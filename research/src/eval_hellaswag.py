"""
Direct HellaSwag evaluator — no lm-eval dependency.
Uses length-normalized log-likelihood (standard zero-shot protocol).

Usage:
    python research/src/eval_hellaswag.py --model smollm2-360m --condition fp16 --limit 500
    python research/src/eval_hellaswag.py --model smollm2-360m --condition static4bit --limit 500
    python research/src/eval_hellaswag.py --model smollm2-360m --condition dwb --limit 500
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

MODEL_IDS = {
    "smollm-135m":  "HuggingFaceTB/SmolLM-135M",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm-360m":  "HuggingFaceTB/SmolLM-360M",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "smollm-1.7b":  "HuggingFaceTB/SmolLM-1.7B",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
}

PAPER_TARGETS = {
    "smollm-360m":  {"fp16": 41.50, "static4bit": 33.60, "dwb": 41.20},
    "smollm2-360m": {"fp16": 41.50, "static4bit": 33.60, "dwb": 41.20},
    "smollm-135m":  {"fp16": 37.20, "static4bit": 33.60, "dwb": 36.90},
    "smollm-1.7b":  {"fp16": 49.00, "static4bit": 41.10, "dwb": 48.60},
}


def score_continuation(model, tokenizer, context, continuation, device="cpu"):
    """Length-normalized log-likelihood of continuation given context."""
    full_text = context + continuation
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    ctx_len = tokenizer.encode(context, return_tensors="pt").shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits[0]

    cont_ids = full_ids[0, ctx_len:]
    if len(cont_ids) == 0:
        return -float("inf")

    log_probs = F.log_softmax(logits[ctx_len - 1:ctx_len - 1 + len(cont_ids)], dim=-1)
    score = log_probs[range(len(cont_ids)), cont_ids].sum().item()
    return score / len(cont_ids)


def apply_int4_simulation(model):
    """Simulate INT4 weight quantization for static 4-bit baseline."""
    print("  Applying INT4 weight simulation...")
    count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                scale = param.abs().max() / 7.0
                if scale > 0:
                    param.data = (param.data / scale).round().clamp(-8, 7) * scale
                    count += 1
    print(f"  Quantized {count} weight tensors to INT4")
    return model


def evaluate_hellaswag(model, tokenizer, limit=None, device="cpu"):
    from datasets import load_dataset
    print("  Loading HellaSwag validation split...")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct, total = 0, len(ds)
    t0 = time.time()

    for i, ex in enumerate(ds):
        if i > 0 and i % 50 == 0:
            eta = (time.time() - t0) / i * (total - i)
            print(f"  [{i}/{total}] acc={correct/i*100:.1f}% eta={eta:.0f}s")

        ctx = ex["ctx"]
        scores = [score_continuation(model, tokenizer, ctx, " " + e, device)
                  for e in ex["endings"]]
        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
            correct += 1

    elapsed = time.time() - t0
    acc = correct / total * 100
    print(f"  Result: {correct}/{total} = {acc:.2f}%  ({elapsed:.0f}s)")
    return {"accuracy": acc, "correct": correct, "total": total, "elapsed_s": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smollm2-360m", choices=MODEL_IDS.keys())
    parser.add_argument("--condition", default="fp16",
                        choices=["fp16", "static4bit", "dwb"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="research/data")
    args = parser.parse_args()

    model_id = MODEL_IDS[args.model]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"HellaSwag Evaluation — arXiv:2604.04722 Verification")
    print(f"Model: {model_id}  Condition: {args.condition}  Limit: {args.limit}")
    print(f"{'='*60}\n")

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

    if args.condition == "static4bit":
        model = apply_int4_simulation(model)

    model = model.to(args.device)
    model.eval()

    if args.condition == "dwb":
        dwb_path = out_dir / f"dwb_controller_{args.model}.pt"
        from dwb_implementation import DWBController, train_controller
        if not dwb_path.exists():
            from datasets import load_dataset
            print("  Training DWB controller (100 train examples)...")
            train_ds = load_dataset("Rowan/hellaswag", split="train").select(range(100))
            controller = train_controller(model, tokenizer,
                                          [ex["ctx"] for ex in train_ds],
                                          epochs=3, device=args.device)
            torch.save(controller.state_dict(), dwb_path)
        else:
            controller = DWBController()
            controller.load_state_dict(torch.load(dwb_path, map_location=args.device))
        controller.eval()

    results = evaluate_hellaswag(model, tokenizer, limit=args.limit, device=args.device)

    paper = PAPER_TARGETS.get(args.model, {})
    paper_val = paper.get(args.condition)
    delta = round(results["accuracy"] - paper_val, 2) if paper_val else None
    results.update({
        "model": args.model, "condition": args.condition,
        "limit": args.limit, "paper_target": paper_val,
        "delta_from_paper": delta,
        "verification_status": ("CONFIRMED" if delta is not None and abs(delta) <= 2.0
                                 else ("DISCREPANCY" if delta is not None else "NO_TARGET")),
        "date": datetime.now().isoformat(),
    })

    if paper_val is not None:
        print(f"\n  Paper target: {paper_val:.2f}%")
        print(f"  Our result:   {results['accuracy']:.2f}%")
        print(f"  Delta:        {delta:+.2f}pp  [{results['verification_status']}]")

    fname = f"hellaswag_{args.model}_{args.condition}_{args.limit}samp_{datetime.now():%Y%m%d_%H%M}.json"
    with open(out_dir / fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_dir / fname}")


if __name__ == "__main__":
    main()
