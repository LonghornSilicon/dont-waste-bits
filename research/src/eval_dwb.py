"""
DWB Adaptive KV Evaluation for HellaSwag.

Two-pass approach:
  Pass 1 (FP16): extract per-token signals [H_t, R_t, V_t, C_t] for each example.
  Controller: map signals to bit-widths using trained DWBController.
  Pass 2 (DWB): re-run with per-token quantization applied via k_proj/v_proj hooks.

This approximates the paper's autoregressive DWB evaluation using single-pass
HellaSwag scoring, matching the 'acc' (unnormalized) metric the paper uses.
"""

import torch
import torch.nn.functional as F
import time
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List


BIT_QUANT_FNS = {}


def _make_quant_fns():
    from kv_cache_quant import quantize_int2, quantize_int4, quantize_int8
    return {
        2: quantize_int2,
        4: quantize_int4,
        8: quantize_int8,
        16: lambda x: x,
    }


def make_per_token_hook(bit_widths_per_position: List[int]):
    """
    Creates a forward hook for k_proj or v_proj that applies per-token quantization.
    bit_widths_per_position: list of bit-widths indexed by sequence position.
    """
    quant_fns = _make_quant_fns()

    def hook(module, input, output):
        # output: (batch, seq_len, d_k)
        result = output.clone()
        seq_len = output.shape[1]
        for t in range(min(seq_len, len(bit_widths_per_position))):
            bits = bit_widths_per_position[t]
            qfn = quant_fns.get(bits, lambda x: x)
            result[:, t] = qfn(output[:, t])
        return result

    return hook


def extract_signals_for_sequence(model, tokenizer, text, freq_table, device="cpu", max_length=128):
    """
    Extract per-token signals [H_t, R_t, V_t, C_t] from a single text.
    Returns (signals tensor: [seq_len, 4], token_ids: list)
    """
    from dwb_implementation import compute_entropy, compute_rarity, compute_attention_variance, compute_confidence

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length).to(device)
    input_ids = inputs["input_ids"][0]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    logits_seq = outputs.logits[0]       # (seq_len, vocab)
    attn = outputs.attentions[-1][0]     # (num_heads, seq_len, seq_len)

    signals = []
    for t in range(len(input_ids)):
        H = compute_entropy(logits_seq[t]).item()
        R = compute_rarity(input_ids[t].item(), freq_table)
        V = compute_attention_variance(attn[:, :t+1, :t+1] if t > 0 else attn[:, :1, :1])
        C = compute_confidence(logits_seq[t])
        signals.append([H, R, V, C])

    return torch.tensor(signals, dtype=torch.float32), input_ids.tolist()


def predict_bit_widths(controller, signals: torch.Tensor, device="cpu") -> List[int]:
    """Use DWB controller to predict bit-widths for each token."""
    controller.eval()
    with torch.no_grad():
        logits = controller(signals.to(device))
        classes = logits.argmax(dim=-1)
    bit_classes = [2, 4, 8, 16]
    return [bit_classes[c] for c in classes.tolist()]


def score_continuation_dwb(model, tokenizer, context, continuation, bit_widths: List[int], device="cpu"):
    """
    Score a continuation using per-token DWB quantization.
    bit_widths: predicted bit-widths for each position in the full sequence.
    """
    full_ids = tokenizer.encode(context + continuation, return_tensors="pt").to(device)
    ctx_len = tokenizer.encode(context, return_tensors="pt").shape[1]

    hooks = []
    for name, module in model.named_modules():
        if name.split(".")[-1] in ("k_proj", "v_proj"):
            h = module.register_forward_hook(make_per_token_hook(bit_widths))
            hooks.append(h)

    try:
        with torch.no_grad():
            logits = model(full_ids).logits[0]
    finally:
        for h in hooks:
            h.remove()

    cont_ids = full_ids[0, ctx_len:]
    if len(cont_ids) == 0:
        return -1e9

    log_probs = F.log_softmax(logits[ctx_len-1:ctx_len-1+len(cont_ids)], dim=-1)
    return log_probs[range(len(cont_ids)), cont_ids].sum().item()


def evaluate_hellaswag_dwb(model, tokenizer, controller, limit=200, device="cpu"):
    """
    Evaluate HellaSwag with DWB adaptive quantization.
    Two-pass: extract signals first, then score with per-token bit-widths.
    """
    from datasets import load_dataset

    print("  Loading HellaSwag...", flush=True)
    ds = load_dataset("Rowan/hellaswag", split="validation").select(range(limit))

    # Build frequency table from first pass
    print("  Building token frequency table...", flush=True)
    freq_table = Counter()
    for ex in ds:
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        freq_table.update(tokenizer.encode(ctx))

    correct = 0
    total = limit
    t0 = time.time()
    bit_distribution = Counter()

    for i, ex in enumerate(ds):
        if i > 0 and i % 50 == 0:
            eta = (time.time() - t0) / i * (total - i)
            avg_bits = sum(b * c for b, c in bit_distribution.items()) / max(1, sum(bit_distribution.values()))
            print(f"  [{i}/{total}] acc={correct/i*100:.1f}% avg_bits={avg_bits:.2f} eta={eta:.0f}s", flush=True)

        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()

        # Pass 1: extract signals from context
        signals, token_ids = extract_signals_for_sequence(
            model, tokenizer, ctx, freq_table, device
        )

        # Controller prediction
        bit_widths = predict_bit_widths(controller, signals, device)
        for b in bit_widths:
            bit_distribution[b] += 1

        # Pass 2: score each ending with DWB quantization
        scores = [
            score_continuation_dwb(model, tokenizer, ctx, " " + e, bit_widths, device)
            for e in ex["endings"]
        ]

        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
            correct += 1

    elapsed = time.time() - t0
    acc = correct / total * 100
    total_tokens = sum(bit_distribution.values())
    bit_pct = {b: round(c / total_tokens * 100, 1) for b, c in sorted(bit_distribution.items())}
    avg_bits = sum(b * c for b, c in bit_distribution.items()) / max(1, total_tokens)

    print(f"  Result: {correct}/{total} = {acc:.2f}%  ({elapsed:.0f}s)", flush=True)
    print(f"  Bit distribution: {bit_pct}", flush=True)
    print(f"  Average bits: {avg_bits:.2f}", flush=True)

    return {
        "accuracy": acc, "correct": correct, "total": total, "elapsed_s": elapsed,
        "bit_distribution_pct": bit_pct, "avg_bits": avg_bits,
    }


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smollm-360m")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--controller_path", default=None)
    parser.add_argument("--train_samples", type=int, default=100)
    parser.add_argument("--output_dir", default="research/data")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dwb_implementation import DWBController, train_controller

    MODEL_IDS = {
        "smollm-360m": "HuggingFaceTB/SmolLM-360M",
        "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    }
    model_id = MODEL_IDS.get(args.model, args.model)

    print(f"Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model = model.to(args.device)
    model.eval()

    ctrl_path = Path(args.controller_path or f"{args.output_dir}/dwb_controller_{args.model}.pt")

    if ctrl_path.exists():
        print(f"Loading controller from {ctrl_path}...", flush=True)
        controller = DWBController()
        controller.load_state_dict(torch.load(ctrl_path, map_location=args.device))
    else:
        from datasets import load_dataset
        print(f"Training DWB controller ({args.train_samples} train examples)...", flush=True)
        train_ds = load_dataset("Rowan/hellaswag", split="train").select(range(args.train_samples))
        train_texts = [
            ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
            for ex in train_ds
        ]
        controller = train_controller(model, tokenizer, train_texts,
                                      epochs=5, device=args.device)
        ctrl_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(controller.state_dict(), ctrl_path)
        print(f"Controller saved to {ctrl_path}", flush=True)

    controller = controller.to(args.device)
    controller.eval()

    print(f"\nEvaluating DWB on HellaSwag ({args.limit} samples)...", flush=True)
    results = evaluate_hellaswag_dwb(model, tokenizer, controller,
                                      limit=args.limit, device=args.device)

    results.update({
        "model": args.model, "condition": "dwb",
        "paper_target": 41.2, "paper_fp16": 41.5, "paper_kv4": 33.6,
        "delta_from_paper": round(results["accuracy"] - 41.2, 2),
        "date": datetime.now().isoformat(),
    })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"dwb_eval_{args.model}_{args.limit}samp_{datetime.now():%Y%m%d_%H%M}.json"
    with open(out_dir / fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_dir / fname}", flush=True)

    paper_target = 41.2
    delta = results["accuracy"] - paper_target
    status = "CONFIRMED" if abs(delta) <= 2.0 else "DISCREPANCY"
    print(f"\nPaper target: {paper_target:.2f}%")
    print(f"Our result:   {results['accuracy']:.2f}%")
    print(f"Delta:        {delta:+.2f}pp  [{status}]")
