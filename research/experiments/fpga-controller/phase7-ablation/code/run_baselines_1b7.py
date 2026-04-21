"""
Phase 7 baselines: FP16 and static INT4 at SmolLM-1.7B / HellaSwag (n=500)
===========================================================================
The paper's tab:main cites FP16 49.0% and static INT4 41.1% from the DWB paper
table. Here we measure both directly on the same A4000 + same n=500 HellaSwag
evaluation harness used for Phase 7d/7g/7i — so the Pareto frontier endpoints
are internally consistent.

Two conditions, both single-seed n=500:
  1. FP16 — no quantization hooks at all
  2. Static INT4 — symmetric per-token INT4 on every k_proj and v_proj output

Output: phase7-ablation/results/baselines_1b7.json
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"
N_EVAL     = 500
SEED       = 0
FP16_COST  = 1.010
INT4_COST  = 0.290

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "baselines_1b7.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


def eval_hellaswag(model, tokenizer, quant_bits, device, n_samples=N_EVAL):
    """quant_bits=None → FP16 (no hooks); quant_bits=4 → symmetric INT4 on all tokens."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)

    handles = []
    if quant_bits is not None:
        def make_hook():
            def hook(module, inp, out):
                q = out.clone()
                for t in range(out.shape[1]):
                    q[0, t] = quantize_tensor(out[0, t].float(), quant_bits).to(out.dtype)
                return q
            return hook

        for li, layer in enumerate(model.model.layers):
            handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook()))
            handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook()))

    correct, total = 0, 0
    for item in list(ds)[:n_samples]:
        ctx = item["activity_label"] + ": " + item["ctx"]
        scores = []
        for ending in item["endings"]:
            inputs = tokenizer(ctx + " " + ending, return_tensors="pt",
                               truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            scores.append(-out.loss.item())
        if scores.index(max(scores)) == int(item["label"]):
            correct += 1
        total += 1
        if total % 50 == 0:
            print(f"    eval {total}/{n_samples} acc={correct/total:.3f}", flush=True)

    for h in handles:
        h.remove()

    return correct / total


def main():
    t_all = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  n_samples={N_EVAL}  seed={SEED}", flush=True)

    print("\n=== Loading SmolLM-1.7B ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    results = {}

    for name, bits, cost in [("fp16", None, FP16_COST),
                             ("static_int4", 4, INT4_COST)]:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        acc = eval_hellaswag(model, tokenizer, bits, device, n_samples=N_EVAL)
        elapsed = time.time() - t0
        speedup = FP16_COST / cost
        results[name] = {
            "accuracy_pct": round(100 * acc, 2),
            "fpga_cost":    round(cost, 3),
            "fpga_speedup": round(speedup, 3),
            "elapsed_s":    round(elapsed, 1),
        }
        print(f"  {name}: {results[name]}", flush=True)

    result = {
        "experiment": "baselines_1b7",
        "model":      MODEL_NAME,
        "n_samples":  N_EVAL,
        "seed":       SEED,
        "conditions": results,
        "paper_reference": {
            "fp16_paper":  {"accuracy_pct": 49.0, "source": "DWB Table 3"},
            "int4_paper":  {"accuracy_pct": 41.1, "source": "DWB Table 3"},
        },
        "elapsed_s": round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)
    print(f"\nFP16:        {results['fp16']['accuracy_pct']}% at {results['fp16']['fpga_speedup']}×  "
          f"(paper: 49.0%)", flush=True)
    print(f"Static INT4: {results['static_int4']['accuracy_pct']}% at {results['static_int4']['fpga_speedup']}×  "
          f"(paper: 41.1%)", flush=True)


if __name__ == "__main__":
    main()
