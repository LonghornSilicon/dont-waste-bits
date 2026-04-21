"""
FP16 5-subset baseline on SmolLM-1.7B / HellaSwag, matching Phase 7j's protocol.

Phase 7j found that static INT4 on random 500-item HellaSwag subsets gives
~56–62% — much higher than both the DWB paper's 41.1% and our own first-500
measurement of 47.0%. Hypothesis: HellaSwag validation is ordered by
activity/difficulty, and first-500 is systematically harder than random
subsets.

This script tests the hypothesis by running FP16 (no quantization) on the
SAME 5 random subsets as Phase 7j (same seeds, same `np.random.default_rng`
selection). If FP16 on random subsets also shows +10pp over the first-500
baseline of 52.0%, the subset-selection effect is the dominant signal and
all our absolute-accuracy numbers need re-anchoring to FP16-per-subset.

Per-subset INT4 minus FP16 delta is what matters for the lossiness story.

Output: phase7-ablation/results/fp16_multiseed_matched.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"
N_EVAL     = 500
SEEDS      = [0, 1, 2, 3, 4]
FP16_COST  = 1.010

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "fp16_multiseed_matched.json"


def eval_fp16(model, tokenizer, subset, device):
    """FP16 HellaSwag eval — no hooks."""
    correct, total = 0, 0
    for item in subset:
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
            print(f"    eval {total}/{len(subset)} acc={correct/total:.3f}",
                  flush=True)
    return correct / total


def main():
    t_all = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  n_samples={N_EVAL}  seeds={SEEDS}", flush=True)

    print("\n=== Loading SmolLM-1.7B ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    ds_list = list(ds)
    total_N = len(ds_list)
    print(f"  HellaSwag validation set: {total_N} items", flush=True)

    per_seed = {}
    for s in SEEDS:
        print(f"\n=== Seed {s} (subset selection — matches Phase 7j) ===",
              flush=True)
        rng = np.random.default_rng(seed=s)
        idx = rng.choice(total_N, size=N_EVAL, replace=False)
        subset = [ds_list[int(i)] for i in idx]
        t0 = time.time()
        acc = eval_fp16(model, tokenizer, subset, device)
        elapsed = time.time() - t0
        per_seed[str(s)] = {
            "accuracy_pct": round(100.0 * acc, 2),
            "elapsed_s":    round(elapsed, 1),
        }
        print(f"  seed {s}: {per_seed[str(s)]}", flush=True)

    accs = np.array([per_seed[str(s)]["accuracy_pct"] for s in SEEDS])
    mean = float(accs.mean()); std = float(accs.std(ddof=1))

    result = {
        "experiment": "fp16_multiseed_matched_to_phase7j",
        "model":      MODEL_NAME,
        "n_samples":  N_EVAL,
        "seeds":      SEEDS,
        "per_seed":   per_seed,
        "summary": {
            "accuracy_mean":    round(mean, 2),
            "accuracy_std":     round(std, 3),
            "accuracy_range":   [round(float(accs.min()), 2),
                                 round(float(accs.max()), 2)],
        },
        "compare": {
            "fp16_first500_subset":  52.0,
            "int4_first500_subset":  47.0,
            "dwb_paper_fp16":        49.0,
            "dwb_paper_int4":        41.1,
        },
        "elapsed_s": round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)
    print(f"\nFP16 @ SmolLM-1.7B (5 random subsets matching Phase 7j, n={N_EVAL}): "
          f"{result['summary']['accuracy_mean']}% ± {result['summary']['accuracy_std']}pp, "
          f"range {result['summary']['accuracy_range']}", flush=True)


if __name__ == "__main__":
    main()
