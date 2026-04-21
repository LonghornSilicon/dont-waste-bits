"""
Phase 7j: 5-seed validation of static INT4 on SmolLM-1.7B / HellaSwag.
======================================================================
The single-subset baseline (run_baselines_1b7.py) measured static INT4 at 47.0%
on the first 500 HellaSwag examples on A4000 — 5.9pp higher than the DWB paper's
reported 41.1%. If robust, this changes the paper's 1.7B narrative: the "static
INT4 cliff" may be mostly a measurement artifact of the paper, not the method.

Validation protocol:
  - Static INT4 has no routing randomness — seeding only changes the HellaSwag
    subset used. Each "seed" picks 500 items uniformly at random from the full
    10,042 validation set (with replacement within a seed disabled; same seed
    gives the same subset).
  - 5 seeds {0,1,2,3,4}. Mean ± std tells us whether 47.0% is robust.

Output: phase7-ablation/results/phase7j_static_int4_multiseed.json
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
INT4_COST  = 0.290

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "phase7j_static_int4_multiseed.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


def eval_static_int4(model, tokenizer, subset, device):
    """Run static INT4 eval on a pre-shuffled subset of HellaSwag examples."""
    def make_hook():
        def hook(module, inp, out):
            q = out.clone()
            for t in range(out.shape[1]):
                q[0, t] = quantize_tensor(out[0, t].float(), 4).to(out.dtype)
            return q
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook()))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook()))

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

    for h in handles:
        h.remove()
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
    print(f"  HellaSwag validation set: {total_N} items (will sample {N_EVAL} per seed)",
          flush=True)

    per_seed = {}
    for s in SEEDS:
        print(f"\n=== Seed {s} (subset selection) ===", flush=True)
        rng = np.random.default_rng(seed=s)
        idx = rng.choice(total_N, size=N_EVAL, replace=False)
        subset = [ds_list[int(i)] for i in idx]
        t0 = time.time()
        acc = eval_static_int4(model, tokenizer, subset, device)
        elapsed = time.time() - t0
        speedup = FP16_COST / INT4_COST
        per_seed[str(s)] = {
            "accuracy_pct": round(100.0 * acc, 2),
            "fpga_speedup": round(speedup, 3),
            "elapsed_s":    round(elapsed, 1),
        }
        print(f"  seed {s}: {per_seed[str(s)]}", flush=True)

    accs = np.array([per_seed[str(s)]["accuracy_pct"] for s in SEEDS])
    mean = float(accs.mean()); std = float(accs.std(ddof=1))
    sem  = std / np.sqrt(len(accs))
    ci95 = 1.96 * sem

    summary = {
        "accuracy_mean":    round(mean, 2),
        "accuracy_std":     round(std, 3),
        "accuracy_sem":     round(sem, 3),
        "accuracy_ci95_pp": round(ci95, 3),
        "accuracy_range":   [round(float(accs.min()), 2),
                             round(float(accs.max()), 2)],
        "fpga_speedup":     round(FP16_COST / INT4_COST, 3),
    }

    result = {
        "experiment": "phase7j_static_int4_multiseed",
        "model":      MODEL_NAME,
        "n_samples":  N_EVAL,
        "seeds":      SEEDS,
        "per_seed":   per_seed,
        "summary":    summary,
        "compare_paper": {
            "dwb_paper_fp16": 49.0,
            "dwb_paper_int4": 41.1,
            "our_baseline_first500_int4": 47.0,
        },
        "elapsed_s": round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)
    print(f"\nStatic INT4 @ SmolLM-1.7B (5 random subsets, n={N_EVAL}): "
          f"{summary['accuracy_mean']}% ± {summary['accuracy_std']}pp "
          f"at {summary['fpga_speedup']}× speedup. "
          f"Range [{summary['accuracy_range'][0]}, {summary['accuracy_range'][1]}].", flush=True)
    print(f"  Paper reports 41.1% — our measurement is "
          f"+{summary['accuracy_mean'] - 41.1:.1f}pp higher.", flush=True)


if __name__ == "__main__":
    main()
