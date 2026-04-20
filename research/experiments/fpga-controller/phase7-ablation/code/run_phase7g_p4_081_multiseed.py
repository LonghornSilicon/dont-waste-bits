"""
Phase 7g: 5-seed validation at p4=0.81 on SmolLM-1.7B / HellaSwag (n=500)
=========================================================================
Phase 7e single-seed n=200 found a flat 48.0% accuracy plateau over
p4 ∈ [0.60, 0.81], then a ~1pp dip at 0.88 and a ~7pp cliff at 1.00. The
p4=0.81 point offers +16% speedup over the p4=0.74 headline at (apparently)
identical accuracy. This script validates that claim with the same 5-seed
n=500 protocol as Phase 7d.

Target: mean ≈ 48% with std ≤ 1pp → shift paper headline to p4=0.81
(48.0% @ 2.96× FPGA speedup, +21% vs DWB 2.44×).

Output: phase7-ablation/results/phase7g_p4_081_multiseed.json
"""
import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"
N_EVAL     = 500
SEEDS      = [0, 1, 2, 3, 4]
TARGET_P4  = 0.81
FP16_COST  = 1.010

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "phase7g_p4_081_multiseed.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


def make_random_router(p4, seed):
    gen = torch.Generator().manual_seed(seed)
    def route(k, li, T):
        u = torch.rand(T, generator=gen)
        return [4 if u[t].item() < p4 else 8 for t in range(T)]
    return route


def eval_one(model, tokenizer, route_fn, device, n_samples):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)

    bits_buf = {}

    def make_k_hook(li):
        def hook(module, inp, out):
            T = out.shape[1]
            bits_list = route_fn(out, li, T)
            bits_buf[li] = bits_list
            q = out.clone()
            for t, b in enumerate(bits_list):
                q[0, t] = quantize_tensor(out[0, t].float(), b).to(out.dtype)
            return q
        return hook

    def make_v_hook(li):
        def hook(module, inp, out):
            bits_list = bits_buf.get(li, [4] * out.shape[1])
            q = out.clone()
            for t, b in enumerate(bits_list):
                q[0, t] = quantize_tensor(out[0, t].float(), b).to(out.dtype)
            return q
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(li)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(li)))

    correct, total, all_bits = 0, 0, []
    for item in list(ds)[:n_samples]:
        ctx = item["activity_label"] + ": " + item["ctx"]
        scores = []
        for ending in item["endings"]:
            inputs = tokenizer(ctx + " " + ending, return_tensors="pt",
                               truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            bits_buf.clear()
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            scores.append(-out.loss.item())
        if scores.index(max(scores)) == int(item["label"]):
            correct += 1
        total += 1
        for bl in bits_buf.values():
            all_bits.extend(bl)
        if total % 50 == 0:
            print(f"    eval {total}/{n_samples} acc={correct/total:.3f}",
                  flush=True)

    for h in handles:
        h.remove()

    c  = Counter(all_bits)
    n  = max(len(all_bits), 1)
    p4 = c.get(4, 0) / n; p8 = c.get(8, 0) / n
    cost = p4 * 0.29 + p8 * 0.56
    return {
        "accuracy_pct": round(100.0 * correct / total, 2),
        "avg_bits":     round(4 * p4 + 8 * p8, 3),
        "fpga_cost":    round(cost, 4),
        "fpga_speedup": round(FP16_COST / cost, 3) if cost > 0 else 0,
        "p4_pct":       round(100 * p4, 2),
        "p8_pct":       round(100 * p8, 2),
        "n_tokens":     n,
    }


def main():
    t_all = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  n_samples={N_EVAL}  |  seeds={SEEDS}  "
          f"|  target_p4={TARGET_P4}", flush=True)

    print("\n=== Loading SmolLM-1.7B ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    per_seed = {}
    for s in SEEDS:
        print(f"\n=== Seed {s} ===", flush=True)
        t0 = time.time()
        route = make_random_router(TARGET_P4, seed=s)
        metrics = eval_one(model, tokenizer, route, device, n_samples=N_EVAL)
        metrics["elapsed_s"] = round(time.time() - t0, 1)
        print(f"  seed {s}: {metrics}", flush=True)
        per_seed[str(s)] = metrics

    accs = np.array([per_seed[str(s)]["accuracy_pct"] for s in SEEDS])
    mean = float(accs.mean()); std = float(accs.std(ddof=1))
    sem  = std / np.sqrt(len(accs))
    ci95 = 1.96 * sem
    sp   = np.array([per_seed[str(s)]["fpga_speedup"] for s in SEEDS])

    summary = {
        "accuracy_mean":    round(mean, 2),
        "accuracy_std":     round(std, 3),
        "accuracy_sem":     round(sem, 3),
        "accuracy_ci95_pp": round(ci95, 3),
        "accuracy_range":   [round(float(accs.min()), 2),
                             round(float(accs.max()), 2)],
        "fpga_speedup_mean": round(float(sp.mean()), 3),
    }

    result = {
        "experiment": "phase7g_p4_081_multiseed",
        "model":      MODEL_NAME,
        "n_samples":  N_EVAL,
        "seeds":      SEEDS,
        "target_p4":  TARGET_P4,
        "per_seed":   per_seed,
        "summary":    summary,
        "compare_phase7d_p4_074": {
            "accuracy_mean_pct": 48.04,
            "accuracy_std_pp": 0.754,
            "fpga_speedup_mean": 2.804,
        },
        "elapsed_s": round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)
    print(f"\nRandom routing @ p4={TARGET_P4} (5 seeds, n={N_EVAL}): "
          f"{summary['accuracy_mean']}% ± {summary['accuracy_std']}pp "
          f"(±{summary['accuracy_ci95_pp']}pp CI) at "
          f"{summary['fpga_speedup_mean']}× speedup", flush=True)


if __name__ == "__main__":
    main()
