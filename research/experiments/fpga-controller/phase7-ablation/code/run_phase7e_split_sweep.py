"""
Phase 7e: Split-ratio sweep on SmolLM-1.7B (HellaSwag n=200, seed 0)
====================================================================
Random routing at fixed seed 0, varying p4 ∈ {0.60, 0.67, 0.81, 0.88} to trace
the accuracy / FPGA-speedup Pareto frontier. Phase 7d already pinned the
p4=0.74 operating point (48.04% ± 0.75pp @ 2.80×) with 5-seed stats; this
sweep uses single-seed n=200 to get a cheap frontier.

Reference points (no re-run needed):
  p4 = 1.00  ⇒  static INT4, acc = 41.1% (paper), speedup = 3.48×
  p4 = 0.74  ⇒  48.04% ± 0.75pp (Phase 7d 5-seed), speedup = 2.80×
  p4 = 0.00  ⇒  static INT8, acc ≈ 48.5% (paper), speedup = 1.80×

Output: phase7-ablation/results/phase7e_split_sweep.json
"""
import json
import sys
import time
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"
N_EVAL     = 200
SEED       = 0
P4_SWEEP   = [0.60, 0.67, 0.81, 0.88]
FP16_COST  = 1.010

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "phase7e_split_sweep.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


def make_random_router(p4, seed):
    gen = torch.Generator().manual_seed(seed)
    def route(k, li, T):
        u = torch.rand(T, generator=gen)
        return [4 if u[t].item() < p4 else 8 for t in range(T)]
    return route


def eval_hellaswag(model, tokenizer, route_fn, device, n_samples=N_EVAL):
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
        if total % 20 == 0:
            print(f"    eval {total}/{n_samples} acc={correct/total:.3f}", flush=True)

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
    torch.manual_seed(SEED)
    print(f"Device: {device}  |  seed={SEED}  |  p4 sweep={P4_SWEEP}", flush=True)

    print("\n=== Loading SmolLM-1.7B ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    per_p4 = {}
    for p4 in P4_SWEEP:
        print(f"\n=== Eval: p4={p4} ===", flush=True)
        t0 = time.time()
        route = make_random_router(p4=p4, seed=SEED)
        metrics = eval_hellaswag(model, tokenizer, route, device, n_samples=N_EVAL)
        metrics["elapsed_s"] = round(time.time() - t0, 1)
        print(f"  p4={p4}: {metrics}", flush=True)
        per_p4[f"{p4:.2f}"] = metrics

    result = {
        "experiment": "phase7e_split_sweep",
        "model":      MODEL_NAME,
        "n_samples":  N_EVAL,
        "seed":       SEED,
        "p4_sweep":   P4_SWEEP,
        "strategy":   "random routing (Bernoulli(p4))",
        "per_p4":     per_p4,
        "reference_points": {
            "p4_0.00_static_int8": {"accuracy_pct_paper": 48.5, "fpga_speedup": 1.80},
            "p4_0.74_phase7d":     {"accuracy_pct": 48.04, "std_pp": 0.75,
                                    "fpga_speedup": 2.80, "source": "5-seed n=500"},
            "p4_1.00_static_int4": {"accuracy_pct_paper": 41.1, "fpga_speedup": 3.48},
        },
        "elapsed_s": round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)
    print("\nPareto sweep summary:", flush=True)
    print(f"  p4=0.00 (static INT8, paper):   48.5%    1.80×", flush=True)
    for p4 in P4_SWEEP:
        m = per_p4[f"{p4:.2f}"]
        print(f"  p4={p4:.2f}:  {m['accuracy_pct']:5.2f}%    "
              f"{m['fpga_speedup']:.2f}×", flush=True)
    print(f"  p4=0.74 (phase7d 5-seed):       48.04%   2.80×", flush=True)
    print(f"  p4=1.00 (static INT4, paper):   41.1%    3.48×", flush=True)


if __name__ == "__main__":
    main()
