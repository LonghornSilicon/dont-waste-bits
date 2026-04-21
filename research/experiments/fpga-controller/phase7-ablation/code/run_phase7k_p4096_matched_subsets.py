"""
Phase 7k: random routing at p4=0.96 on the 5 matched HellaSwag subsets
======================================================================
Our Phase 7i 5-seed result (48.04% etc.) used different *routing* seeds on the
same first-500 HellaSwag subset. Phase 7j + fp16_multiseed_matched use 5
different *subsets* of the full 10,042-example validation set. To compare our
Pareto-best point directly against the matching-subset FP16, we re-run p4=0.96
on the same 5 subsets.

This yields paired (FP16, INT-quantized-at-p4=0.96) deltas per subset. The
question it answers: "is p4=0.96 random routing statistically indistinguishable
from FP16 on this model and eval?"

Decision rule:
  - mean delta (p4=0.96 − FP16) > −2.0pp: effectively lossless; paper headline
    becomes "lossless at 3.36× FPGA speedup" on our eval.
  - delta ∈ [−5, −2] pp: meaningfully lossy; the Phase 7 narrative stands but
    quantitative numbers update.
  - delta < −5pp: rotation is mandatory for the lossless story.

Protocol: same subset selection as Phase 7j (np.random.default_rng seeds 0–4),
same Bernoulli(p4=0.96) routing-seed matched to the subset-selection seed for
reproducibility.

Output: phase7-ablation/results/phase7k_p4096_matched.json
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
TARGET_P4  = 0.96
FP16_COST  = 1.010

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "phase7k_p4096_matched.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


def make_random_router(p4, seed):
    gen = torch.Generator().manual_seed(seed)
    def route(k, li, T):
        u = torch.rand(T, generator=gen)
        return [4 if u[t].item() < p4 else 8 for t in range(T)]
    return route


def eval_quantized(model, tokenizer, subset, route_fn, device):
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
    for item in subset:
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
            print(f"    eval {total}/{len(subset)} acc={correct/total:.3f}",
                  flush=True)

    for h in handles:
        h.remove()

    from collections import Counter
    c  = Counter(all_bits); n = max(len(all_bits), 1)
    p4 = c.get(4, 0) / n; p8 = c.get(8, 0) / n
    cost = p4 * 0.29 + p8 * 0.56
    return {
        "accuracy_pct": round(100.0 * correct / total, 2),
        "fpga_speedup": round(FP16_COST / cost, 3),
        "p4_pct":       round(100 * p4, 2),
    }


def main():
    t_all = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  n_samples={N_EVAL}  seeds={SEEDS}  p4={TARGET_P4}",
          flush=True)

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

    # Load matching FP16 results for paired deltas.
    fp16_path = OUT_DIR / "fp16_multiseed_matched.json"
    fp16_per_seed = {}
    if fp16_path.exists():
        fp16_data = json.load(open(fp16_path))
        fp16_per_seed = fp16_data["per_seed"]

    per_seed = {}
    for s in SEEDS:
        print(f"\n=== Seed {s} (subset matched to Phase 7j / fp16_multiseed) ===",
              flush=True)
        rng = np.random.default_rng(seed=s)
        idx = rng.choice(total_N, size=N_EVAL, replace=False)
        subset = [ds_list[int(i)] for i in idx]
        route = make_random_router(TARGET_P4, seed=s)
        t0 = time.time()
        m = eval_quantized(model, tokenizer, subset, route, device)
        m["elapsed_s"] = round(time.time() - t0, 1)
        fp16_acc = fp16_per_seed.get(str(s), {}).get("accuracy_pct")
        m["fp16_matched"] = fp16_acc
        m["delta_vs_fp16_pp"] = (round(m["accuracy_pct"] - fp16_acc, 2)
                                 if fp16_acc is not None else None)
        per_seed[str(s)] = m
        print(f"  seed {s}: {m}", flush=True)

    accs = np.array([per_seed[str(s)]["accuracy_pct"] for s in SEEDS])
    mean = float(accs.mean()); std = float(accs.std(ddof=1))

    deltas = [per_seed[str(s)]["delta_vs_fp16_pp"] for s in SEEDS
              if per_seed[str(s)]["delta_vs_fp16_pp"] is not None]
    delta_mean = round(float(np.mean(deltas)), 2) if deltas else None
    delta_std  = round(float(np.std(deltas, ddof=1)), 2) if len(deltas) > 1 else None

    summary = {
        "accuracy_mean":    round(mean, 2),
        "accuracy_std":     round(std, 3),
        "accuracy_range":   [round(float(accs.min()), 2),
                             round(float(accs.max()), 2)],
        "paired_delta_vs_fp16_mean_pp": delta_mean,
        "paired_delta_vs_fp16_std_pp":  delta_std,
    }

    result = {
        "experiment": "phase7k_p4096_on_matched_subsets",
        "model":      MODEL_NAME,
        "n_samples":  N_EVAL,
        "seeds":      SEEDS,
        "target_p4":  TARGET_P4,
        "per_seed":   per_seed,
        "summary":    summary,
        "elapsed_s":  round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)
    print(f"\nRandom routing p4={TARGET_P4} on matched subsets (n={N_EVAL}, 5 seeds): "
          f"{mean:.2f}% ± {std:.2f}pp, range {summary['accuracy_range']}", flush=True)
    if delta_mean is not None:
        print(f"Paired Δ vs FP16 (matched subsets): mean {delta_mean:+.2f}pp, "
              f"std ±{delta_std}pp", flush=True)


if __name__ == "__main__":
    main()
