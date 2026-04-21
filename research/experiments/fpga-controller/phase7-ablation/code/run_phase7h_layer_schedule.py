"""
Phase 7h: Layer-tuned bit-schedule on SmolLM-1.7B / HellaSwag (n=200 first, 5-seed n=500 if worth)
==================================================================================================
Per-layer diagnostic (results/per_layer_q_local.json) found a monotonic gradient
with a single outlier: L23 has q4_local = 0.45 (vs 0.55-0.73 for the other 23
layers), driven by the largest KV norms (107 vs 38-105). Hypothesis: L23 alone
accounts for most of the accuracy loss at high p4.

Three layer-schedules to compare (single-seed n=200 per schedule):

  H0 (baseline uniform p4=0.96 random)            → effective p4=0.96, 3.36× predicted
  H1 ({4: L0-L22, 8: L23} static)                 → effective p4=0.96, 3.35× predicted
  H2 ({4: L0-L20, 8: L21-L23} static)             → effective p4=0.875, 3.13× predicted
  H3 ({4: L0-L16, 8: L17-L23} static, top-30% 8b) → effective p4=0.708, 2.74× predicted

H1 is the headline. If H1 ≥ 48% at 3.35×, we have +37% vs DWB (2.44×) at matched accuracy.

Output: phase7-ablation/results/phase7h_layer_schedule.json
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
FP16_COST  = 1.010

NUM_LAYERS = 24  # SmolLM-1.7B

SCHEDULES = {
    # name: dict[layer_idx -> bits]; default 4 for any missing layer
    "H0_uniform_p4_096": {"_all_": "random", "_p4_": 0.96},
    "H1_protect_L23":    {i: 4 for i in range(23)} | {23: 8},
    "H2_protect_top3":   {i: 4 for i in range(21)} | {21: 8, 22: 8, 23: 8},
    "H3_protect_top7":   {i: 4 for i in range(17)} | {**{i: 8 for i in range(17, 24)}},
}

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "phase7h_layer_schedule.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


def make_uniform_random_router(p4, seed):
    gen = torch.Generator().manual_seed(seed)
    def route(k, li, T):
        u = torch.rand(T, generator=gen)
        return [4 if u[t].item() < p4 else 8 for t in range(T)]
    return route


def make_layer_schedule_router(schedule):
    """schedule: dict[layer_idx -> 4 or 8]."""
    def route(k, li, T):
        b = schedule.get(li, 4)
        return [b] * T
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
    print(f"Device: {device}  seed={SEED}  n_samples={N_EVAL}", flush=True)

    print("\n=== Loading SmolLM-1.7B ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    per_schedule = {}
    for name, sched in SCHEDULES.items():
        print(f"\n=== Eval: {name} ===", flush=True)
        t0 = time.time()
        if sched.get("_all_") == "random":
            route = make_uniform_random_router(p4=sched["_p4_"], seed=SEED)
            desc  = f"uniform random p4={sched['_p4_']}"
        else:
            route = make_layer_schedule_router(sched)
            bits_by_layer = [sched.get(i, 4) for i in range(NUM_LAYERS)]
            eight_layers = [i for i, b in enumerate(bits_by_layer) if b == 8]
            desc  = f"static layers-at-8bit={eight_layers}"
        metrics = eval_hellaswag(model, tokenizer, route, device, n_samples=N_EVAL)
        metrics["description"] = desc
        metrics["elapsed_s"]   = round(time.time() - t0, 1)
        print(f"  {name}: {metrics}", flush=True)
        per_schedule[name] = metrics

    result = {
        "experiment":  "phase7h_layer_schedule",
        "model":       MODEL_NAME,
        "n_samples":   N_EVAL,
        "seed":        SEED,
        "schedules":   {k: v if isinstance(v.get("_all_", None), str)
                        else {str(kk): vv for kk, vv in v.items()}
                        for k, v in SCHEDULES.items()},
        "per_schedule": per_schedule,
        "reference": {
            "p4_074_5seed_phase7d": {"acc_pct": 48.04, "std_pp": 0.754, "speedup": 2.804},
            "p4_081_5seed_phase7g": {"acc_pct": 48.32, "std_pp": 0.944, "speedup": 2.960},
            "dwb_paper": {"acc_pct": 48.6, "speedup": 2.44},
        },
        "elapsed_s":   round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)

    print("\n=== Phase 7h summary ===", flush=True)
    for name, m in per_schedule.items():
        print(f"  {name:20s}  acc={m['accuracy_pct']:.2f}%  "
              f"p4={m['p4_pct']:.1f}%  speedup={m['fpga_speedup']:.2f}×  "
              f"[{m['description']}]", flush=True)


if __name__ == "__main__":
    main()
