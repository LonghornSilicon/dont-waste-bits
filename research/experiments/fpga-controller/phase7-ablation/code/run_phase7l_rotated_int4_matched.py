"""
Phase 7l: rotated static INT4 on the 5 matched HellaSwag subsets
================================================================
Combines:
  - Phase 7j protocol (static INT4, 5 random subsets, paired with FP16)
  - analyze_rotated_q_local's offline Hadamard rotation of K/V projection weights
    (baked in at model-load time, no runtime rotation — 130nm-friendly)

The rotation diagnostic on turboquant-integration showed q4_local spread
collapsing 0.28 → 0.051 (L23 worst: 0.45 → 0.83). If that signal translates to
HellaSwag accuracy, rotated static INT4 should close most of the 5.7pp unrotated
INT4-vs-FP16 gap — giving us a static-INT4 Pareto winner at 3.48× speedup and
near-lossless accuracy. That's the 130nm dream: ~10K gates, no runtime
Hadamard, no per-token routing, no 8-bit path.

Protocol: 5 subsets (np.random.default_rng seeds 0..4, matched to Phase 7j
and fp16_multiseed_matched). For each subset:
  1. Load SmolLM-1.7B.
  2. Apply offline random Hadamard rotation (per layer, seeded) to k_proj
     and v_proj weights: W'_{k,v} = R^T · W_{k,v}.
  3. Run static INT4 HellaSwag eval on that subset.
  4. Compare against fp16_multiseed_matched.json's FP16 accuracy on the
     same subset (paired Δ).

Output: phase7-ablation/results/phase7l_rotated_int4_matched.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"
N_EVAL     = 500
SEEDS      = [0, 1, 2, 3, 4]
ROT_SEED   = 42                       # fixed rotation seed (matches diagnostic)
FP16_COST  = 1.010
INT4_COST  = 0.290

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "phase7l_rotated_int4_matched.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


# ----------------------------------------------------------------------------
# Offline rotation (copied inline from turboquant-integration diagnostic).
def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalized fast Walsh-Hadamard transform on last dim (power-of-2)."""
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"Hadamard requires power-of-2 dim, got {d}"
    h = x.clone()
    step = 1
    while step < d:
        for i in range(0, d, step * 2):
            a = h[..., i:i + step].clone()
            b = h[..., i + step:i + step * 2].clone()
            h[..., i:i + step] = a + b
            h[..., i + step:i + step * 2] = a - b
        step *= 2
    return h / math.sqrt(d)


def rotate_kv_weights_inplace(model, rot_seed=ROT_SEED):
    """For each attention layer, replace W_k, W_v with W' = R^T @ W where
    R = H @ diag(random ±1). Applied in-place; no runtime rotation."""
    gen = torch.Generator().manual_seed(rot_seed)
    for li, layer in enumerate(model.model.layers):
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        d_out = k_proj.weight.shape[0]
        assert d_out == v_proj.weight.shape[0]
        assert d_out & (d_out - 1) == 0, f"d_kv not power-of-2: {d_out}"
        signs = (torch.randint(0, 2, (d_out,), generator=gen).float() * 2 - 1).to(
            k_proj.weight.device).to(k_proj.weight.dtype)
        with torch.no_grad():
            # R^T @ W  ==  H @ diag(signs) @ W  (Hadamard is self-transpose, normalized)
            new_k = hadamard_transform(k_proj.weight.float().T).T * signs.unsqueeze(-1)
            new_v = hadamard_transform(v_proj.weight.float().T).T * signs.unsqueeze(-1)
            k_proj.weight.copy_(new_k.to(k_proj.weight.dtype))
            v_proj.weight.copy_(new_v.to(v_proj.weight.dtype))
    print(f"  rotated K/V weights in {len(model.model.layers)} layers "
          f"(rotation seed {rot_seed})", flush=True)


# ----------------------------------------------------------------------------
def eval_static_int4(model, tokenizer, subset, device):
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
    print(f"Device: {device}  n_samples={N_EVAL}  seeds={SEEDS}  rot_seed={ROT_SEED}",
          flush=True)

    print("\n=== Loading SmolLM-1.7B ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print("\n=== Baking offline Hadamard rotation into K/V weights ===",
          flush=True)
    rotate_kv_weights_inplace(model, rot_seed=ROT_SEED)

    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    ds_list = list(ds)
    total_N = len(ds_list)

    fp16_path = OUT_DIR / "fp16_multiseed_matched.json"
    fp16_per_seed = {}
    if fp16_path.exists():
        fp16_per_seed = json.load(open(fp16_path))["per_seed"]

    per_seed = {}
    for s in SEEDS:
        print(f"\n=== Seed {s} (matched subset) ===", flush=True)
        rng = np.random.default_rng(seed=s)
        idx = rng.choice(total_N, size=N_EVAL, replace=False)
        subset = [ds_list[int(i)] for i in idx]
        t0 = time.time()
        acc = eval_static_int4(model, tokenizer, subset, device)
        elapsed = time.time() - t0
        fp16_acc = fp16_per_seed.get(str(s), {}).get("accuracy_pct")
        per_seed[str(s)] = {
            "accuracy_pct":      round(100.0 * acc, 2),
            "fpga_speedup":      round(FP16_COST / INT4_COST, 3),
            "fp16_matched":      fp16_acc,
            "delta_vs_fp16_pp":  round(100*acc - fp16_acc, 2) if fp16_acc else None,
            "elapsed_s":         round(elapsed, 1),
        }
        print(f"  seed {s}: {per_seed[str(s)]}", flush=True)

    accs = np.array([per_seed[str(s)]["accuracy_pct"] for s in SEEDS])
    mean = float(accs.mean()); std = float(accs.std(ddof=1))
    deltas = [per_seed[str(s)]["delta_vs_fp16_pp"] for s in SEEDS
              if per_seed[str(s)]["delta_vs_fp16_pp"] is not None]
    delta_mean = float(np.mean(deltas)) if deltas else None
    delta_std  = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None

    summary = {
        "accuracy_mean":    round(mean, 2),
        "accuracy_std":     round(std, 3),
        "accuracy_range":   [round(float(accs.min()), 2),
                             round(float(accs.max()), 2)],
        "paired_delta_vs_fp16_mean_pp": round(delta_mean, 2) if delta_mean else None,
        "paired_delta_vs_fp16_std_pp":  round(delta_std, 2) if delta_std else None,
    }

    result = {
        "experiment":  "phase7l_rotated_static_int4_matched",
        "model":       MODEL_NAME,
        "n_samples":   N_EVAL,
        "seeds":       SEEDS,
        "rotation_seed": ROT_SEED,
        "per_seed":    per_seed,
        "summary":     summary,
        "compare_unrotated_phase7j": {
            "int4_mean":  59.08,
            "int4_std":   2.24,
            "delta_vs_fp16_mean_pp":  -5.76,
            "delta_vs_fp16_std_pp":   0.95,
        },
        "elapsed_s":   round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)
    print(f"\nRotated static INT4 (5 matched subsets, n={N_EVAL}): "
          f"{mean:.2f}% ± {std:.2f}pp", flush=True)
    if delta_mean is not None:
        print(f"Paired Δ vs FP16: mean {delta_mean:+.2f}pp, std ±{delta_std:.2f}pp",
              flush=True)
        print(f"(Unrotated Phase 7j was −5.76 ± 0.95pp; rotation recovers "
              f"{-5.76 - delta_mean:+.2f}pp)", flush=True)


if __name__ == "__main__":
    main()
