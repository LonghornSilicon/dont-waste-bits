"""
Phase 7m (fast, vectorized): R2-only offline rotation + static INT4.
====================================================================
Same experiment as run_phase7m_r2_rotation.py but vectorizes the per-token
INT4 quantization. Prior version did a Python loop `for t in range(T):
quantize_tensor(x[0, t])` inside every k_proj/v_proj hook, across 24 layers ×
~100 tokens × 4 endings × 500 items × 5 seeds = ~24M Python-level calls.
GPU sat at ~15% utilization; CPU was the bottleneck.

Vectorized INT4 quantization (per-token symmetric, 15-level — identical math):

    def quantize_int4_per_token_batched(x):           # x: (T, d_model)
        levels = 7                                    # 2^(4-1) - 1
        abs_max = x.abs().max(dim=-1, keepdim=True).values  # (T, 1)
        scale   = (abs_max / levels).clamp(min=1e-10)
        return (x / scale).round().clamp(-levels - 1, levels) * scale

No Python per-token loop, all GPU ops. Expected ~10× speedup per forward pass.
Output: phase7-ablation/results/phase7m_r2_rotation.json (same path as slow version)
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
ROT_SEED   = 42
FP16_COST  = 1.010
INT4_COST  = 0.290

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "phase7m_r2_rotation.json"


# ----------------------------------------------------------------------------
# Batched per-token INT4 (symmetric, 15-level; identical math to the scalar
# version in src/kv_cache_quant.py, just vectorized).
def quantize_int4_per_token_batched(x: torch.Tensor) -> torch.Tensor:
    levels = 7  # 2^(4-1) - 1
    # x: (T, d_model) or (B, T, d_model). Take max over LAST dim per-position.
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / levels).clamp(min=1e-10)
    return (x / scale).round().clamp(-levels - 1, levels) * scale


# ----------------------------------------------------------------------------
# Rotation utils (identical to run_phase7m_r2_rotation.py)
def hadamard(d: int) -> torch.Tensor:
    assert d & (d - 1) == 0, f"Hadamard requires power-of-2, got {d}"
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(d)


def random_orthogonal(d: int, gen: torch.Generator) -> torch.Tensor:
    signs = (torch.randint(0, 2, (d,), generator=gen).float() * 2 - 1)
    return hadamard(d) * signs.unsqueeze(0)


def apply_r2_rotation(model, rot_seed=ROT_SEED):
    gen = torch.Generator().manual_seed(rot_seed)
    n_heads = model.config.num_attention_heads
    n_kv    = model.config.num_key_value_heads
    d_model = model.config.hidden_size
    d_head  = d_model // n_heads

    for li, layer in enumerate(model.model.layers):
        W_v = layer.self_attn.v_proj.weight.data
        W_o = layer.self_attn.o_proj.weight.data
        dev = W_v.device; dt = W_v.dtype
        R = random_orthogonal(d_head, gen).to(dev).to(dt)

        with torch.no_grad():
            W_v_r = W_v.view(n_kv, d_head, d_model)
            W_v_rot = torch.einsum('ij,hjk->hik', R.T, W_v_r)
            layer.self_attn.v_proj.weight.data.copy_(
                W_v_rot.reshape(n_kv * d_head, d_model).contiguous())

            W_o_r = W_o.view(d_model, n_heads, d_head)
            W_o_rot = torch.einsum('dhj,jk->dhk', W_o_r, R)
            layer.self_attn.o_proj.weight.data.copy_(
                W_o_rot.reshape(d_model, n_heads * d_head).contiguous())

    print(f"  R2 rotation applied to V + W_o in {len(model.model.layers)} layers "
          f"(rot_seed={rot_seed}, RoPE-safe)", flush=True)


# ----------------------------------------------------------------------------
# Gates
def gate_fp16_sanity(model_unrot, model_rot, tokenizer, device, sample):
    ctx = sample["activity_label"] + ": " + sample["ctx"]
    max_diff = 0.0
    any_nan = False
    for ending in sample["endings"]:
        inputs = tokenizer(ctx + " " + ending, return_tensors="pt",
                           truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            l_unrot = model_unrot(**inputs, labels=inputs["input_ids"]).loss.item()
            l_rot   = model_rot(**inputs, labels=inputs["input_ids"]).loss.item()
        if not (math.isfinite(l_unrot) and math.isfinite(l_rot)):
            any_nan = True
        max_diff = max(max_diff, abs(l_unrot - l_rot))
    return max_diff, any_nan


def eval_static_int4_fast(model, tokenizer, subset, device):
    """Vectorized INT4 eval — no Python per-token loop in the hook."""
    def make_hook():
        def hook(module, inp, out):
            # out shape: (1, T, d_model). Cast to fp32 for accurate scale, then back.
            q = quantize_int4_per_token_batched(out.float()).to(out.dtype)
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
        if total % 100 == 0:
            print(f"    eval {total}/{len(subset)} acc={correct/total:.3f}",
                  flush=True)
    for h in handles:
        h.remove()
    return correct / total


# ----------------------------------------------------------------------------
def main():
    t_all = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  n_samples={N_EVAL}  seeds={SEEDS}  rot_seed={ROT_SEED}  "
          f"(vectorized INT4)", flush=True)

    print("\n=== Loading SmolLM-1.7B (unrotated reference) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_unrot = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model_unrot.eval()

    print("\n=== Loading second copy + R2 rotation ===", flush=True)
    model_rot = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model_rot.eval()
    apply_r2_rotation(model_rot, rot_seed=ROT_SEED)

    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    ds_list = list(ds)
    total_N = len(ds_list)

    # Gate 1
    print("\n=== GATE 1: FP16 rotated == unrotated (single item) ===", flush=True)
    max_diff, any_nan = gate_fp16_sanity(model_unrot, model_rot, tokenizer, device,
                                          ds_list[0])
    print(f"  max |Δ loss|: {max_diff:.6f}   NaN/Inf: {any_nan}", flush=True)
    if any_nan or max_diff > 0.01:
        print("  FAIL", flush=True); sys.exit(1)
    print("  PASS ✓", flush=True)

    del model_unrot
    torch.cuda.empty_cache()

    # Gate 2: vectorized smoke — 50 samples on seed 0
    print("\n=== GATE 2: vectorized INT4 smoke (50 samples, seed 0 subset) ===",
          flush=True)
    rng = np.random.default_rng(seed=0)
    idx = rng.choice(total_N, size=50, replace=False)
    smoke_subset = [ds_list[int(i)] for i in idx]
    t0 = time.time()
    smoke_acc = eval_static_int4_fast(model_rot, tokenizer, smoke_subset, device)
    smoke_time = time.time() - t0
    print(f"  smoke acc: {smoke_acc*100:.1f}%   elapsed: {smoke_time:.1f}s "
          f"({smoke_time/50*1000:.0f} ms/sample)", flush=True)
    if smoke_acc < 0.35:
        print(f"  FAIL", flush=True); sys.exit(1)
    print("  PASS ✓", flush=True)

    # Full 5-subset INT4 eval
    print("\n=== Full 5-subset INT4 eval (vectorized) ===", flush=True)
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
        acc = eval_static_int4_fast(model_rot, tokenizer, subset, device)
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
        "accuracy_mean":               round(mean, 2),
        "accuracy_std":                round(std, 3),
        "accuracy_range":              [round(float(accs.min()), 2),
                                        round(float(accs.max()), 2)],
        "paired_delta_vs_fp16_mean_pp": round(delta_mean, 2) if delta_mean else None,
        "paired_delta_vs_fp16_std_pp":  round(delta_std, 2) if delta_std else None,
        "sanity_max_diff":              round(max_diff, 6),
        "smoke_acc_pct":                round(100*smoke_acc, 2),
    }

    result = {
        "experiment":    "phase7m_r2_rotation_fast",
        "model":         MODEL_NAME,
        "n_samples":     N_EVAL,
        "seeds":         SEEDS,
        "rotation_seed": ROT_SEED,
        "quant_kernel":  "vectorized_int4_per_token_symmetric_15level",
        "per_seed":      per_seed,
        "summary":       summary,
        "compare_unrotated_phase7j": {
            "int4_mean":              59.08,
            "int4_std":               2.24,
            "delta_vs_fp16_mean_pp":  -5.76,
            "delta_vs_fp16_std_pp":   0.95,
        },
        "elapsed_s":     round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)
    print(f"\nR2-rotated static INT4 (5 matched subsets, n={N_EVAL}, vectorized): "
          f"{mean:.2f}% ± {std:.2f}pp", flush=True)
    if delta_mean is not None:
        print(f"Paired Δ vs FP16: {delta_mean:+.2f} ± {delta_std:.2f}pp  "
              f"(unrotated 7j: −5.76 ± 0.95pp; recovery: "
              f"{-5.76 - delta_mean:+.2f}pp)", flush=True)


if __name__ == "__main__":
    main()
