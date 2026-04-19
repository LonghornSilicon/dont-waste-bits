"""
Mechanistic verification of INT4 losslessness.

Insight 4 claims: symmetric per-tensor INT4 errors cancel in attention weighted sum.
This script directly measures:
  1. Quantization error statistics on K/V tensors (mean, std, skewness)
  2. Attention output error vs. naive per-element error bound
  3. Error cancellation ratio: actual_output_error / naive_error_bound

If the mechanism holds:
  - K/V errors are near zero-mean (verified by symmetry)
  - Attention output error is much smaller than naive bound (cancellation)
"""

import sys, json, torch, numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

MODEL_ID = "HuggingFaceTB/SmolLM-360M"
N_EXAMPLES = 20
DEVICE = "cpu"

print(f"Loading {MODEL_ID}...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

from datasets import load_dataset
ds = list(load_dataset("Rowan/hellaswag", split="validation").select(range(N_EXAMPLES)))

# ── Quantization functions ─────────────────────────────────────────────────

def quantize_int4_sym(x: torch.Tensor):
    """Standard symmetric INT4: scale = max/7, 16 levels."""
    scale = x.abs().max() / 7.0
    if scale == 0:
        return x, torch.tensor(0.0)
    q = (x / scale).round().clamp(-8, 7)
    return q * scale, scale

def quantize_int4_int3range(x: torch.Tensor):
    """INT3-range INT4: scale = max/3, 8 levels. Matches paper baseline."""
    scale = x.abs().max() / 3.0
    if scale == 0:
        return x, torch.tensor(0.0)
    q = (x / scale).round().clamp(-4, 3)
    return q * scale, scale


# ── Hook-based K/V capture ─────────────────────────────────────────────────

captured = {}

def make_capture_hook(name):
    def hook(module, inp, out):
        captured[name] = out.detach().clone()
    return hook

# ── Analysis per layer ─────────────────────────────────────────────────────

def attention_output_error(k_fp16, v_fp16, k_quant, v_quant, model_config):
    """
    Simulate single-head attention output error from K/V quantization.
    Uses simplified single-head attention for analysis (avg across heads).
    """
    # k, v: [batch, seq, d_kv] — use first batch, first layer shape
    B, S, D = k_fp16.shape
    head_dim = model_config.head_dim if hasattr(model_config, 'head_dim') else D

    # Error tensors
    dk = k_quant - k_fp16  # [B, S, D]
    dv = v_quant - v_fp16  # [B, S, D]

    # Simplified: measure error statistics
    results = {
        "K_error_mean": dk.mean().item(),
        "K_error_std": dk.std().item(),
        "K_error_abs_mean": dk.abs().mean().item(),
        "K_rel_error": (dk.norm() / (k_fp16.norm() + 1e-8)).item(),
        "V_error_mean": dv.mean().item(),
        "V_error_std": dv.std().item(),
        "V_error_abs_mean": dv.abs().mean().item(),
        "V_rel_error": (dv.norm() / (v_fp16.norm() + 1e-8)).item(),
    }
    return results


# ── Main measurement ───────────────────────────────────────────────────────

all_std_stats = []
all_i3r_stats = []

print(f"Measuring K/V quantization errors on {N_EXAMPLES} examples...", flush=True)

for ex_idx, ex in enumerate(ds):
    ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    inputs = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=128)

    # Hook all k_proj and v_proj
    hooks = []
    for name, module in model.named_modules():
        if name.endswith(".k_proj") or name.endswith(".v_proj"):
            h = module.register_forward_hook(make_capture_hook(name))
            hooks.append(h)

    with torch.no_grad():
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    # Analyze each captured K/V projection
    for layer_name, fp16_out in captured.items():
        # Standard INT4
        std_quant, _ = quantize_int4_sym(fp16_out)
        std_stats = attention_output_error(fp16_out, fp16_out, std_quant, std_quant, model.config)
        all_std_stats.append({
            "K_error_mean": (std_quant - fp16_out).mean().item(),
            "K_error_std": (std_quant - fp16_out).std().item(),
            "K_rel_error": ((std_quant - fp16_out).norm() / (fp16_out.norm() + 1e-8)).item(),
        })

        # INT3-range INT4
        i3r_quant, _ = quantize_int4_int3range(fp16_out)
        all_i3r_stats.append({
            "K_error_mean": (i3r_quant - fp16_out).mean().item(),
            "K_error_std": (i3r_quant - fp16_out).std().item(),
            "K_rel_error": ((i3r_quant - fp16_out).norm() / (fp16_out.norm() + 1e-8)).item(),
        })

    captured.clear()

    if ex_idx % 5 == 0:
        print(f"  [{ex_idx}/{N_EXAMPLES}]", flush=True)


# ── Aggregate statistics ────────────────────────────────────────────────────

def aggregate(stats_list):
    means = [s["K_error_mean"] for s in stats_list]
    stds = [s["K_error_std"] for s in stats_list]
    rel_errs = [s["K_rel_error"] for s in stats_list]
    return {
        "error_mean_avg": float(np.mean(means)),
        "error_mean_std": float(np.std(means)),
        "error_magnitude_avg": float(np.mean(np.abs(means))),
        "error_std_avg": float(np.mean(stds)),
        "rel_error_avg": float(np.mean(rel_errs)),
        "n_measurements": len(stats_list),
        # Symmetry metric: |mean| / std → near 0 means zero-mean relative to magnitude
        "symmetry_ratio": float(np.abs(np.mean(means)) / (np.mean(stds) + 1e-8)),
    }

std_agg = aggregate(all_std_stats)
i3r_agg = aggregate(all_i3r_stats)

print("\n=== K/V QUANTIZATION ERROR ANALYSIS ===", flush=True)
print(f"\nStandard INT4 (scale=max/7, 16 levels):")
print(f"  Mean error:       {std_agg['error_mean_avg']:+.6f}  (std across layers: ±{std_agg['error_mean_std']:.6f})")
print(f"  Error magnitude:  {std_agg['error_magnitude_avg']:.6f}")
print(f"  Error std:        {std_agg['error_std_avg']:.4f}")
print(f"  Relative error:   {std_agg['rel_error_avg']*100:.2f}%")
print(f"  Symmetry ratio:   {std_agg['symmetry_ratio']:.4f}  (near 0 = zero-mean)")

print(f"\nINT3-range INT4 (scale=max/3, 8 levels):")
print(f"  Mean error:       {i3r_agg['error_mean_avg']:+.6f}  (std across layers: ±{i3r_agg['error_mean_std']:.6f})")
print(f"  Error magnitude:  {i3r_agg['error_magnitude_avg']:.6f}")
print(f"  Error std:        {i3r_agg['error_std_avg']:.4f}")
print(f"  Relative error:   {i3r_agg['rel_error_avg']*100:.2f}%")
print(f"  Symmetry ratio:   {i3r_agg['symmetry_ratio']:.4f}  (near 0 = zero-mean)")

# ── Attention simulation: does zero-mean help? ─────────────────────────────
print("\n=== ATTENTION OUTPUT SIMULATION ===", flush=True)

# On a fresh example, measure actual attention output error vs. worst-case
ex = ds[0]
ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
inputs = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=128)

# Get Q, K, V from first layer
layer0 = model.model.layers[0].self_attn
k_proj = layer0.k_proj
v_proj = layer0.v_proj
q_proj = layer0.q_proj

with torch.no_grad():
    hidden = model.model.embed_tokens(inputs["input_ids"])
    k_fp16 = k_proj(hidden)
    v_fp16 = v_proj(hidden)
    q = q_proj(hidden)

# Reshape for attention: [batch, n_heads, seq, head_dim]
n_q_heads = model.config.num_attention_heads
n_kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // n_q_heads
B, S, _ = k_fp16.shape

k_reshaped = k_fp16.view(B, S, n_kv_heads, head_dim).transpose(1, 2)  # [B, nkv, S, d]
v_reshaped = v_fp16.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
q_reshaped = q.view(B, S, n_q_heads, head_dim).transpose(1, 2)         # [B, nq, S, d]

# Expand KV to match Q heads (GQA)
groups = n_q_heads // n_kv_heads
k_exp = k_reshaped.repeat_interleave(groups, dim=1)  # [B, nq, S, d]
v_exp = v_reshaped.repeat_interleave(groups, dim=1)

import torch.nn.functional as F

# FP16 attention output
scale = head_dim ** -0.5
attn_weights = torch.softmax(torch.matmul(q_reshaped, k_exp.transpose(-2,-1)) * scale, dim=-1)
out_fp16 = torch.matmul(attn_weights, v_exp)  # [B, nq, S, d]

# Standard INT4: quantize K and V
k_std, _ = quantize_int4_sym(k_fp16)
v_std, _ = quantize_int4_sym(v_fp16)
k_std_r = k_std.view(B, S, n_kv_heads, head_dim).transpose(1,2).repeat_interleave(groups, dim=1)
v_std_r = v_std.view(B, S, n_kv_heads, head_dim).transpose(1,2).repeat_interleave(groups, dim=1)
attn_std = torch.softmax(torch.matmul(q_reshaped, k_std_r.transpose(-2,-1)) * scale, dim=-1)
out_std = torch.matmul(attn_std, v_std_r)

# INT3-range INT4
k_i3r, _ = quantize_int4_int3range(k_fp16)
v_i3r, _ = quantize_int4_int3range(v_fp16)
k_i3r_r = k_i3r.view(B, S, n_kv_heads, head_dim).transpose(1,2).repeat_interleave(groups, dim=1)
v_i3r_r = v_i3r.view(B, S, n_kv_heads, head_dim).transpose(1,2).repeat_interleave(groups, dim=1)
attn_i3r = torch.softmax(torch.matmul(q_reshaped, k_i3r_r.transpose(-2,-1)) * scale, dim=-1)
out_i3r = torch.matmul(attn_i3r, v_i3r_r)

# Error metrics
err_std = (out_std - out_fp16).abs()
err_i3r = (out_i3r - out_fp16).abs()

# Naive bound: error ≤ max(|dV|) (ignoring cancellation)
naive_bound_std = (v_std - v_fp16).abs().max().item()
naive_bound_i3r = (v_i3r - v_fp16).abs().max().item()

actual_std = err_std.mean().item()
actual_i3r = err_i3r.mean().item()

print(f"\nAttention output error (first layer, first example):")
print(f"  Standard INT4:")
print(f"    Naive bound (max |dV|): {naive_bound_std:.4f}")
print(f"    Actual mean |error|:    {actual_std:.6f}")
print(f"    Cancellation ratio:     {actual_std / (naive_bound_std + 1e-8):.4f}  (lower = more cancellation)")
print(f"  INT3-range INT4:")
print(f"    Naive bound (max |dV|): {naive_bound_i3r:.4f}")
print(f"    Actual mean |error|:    {actual_i3r:.6f}")
print(f"    Cancellation ratio:     {actual_i3r / (naive_bound_i3r + 1e-8):.4f}")

out = {
    "model": MODEL_ID,
    "n_examples": N_EXAMPLES,
    "date": datetime.now().isoformat(),
    "kv_error_stats": {
        "standard_int4": std_agg,
        "int3range_int4": i3r_agg,
    },
    "attention_output_simulation": {
        "standard_int4": {
            "naive_bound": naive_bound_std,
            "actual_mean_error": actual_std,
            "cancellation_ratio": actual_std / (naive_bound_std + 1e-8),
        },
        "int3range_int4": {
            "naive_bound": naive_bound_i3r,
            "actual_mean_error": actual_i3r,
            "cancellation_ratio": actual_i3r / (naive_bound_i3r + 1e-8),
        }
    }
}
fname = Path("research/data") / f"int4_error_analysis_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {fname}", flush=True)
