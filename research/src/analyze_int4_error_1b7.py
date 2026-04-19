"""
Mechanistic verification of INT4 losslessness at SmolLM-1.7B.

Hypothesis: at 1.7B (32 attention heads vs 15 at 360M), the zero-mean
error cancellation mechanism is WEAKER — higher activation variance and
richer inter-head structure reduce cancellation effectiveness.

Expected result:
  - Symmetry ratio remains near 0 (errors are still zero-mean)
  - BUT relative error magnitude is HIGHER than at 360M
  - AND/OR cancellation ratio is CLOSER to 1 (less cancellation)
  - Together these explain the 10pp accuracy gap at 1.7B

Compare to 360M results:
  Standard INT4: symmetry=0.0027, rel_error=26.95%, cancellation_ratio=0.30 (3.3x below naive)
  INT3-range:    symmetry=0.0037, rel_error=55.79%, cancellation_ratio=0.24 (4.2x below naive)
"""

import sys, json, torch, numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

MODEL_ID = "HuggingFaceTB/SmolLM-1.7B"
N_EXAMPLES = 20
DEVICE = "cpu"

print(f"Loading {MODEL_ID}...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

from datasets import load_dataset
ds = list(load_dataset("Rowan/hellaswag", split="validation").select(range(N_EXAMPLES)))

print(f"Model: {MODEL_ID}")
print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
print(f"  num_attention_heads: {model.config.num_attention_heads}")
print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
print(f"  hidden_size: {model.config.hidden_size}", flush=True)


def quantize_int4_sym(x):
    scale = x.abs().max() / 7.0
    if scale == 0: return x, torch.tensor(0.0)
    q = (x / scale).round().clamp(-8, 7)
    return q * scale, scale

def quantize_int4_int3range(x):
    scale = x.abs().max() / 3.0
    if scale == 0: return x, torch.tensor(0.0)
    q = (x / scale).round().clamp(-4, 3)
    return q * scale, scale


captured = {}

def make_capture_hook(name):
    def hook(module, inp, out):
        captured[name] = out.detach().clone()
    return hook


all_std_stats = []
all_i3r_stats = []

print(f"\nMeasuring K/V quantization errors on {N_EXAMPLES} examples...", flush=True)

for ex_idx, ex in enumerate(ds):
    ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    inputs = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=128)

    hooks = []
    for name, module in model.named_modules():
        if name.endswith(".k_proj") or name.endswith(".v_proj"):
            h = module.register_forward_hook(make_capture_hook(name))
            hooks.append(h)

    with torch.no_grad():
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    for layer_name, fp16_out in captured.items():
        std_quant, _ = quantize_int4_sym(fp16_out)
        diff_std = std_quant - fp16_out
        all_std_stats.append({
            "K_error_mean": diff_std.mean().item(),
            "K_error_std": diff_std.std().item(),
            "K_rel_error": (diff_std.norm() / (fp16_out.norm() + 1e-8)).item(),
        })
        i3r_quant, _ = quantize_int4_int3range(fp16_out)
        diff_i3r = i3r_quant - fp16_out
        all_i3r_stats.append({
            "K_error_mean": diff_i3r.mean().item(),
            "K_error_std": diff_i3r.std().item(),
            "K_rel_error": (diff_i3r.norm() / (fp16_out.norm() + 1e-8)).item(),
        })

    captured.clear()
    if ex_idx % 5 == 0:
        print(f"  [{ex_idx}/{N_EXAMPLES}]", flush=True)


def aggregate(stats_list):
    means = [s["K_error_mean"] for s in stats_list]
    stds = [s["K_error_std"] for s in stats_list]
    rel_errs = [s["K_rel_error"] for s in stats_list]
    return {
        "error_mean_avg": float(np.mean(means)),
        "error_mean_std": float(np.std(means)),
        "error_std_avg": float(np.mean(stds)),
        "rel_error_avg": float(np.mean(rel_errs)),
        "n_measurements": len(stats_list),
        "symmetry_ratio": float(np.abs(np.mean(means)) / (np.mean(stds) + 1e-8)),
    }

std_agg = aggregate(all_std_stats)
i3r_agg = aggregate(all_i3r_stats)

# ── Attention output simulation ─────────────────────────────────────────────
ex = ds[0]
ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
inputs = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=128)

layer0 = model.model.layers[0].self_attn
k_proj = layer0.k_proj
v_proj = layer0.v_proj
q_proj = layer0.q_proj

with torch.no_grad():
    hidden = model.model.embed_tokens(inputs["input_ids"])
    k_fp16 = k_proj(hidden)
    v_fp16 = v_proj(hidden)
    q = q_proj(hidden)

n_q_heads = model.config.num_attention_heads
n_kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // n_q_heads
B, S, _ = k_fp16.shape

import torch.nn.functional as F

k_r = k_fp16.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
v_r = v_fp16.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
q_r = q.view(B, S, n_q_heads, head_dim).transpose(1, 2)
groups = n_q_heads // n_kv_heads
k_exp = k_r.repeat_interleave(groups, dim=1)
v_exp = v_r.repeat_interleave(groups, dim=1)

scale = head_dim ** -0.5
attn_w = torch.softmax(torch.matmul(q_r, k_exp.transpose(-2,-1)) * scale, dim=-1)
out_fp16 = torch.matmul(attn_w, v_exp)

def attn_output(k_in, v_in):
    kr = k_in.view(B, S, n_kv_heads, head_dim).transpose(1,2).repeat_interleave(groups, dim=1)
    vr = v_in.view(B, S, n_kv_heads, head_dim).transpose(1,2).repeat_interleave(groups, dim=1)
    aw = torch.softmax(torch.matmul(q_r, kr.transpose(-2,-1)) * scale, dim=-1)
    return torch.matmul(aw, vr)

k_std, _ = quantize_int4_sym(k_fp16)
v_std, _ = quantize_int4_sym(v_fp16)
out_std = attn_output(k_std, v_std)

k_i3r, _ = quantize_int4_int3range(k_fp16)
v_i3r, _ = quantize_int4_int3range(v_fp16)
out_i3r = attn_output(k_i3r, v_i3r)

naive_std = (v_std - v_fp16).abs().max().item()
naive_i3r = (v_i3r - v_fp16).abs().max().item()
actual_std = (out_std - out_fp16).abs().mean().item()
actual_i3r = (out_i3r - out_fp16).abs().mean().item()
cancel_std = actual_std / (naive_std + 1e-8)
cancel_i3r = actual_i3r / (naive_i3r + 1e-8)

print("\n=== K/V ERROR ANALYSIS (SmolLM-1.7B) ===")
print(f"\nStandard INT4 (scale=max/7, 16 levels):")
print(f"  Mean error:       {std_agg['error_mean_avg']:+.6f}")
print(f"  Relative error:   {std_agg['rel_error_avg']*100:.2f}%")
print(f"  Symmetry ratio:   {std_agg['symmetry_ratio']:.4f}  (360M was: 0.0027)")
print(f"  Cancellation:     {cancel_std:.4f}  (360M was: 0.30)")
print(f"\nINT3-range INT4 (scale=max/3, 8 levels):")
print(f"  Mean error:       {i3r_agg['error_mean_avg']:+.6f}")
print(f"  Relative error:   {i3r_agg['rel_error_avg']*100:.2f}%")
print(f"  Symmetry ratio:   {i3r_agg['symmetry_ratio']:.4f}  (360M was: 0.0037)")
print(f"  Cancellation:     {cancel_i3r:.4f}  (360M was: 0.24)")

print(f"\n=== COMPARISON TO 360M ===")
print(f"  360M std INT4 rel_error: 26.95%  |  1.7B: {std_agg['rel_error_avg']*100:.2f}%")
print(f"  360M std INT4 cancellation: 0.30 |  1.7B: {cancel_std:.4f}")
print(f"  Hypothesis: 1.7B has HIGHER rel_error OR WEAKER cancellation → explains lossiness")

out = {
    "model": MODEL_ID,
    "n_examples": N_EXAMPLES,
    "date": datetime.now().isoformat(),
    "num_attention_heads": n_q_heads,
    "num_kv_heads": n_kv_heads,
    "kv_error_stats": {"standard_int4": std_agg, "int3range_int4": i3r_agg},
    "attention_output_simulation": {
        "standard_int4": {"naive_bound": naive_std, "actual_mean_error": actual_std, "cancellation_ratio": cancel_std},
        "int3range_int4": {"naive_bound": naive_i3r, "actual_mean_error": actual_i3r, "cancellation_ratio": cancel_i3r},
    },
    "comparison_360m": {
        "std_int4_rel_error_360m": 0.2695,
        "std_int4_cancellation_360m": 0.30,
        "std_int4_symmetry_360m": 0.0027,
    }
}
fname = Path("research/data") / f"int4_error_1b7_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {fname}")
