"""
GPT-2 Large (774M, d=1280, 36 layers, 20 heads) beta calibration.
Tests floor gap_mean hypothesis: does Large continue downward from Medium (0.188)?
Same Conv1D c_attn architecture as Small/Medium.
"""

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "openai/gpt2-large"
N_TEXTS = 10
MAX_LEN = 128
RESULTS_DIR = r"C:\Users\themo\Desktop\Dont Waste Bits!\research\experiments\fpga-controller\phase5-benchmark\results"

TEXTS = [
    "The transformer architecture has revolutionized natural language processing.",
    "Attention mechanisms allow models to focus on relevant parts of the input.",
    "Large language models require significant computational resources to train.",
    "Quantization reduces memory footprint while preserving model accuracy.",
    "The key-value cache stores intermediate attention states during inference.",
    "Hardware-aware optimization can significantly improve inference throughput.",
    "Grouped query attention reduces the number of key and value heads.",
    "Instruction fine-tuning adapts pre-trained models to follow human directions.",
    "The FPGA BRAM port has a minimum width of four bits on Xilinx devices.",
    "Cross-architecture validation ensures formula generalization across model families.",
]

def quant_int4(x):
    scale = x.abs().max(dim=-1, keepdim=True).values / 7.0
    scale = scale.clamp(min=1e-8)
    xq = (x / scale).round().clamp(-7, 7) * scale
    return xq

def quant_int8(x):
    scale = x.abs().max(dim=-1, keepdim=True).values / 127.0
    scale = scale.clamp(min=1e-8)
    xq = (x / scale).round().clamp(-127, 127) * scale
    return xq

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

config = model.config
n_heads = config.n_head
n_embd = config.n_embd
head_dim = n_embd // n_heads
n_layers = config.n_layer
print(f"Config: n_heads={n_heads}, n_embd={n_embd}, head_dim={head_dim}, n_layers={n_layers}")

signals = []
hooks = []

def make_hook(layer_idx):
    def hook_fn(module, inp, out):
        # out: (B, T, 3*n_embd) — split into Q, K, V
        hidden = out.shape[-1] // 3
        k = out[..., hidden:2*hidden].detach()   # (B, T, n_embd)
        v = out[..., 2*hidden:].detach()
        for b in range(k.shape[0]):
            for t in range(k.shape[1]):
                kv = torch.cat([k[b, t], v[b, t]], dim=0)
                q4 = 1.0 - (quant_int4(kv) - kv).norm() / (kv.norm() + 1e-8)
                q8 = 1.0 - (quant_int8(kv) - kv).norm() / (kv.norm() + 1e-8)
                signals.append((q4.item(), q8.item()))
    return hook_fn

for i, block in enumerate(model.transformer.h):
    h = block.attn.c_attn.register_forward_hook(make_hook(i))
    hooks.append(h)

with torch.no_grad():
    for text in TEXTS:
        ids = tokenizer(text, return_tensors="pt", max_length=MAX_LEN, truncation=True)
        model(**ids)

for h in hooks:
    h.remove()

q4s = np.array([s[0] for s in signals])
q8s = np.array([s[1] for s in signals])
gaps = q8s - q4s
gap_mean = float(gaps.mean())
gap_std = float(gaps.std())
beta_star = gap_mean / 0.267
n_signals = len(signals)

print(f"\nGPT-2 Large results:")
print(f"  n_signals: {n_signals}")
print(f"  gap_mean:  {gap_mean:.4f}")
print(f"  gap_std:   {gap_std:.4f}")
print(f"  beta*:     {beta_star:.3f}")
print(f"\nComparison:")
print(f"  GPT-2 Small  (124M, d=768):  gap_mean=0.1956, beta*=0.733")
print(f"  GPT-2 Medium (345M, d=1024): gap_mean=0.1880, beta*=0.704")
print(f"  GPT-2 Large  (774M, d=1280): gap_mean={gap_mean:.4f}, beta*={beta_star:.3f}")

result = {
    "model": MODEL_ID,
    "n_params": "774M",
    "n_embd": n_embd,
    "n_heads": n_heads,
    "head_dim": head_dim,
    "n_layers": n_layers,
    "n_signals": n_signals,
    "gap_mean": gap_mean,
    "gap_std": gap_std,
    "beta_star": beta_star,
    "formula": "beta* = gap_mean / 0.267"
}

out_path = f"{RESULTS_DIR}/gpt2_large_cal.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out_path}")

# Now do beta sweep
print("\nRunning beta sweep [0.60, 0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.80]...")
betas = [0.60, 0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.80]
sweep_results = {}
all_gaps = gaps  # reuse

for beta in betas:
    threshold = beta * 0.267
    p4 = float((all_gaps < threshold).mean()) * 100
    regime = "4-bit" if p4 > 90 else "8-bit" if p4 < 10 else "MIXED"
    sweep_results[str(beta)] = {"threshold": round(threshold, 4), "p4_pct": round(p4, 1), "regime": regime}
    print(f"  b={beta:.2f} thr={threshold:.4f} gap={gap_mean:.4f} p4={p4:.1f}% [{regime}]")

# Detect transition
prev_regime = None
transition_start = None
for beta, vals in sweep_results.items():
    regime = vals["regime"]
    if prev_regime == "8-bit" and regime in ("MIXED", "4-bit"):
        transition_start = float(beta)
    prev_regime = regime

if transition_start:
    mid = transition_start
    error = abs(mid - beta_star)
    confirmed = error <= 0.04
    print(f"\nMeasured transition start: {transition_start}, mid={mid:.3f}, error={error:.3f} -- Confirmed <=0.04: {confirmed}")
    sweep_results["transition_start"] = transition_start
    sweep_results["formula_error"] = round(error, 3)
    sweep_results["confirmed"] = confirmed

out_path2 = f"{RESULTS_DIR}/gpt2_large_sweep.json"
with open(out_path2, "w") as f:
    json.dump(sweep_results, f, indent=2)
print(f"Saved sweep to {out_path2}")
