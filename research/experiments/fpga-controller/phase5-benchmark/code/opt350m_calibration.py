"""
OPT-350M (facebook, hidden=1024, 24L, 16H) beta calibration.
Tests: does OPT family show floor clustering like GPT-2, or does hidden_dim scaling apply?
OPT-125M: hidden=768, gap_mean=0.213. Naive prediction for 350M (hidden=1024): higher.
"""
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "facebook/opt-350m"
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
    return (x / scale).round().clamp(-7, 7) * scale

def quant_int8(x):
    scale = x.abs().max(dim=-1, keepdim=True).values / 127.0
    scale = scale.clamp(min=1e-8)
    return (x / scale).round().clamp(-127, 127) * scale

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

cfg = model.config
print(f"hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, heads={cfg.num_attention_heads}")

signals = []
hooks = []
k_buf = {}  # layer_idx -> k tensor

def make_k_hook(layer_idx):
    def hook_fn(module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        k_buf[layer_idx] = x.detach().clone()
    return hook_fn

def make_v_hook(layer_idx):
    def hook_fn(module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        v = x.detach()
        k = k_buf.get(layer_idx)
        if k is None:
            return
        for b in range(k.shape[0]):
            for t in range(k.shape[1]):
                kv = torch.cat([k[b, t], v[b, t]], dim=0)  # concatenated, matches other checkpoints
                q4 = 1.0 - (quant_int4(kv) - kv).norm() / (kv.norm() + 1e-8)
                q8 = 1.0 - (quant_int8(kv) - kv).norm() / (kv.norm() + 1e-8)
                signals.append((q4.item(), q8.item()))
    return hook_fn

# OPT uses self_attn.k_proj and v_proj — concatenate k+v per token (consistent with other checkpoints)
for i, layer in enumerate(model.model.decoder.layers):
    hooks.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(i)))
    hooks.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(i)))

with torch.no_grad():
    for text in TEXTS:
        ids = tokenizer(text, return_tensors="pt", max_length=MAX_LEN, truncation=True)
        model(**ids)

for h in hooks:
    h.remove()

# Pair k and v signals (alternating: k, v, k, v, ...)
# Each layer contributes 2 hooks, each hook contributes T signals per text
# Re-pair: group by token-layer to get combined k+v signal
# Actually simpler: treat each hook output independently as a signal (consistent with OPT-125M method)
q4s = np.array([s[0] for s in signals])
q8s = np.array([s[1] for s in signals])
gaps = q8s - q4s
gap_mean = float(gaps.mean())
gap_std = float(gaps.std())
beta_star = gap_mean / 0.267
n_signals = len(signals)

print(f"\nOPT-350M results:")
print(f"  n_signals: {n_signals}")
print(f"  gap_mean:  {gap_mean:.4f}")
print(f"  gap_std:   {gap_std:.4f}")
print(f"  beta*:     {beta_star:.3f}")
print(f"\nOPT family comparison:")
print(f"  OPT-125M (hidden=768):  gap_mean=0.2131, beta*=0.798")
print(f"  OPT-350M (hidden=1024): gap_mean={gap_mean:.4f}, beta*={beta_star:.3f}")

result = {
    "model": MODEL_ID, "n_params": "350M",
    "hidden_size": cfg.hidden_size, "n_layers": cfg.num_hidden_layers,
    "n_heads": cfg.num_attention_heads, "n_signals": n_signals,
    "gap_mean": gap_mean, "gap_std": gap_std, "beta_star": beta_star,
}
with open(f"{RESULTS_DIR}/opt350m_cal.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nRunning beta sweep...")
betas = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]
sweep = {}
for beta in betas:
    thr = beta * 0.267
    p4 = float((gaps < thr).mean()) * 100
    regime = "4-bit" if p4 > 90 else "8-bit" if p4 < 10 else "MIXED"
    sweep[str(beta)] = {"threshold": round(thr,4), "p4_pct": round(p4,1), "regime": regime}
    print(f"  b={beta:.2f} thr={thr:.4f} p4={p4:.1f}% [{regime}]")

# Detect 50% crossing
prev_p4 = None
crossing = None
for b_str, v in sweep.items():
    p4 = v["p4_pct"]
    if prev_p4 is not None and prev_p4 < 50 and p4 >= 50:
        crossing = float(b_str)
    prev_p4 = p4

if crossing:
    error = abs(crossing - beta_star)
    confirmed = error <= 0.04
    print(f"\n50%-4bit crossing at beta={crossing:.2f}, theory={beta_star:.3f}, error={error:.3f}, confirmed={confirmed}")
    sweep["crossing_50pct"] = crossing
    sweep["formula_error"] = round(error, 3)
    sweep["confirmed"] = confirmed

with open(f"{RESULTS_DIR}/opt350m_sweep.json", "w") as f:
    json.dump(sweep, f, indent=2)
print(f"Saved results.")
