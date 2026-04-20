"""
Fine beta sweep for SmolLM2-360M around the transition zone [1.05, 1.15].
Purpose: Determine precise 50%-4bit crossing to refine formula error from coarse interpolation (0.044).
Theory: beta* = 0.2826/0.267 = 1.058. If crossing <= 1.098, error <= 0.040 -> confirmed.
"""
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, local_files_only=True)
model.eval()

signals = []
hooks = []
k_buf = {}

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
                kv = torch.cat([k[b, t], v[b, t]], dim=0)
                q4 = 1.0 - (quant_int4(kv) - kv).norm() / (kv.norm() + 1e-8)
                q8 = 1.0 - (quant_int8(kv) - kv).norm() / (kv.norm() + 1e-8)
                signals.append((q4.item(), q8.item()))
    return hook_fn

for i, layer in enumerate(model.model.layers):
    hooks.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(i)))
    hooks.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(i)))

with torch.no_grad():
    for text in TEXTS:
        ids = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        model(**ids)

for h in hooks:
    h.remove()

gaps = np.array([s[1] - s[0] for s in signals])
gap_mean = float(gaps.mean())
beta_star = gap_mean / 0.267
print(f"gap_mean={gap_mean:.4f}, beta*={beta_star:.3f}, n={len(gaps)}")

# Fine sweep around predicted transition
betas = [1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.15, 1.20]
print("\nFine beta sweep:")
results = {}
prev_p4 = None
crossing = None
for beta in betas:
    thr = beta * 0.267
    p4 = float((gaps < thr).mean()) * 100
    results[beta] = p4
    regime = "4-bit" if p4 > 90 else "8-bit" if p4 < 10 else "MIXED"
    print(f"  b={beta:.2f} thr={thr:.4f} p4={p4:.1f}% [{regime}]")
    if prev_p4 is not None and prev_p4 < 50 and p4 >= 50:
        crossing = beta
    prev_p4 = p4

# Interpolate precise crossing
crossing_interp = None
betas_list = list(results.keys())
for i in range(len(betas_list)-1):
    b1, b2 = betas_list[i], betas_list[i+1]
    p1, p2 = results[b1], results[b2]
    if p1 < 50 <= p2:
        crossing_interp = b1 + (50 - p1)/(p2 - p1) * (b2 - b1)
        break

if crossing_interp:
    error = abs(crossing_interp - beta_star)
    confirmed_04 = error <= 0.040
    confirmed_05 = error <= 0.050
    print(f"\nInterpolated 50%-crossing: beta={crossing_interp:.4f}")
    print(f"Theory beta*={beta_star:.3f}")
    print(f"Formula error: {error:.4f}")
    print(f"Confirmed (<=0.040): {confirmed_04}")
    print(f"Confirmed (<=0.050): {confirmed_05}")
    with open(f"{RESULTS_DIR}/smollm2_fine_sweep.json", "w") as f:
        json.dump({
            "gap_mean": gap_mean, "beta_star": beta_star,
            "crossing_interp": round(crossing_interp, 4),
            "formula_error": round(error, 4),
            "confirmed_04": confirmed_04, "confirmed_05": confirmed_05,
            "sweep": {str(b): round(p, 1) for b, p in results.items()}
        }, f, indent=2)
    print("Saved smollm2_fine_sweep.json")
