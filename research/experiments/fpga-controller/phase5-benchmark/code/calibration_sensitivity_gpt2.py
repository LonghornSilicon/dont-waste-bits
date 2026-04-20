"""
Calibration corpus sensitivity analysis for GPT-2 Small.
Shows that a 1-text corpus estimates beta* within ±0.015 of 10-text aggregate.
Cross-validates the SmolLM-360M sensitivity result on a different architecture.
"""
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "gpt2"
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
cfg = model.config
hidden = cfg.n_embd
n_heads = cfg.n_head

def extract_gaps(text_list):
    """Extract all gap signals for a list of texts."""
    signals = []
    hooks = []
    k_buf = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            # GPT-2 c_attn output: [batch, seq, 3*hidden]; split Q,K,V
            k = x[..., hidden:2*hidden]
            v = x[..., 2*hidden:]
            k_buf[layer_idx] = {'k': k.detach().clone(), 'v': v.detach()}
        return hook_fn

    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.c_attn.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for text in text_list:
            ids = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            model(**ids)

    for h in hooks:
        h.remove()

    for layer_idx, buf in k_buf.items():
        k = buf['k']
        v = buf['v']
        for b in range(k.shape[0]):
            for t in range(k.shape[1]):
                kv = torch.cat([k[b, t], v[b, t]], dim=0)
                q4 = 1.0 - (quant_int4(kv) - kv).norm() / (kv.norm() + 1e-8)
                q8 = 1.0 - (quant_int8(kv) - kv).norm() / (kv.norm() + 1e-8)
                signals.append((q4.item(), q8.item()))
    return signals

# 10-text aggregate (ground truth)
print("Running 10-text aggregate...")
all_signals = extract_gaps(TEXTS)
gaps_all = np.array([s[1] - s[0] for s in all_signals])
gap_mean_10 = float(gaps_all.mean())
beta_star_10 = gap_mean_10 / 0.267
print(f"10-text: gap_mean={gap_mean_10:.4f}, beta*={beta_star_10:.3f}, n={len(all_signals)}")

# Per-text estimates
print("\nPer-text sensitivity:")
per_text = []
for i, text in enumerate(TEXTS):
    sigs = extract_gaps([text])
    gaps = np.array([s[1] - s[0] for s in sigs])
    gm = float(gaps.mean())
    bs = gm / 0.267
    error = abs(bs - beta_star_10)
    per_text.append({"text_idx": i, "n_tokens": len(sigs), "gap_mean": round(gm, 4),
                     "beta_star": round(bs, 3), "error_vs_10text": round(error, 3)})
    print(f"  text {i+1}: n={len(sigs):3d}, gap_mean={gm:.4f}, beta*={bs:.3f}, error={error:.3f}")

max_error = max(x["error_vs_10text"] for x in per_text)
mean_error = float(np.mean([x["error_vs_10text"] for x in per_text]))
print(f"\n1-text corpus sensitivity (GPT-2 Small):")
print(f"  max error vs 10-text: {max_error:.3f}")
print(f"  mean error vs 10-text: {mean_error:.3f}")
print(f"  All within ±0.015: {max_error <= 0.015}")
print(f"  All within ±0.020: {max_error <= 0.020}")

result = {
    "model": MODEL_ID,
    "gap_mean_10text": round(gap_mean_10, 4),
    "beta_star_10text": round(beta_star_10, 3),
    "per_text_results": per_text,
    "max_error_1text": round(max_error, 4),
    "mean_error_1text": round(mean_error, 4),
    "confirmed_015": max_error <= 0.015,
    "confirmed_020": max_error <= 0.020,
}
with open(f"{RESULTS_DIR}/gpt2_cal_sensitivity.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved gpt2_cal_sensitivity.json")
