"""
Calibration corpus sensitivity for GPT-2 Medium (345M, OpenAI Conv1D).
GPT-2 Medium is in the floor cluster (gap_mean=0.188, beta*=0.704).
Tests: is calibration sensitivity scale-invariant within the GPT-2 family?
GPT-2 Small: gap_std=0.033, max_error=0.018. Does Medium (gap_std~0.045) differ?
"""
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "gpt2-medium"
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
    scale = x.abs().max().clamp(min=1e-8) / 7.0
    x_q = (x / scale).round().clamp(-7, 7) * scale
    return float(1.0 - (x_q - x).norm() / x.norm().clamp(min=1e-8))

def quant_int8(x):
    scale = x.abs().max().clamp(min=1e-8) / 127.0
    x_q = (x / scale).round().clamp(-127, 127) * scale
    return float(1.0 - (x_q - x).norm() / x.norm().clamp(min=1e-8))

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, local_files_only=True)
model.eval()
n_layers = model.config.n_layer

def extract_gaps(text_list):
    """GPT-2 uses combined c_attn (QKV fused). Extract K and V from the split."""
    signals = []
    hooks = []
    kv_store = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # c_attn output: [batch, seq, 3*n_embd] — split into Q, K, V
            x = out if not isinstance(out, tuple) else out[0]
            n_embd = x.shape[-1] // 3
            k = x[..., n_embd:2*n_embd].detach().float()
            v = x[..., 2*n_embd:].detach().float()
            kv_store[layer_idx] = (k, v)
        return hook

    for i in range(n_layers):
        hooks.append(model.transformer.h[i].attn.c_attn.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for text in text_list:
            ids = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            kv_store.clear()
            model(**ids)
            for i in range(n_layers):
                kv = kv_store.get(i)
                if kv is None:
                    continue
                k, v = kv
                for b in range(k.shape[0]):
                    for t in range(k.shape[1]):
                        vec = torch.cat([k[b, t], v[b, t]], dim=0)
                        q4 = quant_int4(vec)
                        q8 = quant_int8(vec)
                        signals.append((q4, q8))

    for h in hooks:
        h.remove()
    return signals

# 10-text aggregate
print("Running 10-text aggregate...")
all_signals = extract_gaps(TEXTS)
gaps_all = np.array([s[1] - s[0] for s in all_signals])
gap_mean_10 = float(gaps_all.mean())
gap_std_10 = float(gaps_all.std())
beta_star_10 = gap_mean_10 / 0.267
print(f"10-text: gap_mean={gap_mean_10:.4f}, gap_std={gap_std_10:.4f}, beta*={beta_star_10:.3f}, n={len(all_signals)}")

# Per-text estimates
print("\nPer-text sensitivity:")
per_text = []
for i, text in enumerate(TEXTS):
    sigs = extract_gaps([text])
    gaps = np.array([s[1] - s[0] for s in sigs])
    gm = float(gaps.mean())
    bs = gm / 0.267
    error = abs(bs - beta_star_10)
    per_text.append({
        "text_idx": i,
        "n_tokens": len(sigs),
        "gap_mean": round(gm, 4),
        "gap_std": round(float(gaps.std()), 4),
        "beta_star": round(bs, 3),
        "error_vs_10text": round(error, 3)
    })
    print(f"  text {i+1}: n={len(sigs):4d}, gap_mean={gm:.4f}, gap_std={gaps.std():.4f}, beta*={bs:.3f}, error={error:.3f}")

max_error = max(x["error_vs_10text"] for x in per_text)
mean_error = float(np.mean([x["error_vs_10text"] for x in per_text]))
print(f"\n1-text corpus sensitivity (GPT-2 Medium 345M):")
print(f"  10-text gap_std: {gap_std_10:.4f}")
print(f"  max error vs 10-text: {max_error:.3f}")
print(f"  mean error vs 10-text: {mean_error:.3f}")
print(f"  All within +-0.015: {max_error <= 0.015}")
print(f"  All within +-0.020: {max_error <= 0.020}")

result = {
    "model": MODEL_ID,
    "architecture": "GPT-2/Conv1D (OpenAI, 345M, 24L, 16H)",
    "note": "Within-GPT2-family sensitivity. Gap_mean at floor (0.188). Tests scale-invariance of calibration sensitivity.",
    "gap_mean_10text": round(gap_mean_10, 4),
    "gap_std_10text": round(gap_std_10, 4),
    "beta_star_10text": round(beta_star_10, 3),
    "per_text_results": per_text,
    "max_error_1text": round(max_error, 4),
    "mean_error_1text": round(mean_error, 4),
    "confirmed_015": max_error <= 0.015,
    "confirmed_020": max_error <= 0.020,
}
with open(f"{RESULTS_DIR}/gpt2_medium_cal_sensitivity.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved gpt2_medium_cal_sensitivity.json")
