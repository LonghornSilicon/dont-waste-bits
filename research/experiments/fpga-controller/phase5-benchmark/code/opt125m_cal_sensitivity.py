"""
Calibration corpus sensitivity for OPT-125M (Meta decoder-only).
Closes the 4-architecture formal verification set (LLaMA-MHA, LLaMA-GQA, Conv1D, OPT).
Inline paper result: max_error=0.006 (best of all architectures).
"""
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "facebook/opt-125m"
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

n_layers = model.config.num_hidden_layers

def extract_gaps(text_list):
    """OPT uses model.decoder.layers[i].self_attn.{k_proj, v_proj}."""
    k_buf = {}
    hooks = []

    def make_k_hook(i):
        def hook(module, inp, out):
            k_buf[i] = out.detach().float()
        return hook

    def make_v_hook(i):
        def hook(module, inp, out):
            k = k_buf.get(i)
            if k is None:
                return
            signals_local = []
            # out: [batch, seq, kv_dim] — OPT uses MHA (no GQA)
            for b in range(k.shape[0]):
                for t in range(k.shape[1]):
                    kv = torch.cat([k[b, t], out[b, t].detach()], dim=0)
                    q4 = quant_int4(kv)
                    q8 = quant_int8(kv)
                    signals_local.append((q4, q8))
            return signals_local
        return hook

    all_signals = []
    # Collect via paired approach
    k_store = {}
    v_store = {}

    def make_k_h(i):
        def h(module, inp, out):
            k_store[i] = out.detach().float()
        return h

    def make_v_h(i):
        def h(module, inp, out):
            v_store[i] = out.detach().float()
        return h

    for i in range(n_layers):
        layer = model.model.decoder.layers[i]
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_k_h(i)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_v_h(i)))

    signals = []
    with torch.no_grad():
        for text in text_list:
            ids = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            k_store.clear()
            v_store.clear()
            model(**ids)
            for i in range(n_layers):
                k = k_store.get(i)
                v = v_store.get(i)
                if k is None or v is None:
                    continue
                for b in range(k.shape[0]):
                    for t in range(k.shape[1]):
                        kv = torch.cat([k[b, t], v[b, t]], dim=0)
                        q4 = quant_int4(kv)
                        q8 = quant_int8(kv)
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
print(f"\n1-text corpus sensitivity (OPT-125M):")
print(f"  10-text gap_std: {gap_std_10:.4f}")
print(f"  max error vs 10-text: {max_error:.3f}")
print(f"  mean error vs 10-text: {mean_error:.3f}")
print(f"  All within +-0.015: {max_error <= 0.015}")
print(f"  All within +-0.020: {max_error <= 0.020}")

result = {
    "model": MODEL_ID,
    "architecture": "OPT/Meta (decoder-only, separate k/v projections)",
    "note": "Formal verification of inline paper result (max_error~=0.006). 4th architecture type.",
    "gap_mean_10text": round(gap_mean_10, 4),
    "gap_std_10text": round(gap_std_10, 4),
    "beta_star_10text": round(beta_star_10, 3),
    "per_text_results": per_text,
    "max_error_1text": round(max_error, 4),
    "mean_error_1text": round(mean_error, 4),
    "confirmed_015": max_error <= 0.015,
    "confirmed_020": max_error <= 0.020,
}
with open(f"{RESULTS_DIR}/opt125m_cal_sensitivity.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved opt125m_cal_sensitivity.json")
