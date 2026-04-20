"""
Calibration sensitivity for OPT-350M (Meta, 1024d, 24L, 16H).
Extends the OPT within-family comparison: OPT-125M (max_error=0.006) → OPT-350M (?).
Predicts: larger OPT model → lower max_error (scale-driven, same pattern as GPT-2 family).
Methodology: paired [k,v] concatenation, consistent with opt125m_cal_sensitivity.py.
"""
import torch, json, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "facebook/opt-350m"
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

def quant_quality(x, levels):
    scale = x.abs().max().clamp(min=1e-8) / levels
    xq = (x / scale).round().clamp(-levels, levels) * scale
    return float(1.0 - (xq - x).norm() / x.norm().clamp(min=1e-8))

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, local_files_only=True)
model.eval()
n_layers = model.config.num_hidden_layers

def extract_gaps(text_list):
    signals, kv_store = [], {}
    hooks = []

    def make_hook(i):
        def h(module, inp, out):
            kv_store[("k", i)] = out.detach().float()
        return h

    def make_v_hook(i):
        def h(module, inp, out):
            kv_store[("v", i)] = out.detach().float()
        return h

    for i in range(n_layers):
        hooks.append(model.model.decoder.layers[i].self_attn.k_proj.register_forward_hook(make_hook(i)))
        hooks.append(model.model.decoder.layers[i].self_attn.v_proj.register_forward_hook(make_v_hook(i)))

    with torch.no_grad():
        for text in text_list:
            ids = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            kv_store.clear()
            model(**ids)
            for i in range(n_layers):
                k = kv_store.get(("k", i))
                v = kv_store.get(("v", i))
                if k is None or v is None:
                    continue
                for b in range(k.shape[0]):
                    for t in range(k.shape[1]):
                        vec = torch.cat([k[b, t], v[b, t]], dim=0)
                        signals.append((quant_quality(vec, 7), quant_quality(vec, 127)))

    for h in hooks:
        h.remove()
    return signals

print("Running 10-text aggregate...")
all_sig = extract_gaps(TEXTS)
gaps = np.array([s[1] - s[0] for s in all_sig])
gm10, gs10, bs10 = gaps.mean(), gaps.std(), gaps.mean() / 0.267
print(f"10-text: gap_mean={gm10:.4f}, gap_std={gs10:.4f}, beta*={bs10:.3f}, n={len(all_sig)}")

print("\nPer-text:")
per_text = []
for i, text in enumerate(TEXTS):
    s = extract_gaps([text])
    g = np.array([x[1] - x[0] for x in s])
    gm, bs, err = g.mean(), g.mean() / 0.267, abs(g.mean() / 0.267 - bs10)
    per_text.append({"text_idx": i, "n_tokens": len(s), "gap_mean": round(float(gm), 4),
                     "gap_std": round(float(g.std()), 4), "beta_star": round(float(bs), 3),
                     "error_vs_10text": round(float(err), 3)})
    print(f"  text {i+1}: n={len(s):4d}, gap_mean={gm:.4f}, gap_std={g.std():.4f}, beta*={bs:.3f}, error={err:.3f}")

max_e = max(x["error_vs_10text"] for x in per_text)
mean_e = float(np.mean([x["error_vs_10text"] for x in per_text]))
print(f"\nOPT-350M sensitivity: max_error={max_e:.3f}, mean_error={mean_e:.3f}")
print(f"  OPT-125M was: max_error=0.006, mean_error=0.002")
print(f"  Within +-0.015: {max_e <= 0.015}  Within +-0.020: {max_e <= 0.020}")

result = {"model": MODEL_ID, "architecture": "OPT/Meta (350M, 1024d, 24L, 16H)",
          "note": "OPT within-family comparison: 125M (max_error=0.006) vs 350M. Scale-driven calibration improvement prediction.",
          "gap_mean_10text": round(float(gm10), 4), "gap_std_10text": round(float(gs10), 4),
          "beta_star_10text": round(float(bs10), 3), "per_text_results": per_text,
          "max_error_1text": round(float(max_e), 4), "mean_error_1text": round(float(mean_e), 4),
          "confirmed_015": max_e <= 0.015, "confirmed_020": max_e <= 0.020}
with open(f"{RESULTS_DIR}/opt350m_cal_sensitivity.json", "w") as f:
    json.dump(result, f, indent=2)
print("Saved opt350m_cal_sensitivity.json")
