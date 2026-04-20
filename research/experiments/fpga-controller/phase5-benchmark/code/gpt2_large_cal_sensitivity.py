"""
Calibration sensitivity for GPT-2 Large (774M). Completes the full GPT-2 family sweep:
Small (124M, gap_std=0.033, max_error=0.018) → Medium (345M, gap_std=0.052, max_error=0.011)
→ Large (774M, gap_std=0.026) — gap_std is NON-MONOTONIC with scale (M > S > L).
Finding 9 predicts max_error is uncorrelated with gap_std regardless.
"""
import torch, json, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "gpt2-large"
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
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, local_files_only=True)
model.eval()
n_layers = model.config.n_layer

def extract_gaps(text_list):
    signals, kv_store = [], {}
    hooks = []
    def make_hook(i):
        def h(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            n = x.shape[-1] // 3
            kv_store[i] = (x[..., n:2*n].detach().float(), x[..., 2*n:].detach().float())
        return h
    for i in range(n_layers):
        hooks.append(model.transformer.h[i].attn.c_attn.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        for text in text_list:
            ids = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            kv_store.clear()
            model(**ids)
            for i in range(n_layers):
                if i not in kv_store: continue
                k, v = kv_store[i]
                for b in range(k.shape[0]):
                    for t in range(k.shape[1]):
                        vec = torch.cat([k[b,t], v[b,t]], dim=0)
                        signals.append((quant_quality(vec, 7), quant_quality(vec, 127)))
    for h in hooks: h.remove()
    return signals

print("Running 10-text aggregate...")
all_sig = extract_gaps(TEXTS)
gaps = np.array([s[1]-s[0] for s in all_sig])
gm10, gs10, bs10 = gaps.mean(), gaps.std(), gaps.mean()/0.267
print(f"10-text: gap_mean={gm10:.4f}, gap_std={gs10:.4f}, beta*={bs10:.3f}, n={len(all_sig)}")

print("\nPer-text:")
per_text = []
for i, text in enumerate(TEXTS):
    s = extract_gaps([text])
    g = np.array([x[1]-x[0] for x in s])
    gm, bs, err = g.mean(), g.mean()/0.267, abs(g.mean()/0.267 - bs10)
    per_text.append({"text_idx":i,"n_tokens":len(s),"gap_mean":round(float(gm),4),
                     "gap_std":round(float(g.std()),4),"beta_star":round(float(bs),3),
                     "error_vs_10text":round(float(err),3)})
    print(f"  text {i+1}: n={len(s):4d}, gap_mean={gm:.4f}, gap_std={g.std():.4f}, beta*={bs:.3f}, error={err:.3f}")

max_e = max(x["error_vs_10text"] for x in per_text)
mean_e = float(np.mean([x["error_vs_10text"] for x in per_text]))
print(f"\nGPT-2 Large sensitivity: max_error={max_e:.3f}, mean_error={mean_e:.3f}")
print(f"  Within +-0.015: {max_e<=0.015}  Within +-0.020: {max_e<=0.020}")

result = {"model":MODEL_ID,"architecture":"GPT-2/Conv1D (774M, 36L, 20H)",
          "note":"Completes GPT-2 family sweep. gap_std=0.026 is NON-MONOTONIC (M=0.052 > S=0.033 > L=0.026).",
          "gap_mean_10text":round(float(gm10),4),"gap_std_10text":round(float(gs10),4),
          "beta_star_10text":round(float(bs10),3),"per_text_results":per_text,
          "max_error_1text":round(float(max_e),4),"mean_error_1text":round(float(mean_e),4),
          "confirmed_015":max_e<=0.015,"confirmed_020":max_e<=0.020}
with open(f"{RESULTS_DIR}/gpt2_large_cal_sensitivity.json","w") as f:
    json.dump(result, f, indent=2)
print("Saved gpt2_large_cal_sensitivity.json")
