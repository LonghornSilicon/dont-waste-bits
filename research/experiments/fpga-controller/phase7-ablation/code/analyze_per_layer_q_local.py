"""
Per-layer q_local diagnostic for SmolLM-1.7B (CPU or GPU, ~10 min).

Question: does the per-token quantization quality proxy q_local (= 1 - quant_err/||kv||)
vary systematically across transformer layers? If yes → a layer-tuned p4 schedule could
beat the uniform p4 used in Phase 7. If no → the paper's uniform-p4 claim is watertight.

Procedure: re-extract 30 WikiText-2 texts through SmolLM-1.7B, hook k_proj + v_proj,
compute q4_local and q8_local per token per layer, and dump aggregate statistics per
layer. Decision rule (arbitrary but concrete):
  - "uniform" if max(layer_q4) - min(layer_q4) ≤ 0.05
  - "layer-dependent" if > 0.10 (worth a follow-up)
  - "borderline" between those.

Output: phase7-ablation/results/per_layer_q_local.json
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"
N_TEXTS    = 30
MAX_LEN    = 128

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "per_layer_q_local.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  n_texts={N_TEXTS} max_len={MAX_LEN}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    num_layers = len(model.model.layers)

    kv_buf, handles = {}, []

    def make_hook(li, which):
        def hook(module, inp, out):
            kv_buf[f"{li}_{which}"] = out.detach().cpu().float()
        return hook

    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(li, "v")))

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50][:N_TEXTS]

    # Aggregate per-layer: list of per-token q4 and q8 across all texts
    per_layer_q4 = {li: [] for li in range(num_layers)}
    per_layer_q8 = {li: [] for li in range(num_layers)}
    per_layer_norms = {li: [] for li in range(num_layers)}

    def qlocal(kv, bits):
        kv_q = torch.stack([quantize_tensor(kv[t], bits) for t in range(kv.shape[0])])
        err  = (kv_q - kv).norm(dim=-1)
        nrm  = kv.norm(dim=-1).clamp(min=1e-8)
        return (1.0 - (err / nrm)).clamp(0, 1)

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 5 == 0:
                print(f"  text {i+1}/{len(texts)}", flush=True)
            kv_buf.clear()
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            inp = {k: v.to(device) for k, v in inp.items()}
            model(**inp)
            for li in range(num_layers):
                k = kv_buf.get(f"{li}_k"); v = kv_buf.get(f"{li}_v")
                if k is None or v is None:
                    continue
                kv = torch.cat([k[0], v[0]], dim=-1)
                per_layer_q4[li].extend(qlocal(kv, 4).tolist())
                per_layer_q8[li].extend(qlocal(kv, 8).tolist())
                per_layer_norms[li].extend(kv.norm(dim=-1).tolist())

    for h in handles:
        h.remove()
    del model

    summary = []
    for li in range(num_layers):
        q4 = torch.tensor(per_layer_q4[li])
        q8 = torch.tensor(per_layer_q8[li])
        norms = torch.tensor(per_layer_norms[li])
        summary.append({
            "layer":      li,
            "n_tokens":   len(q4),
            "q4_mean":    round(q4.mean().item(), 4),
            "q4_std":     round(q4.std().item(), 4),
            "q4_p10":     round(q4.quantile(0.10).item(), 4),
            "q4_p90":     round(q4.quantile(0.90).item(), 4),
            "q8_mean":    round(q8.mean().item(), 4),
            "q8_std":     round(q8.std().item(), 4),
            "gap_mean":   round((q8 - q4).mean().item(), 4),
            "gap_std":    round((q8 - q4).std().item(), 4),
            "norm_mean":  round(norms.mean().item(), 3),
            "norm_std":   round(norms.std().item(), 3),
        })

    q4_means = [s["q4_mean"] for s in summary]
    q8_means = [s["q8_mean"] for s in summary]
    gap_means = [s["gap_mean"] for s in summary]

    q4_spread = max(q4_means) - min(q4_means)
    gap_spread = max(gap_means) - min(gap_means)

    if q4_spread <= 0.05:
        verdict = "UNIFORM — layer-tuned p4 unlikely to help"
    elif q4_spread >= 0.10:
        verdict = "LAYER-DEPENDENT — layer-tuned p4 worth exploring"
    else:
        verdict = "BORDERLINE — mild layer gradient; follow-up might help marginally"

    result = {
        "experiment":   "per_layer_q_local",
        "model":        MODEL_NAME,
        "n_texts":      N_TEXTS,
        "max_len":      MAX_LEN,
        "num_layers":   num_layers,
        "per_layer":    summary,
        "q4_mean_across_layers": {
            "min":    round(min(q4_means), 4),
            "max":    round(max(q4_means), 4),
            "spread": round(q4_spread, 4),
            "argmin": q4_means.index(min(q4_means)),
            "argmax": q4_means.index(max(q4_means)),
        },
        "gap_mean_across_layers": {
            "min":    round(min(gap_means), 4),
            "max":    round(max(gap_means), 4),
            "spread": round(gap_spread, 4),
        },
        "verdict":       verdict,
        "elapsed_s":     round(time.time() - t0, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)

    print("\n=== Per-layer q4_local summary ===", flush=True)
    print(f"{'layer':>5}  {'q4_mean':>8}  {'q8_mean':>8}  {'gap':>7}  {'norm_mean':>10}",
          flush=True)
    for s in summary:
        print(f"{s['layer']:>5}  {s['q4_mean']:>8.4f}  {s['q8_mean']:>8.4f}  "
              f"{s['gap_mean']:>7.4f}  {s['norm_mean']:>10.3f}", flush=True)

    print(f"\nq4 spread across layers: min={min(q4_means):.4f} max={max(q4_means):.4f} "
          f"(Δ={q4_spread:.4f}, argmin=L{q4_means.index(min(q4_means))}, "
          f"argmax=L{q4_means.index(max(q4_means))})", flush=True)
    print(f"gap spread across layers: Δ={gap_spread:.4f}", flush=True)
    print(f"\n>>> {verdict} <<<", flush=True)


if __name__ == "__main__":
    main()
