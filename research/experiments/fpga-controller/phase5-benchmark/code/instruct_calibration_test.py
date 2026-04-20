"""
Tests whether beta* calibration transfers from base to instruction-tuned model.

Compares SmolLM-360M-Instruct vs SmolLM-360M-Base (both same architecture).
If gap_mean is similar (within +-0.02), beta* calibrated on base model transfers.
If different, re-calibration is needed after fine-tuning.

Practical implication: can users calibrate on the base model, or must they re-run
after instruction fine-tuning?

Prediction: gap_mean should be close to 0.337 (base model value) since
RLHF/SFT does not dramatically shift KV activation distributions.
"""
import json
import time
from pathlib import Path

import torch

N_TEXTS = 10
MAX_LEN = 64
RESULT_PATH = (Path(__file__).parent.parent
               / "results" / "instruct_calibration_test.json")

BASE_GAP_MEAN = 0.3367  # SmolLM-360M base, measured Session 13


def quant_quality(x: torch.Tensor, bits: int) -> float:
    n_levels = 2 ** bits - 1
    half = n_levels // 2
    scale = x.abs().max().clamp(min=1e-8) / half
    x_q = (x / scale).round().clamp(-half, half) * scale
    norm = x.norm().clamp(min=1e-8)
    return float(1.0 - (x_q - x).norm() / norm)


def measure_gap_mean(model_id: str, label: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"Loading {label} ({model_id})...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, hidden={model.config.hidden_size}")

    all_q4, all_q8 = [], []
    captured = {i: {"k": None, "v": None} for i in range(n_layers)}

    def make_hook(i):
        def hook(module, inp, out):
            captured[i]["k"] = out.detach()
        return hook
    def make_hook_v(i):
        def hook(module, inp, out):
            captured[i]["v"] = out.detach()
        return hook

    hooks = []
    for i in range(n_layers):
        layer = model.model.layers[i].self_attn
        hooks.append(layer.k_proj.register_forward_hook(make_hook(i)))
        hooks.append(layer.v_proj.register_forward_hook(make_hook_v(i)))

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [r["text"] for r in ds if len(r["text"].split()) > 20][:N_TEXTS]

    t0 = time.time()
    with torch.no_grad():
        for text in texts:
            enc = tok(text, return_tensors="pt",
                      max_length=MAX_LEN, truncation=True)
            model(**enc)
            for i in range(n_layers):
                k = captured[i]["k"][0]  # [seq, dim]
                v = captured[i]["v"][0]
                for t in range(k.shape[0]):
                    all_q4.append(quant_quality(k[t], 4))
                    all_q8.append(quant_quality(k[t], 8))
                    all_q4.append(quant_quality(v[t], 4))
                    all_q8.append(quant_quality(v[t], 8))

    for h in hooks:
        h.remove()

    q4 = torch.tensor(all_q4)
    q8 = torch.tensor(all_q8)
    gap = (q8 - q4).numpy()
    gap_mean = float(gap.mean())
    gap_std = float(gap.std())
    beta_star = gap_mean / 0.267
    n_tokens = len(all_q4)
    elapsed = time.time() - t0

    delta_from_base = gap_mean - BASE_GAP_MEAN
    transfers = abs(delta_from_base) < 0.02

    print(f"  {n_tokens} tokens in {elapsed:.1f}s")
    print(f"  gap_mean={gap_mean:.4f}  gap_std={gap_std:.4f}")
    print(f"  beta*={beta_star:.3f}")
    print(f"  delta from base: {delta_from_base:+.4f}  "
          f"(transfers={'YES' if transfers else 'NO'}, threshold +-0.02)")

    return {
        "model": model_id,
        "label": label,
        "n_tokens": n_tokens,
        "gap_mean": round(gap_mean, 4),
        "gap_std": round(gap_std, 4),
        "beta_star": round(beta_star, 3),
        "delta_from_base": round(delta_from_base, 4),
        "transfers_to_instruct": transfers,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    results = {}

    # Instruct model (base already known: gap_mean=0.3367, beta*=1.261)
    results["base_reference"] = {
        "model": "HuggingFaceTB/SmolLM-360M",
        "label": "SmolLM-360M Base (reference)",
        "gap_mean": BASE_GAP_MEAN,
        "beta_star": round(BASE_GAP_MEAN / 0.267, 3),
        "note": "Pre-measured in Session 13 (89,856 tokens)",
    }

    try:
        results["instruct"] = measure_gap_mean(
            "HuggingFaceTB/SmolLM-360M-Instruct",
            "SmolLM-360M-Instruct (SFT)"
        )
    except Exception as e:
        print(f"ERROR loading instruct model: {e}")
        results["instruct"] = {"error": str(e)}

    # Summary
    print("\n" + "="*60)
    print("SUMMARY — Base-to-Instruct Calibration Transfer")
    print("="*60)
    if "gap_mean" in results.get("instruct", {}):
        base_gm = BASE_GAP_MEAN
        inst_gm = results["instruct"]["gap_mean"]
        delta = inst_gm - base_gm
        print(f"  Base:    gap_mean={base_gm:.4f}  beta*={base_gm/0.267:.3f}")
        print(f"  Instruct: gap_mean={inst_gm:.4f}  beta*={inst_gm/0.267:.3f}")
        print(f"  Delta:   {delta:+.4f}")
        print(f"  Transfers (|delta|<0.02): "
              f"{'YES - base calibration is valid for instruct' if abs(delta)<0.02 else 'NO - re-calibrate after fine-tuning'}")

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
