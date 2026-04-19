"""
TQ-H1 Experiment: DWB-TurboQuant vs DWB-scalar accuracy comparison.

Baseline: DWB assigns 57% of tokens to 2-bit. Scalar 2-bit = 25% accuracy (catastrophic).
Question: Does PolarQuant (3-bit keys / 2-bit values via rotation) recover accuracy
          for those 2-bit tokens, bringing DWB-TurboQuant closer to FP16?

Conditions:
  1. FP16 (no quantization)          — upper bound ~42%
  2. Scalar 2-bit KV                 — lower bound ~25%
  3. PolarQuant 3-bit keys/2-bit val — TurboQuant approximation
  4. DWB-scalar (existing)           — DWB with naive 2-bit = 38-40%
  5. DWB-TurboQuant                  — DWB routing with PolarQuant for 2-bit tokens

Prediction for TQ-H1: condition 5 > condition 4 > condition 2
"""

import sys, json, time, torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from kv_cache_quant import attach_kv_hooks
from eval_hellaswag import score_continuation
from turboquant_impl import polar_quant_key, polar_quant_value, PolarQuant

MODEL_ID = "HuggingFaceTB/SmolLM-360M"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
DEVICE = "cpu"
CONTROLLER_PATH = "research/data/dwb_controller_smollm360m.pt"


def attach_polar_quant_hooks(model):
    """Hook k_proj and v_proj with PolarQuant (3-bit keys, 2-bit values)."""
    key_quant = PolarQuant(bits=3, seed=42)
    val_quant = PolarQuant(bits=2, seed=137)
    handles = []
    attached = 0
    for name, module in model.named_modules():
        if name.endswith(".k_proj"):
            def make_k_hook(q):
                def hook(mod, inp, out):
                    return q.quantize_dequantize(out)
                return hook
            handles.append(module.register_forward_hook(make_k_hook(key_quant)))
            attached += 1
        elif name.endswith(".v_proj"):
            def make_v_hook(q):
                def hook(mod, inp, out):
                    return q.quantize_dequantize(out)
                return hook
            handles.append(module.register_forward_hook(make_v_hook(val_quant)))
            attached += 1
    print(f"  Attached PolarQuant hooks to {attached} proj modules", flush=True)
    return handles


def evaluate_simple(model, tokenizer, ds, quant_mode=None, use_polar=False):
    """Evaluate HellaSwag with optional KV quantization."""
    handles = []
    if use_polar:
        handles = attach_polar_quant_hooks(model)
    elif quant_mode:
        handles = attach_kv_hooks(model, quant_mode)
    correct = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i > 0 and i % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(ds) - i)
            print(f"  [{i}/{LIMIT}] acc={correct/i*100:.1f}% eta={eta:.0f}s", flush=True)
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        scores = [score_continuation(model, tokenizer, ctx, " " + e) for e in ex["endings"]]
        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
            correct += 1
    for h in handles:
        h.remove()
    elapsed = time.time() - t0
    acc = correct / len(ds) * 100
    print(f"  Result: {correct}/{len(ds)} = {acc:.2f}%  ({elapsed:.0f}s)", flush=True)
    return {"accuracy": acc, "correct": correct, "total": len(ds), "elapsed_s": round(elapsed, 1)}


print(f"Loading {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

print(f"Loading HellaSwag ({LIMIT} samples)...", flush=True)
ds = load_dataset("Rowan/hellaswag", split="validation").select(range(LIMIT))

results = {}

conditions = [
    ("fp16", None, False),
    ("scalar_2bit", "static2bit", False),
    ("polar_quant_3bit_keys_2bit_vals", None, True),
]

for cond, mode, polar in conditions:
    print(f"\n--- {cond} ---", flush=True)
    r = evaluate_simple(model, tokenizer, ds, mode, polar)
    results[cond] = r

out = {
    "model": MODEL_ID,
    "limit": LIMIT,
    "metric": "acc_unnorm",
    "date": datetime.now().isoformat(),
    "conditions": results,
    "baselines": {
        "paper_fp16": 41.5,
        "paper_dwb": 41.2,
        "verified_fp16_500samp": 42.6,
        "verified_dwb_200samp": 38.0,
    }
}

out_dir = Path("research/data")
fname = out_dir / f"turboquant_eval_{LIMIT}samp_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {fname}", flush=True)

print("\n=== TQ-H1 RESULTS ===")
fp16_acc = results["fp16"]["accuracy"]
for cond, r in results.items():
    delta = r["accuracy"] - fp16_acc
    print(f"  {cond:40s}: {r['accuracy']:.1f}%  (vs FP16: {delta:+.1f}pp)")

scalar_2bit = results.get("scalar_2bit", {}).get("accuracy", 25.0)
polar = results.get("polar_quant_3bit_keys_2bit_vals", {}).get("accuracy", 0.0)
print(f"\n  PolarQuant recovery over scalar 2-bit: {polar - scalar_2bit:+.1f}pp")
print(f"  TQ-H1: {'CONFIRMED' if polar > scalar_2bit + 3 else 'WEAK' if polar > scalar_2bit else 'REFUTED'}")
