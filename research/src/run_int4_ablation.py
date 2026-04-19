"""
INT4 degradation ablation: step size vs. range clipping.

int4_int3range: scale=max/3, clamp(-4,3) — 8 effective levels → 33.0% ≈ paper baseline
Question: is degradation from (a) coarser step size or (b) reduced clamp range?

Conditions:
  A: scale=max/7, clamp(-8,7)  — standard INT4, 16 levels          → ~42% (baseline)
  B: scale=max/3, clamp(-4,3)  — int4_int3range, 8 levels           → ~33% (paper match)
  C: scale=max/3, clamp(-8,7)  — coarse step only, full range       → ?
  D: scale=max/7, clamp(-4,3)  — normal step, narrow range          → ?
  E: scale=max/5, clamp(-8,7)  — intermediate step                  → ?

If C ≈ B (33%): degradation is from coarser step size, not range clipping.
If C ≈ A (42%): degradation is from range clipping, not step size.
"""
import sys, json, torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

MODEL_ID = "HuggingFaceTB/SmolLM-360M"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 50
DEVICE = "cpu"

# Quantization functions for each condition
def quant_A(x):
    """Standard INT4: scale=max/7, clamp(-8,7)"""
    s = x.abs().max() / 7.0
    return (x / s).round().clamp(-8, 7) * s if s > 0 else x

def quant_B(x):
    """int4_int3range: scale=max/3, clamp(-4,3)"""
    s = x.abs().max() / 3.0
    return (x / s).round().clamp(-4, 3) * s if s > 0 else x

def quant_C(x):
    """Coarse step only: scale=max/3, clamp(-8,7) — tests step size effect"""
    s = x.abs().max() / 3.0
    return (x / s).round().clamp(-8, 7) * s if s > 0 else x

def quant_D(x):
    """Narrow range only: scale=max/7, clamp(-4,3) — tests range clipping effect"""
    s = x.abs().max() / 7.0
    return (x / s).round().clamp(-4, 3) * s if s > 0 else x

def quant_E(x):
    """Intermediate: scale=max/5, clamp(-8,7)"""
    s = x.abs().max() / 5.0
    return (x / s).round().clamp(-8, 7) * s if s > 0 else x

CONDITIONS = {
    "A_standard_int4": quant_A,
    "B_int4_int3range": quant_B,
    "C_coarse_step_full_range": quant_C,
    "D_normal_step_narrow_range": quant_D,
    "E_intermediate_step": quant_E,
}

print(f"Loading {MODEL_ID}...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

from datasets import load_dataset
ds = list(load_dataset("Rowan/hellaswag", split="validation").select(range(LIMIT)))

def make_quant_hook(qfn):
    def hook(module, inp, out):
        return qfn(out)
    return hook

def eval_condition(name, qfn):
    hooks = []
    for n, module in model.named_modules():
        if n.endswith(".k_proj") or n.endswith(".v_proj"):
            hooks.append(module.register_forward_hook(make_quant_hook(qfn)))
    correct = 0
    for ex in ds:
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        scores = []
        for ending in ex["endings"]:
            full = tokenizer.encode(ctx + " " + ending, return_tensors="pt")
            ctx_len = tokenizer.encode(ctx, return_tensors="pt").shape[1]
            with torch.no_grad():
                logits = model(full).logits[0]
            cont = full[0, ctx_len:]
            if len(cont) == 0:
                scores.append(-1e9); continue
            lp = F.log_softmax(logits[ctx_len-1:ctx_len-1+len(cont)], dim=-1)
            scores.append(lp[range(len(cont)), cont].sum().item())
        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
            correct += 1
    for h in hooks:
        h.remove()
    acc = correct / len(ds) * 100
    print(f"  {name}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    return acc

print(f"\nRunning INT4 ablation ({LIMIT} samples)...", flush=True)
results = {}
for name, qfn in CONDITIONS.items():
    results[name] = eval_condition(name, qfn)

print("\n=== INT4 ABLATION RESULTS ===")
a, b, c, d, e = [results[k] for k in ["A_standard_int4","B_int4_int3range","C_coarse_step_full_range","D_normal_step_narrow_range","E_intermediate_step"]]
print(f"  A: standard INT4 (max/7, clamp±8):     {a:.1f}%  [reference]")
print(f"  B: int4_int3range (max/3, clamp[-4,3]): {b:.1f}%  [paper match]")
print(f"  C: coarse step (max/3, clamp±8):        {c:.1f}%  [step size effect]")
print(f"  D: narrow range (max/7, clamp[-4,3]):   {d:.1f}%  [range clipping effect]")
print(f"  E: intermediate (max/5, clamp±8):       {e:.1f}%")
print()
if abs(c - b) < abs(c - a):
    print(f"  → Degradation driven primarily by COARSE STEP SIZE (C≈B={c:.1f}%≈{b:.1f}%)")
elif abs(c - a) < abs(c - b):
    print(f"  → Degradation driven primarily by RANGE CLIPPING (C≈A={c:.1f}%≈{a:.1f}%)")
else:
    print(f"  → Mixed contribution")

out = {"model": MODEL_ID, "limit": LIMIT, "date": datetime.now().isoformat(),
       "results": results,
       "interpretation": {
           "step_size_effect": round(a - c, 1),
           "range_clip_effect": round(c - b, 1),
           "total_degradation": round(a - b, 1),
       }}
fname = Path("research/data") / f"int4_ablation_{LIMIT}samp_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved: {fname}", flush=True)
