"""Run just the remaining INT4 variants (int4_int3range, int4_block64, int4_offline_scale_2x).
Previous run crashed on Unicode with int4_int3range — that variant matched paper's 33.6%!
"""

import sys
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))

MODEL_ID = "HuggingFaceTB/SmolLM-360M"
LIMIT = 100
DEVICE = "cpu"
PAPER_KV4 = 33.6
PAPER_FP16 = 41.0  # 100-sample result


def q_int3range(x):
    """INT4 with INT3 effective range [-4,3] = 8 values instead of 16."""
    scale = x.abs().max() / 3.0
    return (x / scale).round().clamp(-4, 3) * scale if scale > 0 else x


def q_block64(x):
    """Block quantization: group_size=64 per feature dimension."""
    if x.dim() != 3:
        scale = x.abs().max() / 7.0
        return (x / scale).round().clamp(-8, 7) * scale if scale > 0 else x
    B, T, D = x.shape
    r = x.clone()
    for g in range(0, D, 64):
        chunk = x[:, :, g:g+64]
        scale = chunk.abs().max() / 7.0
        if scale > 0:
            r[:, :, g:g+64] = (chunk / scale).round().clamp(-8, 7) * scale
    return r


def q_offline_scale_2x(x, fixed_scale=2.0):
    return (x / fixed_scale).round().clamp(-8, 7) * fixed_scale


VARIANTS = [
    ("int4_int3range", q_int3range),
    ("int4_block64", q_block64),
    ("int4_offline_scale_2x", q_offline_scale_2x),
]


def make_hook(qfn):
    def hook(module, input, output):
        return qfn(output)
    return hook


def score_continuation(model, tokenizer, context, continuation):
    full_ids = tokenizer.encode(context + continuation, return_tensors="pt")
    ctx_len = tokenizer.encode(context, return_tensors="pt").shape[1]
    with torch.no_grad():
        logits = model(full_ids).logits[0]
    cont_ids = full_ids[0, ctx_len:]
    if len(cont_ids) == 0:
        return -float("inf")
    lp = F.log_softmax(logits[ctx_len-1:ctx_len-1+len(cont_ids)], dim=-1)
    return lp[range(len(cont_ids)), cont_ids].sum().item()


def evaluate(model, tokenizer, ds, qfn=None):
    hooks = []
    if qfn is not None:
        for name, module in model.named_modules():
            if name.split(".")[-1] in ("k_proj", "v_proj"):
                hooks.append(module.register_forward_hook(make_hook(qfn)))
    correct = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        scores = [score_continuation(model, tokenizer, ctx, " " + e) for e in ex["endings"]]
        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
            correct += 1
    for h in hooks:
        h.remove()
    elapsed = time.time() - t0
    acc = correct / len(ds) * 100
    return acc, elapsed


print(f"Loading {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
print(f"Loading HellaSwag ({LIMIT} samples)...", flush=True)
ds = load_dataset("Rowan/hellaswag", split="validation").select(range(LIMIT))

results = {"fp16": PAPER_FP16}  # From previous run

print(f"\nVariant                        Acc     Delta vs FP16  vs Paper 33.6%")
print("-" * 72)

for vname, qfn in VARIANTS:
    print(f"Running {vname}...", flush=True)
    acc, elapsed = evaluate(model, tokenizer, ds, qfn)
    delta_fp16 = acc - PAPER_FP16
    delta_paper = acc - PAPER_KV4
    match = " <-- MATCHES PAPER!" if abs(acc - PAPER_KV4) < 3 else ""
    results[vname] = acc
    print(f"{vname:<30} {acc:>6.1f}%  {delta_fp16:>+10.1f}pp  {delta_paper:>+10.1f}pp{match}", flush=True)

# Combined results including prior run
all_results = {
    "fp16": PAPER_FP16,
    "int4_sym_pertensor": 44.0,
    "int4_asym_pertensor": 43.0,
    "int4_sym_pertoken": 44.0,
    "int4_asym_pertoken": 39.0,
    **results,
}

out_dir = Path("research/data")
fname = out_dir / f"int4_investigation_{LIMIT}samp_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump({"model": "smollm-360m", "limit": LIMIT, "paper_fp16": PAPER_FP16,
               "paper_kv4": PAPER_KV4, "results": all_results,
               "date": datetime.now().isoformat()}, f, indent=2)
print(f"\nSaved: {fname}", flush=True)

print("\n=== FULL RESULTS SUMMARY ===")
print(f"{'Variant':<30} {'Acc':>7}  {'Delta FP16':>12}  {'Delta Paper':>12}")
print("-" * 68)
for variant, acc in all_results.items():
    d_fp16 = acc - PAPER_FP16
    d_paper = acc - PAPER_KV4
    match = " <-- NEAR PAPER" if abs(acc - PAPER_KV4) < 4 else ""
    print(f"{variant:<30} {acc:>6.1f}%  {d_fp16:>+10.1f}pp  {d_paper:>+10.1f}pp{match}")
