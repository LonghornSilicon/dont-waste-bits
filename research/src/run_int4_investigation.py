"""
INT4 Quantization Investigation — why doesn't our INT4 reproduce paper's 33.6%?

Tests multiple INT4 variants on 100 HellaSwag samples to find what produces
the paper's claimed 7.9pp accuracy drop from FP16.

Variants:
1. symmetric per-tensor (our current: ~44.5%, paper: 33.6%)
2. asymmetric per-tensor
3. symmetric per-token
4. asymmetric per-token
5. fixed offline scale (computed on calibration, not per-forward-pass)
6. low effective bits: clamp INT4 to INT3 range [-4,3] (simulates poor implementation)
7. block quantization (group_size=64 per output channel)
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
PAPER_FP16 = 41.5


def q_sym_pertensor(x):
    scale = x.abs().max() / 7.0
    return (x / scale).round().clamp(-8, 7) * scale if scale > 0 else x


def q_asym_pertensor(x):
    xmin, xmax = x.min(), x.max()
    scale = (xmax - xmin) / 15.0
    if scale == 0: return x
    zp = (-xmin / scale).round().clamp(0, 15)
    return ((x / scale + zp).round().clamp(0, 15) - zp) * scale


def q_sym_pertoken(x):
    if x.dim() != 3: return q_sym_pertensor(x)
    r = x.clone()
    for t in range(x.shape[1]):
        s = x[:, t, :].abs().max() / 7.0
        if s > 0: r[:, t, :] = (x[:, t, :] / s).round().clamp(-8, 7) * s
    return r


def q_asym_pertoken(x):
    if x.dim() != 3: return q_asym_pertensor(x)
    r = x.clone()
    for t in range(x.shape[1]):
        tok = x[:, t, :]
        xmin, xmax = tok.min(), tok.max()
        s = (xmax - xmin) / 15.0
        if s > 0:
            zp = (-xmin / s).round().clamp(0, 15)
            r[:, t, :] = ((tok / s + zp).round().clamp(0, 15) - zp) * s
    return r


def q_int3range(x):
    """INT4 with INT3 effective range — simulates 3-bit masquerading as 4-bit."""
    scale = x.abs().max() / 3.0
    return (x / scale).round().clamp(-4, 3) * scale if scale > 0 else x


def q_block64(x):
    """Block quantization: group_size=64 per feature dimension."""
    if x.dim() != 3: return q_sym_pertensor(x)
    B, T, D = x.shape
    r = x.clone()
    for g in range(0, D, 64):
        chunk = x[:, :, g:g+64]
        scale = chunk.abs().max() / 7.0
        if scale > 0:
            r[:, :, g:g+64] = (chunk / scale).round().clamp(-8, 7) * scale
    return r


def q_offline_scale(x, fixed_scale=1.0):
    """Fixed offline-computed scale (doesn't adapt to input).
    Simulates quantizing with a calibration-set scale.
    If calibration over-estimates the scale, error is larger.
    """
    return (x / fixed_scale).round().clamp(-8, 7) * fixed_scale


VARIANTS = [
    ("fp16", lambda x: x),
    ("int4_sym_pertensor", q_sym_pertensor),
    ("int4_asym_pertensor", q_asym_pertensor),
    ("int4_sym_pertoken", q_sym_pertoken),
    ("int4_asym_pertoken", q_asym_pertoken),
    ("int4_int3range", q_int3range),
    ("int4_block64", q_block64),
    ("int4_offline_scale_2x", lambda x: q_offline_scale(x, fixed_scale=2.0)),
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

results = {}
print(f"\n{'Variant':<30} {'Acc':>8} {'Delta':>10} {'vs Paper':>10}")
print("-" * 62)

for vname, qfn in VARIANTS:
    print(f"Running {vname}...", flush=True, end="")
    acc, elapsed = evaluate(model, tokenizer, ds, None if vname == "fp16" else qfn)
    baseline = results.get("fp16", acc)
    delta_fp16 = acc - baseline if vname != "fp16" else 0
    delta_paper = acc - (PAPER_FP16 if vname == "fp16" else PAPER_KV4)
    results[vname] = acc
    status = ""
    if vname != "fp16":
        if abs(acc - PAPER_KV4) < 3:
            status = " <-- MATCHES PAPER"
        elif abs(delta_fp16) < 3:
            status = " (same as FP16)"
    print(f"\r{vname:<30} {acc:>7.1f}%  {delta_fp16:>+8.1f}pp  {delta_paper:>+8.1f}pp{status}", flush=True)

out_dir = Path("research/data")
fname = out_dir / f"int4_investigation_{LIMIT}samp_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump({"model": "smollm-360m", "limit": LIMIT, "paper_fp16": PAPER_FP16,
               "paper_kv4": PAPER_KV4, "results": results,
               "date": datetime.now().isoformat()}, f, indent=2)
print(f"\nSaved: {fname}", flush=True)
