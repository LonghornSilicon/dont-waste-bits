"""
Multi-condition KV quantization comparison for DWB verification.
Uses eval_hellaswag.py's score_continuation with normalize=False (paper metric).

Runs: FP16 | static4bit_per_tensor | static4bit_per_token | static4bit_asym
Saves JSON to research/data/
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
from kv_cache_quant import attach_kv_hooks, detach_kv_hooks

MODEL_ID = "HuggingFaceTB/SmolLM-360M"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DEVICE = "cpu"

PAPER = {"fp16": 41.50, "static4bit": 33.60, "dwb": 41.20}
CONDITIONS = [
    ("fp16", None),
    ("static4bit", "static4bit"),
    ("static4bit_per_token", "static4bit_per_token"),
    ("static4bit_asym", "static4bit_asym"),
]


def score_continuation(model, tokenizer, context, continuation, device=DEVICE):
    """Raw log-likelihood sum (unnormalized) — matches paper's 'acc' metric."""
    full_text = context + continuation
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    ctx_len = tokenizer.encode(context, return_tensors="pt").shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits[0]

    cont_ids = full_ids[0, ctx_len:]
    if len(cont_ids) == 0:
        return -float("inf")

    log_probs = F.log_softmax(logits[ctx_len - 1:ctx_len - 1 + len(cont_ids)], dim=-1)
    return log_probs[range(len(cont_ids)), cont_ids].sum().item()  # unnormalized


def evaluate(model, tokenizer, ds):
    correct, total = 0, len(ds)
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i > 0 and i % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (total - i)
            print(f"  [{i}/{total}] acc={correct/i*100:.1f}% eta={eta:.0f}s", flush=True)
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        scores = [score_continuation(model, tokenizer, ctx, " " + e) for e in ex["endings"]]
        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
            correct += 1
    elapsed = time.time() - t0
    acc = correct / total * 100
    print(f"  Result: {correct}/{total} = {acc:.2f}%  ({elapsed:.0f}s)", flush=True)
    return {"accuracy": acc, "correct": correct, "total": total, "elapsed_s": round(elapsed, 1)}


print(f"Loading {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE)
model.eval()

print(f"Loading HellaSwag validation ({LIMIT} samples)...", flush=True)
ds = load_dataset("Rowan/hellaswag", split="validation").select(range(LIMIT))

results = {"model": "smollm-360m", "limit": LIMIT, "metric": "acc_unnorm",
           "date": datetime.now().isoformat(), "paper_targets": PAPER, "conditions": {}}

for cond_name, hook_mode in CONDITIONS:
    print(f"\n--- Condition: {cond_name} ---", flush=True)
    hooks = []
    if hook_mode:
        hooks = attach_kv_hooks(model, mode=hook_mode)
    r = evaluate(model, tokenizer, ds)
    if hooks:
        detach_kv_hooks(hooks)

    paper_val = PAPER.get("static4bit" if "4bit" in cond_name else cond_name)
    r["paper_target"] = paper_val
    r["delta"] = round(r["accuracy"] - paper_val, 2) if paper_val else None
    results["conditions"][cond_name] = r
    print(f"  {cond_name}: {r['accuracy']:.2f}%  (paper: {paper_val}%  delta: {r['delta']:+.2f}pp)" if r["delta"] is not None else f"  {cond_name}: {r['accuracy']:.2f}%", flush=True)

out_dir = Path("research/data")
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / f"kv_comparison_smollm360m_{LIMIT}samp_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {fname}", flush=True)

# Summary table
print("\n=== Summary ===")
print(f"{'Condition':<28} {'Ours':>8} {'Paper':>8} {'Delta':>8}")
print("-" * 56)
for cond, r in results["conditions"].items():
    pt = f"{r['paper_target']:.1f}%" if r["paper_target"] else "—"
    delta = f"{r['delta']:+.2f}pp" if r["delta"] is not None else "—"
    print(f"{cond:<28} {r['accuracy']:>7.2f}%  {pt:>7}  {delta:>8}")
