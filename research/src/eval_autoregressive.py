"""
Autoregressive HellaSwag scoring with KV cache quantization.

Difference from single-pass scoring:
  Single-pass: encode(context + continuation) in one forward pass — KV not reused
  Autoregressive: encode context once, then generate continuation token-by-token
                  (KV cache grows; past keys/values are read back at every step)

The paper evaluates via actual autoregressive generation (KV cache is used and
quantized as it grows). Our single-pass approach doesn't exercise this path.

This tests whether quantization accumulation during generation explains why
the paper sees 7.9pp accuracy drop for INT4 while we see ~0pp.

SLOW: ~10x more forward passes per example than single-pass. Use limit=50.
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
from kv_cache_quant import _QUANT_FNS

MODEL_ID = "HuggingFaceTB/SmolLM-360M"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 50
DEVICE = "cpu"

PAPER = {"fp16": 41.50, "static4bit": 33.60}
CONDITIONS = [("fp16", None), ("static4bit", "static4bit")]


def score_continuation_autoregressive(model, tokenizer, context, continuation,
                                       quant_mode=None, device=DEVICE):
    """
    Score continuation autoregressively using KV cache.
    Each new token attends to past KV (possibly quantized).

    Returns sum of log-probs of continuation tokens (unnormalized acc metric).
    """
    ctx_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)

    if len(cont_ids) == 0:
        return -float("inf")

    # Encode context, get KV cache
    with torch.no_grad():
        ctx_out = model(ctx_ids, use_cache=True)

    pkv = ctx_out.past_key_values

    # Quantize initial KV cache if needed
    if quant_mode is not None:
        qfn = _QUANT_FNS[quant_mode]
        if hasattr(pkv, "key_cache"):
            # DynamicCache
            for i in range(len(pkv.key_cache)):
                pkv.key_cache[i] = qfn(pkv.key_cache[i])
                pkv.value_cache[i] = qfn(pkv.value_cache[i])

    total_log_prob = 0.0

    # Generate continuation token by token
    for tok_id in cont_ids:
        tok_tensor = torch.tensor([[tok_id]], device=device)
        with torch.no_grad():
            out = model(tok_tensor, past_key_values=pkv, use_cache=True)

        # Get log prob of this token from previous step's logits
        # (we need the logit at the position BEFORE this token)
        logits_prev = ctx_out.logits[0, -1, :] if total_log_prob == 0 else out.logits[0, -1, :]
        log_prob = F.log_softmax(ctx_out.logits[0, -1, :], dim=-1)[tok_id].item()

        # Update for next step
        pkv = out.past_key_values
        ctx_out = out

        # Quantize new KV entries
        if quant_mode is not None:
            qfn = _QUANT_FNS[quant_mode]
            if hasattr(pkv, "key_cache"):
                for i in range(len(pkv.key_cache)):
                    pkv.key_cache[i] = qfn(pkv.key_cache[i])
                    pkv.value_cache[i] = qfn(pkv.value_cache[i])

        # Accumulate log prob of this continuation token
        total_log_prob += F.log_softmax(
            (ctx_out.logits if total_log_prob == 0 else out.logits)[0, -1, :], dim=-1
        )[tok_id].item()

    return total_log_prob


def score_continuation_ar_correct(model, tokenizer, context, continuation,
                                   quant_mode=None, device=DEVICE):
    """Corrected autoregressive scoring with proper log-prob accumulation."""
    ctx_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)

    if len(cont_ids) == 0:
        return -float("inf")

    qfn = _QUANT_FNS.get(quant_mode, lambda x: x) if quant_mode else lambda x: x

    def quantize_pkv(pkv):
        if pkv is None:
            return pkv
        if hasattr(pkv, "key_cache"):
            for i in range(len(pkv.key_cache)):
                if pkv.key_cache[i] is not None:
                    pkv.key_cache[i] = qfn(pkv.key_cache[i])
                    pkv.value_cache[i] = qfn(pkv.value_cache[i])
        return pkv

    # Encode context
    with torch.no_grad():
        ctx_out = model(ctx_ids, use_cache=True)
    pkv = quantize_pkv(ctx_out.past_key_values)

    # Last context logits predict first continuation token
    prev_logits = ctx_out.logits[0, -1, :]
    total_log_prob = 0.0

    for tok_id in cont_ids:
        # Log prob of this token
        total_log_prob += F.log_softmax(prev_logits, dim=-1)[tok_id].item()

        # Generate next logits
        tok_tensor = torch.tensor([[tok_id]], device=device)
        with torch.no_grad():
            out = model(tok_tensor, past_key_values=pkv, use_cache=True)

        pkv = quantize_pkv(out.past_key_values)
        prev_logits = out.logits[0, -1, :]

    return total_log_prob


def evaluate(model, tokenizer, ds, quant_mode=None):
    correct = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i > 0 and i % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(ds) - i)
            print(f"  [{i}/{len(ds)}] acc={correct/i*100:.1f}% eta={eta:.0f}s", flush=True)
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        scores = [score_continuation_ar_correct(model, tokenizer, ctx, " " + e, quant_mode)
                  for e in ex["endings"]]
        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
            correct += 1
    elapsed = time.time() - t0
    acc = correct / len(ds) * 100
    print(f"  Result: {correct}/{len(ds)} = {acc:.2f}%  ({elapsed:.0f}s)", flush=True)
    return {"accuracy": acc, "correct": correct, "total": len(ds), "elapsed_s": round(elapsed, 1)}


print(f"Loading {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

print(f"Loading HellaSwag ({LIMIT} samples)...", flush=True)
ds = load_dataset("Rowan/hellaswag", split="validation").select(range(LIMIT))

out = {"model": "smollm-360m", "limit": LIMIT, "metric": "acc_unnorm_autoregressive",
       "date": datetime.now().isoformat(), "paper": PAPER, "conditions": {}}

for cond, mode in CONDITIONS:
    print(f"\n--- {cond} (autoregressive) ---", flush=True)
    r = evaluate(model, tokenizer, ds, mode)
    r["paper_target"] = PAPER.get(cond)
    r["delta"] = round(r["accuracy"] - PAPER[cond], 2) if cond in PAPER else None
    out["conditions"][cond] = r
    print(f"  {cond}: {r['accuracy']:.2f}%  paper: {PAPER.get(cond)}%  delta: {r['delta']:+.2f}pp" if r["delta"] is not None else "", flush=True)

out_dir = Path("research/data")
fname = out_dir / f"ar_kv_comparison_{LIMIT}samp_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {fname}", flush=True)

print("\n=== AUTOREGRESSIVE vs SINGLE-PASS COMPARISON ===")
for cond, r in out["conditions"].items():
    print(f"  {cond}: {r['accuracy']:.2f}% (AR)  paper: {r.get('paper_target','—')}%")
