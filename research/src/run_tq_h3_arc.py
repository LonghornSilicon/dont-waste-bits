"""
TQ-H3: DWB-TurboQuant vs DWB-scalar on ARC-Challenge (reasoning benchmark).

Hypothesis: the PolarQuant gain at the 2-bit tier is *larger* on reasoning
tasks (ARC-Challenge) than on commonsense completion (HellaSwag, +2pp).

Prediction: DWB-TurboQuant > DWB-scalar by >= 2pp on ARC-Challenge.
"""

import sys, json, time, torch
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List

sys.path.insert(0, str(Path(__file__).parent))
from kv_cache_quant import quantize_int4, quantize_int8, quantize_int2
from turboquant_impl import PolarQuant

MODEL_ID = "HuggingFaceTB/SmolLM-360M"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
DEVICE = "cpu"
CONTROLLER_PATH = "research/data/dwb_controller_smollm360m.pt"
HEAD_DIM = 64


def polar_quant_single_token(x: torch.Tensor, quant: PolarQuant) -> torch.Tensor:
    batch, d = x.shape
    n_heads = d // HEAD_DIM
    if d % HEAD_DIM != 0:
        return quant.quantize_dequantize(x)
    x_heads = x.view(batch * n_heads, HEAD_DIM)
    x_out = quant.quantize_dequantize(x_heads)
    return x_out.view(batch, d)


def make_hook_tq(bit_widths: List[int], is_key: bool):
    key_quant = PolarQuant(bits=3, seed=42)
    val_quant = PolarQuant(bits=2, seed=137)
    polar = key_quant if is_key else val_quant
    quant_fns = {
        2: lambda x: polar_quant_single_token(x, polar),
        4: quantize_int4, 8: quantize_int8, 16: lambda x: x,
    }
    def hook(module, inp, out):
        result = out.clone()
        for t in range(min(out.shape[1], len(bit_widths))):
            result[:, t] = quant_fns.get(bit_widths[t], lambda x: x)(out[:, t])
        return result
    return hook


def make_hook_scalar(bit_widths: List[int]):
    quant_fns = {2: quantize_int2, 4: quantize_int4, 8: quantize_int8, 16: lambda x: x}
    def hook(module, inp, out):
        result = out.clone()
        for t in range(min(out.shape[1], len(bit_widths))):
            result[:, t] = quant_fns.get(bit_widths[t], lambda x: x)(out[:, t])
        return result
    return hook


def score_continuation(model, tokenizer, context, continuation, bit_widths, use_tq, device=DEVICE):
    import torch.nn.functional as F
    full_ids = tokenizer.encode(context + " " + continuation, return_tensors="pt").to(device)
    ctx_len = tokenizer.encode(context, return_tensors="pt").shape[1]
    hooks = []
    for name, module in model.named_modules():
        if name.endswith(".k_proj"):
            h = module.register_forward_hook(make_hook_tq(bit_widths, True) if use_tq else make_hook_scalar(bit_widths))
            hooks.append(h)
        elif name.endswith(".v_proj"):
            h = module.register_forward_hook(make_hook_tq(bit_widths, False) if use_tq else make_hook_scalar(bit_widths))
            hooks.append(h)
    try:
        with torch.no_grad():
            logits = model(full_ids).logits[0]
    finally:
        for h in hooks:
            h.remove()
    cont_ids = full_ids[0, ctx_len:]
    if len(cont_ids) == 0:
        return -1e9
    log_probs = F.log_softmax(logits[ctx_len-1:ctx_len-1+len(cont_ids)], dim=-1)
    return log_probs[range(len(cont_ids)), cont_ids].sum().item()


def answer_key_to_idx(key, num_choices):
    """Convert ARC answerKey ('A','B','C','D' or '1','2','3','4') to 0-based index."""
    if key in "ABCDE":
        return ord(key) - ord('A')
    try:
        return int(key) - 1
    except ValueError:
        return 0


def run_arc_eval(model, eager_model, tokenizer, controller, ds, use_tq, freq_table):
    from eval_dwb import extract_signals_for_sequence, predict_bit_widths
    correct = 0
    bit_dist = Counter()
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i > 0 and i % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(ds) - i)
            avg_bits = sum(b*c for b,c in bit_dist.items()) / max(1, sum(bit_dist.values()))
            print(f"  [{i}/{LIMIT}] acc={correct/i*100:.1f}% avg_bits={avg_bits:.2f} eta={eta:.0f}s", flush=True)

        question = ex["question"]
        choices = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        correct_idx = answer_key_to_idx(answer_key, len(choices))

        # Extract DWB signals from question context
        signals, _ = extract_signals_for_sequence(eager_model, tokenizer, question, freq_table, DEVICE)
        bit_widths = predict_bit_widths(controller, signals, DEVICE)
        for b in bit_widths:
            bit_dist[b] += 1

        scores = [score_continuation(model, tokenizer, question, c, bit_widths, use_tq) for c in choices]
        if scores and max(range(len(scores)), key=lambda j: scores[j]) == correct_idx:
            correct += 1

    elapsed = time.time() - t0
    acc = correct / len(ds) * 100
    total_tokens = sum(bit_dist.values())
    bit_pct = {b: round(c/total_tokens*100, 1) for b,c in sorted(bit_dist.items())}
    avg_bits = sum(b*c for b,c in bit_dist.items()) / max(1, total_tokens)
    print(f"  Result: {correct}/{len(ds)} = {acc:.2f}%  ({elapsed:.0f}s)", flush=True)
    print(f"  Bit distribution: {bit_pct}, avg={avg_bits:.2f}", flush=True)
    return {"accuracy": acc, "correct": correct, "total": len(ds),
            "elapsed_s": round(elapsed, 1), "bit_pct": bit_pct, "avg_bits": avg_bits}


# ── Load models ──────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID}...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

print(f"Reloading with eager attention...", flush=True)
eager_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager"
).eval()

from dwb_implementation import DWBController
controller = DWBController()
controller.load_state_dict(torch.load(CONTROLLER_PATH, map_location=DEVICE, weights_only=True))
controller.eval()

from datasets import load_dataset
print(f"Loading ARC-Challenge ({LIMIT} samples)...", flush=True)
arc_ds = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(LIMIT)))

# Build freq table from question text
freq_table = Counter()
for ex in arc_ds:
    freq_table.update(tokenizer.encode(ex["question"]))

results = {}

# FP16 baseline on ARC
print(f"\n--- FP16 baseline (ARC-Challenge) ---", flush=True)
from eval_dwb import extract_signals_for_sequence, predict_bit_widths
import torch.nn.functional as F

def score_fp16(model, tokenizer, question, choice, device=DEVICE):
    full_ids = tokenizer.encode(question + " " + choice, return_tensors="pt").to(device)
    ctx_len = tokenizer.encode(question, return_tensors="pt").shape[1]
    with torch.no_grad():
        logits = model(full_ids).logits[0]
    cont_ids = full_ids[0, ctx_len:]
    if len(cont_ids) == 0:
        return -1e9
    log_probs = F.log_softmax(logits[ctx_len-1:ctx_len-1+len(cont_ids)], dim=-1)
    return log_probs[range(len(cont_ids)), cont_ids].sum().item()

correct_fp16 = 0
for i, ex in enumerate(arc_ds):
    scores = [score_fp16(model, tokenizer, ex["question"], c) for c in ex["choices"]["text"]]
    correct_idx = answer_key_to_idx(ex["answerKey"], len(ex["choices"]["text"]))
    if max(range(len(scores)), key=lambda j: scores[j]) == correct_idx:
        correct_fp16 += 1
fp16_acc = correct_fp16 / len(arc_ds) * 100
print(f"  FP16: {correct_fp16}/{len(arc_ds)} = {fp16_acc:.1f}%", flush=True)
results["fp16"] = {"accuracy": fp16_acc, "correct": correct_fp16, "total": len(arc_ds)}

# DWB-scalar
print(f"\n--- DWB-scalar (ARC-Challenge) ---", flush=True)
results["dwb_scalar"] = run_arc_eval(model, eager_model, tokenizer, controller, arc_ds, use_tq=False, freq_table=freq_table)

# DWB-TurboQuant
print(f"\n--- DWB-TurboQuant (ARC-Challenge) ---", flush=True)
results["dwb_turboquant"] = run_arc_eval(model, eager_model, tokenizer, controller, arc_ds, use_tq=True, freq_table=freq_table)

del eager_model

out = {
    "model": MODEL_ID, "benchmark": "ARC-Challenge", "limit": LIMIT,
    "metric": "acc_unnorm", "date": datetime.now().isoformat(),
    "hellaswag_reference": {"dwb_scalar": 40.0, "dwb_turboquant": 42.0, "delta_pp": 2.0},
    "conditions": results
}
fname = Path("research/data") / f"tq_h3_arc_{LIMIT}samp_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {fname}", flush=True)

print("\n=== TQ-H3 RESULTS (ARC-Challenge) ===")
fp16 = results["fp16"]["accuracy"]
scalar = results["dwb_scalar"]["accuracy"]
tq = results["dwb_turboquant"]["accuracy"]
hellaswag_delta = 2.0
arc_delta = tq - scalar
print(f"  FP16:            {fp16:.1f}%")
print(f"  DWB-scalar:      {scalar:.1f}%")
print(f"  DWB-TurboQuant:  {tq:.1f}%")
print(f"  Delta (TQ-H3):   {arc_delta:+.1f}pp")
print(f"  HellaSwag delta: +{hellaswag_delta:.1f}pp (reference)")
print(f"  TQ-H3 (larger on reasoning): {'CONFIRMED' if arc_delta > hellaswag_delta else 'SAME' if arc_delta >= hellaswag_delta - 1 else 'NOT CONFIRMED'}")
