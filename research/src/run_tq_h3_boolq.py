"""
TQ-H3 extension: DWB-TurboQuant vs DWB-scalar on BoolQ.

BoolQ: passage + question → Yes/No answer.
Scoring: log P("Yes"|passage+question) vs log P("No"|passage+question)
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
    return quant.quantize_dequantize(x.view(batch * n_heads, HEAD_DIM)).view(batch, d)

def make_hook_tq(bit_widths: List[int], is_key: bool):
    polar = PolarQuant(bits=3, seed=42) if is_key else PolarQuant(bits=2, seed=137)
    quant_fns = {2: lambda x: polar_quant_single_token(x, polar),
                 4: quantize_int4, 8: quantize_int8, 16: lambda x: x}
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

def score_token(model, tokenizer, context: str, token_str: str, bit_widths, use_tq, device=DEVICE):
    """Score P(token_str | context) under quantized KV cache."""
    import torch.nn.functional as F
    ctx_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    tok_id = tokenizer.encode(" " + token_str, add_special_tokens=False)
    if not tok_id:
        return -1e9
    full_ids = torch.cat([ctx_ids, torch.tensor([tok_id[:1]], device=device)], dim=1)

    hooks = []
    for name, module in model.named_modules():
        if name.endswith(".k_proj"):
            hooks.append(module.register_forward_hook(make_hook_tq(bit_widths, True) if use_tq else make_hook_scalar(bit_widths)))
        elif name.endswith(".v_proj"):
            hooks.append(module.register_forward_hook(make_hook_tq(bit_widths, False) if use_tq else make_hook_scalar(bit_widths)))
    try:
        with torch.no_grad():
            logits = model(full_ids).logits[0]
    finally:
        for h in hooks:
            h.remove()

    last_logits = logits[ctx_ids.shape[1] - 1]
    log_probs = F.log_softmax(last_logits, dim=-1)
    return log_probs[tok_id[0]].item()

def run_boolq_eval(model, eager_model, tokenizer, controller, ds, use_tq, freq_table):
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

        # BoolQ: passage + question as context
        passage = ex["passage"][:300]  # truncate long passages
        question = ex["question"]
        context = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
        label = 1 if ex["answer"] else 0  # True→1 (Yes), False→0 (No)

        signals, _ = extract_signals_for_sequence(eager_model, tokenizer, context, freq_table, DEVICE)
        bit_widths = predict_bit_widths(controller, signals, DEVICE)
        for b in bit_widths:
            bit_dist[b] += 1

        score_yes = score_token(model, tokenizer, context, "Yes", bit_widths, use_tq)
        score_no  = score_token(model, tokenizer, context, "No",  bit_widths, use_tq)
        predicted = 1 if score_yes > score_no else 0
        if predicted == label:
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

# ── Load ──────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID}...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
eager_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager").eval()

from dwb_implementation import DWBController
controller = DWBController()
controller.load_state_dict(torch.load(CONTROLLER_PATH, map_location=DEVICE, weights_only=True))
controller.eval()

from datasets import load_dataset
print(f"Loading BoolQ ({LIMIT} samples)...", flush=True)
bq_ds = list(load_dataset("google/boolq", split="validation").select(range(LIMIT)))

freq_table = Counter()
for ex in bq_ds:
    passage = ex["passage"][:300]
    context = f"Passage: {passage}\nQuestion: {ex['question']}\nAnswer:"
    freq_table.update(tokenizer.encode(context))

results = {}

# FP16 baseline
print(f"\n--- FP16 baseline (BoolQ) ---", flush=True)
import torch.nn.functional as F
correct_fp16 = 0
for ex in bq_ds:
    passage = ex["passage"][:300]
    context = f"Passage: {passage}\nQuestion: {ex['question']}\nAnswer:"
    ctx_ids = tokenizer.encode(context, return_tensors="pt")
    with torch.no_grad():
        logits = model(ctx_ids).logits[0, -1]
    lp = F.log_softmax(logits, dim=-1)
    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode(" No",  add_special_tokens=False)[0]
    pred = 1 if lp[yes_id] > lp[no_id] else 0
    if pred == (1 if ex["answer"] else 0):
        correct_fp16 += 1
fp16_acc = correct_fp16 / len(bq_ds) * 100
print(f"  FP16: {correct_fp16}/{len(bq_ds)} = {fp16_acc:.1f}%", flush=True)
results["fp16"] = {"accuracy": fp16_acc, "correct": correct_fp16, "total": len(bq_ds)}

print(f"\n--- DWB-scalar (BoolQ) ---", flush=True)
results["dwb_scalar"] = run_boolq_eval(model, eager_model, tokenizer, controller, bq_ds, use_tq=False, freq_table=freq_table)

print(f"\n--- DWB-TurboQuant (BoolQ) ---", flush=True)
results["dwb_turboquant"] = run_boolq_eval(model, eager_model, tokenizer, controller, bq_ds, use_tq=True, freq_table=freq_table)

out = {"model": MODEL_ID, "benchmark": "BoolQ", "limit": LIMIT, "date": datetime.now().isoformat(),
       "cross_benchmark_reference": {"hellaswag_delta": 2.0, "arc_delta": 3.0},
       "conditions": results}
fname = Path("research/data") / f"tq_h3_boolq_{LIMIT}samp_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {fname}", flush=True)

print("\n=== TQ-H3 BoolQ RESULTS ===")
fp16 = results["fp16"]["accuracy"]
scalar = results["dwb_scalar"]["accuracy"]
tq = results["dwb_turboquant"]["accuracy"]
delta = tq - scalar
print(f"  FP16:            {fp16:.1f}%")
print(f"  DWB-scalar:      {scalar:.1f}%")
print(f"  DWB-TurboQuant:  {tq:.1f}%")
print(f"  Delta:           {delta:+.1f}pp")
print(f"  HellaSwag:       +2.0pp | ARC-Challenge: +3.0pp | BoolQ: {delta:+.1f}pp")
