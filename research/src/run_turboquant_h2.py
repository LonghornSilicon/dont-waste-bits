"""
TQ-H2: DWB-TurboQuant full pipeline.

Comparison:
  DWB-scalar (baseline): per-token {2,4,8,16} bit routing with scalar quantization
  DWB-TurboQuant (novel): same routing, but 2-bit tier uses PolarQuant (per-head WHT)

Prediction: DWB-TurboQuant > DWB-scalar on HellaSwag accuracy (H3 recovered).
Expected: +2 to +5pp improvement by improving quality of 57%-of-tokens at 2-bit tier.
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
HEAD_DIM = 64  # SmolLM-360M: head_dim=64


def polar_quant_single_token(x: torch.Tensor, quant: PolarQuant) -> torch.Tensor:
    """
    Apply PolarQuant per head to a single token slice.
    x: [batch, n_kv_heads * head_dim]  (a single token position)
    """
    batch, d = x.shape
    n_heads = d // HEAD_DIM
    if d % HEAD_DIM != 0:
        return quant.quantize_dequantize(x)
    x_heads = x.view(batch * n_heads, HEAD_DIM)
    x_out = quant.quantize_dequantize(x_heads)
    return x_out.view(batch, d)


def make_per_token_hook_tq(bit_widths: List[int], is_key: bool):
    """
    Per-token hook that routes 2-bit assignments to PolarQuant.
    is_key: True for k_proj (3-bit PolarQuant), False for v_proj (2-bit PolarQuant).
    """
    key_quant = PolarQuant(bits=3, seed=42)
    val_quant = PolarQuant(bits=2, seed=137)
    polar = key_quant if is_key else val_quant

    quant_fns = {
        2: lambda x: polar_quant_single_token(x, polar),
        4: quantize_int4,
        8: quantize_int8,
        16: lambda x: x,
    }

    def hook(module, inp, out):
        result = out.clone()
        for t in range(min(out.shape[1], len(bit_widths))):
            bits = bit_widths[t]
            result[:, t] = quant_fns.get(bits, lambda x: x)(out[:, t])
        return result

    return hook


def make_per_token_hook_scalar(bit_widths: List[int]):
    """Original DWB scalar routing (baseline)."""
    quant_fns = {2: quantize_int2, 4: quantize_int4, 8: quantize_int8, 16: lambda x: x}

    def hook(module, inp, out):
        result = out.clone()
        for t in range(min(out.shape[1], len(bit_widths))):
            bits = bit_widths[t]
            result[:, t] = quant_fns.get(bits, lambda x: x)(out[:, t])
        return result

    return hook


def score_dwb(model, tokenizer, context, continuation, bit_widths, use_tq=False, device=DEVICE):
    """Score HellaSwag continuation with DWB routing."""
    full_ids = tokenizer.encode(context + continuation, return_tensors="pt").to(device)
    ctx_len = tokenizer.encode(context, return_tensors="pt").shape[1]

    hooks = []
    for name, module in model.named_modules():
        if name.endswith(".k_proj"):
            if use_tq:
                h = module.register_forward_hook(make_per_token_hook_tq(bit_widths, is_key=True))
            else:
                h = module.register_forward_hook(make_per_token_hook_scalar(bit_widths))
            hooks.append(h)
        elif name.endswith(".v_proj"):
            if use_tq:
                h = module.register_forward_hook(make_per_token_hook_tq(bit_widths, is_key=False))
            else:
                h = module.register_forward_hook(make_per_token_hook_scalar(bit_widths))
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
    import torch.nn.functional as F
    log_probs = F.log_softmax(logits[ctx_len-1:ctx_len-1+len(cont_ids)], dim=-1)
    return log_probs[range(len(cont_ids)), cont_ids].sum().item()


def run_eval(model, eager_model, tokenizer, controller, ds, use_tq, freq_table):
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
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        signals, _ = extract_signals_for_sequence(eager_model, tokenizer, ctx, freq_table, DEVICE)
        bit_widths = predict_bit_widths(controller, signals, DEVICE)
        for b in bit_widths:
            bit_dist[b] += 1
        scores = [score_dwb(model, tokenizer, ctx, " " + e, bit_widths, use_tq) for e in ex["endings"]]
        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
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


# Load model and controller
print(f"Loading {MODEL_ID}...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

print(f"Reloading with eager attention for signal extraction...", flush=True)
eager_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager"
).eval()

from dwb_implementation import DWBController
controller = DWBController()
controller.load_state_dict(torch.load(CONTROLLER_PATH, map_location=DEVICE, weights_only=True))
controller.eval()

from datasets import load_dataset
print(f"Loading HellaSwag ({LIMIT} samples)...", flush=True)
ds = list(load_dataset("Rowan/hellaswag", split="validation").select(range(LIMIT)))

# Build freq table
freq_table = Counter()
for ex in ds:
    ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    freq_table.update(tokenizer.encode(ctx))

results = {}

# DWB-scalar (baseline)
print(f"\n--- DWB-scalar (baseline) ---", flush=True)
results["dwb_scalar"] = run_eval(model, eager_model, tokenizer, controller, ds, use_tq=False, freq_table=freq_table)

# DWB-TurboQuant (novel)
print(f"\n--- DWB-TurboQuant (2-bit routed to PolarQuant) ---", flush=True)
results["dwb_turboquant"] = run_eval(model, eager_model, tokenizer, controller, ds, use_tq=True, freq_table=freq_table)

del eager_model

out = {"model": MODEL_ID, "limit": LIMIT, "metric": "acc_unnorm",
       "date": datetime.now().isoformat(),
       "baselines": {"verified_fp16_500samp": 42.6, "paper_dwb": 41.2, "verified_dwb_200samp": 38.0},
       "conditions": results}
fname = Path("research/data") / f"tq_h2_eval_{LIMIT}samp_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {fname}", flush=True)

print("\n=== TQ-H2 RESULTS ===")
scalar = results["dwb_scalar"]["accuracy"]
tq = results["dwb_turboquant"]["accuracy"]
print(f"  DWB-scalar:      {scalar:.1f}%")
print(f"  DWB-TurboQuant:  {tq:.1f}%")
print(f"  Delta:           {tq-scalar:+.1f}pp")
print(f"  TQ-H2: {'CONFIRMED' if tq > scalar + 1 else 'WEAK' if tq > scalar else 'REFUTED'}")
