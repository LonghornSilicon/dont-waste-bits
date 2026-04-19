"""
Controller Behavior Analysis — What did the DWB controller learn?

Collects per-token signals [H_t, R_t, V_t, C_t] and their assigned bit-widths
across a HellaSwag sample. Reports:
  - Mean/std of each signal per bit tier (2/4/8/16)
  - Which signal most distinguishes 2-bit (unimportant) vs 16-bit (critical)
  - Token type examples per tier (actual token strings)

This is purely analytical — no new model training. Uses existing controller.
"""

import sys, json, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from dwb_implementation import DWBController, BIT_CLASSES
from eval_dwb import extract_signals_for_sequence, predict_bit_widths

MODEL_ID = "HuggingFaceTB/SmolLM-360M"
CONTROLLER_PATH = "research/data/dwb_controller_smollm360m.pt"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 50
DEVICE = "cpu"
SIGNAL_NAMES = ["H_t (entropy)", "R_t (rarity)", "V_t (attn_var)", "C_t (confidence)"]

print(f"Loading {MODEL_ID}...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
eager_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager"
).eval()

controller = DWBController()
controller.load_state_dict(torch.load(CONTROLLER_PATH, map_location=DEVICE, weights_only=True))
controller.eval()

from datasets import load_dataset
ds = list(load_dataset("Rowan/hellaswag", split="validation").select(range(LIMIT)))

# Build frequency table
freq_table = Counter()
for ex in ds:
    ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    freq_table.update(tokenizer.encode(ctx))

# Collect signals + bit assignments + token strings per tier
signals_by_bit = defaultdict(list)   # {bits: [[H,R,V,C], ...]}
tokens_by_bit = defaultdict(list)    # {bits: [token_str, ...]}
all_signals = []
all_bits = []

print(f"Collecting signals from {LIMIT} HellaSwag examples...", flush=True)
for i, ex in enumerate(ds):
    if i % 10 == 0:
        print(f"  [{i}/{LIMIT}]", flush=True)
    ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    signals, token_ids = extract_signals_for_sequence(eager_model, tokenizer, ctx, freq_table, DEVICE)
    bit_widths = predict_bit_widths(controller, signals, DEVICE)

    for t, (sig, bits) in enumerate(zip(signals, bit_widths)):
        s = sig.tolist()
        signals_by_bit[bits].append(s)
        all_signals.append(s)
        all_bits.append(bits)
        # Decode token
        if t < len(token_ids):
            tok_str = tokenizer.decode([token_ids[t]])
            tokens_by_bit[bits].append(tok_str)

print(f"\nCollected {len(all_bits)} tokens total.", flush=True)

# ── Summary statistics ─────────────────────────────────────────────────────
results = {}
total_tokens = len(all_bits)
bit_counts = Counter(all_bits)

print("\n=== BIT DISTRIBUTION ===")
for bits in BIT_CLASSES:
    n = bit_counts[bits]
    pct = n / total_tokens * 100
    print(f"  {bits:2d}-bit: {n:5d} tokens ({pct:.1f}%)")

print("\n=== SIGNAL MEANS BY BIT TIER (± std) ===")
print(f"{'Signal':<22} {'2-bit':>16} {'4-bit':>16} {'8-bit':>16} {'16-bit':>16}")
print("-" * 90)

signal_stats = {}
for sig_idx, sig_name in enumerate(SIGNAL_NAMES):
    row = {}
    for bits in BIT_CLASSES:
        vals = [s[sig_idx] for s in signals_by_bit[bits]] if signals_by_bit[bits] else [0]
        arr = np.array(vals)
        row[bits] = {"mean": float(arr.mean()), "std": float(arr.std())}
    signal_stats[sig_name] = row
    line = f"  {sig_name:<20}"
    for bits in BIT_CLASSES:
        line += f"  {row[bits]['mean']:+.3f}±{row[bits]['std']:.3f}"
    print(line)

# ── Discriminative power: which signal best separates 2-bit from 16-bit ──
print("\n=== DISCRIMINATIVE POWER: 2-bit vs 16-bit ===")
discriminability = {}
for sig_idx, sig_name in enumerate(SIGNAL_NAMES):
    vals_2 = np.array([s[sig_idx] for s in signals_by_bit[2]]) if signals_by_bit[2] else np.array([0])
    vals_16 = np.array([s[sig_idx] for s in signals_by_bit[16]]) if signals_by_bit[16] else np.array([0])
    # Cohen's d
    pooled_std = np.sqrt((vals_2.std()**2 + vals_16.std()**2) / 2 + 1e-8)
    d = (vals_16.mean() - vals_2.mean()) / pooled_std
    discriminability[sig_name] = abs(d)
    print(f"  {sig_name:<22}: Cohen's d = {d:+.3f}  (|d|={abs(d):.3f})")

most_disc = max(discriminability, key=discriminability.get)
print(f"\n  Most discriminative signal: {most_disc}")

# ── Example tokens per tier ────────────────────────────────────────────────
print("\n=== SAMPLE TOKENS PER BIT TIER (most common 20) ===")
for bits in BIT_CLASSES:
    toks = tokens_by_bit[bits]
    # Count and normalize
    cnt = Counter(t.strip() for t in toks if t.strip())
    top = cnt.most_common(20)
    top_str = ", ".join(f'"{t}"({c})' for t, c in top[:15])
    print(f"  {bits:2d}-bit: {top_str}")

# ── Save results ──────────────────────────────────────────────────────────
out = {
    "model": MODEL_ID,
    "n_examples": LIMIT,
    "n_tokens": total_tokens,
    "date": datetime.now().isoformat(),
    "bit_distribution": {str(b): {"count": bit_counts[b], "pct": round(bit_counts[b]/total_tokens*100,1)} for b in BIT_CLASSES},
    "signal_stats_by_bit": {
        sig_name: {str(b): row[b] for b in BIT_CLASSES}
        for sig_name, row in signal_stats.items()
    },
    "discriminability_cohens_d": {k: round(v, 4) for k, v in discriminability.items()},
    "most_discriminative_signal": most_disc,
    "top_tokens_by_bit": {
        str(b): [t for t, _ in Counter(s.strip() for s in tokens_by_bit[b] if s.strip()).most_common(30)]
        for b in BIT_CLASSES
    }
}
fname = Path("research/data") / f"controller_analysis_{LIMIT}ex_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {fname}", flush=True)
