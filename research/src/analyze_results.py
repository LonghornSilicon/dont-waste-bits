"""Quick analysis of all INT4 investigation + KV comparison results."""
import json
from pathlib import Path
from glob import glob

data_dir = Path("research/data")

# INT4 investigation
inv_files = sorted(glob(str(data_dir / "int4_investigation_*.json")))
if inv_files:
    with open(inv_files[-1]) as f:
        inv = json.load(f)
    print("=== INT4 Investigation ===")
    print(f"FP16 baseline: {inv['results']['fp16']:.1f}%  (paper: {inv['paper_fp16']}%)")
    print(f"\n{'Variant':<30} {'Acc':>7}  {'Delta vs FP16':>14}  {'vs Paper 33.6%':>15}")
    print("-" * 72)
    fp16 = inv['results']['fp16']
    for variant, acc in inv['results'].items():
        if variant == 'fp16':
            continue
        delta_fp16 = acc - fp16
        delta_paper = acc - inv['paper_kv4']
        match = " ← MATCHES PAPER" if abs(acc - inv['paper_kv4']) < 3 else ""
        print(f"{variant:<30} {acc:>6.1f}%  {delta_fp16:>+12.1f}pp  {delta_paper:>+12.1f}pp{match}")
else:
    print("INT4 investigation not yet complete")

# KV comparison
kv_files = sorted(glob(str(data_dir / "kv_comparison_*.json")))
if kv_files:
    with open(kv_files[-1]) as f:
        kv = json.load(f)
    print("\n=== KV Comparison (200 samples) ===")
    for cond, r in kv.get("conditions", {}).items():
        delta_str = f"{r['delta']:+.2f}pp" if r.get('delta') is not None else "—"
        print(f"  {cond:<28} {r['accuracy']:>6.2f}%  paper: {r.get('paper_target','—')}%  delta: {delta_str}")
