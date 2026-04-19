# Phase 1 Gumbel Analysis

## Results Summary

| Version | Accuracy | avg_bits | Bit Dist | Notes |
|---|---|---|---|---|
| v1 (raw CE + β·bits) | 26.5% | 2.0 | {2:100%} | 2-bit collapse — CE and bits on different scales |
| v2 (normalized quality) | **41.0%** | **4.0** | {4:100%} | 4-bit collapse — actually matches paper accuracy |
| Paper DWB | 41.2% | 5.05 | mixed | Target |

## Root Cause of v2 Collapse (to 4-bit)

v2 fixed the 2-bit collapse but collapsed to 4-bit instead. Root cause:

**Quality scores derived from standard INT4 overestimate 4-bit quality at 360M.**

With alpha=1.0, beta=0.3, and quality scores from standard INT4 (41.6%):
```
2-bit:  1.0*(1-0.0)  + 0.3*(2/16)  = 1.038  ← worst
4-bit:  1.0*(1-0.75) + 0.3*(4/16)  = 0.325  ← local minimum
8-bit:  1.0*(1-0.98) + 0.3*(8/16)  = 0.170  ← global minimum (should win)
16-bit: 1.0*(1-1.0)  + 0.3*(16/16) = 0.300
```

Analytically, 8-bit should dominate. But the controller converges to 4-bit because:
1. **Tau decays too fast** — tau reaches 0.65 by epoch 3, making Gumbel very sharp.
   Once controller latches onto 4-bit (epoch 2: 93%), tau locks it in.
2. **INT4 losslessness** — standard 4-bit IS nearly lossless at 360M (41.6% vs 42.6% FP16).
   The quality gap between 4-bit and 8-bit is tiny (0.02 loss_units), easily lost in noise.

## Key Insight: Wrong Quality Baseline

The paper's controller operates on top of its own **lossy 4-bit baseline (int3-range, 8 levels)**.
Our H2 finding established: int3-range gives 33.0% at 360M (vs 41.6% for standard INT4).

With int3-range as the 4-bit quality:
```
4-bit quality = (33.0 - 25.0) / (42.6 - 25.0) = 0.455   (lossy — was 0.75)
```

Recalculating per-token cost with int3-range quality, alpha=1.0, beta=0.3:
```
2-bit:  1.0*(1-0.0)   + 0.3*(2/16)  = 1.038
4-bit:  1.0*(1-0.455) + 0.3*(4/16)  = 0.620  ← high, not attractive
8-bit:  1.0*(1-0.966) + 0.3*(8/16)  = 0.184  ← global minimum
16-bit: 1.0*(1-1.0)   + 0.3*(16/16) = 0.300
```

With int3-range quality scores, **8-bit wins clearly over 4-bit** (0.184 vs 0.620).
The controller would then learn: promote lossy-4-bit tokens to 8-bit where it matters,
keep 2-bit where tokens are unimportant — producing exactly the mixed distribution
the paper shows.

## Why the Paper Gets 5.05 avg_bits

Paper's dynamic allocation is correction over a **lossy** baseline:
- Lossy 2/4-bit for unimportant tokens (pushes average down)
- 8/16-bit for important tokens (restores accuracy)
- Average: ~5.05 bits

With standard INT4 (lossless at 360M), dynamic allocation is unnecessary — 4-bit everywhere
achieves the same accuracy. The paper's gain is specifically over their lossy int3-range baseline.

## Plan: Phase 1 v3

Use int3-range quality scores to drive genuine mixed allocation:

```python
# Quality from int3-range (paper's actual 4-bit), not standard INT4
INT3_RANGE_ACC = 33.0   # our measurement
QUALITY_SCORES = [
    (25.0 - 25.0) / (42.6 - 25.0),  # 2-bit → 0.0
    (33.0 - 25.0) / (42.6 - 25.0),  # 4-bit (int3-range) → 0.455
    (42.0 - 25.0) / (42.6 - 25.0),  # 8-bit → 0.966
    (42.6 - 25.0) / (42.6 - 25.0),  # 16-bit → 1.0
]
```

Also fix tau schedule:
- Start higher (tau=3.0) for more exploration
- Decay more slowly (end at tau=0.3, not 0.1) to avoid premature locking

Expected result: mixed {2, 4, 8} distribution averaging ~5.05 bits at ~41%+ accuracy,
replicating the paper's dynamic allocation pattern.
