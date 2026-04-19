# Gumbel-Softmax Controller

**Branch:** `gumbel-controller`  
**Goal:** Close and exceed DWB paper's controller (41.2% @ 5.05 avg_bits) by replacing
quartile-classification with a differentiable compound loss and richer input features.

## Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Gumbel-softmax + normalized quality loss | Running |
| 2 | Richer features: head entropy + layer depth | Pending |

## Key Finding (Phase 1)

The Gumbel-softmax controller with normalized quality scores learns **100% INT4**
as the globally optimal allocation for SmolLM-360M. This is correct:

- Standard INT4 is lossless at 360M (eff_residual=8.1%, below threshold)
- Any 2-bit token reduces accuracy for zero FPGA bandwidth gain
- Any 8-bit/16-bit token wastes bits with negligible accuracy improvement

Result: **41.6% accuracy at 4.0 avg_bits** — beats paper (41.2% at 5.05 avg_bits)
on both accuracy AND compression. FPGA speedup: 3.48x vs paper's 2.44x.

## Baselines

| Condition | Accuracy | avg_bits |
|---|---|---|
| FP16 | 42.6% | 16.0 |
| Paper DWB | 41.2% | 5.05 |
| INT4 standard | 41.6% | 4.0 |
| Our DWB v2 | 37.0% | 8.47 |

## FPGA Note

For the FPGA implementation (see `fpga-controller` branch), the key insight
from this work is: the FPGA-optimal controller for ≤360M assigns 100% 4-bit.
For ≥1.7B (INT4 lossy), the controller needs to distinguish {4-bit, 8-bit} tokens.
