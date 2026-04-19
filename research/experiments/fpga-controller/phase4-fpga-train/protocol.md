# Phase 4 Protocol: FPGA-Aware Controller Training

**Hypothesis:** Replacing avg_bits with the FPGA latency proxy in the compound loss will yield
a controller that finds a better Pareto front on FPGA hardware — potentially more aggressive
compression (fewer bits) at equal or better accuracy vs paper's CPU-optimized controller.

**Prediction:** Phase 4 controller achieves ≥41.2% HellaSwag at avg_bits ≤4.5 on the FPGA
latency model (better compression than paper at equal accuracy).

## Method

Start from best Phase 2 config (Gumbel-softmax + richer features).
Replace loss term:
```
# Before (Phase 2):
loss = alpha * ce_loss + beta * avg_bits

# After (Phase 4):
from fpga_latency_model import fpga_latency_cycles
fpga_lat = fpga_latency_cycles(bits_per_token, ...)
loss = alpha * ce_loss + beta * fpga_lat
```

## Why This Could Beat the Paper

Paper optimizes for `avg_bits` (CPU proxy). On FPGA:
- 3-bit has same port cost as 4-bit → controller can target 3.x avg_bits without FPGA penalty
- 5-bit to 7-bit cost same as 8-bit → avoid this range entirely
- Optimal FPGA allocation clusters at port boundaries: mostly 4-bit with some 8-bit, minimal 16-bit

A controller trained on the FPGA cost function will naturally cluster at 4/8-bit boundaries —
potentially achieving lower avg_bits than the paper at equivalent or better FPGA latency.

## Files
- `code/run_phase4_fpga_train.py`
- `results/`
- `analysis.md`
