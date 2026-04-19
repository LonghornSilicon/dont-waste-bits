# Phase 5 Protocol: Full Benchmark

**Goal:** Comprehensive comparison of all controllers on all tasks. Produce paper-ready results tables.

## Conditions

| Condition | Description |
|---|---|
| FP16 | Baseline, no quantization |
| INT4-standard | Symmetric INT4, per-tensor scale |
| Paper DWB | Paper's reported numbers (from Table 3) |
| DWB-v2 | Our prior best (quartile-classification, 500 train, 10 ep) |
| Phase 1 | Gumbel-softmax, [C_t, R_t] features |
| Phase 2 | Gumbel-softmax, [C_t, R_t, H_t, L] features |
| Phase 4 | Phase 2 + FPGA latency target |
| DWB-TurboQuant | Phase 2/4 + WHT on 2-bit tier (extend TurboQuant) |

## Tasks

| Task | Metric | Samples |
|---|---|---|
| HellaSwag | acc (unnorm) | 500 |
| ARC-Challenge | acc (unnorm) | 500 |
| BoolQ | acc | 500 |

## Output

- Per-condition results table (accuracy × task)
- Pareto plot: accuracy vs avg_bits (CPU) and vs fpga_latency_cycles
- Bit distribution histogram per condition
- Controller sensitivity analysis (feature importance)

## Files
- `code/run_full_benchmark.py`
- `results/full_benchmark_YYYYMMDD.json`
- `analysis.md`
