# FPGA-Aware Adaptive KV-Cache Quantization

**Branch:** `fpga-controller`  
**Started:** 2026-04-19  
**Goal:** Close and beat the DWB paper's controller (41.2% @ 5.05 avg_bits on SmolLM-360M HellaSwag) by:
1. Replacing quartile-classification loss with a differentiable compound loss (Gumbel-softmax)
2. Adding richer controller features (head entropy, layer depth)
3. Replacing the avg_bits latency proxy with a cycle-accurate FPGA memory model

## Hypothesis

The paper's compound loss `L = α·CE + β·latency + γ·quality` forces the controller to find the
accuracy/compression Pareto front. Our quartile-classification approach cannot simultaneously optimize
both objectives — confirmed by beta sweep (no β achieves 41.2% at 5.05 bits).

By re-implementing the compound loss with richer features and an FPGA-specific latency target,
we expect to:
- Match or exceed 41.2% accuracy at ≤5.05 avg_bits (CPU baseline)
- Show a better Pareto front than the paper on FPGA-relevant hardware

## Phases

| Phase | Description | Target |
|---|---|---|
| 1 | Gumbel-softmax compound loss | ≥41.2% @ ≤5.05 bits |
| 2 | Richer features (head entropy + layer depth) | Improve over Phase 1 |
| 3 | FPGA cycle-accurate latency model | Latency proxy for FPGA |
| 4 | Retrain with FPGA latency target | Better Pareto on FPGA |
| 5 | Full benchmark (HellaSwag + ARC + BoolQ) | Beat paper across tasks |

## Key Baselines

| Condition | HellaSwag | avg_bits | Source |
|---|---|---|---|
| FP16 | 42.6% | 16.0 | Our measurement |
| Paper DWB | 41.2% | 5.05 | Table 3 |
| Our DWB v2 | 37.0% | 8.47 | run dwb_v2_500train |
| INT4 standard | 41.6% | 4.0 | Our measurement |

## Files

- `phase1-gumbel/` — Gumbel-softmax compound loss
- `phase2-features/` — Enriched controller features
- `phase3-fpga-model/` — FPGA latency model
- `phase4-fpga-train/` — FPGA-aware training
- `phase5-benchmark/` — Full evaluation
