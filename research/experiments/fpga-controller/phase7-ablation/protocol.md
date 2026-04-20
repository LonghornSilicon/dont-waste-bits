# Phase 7 Protocol: Routing Ablation on SmolLM-1.7B

**Goal**: Test whether the learned binary controller provides any routing-quality advantage over two trivial baselines, and whether random routing itself is stable across seeds.

## Phase 7c — Routing-strategy ablation (single seed, n=200)

**Question**: Given a fixed 4-bit / 8-bit split ratio that targets FPGA speedup ~2.81×, does the controller outperform (a) uniform random routing and (b) KV-norm (bottom-p by L2 norm → 4-bit)?

**Model**: `HuggingFaceTB/SmolLM-1.7B`
**Task**: HellaSwag validation, unnormalized log-likelihood, n=200 (95% CI ≈ ±7pp)
**Target split**: 74% 4-bit / 26% 8-bit → FPGA cost = 0.29·0.74 + 0.56·0.26 = 0.3602 → 1.010/0.3602 ≈ **2.81×**

**Strategies**:
| Name | Selection |
|---|---|
| random | Per-token Bernoulli(p=0.74) → 4-bit, else 8-bit |
| controller | Learned binary Gumbel controller (Phase 5 recipe; features [kv_norm, pos]) |
| kv_norm | Bottom-74% per-layer by L2 norm → 4-bit, top-26% → 8-bit |

**Outputs**: `results/phase7c_routing_ablation.json` with accuracy, bit distribution, FPGA cost/speedup per strategy.

## Phase 7d — 5-seed random-routing validation (n=500)

**Question**: How stable is random routing across seeds, and what is the mean±std accuracy that the controller / KV-norm must beat?

**Model / Task**: Same, n=500 (95% CI ≈ ±4.4pp per seed)
**Seeds**: 0, 1, 2, 3, 4
**Target split**: 74% 4-bit / 26% 8-bit (identical ratio to Phase 7c, speedup 2.81×)

**Outputs**: `results/phase7d_random_multiseed.json` — per-seed accuracy, mean, std, CI, bit distribution.

## Decision rule

If `controller_acc` and `kv_norm_acc` both fall inside `random_mean ± 2·random_std`, the controller confers no routing advantage at this ratio — random routing is the honest baseline the paper should report.
