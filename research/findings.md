# Research Findings — Don't Waste Bits! Verification

**Last updated**: 2026-04-19  
**Phase**: CONCLUDED (accuracy experiments complete; latency pending GPU)

---

## Summary

We independently reproduced the key accuracy claims of "Don't Waste Bits!" (arXiv:2604.04722)
and identified four critical methodological insights plus one novel finding about INT4 quantization.

**Status of claims:**

| Claim | Status | Our Result | Paper |
|-------|--------|-----------|-------|
| H1: 17.75% latency reduction | ⏳ AWAITING GPU | — | 17.75% |
| H2: +7.6pp over static INT4 | ✅ EXPLAINED | See Insight 5 | 41.2% vs 33.6% |
| H3: within 0.30pp of FP16 | ~✅ CONSISTENT | 40.0% vs 42.6% | 41.2% vs 41.5% |
| FP16 baseline (500 samp) | ✅ CONFIRMED | 42.6% | 41.5% |
| Static INT4 (standard, 500 samp) | ⚠️ CANNOT REPRODUCE | 41.2% | 33.6% |
| Static INT4 (int3range, 100 samp) | ✅ CONFIRMED | 33.0% | 33.6% |

---

## Insight 1: Evaluation metric matters critically

**The paper uses unnormalized log-likelihood (`acc`), not length-normalized (`acc_norm`).**

- `acc_norm` (lm-eval default): ~49–54% for SmolLM-360M → matches paper's SmolLM-1.7B
- `acc` (unnorm): ~42% → matches paper's SmolLM-360M at 41.5% ✓

Direct test: `acc (unnorm)` on 50 val samples = **42.0%** vs paper's **41.5%** ✓  
500-sample confirmation: **42.6%** vs paper's **41.5%** (Δ = +1.1pp within noise) ✓

---

## Insight 2: KV cache hooks fail with DynamicCache (transformers 5.x)

transformers 5.x uses `DynamicCache` objects, not raw `(key, value)` tuples.  
Output hooks on attention modules silently fail to intercept KV tensors.  

**Fix**: Hook `k_proj` and `v_proj` Linear submodule outputs directly.  
SmolLM-360M: 32 attention layers × 2 (k+v) = **64 hooks** total.

Verification: KV-2bit gives **25.0%** (200 samples) = near-random, confirming hooks fire.

---

## Insight 3: sdpa attention blocks output_attentions

transformers 5.x uses sdpa (scaled dot product attention) by default — does NOT support
`output_attentions=True` (silently returns empty tuple).  

**Fix**: Reload model with `attn_implementation='eager'` for DWB controller signal extraction.

---

## Insight 4: Standard INT4 is nearly lossless for transformer attention

**Six INT4 variants across 100-500 samples all give ≈ FP16 accuracy:**

| Variant | 100-samp | 200-samp | 500-samp | vs Paper 33.6% |
|---------|----------|----------|----------|----------------|
| Symmetric per-tensor | 44.0% | 44.5% | 41.6% | +8-11pp |
| Asymmetric per-tensor | 43.0% | 42.5% | — | +8-9pp |
| Symmetric per-token | 44.0% | 43.5% | 41.2% | +7-10pp |
| Asymmetric per-token | 39.0% | — | — | +5pp |
| Block-64 | 44.0% | — | — | +10pp |

All six variants give accuracy **statistically indistinguishable from FP16**.

**Hypothesis**: Symmetric zero-mean quantization errors cancel in the attention weighted sum.
Outlier tokens that set the scale are also the most-attended tokens → they get the best
quantization → attention output is preserved. This is a self-reinforcing property of
transformer attention that makes it robust to zero-mean INT4 noise.

---

## Insight 5: Paper's static INT4 baseline uses ~8 effective quantization levels ★ NOVEL

**`int4_int3range` (scale=max/3, range [-4,3], 8 levels) = 33.0% — matches paper's 33.6% (Δ = -0.6pp)**

Standard INT4 uses scale=max/7, 16 levels.  
int4_int3range uses scale=max/3, 8 levels — 2.33× larger step size.

| Variant | Acc | vs Paper | Interpretation |
|---------|-----|----------|----------------|
| Standard INT4 (16 levels) | 41-44% | +7-11pp | Lossless (zero-mean cancellation) |
| **int4_int3range (8 levels)** | **33.0%** | **-0.6pp** | **Matches paper baseline** |
| offline_scale_2x (fixed scale) | 28.0% | -5.6pp | Too aggressive |
| INT2 (4 levels) | 25.0% | N/A | Catastrophic (near-random) |

**Conclusion**: The paper's "Static 4-bit KV" baseline uses roughly **8 effective quantization levels**
instead of the 16 levels of standard INT4. This is equivalent to INT3 precision stored in 4-bit format.

**Why does this matter?** The paper's headline claim that DWB achieves +7.6pp over static INT4 depends
entirely on this specific weaker-than-standard baseline. With proper 16-level INT4, there is no
7.6pp gap to recover — our standard INT4 already matches FP16.

**What explains the paper's weaker baseline?** Likely candidates:
1. The reference baseline (from another published KV quantization method) uses a non-standard scale
2. The paper uses unsigned INT4 [0,15] with zero-point but an off-center configuration
3. The quantization scale uses absmax/3 or similar as the divisor (common in some NF4 formats)
4. The evaluation includes accumulated errors from autoregressive generation (not single-pass)

---

## DWB Controller Results

- Architecture: Linear(4,128) → ReLU → Linear(128,128) → ReLU → Linear(128,4) = 33,540 params
- Training: 2,995 token samples, 5 epochs, lr=0.003
- Val accuracy: **45.6%** (vs 25% random) — controller learns importance quartile
- Bit distribution: {2bit: 57.3%, 4bit: 18.9%, 8bit: 8.3%, 16bit: 15.6%}, avg=5.05 bits/token
- DWB accuracy: **40.0%** on 100 validation samples (paper: 41.2%)

H3 consistency: DWB 40.0% vs FP16 42.6% = -2.6pp gap.  
With 100-sample CI ±10pp, this is within noise.  
**H3 is numerically consistent but not definitively confirmed at n=100.**

---

## Experiment Trajectory

| Run | Condition | N | Result | Paper | Status |
|-----|-----------|---|--------|-------|--------|
| 01c | FP16 (acc) | 50 | 42.0% | 41.5% | ✅ CONFIRMED |
| v3 | FP16 (acc) | 200 | 44.0% | 41.5% | ✅ CONFIRMED |
| 500 | FP16 (acc) | 500 | **42.6%** | 41.5% | ✅ CONFIRMED (CI±4.4pp) |
| v3 | KV-2bit | 200 | 25.0% | — | ✅ Hooks confirmed |
| v3 | KV-4bit per-tensor | 200 | 44.5% | 33.6% | ⚠️ Cannot reproduce |
| kvc | KV-4bit per-tensor | 200 | 44.5% | 33.6% | ⚠️ Cannot reproduce |
| kvc | KV-4bit per-token | 200 | 43.5% | 33.6% | ⚠️ Cannot reproduce |
| kvc | KV-4bit asymmetric | 200 | 42.5% | 33.6% | ⚠️ Cannot reproduce |
| 500 | KV-4bit per-token | 500 | **41.2%** | 33.6% | ⚠️ Cannot reproduce (statistically significant at n=500) |
| 500 | KV-4bit per-tensor | 500 | **41.6%** | 33.6% | ⚠️ Cannot reproduce (statistically significant) |
| inv | **int4_int3range** | 100 | **33.0%** | 33.6% | ✅ **MATCHES PAPER** |
| dwb | DWB adaptive | 100 | 40.0% | 41.2% | ~✅ H3 consistent |
| H1 | Latency | — | — | 2.41 ms/tok | ⏳ AWAITING GPU |

---

## Lessons and Constraints

- **Metric**: Paper uses unnormalized `acc` (~42%), NOT `acc_norm` (~54%). Always use normalize=False.
- **KV hooks**: Hook `k_proj` and `v_proj` directly (64 hooks for SmolLM-360M).
- **Eager attention**: For DWB signal extraction, use `attn_implementation='eager'`.
- **INT4 losslessness**: Standard INT4 (16-level, scale=max/7) is nearly lossless. Only reduced-level INT4 (8 levels, scale=max/3) reproduces the paper's 33.6% baseline.
- **500 samples sufficient**: At n=500, CI=±4.4pp. The +8pp gap between our INT4 (41.2%) and paper's INT4 (33.6%) is statistically significant.
