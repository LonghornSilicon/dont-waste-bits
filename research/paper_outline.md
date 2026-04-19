# Paper Outline: Independent Verification of "Don't Waste Bits!"

**Target venue**: arXiv preprint (verification report) or NeurIPS/ICLR workshop
**Paper type**: Reproducibility study
**Status**: In progress — results completing

---

## Title Options

1. "Independent Verification of Don't Waste Bits!: Adaptive KV-Cache Quantization for Lightweight LLMs"
2. "Reproducing Adaptive KV-Cache Quantization: Lessons from Verifying arXiv:2604.04722"

---

## Abstract (draft)

We present an independent reproduction attempt of the key claims in "Don't Waste Bits!" (arXiv:2604.04722, CVPR 2026) which proposes adaptive KV-cache quantization for on-device LLMs via a learned MLP controller. Since the original code is not yet public (expected June 2026), we re-implement the full method from the paper equations. We identify three methodological subtleties that are critical for reproducing paper results: (1) the evaluation uses unnormalized log-likelihood (`acc`), not the length-normalized `acc_norm` used by default in lm-eval; (2) KV cache quantization must hook the projection layers directly, not the cached output (`DynamicCache` in transformers 5.x silently breaks output hooks); (3) per-token quantization granularity significantly affects quantization noise magnitude. We confirm the FP16 baseline (42.0% vs paper's 41.5%, ±0.5pp) and present our re-implementation's performance on HellaSwag for SmolLM-360M.

---

## Sections

### 1. Introduction
- Motivation: CVPR 2026 paper proposes adaptive KV-cache quantization with strong claims
- Why reproduce: code not yet public, claims important for on-device LLM deployment
- Our contribution: complete re-implementation + methodological insights

### 2. Background
- KV cache quantization for LLM inference
- HellaSwag evaluation protocol
- SmolLM model family

### 3. Method Re-implementation
- Controller architecture (MLP, 4 inputs → 4 outputs, ~33K params)
- Token importance signals (H_t, R_t, V_t, C_t) from paper Eq. 14-17
- Training loss (Eq. 28): α·CE + β·latency + γ·quality
- KV cache quantization: hook k_proj/v_proj vs past_key_values

### 4. Evaluation Setup
- Dataset: HellaSwag validation, first N examples
- Metric: acc (unnormalized log-likelihood) — critical for paper match
- Hardware: CPU (accuracy); RTX 4090 for latency (pending)
- KV quantization variants tested: per-tensor, per-token, symmetric, asymmetric

### 5. Results
- Table: comparison of our results vs paper Table 3
- FP16 baseline: 42.0% (ours) vs 41.5% (paper) ✅
- Static KV-4bit: [PENDING]
- DWB adaptive: [PENDING]
- Latency: [PENDING - needs GPU]

### 6. Methodological Insights

**Insight 1**: Evaluation metric matters critically
- acc_norm (lm-eval default) gives ~54% for SmolLM-360M
- acc (unnorm) gives ~42%, matching paper's 41.5%
- Cross-model evidence: our 360M acc_norm ≈ paper's 1.7B acc → wrong metric

**Insight 2**: KV cache hooks in transformers 5.x
- past_key_values is a DynamicCache object, not raw tuples
- Output hooks on attention modules silently fail to quantize
- Fix: hook k_proj and v_proj Linear submodules directly

**Insight 3**: Quantization granularity
- Per-tensor INT4 uses one scale driven by outlier tokens
- Layer analysis shows max/mean_abs ratios up to 10× in some layers
- Per-token quantization matches KV cache behavior during generation
- Symmetric vs asymmetric affects whether errors cancel

### 7. Discussion
- What we confirmed vs what remains uncertain
- Limitations: no GPU for latency, no original code for comparison
- DWB method details underspecified in paper (exact quantization granularity)

### 8. Conclusion

---

## Key Data for Paper

| Condition | Our Result | Paper Target | Delta | Status |
|-----------|-----------|--------------|-------|--------|
| FP16 (acc) | 42.0% | 41.5% | +0.5pp | ✅ CONFIRMED |
| KV-4bit per-tensor (acc) | [PENDING] | 33.6% | — | Running |
| KV-4bit per-token (acc) | [PENDING] | 33.6% | — | Running |
| DWB adaptive (acc) | [PENDING] | 41.2% | — | Queued |
| FP16 latency | — | 3.50 ms/tok | — | Needs GPU |
| KV-4bit latency | — | 2.93 ms/tok | — | Needs GPU |
| DWB latency | — | 2.41 ms/tok | — | Needs GPU |

## SmolLM2 Comparison Table

| Model | FP16 (acc_norm) | Notes |
|-------|----------------|-------|
| SmolLM-360M | ~42% (acc) | Paper's target model |
| SmolLM2-360M | 45.33% (acc_norm) | Improved architecture |
| SmolLM2-360M | ~49% (acc) | Paper not evaluated on this |
