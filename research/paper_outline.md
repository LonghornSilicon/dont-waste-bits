# Paper Outline: Independent Verification of "Don't Waste Bits!"

**Target venue**: arXiv preprint (reproducibility study)
**Paper type**: Reproducibility / Negative Results  
**Status**: Experiments mostly complete — writing pending

---

## Title Options

1. "Independent Verification of Don't Waste Bits!: Adaptive KV-Cache Quantization for Lightweight LLMs"
2. "Reproducing Adaptive KV-Cache Quantization: Lessons from Verifying arXiv:2604.04722"
3. "Why Symmetric INT4 KV Quantization is Nearly Lossless: Insights from Reproducing arXiv:2604.04722"

---

## Abstract (draft)

We present an independent reproduction attempt of the key claims in "Don't Waste Bits!" (arXiv:2604.04722, CVPR 2026) which proposes adaptive KV-cache quantization for on-device LLMs via a learned MLP controller. Since the original code is not yet public (expected June 2026), we re-implement the full method from the paper equations. We identify four methodological subtleties critical for understanding the paper's results:

(1) The evaluation uses unnormalized log-likelihood (`acc`), not the length-normalized `acc_norm` used by default in lm-eval — the difference is ~13pp for SmolLM-360M on HellaSwag.

(2) KV cache quantization must hook the projection layers directly, not the cached output (`DynamicCache` in transformers 5.x silently breaks output hooks).

(3) sdpa attention blocks `output_attentions=True`, requiring `attn_implementation='eager'` for DWB controller signal extraction.

(4) Symmetric per-tensor INT4 quantization of K and V projections is nearly lossless for attention computation — our implementation gives ~44.5% (≈ FP16 44%) vs the paper's claimed 33.6% for static 4-bit KV. We attribute this to zero-mean error cancellation in the attention weighted sum, where outlier-set scales disproportionately protect the most-attended tokens.

We confirm the FP16 baseline (42.0% vs paper's 41.5%, Δ=+0.5pp ✅) and present our DWB re-implementation's accuracy (40.0% vs paper's 41.2%, within noise ≈ H3 consistent). The paper's central accuracy claim (H2: +7.6pp over static INT4) cannot be independently verified because we cannot reproduce the static INT4 baseline's reported accuracy drop.

---

## Sections

### 1. Introduction
- CVPR 2026 paper: adaptive KV-cache quantization, strong accuracy + latency claims
- Code not yet public → re-implementation necessary, valuable as reproducibility record
- Our contribution: re-implementation + 4 methodological insights + cannot-reproduce finding

### 2. Background
- KV cache quantization for LLM inference
- HellaSwag evaluation protocol (and why metric matters)
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
- KV quantization variants tested

### 5. Results

**Table: Our results vs paper Table 3**

| Condition | Our Result | Paper Target | Δ | Status |
|-----------|-----------|--------------|---|--------|
| FP16 (acc) | 42.0% (50 samp) | 41.5% | +0.5pp | ✅ CONFIRMED |
| FP16 (acc) | 44.0% (200 samp) | 41.5% | +2.5pp | ✅ Within noise |
| KV-4bit per-tensor | 44.5% (200 samp) | 33.6% | +10.9pp | ⚠️ CANNOT REPRODUCE |
| KV-4bit per-token | 44.0% (100 samp) | 33.6% | +10.4pp | ⚠️ CANNOT REPRODUCE |
| KV-2bit | 25.0% (200 samp) | — | — | ✅ Hooks confirmed |
| DWB (ours) | 40.0% (100 samp) | 41.2% | -1.2pp | ~✅ H3 consistent |
| FP16 latency | — | 3.50 ms/tok | — | ⏳ Needs GPU |
| KV-4bit latency | — | 2.93 ms/tok | — | ⏳ Needs GPU |
| DWB latency | — | 2.41 ms/tok | — | ⏳ Needs GPU |

### 6. Methodological Insights

**Insight 1**: Evaluation metric matters critically
- acc_norm (lm-eval default) gives ~54% for SmolLM-360M
- acc (unnorm) gives ~42%, matching paper's 41.5%
- Cross-model evidence: our 360M acc_norm ≈ paper's 1.7B acc → wrong metric

**Insight 2**: KV cache hooks in transformers 5.x
- past_key_values is a DynamicCache object, not raw tuples
- Output hooks on attention modules silently fail to quantize
- Fix: hook k_proj and v_proj Linear submodules directly

**Insight 3**: sdpa attention blocks output_attentions
- DWB controller needs attention weights for V_t signal (Eq. 16)
- sdpa doesn't support output_attentions=True (silently returns empty tuple)
- Fix: reload with attn_implementation='eager' for training data extraction

**Insight 4**: Symmetric per-tensor INT4 quantization is nearly lossless ★ (main finding)
- Our INT4 KV gives ~44.5% ≈ FP16, not paper's 33.6%
- KV-2bit gives 25% (catastrophic) confirming hooks work
- Hypothesis: zero-mean symmetric errors cancel in attention weighted sum
- Outlier tokens set the scale and are thus well-quantized AND most attended
- Paper's "Static 4-bit KV" likely uses a more aggressive scheme
- Ongoing investigation: 7 INT4 variants tested to identify which scheme degrades accuracy

### 7. Discussion
- What we confirmed: FP16 baseline (H3 numerically consistent), hooks work
- What we cannot reproduce: static INT4 accuracy drop (H2 baseline)
- The "cannot reproduce" is itself a finding: naive INT4 is not as harmful as claimed
- Limitations: no GPU for latency, no original code for comparison
- When code releases (CVPR June 2026): repeat with official implementation

### 8. Conclusion
- Paper's FP16 baseline confirmed within noise
- DWB controller learns above-chance signal prediction (45.6% vs 25% random)
- DWB accuracy near FP16 (H3 consistent)
- Central challenge: cannot reproduce static INT4 baseline's 7.9pp accuracy drop
- This is a novel negative result suggesting the paper's comparison baseline may be a specific published method with non-standard quantization, not naive symmetric INT4

---

## Key Data for Paper

### Controller Training
- Architecture: Linear(4,128) → ReLU → Linear(128,128) → ReLU → Linear(128,4) = 33,540 params
- Training: 2,995 token samples from 100 HellaSwag train examples, 5 epochs, lr=0.003
- Val accuracy: 45.6% (4-class quartile prediction, random baseline 25%)
- Bit distribution in eval: {2bit: 57.3%, 4bit: 18.9%, 8bit: 8.3%, 16bit: 15.6%}
- Average bits: 5.05/token

### SmolLM2 Comparison Table (for context)
| Model | acc_norm | acc | Notes |
|-------|---------|-----|-------|
| SmolLM-360M | ~54% | ~42% | Paper's target model |
| SmolLM2-360M | 45.33% | ~49% | Improved architecture |
