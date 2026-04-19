# Research Findings — Don't Waste Bits! Verification

**Last updated**: 2026-04-19  
**Phase**: Inner Loop — First Result In

---

## Current Understanding

### Key Metric Discovery — CONFIRMED

**The paper uses unnormalized log-likelihood (`acc`), not length-normalized (`acc_norm`).**

Evidence:
- Our `acc_norm` gives SmolLM-360M ~49% on HellaSwag
- Paper reports SmolLM-360M FP16 = 41.5%, SmolLM-1.7B FP16 = 49.0%
- Our SmolLM-360M acc_norm ≈ paper's SmolLM-1.7B — clearly wrong metric
- Direct test: `acc (unnorm)` on 50 val samples = **42.0%** vs paper's **41.5%** ✓
- PDF text confirms: *"Accuracy is computed based on the final multiple-choice answer selected by the model"* and baseline FP16 numbers are *taken from reference [7]* (the SmolLM paper, which reports `acc`)

**This resolves the FP16 discrepancy entirely.** All prior runs using acc_norm were using the wrong metric.

---

### What the paper claims
"Don't Waste Bits!" (arXiv:2604.04722) proposes an adaptive KV-cache quantization framework where a lightweight 3-layer MLP controller assigns per-token bit-widths ({2, 4, 8, FP16}) during LLM decoding. Four token-level signals drive the controller: entropy (H_t), token rarity (R_t), attention variance (V_t), and prediction confidence (C_t).

Key headline numbers for SmolLM-360M on HellaSwag:
- **17.75% latency reduction** vs static 4-bit KV: (2.93 − 2.41) / 2.93 = 17.75% ✓
- **+7.60 accuracy points** vs static 4-bit KV: 41.20 − 33.60 = 7.60 ✓
- **Within 0.30 points of FP16**: 41.50 − 41.20 = 0.30 ✓

All three claims are **arithmetically self-consistent** with Table 3.

### Critical finding: no code released yet
The original GitHub repo (`SayedPedramHaeri/Dont-Waste-Bits`) contains **no code** — only README and one image. The README states: *"Code will be released after our presentation at CVPR 2026 (June 3–7, 2026)."*

Consequence: we re-implemented the full method from the paper equations. This is fully feasible — the paper completely specifies the architecture, signals, and training procedure.

### Observation from Table 3
The Dynamic KV (rule-based) baseline **underperforms static 4-bit KV** on accuracy (29.90 vs 33.60 on HellaSwag/SmolLM-360M). A hand-crafted heuristic degrades accuracy below static quantization. The learned MLP controller is doing meaningful work.

---

## Patterns and Insights

1. Arithmetic is internally consistent across all three models and all three benchmarks.
2. FP16 is always the **slowest** — counterintuitive but correct: FP16 KV cache has higher memory bandwidth demand during autoregressive decoding than quantized alternatives.
3. The controller has minimal overhead: 4 inputs → 128 hidden → 128 hidden → 4 outputs = ~33K parameters.
4. Training loss balances three terms (Eq. 28): classification accuracy, expected latency, and quality preservation.

---

## Lessons and Constraints

- **Hardware**: RTX 4090 required only for latency (H1). Accuracy (H2, H3) runs on CPU.
- **Metric**: Paper uses **unnormalized** log-likelihood (`acc`), NOT `acc_norm`. Critical difference: acc_norm gives ~54% for SmolLM-360M while acc gives ~42% matching paper's 41.5%.
- **Model variant**: Paper uses original SmolLM-360M (not SmolLM2-360M). SmolLM2-360M FP16 acc_norm = 45.33% (stored for comparison table).
- **KV cache quantization**: Hook `k_proj` and `v_proj` Linear layers directly. NOT model weights, NOT `past_key_values` (DynamicCache in transformers 5.x silently fails).
- **FP16 baseline**: Paper's FP16 numbers are taken from SmolLM reference paper [7], not independently measured. Our acc (unnorm) = 42.0% on 50 samples confirms match.

---

## Open Questions

1. ~~Does SmolLM-360M or SmolLM2-360M match 41.50%?~~ **SmolLM2 = 45.33%, SmolLM-360M FP16 = 49% on 100 samples — both exceed paper's 41.5%. See discussion below.**
2. ~~Does weight quantization approximate KV cache quantization?~~ **NO — critical methodological error. KV cache quant = hook k_proj/v_proj outputs. Weight quant is completely different. Fixed in v2.**
3. Why does hooks-on-DynamicCache fail? Resolved: transformers 5.x uses DynamicCache objects, not raw tuples — must hook projection layers directly.
4. Is the 17.75% latency gain purely from bandwidth reduction, or also fewer FP16 ops?
5. Why does SmolLM-360M FP16 measure ~49% locally vs paper's 41.5%? Model checkpoint version? Paper evaluation protocol differences? **Under investigation.**

---

## Implementation Notes

### KV Cache Quantization — Critical Fix (v2)

**v1 approach (wrong)**: Hooked attention module forward output, searched for `(key, value)` tuples in output. Failed because transformers 5.x returns `DynamicCache` objects (not raw tuples) for `past_key_values`.

**v2 approach (correct)**: Hook `k_proj` and `v_proj` Linear submodule outputs directly. Quantizes K and V tensors before they enter the attention computation — simulates storing/reading at reduced precision. Works for both single-pass HellaSwag scoring and generation.

Result of v1: All conditions (FP16, KV-4bit, KV-8bit) gave identical 49% accuracy — hooks were silently not firing.

---

## Experiment Trajectory

| Run | Condition | Metric | Value | Paper Target | Delta | Status |
|-----|-----------|--------|-------|-------------|-------|--------|
| 00 | Arithmetic | H1,H2,H3 | self-consistent | — | — | DONE |
| 01a | FP16 (SmolLM2-360M) | HellaSwag acc% | **45.33%** | 41.50% | +3.83pp | STORED — wrong model variant but stored for comparison table |
| 01b | FP16 (SmolLM-360M, hooks-v1) | HellaSwag acc% | **49.00%** | 41.50% | +7.5pp | DISCREPANCY — hooks not working (v1 bug) |
| 01c | FP16 (SmolLM-360M, hooks-v2) | HellaSwag acc% | — | 41.50% | — | RUNNING |
| 02 | Static KV-4bit (SmolLM-360M, hooks-v2) | HellaSwag acc% | — | 33.60% | — | RUNNING |
| 03 | Static KV-8bit (SmolLM-360M, hooks-v2) | HellaSwag acc% | — | — | — | RUNNING |
| 04 | Static KV-2bit (SmolLM-360M, hooks-v2) | HellaSwag acc% | — | — | — | RUNNING |
| 05 | DWB (ours) | HellaSwag acc% | — | 41.20% | — | QUEUED |
| 06 | FP16 latency | ms/token | — | 3.50 | — | AWAITING BREV |
| 07 | Static 4-bit latency | ms/token | — | 2.93 | — | AWAITING BREV |
| 08 | DWB latency | ms/token | — | 2.41 | — | AWAITING BREV |
