# Research Findings — Don't Waste Bits! Verification

**Last updated**: 2026-04-19  
**Phase**: Inner Loop — Baselines Running

---

## Current Understanding

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
- **Accuracy evaluation**: `eval_hellaswag.py` uses length-normalized log-likelihood (standard zero-shot HellaSwag protocol).
- **Model variant**: Checking SmolLM-360M vs SmolLM2-360M — FP16 baseline probe running now.
- **4-bit baseline**: Paper applies 4-bit to KV cache only; must simulate KV quantization, not weight quantization.

---

## Open Questions

1. Does SmolLM-360M or SmolLM2-360M reproduce the paper's FP16 = 41.50%? **(probing now)**
2. Does 4-bit weight quantization approximate 4-bit KV quantization in accuracy impact?
3. Is the 17.75% latency gain purely from bandwidth reduction, or also fewer FP16 ops?

---

## Experiment Trajectory

| Run | Condition | Metric | Value | Paper Target | Delta | Status |
|-----|-----------|--------|-------|-------------|-------|--------|
| 00 | Arithmetic | H1,H2,H3 | self-consistent | — | — | DONE |
| 01 | FP16 baseline | HellaSwag acc% | — | 41.50% | — | RUNNING (300 samp) |
| 02 | Static 4-bit | HellaSwag acc% | — | 33.60% | — | QUEUED |
| 03 | DWB (ours) | HellaSwag acc% | — | 41.20% | — | QUEUED |
| 04 | FP16 latency | ms/token | — | 3.50 | — | AWAITING BREV |
| 05 | Static 4-bit latency | ms/token | — | 2.93 | — | AWAITING BREV |
| 06 | DWB latency | ms/token | — | 2.41 | — | AWAITING BREV |
