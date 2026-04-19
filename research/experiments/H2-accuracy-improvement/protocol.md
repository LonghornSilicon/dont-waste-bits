# Experiment H2: Accuracy Improvement Verification

**Hypothesis**: The DWB method achieves +7.60 accuracy points over static 4-bit KV
on SmolLM-360M / HellaSwag (zero-shot).

**Claimed values (Table 3)**:
- Static 4-bit KV accuracy: 33.60%
- DWB method accuracy: 41.20%
- Improvement: 41.20 - 33.60 = 7.60 points

**Protocol** (locked before running):
1. Use lm-evaluation-harness (lm-eval) in zero-shot mode
2. Evaluate SmolLM-360M (HuggingFaceTB/SmolLM-360M) on HellaSwag
3. Three conditions: FP16 (baseline), static 4-bit KV, DWB adaptive
4. Report accuracy % and compare to Table 3

**Prediction**: Accuracy within ±1.0 point of paper values.
FP16 baseline ~41.5%, static 4-bit ~33-35%, DWB ~40-42%.

**Key concern**: lm-eval version sensitivity. HellaSwag scores can vary by 1-3 points
depending on normalization (by token length vs raw). Must use same normalization as paper.
The paper does not specify lm-eval version — this is a potential source of discrepancy.

**Sanity checks**:
- SmolLM-360M FP16 on HellaSwag should be ~40-45% (known range from HuggingFace model card)
- Static 4-bit should degrade ~5-10 points (expected from quantization noise)
- DWB should recover most of that gap

**Hardware**: NVIDIA RTX 4090 (24GB) on NVIDIA Brev
**Status**: AWAITING GPU
