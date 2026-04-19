# Experiment H3: FP16 Parity Verification

**Hypothesis**: The DWB method achieves accuracy within 0.30 points of FP16 on
SmolLM-360M / HellaSwag.

**Claimed values (Table 3)**:
- FP16 accuracy: 41.50%
- DWB accuracy: 41.20%
- Gap: 41.50 - 41.20 = 0.30 points

**Protocol** (locked before running):
1. Same setup as H2 experiment
2. Focus metric: absolute gap between FP16 and DWB accuracy
3. Gap ≤ 1.0 point would be a strong confirmation; gap ≤ 0.50 would be excellent

**Prediction**: Gap will be 0.1–1.0 points. The exact 0.30 is unlikely to reproduce
exactly (stochastic evaluation), but near-FP16 accuracy should hold.

**Hardware**: NVIDIA RTX 4090 (24GB) on NVIDIA Brev
**Status**: AWAITING GPU
