# Experiment H1: Latency Reduction Verification

**Hypothesis**: The Don't Waste Bits! method achieves 17.75% decoding latency reduction
vs static 4-bit KV quantization on SmolLM-360M / HellaSwag.

**Claimed values (Table 3)**:
- Static 4-bit KV latency: 2.93 s/token
- DWB method latency: 2.41 s/token
- Reduction: (2.93 - 2.41) / 2.93 * 100 = 17.75%

**Protocol** (locked before running):
1. Load SmolLM-360M in FP16 on RTX 4090
2. Measure baseline latency (FP16, no quantization): target 3.50 s/token
3. Apply static 4-bit KV quantization: target 2.93 s/token
4. Apply DWB adaptive method (load controller): target 2.41 s/token
5. Measure latency as ms/token averaged over HellaSwag validation set (10,042 samples)
6. Report: absolute latency values, percentage reduction, whether within ±5% of paper

**Prediction**: Latency reduction will be 15–20% (bracketing the reported 17.75%).
Small deviations expected due to different hardware clock states, but the relative ordering
(DWB < static 4-bit < FP16 in latency) should hold.

**Sanity checks**:
- FP16 should be slowest (memory bandwidth bound)
- Static 4-bit should be ~15-20% faster than FP16
- DWB should be faster than static 4-bit (adaptive allocation skews toward lower bits)

**Hardware**: NVIDIA RTX 4090 (24GB) on NVIDIA Brev
**Status**: AWAITING GPU
