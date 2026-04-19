# Don't Waste Bits! + TurboQuant — Integration Research

**Branch**: `turboquant-integration`  
**Base paper**: [arXiv:2604.04722](https://arxiv.org/abs/2604.04722) · *Don't Waste Bits! Adaptive KV-Cache Quantization*  
**TurboQuant**: [Google Research, ICLR 2026](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

> See the `main` branch for independent verification of the base paper's claims.

---

## The Idea

**Don't Waste Bits! (DWB)** learns which tokens are important and assigns them higher KV-cache precision. Its lowest tier (2-bit) currently uses naive scalar quantization.

**TurboQuant** compresses all KV tokens uniformly using vector quantization (PolarQuant + QJL correction) — achieving 3-bit keys / 2-bit values with no accuracy loss. But it treats all tokens equally.

**This branch combines them**: DWB ranks token importance, then routes low-importance tokens through TurboQuant's principled vector quantization instead of naive 2-bit scalar quantization.

```
Token t →  DWB controller signals [H_t, R_t, V_t, C_t]
        →  MLP → bit-width ∈ {2, 4, 8, 16}

Routing:
  16-bit  →  FP16 storage              (critical tokens)
   8-bit  →  INT8 scalar quantization
   4-bit  →  INT4 scalar quantization
   2-bit  →  TurboQuant vector quant   ← novel contribution
```

Expected: DWB-TurboQuant recovers accuracy lost by naive 2-bit scalar while maintaining
TurboQuant's compression ratio, especially on reasoning-heavy benchmarks.

---

## Baseline Numbers (from main branch verification)

| Condition | SmolLM-360M | SmolLM-135M | Source |
|-----------|-------------|-------------|--------|
| FP16 | 42.6% | 40.0% | Verified (500/100 samp) |
| Standard INT4 KV | 41.2–44.5% | 39.0% | Verified (lossless) |
| int4_int3range (8-level) | 33.0% | 32.0% | Verified (matches paper) |
| DWB adaptive | 38.0–40.0% | — | Verified (200/100 samp) |
| Scalar 2-bit | 25.0% | — | Verified (catastrophic) |

**Key finding from main**: Standard INT4 is near-lossless (zero-mean error cancellation).
DWB 2-bit assignment (57% of tokens) uses naive scalar 2-bit → 25% accuracy. TurboQuant's
vector quantization should recover 5–15pp for these tokens.

---

## Hypotheses

### TQ-H1: Accuracy Recovery
**Claim**: DWB-TurboQuant achieves higher accuracy than DWB-scalar for 2-bit tokens.  
**Prediction**: +3–10pp over DWB-scalar on HellaSwag (from 38% toward 42%).  
**Mechanism**: PolarQuant rotation decorrelates KV dimensions, reducing worst-case error.

### TQ-H2: Compression Parity
**Claim**: DWB-TurboQuant matches TurboQuant's compression ratio.  
**Prediction**: avg_bits ≈ 4.5–5.0 (DWB routing) with <0.5pp accuracy difference from DWB-scalar.

### TQ-H3: Benchmark Robustness
**Claim**: Gains are larger on reasoning tasks than factual recall.  
**Status**: Untested (requires ARC/BoolQ in addition to HellaSwag).

---

## Status

- [x] Baseline numbers established (from main branch)
- [x] Protocol committed
- [x] `turboquant_pipeline.py` scaffold created
- [x] `turboquant_impl.py` — self-contained PolarQuant implementation (no external deps)
- [ ] TQ-H1 experiment run
- [ ] TQ-H2 experiment run
- [ ] TQ-H3 (additional benchmarks)

---

## Running

```bash
# Run DWB-TurboQuant vs DWB-scalar comparison (100 samples)
python research/src/run_turboquant_eval.py --limit 100

# Or directly:
python research/src/turboquant_pipeline.py
```
