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

---

## Hypotheses

| ID | Claim | Prediction |
|----|-------|-----------|
| TQ-H1 | DWB+TurboQuant > DWB alone on accuracy | +0.5–2.0 pp on HellaSwag |
| TQ-H2 | Compression maintained or improved vs DWB alone | Avg bits ≤ DWB |
| TQ-H3 | Larger benefit on ARC-Challenge than HellaSwag | More reasoning benefit |

---

## Why This Should Work

TurboQuant's QJL residual correction eliminates quantization bias that naive 2-bit scalar quantization introduces. For tokens DWB identifies as low-importance (where some error is tolerable), TurboQuant provides better-quality compression at the same bit budget.

**Key risk**: TurboQuant's rotation matrix is calibrated globally. Applying it to a subset of tokens may require per-token re-calibration. The `turboquant-pytorch` fork (which replaced QJL with asymmetric K/V allocation) may integrate more cleanly.

---

## TurboQuant Implementation

Using [`tonbistudio/turboquant-pytorch`](https://github.com/tonbistudio/turboquant-pytorch) — pure PyTorch, no vLLM dependency:
- Asymmetric K/V allocation (more bits for keys, fewer for values)
- Layer-adaptive precision (protects first/last transformer layers)
- Residual windowing: recent tokens stay in FP16

Pipeline code: `research/src/turboquant_pipeline.py`

---

## Experiments Plan

| Run | Description | Primary Metric |
|-----|-------------|---------------|
| TQ-base | TurboQuant alone (uniform 3/2-bit) | HellaSwag acc%, latency |
| DWB-base | DWB alone (from main branch) | HellaSwag acc%, latency |
| DWB+TQ | DWB controller + TQ at 2-bit tier | HellaSwag acc%, latency |
| DWB+TQ-ablation | TQ at all tiers | HellaSwag acc%, latency |

All on SmolLM-360M. Expand to ARC-C and OBQA after primary results.

---

## Status

- [x] Pipeline architecture designed (`turboquant_pipeline.py`)
- [x] Research protocols locked
- [ ] TurboQuant PyTorch integration (attention hooks)
- [ ] DWB baseline numbers from `main` branch
- [ ] TQ-H1/H2/H3 experiments
- [ ] Paper
