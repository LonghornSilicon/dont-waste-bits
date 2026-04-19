# Research Findings — DWB + TurboQuant Integration

**Branch**: turboquant-integration  
**Last updated**: 2026-04-19  
**Phase**: Bootstrap

---

## The Core Idea

TurboQuant (ICLR 2026) achieves state-of-the-art KV cache compression via vector
quantization — but it treats all tokens uniformly. Don't Waste Bits! (DWB) has a
learned controller that knows which tokens are important — but its 2-bit tier uses
naive scalar quantization.

**Combining them**: DWB identifies importance; TurboQuant compresses the unimportant.

```
DWB controller output:
  bit_width = 16 → keep in FP16 (critical tokens: named entities, rare words)
  bit_width = 8  → INT8 scalar
  bit_width = 4  → INT4 scalar
  bit_width = 2  → TurboQuant (PolarQuant + QJL) ← novel routing
```

---

## Current Understanding

### Why this should work (theory)

1. **TurboQuant vs scalar 2-bit**: QJL residual correction eliminates scalar quantization
   bias. For low-importance tokens that get 2-bit assignment from DWB, TurboQuant's
   error-corrected vector quantization should produce less corruption than naive 2-bit.

2. **Selective protection**: FP16 and 8-bit tiers protect tokens the DWB controller
   identifies as high-importance. TurboQuant gets only the tokens DWB already judged
   as low-importance — so any residual TQ error lands on tokens that matter less.

3. **Compression**: DWB's baseline already achieves a mixed-precision KV cache. Adding
   TurboQuant at the 2-bit tier doesn't increase memory vs DWB — it improves quality
   at the same compression level.

### Risks / potential failure modes

1. **TurboQuant calibration**: TurboQuant assumes all tokens see its rotation matrix.
   Running it only on a subset (low-importance tokens) may violate this assumption if
   the rotation is calibrated globally. Need to check TurboQuant code.

2. **Latency overhead**: TurboQuant adds PolarQuant rotation at store time. If this
   overhead exceeds the latency savings from better compression, the pipeline loses.
   Must measure carefully.

3. **DWB controller distribution**: If DWB assigns very few 2-bit tokens (most get 4+),
   TurboQuant has little scope to help. Need to check the typical bit-width distribution.

---

## Open Questions

1. What fraction of tokens does DWB assign to 2-bit vs higher?
2. Does TurboQuant support partial cache quantization (not all tokens)?
3. How does PolarQuant's rotation interact with per-token selective application?
4. Is the TurboQuant repo production-ready or research code?

---

## Planned Experiments

| Run | Description | Metric | Status |
|-----|-------------|--------|--------|
| TQ-base | TurboQuant alone (uniform) | HellaSwag acc, latency | PENDING |
| DWB-base | DWB alone (from main) | HellaSwag acc, latency | PENDING |
| DWB+TQ | DWB controller + TQ at 2-bit tier | HellaSwag acc, latency | PENDING |
| DWB+TQ-ablation | TQ at all tiers | HellaSwag acc, latency | PENDING |
