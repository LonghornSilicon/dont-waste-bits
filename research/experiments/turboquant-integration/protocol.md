# TurboQuant Integration — Research Protocol

**Branch**: turboquant-integration  
**Novel hypothesis**: Using Don't Waste Bits! controller as a precursor ranking stage
for TurboQuant produces better accuracy/compression tradeoff than uniform TurboQuant alone.

---

## Background

### Don't Waste Bits! (DWB)
- Assigns per-token bit-widths {2,4,8,16} via learned MLP controller
- Trained on token importance signals: entropy, rarity, attention variance, confidence
- Achieves near-FP16 accuracy at reduced latency

### TurboQuant (Google, ICLR 2026)
- Vector quantization for KV cache: PolarQuant + QJL residual correction
- Achieves 3-bit keys / 2-bit values uniformly across all tokens
- Training-free, data-oblivious — no per-token importance distinction
- 6× compression, 8× speedup on H100

---

## The Gap

TurboQuant treats all tokens equally. But some tokens (named entities, rare words,
high-attention positions) carry disproportionate semantic weight. Uniform compression
risks over-quantizing these critical tokens.

DWB knows which tokens are important. But its 2-bit scalar quantization for low-importance
tokens is naive compared to TurboQuant's principled vector quantization.

---

## Proposed Pipeline

```
Token t arrives at step d (decoding)
  ↓
Compute signals: [H_t, R_t, V_t, C_t]
  ↓
DWB controller → predicted bit-width b̂_t ∈ {2, 4, 8, 16}
  ↓
Routing:
  b̂_t = 16  →  FP16 storage (no compression)
  b̂_t = 8   →  INT8 scalar quantization
  b̂_t = 4   →  INT4 scalar quantization
  b̂_t = 2   →  TurboQuant (PolarQuant + QJL, ~2.5 bits effective)
  ↓
Store quantized (k_t, v_t) in adaptive KV cache
```

Key change: 2-bit assignments go through TurboQuant's vector quantization
instead of naive scalar 2-bit — better noise properties, error correction via QJL.

---

## Hypotheses

### TQ-H1: Accuracy Recovery
DWB+TurboQuant achieves higher accuracy than DWB alone on low-importance tokens,
because TurboQuant's QJL correction reduces quantization error.

**Prediction**: +0.5–2.0 accuracy points vs DWB-alone on HellaSwag (SmolLM-360M)

### TQ-H2: Compression Maintained
DWB+TurboQuant maintains similar or better compression ratio vs DWB alone,
since ~30-50% of tokens are assigned 2-bit and benefit from TurboQuant's
better compression.

**Prediction**: Average KV bits ≤ DWB alone, latency similar or better

### TQ-H3: Reasoning Benefit
The benefit is larger on complex reasoning benchmarks (ARC-Challenge) than
on easier tasks, because important tokens matter more in multi-step reasoning.

**Prediction**: Larger improvement on ARC-C than HellaSwag

---

## Experiments to Run

1. **Baseline**: DWB alone (from verification branch results)
2. **TurboQuant alone**: Uniform 3/2-bit vector quantization across all tokens
3. **DWB + TurboQuant**: Adaptive routing with TurboQuant at the 2-bit tier
4. **Ablation**: DWB + TurboQuant at all tiers (replace scalar with TQ everywhere)

---

## Implementation Steps

1. Clone TurboQuant: `git clone https://github.com/0xSero/turboquant`
2. Integrate TurboQuantKV into `research/src/turboquant_pipeline.py`
3. Load pre-trained DWB controller from verification experiments
4. Run combined pipeline on HellaSwag, ARC-C, OBQA
5. Compare accuracy and latency vs all baselines

**Status**: AWAITING GPU + TurboQuant integration
