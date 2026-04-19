# Literature Survey

## Primary Paper

### Don't Waste Bits! (arXiv:2604.04722)
- **Authors**: Haeri Boroujeni et al., Clemson University, 2026
- **Method**: Adaptive KV-cache quantization via learned MLP controller
- **Key idea**: Token importance signals (entropy, rarity, attention variance, confidence) → MLP → bit-width ∈ {2,4,8,16}
- **Training**: Combined loss = cross-entropy + expected latency + quality penalty
- **Results**: SmolLM-360M HellaSwag — 17.75% latency reduction, +7.60 accuracy vs static 4-bit
- **Code**: https://github.com/SayedPedramHaeri/Dont-Waste-Bits
- **Relevance**: Primary paper under verification

---

## KV Cache Quantization

### TurboQuant (Google Research, ICLR 2026)
- **Authors**: Google Research
- **Method**: Two-stage vector quantization: PolarQuant (random rotation + scalar quantization) + QJL residual correction (1-bit error correction)
- **Targets**: 3-bit keys, 2-bit values in KV cache
- **Compression**: ~6× KV cache reduction, up to 8× speedup on H100
- **Key property**: Training-free, data-oblivious
- **Code**: https://github.com/tonbistudio/turboquant-pytorch (pure PyTorch, no vLLM dependency)
- **Relevance**: Integration target for Track 2 — DWB importance signals route tokens needing compression to TurboQuant

### KIVI (arXiv:2402.02750)
- **Method**: Asymmetric 2-bit KV cache quantization, tuning-free
- **Key**: Group quantization with residual full-precision storage for recent tokens
- **Relevance**: Baseline for aggressive KV compression comparison

### KVQuant (NeurIPS 2024)
- **Method**: Salient-channel and per-vector quantization for KV cache
- **Supports**: Up to 10M context length
- **Relevance**: Related work on importance-aware KV quantization

### ZipCache (NeurIPS 2024)
- **Method**: Salient token identification for accurate KV cache quantization
- **Relevance**: Most similar prior work to DWB — also uses token importance

### QAQ (ICCV 2025)
- **Method**: Quality adaptive quantization for LLM KV cache
- **Relevance**: Cited as [5] in DWB paper — adaptive quality-aware baseline

### Ada-KV (arXiv:2407.11550)
- **Method**: Adaptive budget allocation for KV cache eviction
- **Relevance**: Adaptive approach but focuses on eviction vs quantization

---

## Related Models

### SmolLM (HuggingFace)
- SmolLM-135M / 360M / 1.7B: small on-device LLMs from HuggingFace
- Model IDs: `HuggingFaceTB/SmolLM-135M`, `HuggingFaceTB/SmolLM-360M`, `HuggingFaceTB/SmolLM-1.7B`

---

## Evaluation

### lm-evaluation-harness (EleutherAI)
- Standard framework for zero-shot LLM evaluation
- HellaSwag, ARC-Challenge, OpenBookQA all supported
- **Note**: Score normalization can vary by version — potential source of 1-3 point discrepancy vs paper
