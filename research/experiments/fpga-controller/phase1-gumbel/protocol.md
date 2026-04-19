# Phase 1 Protocol: Gumbel-Softmax Compound Loss Controller

**Hypothesis:** Replacing quartile-classification with a differentiable compound loss will close
the 4.2pp H3 gap and reach ≥41.2% at ≤5.05 avg_bits.

**Prediction:** Gumbel-softmax + compound loss achieves ≥41.2% HellaSwag at avg_bits ≤5.5
(within 10% of paper's compression target).

## What Changes

| | Our DWB v2 (baseline) | Phase 1 |
|---|---|---|
| Bit selection | argmax → hard class | Gumbel-softmax → soft weighted sum |
| Loss | cross-entropy on quartile label | α·CE_LM + β·avg_bits |
| Gradient | stops at argmax | flows through bit allocation |
| Compression pressure | none | β·avg_bits term |

## Method

### Controller Architecture (unchanged)
- Input: `[C_t, R_t]` — cumulative attention score, residual norm (same as paper)
- Hidden: 3-layer MLP, 64 units, ReLU
- Output: logits over {2, 4, 8, 16} bit classes (4-way)

### Training Loss
```
logits = controller([C_t, R_t])                  # (T, 4)
probs = gumbel_softmax(logits, tau=1.0, hard=False)  # soft weights during train
bits_t = probs @ [2, 4, 8, 16]                   # expected bits per token (T,)
avg_bits = bits_t.mean()

# Quantize KV with expected bits (differentiable via STE)
kv_quant = quantize_ste(kv, bits_t)

# Loss
ce_loss = cross_entropy(lm_logits(kv_quant), targets)
loss = alpha * ce_loss + beta * avg_bits
```

During eval: use hard argmax (same as paper inference).

### Hyperparameters to sweep
- `tau` (Gumbel temperature): {2.0, 1.0, 0.5}
- `beta` (compression weight): {0.05, 0.1, 0.2}
- `alpha` = 1.0 (fixed)

Start with tau=1.0, beta=0.1. Adjust based on avg_bits convergence.

### Training Setup
- Model: SmolLM-360M
- Train samples: 200 (wikitext-2, same as DWB v2)
- Epochs: 10
- LR: 1e-3 (Adam)
- Eval: HellaSwag 200 samples (fast), then 500 if promising

## Success Criteria
- HellaSwag ≥ 41.2% (paper target)
- avg_bits ≤ 5.5
- Both simultaneously

## Expected Failure Modes
- tau too low → collapse to argmax early, gradient vanishes
- beta too high → controller assigns all 2-bit (same collapse as beta sweep)
- STE approximation introduces training/eval distribution shift

## Files
- `code/run_phase1_gumbel.py` — main training + eval script
- `results/` — JSON output per hyperparameter config
- `analysis.md` — what we learned
