# Phase 2 Protocol: Richer Controller Features

**Hypothesis:** Adding per-head attention entropy and layer depth to the controller input will
improve bit allocation precision beyond Phase 1, since the paper's [C_t, R_t] features are
informationally incomplete — our mechanistic analysis shows head entropy and layer depth
are strong predictors of quantization sensitivity.

**Prediction:** Phase 2 controller achieves ≥ Phase 1 accuracy at equal or fewer avg_bits.

## New Features

### H_t — Per-head attention entropy
```python
# For each head h at position t:
attn_weights = softmax(Q_h K_h^T / sqrt(d))   # (T, T)
entropy_h = -sum(p * log(p + 1e-9))            # scalar per head
H_t = mean(entropy_h over all heads)           # scalar per token
```
High entropy = uniformly attending = less important = safe to compress more.

### L — Layer depth (normalized)
```python
L = layer_idx / num_layers   # scalar in [0, 1]
```
From our 1.7B mechanistic analysis: later layers have higher KV variance (eff_residual 12.4%
vs 8.1%). Layer depth is a proxy for quantization sensitivity even within the same model.

## Controller Input
```
[C_t, R_t, H_t, L]   # 4-dim vs paper's 2-dim
```

## Everything Else
Same as best Phase 1 config (tau, beta, alpha, training setup).

## Files
- `code/run_phase2_features.py`
- `results/`
- `analysis.md`
