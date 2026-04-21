# Rotation implementation plan (before coding)

## Findings from QuaRot (Ashkboos et al., NeurIPS 2024, arxiv 2404.00456)

From the paper's Section 4.1 (method description):

1. **R1** — residual stream rotation of dim `d_model × d_model`. Fused into LayerNorm's weights and into W_q/k/v/up on the left, W_o/down on the right. **Baked offline.** Reduces outliers in the residual stream.

2. **R2** — per-head `d_head × d_head` Hadamard applied to V and to the corresponding slice of W_o. Fused into both. **Baked offline.** Reduces outliers in V, improving V-cache quantization.

3. **R3** — per-head `d_head × d_head` Hadamard applied to Q and K. Because of RoPE, this **cannot be baked offline** — RoPE doesn't commute with arbitrary rotation. QuaRot applies R3 **at inference time (online)**, post-RoPE, as a separate Hadamard block.

4. **R4** — a Hadamard applied inline before MLP down_proj. Also runtime, fused with down_proj output.

## Consequence for our 130nm constraint

The user wants weights baked in, zero runtime rotation hardware. QuaRot tells us this is achievable for R1 and R2 but not R3 or R4:

| Rotation | Offline-only? | Helps K cache? | Helps V cache? |
|----|---|---|---|
| R1 | ✅ | indirectly (via residual stream decorrelation) | indirectly |
| R2 (V + W_o) | ✅ | no | **yes (direct)** |
| R3 (Q, K) | ❌ needs runtime HW | **yes (direct)** | no |
| R4 (MLP) | ❌ needs runtime HW | no | no |

**So: for a 130nm tape-out that bakes everything in, we can apply R1 + R2. K quantization stays at its current −5.7pp loss (unrotated). V quantization should improve.**

Expected impact: modest recovery (likely 1-3pp on Δ vs FP16), not near-lossless.

## Steps for implementation (before coding)

### Step 1: Minimal scope — R2 only (V + W_o)
- Safest first step. V rotation doesn't touch RoPE.
- Per layer: pick random orthogonal `R ∈ R^{d_head × d_head}`.
- `W_v_new[h] = R.T @ W_v[h]` — rotate V output per head.
- `W_o_new[:, h, :] = W_o[:, h, :] @ R` — unrotate input per head.
- Sanity: FP16 rotated == FP16 unrotated to within fp16 roundoff.
- Eval: INT4 on 5 matched subsets, paired Δ vs FP16.

### Step 2: If Step 1 passes and gives ≥2pp recovery, add R1 (residual stream)
- More complex: requires fusing into RMSNorm and all projections (W_q, W_k, W_v, W_up, W_gate) on one side, and W_o, W_down on the other.
- For SmolLM-1.7B with RMSNorm: fuse into RMSNorm `weight` parameter and the subsequent linear weights.
- Sanity check same as Step 1.
- Eval same as Step 1.

### Step 3: Document findings
- If R2 alone gives near-zero benefit: conclude "V rotation helps K-cache quantization negligibly; the L23 outlier is K-side; rotation of K requires runtime Hadamard which is out of scope for 130nm."
- If R2 gives 1-2pp: "V rotation recovers some of the INT4 gap; full gap requires R3 online hardware."
- If R2 + R1 together close the gap significantly: "Partial offline rotation gives XXpp recovery — still below QuaRot's near-lossless but a ~60% gap closure at zero runtime cost."

## Verification protocol

Each step MUST pass these three gates before running INT4 eval:

1. **FP16 sanity gate**: rotated model on single HellaSwag item produces log-likelihoods within 0.01 of unrotated model (per-token fp16 roundoff bound).
2. **Attention smoke gate**: computes a forward pass without NaN or Inf.
3. **Single-subset INT4 gate**: runs 50 samples of static INT4 on rotated model; accuracy must be ≥35% (above random chance 25%). If below, rotation is broken.

Only if all three gates pass do we launch the full 5-subset eval.

## Source code references

- QuaRot github (linked from paper): https://github.com/spcl/QuaRot — can clone and inspect weight-fusion code directly if needed
- Their `fuse_norms.py` and `fuse_qk.py` would show the exact Python.

## What to tell user before coding

- R2 (V + W_o only) is the correct first step. RoPE blocks K rotation offline.
- Expected ceiling on recovery: maybe 1-3pp, not full −5.7pp → 0pp.
- If insufficient: the 130nm design needs a small Hadamard unit (~5K gates) for runtime K rotation. Still simpler than DWB's 4-way router, but not zero.
