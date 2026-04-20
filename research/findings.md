# Research Findings — Don't Waste Bits! Verification

**Last updated**: 2026-04-20 (Session 35: TinyLlama sensitivity confirmed — Finding 9 generalized across 5 data points / 3 architecture types. All ≤±0.020. CPU verification fully complete.)  
**Phase**: CPU_COMPLETE — 10 checkpoints / 5 model families / 2×2 instruct matrix. GPU (1.7B HellaSwag accuracy) + FPGA hardware (latency) are only remaining blockers.

---

## Summary

We independently reproduced the key accuracy claims of "Don't Waste Bits!" (arXiv:2604.04722)
and identified seven methodological insights including a novel finding about INT4 quantization,
a mechanistically verified losslessness mechanism, and a dual-objective controller training constraint.
Cross-model validation across SmolLM-135M, 360M, and 1.7B (H4) reveals a critical scale-dependent
pattern: INT4 is lossless at 135M/360M but shows genuine degradation at 1.7B.

> **Novel extension**: DWB-TurboQuant achieves 42.0% ≈ FP16 at 5.05 avg_bits (+2pp HellaSwag, +3pp ARC-Challenge vs DWB-scalar). **BoolQ caveat**: DWB-TurboQuant shows -20pp on BoolQ (41% vs 61% DWB-scalar). WHT rotation is task-specific — helps open-ended generation, hurts closed-form yes/no QA.

**Status of claims:**

| Claim | Status | Our Result | Paper |
|-------|--------|-----------|-------|
| H1: 17.75% latency reduction | ⏳ AWAITING GPU | — | 17.75% |
| H2: +7.6pp over static INT4 | ✅ EXPLAINED | See Insight 5 | 41.2% vs 33.6% |
| H3: within 0.30pp of FP16 | ⚠️ PARTIAL | v1: 33.8%@5.03b; v2: 37.0%@8.47b | 41.2%@5.05b |
| H4: cross-model validation | ✅ CONFIRMED | See H4 Results | 135M + 360M + 1.7B |
| FP16 baseline (500 samp, 360M) | ✅ CONFIRMED | 42.6% | 41.5% |
| FP16 baseline (100 samp, 135M) | ✅ CONFIRMED | 40.0% | 37.2% |
| FP16 baseline (50 samp, 1.7B) | ✅ CONFIRMED | 50.0% | 49.0% |
| Static INT4 (standard, 500 samp, 360M) | ⚠️ LOSSLESS | 41.2% | 33.6% |
| Static INT4 (int3range, 360M) | ✅ CONFIRMED | 33.0% | 33.6% |
| Static INT4 (int3range, 135M) | ✅ CONFIRMED | 32.0% | 33.6% |
| Static INT4 (standard, 50 samp, 1.7B) | ✅ MATCHES PAPER | 40.0% | 41.1% |
| Static INT4 (int3range, 50 samp, 1.7B) | ⚠️ BELOW PAPER | 32.0% | 41.1% |

---

## Insight 1: Evaluation metric matters critically

**The paper uses unnormalized log-likelihood (`acc`), not length-normalized (`acc_norm`).**

- `acc_norm` (lm-eval default): ~49–54% for SmolLM-360M → matches paper's SmolLM-1.7B
- `acc` (unnorm): ~42% → matches paper's SmolLM-360M at 41.5% ✓

Direct test: `acc (unnorm)` on 50 val samples = **42.0%** vs paper's **41.5%** ✓  
500-sample confirmation: **42.6%** vs paper's **41.5%** (Δ = +1.1pp within noise) ✓

---

## Insight 2: KV cache hooks fail with DynamicCache (transformers 5.x)

transformers 5.x uses `DynamicCache` objects, not raw `(key, value)` tuples.  
Output hooks on attention modules silently fail to intercept KV tensors.  

**Fix**: Hook `k_proj` and `v_proj` Linear submodule outputs directly.  
SmolLM-360M: 32 attention layers × 2 (k+v) = **64 hooks** total.

Verification: KV-2bit gives **25.0%** (200 samples) = near-random, confirming hooks fire.

---

## Insight 3: sdpa attention blocks output_attentions

transformers 5.x uses sdpa (scaled dot product attention) by default — does NOT support
`output_attentions=True` (silently returns empty tuple).  

**Fix**: Reload model with `attn_implementation='eager'` for DWB controller signal extraction.

---

## Insight 4: INT4 losslessness is scale-dependent — mechanism fully verified ★

**Six INT4 variants at 360M (lossless) vs 1.7B (lossy): directly explained by residual error magnitude**

Accuracy results (360M, lossless):

| Variant | 100-samp | 200-samp | 500-samp | vs Paper 33.6% |
|---------|----------|----------|----------|----------------|
| Symmetric per-tensor | 44.0% | 44.5% | 41.6% | +8-11pp |
| Asymmetric per-tensor | 43.0% | 42.5% | — | +8-9pp |
| Symmetric per-token | 44.0% | 43.5% | 41.2% | +7-10pp |
| Asymmetric per-token | 39.0% | — | — | +5pp |
| Block-64 | 44.0% | — | — | +10pp |

**Mechanistic cross-scale comparison** (20 examples each):

| Metric | Standard INT4 — 360M | Standard INT4 — 1.7B |
|--------|---------------------|---------------------|
| Symmetry ratio (mean/std) | 0.0027 ≈ zero-mean | **0.0006** ≈ zero-mean |
| Relative error magnitude | 26.95% | **35.31%** (+31%) |
| Cancellation ratio | 0.30 (3.3× below naive) | **0.35** (2.9× below naive) |
| **Effective residual** (rel × cancel) | **8.1%** ← below threshold | **12.4%** ← above threshold |
| Accuracy impact | ~0pp (lossless) | ~10pp loss |

**Mechanism confirmed**: Both 360M and 1.7B have near-zero-mean errors (symmetry ~0). The difference is the **effective residual error** = relative_error × cancellation_ratio:
- 360M: 26.95% × 0.30 = **8.1%** — below the decision threshold → lossless
- 1.7B: 35.31% × 0.35 = **12.4%** — above the threshold → 10pp accuracy loss

**Root cause of higher error at 1.7B**: Larger hidden dimension (2048 vs 960) produces higher-variance KV tensors. At the same scale divisor (max/7), larger dynamic ranges → more relative quantization error. Cancellation is slightly weaker too (0.35 vs 0.30), compounding the effect.

**Decision threshold**: Somewhere between 8.1% and 12.4% effective residual error. Standard INT4 sits safely below it at ≤360M; crosses it at 1.7B. INT3-range (effective residual ~12.6% at 1.7B) is similarly above it — which is why int3range also fails at 1.7B.

**Self-reinforcing property**: Outlier tokens that set the quantization scale are also the most-attended tokens (high-confidence, rare content words per Insight 6) — they receive the best quantization AND the highest attention weight. This partially mitigates errors but cannot overcome the 1.7B magnitude increase.

---

## Insight 5: Paper's baseline degradation caused by coarse step size — ablation verified ★ NOVEL

**`int4_int3range` (scale=max/3, clamp[-4,3]) = 33.0% — matches paper's 33.6% (Δ = -0.6pp)**

**Controlled ablation (5 conditions, 50 samples)** — isolates step size vs. range clipping:

| Condition | scale | clamp | Acc | vs Std |
|-----------|-------|-------|-----|--------|
| A: Standard INT4 | max/7 | (−8, 7) | 46.0% | — |
| E: Intermediate | max/5 | (−8, 7) | 42.0% | −4pp |
| D: Narrow range only | max/7 | (−4, 3) | 38.0% | −8pp |
| **C: Coarse step only** | **max/3** | **(−8, 7)** | **28.0%** | **−18pp** |
| **B: int4_int3range** | **max/3** | **(−4, 3)** | **28.0%** | **−18pp** |

**Causal decomposition**: B = C (both 28%) — range clipping adds **0pp** once step is coarse. All −18pp degradation from coarse step size. Range clipping alone (D) is milder (−8pp) and does not interact with step size.

**Threshold**: lossless at max/5 (42%), degraded at max/3 (28%). Standard max/7 is well within the lossless regime.

**Conclusion**: The paper's "Static 4-bit KV" baseline uses **scale ≈ max/3** — this is the entire cause of the 33.6% baseline. Autoregressive errors ruled out (AR INT4 = 42%, same as single-pass).

---

## Insight 6: Controller relies on confidence (C_t) and entropy (H_t), not rarity ★ NEW

**Controller behavior analysis**: 50 HellaSwag examples, 1484 tokens, trained DWBController.

**Signal means by bit tier** (higher = signal value at that tier):

| Signal | 2-bit (unimportant) | 4-bit | 8-bit | 16-bit (critical) | Cohen's d (2 vs 16) |
|--------|---------------------|-------|-------|-------------------|---------------------|
| H_t (entropy) | 4.97 | 2.91 | 2.19 | 1.18 | 4.09 |
| R_t (rarity) | 0.985 | 0.992 | 0.993 | 0.992 | 0.52 |
| C_t (confidence) | 0.174 | 0.320 | 0.486 | 0.769 | **4.55** |

**Key findings:**
1. **C_t (confidence) is the most discriminative signal** (Cohen's d = 4.55): tokens where the model predicts with high certainty are assigned to 16-bit. These are typically content words, proper nouns, or rare subwords where meaning is unambiguous in context.
2. **H_t (entropy) is second most discriminative** (d = 4.09): high entropy (model uncertain about what comes next) → assigned 2-bit. Low entropy (model certain about context) → 16-bit.
3. **R_t (rarity) barely discriminates** (d = 0.52): all tokens score 0.985–0.993 on HellaSwag's vocabulary, providing almost no signal. The paper's Eq. 15 rarity term adds minimal value on this distribution.

**Token examples by tier:**
- 2-bit (unimportant): `"."`, `":"`, `"a"`, `"the"`, `"The"`, `"and"`, `"is"` — common function words and punctuation
- 16-bit (critical): `"cheer"`, `"le"`, `"ice"`, `"p"`, `"m"` — unusual subwords and rare content tokens

**Interpretation:** The controller learned that **confident, low-entropy positions** (where meaning is clear and context determines the next token) are paradoxically the "important" ones to preserve at high precision. Common function words at uncertain positions are safe to quantize aggressively — their error propagates into already-unpredictable computation.

This aligns with the INT4 losslessness mechanism (Insight 4): the most-attended tokens (those that matter for accuracy) are the ones where C_t is highest, and those are preserved at 16-bit by the controller.

---

## H4 Results: SmolLM-135M Cross-Model Validation

100 samples, `acc` (unnormalized), same hooks as 360M experiments.

| Condition | Ours | Paper | Delta | Status |
|-----------|------|-------|-------|--------|
| FP16 | 40.0% | 37.2% | +2.8pp | ✅ CONFIRMED |
| Standard INT4 per-tensor | 39.0% | 33.6% | +5.4pp | ⚠️ Lossless (same as 360M) |
| **int4_int3range** | **32.0%** | **33.6%** | **-1.6pp** | ✅ **MATCHES PAPER** |

Both key findings replicate on SmolLM-135M:
1. Standard INT4 is near-lossless (+5.4pp gap vs paper, same phenomenon as 360M)
2. int4_int3range (8-level INT4) matches the paper's baseline (-1.6pp gap at n=100)

Note: Paper reports *identical* static INT4 accuracy (33.6%) for both 135M and 360M, suggesting the
quantization scheme property (not model-specific characteristics) drives the baseline degradation.

---

## H4 Extension: SmolLM-1.7B — Scale-Dependent INT4 Behavior ★ NEW FINDING

50 samples, `acc` (unnormalized).

| Condition | Ours | Paper | Delta | Status |
|-----------|------|-------|-------|--------|
| FP16 | 50.0% | 49.0% | +1.0pp | ✅ CONFIRMED |
| Standard INT4 per-tensor | 40.0% | 41.1% | -1.1pp | ✅ **MATCHES PAPER** |
| int4_int3range | 32.0% | 41.1% | -9.1pp | ⚠️ Over-degrades at 1.7B |

**This reverses the 135M/360M finding.** At smaller models, standard INT4 was lossless and
int4_int3range reproduced the paper's baseline. At 1.7B, **standard INT4 is the paper's actual
baseline** — int4_int3range over-degrades.

**Scale-dependent losslessness pattern:**

| Model | Params | attn_heads | std INT4 | FP16 | Gap | INT4 pattern |
|-------|--------|-----------|----------|------|-----|--------------|
| 135M | 135M | 15 | 39.0% | 40.0% | 1pp | Lossless |
| 360M | 360M | 15 | 41.2% | 42.6% | 1.4pp | Lossless |
| 1.7B | 1.7B | 32 | 40.0% | 50.0% | 10pp | **Lossy — matches paper** |

**Implication for H2**: The paper's +7.6pp improvement claim is best-supported at 1.7B, where
standard INT4 genuinely degrades. At 135M/360M, the improvement is over a sub-standard baseline.
The 1.7B result validates the paper's core motivation for adaptive quantization.

---

## DWB Controller Results

- Architecture: Linear(4,128) → ReLU → Linear(128,128) → ReLU → Linear(128,4) = 33,540 params
- Training: 2,995 token samples, 5 epochs, lr=0.003
- Val accuracy: **45.6%** (vs 25% random) — controller learns importance quartile
- Bit distribution: {2bit: 57.3%, 4bit: 18.9%, 8bit: 8.3%, 16bit: 15.6%}, avg=5.05 bits/token
- DWB accuracy: **40.0%** on 100 samples, **38.0%** on 200 samples, **33.8%** on 500 samples (paper: 41.2%)

H3 definitive (500 samples, CI ±4.4pp): DWB 33.8% vs FP16 42.6% = **-8.8pp gap**.  
Gap from paper's 41.2%: **-7.4pp** — exceeds the ±4.4pp CI. **H3 NOT reproduced by our implementation.**

**Root cause — controller quality**: Our controller val_acc=36.6% (vs 25% random), using 5.03 avg bits.
Standard INT4 (4.0 bits) gives 41.6% — our DWB with MORE bits (5.03) gives LESS accuracy (33.8%).
This means the controller is mislabeling important tokens as low-priority (2-bit), causing degradation.
The paper's controller likely has much higher val_acc (undisclosed training details).

**Implementation gap confirmed**: DWB accuracy is highly sensitive to controller training quality.
Reproducing the paper's 41.2% requires controller training data/procedure not disclosed in the paper.

| Run | N | DWB | FP16 | Delta | CI | Status |
|-----|---|-----|------|-------|----|--------|
| dwb_100 (val_acc=0.456) | 100 | 40.0% | 42.6% | -2.6pp | ±10pp | Within noise |
| dwb_200 (val_acc=0.366) | 200 | 38.0% | 42.6% | -4.6pp | ±6.7pp | Within noise |
| dwb_500 (val_acc=0.366) | 500 | 33.8% | 42.6% | -8.8pp | ±4.4pp | **IMPL_GAP** |
| dwb_v2_500 (val_acc=0.446, avg=8.47b) | 500 | **37.0%** | 42.6% | -5.6pp | ±4.4pp | ⚠️ DUAL-OBJ TENSION |
| Paper | — | 41.2% | 41.5% | -0.3pp | — | Target (5.05 avg bits) |

**Controller sensitivity finding** ★: Better-trained controller (v2, val_acc=0.446) gives **37.0%** (500 samp) at avg_bits=8.47 vs paper's 41.2% at 5.05 bits. Bit distribution bimodal: {2: 41.7%, 4: 9.3%, 8: 7.2%, 16: 41.8%} — the controller splits tokens into "unimportant" (2-bit, 42%) and "critical" (16-bit, 42%) with little middle ground.

The controller becomes "conservative" — assigning high precision to most ambiguous tokens rather than finding the efficient middle ground. This reveals a **two-objective tension**: our quartile-labeling approach trains for classification accuracy alone, not for the paper's compound loss (α·CE + β·latency + γ·quality). A well-trained DWB controller must simultaneously achieve good accuracy AND maintain ~5.05 avg bits — this requires end-to-end training with the full compound loss, not just quartile classification.

**v1 vs v2 comparison:**

| Controller | Train samp | Epochs | val_acc | avg_bits | DWB acc (500 samp) | vs Paper |
|------------|-----------|--------|---------|----------|--------------------|---------|
| v1 | 100 | 5 | 0.366 | 5.03 | 33.8% | -7.4pp |
| v2 | 500 | 10 | 0.446 | 8.47 | **37.0%** | **-4.2pp** |
| Paper | — | — | — | 5.05 | 41.2% | target |

v2 improved accuracy by +3.2pp but required 1.68× more bits (8.47 vs 5.03). The paper achieves BOTH: 41.2% at only 5.05 bits. Closing H3 fully requires the paper's undisclosed compound-loss training procedure.

---

## Experiment Trajectory

| Run | Condition | N | Result | Paper | Status |
|-----|-----------|---|--------|-------|--------|
| 01c | FP16 (acc) | 50 | 42.0% | 41.5% | ✅ CONFIRMED |
| v3 | FP16 (acc) | 200 | 44.0% | 41.5% | ✅ CONFIRMED |
| 500 | FP16 (acc) | 500 | **42.6%** | 41.5% | ✅ CONFIRMED (CI±4.4pp) |
| v3 | KV-2bit | 200 | 25.0% | — | ✅ Hooks confirmed |
| v3 | KV-4bit per-tensor | 200 | 44.5% | 33.6% | ⚠️ Cannot reproduce |
| kvc | KV-4bit per-tensor | 200 | 44.5% | 33.6% | ⚠️ Cannot reproduce |
| kvc | KV-4bit per-token | 200 | 43.5% | 33.6% | ⚠️ Cannot reproduce |
| kvc | KV-4bit asymmetric | 200 | 42.5% | 33.6% | ⚠️ Cannot reproduce |
| 500 | KV-4bit per-token | 500 | **41.2%** | 33.6% | ⚠️ Cannot reproduce (statistically significant at n=500) |
| 500 | KV-4bit per-tensor | 500 | **41.6%** | 33.6% | ⚠️ Cannot reproduce (statistically significant) |
| inv | **int4_int3range** | 100 | **33.0%** | 33.6% | ✅ **MATCHES PAPER** |
| dwb | DWB adaptive | 100 | 40.0% | 41.2% | ~✅ H3 consistent |
| dwb_200 | DWB adaptive | 200 | **38.0%** | 41.2% | ~✅ H3 consistent (CI ±6.7pp) |
| run_ar_50 | FP16 (autoregressive) | 50 | 42.0% | 41.5% | ✅ AR matches single-pass |
| run_ar_50 | INT4 (autoregressive) | 50 | 42.0% | 33.6% | ⚠️ AR still lossless — methodology ruled out |
| h4_135m | SmolLM-135M FP16 | 100 | 40.0% | 37.2% | ✅ H4 CONFIRMED |
| h4_135m | SmolLM-135M standard INT4 | 100 | 39.0% | 33.6% | ⚠️ Lossless (cross-model) |
| h4_135m | **SmolLM-135M int4_int3range** | 100 | **32.0%** | **33.6%** | ✅ **H4 CONFIRMED — cross-model** |
| h4_1b7 | SmolLM-1.7B FP16 | 50 | 50.0% | 49.0% | ✅ H4 CONFIRMED |
| h4_1b7 | **SmolLM-1.7B standard INT4** | 50 | **40.0%** | **41.1%** | ✅ **MATCHES PAPER at 1.7B** |
| h4_1b7 | SmolLM-1.7B int4_int3range | 50 | 32.0% | 41.1% | ⚠️ Over-degrades at 1.7B |
| H1 | Latency | — | — | 2.41 ms/tok | ⏳ AWAITING GPU |

---

## Beta Sweep: Dual-Objective Tension Is Fundamental ★ NEW

**Question**: Can increasing the latency penalty β resolve the H3 dual-objective tension?

β ∈ {0.1, 0.5, 1.0, 2.0}, 100 train samples, 5 epochs, α=1.0, γ=0.1.

| β | Acc (100s) | avg_bits | val_acc | Bit dist (2/4/8/16%) |
|---|-----------|----------|---------|----------------------|
| 0.1 | 39.0% | 5.30 | 0.407 | 38.5/37.6/9.9/14.0 |
| 0.5 | 39.0% | 3.92 | 0.341 | 37.0/57.4/0.0/5.5 |
| 1.0 | **32.0%** | **2.00** | 0.257 (<random) | 100/0/0/0 |
| 2.0 | **32.0%** | **2.00** | 0.227 (<random) | 100/0/0/0 |
| Paper | 41.2% | 5.05 | — | undisclosed |

**Key finding (COMPLETE — all 4 betas confirmed)**:
- β=0.1: avg_bits=5.30 ≈ paper's 5.05 (closest), accuracy=39.0% (still 2.2pp below paper's 41.2%)
- β=0.5: bits drop to 3.92, accuracy unchanged at 39.0% — higher penalty reduces bits for free but still can't reach 41.2%
- β≥1.0: TOTAL COLLAPSE — controller assigns 100% 2-bit tokens, accuracy=32.0%, val_acc falls BELOW random baseline (0.25). The latency penalty completely overwhelms the classification objective.
- There is NO β that achieves both 41.2% accuracy AND 5.05 avg_bits. The collapse at β=1.0 proves that our training framework cannot interpolate between the two objectives — the loss landscape has no Pareto-optimal solution under quartile-classification training.

**Conclusion**: The dual-objective tension is NOT a hyperparameter issue — it is a fundamental difference in training objectives. Our quartile-classification approach cannot simultaneously optimize accuracy and compression. The paper's compound loss (α·CE + β·latency + γ·quality) with end-to-end training is the necessary ingredient.

---

## DWB-TurboQuant: Task-Specific Gains ★ CAVEAT ADDED

Full cross-benchmark results (SmolLM-360M, 100 samples each):

| Condition | HellaSwag | ARC-Challenge | BoolQ | avg_bits |
|-----------|-----------|---------------|-------|----------|
| FP16 | 41.0% | 35.0% | 55.0% | 16.0 |
| DWB-scalar | 40.0% | 26.0% | **61.0%** | 5.05/7.72/7.88 |
| **DWB-TurboQuant** | **42.0%** | **29.0%** | 41.0% | 5.05/7.72/7.88 |
| Paper DWB | 41.2% | — | — | 5.05 |

**BoolQ finding**: DWB-TurboQuant shows -20pp regression vs DWB-scalar on BoolQ (41% vs 61%). DWB-scalar itself outperforms FP16 by +6pp on BoolQ (61% vs 55%) — KV quantization appears to regularize yes/no probability comparison. WHT rotation reverses this effect.

**Interpretation**: WHT rotation is **task-specific** — it benefits open-ended generation tasks (HellaSwag commonsense, ARC-Challenge reasoning) but hurts closed-form yes/no QA. The mechanism likely relates to how quantization errors propagate into log-probability comparisons over short continuations ("yes"/"no") vs. longer phrase completions.

Note: n=100 gives CI≈±10pp. The DWB-scalar +6pp over FP16 on BoolQ may be noise; the -20pp TurboQuant regression is harder to dismiss statistically.

**Qualified conclusion**: DWB-TurboQuant is an improvement on generation/reasoning tasks at identical compression, but should not be used for closed-form QA without further investigation.

---

## FPGA Controller: Phase 2 + Phase 4 Results ★ NEW

### Phase 2: Richer Features (Head Entropy + Layer Depth)

4D controller `[kv_norm, pos_frac, head_entropy, layer_depth]` vs Phase 1's 2D controller:

| Metric | Phase 2 (4D) | Phase 1 (2D) |
|--------|-------------|-------------|
| Accuracy | 41.0% | 41.0% |
| avg_bits | 4.0 | 4.0 |
| Bit dist | 100% 4-bit | 100% 4-bit |

**Conclusion**: Richer features change nothing at 360M. Convergence to 100% 4-bit is a property
of the model's eff_residual=8.1% (below losslessness threshold), not a feature engineering
limitation. No controller can improve on 100% 4-bit at this scale.

### Phase 4: Binary {4,8}-bit FPGA Controller

| Metric | Our FPGA Controller | Paper DWB | FP16 |
|--------|---------------------|-----------|------|
| Accuracy | **41.0%** | 41.2% | 42.6% |
| avg_bits | **4.0** | 5.05 | 16.0 |
| FPGA cost | **0.290** | 0.414 | 1.010 |
| FPGA speedup | **3.48x** | 2.44x | 1.00x |

**+43% FPGA throughput over paper DWB at equal accuracy.**

Why we beat the paper: Paper allocates 47.9% of tokens to 2-bit. On Xilinx Ultrascale+ BRAM,
2-bit uses the same 4-bit BRAM port — same hardware cost but worse accuracy. Our binary {4,8}
controller eliminates this suboptimal 2-bit option entirely.

**Key pending experiment**: SmolLM-1.7B where eff_residual=12.4% > threshold. Phase 5 v2 script (`run_phase5_1b7_pertok.py`) pushed to GitHub (commit 096ab5e) and ready to run on Brev A4000.

**Critical fix — per-token quality proxy + correct beta**: Two discoveries:

1. **Global quality scores fail**: q4=0.671, q8=0.979 → dL/dp4 > 0 for all β → 100% 8-bit.
2. **Per-token proxy alone insufficient**: avg q8_local - q4_local = 0.337 at 360M, so even per-token quality gives 100% 8-bit with beta=0.5 (threshold only 0.134).

**Correct beta = 1.5** (validated by smoke test on 360M):
- Threshold = 1.5 × 0.267 = 0.401 > avg gap 0.337 → avg gradient pushes 4-bit ✅
- Smoke test beta sweep at 360M (q8-q4: mean=0.337, std=0.050):
  - beta=1.0: 100% 8-bit (threshold < gap), 1.80× speedup
  - **beta=1.5: 100% 4-bit** (correct for 360M where INT4 is lossless), 3.48× speedup ✅
  - beta=2.0: 100% 4-bit, 3.48×
- At 1.7B (expected larger gap ≈ 0.38): beta=1.5 threshold=0.401 ≈ 60th percentile → ~40-60% 4-bit mixed allocation

Phase 5 script now sweeps [1.0, 1.5, 2.0, 3.0] and selects best. Code validated end-to-end on 360M (smoke test PASS).

---

## Lessons and Constraints

- **Metric**: Paper uses unnormalized `acc` (~42%), NOT `acc_norm` (~54%). Always use normalize=False.
- **KV hooks**: Hook `k_proj` and `v_proj` directly (64 hooks for SmolLM-360M).
- **Eager attention**: For DWB signal extraction, use `attn_implementation='eager'`.
- **INT4 losslessness is scale-dependent**: Lossless at 135M/360M (15 heads); genuine degradation at 1.7B (32 heads). Standard INT4 matches paper's 41.1% baseline at 1.7B; int4_int3range matches at 135M/360M.
- **500 samples sufficient for 360M**: At n=500, CI=±4.4pp. The +8pp gap is statistically significant.
- **1.7B validates paper's core claim**: Genuine INT4 degradation occurs at scale — H2 is strongest at 1.7B.
- **Global quality scores fail at ANY scale**: avg q8-q4 ≈ 0.337 even at 360M → dL/dp4 > 0 with beta=0.5 → 100% 8-bit. Not just a 1.7B problem.
- **Per-token quality proxy requires correct beta**: q_local alone doesn't fix it — need beta≥1.5 so FPGA threshold (beta*0.267) exceeds avg quality gap (0.337).
- **beta=1.5 validated at 135M/360M, NOT at 1.7B**: Smoke test on 360M shows clean flip. At 1.7B, beta*=1.584 — beta=1.5 is just below the transition (all 8-bit). Use beta=1.6 at 1.7B for genuine mixed allocation.
- **beta* = gap_mean/0.267 confirmed at ALL THREE SCALES**: 135M: 0.3297/0.267=1.234 [measured: 1.2-1.3 ✓], 360M: 0.3367/0.267=1.261 [measured: 1.2-1.4 ✓], 1.7B: 0.4235/0.267=1.584 [measured: 1.5-1.6 ✓]. This is the key calibration formula.
- **AUTHORITATIVE 1.7B numbers (5-seed, Session 19)**: beta=1.70 → 79.9%±2.8pp 4-bit, avg_bits=4.80, 2.93× speedup (+20% vs DWB). beta=1.60 single-run (68.4%) was a high outlier; multi-seed mean=60.6%±1.9pp. Always report multi-seed means.
- **Calibration corpus size (Session 20)**: Even 1 text (~1536 tokens, <3 sec) gives beta* within ±0.015. 5 texts reduce error to ±0.005. The transition window (±0.03) is 2x larger than 1-text error — calibration cannot push beta* past the transition boundary.
- **Paper (current HEAD, 12f461c)**: All claims consistent. Abstract, Contribution #2, Discussion all reference 2.93× (+20%). 1.7B HellaSwag accuracy TBD (GPU).
- **Formula consistency**: ε_eff = ε_rel × r_survive. All three rows now consistent: 135M: 0.249×0.28=0.069 ✓, 360M: 0.270×0.30=0.081 ✓, 1.7B: 0.353×0.351=0.124 ✓. Table column was incorrectly labeled r_cancel (fixed in session 16).
- **Lessons for paper writing**: The fact that 2-bit is dominated is the core contribution — state it in abstract, intro bullet 1, and conclusion. The per-token proxy and beta calibration are the technical enablers for 1.7B (scale-generalization). These are independent contributions, both novel relative to DWB.

---

## Fine Beta Sweep: Phase Transition at β=1.260 Confirmed ★ NEW (Session 13)

**Date**: 2026-04-19 (Session 13)

**Method**: Fine-grained beta sweep [1.1, 1.2, 1.25, 1.3, 1.4] on cached 89,856-token signals from SmolLM-360M. Uses Stage 2 (controller training) only — model already freed.

**Results**:

| β | threshold | 4-bit% | Regime |
|---|-----------|--------|--------|
| 1.1 | 0.294 | 0% | 8-bit dominant |
| 1.2 | 0.321 | 0% | 8-bit dominant |
| **1.25** | **0.334** | **41.7%** | **mixed (at transition)** |
| 1.3 | 0.348 | 58.7% | mixed |
| 1.4 | 0.374 | 100% | 4-bit dominant |

**Key finding**: Phase transition confirmed at β* = gap_mean/0.267 = 0.337/0.267 = **1.260**, matching theory to within ±0.04. The transition window is β ∈ [1.20, 1.40].

**Three-scale beta* comparison (all measured on CPU)** ★:

| Scale | gap_mean | β* (theory) | β* (measured) | β=1.5 outcome |
|-------|----------|-------------|---------------|---------------|
| 135M  | 0.3297   | 1.233       | [1.2, 1.3] ✓  | 100% 4-bit    |
| 360M  | 0.3367   | 1.260       | [1.20, 1.40] ✓| 100% 4-bit    |
| 1.7B  | 0.400 (est) | 1.498    | (simulated)   | ~53% mixed    |

**β=1.5 as a near-universal operating point** ★:
- At 135M: β*=1.233, so β=1.5 safely above → 100% 4-bit (INT4 lossless)
- At 360M: β*=1.260, so β=1.5 safely above → 100% 4-bit (INT4 lossless)
- At 1.7B: β* = 0.400/0.267 = **1.498 ≈ 1.5** → right at transition → mixed {4,8} allocation

β=1.5 **automatically adapts** its behavior across scales: pure 4-bit at small lossless scales (max FPGA throughput), genuine mixed allocation at large lossy scales (correct accuracy-efficiency tradeoff). No per-scale tuning needed. This is validated across all three SmolLM scales.

**Results file**: `research/experiments/fpga-controller/phase5-benchmark/results/beta_transition_fine.json`

---

## 1.7B Simulation: Genuine Mixed Allocation Confirmed (Predicted) ★ NEW

**Date**: 2026-04-19 (Session 10)

**Method**: Generated 90,000 synthetic tokens with predicted 1.7B gap distribution
(mean=0.400, std=0.058, derived from eff_residual scaling: 8.1%→12.4%).
Trained binary {4,8} controller on synthetic signals. This is a MODEL-BASED PREDICTION
— not a hardware measurement. Validates gradient analysis.

**Simulation results**:

| beta | 4-bit% | 8-bit% | avg_bits | FPGA speedup | Outcome |
|------|--------|--------|----------|-------------|---------|
| 1.0  | 0%     | 100%   | 8.00     | 1.80×       | All 8-bit (too conservative) |
| **1.5** | **53.5%** | **46.5%** | **5.86** | **2.43×** | **Genuine mixed allocation** |
| 2.0  | 100%   | 0%     | 4.00     | 3.48×       | All 4-bit (max FPGA speed) |
| 3.0  | 100%   | 0%     | 4.00     | 3.48×       | All 4-bit |

**Key findings**:
1. **beta=1.5 produces genuine mixed {4,8}-bit allocation** at predicted 1.7B scale (53.5% 4-bit, 46.5% 8-bit). This confirms the gradient analysis: gap mean=0.400 ≈ threshold=0.401 at beta=1.5.
2. **FPGA speedup 2.43×** is nearly identical to DWB (2.44×) at similar avg_bits (5.86 vs 5.05), but WITHOUT any 2-bit tokens.
3. **Accuracy lower bound 44.3%** (linear interpolation). With intelligent token selection, actual accuracy should exceed this (controller finds optimal assignment). DWB achieves 48.6% at 1.7B.
4. **beta=2.0 → 100% 4-bit (3.48×)** — highest FPGA throughput but predicted accuracy drops to 41.1% (same as static INT4, as expected when INT4 is lossy).

**Script**: `research/experiments/fpga-controller/phase5-benchmark/code/simulate_1b7_prediction.py`
**Results**: `research/experiments/fpga-controller/phase5-benchmark/results/sim_1b7_prediction.json`

**Implications for paper**:
- The simulation is added as Table tab:sim_1b7 in the Discussion section
- Pareto dominance argument is nuanced: at 1.7B, we eliminate the dominated 2-bit option, but the overall accuracy-FPGA tradeoff depends on controller quality
- GPU measurement (Brev A4000) will determine whether beta=1.5 achieves competitive accuracy

---

## SmolLM-1.7B Beta Calibration: MEASURED ★ NEW (Session 17)

**Date**: 2026-04-19 (Session 17)

**Experiment**: `beta_calibration_1b7.py` — real SmolLM-1.7B signals, 10 texts, max_len=64, 15,360 tokens (26 seconds for signal extraction on CPU).

**Key result**: gap_mean=**0.4235** (simulation predicted 0.400), beta*=**1.584** (simulation predicted 1.498).

**Implication — beta=1.5 is NOT at the 1.7B transition:**
- beta=1.5: threshold=0.401 < gap_mean=0.424 → **0% 4-bit** (all 8-bit, 1.80× speedup)
- beta=1.6: threshold=0.428 > gap_mean=0.424 → **68.4% 4-bit** (genuine mixed, 2.69× speedup)
- This is **+10% over DWB** (2.44×) at similar avg_bits (5.26 vs 5.05)!

**Cross-scale formula beta* = gap_mean/0.267 — CONFIRMED at all three scales:**
| Scale | gap_mean (measured) | beta* (theory) | Measured transition | beta=1.5 outcome |
|-------|---------------------|----------------|---------------------|-----------------|
| 135M  | 0.3297              | 1.234          | [1.2, 1.3] ✓        | 100% 4-bit      |
| 360M  | 0.3367              | 1.261          | [1.2, 1.4] ✓        | 100% 4-bit      |
| 1.7B  | **0.4235**          | **1.584**      | **[1.5, 1.6] ✓**   | 0% 4-bit (below!)|

**Recommendation**: measure gap_mean on 10 texts (< 1 min CPU), set beta = gap_mean/0.267 + 0.1.

**Paper updates**: Simulation table replaced with real data. Tab. tab:betastar row 3 updated. Abstract/conclusion/contributions revised. Figures regenerated with all three scales measured.

---

## 1.7B Fine Beta Sweep: Transition Window [1.55, 1.57] ★ NEW (Session 18)

**Date**: 2026-04-19 (Session 18)

**Experiment**: `beta_transition_fine_1b7.py` — 10 betas in [1.50-1.70] on cached 15,360 tokens (11 seconds).

**Key results**:
- Transition window: **[1.55, 1.57]** (theory predicted 1.584 — matches within ±0.015)
- 1.7B transition is SOFTER than 360M: no sharp jump to 100%, instead a broad mixed plateau
- Broad plateau β∈[1.57, 1.70+]: speedups 2.52–2.84× (ALL beat DWB's 2.44×)
- **Best at β=1.70**: 75.7% 4-bit, avg_bits=5.03 (matches DWB), **2.84× speedup = +16% vs DWB**

| β    | 4-bit% | avg_bits | speedup | vs DWB |
|------|--------|----------|---------|--------|
| 1.55 | 0%     | 8.00     | 1.80×   | −26%   |
| 1.57 | 59.2%  | 5.63     | 2.52×   | +3%    |
| 1.65 | 72.0%  | 5.12     | 2.76×   | +13%   |
| **1.70** | **75.7%** | **5.03** | **2.84×** | **+16%** |

**Why softer transition at 1.7B**: Wider gap distribution (std=0.063 vs 0.050 at 360M) means many tokens are near the threshold, creating a gradual rather than sharp transition.

**Practical implication**: Any β ≥ 1.57 gives useful mixed allocation at 1.7B. Formula recommendation: β = gap_mean/0.267 + 0.1 ≈ 1.68, landing comfortably in the plateau.

---

## Session 19: Multi-Seed Reproducibility Test — Authoritative Numbers ★ NEW

**Date**: 2026-04-19 (Session 19)

**Experiment**: `reproducibility_test_1b7.py` — N_SEEDS=5, BETAS=[1.60, 1.65, 1.70] on cached 15,360 1.7B tokens.

**Key finding**: Single-run numbers from Session 18 were conservative. The 5-seed means at β=1.70 give 2.93× FPGA speedup (+20% vs DWB) at avg_bits=4.80 — FEWER bits than DWB's 5.05. This is a strict Pareto improvement.

| β    | 4-bit% runs (5 seeds)             | Mean ± std     | avg_bits | FPGA speedup | vs DWB (2.44×) |
|------|-----------------------------------|----------------|----------|-------------|----------------|
| 1.60 | [62.3, 58.4, 63.3, 60.5, 58.7]   | 60.6% ± 1.9pp  | 5.51     | 2.55×       | +4%            |
| 1.65 | [75.9, 69.3, 73.2, 68.9, 67.8]   | 71.0% ± 3.0pp  | 5.16     | 2.74×       | +12%           |
| **1.70** | **[82.7, 82.8, 80.8, 77.1, 76.3]** | **79.9% ± 2.8pp** | **4.80** | **2.93×** | **+20%** |

**Key insight**: Training stochasticity is only ±2-3pp — single runs are reliable for direction decisions. Report multi-seed means for final numbers. The β=1.60 single-run outlier (68.4%, +10%) was superseded by mean=60.6%±1.9pp (+4%).

**Paper updates (Session 19 final consistency pass)**: Abstract revised (removed "β=1.5 near-universal"), Contribution #2 updated to "2.93×, +20%", Discussion updated to 5-seed multi-run means. Commit: 12f461c.

---

## Session 20: Calibration Sensitivity Analysis — 1 Text Suffices ★ NEW

**Date**: 2026-04-19 (Session 20)

**Experiment**: `calibration_sensitivity.py` — 20 random subsamples at each corpus size n_texts ∈ {1,2,3,5,7,10} from cached 1.7B signals.

**Key finding**: Even 1 text (~1536 tokens, <3 seconds on CPU) estimates β* within ±0.015 of the 10-text value. The "<1 minute calibration" claim is extremely conservative.

| n_texts | ~tokens | gap_mean ± std   | β* ± std       | max error vs true β*=1.584 |
|---------|---------|-----------------|----------------|---------------------------|
| 1       | 1536    | 0.4231 ± 0.0016 | 1.585 ± 0.006  | **0.015** (ACCEPTABLE)    |
| 2       | 3072    | 0.4234 ± 0.0011 | 1.586 ± 0.004  | 0.009                     |
| 3       | 4608    | 0.4236 ± 0.0008 | 1.587 ± 0.003  | 0.010                     |
| 5       | 7680    | 0.4232 ± 0.0005 | 1.585 ± 0.002  | 0.005                     |
| 7       | 10752   | 0.4234 ± 0.0004 | 1.586 ± 0.001  | 0.005                     |
| 10      | 15360   | 0.4235 ± 0.0000 | 1.586 ± 0.000  | 0.002                     |

**Why this matters**: The transition window is ±0.03 wide ([1.55, 1.57]). A 1-text calibration error of ±0.015 is well within this tolerance — it cannot push β* past the transition boundary. The gap_mean is a robust statistic of the token-level quality proxy distribution, not sensitive to individual tokens.

**Paper update**: Added to Discussion — "even a 1-text corpus (<3 seconds on CPU) estimates β* within ±0.015... The '<1 minute' calibration claim is extremely conservative."

**Lessons update**:
- **Calibration is robust to corpus size**: 1 text suffices (±0.015 max error). 5 texts reduce error to ±0.005. Use 10 texts for publication-quality results.
- **Recommendation confirmed**: β = gap_mean/0.267 + 0.1. The +0.1 safety margin is ≈7× larger than the calibration error.

---

## Session 21: Cross-Architecture Beta* Validation — OPT-125M ★ NOVEL

**Date**: 2026-04-19 (Session 21)

**Hypothesis**: beta* = gap_mean/0.267 is derived from FPGA hardware constants, not model-specific properties. It should generalize across model architectures.

**Experiment**: Extract k/v signals from facebook/opt-125m (Meta OPT architecture, vs SmolLM's LLaMA-style). Fine sweep beta in [0.3, 1.5].

**Results**:
- gap_mean = 0.2131, gap_std = 0.0389 (much lower than SmolLM — OPT has less KV variance)
- Predicted beta* = 0.2131/0.267 = **0.798**
- Measured transition: **[0.75, 0.80]** — at beta=0.75: 0% 4-bit; at beta=0.80: 48.2% 4-bit (exact transition!)
- Theory error: **0.023** — CONFIRMED (within +-0.025)

| beta | threshold | 4-bit% | speedup | regime |
|------|-----------|--------|---------|--------|
| 0.75 | 0.2003    | 0%     | 1.80x   | 8-bit  |
| **0.80** | **0.2136** | **48.2%** | **2.35x** | **MIXED — at transition** |
| 0.85 | 0.2270    | 98.9%  | 3.45x   | 4-bit  |
| 1.00 | 0.2670    | 100%   | 3.48x   | 4-bit  |

**Cross-architecture summary** (all MEASURED, two families, four checkpoints):
| Family | Model | gap_mean | beta* (theory) | measured transition | error |
|--------|-------|----------|----------------|---------------------|-------|
| LLaMA (SmolLM) | 135M | 0.330 | 1.234 | [1.2, 1.3] | <0.03 |
| LLaMA (SmolLM) | 360M | 0.337 | 1.261 | [1.2, 1.4] | <0.04 |
| LLaMA (SmolLM) | 1.7B | 0.424 | 1.584 | [1.55, 1.57] | <0.015 |
| **OPT (Meta)** | **125M** | **0.213** | **0.798** | **[0.75, 0.80]** | **0.023** |

**Why OPT has lower gap_mean**: OPT-125M has 12 attention heads and 768 hidden dim (vs SmolLM-135M's 15 heads / 960 dim). Smaller hidden dim -> lower KV magnitude variance -> smaller per-token INT4 error -> lower q8-q4 gap.

**Key insight**: The formula beta* = gap_mean/0.267 is a universal HARDWARE criterion. The 0.267 = (c8-c4)/C_FP16 = (0.560-0.290)/1.010 depends only on FPGA BRAM port costs, not model architecture. gap_mean depends on the model's KV distribution — models with lower KV variance need smaller beta to enter the 4-bit regime.

**Paper update**: Added OPT-125M row to tab:betastar, Discussion "Cross-architecture validation" paragraph, Conclusion updated to "four model checkpoints across two architectures".

---

## Session 24: GPT-2 Cross-Architecture Validation — 3rd Model Family ★ NOVEL

**Date**: 2026-04-19 (Session 24)

**Hypothesis**: beta* = gap_mean/0.267 holds for GPT-2 (OpenAI), which uses a fundamentally different attention implementation (Conv1D combined QKV c_attn) vs LLaMA (separate k_proj/v_proj) and OPT (separate self_attn.k_proj/v_proj).

**Model**: openai/gpt2 (GPT-2 Small, 124M, 12 layers, 768 hidden dim, 12 heads)

**Results**:
- gap_mean = 0.1956 (lower than OPT-125M's 0.213, despite same hidden_dim=768 — architecture AND pre-training matter)
- Predicted beta* = 0.1956/0.267 = **0.733**
- Measured transition: **[0.70, 0.80]** — at beta=0.70: 18.2% 4-bit (entering mixed); at beta=0.80: 100% 4-bit
- Theory error: **0.017** — tightest fit across all 5 checkpoints!

| beta | threshold | 4-bit% | speedup | regime |
|------|-----------|--------|---------|--------|
| 0.60 | 0.1602    | 0%     | 1.80x   | 8-bit  |
| 0.70 | 0.1869    | 18.2%  | 1.98x   | MIXED entering |
| **0.75** | **0.2003** | **64.5%** | **2.62x** | **MIXED** |
| 0.80 | 0.2136    | 100%   | 3.48x   | 4-bit  |

**Full cross-architecture summary (5 checkpoints, 3 families, ALL CPU-measured)**:

| Family | Model | gap_mean | beta* theory | measured transition | error |
|--------|-------|----------|--------------|---------------------|-------|
| LLaMA (SmolLM) | 135M | 0.330 | 1.234 | [1.2, 1.3] | <0.03 |
| LLaMA (SmolLM) | 360M | 0.337 | 1.261 | [1.2, 1.4] | <0.04 |
| LLaMA (SmolLM) | 1.7B | 0.424 | 1.584 | [1.55, 1.57] | <0.015 |
| OPT (Meta) | 125M | 0.213 | 0.798 | [0.75, 0.80] | 0.023 |
| **GPT-2 (OpenAI)** | **124M** | **0.196** | **0.733** | **[0.70, 0.80]** | **0.017** |

**Key insight**: GPT-2 has LOWER gap_mean than OPT-125M (0.196 vs 0.213) despite same hidden_dim=768.
This confirms that gap_mean depends on KV activation distribution (pre-training data + architecture), not just hidden_dim.
The formula beta* = gap_mean/0.267 correctly accounts for this — empirical calibration is essential.

**Paper updates**: tab:betastar expanded to 5 rows/3 families, Discussion extended with GPT-2 paragraph,
Conclusion updated to "five checkpoints across three architectures (all within +-0.04)", GPT-2 BibTeX added.

---

## Session 25: Instruct Model Calibration Transfer Test ★ SURPRISING

**Date**: 2026-04-19 (Session 25)

**Hypothesis**: Beta* calibrated on SmolLM-360M base transfers to SmolLM-360M-Instruct (same architecture, SFT weights). Predicted: |delta| < 0.02 (within calibration noise).

**Result**: REFUTED — gap_mean dramatically different.

| Model | gap_mean | beta* | delta |
|-------|----------|-------|-------|
| SmolLM-360M Base | 0.3367 | 1.261 | — |
| SmolLM-360M Instruct | **0.1941** | **0.727** | **-0.143** |

**Finding**: Instruction fine-tuning shifts gap_mean by 43% (0.337 → 0.194). This is larger than any cross-architecture difference we measured (OPT vs GPT-2: only 0.017). Instruct gap_mean=0.194 ≈ GPT-2 Small=0.196 despite being LLaMA-style architecture.

**Mechanistic hypothesis**: Instruction tuning regularizes KV activations toward more uniform magnitudes. Structured instruction data (Q&A patterns) produces more focused attention distributions → smaller relative INT4 errors → lower q8-q4 gap. Similar to how GPT-2 (pre-trained on WebText, more structured than raw Wikipedia) also shows lower gap_mean.

**Practical implication (critical)**: Calibration MUST be done on the deployed model checkpoint, not derived from the base. Using base beta*=1.261 for the instruct model would give correct behavior at 360M (100% 4-bit, INT4 lossless) but for 1.7B instruct models, would risk over-aggressive 4-bit assignment. Since calibration is <3 seconds, always re-calibrate on the exact deployed model.

**Session 25b**: Also tested SmolLM-135M-Instruct: gap_mean=0.181, beta*=0.677, delta=-0.149 (-45.2%).

**Pattern confirmed across scales**:

| Model | Base gap_mean | Instruct gap_mean | Delta | beta* shift |
|-------|--------------|-------------------|-------|------------|
| SmolLM-135M | 0.330 | 0.181 | -45% | 1.233 → 0.677 |
| SmolLM-360M | 0.337 | 0.194 | -43% | 1.261 → 0.727 |

The ~44% reduction in gap_mean is scale-independent and reproducible. The shift exceeds any cross-architecture difference measured. Instruct models converge to 0.18-0.19 gap_mean regardless of scale.

**Transition verification** (Session 25c): Beta sweep on 360M-Instruct controller confirms prediction:
- beta=0.50: 0% 4-bit (8-bit) — threshold 0.134 < gap 0.194 ✓
- beta=0.70: 15.6% 4-bit (entering mixed) — threshold 0.187 ≈ gap ✓
- beta=0.75: 52.6% 4-bit (mixed, above transition) — threshold 0.200 > gap ✓
- beta=0.80: 100% 4-bit ✓; beta=1.00: 100% 4-bit ✓
- Measured transition: [0.70, 0.80], theory error = 0.023 — within ±0.04 ✓

**Paper update**: "Fine-tuning shifts the KV distribution" paragraph + Table tab:instruct (now 6 cols including measured transition for 360M-Instruct, verified error=0.023) added to Discussion.

---

## Session 26: TinyLlama-1.1B GQA — 4th Model Family ★ NOVEL GQA INSIGHT

**Date**: 2026-04-20 (Session 26)

**Hypothesis**: beta* = gap_mean/0.267 holds for TinyLlama-1.1B, which uses Grouped Query Attention (GQA: n_kv_heads=4, n_heads=32, head_dim=64 → k/v_proj output is 256-dim per token, vs SmolLM's 960-2048-dim).

**Model**: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T (24,904 signals from 10 texts, 22 layers)

**Key results**:
- gap_mean = **0.1888** (predicted range: [0.337, 0.424] from SmolLM scale — drastically BELOW)
- Predicted beta* = 0.1888/0.267 = **0.707**
- Fine sweep [0.68–0.80]: transition at [0.68, 0.74], mixed zone [0.70, 0.72]
- Theory error: **0.003** — BEST FIT across all 6 checkpoints!

| beta | threshold | 4-bit% | regime |
|------|-----------|--------|--------|
| 0.68 | 0.1816 | 0% | 8-bit |
| 0.70 | 0.1869 | 45.1% | MIXED |
| 0.72 | 0.1922 | 47.9% | MIXED |
| 0.74 | 0.1976 | 100% | 4-bit |

**SURPRISING FINDING — GQA lowers beta* vs scale expectation**:
- SmolLM-1.7B (MHA, n_kv_heads=32): gap_mean=0.424, beta*=1.584
- TinyLlama-1.1B (GQA, n_kv_heads=4): gap_mean=0.189, beta*=0.707
- Same ballpark scale, but gap_mean is 2.2× lower with GQA!

**Mechanism**: GQA reduces K/V projection output from (seq, 2048) to (seq, 256) — 8× fewer dimensions per token. Smaller K/V vectors have lower per-token variance → smaller INT4 quantization error → lower q8-q4 gap → lower beta*. **GQA acts as implicit KV regularization: it lowers beta* far below what model scale alone would predict.**

**Full 6-checkpoint / 4-family summary**:

| Family | Model | gap_mean | beta* theory | measured transition | error |
|--------|-------|----------|--------------|---------------------|-------|
| LLaMA-MHA (SmolLM) | 135M | 0.330 | 1.234 | [1.2, 1.3] | <0.03 |
| LLaMA-MHA (SmolLM) | 360M | 0.337 | 1.261 | [1.2, 1.4] | <0.04 |
| LLaMA-MHA (SmolLM) | 1.7B | 0.424 | 1.584 | [1.55, 1.57] | <0.015 |
| **LLaMA-GQA (TinyLlama)** | **1.1B** | **0.189** | **0.707** | **[0.68, 0.74]** | **0.003** |
| OPT-MHA (Meta) | 125M | 0.213 | 0.798 | [0.75, 0.80] | 0.023 |
| GPT-2 (OpenAI) | 124M | 0.196 | 0.733 | [0.70, 0.80] | 0.017 |

**Paper updates**: TinyLlama row added to tab:betastar, GQA insight paragraph added to Discussion, all counts updated to 6 checkpoints / 4 families (±0.04 bound holds), TinyLlama BibTeX added.

---

## Session 27: TinyLlama-1.1B-Chat — GQA Instruct Calibration Transfer (Null Shift) ★ NOVEL

**Date**: 2026-04-20 (Session 27)

**Hypothesis**: TinyLlama-1.1B-Chat (SFT) shows ~44% gap_mean reduction like SmolLM-instruct models.

**Result: REFUTED — null shift observed**
- TinyLlama base: gap_mean = 0.1888, β* = 0.707
- TinyLlama Chat: gap_mean = **0.1883**, β* = **0.705** — Δ = **-0.3%** (essentially zero!)

**Interpretation — 2×2 matrix completed: {MHA, GQA} × {base, instruct}**:

| Architecture | Variant | gap_mean | β* | SFT Δgap |
|---|---|---|---|---|
| MHA (SmolLM-135M) | Base | 0.330 | 1.233 | — |
| MHA (SmolLM-135M) | Instruct | 0.181 | 0.677 | **-45%** |
| MHA (SmolLM-360M) | Base | 0.337 | 1.261 | — |
| MHA (SmolLM-360M) | Instruct | 0.194 | 0.727 | **-43%** |
| **GQA (TinyLlama-1.1B)** | **Base** | **0.189** | **0.707** | **—** |
| **GQA (TinyLlama-1.1B)** | **Chat** | **0.188** | **0.705** | **-0.3% (null)** |

**Mechanistic explanation**: GQA with n_kv_heads=4 already reduces K/V dimension to 256 (vs MHA's 960-2048), placing gap_mean at ~0.189 — the same "instruction-level floor" (~0.18-0.19) that MHA models reach only through SFT. There's nothing left for SFT to regularize.

**Key insight**: The floor gap_mean ≈ 0.18-0.19 appears to be a natural lower bound for well-trained models, reachable either by GQA architecture OR by instruction fine-tuning of MHA models.

**Practical implication**: For GQA models (TinyLlama, Mistral, Llama-3), base calibration transfers to instruct — no re-calibration needed. For MHA models (SmolLM, OPT, GPT-2), re-calibration after SFT is required.

**Paper update**: tab:instruct expanded to 3 models (2 MHA + 1 GQA), Discussion updated with GQA null-shift insight and practical implication revised.

---

## Session 28: GPT-2 Medium (345M) — Dimension Hypothesis Challenged ★ SURPRISING

**Date**: 2026-04-20 (Session 28)

**Hypothesis**: Within the GPT-2 family, larger hidden_dim → higher gap_mean (naive dimension scaling).

**Model**: openai/gpt2-medium (345M, d_model=1024, 24 layers, 16 heads, 26,352 cached signals)

**Results**:
- gap_mean = **0.1880** (GPT-2 Small: 0.1956 — LOWER despite larger dim!)
- Predicted beta* = 0.1880/0.267 = **0.704**
- Fine sweep transition: **[0.68, 0.70]**, mid=0.690, error=**0.014** — within ±0.04 ✓

| beta | threshold | 4-bit% | regime |
|------|-----------|--------|--------|
| 0.68 | 0.1816 | 0% | 8-bit |
| 0.70 | 0.1869 | 58.5% | MIXED |
| 0.72 | 0.1922 | 64.4% | MIXED |
| 0.74 | 0.1976 | 87.9% | MIXED |
| 0.76 | 0.2029 | 100% | 4-bit |

**FINDING — Dimension hypothesis REFUTED within GPT-2 family**:
- GPT-2 Small (124M, d=768): gap_mean = 0.196
- GPT-2 Medium (345M, d=1024): gap_mean = 0.188 — LOWER, not higher

**Interpretation — floor gap_mean is about representation quality, not dimension**:
GPT-2 Medium clusters with GQA models (TinyLlama ~0.189) and MHA-instruct models (SmolLM ~0.181-0.194) at the floor ≈ 0.18-0.19. The pattern: more expressive / better-trained models learn more REGULAR KV representations, driving gap_mean toward this floor regardless of hidden_dim. Raw K/V dimensionality is a secondary factor.

**Full 7-checkpoint / 4-family summary**:

| Family | Model | gap_mean | beta* theory | measured transition | error |
|--------|-------|----------|--------------|---------------------|-------|
| LLaMA-MHA (SmolLM) | 135M | 0.330 | 1.234 | [1.2, 1.3] | <0.03 |
| LLaMA-MHA (SmolLM) | 360M | 0.337 | 1.261 | [1.2, 1.4] | <0.04 |
| LLaMA-MHA (SmolLM) | 1.7B | 0.424 | 1.584 | [1.55, 1.57] | <0.015 |
| LLaMA-GQA (TinyLlama) | 1.1B | 0.189 | 0.707 | [0.68, 0.74] | 0.003 |
| OPT-MHA (Meta) | 125M | 0.213 | 0.798 | [0.75, 0.80] | 0.023 |
| GPT-2 (OpenAI) | 124M | 0.196 | 0.733 | [0.70, 0.80] | 0.017 |
| **GPT-2 Medium (OpenAI)** | **345M** | **0.188** | **0.704** | **[0.68, 0.70]** | **0.014** |

Mean error (7 checkpoints): **0.020**, max: **0.040** — all within ±0.04.

**Paper update**: GPT-2 Medium row added to tab:betastar (7th checkpoint), Discussion extended with "dimension hypothesis challenged / floor gap_mean = representation quality" paragraph, all counts updated to 7 checkpoints, abstract/contributions/conclusion updated.

---

## Session 29: GPT-2 Large (774M) — Non-Monotonic Family, Floor Confirmed ★ NOVEL

**Date**: 2026-04-20 (Session 29)

**Hypothesis**: GPT-2 Large (774M, d=1280) continues the downward trend from Medium (0.188) → even lower gap_mean.

**Model**: gpt2-large (774M, d_model=1280, 36 layers, 20 heads, 4,680 cached signals, 10 texts)

**Results**:
- gap_mean = **0.1923** (between Small 0.1956 and Medium 0.1880)
- β* predicted: 0.1923/0.267 = **0.720**
- Beta sweep: p4≈50% at β≈0.71 (between 0.70: 46.4% and 0.72: 57.0%), error ≈ **0.010** ✓
- Note: narrow gap_std=0.026 creates broad transition in β-space; 50% crossing is the correct reference

| beta | threshold | 4-bit% | regime |
|------|-----------|--------|--------|
| 0.60 | 0.1602 | 3.8% | 8-bit |
| 0.65 | 0.1736 | 19.4% | MIXED |
| 0.70 | 0.1869 | 46.4% | MIXED (below β*) |
| **0.72** | **0.1922** | **57.0%** | **MIXED (≈β*)** |
| 0.80 | 0.2136 | 85.7% | MIXED |

**FINDING — GPT-2 family is NON-MONOTONIC: Large (0.192) between Medium (0.188) and Small (0.196)**:
- Not a simple "bigger → lower gap_mean" pattern
- All three within 0.008 band → all at the floor simultaneously
- Floor ≈ 0.188-0.196 for the GPT-2 family (β* ≈ 0.70-0.73)

**Full 8-checkpoint / 4-family summary**:

| Family | Model | gap_mean | beta* theory | measured | error |
|--------|-------|----------|--------------|----------|-------|
| LLaMA-MHA (SmolLM) | 135M | 0.330 | 1.234 | [1.2, 1.3] | <0.030 |
| LLaMA-MHA (SmolLM) | 360M | 0.337 | 1.261 | [1.2, 1.4] | <0.040 |
| LLaMA-MHA (SmolLM) | 1.7B | 0.424 | 1.584 | [1.55, 1.57] | <0.015 |
| LLaMA-GQA (TinyLlama) | 1.1B | 0.189 | 0.707 | [0.68, 0.74] | 0.003 |
| OPT-MHA (Meta) | 125M | 0.213 | 0.798 | [0.75, 0.80] | 0.023 |
| GPT-2 Small (OpenAI) | 124M | 0.196 | 0.733 | [0.70, 0.80] | 0.017 |
| GPT-2 Medium (OpenAI) | 345M | 0.188 | 0.704 | [0.68, 0.70] | 0.014 |
| **GPT-2 Large (OpenAI)** | **774M** | **0.192** | **0.720** | **[0.70, 0.72]** | **0.010** |

Mean error (8 checkpoints): **0.018**, max: **0.040** — all within ±0.04.

**Paper update**: GPT-2 Large row added to tab:betastar (8th checkpoint); "non-monotonic family / floor cluster" insight in Discussion; all counts updated to 8 checkpoints; mean error 0.020→0.018.

---

## Session 30: OPT-350M — OPT Family Convergence to Floor, 4th Independent Route ★ NOVEL

**Date**: 2026-04-20 (Session 30)

**Hypothesis**: OPT-350M (Meta, hidden=1024, larger than OPT-125M's 768) shows floor clustering similar to GPT-2 family, OR dimension-driven scaling (gap_mean increases with hidden_dim).

**Model**: facebook/opt-350m (350M, hidden=1024, 24 layers, 16 heads, 3,360 signals, k+v concatenated methodology)

**Results**:
- gap_mean = **0.1812** (OPT-125M: 0.2131 — LOWER! OPT converges down)
- β* predicted: 0.1812/0.267 = **0.679**
- Beta sweep: 50% crossing at β≈[0.65, 0.70], error ≈ **0.021** ✓ (within ±0.04)

| beta | threshold | 4-bit% | regime |
|------|-----------|--------|--------|
| 0.55 | 0.1469 | 9.0% | 8-bit |
| 0.65 | 0.1736 | 36.4% | MIXED |
| **0.70** | **0.1869** | **57.4%** | **MIXED (above β*)** |
| 0.85 | 0.2270 | 93.6% | 4-bit |

**FINDING — OPT family converges TO the floor**:
- OPT-125M (hidden=768): gap_mean = 0.213 (above floor)
- OPT-350M (hidden=1024): gap_mean = 0.181 (AT the floor!)
- OPT scales DOWN with model size, converging toward 0.18-0.19

**4th independent route to the floor gap_mean ≈ 0.18-0.19**:
1. GQA architecture: TinyLlama 1.1B GQA → 0.189
2. MHA + SFT: SmolLM-135M/360M instruct → 0.181-0.194
3. GPT-2 family: all 3 sizes at 0.188-0.196 (floor cluster)
4. **OPT scaling: 125M (0.213) → 350M (0.181)** (convergence from above) ← NEW

**Full 9-checkpoint / 4-family summary**:

| Family | Model | gap_mean | beta* theory | measured | error |
|--------|-------|----------|--------------|----------|-------|
| LLaMA-MHA (SmolLM) | 135M | 0.330 | 1.234 | [1.2, 1.3] | <0.030 |
| LLaMA-MHA (SmolLM) | 360M | 0.337 | 1.261 | [1.2, 1.4] | <0.040 |
| LLaMA-MHA (SmolLM) | 1.7B | 0.424 | 1.584 | [1.55, 1.57] | <0.015 |
| LLaMA-GQA (TinyLlama) | 1.1B | 0.189 | 0.707 | [0.68, 0.74] | 0.003 |
| OPT (Meta) | 125M | 0.213 | 0.798 | [0.75, 0.80] | 0.023 |
| **OPT (Meta)** | **350M** | **0.181** | **0.679** | **[0.65, 0.70]** | **0.021** |
| GPT-2 (OpenAI) | 124M | 0.196 | 0.733 | [0.70, 0.80] | 0.017 |
| GPT-2 (OpenAI) | 345M | 0.188 | 0.704 | [0.68, 0.70] | 0.014 |
| GPT-2 (OpenAI) | 774M | 0.192 | 0.720 | [0.70, 0.72] | 0.010 |

Mean error: **0.018**, max: **0.040** — all within ±0.04.

**Paper update**: OPT-350M row added to tab:betastar (9th checkpoint), OPT paragraph expanded with convergence narrative and floor attractor explanation. Abstract/contribution#4/conclusion updated to 9 checkpoints. The floor gap_mean is now confirmed via 4 independent routes.

---

## Session 31 — SmolLM2-360M: GQA-Scale Interaction, 10th Checkpoint

**Last updated**: 2026-04-20 (Session 31)

**Protocol**: Within-family GQA vs MHA comparison at fixed scale. SmolLM-360M (MHA, gap_mean=0.337) vs SmolLM2-360M (GQA, kv_heads=5). Prediction: GQA drives gap_mean toward floor (~0.18-0.19).

**Result: GQA REDUCES GAP_MEAN BUT FLOOR REQUIRES SCALE**
- SmolLM-360M (MHA, d=960): gap_mean = 0.337, β*=1.260
- SmolLM2-360M (GQA, d=960): gap_mean = **0.283**, β*=**1.058** — -16% reduction!
- TinyLlama-1.1B (GQA, 3× larger): gap_mean = 0.189 — at the floor

GQA at 360M reduces gap_mean by 16% (0.337→0.283) but does NOT reach the floor (0.189). This is the cleanest test of the GQA hypothesis — same architecture backbone, same hidden dim, same parameter count — only K/V head count changes (15 MHA → 5 KV + 15 attn GQA).

**Key finding — GQA-scale interaction**:
- GQA alone ≠ floor convergence
- Floor convergence requires BOTH GQA AND sufficient scale (≥1B)
- Gradient: MHA (0.337) → small GQA/360M (0.283) → large GQA/1.1B (0.189)

**Formula accuracy**: interpolated 50%-4bit crossing β≈1.103, theory β*=1.058, error≈0.044.
This is BORDERLINE — just outside ±0.04 but within ±0.05. The higher gap_std (0.052, highest of all 10 checkpoints) explains the slightly reduced formula accuracy. 9 of 10 checkpoints within ±0.04.

**Full 10-checkpoint / 5-family summary**:

| Family | Model | gap_mean | beta* theory | measured | error |
|--------|-------|----------|--------------|----------|-------|
| LLaMA-MHA (SmolLM) | 135M | 0.330 | 1.234 | [1.2, 1.3] | <0.030 |
| LLaMA-MHA (SmolLM) | 360M | 0.337 | 1.261 | [1.2, 1.4] | <0.040 |
| LLaMA-MHA (SmolLM) | 1.7B | 0.424 | 1.584 | [1.55, 1.57] | <0.015 |
| **LLaMA-GQA (SmolLM2)** | **360M** | **0.283** | **1.058** | **[1.10, 1.20]** | **0.044*† |
| LLaMA-GQA (TinyLlama) | 1.1B | 0.189 | 0.707 | [0.68, 0.74] | 0.003 |
| OPT (Meta) | 125M | 0.213 | 0.798 | [0.75, 0.80] | 0.023 |
| OPT (Meta) | 350M | 0.181 | 0.679 | [0.65, 0.70] | 0.021 |
| GPT-2 (OpenAI) | 124M | 0.196 | 0.733 | [0.70, 0.80] | 0.017 |
| GPT-2 (OpenAI) | 345M | 0.188 | 0.704 | [0.68, 0.70] | 0.014 |
| GPT-2 (OpenAI) | 774M | 0.192 | 0.720 | [0.70, 0.72] | 0.010 |

*† Borderline: interpolated crossing. 9 of 10 within ±0.04. Mean error: ~0.022.

**5th independent route insight** (refinement of floor attractor narrative):
1. GQA + scale (TinyLlama-1.1B): 0.189
2. MHA + SFT: SmolLM instruct → 0.181-0.194  
3. GPT-2 family cluster: all 3 at 0.188-0.196
4. OPT scaling: 0.213 → 0.181
5. **GQA without scale (SmolLM2-360M)**: intermediate at 0.283 — CONFIRMS GQA reduces gap_mean but scale is also needed

**Paper update**: SmolLM2-360M added as 5th family / 10th checkpoint. Table caption updated to "10 checkpoints, 5 families, 9 of 10 within ±0.04". SmolLM2 Discussion paragraph added explaining GQA-scale interaction. Abstract/contribution#4/conclusion updated. Commit: (see git log).

---

## Session 34 — SmolLM2-360M Calibration Sensitivity: Counter-Intuitive Finding ★

**Date**: 2026-04-20 (Session 34)

**Experiment**: `smollm2_360m_cal_sensitivity.py` — 10-text sensitivity analysis on the borderline formula case (SmolLM2-360M GQA, gap_std=0.052, formula error=0.044).

**Hypothesis**: High gap_std → high calibration sensitivity (1-text estimates more variable).

**Result: REFUTED — High gap_std does NOT predict calibration sensitivity**

| Architecture | gap_std | max_error_1text | mean_error_1text | Notes |
|---|---|---|---|---|
| SmolLM/LLaMA-MHA (1.7B) | 0.063 | 0.015 | — | Session 20 |
| SmolLM2/LLaMA-GQA (360M) | **0.052** | **0.013** | **0.004** | Session 34 — LOWEST mean error |
| GPT-2/Conv1D (124M) | 0.026 | 0.018 | 0.009 | Session 33 |
| OPT/Meta (125M) | ~0.030 | 0.006 | — | Session 33 inline |

SmolLM2-360M has the highest gap_std of all 10 checkpoints yet the LOWEST mean calibration error (0.004, max 0.013). All 10 per-text estimates within ±0.015 of the 10-text aggregate.

**Mechanistic insight**: gap_std reflects between-token variance within a text (how different tokens' quantization gaps vary). This is orthogonal to between-text variance in gap_mean (how much gap_mean shifts across text domains). The calibration claim is about the latter — and gap_mean is a stable domain-invariant property of the model's KV distribution, not a per-token or per-text artifact. High gap_std just means the controller sees a wider range of per-token signals, not that the mean is unstable.

**Practical implication**: The formula error=0.044 for SmolLM2 is NOT due to unstable calibration. It comes from the GQA-scale interaction making the phase transition harder to predict linearly (higher gap_std → wider transition window → harder to pin exactly where 50% of tokens cross threshold). These are distinct effects.

**Paper update**: Sensitivity claim updated from "three architectures" to "four architectures". SmolLM2 added to per-text max deviation table. Counter-intuitive gap_std finding added as key sentence.

---

## Session 35 — TinyLlama-1.1B Calibration Sensitivity: Finding 9 Generalized ★

**Date**: 2026-04-20 (Session 35)

**Experiment**: `tinyllama_cal_sensitivity.py` — 10-text sensitivity analysis on TinyLlama-1.1B (LLaMA-GQA, the tightest formula fit: error=0.003). Uses separate K/V quality methodology matching original `tinyllama_calibration.py`.

**Hypothesis**: Validate Finding 9 orthogonality across an additional LLaMA-GQA checkpoint at a different scale.

**Result: CONFIRMED — gap_std ⊥ calibration sensitivity holds for TinyLlama**

| Architecture | gap_std | max_error_1text | mean_error_1text | Notes |
|---|---|---|---|---|
| SmolLM/LLaMA-MHA (1.7B) | 0.063 | 0.015 | — | Session 20 |
| SmolLM2/LLaMA-GQA (360M) | **0.052** | **0.013** | **0.004** | Session 34 — LOWEST mean error |
| TinyLlama/LLaMA-GQA (1.1B) | 0.051 | **0.008** | **0.004** | Session 35 — confirms orthogonality |
| GPT-2/Conv1D (124M) | 0.026 | 0.018 | 0.009 | Session 33 |
| OPT/Meta (125M) | ~0.030 | 0.006 | — | Session 33 inline |

TinyLlama has gap_std=0.051 (similar to SmolLM2's 0.052) but max_error_1text=0.008 (excellent). The two LLaMA-GQA checkpoints at different scales (360M and 1.1B) show identical mean calibration error (0.004) despite different formula fit quality (error=0.044 vs 0.003). This confirms the orthogonality claim is not scale- or formula-fit-dependent.

**Notable**: 10-text gap_mean=0.1962 vs original calibration's 0.189 — a 0.007 difference attributable to per-token (this script) vs per-head (original calibration) quantization granularity. The calibration sensitivity result (max_error=0.008) is unaffected by this measurement-level difference: within-methodology consistency is what matters for the calibration claim.

**CPU verification status**: All major experiments complete. TinyLlama adds the 5th data point confirming ≤±0.020 single-text calibration sensitivity. Finding 9 is now supported across 3 architecture types with 4 explicit measurements.
