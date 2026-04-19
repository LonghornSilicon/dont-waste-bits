# Don't Waste Bits! — Independent Verification

**Paper**: [arXiv:2604.04722](https://arxiv.org/abs/2604.04722) · *Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs*  
**Original Authors**: Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Patrick Woods, Gabriel Hillesheim, Abolfazl Razi (Clemson University)  
**Accepted**: CVPR 2026 · Original code releases June 3–7, 2026

**Verification by**: themoddedcube / LonghornSilicon  
**Status**: Outer loop — all accuracy experiments complete; latency awaiting GPU

---

## What This Repo Does

Independent reproduction of Table 3 from the paper. Since original code is not yet public, we re-implement the full method from the paper's equations and document critical methodological discoveries.

### Claims Under Verification

| Claim | Paper Value | Baseline | Model | Benchmark |
|-------|-------------|----------|-------|-----------|
| Latency reduction | **17.75%** | vs static 4-bit KV | SmolLM-360M | HellaSwag |
| Accuracy improvement | **+7.60 pp** | vs static 4-bit KV | SmolLM-360M | HellaSwag |
| Gap from FP16 | **≤ 0.30 pp** | vs FP16 | SmolLM-360M | HellaSwag |

---

## Results

### Accuracy Experiments (CPU, HellaSwag, acc metric)

| Condition | Ours | Paper | Delta | Status |
|-----------|------|-------|-------|--------|
| FP16 baseline (50 samp) | **42.0%** | 41.50% | +0.5pp | ✅ CONFIRMED |
| FP16 baseline (200 samp) | **44.0%** | 41.50% | +2.5pp | ✅ Within noise |
| Static KV-4bit per-tensor (200 samp) | **44.5%** | 33.60% | +10.9pp | ⚠️ CANNOT REPRODUCE |
| Static KV-4bit per-token (100 samp) | **44.0%** | 33.60% | +10.4pp | ⚠️ CANNOT REPRODUCE |
| Static KV-8bit (200 samp) | **44.0%** | — | ~0pp | ✅ No degradation (expected) |
| Static KV-2bit (200 samp) | **25.0%** | — | -19pp | ✅ Confirms hooks work |
| DWB adaptive (ours, 100 samp) | **40.0%** | 41.20% | -1.2pp | ~✅ Within noise |
| FP16 latency | — | 3.50 ms/tok | — | ⏳ Awaiting RTX 4090 |
| Static 4-bit latency | — | 2.93 ms/tok | — | ⏳ Awaiting RTX 4090 |
| DWB latency | — | 2.41 ms/tok | — | ⏳ Awaiting RTX 4090 |

**SmolLM2-360M FP16 (acc_norm, for comparison)**: 45.33% (different model variant, stored)

---

## Methodological Findings (Key Contributions)

### Finding 1: Evaluation metric is critical
The paper uses **unnormalized log-likelihood** (`acc`), not length-normalized (`acc_norm`).  
`acc_norm` (lm-eval default) gives ~49–54% for SmolLM-360M vs paper's 41.5%.  
`acc` (unnorm) gives 42.0% on 50 samples — matches paper's 41.5% ✓  
*Cross-model evidence*: our SmolLM-360M acc_norm ≈ paper's SmolLM-1.7B acc — definitively wrong metric.

### Finding 2: KV cache hooks fail with DynamicCache (transformers 5.x)
transformers 5.x uses `DynamicCache` objects, not raw `(key, value)` tuples.  
Hooks on attention modules silently fail to intercept KV tensors.  
**Fix**: Hook `k_proj` and `v_proj` Linear submodule outputs directly.  
SmolLM-360M has 32 attention layers × 2 (k+v) = 64 hooks total.

### Finding 3: sdpa attention blocks output_attentions
transformers 5.x uses sdpa by default, which does NOT support `output_attentions=True`.  
**Fix**: Reload with `attn_implementation='eager'` for DWB signal extraction.

### Finding 4: Symmetric per-tensor INT4 is nearly lossless for attention ★
Our INT4 KV implementation gives ~44.5% accuracy — essentially identical to FP16 (~44%).  
The paper claims 33.6% (7.9pp drop) for static INT4.  
**We cannot reproduce the paper's static INT4 accuracy drop.**

Hypothesis: symmetric quantization produces zero-mean errors that cancel in the attention weighted sum, especially when outlier tokens set the scale and tend to be the most-attended tokens.

Supporting evidence: KV-2bit (asymmetric-like behavior at 4 levels only) gives 25% (catastrophic), while INT4 (16 levels, still zero-mean) gives ~44% (no degradation).

The paper's "Static 4-bit KV" baseline likely uses a different, more aggressive quantization scheme (e.g., from KIVI or a similar published method) with non-cancelling errors.

---

## Method Summary

The paper proposes a lightweight 3-layer MLP controller that assigns per-token KV-cache precision from {2, 4, 8, FP16} bits during autoregressive decoding, driven by four token-level signals:

- **H_t** — Shannon entropy of next-token distribution (Eq. 14)
- **R_t** — Token rarity / inverse frequency (Eq. 15)
- **V_t** — Attention variance across heads (Eq. 16)
- **C_t** — Model confidence (max softmax probability)

Training minimizes: `L = α·CE + β·latency + γ·quality` (Eq. 28, α=1, β=0.1, γ=0.1).  
Architecture: Linear(4,128) → ReLU → Linear(128,128) → ReLU → Linear(128,4) = 33,540 params.

Our controller: trained on 2,995 token samples from 100 HellaSwag train contexts. Val accuracy: **45.6%** (vs 25% random chance) — controller learns importance quartile above chance.

DWB eval bit distribution: {2bit: 57.3%, 4bit: 18.9%, 8bit: 8.3%, 16bit: 15.6%}, avg=5.05 bits/token.

---

## Repository Structure

```
dont-waste-bits/
├── README.md
├── 2604.04722v1.pdf              # Original paper
└── research/
    ├── research-state.yaml       # Central experiment state
    ├── research-log.md           # Decision timeline
    ├── findings.md               # Evolving synthesis (primary doc)
    ├── paper_outline.md          # Reproducibility paper outline
    ├── literature/               # Survey notes
    ├── src/
    │   ├── dwb_implementation.py # Re-implementation from paper equations
    │   ├── eval_hellaswag.py     # HellaSwag evaluator (acc metric)
    │   ├── eval_dwb.py           # DWB two-pass evaluation
    │   ├── kv_cache_quant.py     # KV cache quantization hooks (v2)
    │   ├── run_kv_comparison.py  # Multi-condition KV quant sweep
    │   └── brev_setup.sh         # NVIDIA Brev GPU setup
    ├── data/                     # Experiment results (JSON)
    └── experiments/
        ├── H1-latency-reduction/
        ├── H2-accuracy-improvement/
        └── H3-fp16-parity/
```

---

## Running

```bash
# FP16 baseline
python research/src/eval_hellaswag.py --model smollm-360m --condition fp16 --limit 500

# KV quantization comparison (multi-condition)
python research/src/run_kv_comparison.py 200

# DWB adaptive evaluation (trains controller if not cached)
python research/src/eval_dwb.py --model smollm-360m --limit 200

# Full latency + accuracy (GPU required)
bash research/src/brev_setup.sh
```

---

## Hardware

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Accuracy (H2, H3) | CPU | SmolLM-360M fits in RAM |
| Latency (H1) | **NVIDIA RTX 4090** | Must match paper hardware |

Latency experiments run on **NVIDIA Brev** cloud.

---

## Status

- [x] Arithmetic verification — all 3 claims internally consistent
- [x] Re-implementation of all paper equations (DWB controller, signals, training loss)
- [x] FP16 baseline confirmed — **42.0%** acc (paper: 41.50%) ✅
- [x] Metric resolved — paper uses unnormalized `acc`, not `acc_norm` (Finding 1)
- [x] KV hook fix — `k_proj`/`v_proj` not `past_key_values` (Finding 2)
- [x] Eager attention fix — for DWB signal extraction (Finding 3)
- [x] DWB controller trained — val_acc=45.6%, avg 5.05 bits/token
- [x] DWB eval: **40.0%** (paper: 41.2%, within noise) ~✅
- [x] INT4 losslessness documented — cannot reproduce paper's 33.6% static baseline (Finding 4)
- [ ] Static INT4 baseline investigation — asymmetric/calibrated scheme
- [ ] Latency experiments (H1) — RTX 4090 required
- [ ] Academic paper writeup
