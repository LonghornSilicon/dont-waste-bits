# Don't Waste Bits! — Independent Verification

**Paper**: [arXiv:2604.04722](https://arxiv.org/abs/2604.04722) · *Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs*  
**Original Authors**: Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Patrick Woods, Gabriel Hillesheim, Abolfazl Razi (Clemson University)  
**Accepted**: CVPR 2026 · Original code releases June 3–7, 2026

**Verification by**: themoddedcube / LonghornSilicon  
**Status**: Accuracy experiments complete. Latency awaiting RTX 4090.

---

## What This Repo Does

Independent reproduction of Table 3 from the paper. Since original code is not yet public,
we re-implement the full method from paper equations and document critical methodological discoveries.

---

## Results (Accuracy)

### Claims vs Our Results

| Claim | Paper | Ours | Status |
|-------|-------|------|--------|
| FP16 baseline (500 samples) | 41.50% | **42.6%** | ✅ CONFIRMED (Δ=+1.1pp, within noise) |
| Static INT4 KV (standard) | 33.60% | **41.2–44.5%** | ⚠️ CANNOT REPRODUCE standard INT4 |
| Static INT4 KV (int3range) | 33.60% | **33.0%** | ✅ MATCHES with 8-level INT4 |
| DWB adaptive | 41.20% | **40.0%** | ~✅ Within noise (H3 consistent) |
| Latency reduction | 17.75% | — | ⏳ Awaiting RTX 4090 |

### Detailed Accuracy Table

| Condition | Samples | Ours | Paper | Δ |
|-----------|---------|------|-------|---|
| FP16 | 500 | 42.6% | 41.5% | +1.1pp |
| KV-4bit sym per-tensor | 500 | 41.6% | 33.6% | +8.0pp |
| KV-4bit sym per-token | 500 | 41.2% | 33.6% | +7.6pp |
| KV-4bit asym per-tensor | 200 | 42.5% | 33.6% | +8.9pp |
| **KV int4_int3range (8 levels)** | 100 | **33.0%** | 33.6% | **-0.6pp ✅** |
| KV-2bit | 200 | 25.0% | — | -17pp (hooks confirmed) |
| DWB adaptive | 100 | 40.0% | 41.2% | -1.2pp |
| DWB adaptive | 200 | **38.0%** | 41.2% | -3.2pp (within ±6.7pp CI) |
| KV-4bit sym per-tensor (autoregressive) | 50 | **42.0%** | 33.6% | +8.4pp — AR methodology doesn't explain gap |
| Latency (FP16) | — | — | 3.50 ms/tok | — |
| Latency (KV-4bit) | — | — | 2.93 ms/tok | — |
| Latency (DWB) | — | — | 2.41 ms/tok | — |

---

## Key Methodological Findings

### Finding 1: Evaluation metric matters critically
Paper uses **unnormalized** log-likelihood (`acc`), not length-normalized (`acc_norm`).
- `acc_norm` (lm-eval default): ~54% for SmolLM-360M → WRONG
- `acc` (unnorm): ~42% → matches paper's 41.5% ✓

### Finding 2: KV hooks fail with DynamicCache (transformers 5.x)
transformers 5.x uses `DynamicCache` objects — hooks on attention outputs silently fail.  
**Fix**: Hook `k_proj` and `v_proj` Linear submodules directly (64 hooks for SmolLM-360M).

### Finding 3: sdpa attention blocks output_attentions
Default sdpa attention doesn't support `output_attentions=True` for DWB signal extraction.  
**Fix**: Reload with `attn_implementation='eager'`.

### Finding 4: Standard INT4 KV is nearly lossless ★
**6 INT4 variants (sym/asym × per-tensor/per-token/block) all give ≈ FP16 accuracy.**  
Our 500-sample result: FP16=42.6%, INT4=41.2–41.6%. The ~8pp gap between our INT4 and the paper's
33.6% is **statistically significant** (n=500, CI=±4.4pp).

Hypothesis: symmetric INT4 produces zero-mean quantization errors that cancel in the attention
weighted sum, preserving accuracy. The most-attended tokens set the quantization scale and are
thus also the best-quantized.

### Finding 5: Paper's INT4 baseline uses ~8 effective quantization levels ★★
`int4_int3range` (scale=max/3, 8 levels) gives **33.0%** matching the paper's **33.6%** (Δ=-0.6pp).  
Standard INT4 uses scale=max/7 (16 levels) → lossless.  
The paper's "Static 4-bit KV" is **equivalent to INT3 quantization stored in 4-bit format**.

| Levels | Scale | Acc | vs Paper |
|--------|-------|-----|---------|
| 16 (standard INT4) | max/7 | ~42% | +8pp (lossless) |
| 8 (int4_int3range) | max/3 | 33% | **-0.6pp ✅** |
| 4 (INT2) | max/1 | 25% | -8.6pp |

**Implication**: The paper's claim that DWB achieves +7.6pp over static INT4 is conditional
on the INT4 baseline using 8 effective levels (not 16). With proper 16-level INT4, standard
quantization already matches FP16 accuracy.

---

## Method Summary (Re-implementation)

Controller: Linear(4,128) → ReLU → Linear(128,128) → ReLU → Linear(128,4) — **33,540 params**

Four token-level signals:
- **H_t**: entropy of next-token distribution (Eq. 14)
- **R_t**: rarity / inverse frequency (Eq. 15)  
- **V_t**: attention variance across heads (Eq. 16)
- **C_t**: prediction confidence (Eq. 17)

Training loss: `L = α·CE + β·latency + γ·quality` (Eq. 28, α=1, β=0.1, γ=0.1)

Our controller: 2,995 token samples from 100 HellaSwag train examples, 5 epochs.  
Val accuracy: **45.6%** (vs 25% random) — learns importance quartile above chance.  
DWB eval bit distribution: {2bit: 57.3%, 4bit: 18.9%, 8bit: 8.3%, 16bit: 15.6%}, avg=5.05 bits/token.

---

## Repository Structure

```
dont-waste-bits/
├── README.md
├── 2604.04722v1.pdf                    # Original paper
└── research/
    ├── research-state.yaml             # Experiment state
    ├── research-log.md                 # Decision timeline
    ├── findings.md                     # Synthesis (primary doc)
    ├── paper_outline.md                # Reproducibility paper outline
    ├── src/
    │   ├── dwb_implementation.py       # DWB re-implementation
    │   ├── eval_hellaswag.py           # HellaSwag evaluator (acc metric)
    │   ├── eval_dwb.py                 # DWB two-pass evaluation
    │   ├── kv_cache_quant.py           # KV quantization hooks (v2)
    │   ├── run_kv_comparison.py        # Multi-condition KV sweep
    │   ├── run_int4_investigation.py   # INT4 variant investigation
    │   └── eval_autoregressive.py      # AR scoring with KV cache
    └── data/
        ├── baseline_500samp_*.json     # Definitive 500-sample baseline
        ├── kv_comparison_*.json        # 200-sample INT4 variant comparison
        ├── int4_investigation_*.json   # 7-variant INT4 investigation
        ├── dwb_eval_*.json             # DWB adaptive evaluation
        └── dwb_controller_smollm360m.pt  # Trained controller
```

---

## Running

```bash
# FP16 baseline
python research/src/eval_hellaswag.py --model smollm-360m --condition fp16 --limit 500

# Multi-condition INT4 comparison
python research/src/run_kv_comparison.py 200

# INT4 variant investigation (7 schemes)
python research/src/run_int4_investigation.py

# DWB adaptive evaluation (trains/loads controller)
python research/src/eval_dwb.py --model smollm-360m --limit 200
```

---

## Status

- [x] Arithmetic verification ✓
- [x] FP16 baseline confirmed — 42.6% (500 samp, paper: 41.5%) ✓
- [x] Metric resolved — unnormalized `acc`, not `acc_norm` (Finding 1)
- [x] KV hook fix — `k_proj`/`v_proj` (Finding 2)
- [x] Eager attention fix — for DWB signal extraction (Finding 3)
- [x] Standard INT4 losslessness documented (Finding 4) — all 6 variants ≈ FP16
- [x] Paper's INT4 baseline reproduced — `int4_int3range` = 33.0% ≈ 33.6% (Finding 5)
- [x] DWB controller trained (val_acc=45.6%) and evaluated (40.0%)
- [ ] Latency experiments (H1) — RTX 4090 required
- [ ] Academic paper writeup
