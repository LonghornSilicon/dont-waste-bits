# Don't Waste Bits! — Independent Verification

**Paper**: [arXiv:2604.04722](https://arxiv.org/abs/2604.04722) · *Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs*  
**Original Authors**: Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Patrick Woods, Gabriel Hillesheim, Abolfazl Razi (Clemson University)  
**Accepted**: CVPR 2026 · Original code releases June 3–7, 2026

**Verification by**: themoddedcube / LonghornSilicon  
**Status**: Inner loop — baselines confirmed, DWB evaluation running

---

## What This Repo Does

Independent reproduction of Table 3 from the paper. Since original code is not yet public, we re-implement the full method from the paper's equations.

### Claims Under Verification

| Claim | Paper Value | Baseline | Model | Benchmark |
|-------|-------------|----------|-------|-----------|
| Latency reduction | **17.75%** | vs static 4-bit KV | SmolLM-360M | HellaSwag |
| Accuracy improvement | **+7.60 pp** | vs static 4-bit KV | SmolLM-360M | HellaSwag |
| Gap from FP16 | **≤ 0.30 pp** | vs FP16 | SmolLM-360M | HellaSwag |

---

## Current Results

### Metric clarification (key finding)
The paper uses **unnormalized log-likelihood** (`acc`), not length-normalized (`acc_norm`).  
Evidence: our `acc_norm` gives SmolLM-360M ~54% while paper reports 41.5% — same gap as between SmolLM-360M and SmolLM-1.7B in the paper. Direct test: `acc (unnorm)` on 50 val samples = **42.0%** vs paper's **41.5%** ✓

### Experiment Trajectory

| Run | Condition | Our Result | Paper Target | Delta | Status |
|-----|-----------|-----------|--------------|-------|--------|
| 00 | Arithmetic check | Consistent | — | — | ✅ DONE |
| 01a | SmolLM2-360M FP16 (acc_norm) | 45.33% | 41.50% | +3.83pp | Stored (wrong variant) |
| 01c | SmolLM-360M FP16 (acc, 50 samp) | **42.0%** | 41.50% | **+0.5pp** | ✅ BASELINE CONFIRMED |
| 02 | Static KV-4bit (acc, unnorm) | running | 33.60% | — | 🔄 RUNNING |
| 03 | DWB adaptive (our re-impl) | queued | 41.20% | — | ⏳ QUEUED |
| 04–06 | Latency (FP16/4-bit/DWB) | — | 3.50/2.93/2.41 ms/tok | — | ⏳ Awaiting Brev RTX 4090 |

---

## Method Summary

The paper proposes a lightweight MLP controller that assigns per-token KV-cache precision from {2, 4, 8, FP16} bits during autoregressive decoding, driven by four token-level signals:

- **H_t** — Shannon entropy of next-token distribution (Eq. 14)
- **R_t** — Token rarity / inverse frequency (Eq. 15)
- **V_t** — Attention variance across heads (Eq. 16)
- **C_t** — Model confidence (max softmax probability)

Training minimizes combined loss (Eq. 28): cross-entropy + expected latency + quality penalty.

---

## Implementation Notes

### KV Cache Quantization (Critical)
The paper quantizes **KV cache entries only**, not model weights. For transformers 5.x:
- **Wrong**: hook `past_key_values` output (returns `DynamicCache` object — hooks silently fail)
- **Correct**: hook `k_proj` and `v_proj` Linear submodule outputs directly (`kv_cache_quant.py`)

### Evaluation Metric (Critical)
Use **unnormalized** log-likelihood (`acc`), not `acc_norm`:
```python
# Correct (matches paper ~41.5%):
score = log_probs[range(len(cont)), cont_ids].sum().item()  # raw sum

# Wrong (gives ~54% — not paper's metric):
score = score / len(cont_ids)  # per-token average
```

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
    ├── literature/               # Survey notes
    ├── src/
    │   ├── dwb_implementation.py # Re-implementation from paper equations
    │   ├── eval_hellaswag.py     # HellaSwag evaluator (acc metric)
    │   ├── eval_dwb.py           # DWB two-pass evaluation
    │   ├── kv_cache_quant.py     # KV cache quantization hooks (v2)
    │   ├── run_baselines.py      # Baseline sweep script
    │   └── brev_setup.sh         # NVIDIA Brev GPU setup
    ├── data/                     # Experiment results (JSON)
    ├── experiments/
    │   ├── H1-latency-reduction/
    │   ├── H2-accuracy-improvement/
    │   └── H3-fp16-parity/
    └── to_human/                 # Progress reports (supplementary)
```

---

## Running

```bash
# FP16 baseline (CPU works)
python research/src/eval_hellaswag.py --model smollm-360m --condition fp16 --limit 500

# Static KV-4bit baseline
python research/src/eval_hellaswag.py --model smollm-360m --condition static4bit --limit 500

# DWB adaptive evaluation
python research/src/eval_dwb.py --model smollm-360m --limit 200 --train_samples 100

# Full latency + accuracy (GPU required)
python research/src/run_baselines.py --model smollm-360m --task hellaswag
```

---

## Hardware

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Accuracy (H2, H3) | CPU | SmolLM-360M fits in RAM |
| Latency (H1) | **NVIDIA RTX 4090** | Must match paper hardware |

GPU experiments run on **NVIDIA Brev** cloud.

---

## Status

- [x] Arithmetic verification — all 3 claims internally consistent with Table 3
- [x] Re-implementation of DWB method from paper equations
- [x] FP16 baseline confirmed — **42.0%** acc (paper: 41.50%) ✅
- [x] Metric resolved — paper uses unnormalized `acc`, not `acc_norm`
- [x] KV hooks fixed — hook `k_proj`/`v_proj`, not `past_key_values` (DynamicCache issue)
- [ ] Static 4-bit KV baseline — **running**
- [ ] DWB adaptive accuracy — queued
- [ ] Latency experiments — needs Brev RTX 4090

---

## SmolLM2 Comparison (for paper table)

SmolLM2-360M (improved successor model) FP16 acc_norm = **45.33%** on 300 val samples.  
The paper uses original SmolLM-360M. SmolLM2 result stored for comparison table.
