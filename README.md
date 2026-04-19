# Don't Waste Bits! — Independent Verification

**Paper**: [arXiv:2604.04722](https://arxiv.org/abs/2604.04722) · *Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs*  
**Original Authors**: Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Patrick Woods, Gabriel Hillesheim, Abolfazl Razi (Clemson University)  
**Accepted**: CVPR 2026 (original code releases June 3–7, 2026)

---

## What This Repo Does

Independent reproduction of the paper's key claims from Table 3:

| Claim | Paper Value | Model | Benchmark |
|-------|------------|-------|-----------|
| Latency reduction vs static 4-bit KV | **17.75%** | SmolLM-360M | HellaSwag |
| Accuracy improvement vs static 4-bit KV | **+7.60 pp** | SmolLM-360M | HellaSwag |
| Gap from FP16 inference | **≤ 0.30 pp** | SmolLM-360M | HellaSwag |

Since the original code is not yet public, we re-implement the full method from the paper's equations and verify both baselines and the DWB adaptive method independently.

---

## Method Summary

The paper proposes a lightweight MLP controller that assigns per-token KV-cache precision from {2, 4, 8, FP16} during autoregressive decoding. The controller is trained on four token-level signals:

- **H_t** — entropy of the next-token distribution (Eq. 14)
- **R_t** — token rarity / inverse frequency (Eq. 15)
- **V_t** — attention variance across heads (Eq. 16)
- **C_t** — model confidence (max softmax probability)

Training minimizes a combined loss (Eq. 28): cross-entropy + expected latency + quality penalty.

---

## Repository Structure

```
dont-waste-bits/
├── README.md
├── 2604.04722v1.pdf              # Original paper
└── research/
    ├── research-state.yaml       # Central experiment state
    ├── research-log.md           # Decision timeline
    ├── findings.md               # Evolving synthesis
    ├── literature/               # Survey notes
    ├── src/
    │   ├── dwb_implementation.py # Re-implementation from paper equations
    │   ├── eval_hellaswag.py     # Direct HellaSwag evaluator
    │   ├── run_baselines.py      # Baseline sweep script
    │   └── brev_setup.sh         # NVIDIA Brev GPU setup
    ├── data/                     # Experiment results (JSON)
    ├── experiments/
    │   ├── H1-latency-reduction/ # Protocol + results
    │   ├── H2-accuracy-improvement/
    │   └── H3-fp16-parity/
    └── to_human/                 # Progress reports (HTML)
```

---

## Hardware

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Accuracy (H2, H3) | CPU or any GPU | SmolLM fits in RAM |
| Latency (H1) | **NVIDIA RTX 4090** | Must match paper hardware |

GPU experiments run on **NVIDIA Brev** cloud. Paper used RTX 4090 (24 GB).

---

## Running

```bash
# Setup (GPU, on Brev)
bash research/src/brev_setup.sh

# Accuracy verification (CPU works)
python research/src/eval_hellaswag.py --model smollm2-360m --condition fp16 --limit 500
python research/src/eval_hellaswag.py --model smollm2-360m --condition static4bit --limit 500
python research/src/eval_hellaswag.py --model smollm2-360m --condition dwb --limit 500

# Full latency + accuracy (GPU required)
python research/src/run_baselines.py --model smollm-360m --task hellaswag
```

---

## Status

- [x] Arithmetic verification — all 3 claims internally consistent with Table 3
- [x] Re-implementation of DWB method from paper equations
- [ ] FP16 baseline accuracy — **running** (CPU, 300 samples)
- [ ] Static 4-bit baseline accuracy
- [ ] DWB adaptive accuracy
- [ ] Latency experiments (needs Brev RTX 4090)
