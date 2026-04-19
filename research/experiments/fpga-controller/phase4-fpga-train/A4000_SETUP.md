# Running Phase 4 on NVIDIA A4000

## What This Runs

Binary {4,8}-bit FPGA-aware controller on **SmolLM-1.7B** — the model where INT4 is
genuinely lossy (eff_residual=12.4%). This is the key experiment: at 360M INT4 is lossless
so there's nothing to gain from dynamic allocation, but at 1.7B there's a real 7.9pp accuracy
gap between 4-bit (41.1%) and FP16 (49.0%) that the controller should learn to close.

**Expected result**: controller learns to route high-attention-score tokens to 8-bit and
low-importance tokens to 4-bit, achieving ~48-49% accuracy (matching paper) at avg_bits < 8.0
with better FPGA cost than the paper's CPU-optimized allocation.

## Setup

```bash
# Clone the repo
git clone https://github.com/LonghornSilicon/dont-waste-bits.git
cd dont-waste-bits
git checkout fpga-controller

# Install dependencies
pip install torch transformers datasets

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Run

```bash
cd research/experiments/fpga-controller/phase4-fpga-train/code

# Full run (500 train texts, 500 eval samples, ~30-60 min on A4000)
python run_phase4_1b7_gpu.py --device cuda --train-texts 500 --eval-samples 500

# Quick test first (100 train texts, 200 eval samples, ~10-15 min)
python run_phase4_1b7_gpu.py --device cuda --train-texts 100 --eval-samples 200
```

## What It Does

1. **Stage 1** (~5-10 min): Extract KV cache signals from 1.7B on GPU, cache to disk
   - Saves to `results/phase4_1b7_kv_cache.pt`
   - Rerun with `--skip-cache` to re-extract

2. **Stage 2** (~5 min): Train binary MLP controller on CPU (tiny model, ~10k params)
   - Beta sweep {0.3, 0.5, 0.7} to find target avg_bits ~6.0
   - Full training with best beta

3. **Stage 3** (~15-30 min): Evaluate on HellaSwag 500 samples on GPU
   - Saves result JSON to `results/phase4_1b7_results_YYYYMMDD_HHMM.json`

## Expected Output

```
Phase 4 Binary FPGA 1.7B Result:
  Accuracy:      48.x%  (paper DWB: 48.6%)
  avg_bits:      x.xx   (lower = better FPGA throughput)
  FPGA speedup:  x.xx x vs FP16
  Bit dist:      {'4': xx.x%, '8': xx.x%}
```

## Key Baselines to Compare Against

| Condition | Accuracy | avg_bits | FPGA cost | FPGA speedup |
|---|---|---|---|---|
| FP16 | 49.0% | 16.0 | 1.010 | 1.00x |
| Static 4-bit | 41.1% | 4.0 | 0.290 | 3.48x |
| Paper DWB | 48.6% | 5.05 | ~0.41* | ~2.44x* |
| **Our target** | **≥48%** | **≤8.0** | **≤0.56** | **≥1.80x** |

*Paper DWB FPGA cost is an estimate — their 2-bit tokens have same BRAM cost as 4-bit,
so their effective FPGA cost is higher than their avg_bits suggests.

## After Running

Push results back to the repo:
```bash
git add research/experiments/fpga-controller/phase4-fpga-train/results/
git commit -m "research(results): phase4 1.7B binary FPGA controller — GPU run"
git push origin fpga-controller
```
