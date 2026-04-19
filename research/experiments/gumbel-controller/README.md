# Gumbel-Softmax Controller

**Branch:** `gumbel-controller`  
**Status: CONCLUDED — superseded by `fpga-controller`**

This branch answered its research question and was folded into `fpga-controller`.
All active experiments run on `fpga-controller`. Do not open PRs or continue work here.

---

## Why This Branch Is Not Being Used

### Both phases produced identical results: 100% INT4, 41.0% HellaSwag

| Phase | Features | Accuracy | avg_bits | Conclusion |
|---|---|---|---|---|
| 1 | kv_norm, pos_frac | 41.0% | 4.0 | Standard quality scores → 4-bit global minimum |
| 2 | + head entropy, layer depth | 41.0% | 4.0 | Richer features change nothing |

### Root cause 1: dynamic allocation doesn't help at 360M

SmolLM-360M has **eff_residual = 8.1%** — below the losslessness threshold (~10%).
INT4 is already lossless at this scale regardless of which token, layer, or attention
pattern you look at. There is no per-token variation in quantization sensitivity for a
controller to exploit, so every controller correctly learns: assign 4-bit to everything.

This is not a failure — 100% 4-bit is the **correct global optimum** at 360M:
- 2-bit: same FPGA BRAM cost as 4-bit (both use a 4-bit port), worse accuracy → always dominated
- 8/16-bit: 2–4x BRAM cost, negligible accuracy gain at 360M → not worth it

### Root cause 2: pre-computed quality scores cannot produce mixed allocation

The Gumbel-softmax approach uses static quality scores (global accuracy averages per bit
width). Every token sees the same cost function → every token converges to the same global
minimum. True mixed allocation requires **per-token LM loss gradients** — the paper's actual
compound loss `L = α·CE_LM + β·avg_bits` computed through the LM forward pass with quantized
KV. Without the LM in memory during training, the controller has no signal about which
specific tokens need more bits. This requires a GPU with enough VRAM to hold the LM.

### Where the interesting experiment is

The meaningful dynamic allocation regime is **SmolLM-1.7B**, where:
- eff_residual = 12.4% → INT4 is genuinely lossy (−7.9pp vs FP16)
- Quality gap between 4-bit and 8-bit: **0.305** (normalized) vs **0.021** at 360M
- A binary {4,8}-bit controller has real per-token signal to learn from

---

## Active Branch: `fpga-controller`

```bash
git checkout fpga-controller
python research/experiments/fpga-controller/phase4-fpga-train/code/run_phase4_1b7_gpu.py --device cuda
```

See `research/experiments/fpga-controller/phase4-fpga-train/A4000_SETUP.md` for full setup.

---

## Results (archived)

| Condition | Accuracy | avg_bits | FPGA cost | FPGA speedup |
|---|---|---|---|---|
| FP16 (360M) | 42.6% | 16.0 | 1.010 | 1.00x |
| Paper DWB (360M) | 41.2% | 5.05 | 0.414 | 2.44x |
| This branch — Phase 1 | 41.0% | 4.0 | 0.290 | 3.48x |
| This branch — Phase 2 | 41.0% | 4.0 | 0.290 | 3.48x |
