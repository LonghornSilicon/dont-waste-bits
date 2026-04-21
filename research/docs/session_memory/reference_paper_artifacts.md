---
name: Paper and experiment artifact layout (dont-waste-bits)
description: Where the paper TeX, experiment code, and results JSONs live under research/.
type: reference
originSessionId: eeed6cc0-c5f4-4b71-988a-03ac1ad53025
---
- Paper (active branch: `fpga-controller`): `research/paper/fpga_controller_paper.tex`
  - Main-table row of interest: `tab:main` (SmolLM-360M + SmolLM-1.7B HellaSwag)
  - Beta-calibration table: `tab:betastar` (10 checkpoints, 5 families)
- Experiment tree: `research/experiments/fpga-controller/`
  - `phase5-benchmark/code/run_phase5_1b7.py` — canonical pattern for binary Gumbel controller at 1.7B (Stage 1 signal extraction, Stage 2 controller training, Stage 3 HellaSwag eval via k_proj/v_proj hooks)
  - `phase5-benchmark/code/beta_calibration_1b7.py` — per-token q4_local / q8_local signal extraction (needed for mixed allocation at 1.7B)
  - `phase5-benchmark/code/reproducibility_test_1b7.py` — multi-seed pattern
  - `phase7-ablation/code/run_phase7c_routing_ablation.py` — random vs controller vs KV-norm (n=200)
  - `phase7-ablation/code/run_phase7d_random_multiseed.py` — 5-seed random at n=500
- Shared KV-quant helpers: `research/src/kv_cache_quant.py` (`quantize_tensor(x, bits)` for per-token bit-width routing)
- Env: A4000 / CUDA 12.8 driver. Torch must be `+cu121`; `torch 2.11.0+cu130` fails with "NVIDIA driver too old". `accelerate` is required alongside `transformers` 5.x when using `device_map="auto"`.
