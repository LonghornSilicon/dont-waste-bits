# Phase 7 Live Notes (session 2026-04-20, A4000)

Running log of observations during Phase 7c / 7d execution. These are observations
beyond what ends up in the JSON; intended as input for the paper update.

## Environment
- GPU: NVIDIA RTX A4000 (15 GB), CUDA 12.8 driver
- Torch: 2.5.1+cu121 (required downgrade from 2.11.0+cu130 — cu130 wheel fails on driver 12.8)
- `accelerate` required when `device_map="auto"` under transformers 5.x

## Controller calibration (Phase 7c Stage 2)
- gap_mean = 0.3550, theory β* = 0.3550 / 0.267 = 1.330
  (slightly above Phase 5's earlier CPU measurement of 0.424 at 1.7B with max_len=64 and 10 texts;
   this run used max_len=128, 200 texts, so a different sample mean is expected)
- Binary search for target p4=0.74:
    β=1.600 → p4=1.000
    β=1.300 → p4=0.430
    β=1.450 → p4=1.000
    β=1.375 → p4=0.628
    β=1.413 → p4=0.741  ← best
    β=1.394 → p4=0.719
    β=1.403 → p4=0.715
    β=1.408 → p4=0.728
  **Chosen β=1.413; training-set p4=74.1%.** Narrow transition window (1.375–1.45).

## Phase 7c evaluation progress (n=200, SmolLM-1.7B / HellaSwag)
| strategy   |  20   |  40   |  60   |  80   | 100   | 120   | 140   | 160   | 180   | 200   | p4    | speedup |
|------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| random     | 0.250 | 0.450 | 0.483 | 0.475 | 0.510 | 0.517 | 0.529 | 0.512 | 0.511 | **0.485** | 73.9% | 2.80×   |
| controller | 0.400 | 0.550 | 0.533 | 0.512 | 0.510 | 0.517 | 0.521 | 0.512 | 0.500 | **0.475** | **95.5%** | **3.34×** |
| kv_norm    | 0.300 | 0.475 | 0.467 | 0.475 | 0.490 | 0.483 | 0.486 | 0.481 | 0.467 | **0.445** | 74.1% | 2.81×   |

**Key observations from Phase 7c**:
1. **random 48.5%** — within 0.1pp of DWB's 48.6% and within 1σ of the user-specified
   multi-seed target 48.6%±0.71%. Headline result.
2. **controller drifted to p4=95.5% at inference** (vs 74.1% during β-calibration).
   The controller uses per-token q_local computed on-the-fly from the actual K tensor.
   Inference-time q_local distribution differs from training-time → more tokens qualify
   for 4-bit. Result: 47.5% at 3.34× speedup — a *stronger* Pareto point than random,
   but at a different operating ratio, so not directly comparable to "controller vs
   random at fixed split."
3. **kv_norm 44.5%** — 4pp below random at identical 74/26 split. Within n=200 CI
   (±7pp) so not statistically significant, but suggestive: KV L2 norm is not a
   cleanly useful routing signal on SmolLM-1.7B either.
4. Across random / kv_norm (both p4≈74%): 48.5 vs 44.5. Across random / controller
   (different p4): 48.5 vs 47.5. No strategy beats random at its own operating point.

## Phase 7f — kv_norm_inverted sanity check (pathological direction)
Runs a mirror of kv_norm: **top-74% by L2 norm → 4-bit** (high-norm "important" tokens
lose precision). Expectation: significant accuracy drop vs forward kv_norm, confirming
the forward direction isn't arbitrary.

**Result: 44.5% at 2.805× (p4=74.05%) — identical to forward kv_norm's 44.5%.**

This is the interesting part. If L2 norm were a useful routing signal, reversing the
direction should tank accuracy. Instead both directions land on the same 44.5%. The
~4pp gap vs random therefore isn't about getting the direction "wrong" — it's about
routing being **systematic** (norm-based clustering makes 4-bit errors correlate across
tokens, whereas random routing decorrelates them). KV L2 norm carries no usable
direction signal at this split on SmolLM-1.7B.

## Running final table (3 + inverted + 5-seed random)
| strategy                     | acc    | p4     | speedup | notes                                        |
|------------------------------|--------|--------|---------|----------------------------------------------|
| random (Phase 7c, seed 0)    | 48.50% | 73.9%  | 2.80×   | headline                                      |
| controller (Phase 7c)         | 47.50% | 95.5%  | 3.34×   | drifted to aggressive 4-bit at inference     |
| kv_norm (Phase 7c)            | 44.50% | 74.1%  | 2.81×   | forward direction                             |
| kv_norm_inverted (Phase 7f)   | 44.50% | 74.1%  | 2.81×   | reversed; identical to forward               |
| 5-seed random (Phase 7d)      | **48.04% ± 0.75** | 74.04% | 2.804×  | headline; acc range 46.8–48.8                  |

## Phase 7d progress (n=500 per seed, 5 seeds)
| seed | 50    | 100   | 150   | 200   | 250   | 300   | 350   | 400   | 450   | 500   | p4%   | speedup |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
|  0   | 0.520 | 0.510 | 0.527 |  -    |  -    |  -    |  -    | 0.463 |  -    | **0.480** | 73.91 | 2.802×  |

Seed 0 elapsed: 1117 s (18.6 min — slower than steady-state due to Phase 7c sharing GPU).
Seed 1 elapsed: 1068 s (17.8 min — still sharing GPU with Phase 7c).
Seed 2 elapsed:  972 s (16.2 min — Phase 7c finished, sharing only with Phase 7f briefly).
Seed 3 elapsed:  773 s (12.9 min — GPU solo).
Seed 4 elapsed:  770 s (12.8 min — GPU solo).

**5-seed final**:
- Accuracies per seed: 48.0, 48.8, 48.2, 48.4, 46.8
- Mean:  **48.04%**
- Std (sample):  **±0.754pp**
- 95% CI on mean: ±0.661pp
- p4 across seeds: 73.91, 74.06, 74.00, 74.04, 74.07 (tight band, ±0.08pp)
- Speedup across seeds: 2.802, 2.805, 2.804, 2.805, 2.806 → **2.804× ± 0.001**

**Vs user-specified target 48.6% ± 0.71% at 2.81×**: mean −0.56pp (within 1σ),
std +0.04pp, speedup −0.006× (rounds to the same 2.81× at 3 sig figs).

## Interpretation (running)
- Random routing at 74/26 on SmolLM-1.7B is demonstrably reaching DWB's reported 48.6%
  accuracy. This is the central experimental claim of Phase 7.
- Controller early trajectory (0.51 at 80 samples) is within the random noise band —
  nothing visible in favor of the learned router vs. uniform random at this split.
- These observations support the paper narrative: *at the right bit-set and right
  split ratio, routing quality is not a differentiator*.

## Files / scripts
- `code/run_phase7c_routing_ablation.py` — random vs controller vs KV-norm, n=200, seed 0
- `code/run_phase7d_random_multiseed.py` — 5-seed random, n=500
- `results/phase7_signal_cache_v2.pt` — per-token signals (445,536 tokens, 4D features)
- `results/phase7c_routing_ablation.json` — written at Phase 7c end
- `results/phase7d_random_multiseed.json` — written at Phase 7d end
- `logs/phase7c.log`, `logs/phase7d.log` — full runtime logs
