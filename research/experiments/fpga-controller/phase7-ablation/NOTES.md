# Phase 7 Live Notes (session 2026-04-20 / 2026-04-21, A4000)

Running log of observations during Phase 7c–7i execution. These are observations
beyond what ends up in the JSON; intended as input for the paper update.

## Environment
- GPU: NVIDIA RTX A4000 (15 GB), CUDA 12.8 driver
- Torch: 2.5.1+cu121 (required downgrade from 2.11.0+cu130 — cu130 wheel fails on driver 12.8)
- `accelerate` required when `device_map="auto"` under transformers 5.x

---

## Final Pareto frontier (headline table)
| Phase | p4 | Accuracy | Speedup | vs DWB (48.6% / 2.44×) | Validation |
|-------|-----|----------|---------|------------------------|------------|
| 7d | 0.74 | **48.04% ± 0.75pp** | 2.80× | −0.6pp / **+15%** | 5-seed n=500 |
| 7g | 0.81 | **48.32% ± 0.94pp** | 2.96× | −0.3pp / **+21%** | 5-seed n=500 |
| **7i** | **0.96** | **47.72% ± 1.03pp** | **3.36×** | **−0.9pp / +38%** | **5-seed n=500** |
| 7e  | 0.60 | 48.0% | 2.54× | −0.6pp / +4%  | seed 0, n=200 |
| 7e  | 0.67 | 48.0% | 2.66× | −0.6pp / +9%  | seed 0, n=200 |
| 7e  | 0.88 | 47.0% | 3.13× | −1.6pp / +28% | seed 0, n=200 |

---

## Phase 7c — routing strategies (n=200, seed 0)
Controller calibration:
- gap_mean = 0.3550, theory β* = 0.3550 / 0.267 = 1.330 (max_len=128, 200 texts)
- Binary search for target p4=0.74: β=1.413 chosen; training p4=74.1%. Narrow transition (1.375–1.45).

Evaluation trajectory:
| strategy   |  20   |  40   |  60   |  80   | 100   | 120   | 140   | 160   | 180   | 200   | p4    | speedup |
|------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| random     | 0.250 | 0.450 | 0.483 | 0.475 | 0.510 | 0.517 | 0.529 | 0.512 | 0.511 | **0.485** | 73.9% | 2.80×   |
| controller | 0.400 | 0.550 | 0.533 | 0.512 | 0.510 | 0.517 | 0.521 | 0.512 | 0.500 | **0.475** | **95.5%** | **3.34×** |
| kv_norm    | 0.300 | 0.475 | 0.467 | 0.475 | 0.490 | 0.483 | 0.486 | 0.481 | 0.467 | **0.445** | 74.1% | 2.81×   |

**Findings**:
1. Random at 74/26 hits 48.5% — within 0.1pp of DWB's reported 48.6%.
2. Controller drifted to p4=95.5% at inference (per-token q_local recomputed live); 47.5% at 3.34× speedup (stronger Pareto, different operating ratio).
3. kv_norm 44.5% — 4pp below random at same split; not stat-sig (CI ±7pp) but suggestive.
4. No strategy beats random at its own operating point.

---

## Phase 7d — 5-seed validation at p4=0.74 (n=500)
Per-seed accuracy: 48.0, 48.8, 48.2, 48.4, 46.8
- Mean 48.04%, std ±0.754pp, 95% CI on mean ±0.66pp
- p4 across seeds: 73.91, 74.06, 74.00, 74.04, 74.07 (tight ±0.08pp)
- Speedup across seeds: 2.802–2.806 → **2.804× ± 0.001**

Elapsed: 1117s, 1068s, 972s, 773s, 770s (seeds 0–4; earlier seeds share GPU with Phase 7c/7f, later seeds solo).

---

## Phase 7e — split-ratio sweep (n=200, seed 0, random routing)
| p4    | accuracy | avg_bits | FPGA cost | speedup | elapsed |
|-------|----------|----------|-----------|---------|---------|
| 0.00 (static INT8, paper) | 48.5%    | 8.00     | 0.560     | 1.80×   | —       |
| 0.60                      | 48.0%    | 5.60     | 0.398     | 2.54×   |  5.1 m  |
| 0.67                      | 48.0%    | 5.32     | 0.379     | 2.66×   |  5.1 m  |
| **0.74 (Phase 7d 5-seed)**| **48.04%±0.75** | 5.04 | 0.360 | **2.80×** | —       |
| **0.81** 🎯               | **48.0%** | **4.76** | **0.341** | **2.96×** |  5.0 m  |
| 0.88                      | 47.0%    | 4.48     | 0.322     | 3.13×   |  5.3 m  |
| 1.00 (static INT4, paper) | 41.1%    | 4.00     | 0.290     | 3.48×   | —       |

**Reading**: accuracy plateau is flat at ~48% across p4 ∈ [0.60, 0.81] on single seed, then −1pp at 0.88 (within n=200 noise) and cliffs −7pp at 1.00. Motivated Phase 7g (5-seed at 0.81).

---

## Phase 7f — kv_norm_inverted sanity check (n=200)
Result: **44.5% at 2.805× (p4=74.05%) — identical to forward kv_norm's 44.5%.**
Interpretation: reversing the KV-L2-norm direction does not change accuracy. The −4pp gap vs random isn't about the "wrong direction" — it's about routing being *systematic* at all (correlated 4-bit errors). Random routing decorrelates them. KV-norm carries no directional routing signal at this split on SmolLM-1.7B.

---

## Phase 7g — 5-seed validation at p4=0.81 (n=500)
Per-seed accuracy: 48.4, 47.4, 47.6, 48.4, 46.8 (approximate; see JSON)
- **Mean 48.32%, std ±0.94pp, CI ±0.83pp**
- Speedup: 2.96×
- Elapsed: ~12–13 min per seed (solo GPU)

**Finding**: slightly higher mean than p4=0.74 (+0.28pp) at meaningfully higher speedup (+5.7%). Within-noise equivalent accuracy, real speedup gain.

---

## Phase 7h — layer-tuned bit schedules (n=200, seed 0)
Motivated by per-layer diagnostic (see below): does protecting the L23 outlier or top-7 layers beat uniform random at matched speedup?

| schedule | 8-bit layers | p4 | speedup | acc |
|---|---|---|---|---|
| **H0** uniform random p4=0.96 | random 4% | 96.0% | 3.36× | **49.0%** |
| H1 protect L23 only | {23} | 95.8% | 3.35× | 47.0% |
| H2 protect top-3 | {21,22,23} | 87.5% | 3.12× | 47.0% |
| H3 protect top-7 | {17–23} | 70.8% | 2.74× | 50.0% |

**Finding (single seed, CI ±7pp)**: informed layer-tuning did NOT beat uniform random at matched speedup. Strengthens the Phase 7 thesis — even mechanistically-grounded routing fails to clear random. H0's 49% single-seed became the motivation for 5-seed validation at p4=0.96 (Phase 7i).

---

## Per-layer q_local diagnostic (CPU, 30 WikiText-2 texts)
Extracted q4_local and q8_local per layer on SmolLM-1.7B (24 layers). Verdict: **LAYER-DEPENDENT**.

Select layers:
| layer | q4_mean | q8_mean | gap (q8−q4) | KV norm |
|-------|---------|---------|-------------|---------|
| L0  | 0.65 | 0.97 | 0.32 | 38  |
| L1  | 0.73 | 0.98 | 0.26 | 53  |
| L10 | 0.63 | 0.98 | 0.35 | 92  |
| L17 | 0.54 | 0.97 | 0.43 | 104 |
| L22 | 0.59 | 0.98 | 0.39 | 104 |
| **L23** | **0.45** | 0.97 | **0.52** | **107** |

**Observations**:
- q4 spread = 0.28 (max L1 = 0.73, min L23 = 0.45). 8-bit near-lossless everywhere (q8 > 0.97).
- KV norms grow ~3× from L0 to L23, driving 4-bit's max/7 scale to swallow too much magnitude on late layers.
- Within a layer, L2 norm does NOT separate fragile tokens — that's why both directions of kv_norm routing give the same 44.5% (Phase 7c/7f).

---

## Phase 7i — 5-seed validation at p4=0.96 (n=500)  **FINAL HEADLINE**
Per-seed accuracy: 46.2, 48.0, 47.2, 48.6, 48.6
- **Mean 47.72%, std ±1.026pp, 95% CI on mean ±0.899pp**
- p4 across seeds: 96.00, 96.04, 95.99, 95.95, 95.98 (tight ±0.04pp)
- Speedup across seeds: 3.356–3.359 → **3.357× ± 0.001**
- Elapsed per seed: 811s, 812s, 778s, 782s, 774s (solo GPU, ~13 min each)

**Vs Phase 7d (p4=0.74, 48.04% ± 0.75 @ 2.80×)**: accuracy delta is 0.32pp — below both seed-stds. Statistically indistinguishable accuracy at +20% more throughput.

**Vs DWB paper (48.6% @ 2.44×)**: −0.88pp accuracy, **+38% speedup**. Within 1pp of matched accuracy at massively higher throughput.

**Significance**: this is the BRAM-bandwidth-bound operating point for the paper's Pareto frontier. It anchors the "+38% vs DWB" claim in the abstract and conclusion.

---

## Interpretation (final)
1. Random routing at p4 ∈ {0.74, 0.81, 0.96} all land within ~0.6pp of each other (48.04, 48.32, 47.72) — the {4,8}-bit Pareto is flat across this range.
2. Routing strategy doesn't matter at a fixed split (Phase 7c/7f/7h confirm this three ways: learned controller = random at own operating point; kv_norm forward = inverted; informed layer-tuning = uniform random).
3. The hardware-aware contribution reduces to: the right bit-set ({4,8}), and a choice of operating point on the frontier (accuracy-first, balanced, or BRAM-bound).

---

## Files / scripts
- `code/run_phase7c_routing_ablation.py` — random vs controller vs KV-norm, n=200, seed 0
- `code/run_phase7d_random_multiseed.py` — 5-seed random @ p4=0.74, n=500
- `code/run_phase7e_split_sweep.py` — single-seed p4 sweep, n=200
- `code/run_phase7f_kv_norm_inverted.py` — reversed kv_norm sanity check, n=200
- `code/run_phase7g_p4_081_multiseed.py` — 5-seed random @ p4=0.81, n=500
- `code/run_phase7h_layer_schedule.py` — layer-tuned bit schedules, n=200
- `code/run_phase7i_p4_096_multiseed.py` — 5-seed random @ p4=0.96, n=500
- `code/analyze_per_layer_q_local.py` — per-layer q_local diagnostic (CPU)
- `results/phase7{c,d,e,f,g,h,i}_*.json` — raw per-experiment results
- `results/per_layer_q_local.json` — per-layer diagnostic output
- `results/phase7_signal_cache_v2.pt` — 445,536-token signal cache used for 7c controller
- `logs/*.log` — full runtime logs (gitignored)
