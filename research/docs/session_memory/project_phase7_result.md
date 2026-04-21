---
name: Phase 7 result — routing selection carries no accuracy signal at 74/26 split (1.7B)
description: At the p4=0.74 {4,8}-bit split on SmolLM-1.7B, random / controller / KV-norm routing all land within ±1 seed-std; the paper's 1.7B win comes from the bit-set and ratio, not token selection.
type: project
originSessionId: eeed6cc0-c5f4-4b71-988a-03ac1ad53025
---
Phase 7 ablation (run 2026-04-20 on A4000, in `research/experiments/fpga-controller/phase7-ablation/`) answers a question Phase 5 left open: at the target 74/26 {4,8}-bit split on SmolLM-1.7B, does the learned Gumbel controller beat trivial routing baselines?

**Answer:** no, not measurably.
- **Phase 7d** (5-seed random routing, n=500 HellaSwag): 48.6% ± 0.71% at 2.81× FPGA speedup. Matches DWB's reported 48.6% at +15% more throughput.
- **Phase 7c** (n=200, same split): random, learned binary Gumbel controller (β*-calibrated via binary search), and KV-norm (bottom-74% by L2 norm) all fall inside ±1 seed-std of the Phase 7d mean.

**Why:** Given the right bit-set ({4,8}, eliminating DWB's strictly-dominated 2-bit) and the right split ratio (2.81× target), *which* tokens get 4-bit vs. 8-bit is noise at this scale. The hardware-aware contribution of this paper is fully captured by those two design choices.

**How to apply:** When framing the paper, don't over-claim controller sophistication at 1.7B. The headline is: hardware-aware bit-set + calibrated split ratio = +15% over DWB. A learned controller is optional, not load-bearing. The paper table footnote clarifies: "Ours ($p_4{=}0.74$)" row uses random routing for reproducibility; Phase 7c shows controller/KV-norm within noise.
