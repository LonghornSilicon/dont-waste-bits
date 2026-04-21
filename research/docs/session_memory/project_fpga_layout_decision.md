---
name: FPGA layout decision rule tied to Phase 7j/7k/rotated-INT4 outcomes
description: Given the user's 130nm inference-engine constraint and speedup-first priority, which FPGA layout to commit to depends on whether 7k (p4=0.96 mix) is close enough to 7j (static INT4). Records the decision matrix and the gate-count estimates.
type: project
originSessionId: eeed6cc0-c5f4-4b71-988a-03ac1ad53025
---
User's goal (2026-04-21): tape out on 130nm, weights baked in, inference-only, accuracy ~48% OK, speedup-first. Wants the *simplest* FPGA layout that still hits accuracy target.

**Decision rule**: the question "is 7k successful?" means "is static INT4 close enough to the p4=0.96 mix that we can drop the mix for the simpler hardware?" → we are actively *rooting* for 7k ≈ 7j, because that unlocks ~10K-gate layout instead of ~25K-gate layout.

## Architecture options by accuracy outcome

All options assume offline-rotated K/V projection weights are an optional add-on that trades weight-prep time for closing the ~5.7pp INT4-vs-FP16 gap (pending rotated-INT4 end-to-end eval).

### Option A: rotated static INT4
- Hardware: scale extractor + round/clamp tree + BRAM single 4-bit port + dequant multiply
- Gate count: ~10–15K gates (130nm)
- BRAM cost: 0.290 per KV token → **3.48× vs FP16**
- Accuracy target: Δ ≈ −1pp vs FP16 (hypothesis, pending rotated-INT4 experiment)
- **Wins when**: 7k ≈ 7j AND rotation closes the gap

### Option B: plain static INT4 (no rotation)
- Hardware: same as A
- Gate count: ~10–15K
- BRAM cost: 0.290 → 3.48×
- Accuracy: Δ ≈ −5.7pp vs FP16 (measured: 59.08%±2.24pp matched-subset, 47.0% first-500)
- **Wins when**: 7k ≈ 7j AND rotation doesn't help OR isn't worth the offline cost

### Option C: p4=0.96 random + LFSR router
- Hardware: Option B + LFSR pseudo-random generator + fixed-threshold comparator + two BRAM port widths (4-bit and 8-bit) + 2:1 arbitration mux
- Gate count: ~25–30K
- BRAM cost: 0.301 → 3.36×
- Accuracy: Δ ≈ −4.8pp to −5.2pp vs FP16 (partial 7k: seeds 0,1 only)
- **Wins when**: 7k is meaningfully better than 7j (delta ≥ 2pp), no rotation

### Option D (reject): DWB binary learned controller
- Per-token MLP, feature extraction, 4-class router
- ~100K+ gates
- Phase 7 ablations showed routing choice is noise at a given split — controller gives no accuracy lift over random routing. Not recommended.

## Quantitative tie-in to pending experiments

| 7k outcome (p4=0.96 matched) | rotated-INT4 outcome | ship | gates | speedup | acc vs FP16 |
|---|---|---|---|---|---|
| ≈ 7j (|Δ|<1pp)                      | rotation recovers ≥3pp       | **A** | 10–15K | 3.48× | ≥ −1pp |
| ≈ 7j                                | rotation no help             | **B** | 10–15K | 3.48× | ~−5.7pp |
| Meaningfully > 7j (Δ ≥ 2pp)         | rotation recovers ≥3pp       | A' (rotation > mix) | 10–15K | 3.48× | ≥ −1pp |
| Meaningfully > 7j                   | rotation no help             | **C** | 25–30K | 3.36× | ~−3pp |

**Current trend (partial data)**: 7j done (59.08%±2.24pp). 7k seeds 0,1 → 58.8%, 62.6% (mean 60.7, matches 7j within noise). If remaining 3 seeds hold, the answer is "INT4 is close enough" → ship Option A or B.

**How to apply**: when 7k finishes and rotated-INT4 runs, match the row in the table to the measured deltas and pick the matching ship option. The paper's operating-points table and the 130nm block diagram follow directly.
