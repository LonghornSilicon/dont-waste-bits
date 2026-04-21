---
name: DWB-TurboQuant revisit plan (inference engine, 130nm tape-out)
description: Plan for revisiting turboquant-integration branch after Phase 7 experiments conclude; target is inference-only FPGA on 130nm with weights baked in, prioritizing speedup over 48% accuracy.
type: project
originSessionId: eeed6cc0-c5f4-4b71-988a-03ac1ad53025
---
User set these constraints for the DWB-TurboQuant contribution (2026-04-21):

- **Target**: FPGA tape-out on 130nm, inference engine only, weights baked in offline.
- **Accuracy**: ~48% on HellaSwag/SmolLM-1.7B is acceptable (matching current Phase 7 headline); **prioritize speedup**.
- **Constraint**: full PolarQuant with online Hadamard at every layer likely won't fit on 130nm.

**Preferred research lane**: offline Hadamard / random-rotation pre-processing baked into KV weight prep, followed by plain INT4 (or binary {4,8}) at inference. No runtime Hadamard hardware needed.

**First diagnostic when we revisit** (CPU-only, ~15 min):
1. Apply an offline random orthogonal rotation (or Hadamard) to the KV projection weights of SmolLM-1.7B.
2. Re-extract per-token q4_local and q8_local on the rotated weights using the same Phase 7 pipeline.
3. Compare to the unrotated q4 gradient we measured in `per_layer_q_local.json` (q4 from 0.45 at L23 to 0.73 at L1).
4. If rotation tightens the q4 distribution (e.g., L23 climbs from 0.45 toward ~0.60), the losslessness threshold shifts upward and the 1.7B Pareto frontier moves — potential path to push 48% → higher accuracy, or keep 48% at much higher p4 (and thus higher speedup).

**Branch to inspect**: `turboquant-integration` on GitHub. Prior results (from CLAUDE.md / repo notes): +2pp HellaSwag and +3pp ARC at 5.05 avg_bits; −20pp BoolQ regression (WHT gains are task-specific).

**How to apply**: Before proposing any implementation changes, read `research/turboquant-findings.md` on that branch, understand exactly what PolarQuant/TurboQuant means in this codebase, then align it with the Phase 7 finding that routing doesn't matter (so the story should be "rotation shifts the bit-set quality, not the routing").
