# Research Log

## 2026-04-19 — Bootstrap

**Decision**: Run independent verification of arXiv:2604.04722 across two parallel tracks.

**Track 1 (main branch)**: Reproduce Table 3 results — SmolLM-360M on HellaSwag primary, then all models/benchmarks.

**Track 2 (turboquant-integration branch)**: Novel extension — use DWB controller as precursor ranking stage for TurboQuant (Google, ICLR 2026). Hypothesize that importance-aware routing improves TurboQuant's accuracy on reasoning benchmarks.

**Arithmetic verification** (no GPU needed):
- H1: (2.93 - 2.41) / 2.93 = 17.75% ✓ — matches paper claim exactly
- H2: 41.20 - 33.60 = 7.60 ✓ — matches paper claim exactly
- H3: 41.50 - 41.20 = 0.30 ✓ — matches paper claim exactly

**Observation**: Dynamic KV baseline accuracy (29.90%) is *lower* than static 4-bit KV (33.60%) on HellaSwag for SmolLM-360M. This pattern holds across all models — the rule-based approach is harmful. Supports the paper's narrative that learned policies are needed.

**Next**: Provision RTX 4090 on NVIDIA Brev, clone original repo, run FP16 baseline.

---

## Pending

- [ ] GPU environment setup on Brev
- [ ] Clone + inspect original repo code
- [ ] FP16 baseline run (SmolLM-360M, HellaSwag)
- [ ] Static 4-bit KV baseline
- [ ] DWB method run
- [ ] TurboQuant baseline
- [ ] DWB + TurboQuant combined pipeline
