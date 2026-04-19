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

## 2026-04-19 — Evaluator Debugging (Multiple Fixes)

**Run 01a (SmolLM2-360M FP16)**: Got 45.33% vs paper's 41.5% (+3.83pp). Identified as wrong model variant — paper uses original SmolLM-360M. Result stored for paper comparison table.

**KV Cache Quantization Bug #1 — Wrong Approach**: Initial attempt used `apply_int4_simulation()` which quantized model *weights* rather than KV cache. Paper specifically quantizes keys/values during decoding, not weights. Weight quantization caused catastrophic accuracy loss (~23%).

**KV Cache Quantization Bug #2 — Hooks Not Firing**: Rewrote `kv_cache_quant.py` to hook attention module outputs. Found that transformers 5.x uses `DynamicCache` objects instead of raw `(key, value)` tuples — hooks were silently finding nothing to quantize. Result: FP16, KV-4bit, and KV-8bit all gave identical 49.00% on 100 samples.

**Fix v2 — Correct Approach**: Hook `k_proj` and `v_proj` Linear submodule outputs directly. These layers exist in LLaMA/SmolLM attention modules and produce the raw K and V tensors before attention computation. Quantizing here simulates quantized KV cache storage and retrieval correctly.

**Ongoing**: Corrected experiment (4 conditions: FP16, KV-4bit, KV-8bit, KV-2bit, 100 samples, SmolLM-360M) running on CPU.

**Git authorship fix**: All commits rewritten from `arun.rajendra8@gmail.com` to `themoddedcube@gmail.com` using git filter-branch.

---

---

## 2026-04-19 — Metric Discovery (Major)

**Key finding**: Paper uses unnormalized log-likelihood (`acc`), not length-normalized (`acc_norm`).  
Direct test: `acc (unnorm)` on 50 val samples = **42.0%** vs paper's **41.5%** ✓

Evidence from paper PDF: *"Accuracy is computed based on the final multiple-choice answer selected by the model"* and *"reported accuracy of existing methods are directly taken from [7]"* (SmolLM paper). The FP16 baseline 41.5% is from a citation.

Cross-model sanity check: our acc_norm for SmolLM-360M ≈ 54%, while paper reports SmolLM-1.7B = 49%. Our smaller model matching paper's larger model in acc_norm revealed the metric mismatch.

**KV-4bit investigation**: 
- KV-2bit: 20% (catastrophic, confirms hooks work)
- KV-8bit: 42% (no degradation, expected)
- KV-4bit on 50 samples: 46% (noisy, within CI ±14pp)
- Per-tensor INT4 noise is roughly zero-mean (symmetric quantization), so errors cancel in attention sum
- Running 500-sample definitive test + per-token + asymmetric variants

**DWB controller training**: Running in background (150 train examples, 5 epochs).

**Git**: All commits authored as themoddedcube@gmail.com. History rewritten and force-pushed to GitHub.

---

## Pending

- [x] FP16 baseline confirmed — 42.0% ✅
- [x] Metric resolved — use acc (unnorm)
- [x] KV hooks fixed — k_proj/v_proj
- [x] Git authorship fixed — themoddedcube@gmail.com
- [ ] KV-4bit definitive baseline (500 samp) — RUNNING
- [ ] DWB controller training — RUNNING
- [ ] DWB evaluation
- [ ] GPU environment setup on Brev (for H1 latency)
- [ ] TurboQuant pipeline (turboquant-integration branch)
