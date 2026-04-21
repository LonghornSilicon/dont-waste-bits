# Overnight Summary — Papers A, B, C

**Date**: 2026-04-21  
**Branch**: fpga-controller  
**All three papers compiled with tectonic and pushed to origin/fpga-controller.**

---

## Paper A — `A_fpga_binary_48bit/`

**Status**: Polish complete. No content changes needed — Brev session had already written a correct draft.  
**Pages**: ~7  
**Thesis**: DWB's 2-bit wastes BRAM on Xilinx Ultrascale+; binary {4,8}-bit scheme Pareto-dominates; static INT4 at 3.48× is the dominant operating point; 130nm silicon is ~10K gates.  
**Key headline**: Static INT4 59.08%±2.24pp on matched subsets at 3.48×, +43% vs DWB's 2.44×.  
**What was done**: Added NOTES.md with data provenance verification (all numbers checked against Phase 7 JSONs). No tex edits — paper was already correct.  
**TODOs**: Add `\citep{innerq2026}` in Related Work; update companion-paper stub citations when arXiv IDs exist.

---

## Paper C — `C_routing_is_noise/`

**Status**: Draft v1 complete.  
**Pages**: ~5  
**Thesis**: At fixed {4,8}-bit set and split ratio, per-token routing choice (random, Gumbel controller, KV-norm ×2, layer-tuned schedules, mix vs static INT4) is statistically indistinguishable from uniform Bernoulli. KV-norm is the exception — it's *worse* by ~4pp.  
**Key tables**:
- Routing ablation (7c): random 48.5%, controller 47.5% (drifted p4), kv_norm 44.5%, inverted 44.5%
- 5-seed frontier (7d/7g/7i): flat accuracy 48.04/48.32/47.72% across p4={0.74,0.81,0.96}
- Layer schedules (7h): H0-H3, none beat random at matched speedup
- Matched-subset (7j/7k): mix 58.52%±2.46 vs INT4 59.08%±2.24 — tied
- Per-layer diagnostic (all 24 layers): L23 outlier (q4=0.447, gap=0.520)  
**TODOs**: Update companion citations; optionally add Phase 7e split-sweep figure.

---

## Paper B — `B_beta_calibration_formula/`

**Status**: Draft v1 complete.  
**Pages**: ~10 (including appendix)  
**Thesis**: β* = gap_mean/0.267 predicts the phase transition for mixed KV-cache allocation, validated on 10 checkpoints × 5 families, 9/10 within ±0.04. Single sentence of calibration text suffices.  
**Key tables**:
- SmolLM-360M fine sweep: transition window [1.2, 1.4], error=0.010
- Cross-scale (135M/360M/1.7B): errors 0.014–0.020
- Full 10-checkpoint table: 9/10 within ±0.04, mean error 0.018
- Calibration sensitivity: max_error ≤0.018 across 7 models
- Instruct vs base: MHA -44%, GQA -0.3% (already at floor)
- Appendix: 1.7B fine sweep + extended calibration table  
**Surprises**: Had to add in-text citations for OPT, GPT-2, TinyLlama (they were named in prose but not \citep'd in the mega-paper extract). Flagged in NOTES.md.  
**TODOs**: Add \citep{zhang2022opt}, \citep{radford2019language}, \citep{zhang2024tinyllama} in-text; update companion citations.

---

## Compilation

All three compiled with **tectonic** (installed via scoop — pdflatex not present on this machine).  
Cosmetic overfull hbox warnings in B and C (table columns slightly too wide); zero undefined citations, zero undefined references.  
PDFs committed alongside .tex files.

## What's Not Done (out of scope)
- Paper D (130nm silicon design) — explicitly out of scope per instructions
- Cross-citation stubs: companion paper @misc entries have no real arXiv IDs yet — update when papers are posted
- In-text OPT/GPT-2/TinyLlama \citep{} calls in Paper B (refs present in bib, just not \citep'd inline yet)
