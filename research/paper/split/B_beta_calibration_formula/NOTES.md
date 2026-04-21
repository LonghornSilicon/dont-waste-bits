# Paper B — Editorial Notes

**Status**: Draft v1, compiles cleanly with tectonic (cosmetic overfull hbox warnings only).

## Data sources
All beta calibration data pulled from mega-paper (fpga_controller_paper.tex) and:
- phase5-benchmark/results/*_calibration*.json — individual model calibrations
- phase5-benchmark/results/*_cal_sensitivity.json — within-corpus sensitivity
- phase5-benchmark/results/beta_transition_fine_1b7.json — 1.7B fine sweep
- phase5-benchmark/results/reproducibility_1b7.json — 5-seed multi-run at 1.7B

## Editorial decisions
- Gradient derivation (Eq. 1) uses α=1 (as set in all experiments); this is a convention,
  not a constraint. Stated clearly.
- 0.267 = (c8-c4)/C_FP16 = 0.270/1.010 = 0.2673... rounded to 0.267. Consistent with
  mega-paper and all experiment code.
- SmolLM-360M fine sweep table (Tab. 2) lifted directly from mega-paper Section 5.3.
- Cross-scale validation table (Tab. 3) and cross-arch table (Tab. 4) from mega-paper
  Sections 5.3 and appendix.
- Floor attractor section synthesizes findings from mega-paper Discussion.
- Calibration sensitivity table from mega-paper Appendix B.
- Instruct vs base table from mega-paper Appendix D.
- zhang2022opt and radford2019language included in refs.bib but not explicitly cited
  in-text (OPT and GPT-2 are referenced by name only). TODO: add \citep{} for both.
- zhang2024tinyllama included but not cited in-text. TODO: add \citep{} for TinyLlama.
- Companion paper citations (A and C) as @misc stubs.

## TODO for final submission
- [ ] Add \citep{zhang2022opt} in-text when OPT-125M/350M first mentioned (Section 5)
- [ ] Add \citep{radford2019language} in-text when GPT-2 first mentioned (Section 5)
- [ ] Add \citep{zhang2024tinyllama} when TinyLlama first mentioned (Section 4)
- [ ] Update companion paper citations with real arXiv IDs
- [ ] SmolLM-360M fine sweep figure (fig:beta right panel) labels could be made clearer
      for standalone readability — currently inherits mega-paper caption
- [ ] Page count: ~10 pages with appendix
