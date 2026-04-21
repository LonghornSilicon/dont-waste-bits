# Paper A — Editorial Notes

**Status**: Draft v1, compiles cleanly with tectonic.

## Data verified against JSON sources
- 59.08%±2.24pp static INT4 (matched): phase7j_static_int4_multiseed.json ✓
- 58.52%±2.46pp p4=0.96 mix (matched): phase7k_p4096_matched.json ✓
- 47.72%±1.03pp p4=0.96 (first-500): phase7i_p4_096_multiseed.json ✓
- 48.32%±0.94pp p4=0.81 (first-500): phase7g_p4_081_multiseed.json ✓
- 48.04%±0.75pp p4=0.74 (first-500): phase7d_random_multiseed.json ✓
- R2 rotation: -5.96±1.11pp delta: phase7m_r2_rotation.json ✓
- Plain INT4: -5.76±0.95pp delta: phase7j ✓

## Editorial decisions
- Kept "InnerQ" as plain text in Related Work (no \cite) since fpga_refs.bib has it but
  Paper A refs.bib is a minimal subset. Added innerq2026 to refs.bib would be cleaner —
  TODO for final submission.
- 130nm gate estimates (~10K, ~15K) are design-space estimates, not measured silicon.
  Clearly labeled as such in the table.
- Subset methodology note is prominent — necessary because 59% vs 48% baseline comparison
  would confuse readers without it.
- Paper A refs.bib missing innerq2026. If reviewer asks about InnerQ cite: add it.

## TODO for final submission
- [ ] Add innerq2026 to refs.bib and \citep{innerq2026} in Related Work
- [ ] Companion paper citations (A cites B and C) — currently stub @misc entries in B/C;
      once real DOIs/arXiv IDs exist, update cross-citations
- [ ] FPGA latency hardware validation (Paper D, out of scope for now)
- [ ] Page count: ~7 pages with current content
