# Paper C — Editorial Notes

**Status**: Draft v1, compiles cleanly with tectonic (cosmetic overfull hbox warnings only).

## Data verified against JSON sources
- phase7c_routing_ablation.json: random 48.5%, controller 47.5% (p4=95.5%), kv_norm 44.5% ✓
- phase7f_kv_norm_inverted.json: kv_norm_inverted 44.5% ✓
- phase7d_random_multiseed.json: 5-seed mean 48.04%±0.75pp at p4=0.74, n=500 ✓
- phase7g_p4_081_multiseed.json: 5-seed mean 48.32%±0.94pp at p4=0.81, n=500 ✓
- phase7i_p4_096_multiseed.json: 5-seed mean 47.72%±1.03pp at p4=0.96, n=500 ✓
- phase7h_layer_schedule.json: H0=49%, H1=47%, H2=47%, H3=50% ✓
- phase7j_static_int4_multiseed.json: 59.08%±2.24pp at 3.48x ✓
- phase7k_p4096_matched.json: 58.52%±2.46pp at 3.36x, paired delta -6.32±1.49pp ✓
- per_layer_q_local.json: L0-L23 q4/q8/gap/norm values ✓

## Editorial decisions
- Controller "drifts" to p4=0.955 at argmax — this is the Gumbel→argmax train-test gap.
  Reported accurately; Paper B explains the β* mechanism.
- KV-norm inverted result (44.5%) is the same as forward (44.5%) to 1 decimal place at n=200.
  This is the key sanity-check result. Described as "identical to 2 decimal places."
- Layer H3 (protect top-7) at 50% single-seed: within ±7pp CI of H0 at 49%. Not claiming
  it's different. The point is it uses much lower p4 (70.8%) for no accuracy gain.
- Subset methodology prominently explained — the 13pp gap between first-500 and random-500
  is crucial context for all the ~48% numbers.
- Companion paper citations: Paper A (companion_paper_a) and Paper B (companion_paper_b)
  cited as @misc stubs. TODO: update with real citations.

## TODO for final submission
- [ ] Update companion paper citations (A and B) with real arXiv IDs
- [ ] Phase 7e split sweep results (p4 ∈ {0.60, 0.67, 0.81, 0.88}) could add a figure
      showing the flat accuracy plateau — omitted here for space, present in mega-paper
- [ ] Consider adding a brief note on the controller's p4 drift mechanism for readers
      coming to this paper without reading Paper B
- [ ] Page count: ~5 pages with current content
