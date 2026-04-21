# Paper split: the "mega" paper reorganized into focused deliverables

The original `research/paper/fpga_controller_paper.tex` remains the "mega" paper with every finding from the project. For each paper venue submission, we extract a focused subset into this directory.

Figures live in the **shared** `research/paper/figures/` directory. Each split paper's `main.tex` uses `\graphicspath{{../../figures/}}` to find them. Do not duplicate figures per paper.

Each paper has its own `refs.bib` (a subset of the shared `../fpga_refs.bib` plus any paper-specific cites).

## Compile locally

```bash
cd research/paper/split/A_fpga_binary_48bit
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or if you have `latexmk`:

```bash
cd research/paper/split/A_fpga_binary_48bit
latexmk -pdf main.tex
```

## Paper inventory

| Paper | Directory | Status | One-line thesis |
|---|---|---|---|
| **A** | `A_fpga_binary_48bit/` | **draft v1 in progress** | DWB's 2-bit wastes BRAM; binary {4,8} Pareto-dominates; 130nm silicon in ~10K gates. |
| **B** | `B_beta_calibration_formula/` | TODO | $\beta^* = \overline{q_{t,8}-q_{t,4}}/0.267$ predicts the KV-quant mixed-allocation phase transition across 10 checkpoints, 5 families, 9 within ±0.04. |
| **C** | `C_routing_is_noise/` | TODO | At fixed bit-set and split ratio, per-token routing choice is statistically noise. |
| **D** | `D_130nm_silicon_design/` | TODO (post-tapeout) | 130nm inference engine with baked-in weights + optional Walsh-Hadamard rotation; silicon validation. |

## Relationship

- **A** cites **B** (for $\beta^*$) and **C** (for routing-is-noise) with one-paragraph summaries; A is self-contained.
- **B** and **C** are self-contained; each cites **A** for the hardware context motivating the study.
- **D** cites **A** and **QuaRot**; extends **A** with silicon measurements.
