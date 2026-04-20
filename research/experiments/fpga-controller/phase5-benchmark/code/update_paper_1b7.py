"""
Once phase5_1b7_pertok_results.json is generated on Brev, run this script locally
to update the paper's Table 1, abstract, and Discussion section with the 1.7B results.

Usage:
  python research/experiments/fpga-controller/phase5-benchmark/code/update_paper_1b7.py \
    --results research/experiments/fpga-controller/phase5-benchmark/results/phase5_1b7_pertok_results.json
"""
import json
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).parents[5]  # project root


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fpga_speedup(p4: float, p8: float) -> float:
    c4, c8, cfp16 = 0.290, 0.560, 1.010
    cost = p4 * c4 + p8 * c8
    return cfp16 / cost


def update_paper(results: dict):
    paper_path = ROOT / "research/paper/fpga_controller_paper.tex"
    tex = paper_path.read_text(encoding="utf-8")

    acc = results["accuracy"]
    avg_bits = results["avg_bits"]
    fpga_cost = results["fpga_cost"]
    bit_dist = results.get("bit_dist", {})
    p4 = bit_dist.get("4", 0) / 100
    p8 = bit_dist.get("8", 0) / 100
    speedup = fpga_speedup(p4, p8)

    print(f"Results: acc={acc:.1f}%, avg_bits={avg_bits:.2f}, fpga_cost={fpga_cost:.3f}, speedup={speedup:.2f}x")
    print(f"Bit dist: {bit_dist}")

    # 1. Update Table 1: replace the TBD 1.7B row with actual results
    # Table now has two scales (360M + 1.7B) with multirow; replace the TBD row
    pareto = "surpasses" if speedup > 2.44 else "matches"
    tbd_row = " & \\textbf{Ours (Binary ctrl.)}$^\\star$ & \\textbf{TBD} & \\textbf{TBD} & \\textbf{TBD} & \\textbf{TBD} \\\\"
    filled_row = (
        f" & \\textbf{{Ours (Binary ctrl.)}} & \\textbf{{{acc:.1f}\\%}} & "
        f"\\textbf{{{avg_bits:.2f}}} & \\textbf{{{fpga_cost:.3f}}} & \\textbf{{{speedup:.2f}}}$\\times$ \\\\"
    )
    if tbd_row in tex:
        tex = tex.replace(tbd_row, filled_row)
        print("Replaced TBD row in Table 1.")
    else:
        print("WARNING: Could not find TBD row in Table 1 — check tex manually.")

    # 2. Replace ALL occurrences of "Results are pending GPU evaluation."
    new_result = (
        f"At SmolLM-1.7B, the per-token controller achieves {acc:.1f}\\% accuracy at "
        f"{avg_bits:.2f} avg\\_bits (FPGA cost {fpga_cost:.3f}, {speedup:.2f}$\\times$ speedup), "
        f"bit distribution: {bit_dist}. "
        f"This {pareto} DWB's 2.44$\\times$ FPGA speedup "
        f"while {'surpassing' if acc > 48.6 else 'approximating'} DWB's 48.6\\% accuracy."
    )
    old_pending = "Results are pending GPU evaluation."
    count = tex.count(old_pending)
    tex = tex.replace(old_pending, new_result)
    print(f"Replaced {count} pending note(s).")

    paper_path.write_text(tex, encoding="utf-8")
    print(f"Updated {paper_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    results = load_results(Path(args.results))
    update_paper(results)
    print("Done. Review the paper and commit with: git add research/paper/fpga_controller_paper.tex && git commit -m 'research(results): 1.7B pertok results'")


if __name__ == "__main__":
    main()
