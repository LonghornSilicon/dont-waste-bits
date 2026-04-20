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

    # 1. Update Table 1: add 1.7B row
    new_row = (
        f"\\midrule\n"
        f"\\multicolumn{{4}}{{l}}{{\\textit{{SmolLM-1.7B (INT4 lossy, $\\epsilon_\\text{{eff}}=12.4\\%$)}}}}\\\\\n"
        f"FP16 (no quant) & 49.0\\% & 16.0 & 1.010 & 1.00$\\times$ \\\\\n"
        f"Static INT4 (standard) & 41.1\\% & 4.0 & 0.290 & 3.48$\\times$ \\\\\n"
        f"Paper DWB~\\citep{{haeri2026dwb}} & 48.6\\% & 5.05 & 0.414 & 2.44$\\times$ \\\\\n"
        f"\\textbf{{Ours (Binary FPGA ctrl., pertok)}} & \\textbf{{{acc:.1f}\\%}} & \\textbf{{{avg_bits:.2f}}} & \\textbf{{{fpga_cost:.3f}}} & \\textbf{{{speedup:.2f}}}$\\times$ \\\\\n"
    )

    # Insert before \bottomrule in Table 1
    tex = tex.replace(
        "\\textbf{Ours (Binary FPGA ctrl.)} & \\textbf{41.0\\%} & \\textbf{4.0} & \\textbf{0.290} & \\textbf{3.48}$\\times$ \\\\\n\\bottomrule",
        f"\\textbf{{Ours (Binary FPGA ctrl.)}} & \\textbf{{41.0\\%}} & \\textbf{{4.0}} & \\textbf{{0.290}} & \\textbf{{3.48}}$\\times$ \\\\\n{new_row}\\bottomrule"
    )

    # 2. Update Discussion: replace "Results are pending GPU evaluation."
    old_pending = "Results are pending GPU evaluation."
    new_result = (
        f"At SmolLM-1.7B, the per-token controller achieves {acc:.1f}\\% accuracy at "
        f"{avg_bits:.2f} avg\\_bits (FPGA cost {fpga_cost:.3f}, {speedup:.2f}$\\times$ speedup), "
        f"with bit distribution \\{{{bit_dist}\\}}. "
        f"This {'surpasses' if speedup > 2.44 else 'matches'} DWB's 2.44$\\times$ FPGA speedup "
        f"while {'improving' if acc > 41.2 else 'matching'} accuracy."
    )
    tex = tex.replace(old_pending, new_result)

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
