"""
Generate the three Phase 7 figures for fpga_controller_paper_v2.tex:

  figures/pareto_frontier.{pdf,png}      — Pareto scatter of all 1.7B points
  figures/routing_ablation.{pdf,png}     — bar chart of Phase 7c/d/f strategies
  figures/per_layer_q_local.{pdf,png}    — q4/q8 by layer with L23 flagged

Loads experimental JSONs so the figures track the real numbers.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASE7 = Path(__file__).parent.parent / "experiments/fpga-controller/phase7-ablation/results"
FIG    = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def savefig(name):
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------- pareto
def pareto_figure():
    d_7c = json.load(open(PHASE7 / "phase7c_routing_ablation.json"))
    d_7d = json.load(open(PHASE7 / "phase7d_random_multiseed.json"))
    d_7e = json.load(open(PHASE7 / "phase7e_split_sweep.json"))
    d_7f = json.load(open(PHASE7 / "phase7f_kv_norm_inverted.json"))
    d_7g = json.load(open(PHASE7 / "phase7g_p4_081_multiseed.json"))
    d_7i = json.load(open(PHASE7 / "phase7i_p4_096_multiseed.json"))

    # Baselines (reported in the DWB paper or from arithmetic)
    fp16_pt   = (1.00, 49.0, "FP16")
    int8_pt   = (1.80, 48.5, "Static INT8")
    int4_pt   = (3.48, 41.1, "Static INT4")
    dwb_pt    = (2.44, 48.6, "DWB (paper)")

    # Phase 7d and 7g: 5-seed points with error bars
    p7d = (d_7d["summary"]["fpga_speedup_mean"],
           d_7d["summary"]["accuracy_mean"],
           d_7d["summary"]["accuracy_std"])
    p7g = (d_7g["summary"]["fpga_speedup_mean"],
           d_7g["summary"]["accuracy_mean"],
           d_7g["summary"]["accuracy_std"])
    p7i = (d_7i["summary"]["fpga_speedup_mean"],
           d_7i["summary"]["accuracy_mean"],
           d_7i["summary"]["accuracy_std"])

    # Phase 7e sweep: single seed n=200
    sweep = [(m["fpga_speedup"], m["accuracy_pct"], k)
             for k, m in d_7e["per_p4"].items()]

    fig, ax = plt.subplots(figsize=(6.5, 4.3))

    # Baseline scatter
    bl = [fp16_pt, int8_pt, int4_pt]
    ax.scatter([p[0] for p in bl], [p[1] for p in bl],
               c="#888", marker="s", s=70, zorder=3,
               label="Static baselines")
    for x, y, lbl in bl:
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(7, 4),
                    fontsize=8.5, color="#444")

    # DWB
    ax.scatter([dwb_pt[0]], [dwb_pt[1]],
               c="#c0392b", marker="^", s=110, zorder=4,
               label="DWB (paper)")
    ax.annotate("DWB", (dwb_pt[0], dwb_pt[1]),
                textcoords="offset points", xytext=(-24, 6),
                fontsize=9, color="#c0392b", fontweight="bold")

    # Phase 7e single-seed sweep
    xs = [p[0] for p in sweep]; ys = [p[1] for p in sweep]
    ax.scatter(xs, ys, c="#2980b9", marker="o", s=55, zorder=4,
               label="Phase 7e sweep ($n{=}200$, single seed)")
    for x, y, lbl in sweep:
        ax.annotate(f"$p_4{{=}}${lbl}", (x, y),
                    textcoords="offset points", xytext=(7, -3),
                    fontsize=7.5, color="#2980b9")

    # Phase 7d 5-seed
    ax.errorbar(p7d[0], p7d[1], yerr=p7d[2], fmt="D",
                c="#27ae60", ms=9, capsize=4, zorder=5,
                label=f"Phase 7d 5-seed ($p_4{{=}}0.74$): {p7d[1]:.2f}%$\\pm${p7d[2]:.2f}pp")

    # Phase 7g 5-seed (the Pareto-shifted point)
    ax.errorbar(p7g[0], p7g[1], yerr=p7g[2], fmt="D",
                c="#16a085", ms=9, capsize=4, zorder=5,
                label=f"Phase 7g 5-seed ($p_4{{=}}0.81$): {p7g[1]:.2f}%$\\pm${p7g[2]:.2f}pp")

    # Phase 7i 5-seed (BRAM-bound operating point, +38% vs DWB)
    ax.errorbar(p7i[0], p7i[1], yerr=p7i[2], fmt="D",
                c="#0b7a5a", ms=10, capsize=4, zorder=5,
                label=f"Phase 7i 5-seed ($p_4{{=}}0.96$): {p7i[1]:.2f}%$\\pm${p7i[2]:.2f}pp")

    # Routing-strategy singletons (Phase 7c, n=200)
    s7c = d_7c["strategies"]
    ax.scatter([s7c["random"]["fpga_speedup"]], [s7c["random"]["accuracy_pct"]],
               c="#2980b9", marker="x", s=80, zorder=4)
    ax.scatter([s7c["controller"]["fpga_speedup"]], [s7c["controller"]["accuracy_pct"]],
               c="#8e44ad", marker="*", s=120, zorder=4,
               label="Gumbel controller (drifts to $p_4{=}0.96$)")
    ax.scatter([s7c["kv_norm"]["fpga_speedup"]], [s7c["kv_norm"]["accuracy_pct"]],
               c="#d35400", marker="v", s=65, zorder=4,
               label="KV-norm heuristic")
    kv_inv = d_7f["metrics"]
    ax.scatter([kv_inv["fpga_speedup"]], [kv_inv["accuracy_pct"]],
               c="#d35400", marker="^", s=65, zorder=4, facecolors="none",
               linewidths=1.5, label="KV-norm inverted (identical!)")

    # Shaded "flat plateau" band
    ax.axhspan(d_7d["summary"]["accuracy_mean"] - 1.5 * d_7d["summary"]["accuracy_std"],
               d_7d["summary"]["accuracy_mean"] + 1.5 * d_7d["summary"]["accuracy_std"],
               alpha=0.10, color="#27ae60", label="Phase 7d $\\pm 1.5\\sigma$ band")

    ax.set_xlabel("FPGA speedup vs. FP16 (BRAM cost model)", fontsize=10)
    ax.set_ylabel("HellaSwag accuracy (%)", fontsize=10)
    ax.set_title("SmolLM-1.7B: accuracy–speedup Pareto frontier",
                 fontsize=11)
    ax.set_xlim(0.8, 3.7)
    ax.set_ylim(39, 50.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.95)
    savefig("pareto_frontier")
    print(f"  → figures/pareto_frontier.{{pdf,png}}")


# --------------------------------------------------------------- routing
def routing_figure():
    d_7c = json.load(open(PHASE7 / "phase7c_routing_ablation.json"))
    d_7d = json.load(open(PHASE7 / "phase7d_random_multiseed.json"))
    d_7f = json.load(open(PHASE7 / "phase7f_kv_norm_inverted.json"))

    s7c = d_7c["strategies"]; kv_inv = d_7f["metrics"]
    labels = ["Random\n($p_4{=}0.74$)",
              "KV-norm\n(low$\\to$4b)",
              "KV-norm\ninverted\n(high$\\to$4b)",
              "Gumbel\ncontroller\n($p_4{=}0.96$)"]
    accs   = [s7c["random"]["accuracy_pct"],
              s7c["kv_norm"]["accuracy_pct"],
              kv_inv["accuracy_pct"],
              s7c["controller"]["accuracy_pct"]]
    colors = ["#2980b9", "#d35400", "#d35400", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, accs, color=colors, edgecolor="black", alpha=0.85)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.25, f"{a:.1f}%",
                ha="center", fontsize=9, fontweight="bold")

    # Phase 7d 5-seed band
    mean7d = d_7d["summary"]["accuracy_mean"]
    std7d  = d_7d["summary"]["accuracy_std"]
    ax.axhspan(mean7d - std7d, mean7d + std7d,
               alpha=0.15, color="#27ae60",
               label=f"Phase 7d 5-seed $\\pm 1\\sigma$ = {mean7d:.2f}$\\pm${std7d:.2f}pp")
    ax.axhline(mean7d, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.8)

    # DWB reference line
    ax.axhline(48.6, color="#c0392b", linestyle=":", linewidth=1.2, alpha=0.9)
    ax.text(len(labels) - 0.5, 48.75, "DWB: 48.6%",
            ha="right", fontsize=8.5, color="#c0392b")

    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("HellaSwag accuracy (%)", fontsize=10)
    ax.set_ylim(41, 52)
    ax.set_title("SmolLM-1.7B routing-strategy ablation ($n{=}200$, seed 0)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    savefig("routing_ablation")
    print("  → figures/routing_ablation.{pdf,png}")


# ------------------------------------------------------------ per-layer
def per_layer_figure():
    d = json.load(open(PHASE7 / "per_layer_q_local.json"))
    rows = d["per_layer"]
    layers = [r["layer"] for r in rows]
    q4     = [r["q4_mean"] for r in rows]
    q8     = [r["q8_mean"] for r in rows]
    norms  = [r["norm_mean"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    ax1.plot(layers, q4, "o-", color="#2980b9", label="$q_{t,4}^{\\mathrm{local}}$ (4-bit fidelity)")
    ax1.plot(layers, q8, "s-", color="#27ae60", label="$q_{t,8}^{\\mathrm{local}}$ (8-bit fidelity)")
    ax1.scatter([layers[-1]], [q4[-1]], s=120, marker="o",
                facecolors="none", edgecolors="#c0392b", linewidths=2, zorder=5,
                label=f"L{layers[-1]} outlier (q4$\\,$= {q4[-1]:.2f})")
    ax1.set_xlabel("Transformer layer index", fontsize=10)
    ax1.set_ylabel("Mean quality proxy  $q_{t,b}^{\\mathrm{local}}$", fontsize=10)
    ax1.set_ylim(0.40, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(layers, norms, "^--", color="#888", alpha=0.6,
             label="Mean $\\|[k;v]\\|_2$ (right axis)")
    ax2.set_ylabel("KV L2 norm (mean)", fontsize=10, color="#666")
    ax2.tick_params(axis="y", colors="#666")
    ax2.legend(loc="center right", fontsize=8)

    ax1.set_title("SmolLM-1.7B per-layer quantization quality (30 texts)",
                  fontsize=11)
    savefig("per_layer_q_local")
    print("  → figures/per_layer_q_local.{pdf,png}")


if __name__ == "__main__":
    # TeX-friendly rendering without needing a TeX installation on matplotlib's side
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "text.usetex": False,  # rely on matplotlib's mathtext so no tex-on-Python needed
    })
    print("Generating Phase 7 figures:")
    pareto_figure()
    routing_figure()
    per_layer_figure()
    print("Done.")
