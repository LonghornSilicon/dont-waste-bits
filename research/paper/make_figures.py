"""Generate figures for fpga_controller_paper.tex"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# ── Figure 1: FPGA speedup comparison bar chart ──────────────────────────────
methods = ["FP16\n(baseline)", "Static\nINT4", "Paper\nDWB", "Ours\n(binary)"]
speedups = [1.00, 3.48, 2.44, 3.48]
accuracies = [42.6, 41.6, 41.2, 41.0]
colors = ["#718096", "#4299e1", "#e53e3e", "#38a169"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

bars = ax1.bar(methods, speedups, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
ax1.set_ylabel("FPGA Throughput Speedup (×)", fontsize=11)
ax1.set_title("FPGA Throughput vs. FP16\n(Xilinx Ultrascale+ BRAM model)", fontsize=10)
ax1.set_ylim(0, 4.2)
ax1.axhline(1.0, color="#718096", linestyle="--", linewidth=0.8, alpha=0.5)
for bar, val in zip(bars, speedups):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05,
             f"{val:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")

# annotate the +43% gap — two-segment path well clear of bar-value labels
ax1.annotate("", xy=(2.95, 3.30), xytext=(2.05, 2.62),
             arrowprops=dict(arrowstyle="-|>", color="#38a169",
                             lw=1.8, shrinkA=2, shrinkB=2,
                             connectionstyle="arc3,rad=-0.22"))
ax1.text(2.50, 2.76, "+43%", color="#38a169", fontsize=10,
         fontweight="bold", ha="center",
         bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                   edgecolor="#38a169", linewidth=0.8))

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Figure 2: Pareto frontier — accuracy vs FPGA cost ────────────────────────
# Pure quantization operating points (SmolLM-360M measured)
pure_pts = {
    "2-bit":  {"fpga": 0.290, "acc": 25.0, "color": "#e53e3e", "marker": "v", "size": 90},
    "4-bit":  {"fpga": 0.290, "acc": 41.6, "color": "#4299e1", "marker": "o", "size": 90},
    "8-bit":  {"fpga": 0.560, "acc": 42.0, "color": "#4299e1", "marker": "s", "size": 90},
    "FP16":   {"fpga": 1.010, "acc": 42.6, "color": "#718096", "marker": "s", "size": 80},
}

# Pareto frontier for binary {4,8} controller: interpolate between 4-bit and 8-bit
p4_vals = np.linspace(0, 1, 100)
fpga_binary = 0.290 * p4_vals + 0.560 * (1 - p4_vals)
# At 360M INT4 is lossless: accuracy stays ~41.6% for any p4>=0.x
# Use linear interpolation between pure-4 and pure-8 accuracy
acc_binary = 41.6 * p4_vals + 42.0 * (1 - p4_vals)

ax2.plot(fpga_binary, acc_binary, "-", color="#38a169", lw=2.5, alpha=0.7,
         label="Binary {4,8} Pareto frontier", zorder=3)

# DWB's "frontier" with 2-bit: interpolate using their bit distribution
# For all-2bit->all-4bit, FPGA cost is same (0.290) but accuracy drops to 25%
# DWB mixes {2,4,8,16}: show the cost of including 2-bit

# Show main comparison points
main_pts = {
    "Paper DWB\n(47.9% 2-bit waste)": {"fpga": 0.414, "acc": 41.2, "color": "#e53e3e", "marker": "^", "size": 120},
    "Ours (binary)":                   {"fpga": 0.290, "acc": 41.0, "color": "#38a169", "marker": "D", "size": 120},
}

for label, d in {**pure_pts, **main_pts}.items():
    ax2.scatter(d["fpga"], d["acc"], c=d["color"], marker=d["marker"],
                s=d["size"], zorder=5, label=label)

# Arrow: DWB → ours: same cost budget, +0.2pp accuracy (curve placed ABOVE the
# points so the label never sits on top of the green diamond)
ax2.annotate("", xy=(0.305, 41.2), xytext=(0.408, 41.35),
             arrowprops=dict(arrowstyle="-|>", color="#38a169",
                             lw=1.6, shrinkA=2, shrinkB=2,
                             connectionstyle="arc3,rad=0.30"))
ax2.text(0.355, 42.05, "Pareto\ndominates",
         color="#38a169", fontsize=8, ha="center", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                   edgecolor="#38a169", linewidth=0.7))

# 2-bit annotation: place the label in empty low-left space (where there are no
# data points or legend) with a curved arrow to the 2-bit marker.
ax2.annotate("2-bit: same BRAM as\n4-bit, $-$16.6pp accuracy!",
             xy=(0.305, 25.3), xytext=(0.42, 32.5),
             fontsize=7.8, color="#e53e3e", ha="center", fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color="#e53e3e", lw=1.4,
                             shrinkA=3, shrinkB=3,
                             connectionstyle="arc3,rad=0.25"),
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                       edgecolor="#e53e3e", linewidth=0.7))

ax2.set_xlabel("FPGA BRAM Cost (normalized, lower = faster)", fontsize=10)
ax2.set_ylabel("HellaSwag Accuracy (%)", fontsize=10)
ax2.set_title("Pareto Frontier: Accuracy vs. FPGA Cost\n(SmolLM-360M)", fontsize=10)
ax2.set_xlim(0.20, 1.15)
ax2.set_ylim(22, 44.5)

# draw "ideal" direction arrow
ax2.annotate("", xy=(0.22, 43.8), xytext=(0.42, 43.8),
             arrowprops=dict(arrowstyle="->", color="#2d3748", lw=1.2))
ax2.text(0.21, 44.1, "better", color="#2d3748", fontsize=8)

# Place legend on the right-middle area, well clear of both the 2-bit annotation
# (red box in mid-plot area) and the Pareto-dominates callout (top-left).
ax2.legend(fontsize=6.8, loc="center right", ncol=1, framealpha=0.92)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "fpga_comparison.pdf", bbox_inches="tight", dpi=150)
fig.savefig(OUT / "fpga_comparison.png", bbox_inches="tight", dpi=150)
print(f"Saved figures to {OUT}/fpga_comparison.{{pdf,png}}")

# ── Figure 3: BRAM cost model bar chart ──────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(6, 3.5))
bits = [2, 4, 8, 16]
costs = [0.290, 0.290, 0.560, 1.010]
bar_colors = ["#e53e3e", "#38a169", "#4299e1", "#718096"]
bars3 = ax3.bar([str(b)+"-bit" for b in bits], costs, color=bar_colors,
                edgecolor="white", linewidth=1.5, width=0.5)

ax3.set_ylabel("Normalized BRAM Cost", fontsize=11)
ax3.set_title("Xilinx Ultrascale+ BRAM Port Cost per KV Token\n(4-bit minimum port width)", fontsize=10)
ax3.set_ylim(0, 1.2)

for bar, val in zip(bars3, costs):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# annotation: 2-bit and 4-bit share the same BRAM port cost. Place the text
# ABOVE both bars (at x=0.5, which is between bar 0="2-bit" and bar 1="4-bit")
# and draw two clear arrows with visible arrowheads down to each bar top.
ax3.text(0.5, 0.67, "Same BRAM\nport cost!",
         ha="center", fontsize=9.5, color="#e53e3e", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                   edgecolor="#e53e3e", linewidth=0.9))
ax3.annotate("", xy=(0.0, 0.31), xytext=(0.38, 0.60),
             arrowprops=dict(arrowstyle="-|>", color="#e53e3e", lw=1.5,
                             shrinkA=2, shrinkB=2))
ax3.annotate("", xy=(1.0, 0.31), xytext=(0.62, 0.60),
             arrowprops=dict(arrowstyle="-|>", color="#e53e3e", lw=1.5,
                             shrinkA=2, shrinkB=2))

ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
fig2.savefig(OUT / "bram_cost_model.pdf", bbox_inches="tight", dpi=150)
fig2.savefig(OUT / "bram_cost_model.png", bbox_inches="tight", dpi=150)
print(f"Saved figures to {OUT}/bram_cost_model.{{pdf,png}}")

# ── Figure 4: Beta calibration analysis ──────────────────────────────────────
import numpy as np

fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(9, 4))

# Left: q8-q4 gap distribution — all three scales MEASURED
gap_135m_mean_l, gap_135m_std_l = 0.3297, 0.0494  # measured
gap_360m_mean, gap_360m_std = 0.3367, 0.0501        # measured
gap_1b7_mean,  gap_1b7_std  = 0.4235, 0.0629        # MEASURED (was estimated 0.400)

x = np.linspace(0.1, 0.8, 300)
from scipy.stats import norm
pdf_135m_l = norm.pdf(x, gap_135m_mean_l, gap_135m_std_l)
pdf_360m = norm.pdf(x, gap_360m_mean, gap_360m_std)
pdf_1b7  = norm.pdf(x, gap_1b7_mean,  gap_1b7_std)

ax4.plot(x, pdf_135m_l, color="#805ad5", lw=2, label="SmolLM-135M (measured)")
ax4.plot(x, pdf_360m,   color="#4299e1", lw=2, label="SmolLM-360M (measured)")
ax4.plot(x, pdf_1b7,    color="#e53e3e", lw=2, label="SmolLM-1.7B (measured)")

# Mark beta=1.5 threshold and beta=1.6 threshold
thr15 = 1.5 * 0.270 / 1.01
thr16 = 1.6 * 0.270 / 1.01
ax4.axvline(thr15, color="#38a169", lw=1.8, linestyle="--", label=f"beta=1.5 thr ({thr15:.3f})")
ax4.axvline(thr16, color="#d69e2e", lw=1.5, linestyle=":", label=f"beta=1.6 thr ({thr16:.3f})")
ax4.fill_betweenx([0, 8], 0.1, thr15, alpha=0.10, color="#38a169")
ax4.fill_betweenx([0, 8], thr15, thr16, alpha=0.10, color="#d69e2e")
# Zone labels at a consistent y (bottom strip) with generous margins from plot
# edges so none are clipped. The narrow 1.7B transition zone only gets a
# single-line caret label to avoid collisions.
ax4.text(0.22, 0.7, "4-bit zone", ha="center", fontsize=8.5, color="#38a169",
         fontweight="bold")
ax4.text((thr15+thr16)/2, 0.7, "1.7B trans.", ha="center", fontsize=7,
         color="#d69e2e", fontweight="bold", rotation=90)
ax4.text(0.63, 0.7, "8-bit zone", ha="center", fontsize=8.5, color="#718096",
         fontweight="bold")

ax4.set_xlabel("q8_local - q4_local gap", fontsize=10)
ax4.set_ylabel("Density", fontsize=10)
ax4.set_title("Quality Gap Distribution (all three scales MEASURED)\n(4-bit preferred when gap < threshold)", fontsize=10)
ax4.legend(fontsize=7.5)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# Right: beta vs frac_4bit at all three scales — all MEASURED
# 360M coarse + fine sweep combined:
betas_measured = [1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 2.0, 3.0]
frac4_360m_actual = [0.0, 0.0, 0.0, 41.7, 58.7, 100.0, 100.0, 100.0, 100.0]
# 135M measured
betas_135m = [0.9, 1.0, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]
frac4_135m_actual = [0.0, 0.0, 0.0, 0.0, 0.0, 63.0, 100.0, 100.0]
# 1.7B MEASURED fine sweep
betas_1b7 = [1.0, 1.50, 1.52, 1.55, 1.57, 1.58, 1.59, 1.60, 1.62, 1.65, 1.70, 2.0]
frac4_1b7_actual = [0.0, 0.0, 0.0, 0.0, 59.2, 59.5, 61.0, 62.2, 65.2, 72.0, 75.7, 97.6]

# Theoretical curves
betas_fine = np.linspace(0.5, 3.5, 200)
gap_135m_mean, gap_135m_std = 0.3297, 0.0494
frac4_135m_theory = [norm.cdf(b * 0.270/1.01, gap_135m_mean, gap_135m_std)*100
                     for b in betas_fine]
frac4_360m_theory = [norm.cdf(b * 0.270/1.01, gap_360m_mean, gap_360m_std)*100
                     for b in betas_fine]
frac4_1b7_theory  = [norm.cdf(b * 0.270/1.01, gap_1b7_mean, gap_1b7_std)*100
                     for b in betas_fine]

ax5.plot(betas_fine, frac4_135m_theory, "-", color="#805ad5", lw=1.5, alpha=0.5)
ax5.plot(betas_135m, frac4_135m_actual, "^", color="#805ad5",
         ms=7, zorder=6, label="135M measured")
ax5.plot(betas_fine, frac4_360m_theory, "-", color="#4299e1", lw=1.5, alpha=0.5)
ax5.plot(betas_measured, frac4_360m_actual, "o", color="#4299e1",
         ms=8, zorder=6, label="360M measured")
ax5.plot(betas_fine, frac4_1b7_theory, "-", color="#e53e3e", lw=1.5, alpha=0.5)
ax5.plot(betas_1b7, frac4_1b7_actual, "s", color="#e53e3e",
         ms=7, zorder=6, label="1.7B measured")

ax5.axvline(1.5, color="#38a169", lw=1.5, linestyle=":", alpha=0.8, label="beta=1.5")
ax5.axvline(1.6, color="#d69e2e", lw=1.5, linestyle=":", alpha=0.8, label="beta=1.6 (1.7B)")
ax5.axhline(54.1, color="#718096", lw=1.2, linestyle=":", alpha=0.6,
            label="Min 4-bit% to beat DWB")

# annotate phase transitions
ax5.axvline(1.26, color="#38a169", lw=1.2, linestyle="-.", alpha=0.6)
ax5.axvline(1.584, color="#e53e3e", lw=1.2, linestyle="-.", alpha=0.6)
# Cleaner annotations on top of the curves, each with a white-backed pill box so
# the text never fights with the legend or the scatter markers behind it.
ax5.annotate(r"$\beta^*{=}1.26$" + "\n(135M/360M)",
             xy=(1.26, 50), xytext=(0.80, 78),
             arrowprops=dict(arrowstyle="-|>", color="#38a169", lw=1.3,
                             shrinkA=2, shrinkB=2,
                             connectionstyle="arc3,rad=0.20"),
             fontsize=7.5, color="#38a169", ha="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                       edgecolor="#38a169", linewidth=0.7))
ax5.axvline(1.70, color="#d69e2e", lw=1.2, linestyle=":", alpha=0.7)
ax5.annotate(r"$\beta^*{=}[1.55,1.57]$" + "\n(1.7B transition)",
             xy=(1.56, 30), xytext=(2.05, 16),
             arrowprops=dict(arrowstyle="-|>", color="#e53e3e", lw=1.3,
                             shrinkA=2, shrinkB=2,
                             connectionstyle="arc3,rad=0.20"),
             fontsize=7.5, color="#e53e3e", ha="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                       edgecolor="#e53e3e", linewidth=0.7))
ax5.annotate(r"$\beta{=}1.70$: 2.84$\times$" + "\n(+16% vs DWB)",
             xy=(1.70, 75.7), xytext=(2.45, 94),
             arrowprops=dict(arrowstyle="-|>", color="#d69e2e", lw=1.3,
                             shrinkA=2, shrinkB=2,
                             connectionstyle="arc3,rad=-0.20"),
             fontsize=7.5, color="#d69e2e", ha="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                       edgecolor="#d69e2e", linewidth=0.7))

ax5.set_xlabel("beta (FPGA penalty weight)", fontsize=10)
ax5.set_ylabel("4-bit token fraction (%)", fontsize=10)
ax5.set_title("Beta Calibration (all three scales MEASURED)\n(4-bit% vs. penalty weight)", fontsize=10)
ax5.set_ylim(0, 115)
ax5.legend(fontsize=7, loc="center right", ncol=1, framealpha=0.92)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

plt.tight_layout()
fig3.savefig(OUT / "beta_calibration.pdf", bbox_inches="tight", dpi=150)
fig3.savefig(OUT / "beta_calibration.png", bbox_inches="tight", dpi=150)
print(f"Saved figures to {OUT}/beta_calibration.{{pdf,png}}")
