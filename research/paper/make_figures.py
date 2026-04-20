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

# annotate the +43% gap
ax1.annotate("", xy=(3, 3.48), xytext=(2, 2.44),
             arrowprops=dict(arrowstyle="->", color="#38a169", lw=1.5))
ax1.text(2.55, 3.05, "+43%", color="#38a169", fontsize=9, fontweight="bold")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Figure 2: accuracy vs FPGA cost scatter ───────────────────────────────────
data = {
    "FP16":          {"fpga": 1.010, "acc": 42.6, "color": "#718096", "marker": "s", "size": 80},
    "Static INT4":   {"fpga": 0.290, "acc": 41.6, "color": "#4299e1", "marker": "o", "size": 80},
    "Paper DWB":     {"fpga": 0.414, "acc": 41.2, "color": "#e53e3e", "marker": "^", "size": 100},
    "Ours (binary)": {"fpga": 0.290, "acc": 41.0, "color": "#38a169", "marker": "D", "size": 100},
}

for label, d in data.items():
    ax2.scatter(d["fpga"], d["acc"], c=d["color"], marker=d["marker"],
                s=d["size"], zorder=5, label=label)

ax2.set_xlabel("FPGA BRAM Cost (normalized, lower = faster)", fontsize=10)
ax2.set_ylabel("HellaSwag Accuracy (%)", fontsize=10)
ax2.set_title("Accuracy vs. FPGA Cost\n(SmolLM-360M, 200-sample HellaSwag)", fontsize=10)
ax2.set_xlim(0.20, 1.15)
ax2.set_ylim(38, 44)

# draw "ideal" direction arrow
ax2.annotate("", xy=(0.22, 43.5), xytext=(0.38, 43.5),
             arrowprops=dict(arrowstyle="->", color="#2d3748", lw=1.2))
ax2.text(0.22, 43.7, "better →", color="#2d3748", fontsize=8)

# shade dominated region
ax2.fill_betweenx([38, 44], 0.414, 1.15, alpha=0.06, color="#e53e3e",
                  label="DWB BRAM cost zone")

ax2.legend(fontsize=8, loc="lower left")
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

# annotation: 2-bit = 4-bit cost
ax3.annotate("Same BRAM\nport cost!", xy=(0.5, 0.295), xytext=(0.5, 0.55),
             arrowprops=dict(arrowstyle="-", color="#e53e3e", lw=1.5,
                             connectionstyle="arc3,rad=0"),
             fontsize=9, color="#e53e3e", ha="center")
ax3.annotate("", xy=(1.5, 0.295), xytext=(1.0, 0.55),
             arrowprops=dict(arrowstyle="-", color="#e53e3e", lw=1.5))

ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
fig2.savefig(OUT / "bram_cost_model.pdf", bbox_inches="tight", dpi=150)
fig2.savefig(OUT / "bram_cost_model.png", bbox_inches="tight", dpi=150)
print(f"Saved figures to {OUT}/bram_cost_model.{{pdf,png}}")

# ── Figure 4: Beta calibration analysis ──────────────────────────────────────
import numpy as np

fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(9, 4))

# Left: q8-q4 gap distribution at 360M (measured) and 1.7B (estimated)
gap_360m_mean, gap_360m_std = 0.3367, 0.0501
gap_1b7_mean,  gap_1b7_std  = 0.40,   0.058   # estimated: larger errors at 1.7B

x = np.linspace(0.1, 0.7, 300)
from scipy.stats import norm
pdf_360m = norm.pdf(x, gap_360m_mean, gap_360m_std)
pdf_1b7  = norm.pdf(x, gap_1b7_mean,  gap_1b7_std)

ax4.plot(x, pdf_360m, color="#4299e1", lw=2, label="SmolLM-360M (measured)")
ax4.plot(x, pdf_1b7,  color="#e53e3e", lw=2, label="SmolLM-1.7B (estimated)")

# Mark beta=1.5 threshold
thr15 = 1.5 * 0.270 / 1.01
ax4.axvline(thr15, color="#38a169", lw=1.8, linestyle="--", label=f"beta=1.5 threshold ({thr15:.3f})")
ax4.fill_betweenx([0, 8], 0.1, thr15, alpha=0.12, color="#38a169")
ax4.text(thr15 - 0.005, 7.5, "4-bit\nzone", ha="right", fontsize=8, color="#38a169")
ax4.text(thr15 + 0.005, 7.5, "8-bit\nzone", ha="left", fontsize=8, color="#718096")

ax4.set_xlabel("q8_local - q4_local gap", fontsize=10)
ax4.set_ylabel("Density", fontsize=10)
ax4.set_title("Quality Gap Distribution\n(4-bit preferred when gap < threshold)", fontsize=10)
ax4.legend(fontsize=8)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# Right: beta vs frac_4bit at 360M (actual controller outcomes) and predicted 1.7B
# Actual controller training outcomes from smoke test (hard 0/100% transitions):
betas_measured = [1.0, 1.5, 2.0, 3.0]
frac4_360m_actual = [0.0, 100.0, 100.0, 100.0]  # actual trained controller outputs

# Theoretical curve for 360M (fraction of tokens where gap < threshold)
betas_fine = np.linspace(0.5, 3.5, 200)
frac4_360m_theory = [norm.cdf(b * 0.270/1.01, gap_360m_mean, gap_360m_std)*100
                     for b in betas_fine]
# Predict 1.7B (estimated gap distribution)
frac4_1b7_pred = [norm.cdf(b * 0.270/1.01, gap_1b7_mean, gap_1b7_std)*100
                  for b in betas_fine]

ax5.plot(betas_fine, frac4_360m_theory, "-", color="#4299e1", lw=1.5,
         alpha=0.5, label="360M theoretical")
ax5.plot(betas_measured, frac4_360m_actual, "o", color="#4299e1",
         ms=8, zorder=6, label="360M measured (controller)")
ax5.plot(betas_fine, frac4_1b7_pred, "--", color="#e53e3e",
         lw=2, label="1.7B predicted")

ax5.axvline(1.5, color="#38a169", lw=1.5, linestyle=":", alpha=0.8, label="beta=1.5")
ax5.axhline(54.1, color="#718096", lw=1.2, linestyle=":", alpha=0.6,
            label="Min 4-bit% to beat DWB (54.1%)")

# annotate the sharp transition point
ax5.annotate("Phase\ntransition", xy=(1.5, 50), xytext=(2.0, 30),
             arrowprops=dict(arrowstyle="->", color="#38a169", lw=1.2),
             fontsize=8, color="#38a169")

ax5.set_xlabel("beta (FPGA penalty weight)", fontsize=10)
ax5.set_ylabel("4-bit token fraction (%)", fontsize=10)
ax5.set_title("Beta Calibration\n(4-bit% vs. penalty weight)", fontsize=10)
ax5.set_ylim(0, 115)
ax5.legend(fontsize=7.5, loc="upper left")
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

plt.tight_layout()
fig3.savefig(OUT / "beta_calibration.pdf", bbox_inches="tight", dpi=150)
fig3.savefig(OUT / "beta_calibration.png", bbox_inches="tight", dpi=150)
print(f"Saved figures to {OUT}/beta_calibration.{{pdf,png}}")
