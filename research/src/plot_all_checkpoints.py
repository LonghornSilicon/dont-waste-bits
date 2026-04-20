"""
Plot gap_mean vs model scale for all 10 checkpoints.
Visualizes the floor attractor and GQA-scale interaction.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = r"C:\Users\themo\Desktop\Dont Waste Bits!\research\paper\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# All 10 checkpoints: (label, params_M, gap_mean, family, marker, color, gqa)
checkpoints = [
    ("SmolLM-135M",   135,  0.330, "SmolLM (MHA)",     "o", "#2196F3", False),
    ("SmolLM-360M",   360,  0.337, "SmolLM (MHA)",     "o", "#2196F3", False),
    ("SmolLM-1.7B",  1700,  0.424, "SmolLM (MHA)",     "o", "#2196F3", False),
    ("SmolLM2-360M",  360,  0.283, "SmolLM2 (GQA)",    "^", "#9C27B0", True),
    ("TinyLlama-1.1B",1100, 0.189, "TinyLlama (GQA)",  "^", "#4CAF50", True),
    ("OPT-125M",      125,  0.213, "OPT (Meta)",       "s", "#FF9800", False),
    ("OPT-350M",      350,  0.181, "OPT (Meta)",       "s", "#FF9800", False),
    ("GPT-2-Small",   124,  0.196, "GPT-2 (OpenAI)",   "D", "#F44336", False),
    ("GPT-2-Medium",  345,  0.188, "GPT-2 (OpenAI)",   "D", "#F44336", False),
    ("GPT-2-Large",   774,  0.192, "GPT-2 (OpenAI)",   "D", "#F44336", False),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ─── Left: scatter gap_mean vs scale ───
ax = axes[0]
ax.set_facecolor('#fafafa')

# Floor band
ax.axhspan(0.178, 0.200, alpha=0.15, color='green', label='Floor zone (0.18–0.20)')
ax.axhline(0.189, color='green', lw=1.2, ls='--', alpha=0.6)

# Plot each checkpoint
legend_handles = {}
for (label, params, gap, family, marker, color, gqa) in checkpoints:
    h = ax.scatter(params, gap, marker=marker, color=color, s=120, zorder=5,
                   edgecolors='white', linewidths=1.2)
    if family not in legend_handles:
        legend_handles[family] = mpatches.Patch(color=color, label=family)

# Annotations
offsets = {
    "SmolLM-135M": (-5, 8), "SmolLM-360M": (5, 8), "SmolLM-1.7B": (5, 6),
    "SmolLM2-360M": (-70, -16), "TinyLlama-1.1B": (10, 6),
    "OPT-125M": (-5, 8), "OPT-350M": (5, -16),
    "GPT-2-Small": (-5, 8), "GPT-2-Medium": (-70, 6), "GPT-2-Large": (5, 6),
}
for (label, params, gap, family, marker, color, gqa) in checkpoints:
    dx, dy = offsets.get(label, (5, 6))
    ax.annotate(label, (params, gap), xytext=(dx, dy),
                textcoords='offset points', fontsize=7, color='#333333',
                ha='left' if dx >= 0 else 'right')

# GQA gradient arrow
ax.annotate('', xy=(1100, 0.189), xytext=(360, 0.283),
            arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.5, alpha=0.7))
ax.text(500, 0.238, 'GQA+scale\nconverges to floor', fontsize=7.5, color='#9C27B0', alpha=0.85)

ax.set_xscale('log')
ax.set_xlabel('Model parameters (M, log scale)', fontsize=10)
ax.set_ylabel('gap_mean = E[q₈ − q₄]', fontsize=10)
ax.set_title('All 10 checkpoints: gap_mean vs scale\n(floor attractor ≈ 0.18–0.19)', fontsize=10)
ax.set_xlim(80, 2500)
ax.set_ylim(0.15, 0.46)
handles = list(legend_handles.values()) + [
    mpatches.Patch(color='green', alpha=0.3, label='Floor zone (0.18–0.20)')
]
ax.legend(handles=handles, fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, ls='--')

# ─── Right: formula accuracy bar chart ───
ax2 = axes[1]
ax2.set_facecolor('#fafafa')

labels_short = ["SL-135M","SL-360M","SL-1.7B","SL2-360M","TL-1.1B","OPT-125","OPT-350","GP2-S","GP2-M","GP2-L"]
errors = [0.022, 0.030, 0.014, 0.044, 0.003, 0.023, 0.021, 0.017, 0.014, 0.010]
colors_bar = ["#2196F3","#2196F3","#2196F3","#9C27B0","#4CAF50","#FF9800","#FF9800","#F44336","#F44336","#F44336"]

bars = ax2.bar(range(10), errors, color=colors_bar, edgecolor='white', linewidth=0.8, alpha=0.85)
ax2.axhline(0.040, color='#333', lw=1.5, ls='--', label='±0.04 bound (9 of 10)')
ax2.axhline(0.050, color='#888', lw=1.0, ls=':', label='±0.05 bound (all 10)')

# Highlight the borderline bar
bars[3].set_edgecolor('#9C27B0')
bars[3].set_linewidth(2.5)

ax2.set_xticks(range(10))
ax2.set_xticklabels(labels_short, rotation=40, ha='right', fontsize=8)
ax2.set_ylabel('Formula error |β_crossing − β*|', fontsize=10)
ax2.set_title('β* = gap_mean/0.267: prediction error\nacross all 10 checkpoints', fontsize=10)
ax2.set_ylim(0, 0.058)
ax2.legend(fontsize=8)
ax2.grid(True, axis='y', alpha=0.3, ls='--')

# Value labels on bars
for i, v in enumerate(errors):
    ax2.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout(pad=1.5)
out_path = os.path.join(OUT_DIR, "all_checkpoints_summary.pdf")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
out_png = os.path.join(OUT_DIR, "all_checkpoints_summary.png")
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
print(f"Saved: {out_png}")
