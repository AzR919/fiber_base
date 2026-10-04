"""
One-off script for the ceehrc_26 poster/talk.

Publication-quality grouped bar chart of HepG2_200U per-assay MSE loss across
all 5 training configurations, using the values already reported in
reports/Sep_25.md (HepG2_200U columns only) — hardcoded here rather than
parsed, since the report's numbers are the finalized source of truth.

Usage: python plots_code/ceehrc_26/hepg2_mse_barchart.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_PATH = os.path.join(REPO_ROOT, "plots", "ceehrc_26", "hepg2_mse_barchart.png")

# HepG2_200U columns only (ATAC, H3K4me3, H3K27ac, H3K27me3, Avg), from reports/Sep_25.md.
ASSAYS = ["ATAC", "H3K4me3", "H3K27ac", "H3K27me3", "Avg"]

DATA = {
    "GM12878":              [0.6141, 0.5814, 0.5086, 0.1198, 0.4560],
    "K562":                 [0.6421, 0.3973, 0.4852, 0.1802, 0.4262],
    "HepG2":                [0.7658, 0.3564, 0.4515, 0.1126, 0.4216],
    "GM12878+K562":         [0.6184, 0.4333, 0.4724, 0.1274, 0.4129],
    "GM12878+K562+HepG2":   [0.6523, 0.3896, 0.4900, 0.1170, 0.4122],
}

MODELS = list(DATA.keys())


def main():
    n_models = len(MODELS)
    n_groups = len(ASSAYS)
    x = np.arange(n_groups)
    bar_width = 0.8 / n_models

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, model in enumerate(MODELS):
        offset = (i - (n_models - 1) / 2) * bar_width
        values = DATA[model]
        bars = ax.bar(
            x + offset, values, width=bar_width, label=model,
            color=colors[i], edgecolor='black', linewidth=0.6,
        )
        for rect, v in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.012,
                f"{v:.3f}", ha='center', va='bottom', fontsize=7, rotation=90,
            )

    # Visual separator before "Avg" — it's an aggregate, not a raw assay.
    sep_x = (x[-2] + x[-1]) / 2
    ax.axvline(sep_x, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(ASSAYS, fontsize=13, fontweight='bold')
    ax.set_xlabel("Assay", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel("MSE Loss", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(
        "HepG2 Performance Across Training Configurations",
        fontsize=16, fontweight='bold', pad=18,
    )
    ax.set_ylim(0, max(max(v) for v in DATA.values()) * 1.18)

    ax.legend(
        title="Model Trained On", fontsize=9.5, title_fontsize=10.5,
        loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False,
    )

    ax.yaxis.grid(True, linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
