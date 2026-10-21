"""
Mixture-test (T3) report: compares the mixture-trained model with the single-cell-trained upper bound from their
eval_summary.json files (eval.py --baseline --mixture_reference with the eval24 paired config and the eval25 library
control), and writes a markdown table and a bar-chart figure. See reports/TODO.md, T3.

Usage:
    python t3_report.py --mixture results/<mix eval run>/eval_summary.json \
                        --upper results/<single-cell eval run>/eval_summary.json \
                        [--pair eval24_pair_GM12878_HepG2_200U] [--control eval25_pair_HepG2_100U_200U] \
                        [--out results/t3_report]

Rows per assay:
  recovery r        each cell's prediction (from its own fibers) vs its own track; the mixture track as H0
  cell-specific r   partial correlation given the mixture track (H0 = 0 by construction)
  differential r    corr(p_A - p_B, t_A - t_B) (H0 = 0: same prediction for both)
  sign accuracy     at the pairwise differential bins
  fiber AUC         oriented fiber-origin AUC at those bins (0.5 = no separation)
  library control   mean |AUC - 0.5| and RMS(p_100U - p_200U) on the same bins / positions
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mixture", required=True, help="eval_summary.json of the mixture-trained model")
    p.add_argument("--upper", default=None, help="eval_summary.json of the single-cell-trained upper bound")
    p.add_argument("--pair", default="eval24_pair_GM12878_HepG2_200U")
    p.add_argument("--control", default="eval25_pair_HepG2_100U_200U")
    p.add_argument("--out", default="results/t3_report")
    return p.parse_args()


def _get(d, *keys):
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def _fmt(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def collect(summary, pair, control):
    """{row label: {assay: value}} for one model."""
    res = summary["results"]
    paired = res.get(f"{pair}/paired", {})
    cells = paired.get("cells", [])
    rows = {}
    for c in cells:
        e = res.get(f"{pair}/{c}", {})
        rows[f"recovery r {c}"] = e.get("per_assay_pearson", {})
        rows[f"H0 mixture r {c}"] = {a: v["pearson"] for a, v in (_get(e, "baseline", "per_assay") or {}).items()}
        rows[f"cell-specific r {c}"] = {a: v["pearson"] for a, v in
                                        (_get(e, "partial", "strata", "all", "per_assay") or {}).items()}
    rows["differential r"] = {a: v["diff_pearson"] for a, v in (_get(paired, "strata", "all", "per_assay") or {}).items()}
    loci = paired.get("loci", {})
    rows["sign accuracy"] = {a: v["sign_accuracy"] for a, v in loci.items()}
    rows["fiber AUC"] = {a: v["fiber_auc"] for a, v in loci.items()}
    rows["|AUC - 0.5|"] = {a: v["fiber_auc_abs_dev"] for a, v in loci.items()}
    rows["RMS p_A - p_B"] = {a: v["rms_dp"] for a, v in (_get(paired, "strata", "all", "per_assay") or {}).items()}
    ctrl = res.get(f"{control}/paired", {})
    rows["library |AUC - 0.5|"] = {a: v["fiber_auc_abs_dev"] for a, v in ctrl.get("loci", {}).items()}
    rows["library RMS p_A - p_B"] = {a: v["rms_dp"] for a, v in (_get(ctrl, "strata", "all", "per_assay") or {}).items()}
    return cells, rows


def main():
    args = parse_args()
    models = {"mixture": json.load(open(args.mixture))}
    if args.upper:
        models["upper bound"] = json.load(open(args.upper))
    collected = {name: collect(s, args.pair, args.control) for name, s in models.items()}
    cells = collected["mixture"][0]
    assays = list(next(iter(collected["mixture"][1].values())).keys()) or []
    for _, rows in collected.values():
        for r in rows.values():
            assays += [a for a in r if a not in assays]

    os.makedirs(args.out, exist_ok=True)
    lines = [f"# T3 report: {' vs '.join(cells)}", "", f"Pair config `{args.pair}`, library control `{args.control}`.",
             "Each cell: `mixture model / upper bound` (H0 rows: the mixture track).", ""]
    labels = list(collected["mixture"][1].keys())
    lines.append("| | " + " | ".join(assays) + " | mean |")
    lines.append("|---" * (len(assays) + 2) + "|")
    for label in labels:
        vals, means = [], []
        for a in assays:
            per = [collected[m][1].get(label, {}).get(a) for m in models]
            if label.startswith("H0"):
                per = per[:1]
            vals.append(" / ".join(_fmt(v) for v in per))
        for m in (list(models)[:1] if label.startswith("H0") else models):
            xs = [v for v in collected[m][1].get(label, {}).values() if v is not None]
            means.append(_fmt(float(np.mean(xs)) if xs else None))
        lines.append(f"| {label} | " + " | ".join(vals) + f" | {' / '.join(means)} |")
    md = os.path.join(args.out, "t3_report.md")
    open(md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Figure: per assay, the key readouts side by side
    panels = [(f"recovery r {c}", f"H0 mixture r {c}") for c in cells] + [("differential r", None),
                                                                           ("fiber AUC", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), squeeze=False)
    x = np.arange(len(assays))
    for ax, (label, h0) in zip(axes[0], panels):
        series = [(m, collected[m][1].get(label, {})) for m in models]
        if h0:
            series = [("H0 (mixture track)", collected["mixture"][1].get(h0, {}))] + series
        w = 0.8 / len(series)
        for j, (name, vals) in enumerate(series):
            ax.bar(x + (j - (len(series) - 1) / 2) * w, [np.nan if vals.get(a) is None else vals[a] for a in assays],
                   width=w, label=name)
        if label == "fiber AUC":
            ax.axhline(0.5, color="k", lw=0.8, ls="--")
            ax.set_ylim(0.3, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([a[:8] for a in assays], rotation=45, ha="right")
        ax.set_title(label)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    png = os.path.join(args.out, "t3_report.png")
    fig.savefig(png, dpi=120)
    print(f"\nWrote {md} and {png}")


if __name__ == "__main__":
    main()
