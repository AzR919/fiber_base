"""Nov_20.md section 1 with and without the CTCF sites that have no bulk ChIP data (ctcf_holes.py): compares
ctcf_footprints.py's summaries on all sites (plots/26_nov_13/molecules/) with the rerun without the holes
(plots/26_nov_20/ctcf_noholes/), for every model and training seed.

    python plots_code/26_nov_20/ctcf_noholes_compare.py
Writes plots/26_nov_20/ctcf_noholes/compare.tsv and compare_h5.tsv; prints the tables.
"""
import os

import pandas as pd

OLD, NEW = "plots/26_nov_13/molecules", "plots/26_nov_20/ctcf_noholes"
MODELS = {"DEV4NF": "noFIRE", "DEV4": "DEV4", "m6A + CpG": "m6aCpG", "no accessibility": "noAcc", "no CpG": "noCpG"}
SEEDS = [919, 2]
COLS = ["n_sites", "n_fibers", "footprint_frac_of_accessible", "pred_ctcf_site_mean", "d_fp_open_ctcf_median",
        "d_fp_open_ctcf_frac_pos", "d_fp_open_ctcf_p", "d_fp_open_ctcf_n", "d_ctcf_minus_atac_median",
        "d_ctcf_minus_atac_p", "d_open_nuc_ctcf_median"]


def tag(key, seed):
    return key if seed == 919 else f"{key}_seed{seed}"


def main():
    rows, h5 = [], []
    for label, key in MODELS.items():
        for seed in SEEDS:
            for which, d in (("all sites", OLD), ("no holes", NEW)):
                s = pd.read_csv(os.path.join(d, f"ctcf_summary_{tag(key, seed)}.tsv"), sep="\t")
                for r in s.to_dict("records"):
                    rows.append({"model": label, "seed": seed, "sites": which, "sample": r["sample"],
                                 "site_set": r["site_set"], **{c: r[c] for c in COLS}})
                b = pd.read_csv(os.path.join(d, f"ctcf_bound_vs_unbound_by_score_{tag(key, seed)}.tsv"), sep="\t")
                for sample, g in b.groupby("sample"):
                    h5.append({"model": label, "seed": seed, "sites": which, "sample": sample, "bins": len(g),
                               "bins_bound_higher": int((g.bound_mean > g.unbound_mean).sum()),
                               "n_unbound": int(g.n_unbound.sum())})
    df, h5 = pd.DataFrame(rows), pd.DataFrame(h5)
    df.to_csv(os.path.join(NEW, "compare.tsv"), sep="\t", index=False, float_format="%.4g")
    h5.to_csv(os.path.join(NEW, "compare_h5.tsv"), sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
        for st in ["unbound", "bound", "bound_other_line"]:
            print(f"\n== {st}")
            print(df[df.site_set == st].pivot_table(
                index=["model", "seed", "sample"], columns="sites", sort=False,
                values=["n_sites", "d_fp_open_ctcf_median", "d_fp_open_ctcf_p", "d_ctcf_minus_atac_median",
                        "pred_ctcf_site_mean", "footprint_frac_of_accessible"]).round(4).to_string())
        print("\n== H5 bins (bound > unbound at matched motif score)")
        print(h5.pivot_table(index=["model", "seed", "sample"], columns="sites", sort=False,
                             values=["bins", "bins_bound_higher", "n_unbound"]).to_string())


if __name__ == "__main__":
    main()
