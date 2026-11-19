"""Pick the loci for the single-molecule figures of reports/Nov_20_figs.md, from measured data only (fiber states and
inputs, never model predictions), so the choice cannot favour a model.

CTCF (GM12878, ctcf_sites_chr21.bed, fiber states as in plots_code/26_nov_13/ctcf_footprints.py):
  bound sites    GM12878 has a CTCF peak; the N_BOUND sites with the most fibers in the smaller of the footprint /
                 open states (so both states are well populated)
  unbound site   the N_UNBOUND unbound motif sites with the most accessible (open + footprint) fibers, among the
                 sites with bulk short-read data (not in ctcf_holes.py's holes.txt; earlier picks without this
                 filter, ctcf_unbound_26 and _52, were in regions with no CTCF / no ATAC reads)
X inactivation (GM12878, tss_chrX_chr21.bed, per-gene inputs from plots_code/26_nov_13/xi_promoters.py):
  silenced CGI   silenced genes with a CGI promoter and >= MIN_HP fibers on each haplotype; the N_SILENCED with the
                 largest measured CpG-methylation difference (Xi more methylated)
  escape         the escape gene (not PAR) with the most fibers on the less-covered haplotype
  chr21          the chr21 CGI-promoter gene with the most fibers on the less-covered haplotype
  XIST
The fiber counts come from the seed-919 DEV4NF molecule run; they are data, the same for every model.

    python plots_code/26_nov_20/select_loci.py
Writes plots/26_nov_20/loci/loci_ctcf.bed and loci_tss.bed (same columns and header as the source BEDs, for
molecule_predict.py --save_arrays) and selection.tsv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "26_nov_13"))
from abl_common import molecule_dir
from ctcf_footprints import fiber_state

TARGETS = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/regions/targets"
OUT = "plots/26_nov_20/loci"
HOLES = "plots/26_nov_20/ctcf_noholes/holes.txt"   # ctcf_holes.py
N_BOUND, N_UNBOUND, N_SILENCED, MIN_HP = 3, 1, 3, 15


def read_bed(path):
    header = open(path).readline().lstrip("#").rstrip("\n").split("\t")
    return pd.read_csv(path, sep="\t", comment="#", header=None, names=header)


def write_bed(df, path):
    with open(path, "w") as f:
        f.write("#" + "\t".join(df.columns) + "\n")
        df.to_csv(f, sep="\t", header=False, index=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []

    # CTCF
    d = pd.read_csv(os.path.join(molecule_dir("noFIRE", "ctcf_sites_chr21"), "fibers_GM12878.tsv.gz"), sep="\t")
    d["state"] = fiber_state(d)
    n = d.groupby(["target", "state"]).size().unstack(fill_value=0)
    bed = read_bed(os.path.join(TARGETS, "ctcf_sites_chr21.bed"))
    bed = bed.join(n, on="name")
    bound = bed[(bed.set == "bound") & (bed.GM12878 > 0)].copy()
    bound["rank_key"] = bound[["footprint", "open"]].min(1)
    bound = bound.sort_values("rank_key", ascending=False).head(N_BOUND)
    holes = set(open(HOLES).read().split())   # no bulk ChIP data there, so not a valid unbound control
    unb = bed[(bed.set == "unbound") & ~bed.name.isin(holes)].copy()
    unb["rank_key"] = unb.footprint + unb.open
    unb = unb.sort_values("rank_key", ascending=False).head(N_UNBOUND)
    sel = pd.concat([bound, unb])
    for r in sel.itertuples():
        rows.append({"test": "ctcf", "name": r.name, "group": "bound" if r.set == "bound" else "unbound",
                     "chrom": r.chrom, "start": r.start, "end": r.end, "motif_score": r.score,
                     "n_footprint": r.footprint, "n_open": r.open, "n_nucleosome": r.nucleosome})
    write_bed(sel[read_bed(os.path.join(TARGETS, "ctcf_sites_chr21.bed")).columns], os.path.join(OUT, "loci_ctcf.bed"))

    # X inactivation
    g = pd.read_csv("plots/26_nov_13/molecules/xi_per_gene_noFIRE.tsv", sep="\t")
    g["n_min"] = g[["n_xa", "n_xi"]].min(1)
    ok = g[g.n_min >= MIN_HP]
    picks = [("silenced CGI", ok[(ok["class"] == "subject") & (ok.cgi == 1)].sort_values("cpg_meth_t").head(N_SILENCED)),
             ("escape", ok[ok["class"] == "escape"].sort_values("n_min", ascending=False).head(1)),
             ("chr21", ok[(ok["class"] == "autosomal") & (ok.cgi == 1)].sort_values("n_min", ascending=False).head(1)),
             ("XIST", g[g["class"] == "xist"])]
    tss = read_bed(os.path.join(TARGETS, "tss_chrX_chr21.bed"))
    names = []
    for grp, p in picks:
        for r in p.itertuples():
            b = tss[tss.name == r.target].iloc[0]
            names.append(r.target)
            rows.append({"test": "xi", "name": r.target, "group": grp, "gene": r.gene, "chrom": b.chrom,
                         "start": b.start, "end": b.end, "strand": b.strand, "n_xa": r.n_xa, "n_xi": r.n_xi,
                         "d_cpg_meth": r.cpg_meth_t, "d_msp": r.msp_t})
    write_bed(tss[tss.name.isin(names)], os.path.join(OUT, "loci_tss.bed"))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "selection.tsv"), sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
