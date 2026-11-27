"""Phase-block maps of the sealed X-inactivation cells (reports/TODO.md section 3 #2): which promoters share a HiPhase
block (PS tag). Used to validate block-phased XI on GM12878 (xi_blocks.py) with a realistic block layout, and later to
run XI on the sealed cells themselves.

Reads only the reads' HP / PS tags at each target of tss_chrX_chr21.bed (TSS +- 500 bp); no methylation, no model, so
nothing of the sealed cells' test readouts is looked at.

Per cell and target: the majority PS among HP-tagged reads overlapping the target, its share of those reads, and the
HP1 / HP2 read counts. Targets with no HP-tagged read get no block.

    python plots_code/26_dec_04/xi_block_maps.py
Writes plots/26_dec_04/xi_blocks/block_map_<cell>.tsv and prints per-cell block statistics.
"""
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd
import pysam

D = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/fiber_multi_cell"
BED = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/regions/targets/tss_chrX_chr21.bed"
FA = "/home/azr/projects/def-maxwl/azr/data/misc/hg38.fa"
OUT = "plots/26_dec_04/xi_blocks"
CELLS = {"HG008_3E4": "HG008_3E4*", "GM25455_LCL": "GM25455_LCL*", "HEK293T": "Hek293T_*200U*"}


def main():
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_csv(BED, sep="\t")
    t.columns = [c.lstrip("#") for c in t.columns]
    for cell, pat in CELLS.items():
        paths = sorted(glob.glob(f"{D}/{pat}.cram"), key=lambda p: "100U" in p)
        if not paths:
            print(f"== {cell}: no CRAM for {pat}", flush=True)
            continue
        aln = pysam.AlignmentFile(paths[0], "rc", reference_filename=FA)
        rows = []
        for r in t.itertuples(index=False):
            ps, hp = Counter(), Counter()
            for a in aln.fetch(r.chrom, r.start, r.end):
                if a.has_tag("HP") and a.has_tag("PS"):
                    ps[a.get_tag("PS")] += 1
                    hp[a.get_tag("HP")] += 1
            n = sum(ps.values())
            top, k = ps.most_common(1)[0] if n else (None, 0)
            rows.append({"target": r.name, "chrom": r.chrom, "start": r.start, "end": r.end, "gene": r.gene,
                         "class": r[7], "cgi": r.cgi, "ps": top, "ps_share": k / n if n else np.nan,
                         "n_hp1": hp.get(1, 0), "n_hp2": hp.get(2, 0)})
        df = pd.DataFrame(rows)
        df.to_csv(f"{OUT}/block_map_{cell}.tsv", sep="\t", index=False)
        x = df[(df.chrom == "chrX") & df.ps.notna()]
        per = x.groupby("ps").size()
        sc = x[(x["class"] == "subject") & (x.cgi == 1)].groupby("ps").size()
        print(f"== {cell} ({os.path.basename(paths[0])})", flush=True)
        print(f"   chrX targets with a block: {len(x)} / {int((df.chrom == 'chrX').sum())}; blocks {len(per)}; "
              f"targets per block median {per.median():.0f}, mean {per.mean():.1f}, max {per.max()}", flush=True)
        print(f"   subject CGI promoters per block: median {sc.median():.0f}; blocks with >= 2: {(sc >= 2).sum()} "
              f"holding {sc[sc >= 2].sum()} of {sc.sum()} promoters; majority-PS share median "
              f"{x.ps_share.median():.2f}", flush=True)
        c = df[(df.chrom == "chr21") & df.ps.notna()]
        print(f"   chr21 targets with a block: {len(c)}; blocks {c.ps.nunique()}", flush=True)


if __name__ == "__main__":
    main()
