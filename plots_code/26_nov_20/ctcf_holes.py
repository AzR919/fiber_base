"""CTCF motif sites (ctcf_sites_chr21.bed) without bulk short-read data: the ENCODE CTCF or ATAC fold change is
exactly 0 over the whole motif +- 1 kb in a dev line (GM12878, K562, HepG2, H1; H1 has no ATAC). There, "no CTCF
peak in any dev line" means "not measured", so such sites cannot serve as unbound controls (found 18 Nov,
reports/Nov_20_figs.md).

Why both tracks: CTCF fold change is sparse (exact zeros cover a median 28% of bp around bound sites), so only a
fully zero 2 kb is a hole. ATAC is dense (no bound site has a fully zero 2 kb; the most zeros at any bound site is
66%), so a fully zero ATAC window is a region where reads do not map even when the CTCF track has a few non-zero
bins (e.g. ctcf_unbound_52, 105 accessible fibers but no ATAC reads).

    python plots_code/26_nov_20/ctcf_holes.py
Writes plots/26_nov_20/ctcf_noholes/ctcf_sites_bulk_holes.tsv (per site and line: hole_<assay>_<line>, and
hole_ctcf, hole_atac, hole_any) and holes.txt (names of the hole_any sites, for ctcf_footprints.py --exclude).
"""
import os
import sys

import numpy as np
import pandas as pd
import pyBigWig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repo root (utils)
from utils import load_cell_assays, load_metapaths

BED = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/regions/targets/ctcf_sites_chr21.bed"
OUT = "plots/26_nov_20/ctcf_noholes"
CELLS = {"GM12878": "GM12878", "K562": "K562_200U", "HepG2": "HepG2_200U", "H1": "H1_200U"}
ASSAYS = ["ctcf", "atac"]
FLANK = 1000


def main():
    os.makedirs(OUT, exist_ok=True)
    bed = pd.read_csv(BED, sep="\t")
    bed.columns = [c.lstrip("#") for c in bed.columns]
    m = load_metapaths("configs/metapaths.yaml")
    for assay in ASSAYS:
        cols = []
        for c, key in CELLS.items():
            paths = load_cell_assays(m["cell_types"][key])
            if assay not in paths:
                continue
            bw = pyBigWig.open(paths[assay])
            col = f"hole_{assay}_{c}"
            bed[col] = [float(np.nan_to_num(np.asarray(bw.values(r.chrom, r.start - FLANK, r.end + FLANK,
                                                                   numpy=True))).max()) == 0
                        for r in bed.itertuples()]
            cols.append(col)
        bed[f"hole_{assay}"] = bed[cols].any(axis=1)
    bed["hole_any"] = bed[[f"hole_{a}" for a in ASSAYS]].any(axis=1)
    hc = [c for c in bed.columns if c.startswith("hole_")]
    bed[["name", "chrom", "start", "end", "set"] + hc].to_csv(os.path.join(OUT, "ctcf_sites_bulk_holes.tsv"),
                                                              sep="\t", index=False)
    with open(os.path.join(OUT, "holes.txt"), "w") as f:
        f.write("\n".join(bed[bed.hole_any].name) + "\n")
    print(bed.groupby("set")[hc].sum().T.to_string(), flush=True)
    print(bed.groupby("set").size().to_string(), flush=True)
    print("hole_any sites by Mb:", np.floor(bed[bed.hole_any].start / 1e6).astype(int).value_counts().sort_index()
          .to_dict(), flush=True)


if __name__ == "__main__":
    main()
