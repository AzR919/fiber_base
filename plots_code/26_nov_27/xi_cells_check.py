"""One-off check (25 Nov, reports/TODO.md section 3): Which cells have an inactive X and haplotype tags? Per cell (200U libraries + GM12878):
- HP / PS tags on reads at chrX promoters; number of phase blocks (PS) on chrX and their spans
- per-fiber CpG methylation at CGI-like promoters (TSS +-500, >= 40 CpGs) on chrX vs chr21: with an Xi, about half
  of the chrX fibers are methylated (silenced allele), chr21 near 0
- XIST promoter: fraction of fibers unmethylated (the Xi's XIST is unmethylated), overall and per HP
Run as a CPU job (19 CRAMs, ~10 min): python -u plots_code/26_nov_27/xi_cells_check.py > plots/26_nov_27/xi_cells/log_xi_cells.txt
"""
import glob
import sys
from collections import defaultdict

import numpy as np
import pysam
import pyft

D = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/fiber_multi_cell"
BED = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/regions/targets/tss_chrX_chr21.bed"
FA = pysam.FastaFile("/home/azr/projects/def-maxwl/azr/data/misc/hg38.fa")
CELLS = {"GM12878": "GM12878-fire-v0.1-filtered", "K562": "K562_*200U*", "HepG2": "HepG2_*200U*", "H1": "H1_*200U*",
         "H9": "H9_*200U*", "HCT116": "HCT116_*200U*", "MCF-7": "MCF_7_*200U*",
         # other Fiber-seq lines in the folder (no ENCODE tracks used so far): candidates for a sealed XI cell
         "GM25455_LCL": "GM25455_LCL*", "GM28570_LCL": "GM28570_LCL*", "GM25456_Fib": "GM25456_Fibroblast*",
         "GM28572_Fib": "GM28572_Fibroblast*", "GM27730_iPSC": "GM27730_iPSC*", "HEK293T": "Hek293T_*200U*",
         "WTC11": "WTC_11_*200U*", "A549": "A549_*200U*", "Caco2": "Caco_2_*", "Jurkat": "Jurkat_*200U*",
         "THP1": "THP_1_*200U*", "HG008_3E4": "HG008_3E4*"}
FLANK, MIN_CPG_SITE, MIN_CPG_FIBER, Q = 500, 40, 5, 200

tss = []
for line in open(BED):
    f = line.split("\t")
    if not f[0].startswith("chr"):
        continue
    s, e = int(f[1]), int(f[2])
    mid = (s + e) // 2
    tss.append((f[0], mid - FLANK, mid + FLANK, f[3]))
promoters = []
for chrom, s, e, name in tss:
    seq = FA.fetch(chrom, s, e).upper()
    cpg = np.array([i for i in range(len(seq) - 1) if seq[i:i + 2] == "CG"]) + s
    if len(cpg) >= MIN_CPG_SITE or "XIST" in name:
        promoters.append((chrom, s, e, name, cpg))
print(f"CGI-like promoters: chrX {sum(p[0] == 'chrX' for p in promoters)}, "
      f"chr21 {sum(p[0] == 'chr21' for p in promoters)}", flush=True)


def fiber_meth(fb, cpg):
    """Fraction of a fiber's covered CpGs (by reference C or G position) with a 5mC call."""
    cov = cpg[(cpg >= fb.start) & (cpg < fb.end - 1)]
    if len(cov) < MIN_CPG_FIBER:
        return None
    st = fb.cpg.reference_starts
    ml = np.asarray(fb.cpg.ml)
    pos = np.array([-1 if p is None else p for p in st], dtype=np.int64)
    calls = set(pos[ml >= Q].tolist())
    m = sum((c in calls) or (c + 1 in calls) for c in cov)
    return m / len(cov)


for cell, pat in CELLS.items():
    paths = sorted(glob.glob(f"{D}/{pat}.cram"), key=lambda p: "100U" in p)   # prefer 200U
    if not paths:
        print(f"\n== {cell}: no CRAM for {pat}")
        continue
    path = paths[0]
    bam, aln = pyft.Fiberbam(path), pysam.AlignmentFile(path, "rc", reference_filename=FA.filename.decode())
    tags = {}
    fr = {"chrX": [], "chr21": []}
    per_gene_bimodal = {"chrX": [], "chr21": []}
    xist = []
    for chrom, s, e, name, cpg in promoters:
        for r in aln.fetch(chrom, s, e):
            tags[(r.query_name, r.reference_start)] = (r.get_tag("HP") if r.has_tag("HP") else None,
                                                      r.get_tag("PS") if r.has_tag("PS") else None, r.reference_end)
        vals = []
        for fb in bam.fetch(chrom, s, e):
            v = fiber_meth(fb, cpg)
            if v is None:
                continue
            hp = tags.get((fb.qname, fb.start), (None, None, None))[0]
            if "XIST" in name:
                xist.append((v, hp))
            else:
                vals.append(v)
        if "XIST" not in name and len(vals) >= 8:
            v = np.array(vals)
            fr[chrom].extend(vals)
            per_gene_bimodal[chrom].append(0.2 <= (v > 0.5).mean() <= 0.8)
    hp_frac = np.mean([t[0] is not None for t in tags.values()]) if tags else 0
    ps_x = defaultdict(list)
    for (q, st), (hp, ps, end) in tags.items():
        if ps is not None and hp is not None:
            ps_x[ps].append((st, end))
    xblocks = [ps for ps in ps_x]
    print(f"\n== {cell}  ({path.split('/')[-1][:50]})", flush=True)
    print(f"   reads with HP tag at the promoters: {hp_frac:.2f}; distinct PS blocks over the promoters: {len(xblocks)}")
    for chrom in ["chrX", "chr21"]:
        v = np.array(fr[chrom])
        b = per_gene_bimodal[chrom]
        print(f"   {chrom}: {len(b)} promoters, {len(v)} fibers; fibers methylated (>0.5) {(v > 0.5).mean():.2f}; "
              f"promoters with 20-80% methylated fibers {np.mean(b):.2f}" if len(v) else f"   {chrom}: no fibers")
    if xist:
        xv = np.array([x[0] for x in xist])
        line = f"   XIST promoter: {len(xv)} fibers, unmethylated (<0.3) {(xv < 0.3).mean():.2f}"
        for h in sorted({x[1] for x in xist}, key=str):
            hv = np.array([x[0] for x in xist if x[1] == h])
            line += f" | HP {h}: n {len(hv)}, unmeth {(hv < 0.3).mean():.2f}"
        print(line, flush=True)
