"""
Per-base annotation tracks for baseline ladder rung 1 (hand-crafted rules, reports/TODO_extended.md T11). Only rung 1
uses them; the network and rungs 2 / 4 see the fiber and the reference DNA only.

One uint8 array per chromosome, a bit per annotation:
  1   CGI          inside a UCSC CpG island (cpgIslandExt)
  2   TSS_2KB      within 2 kb of a protein-coding TSS (GENCODE basic)
  4   GENE_BODY    protein-coding gene body from TSS + 2 kb to the TES (strand-aware)
  8   REPEAT       soft-masked (lowercase) in the reference: every RepeatMasker / TRF class. Stand-in until
                   RepeatMasker classes are decided on (user, 25 Nov: skip the rmsk download for now); broader than the
                   LTR / satellite repeats the H3K9me3 rule is about
  16  ZNF          gene body of a ZNF* protein-coding gene (KRAB-ZNF clusters carry H3K9me3)
  32  CTCF_MOTIF   inside a JASPAR MA0139.2 hit at p <= 1e-5 on either strand (same motif, pseudocount and threshold as
                   build_targets.py / the CTCF footprint test)

Reads the annotation paths from data_prep/targets_spec.yaml. Writes <out>/<chrom>.npy (memory-mapped by the data
loader, data_utils.SingleCellFiberDataset with dna_type 'ref_annot') and <out>/manifest.json.

    python data_prep/build_annotation_tracks.py [--chroms chr20 chr21 chr22 chrX] [--out DIR]
Default DIR: <directory of the reference FASTA>/annotation_tracks.
"""
import argparse
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pysam
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_targets import read_jaspar, log_odds, score_tail, score_threshold, scan, read_cgi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BITS = {"CGI": 1, "TSS_2KB": 2, "GENE_BODY": 4, "REPEAT": 8, "ZNF": 16, "CTCF_MOTIF": 32}
TSS_REACH, BODY_OFFSET = 2000, 2000


def read_gene_bodies(gtf, chroms):
    rows = []
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] != "gene" or p[0] not in chroms:
                continue
            if 'gene_type "protein_coding"' not in p[8]:
                continue
            name = re.search(r'gene_name "([^"]+)"', p[8]).group(1)
            rows.append({"chrom": p[0], "start": int(p[3]) - 1, "end": int(p[4]), "strand": p[6], "gene": name})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroms", nargs="+", default=["chr20", "chr21", "chr22", "chrX"])
    ap.add_argument("--spec", default=os.path.join(HERE, "targets_spec.yaml"))
    ap.add_argument("--fasta", default=None, help="reference FASTA (default: configs/reference.yaml fasta_path)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec))
    ann = spec["annotations"]
    fasta_path = args.fasta or yaml.safe_load(open(os.path.join(HERE, "..", "configs", "reference.yaml")))["fasta_path"]
    out = args.out or os.path.join(os.path.dirname(fasta_path), "annotation_tracks")
    os.makedirs(out, exist_ok=True)
    fasta = pysam.FastaFile(fasta_path)

    c = spec["ctcf"]
    pwm = log_odds(read_jaspar(os.path.join(ann, spec["ctcf_motif"])), c["pseudocount"])
    tail, offset = score_tail(pwm)
    thr, p_at = score_threshold(tail, offset, c["motif_pvalue"])
    genes = read_gene_bodies(os.path.join(ann, spec["gtf"]), args.chroms)
    cgi = read_cgi(os.path.join(ann, spec["cpg_islands"]), args.chroms)
    print(f"CTCF motif threshold {thr:.2f} bits (p = {p_at:.2e}); {len(genes)} protein-coding genes; {len(cgi)} CGIs",
          flush=True)

    manifest = {"bits": BITS, "fasta": fasta_path, "spec": args.spec, "ctcf_threshold_bits": thr,
                "tss_reach": TSS_REACH, "body_offset": BODY_OFFSET, "chroms": {}}
    for chrom in args.chroms:
        seq = fasta.fetch(chrom)
        n = len(seq)
        a = np.zeros(n, dtype=np.uint8)
        b = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        a[(b >= ord("a")) & (b <= ord("z"))] |= BITS["REPEAT"]
        for s, e in cgi[cgi.chrom == chrom][["start", "end"]].itertuples(index=False):
            a[s:e] |= BITS["CGI"]
        g = genes[genes.chrom == chrom]
        for s, e, strand, name in g[["start", "end", "strand", "gene"]].itertuples(index=False):
            tss = s if strand == "+" else e - 1
            a[max(0, tss - TSS_REACH):tss + TSS_REACH + 1] |= BITS["TSS_2KB"]
            bs, be = (s + BODY_OFFSET, e) if strand == "+" else (s, e - BODY_OFFSET)
            if be > bs:
                a[bs:be] |= BITS["GENE_BODY"]
            if name.startswith("ZNF"):
                a[s:e] |= BITS["ZNF"]
        scores, _ = scan(seq, pwm)
        hits = np.flatnonzero(scores >= thr)
        L = len(pwm)
        for h in hits:
            a[h:h + L] |= BITS["CTCF_MOTIF"]
        np.save(os.path.join(out, f"{chrom}.npy"), a)
        frac = {k: float((a & v > 0).mean()) for k, v in BITS.items()}
        manifest["chroms"][chrom] = {"length": n, "ctcf_hits": int(len(hits)), "fraction": frac}
        print(f"{chrom}: {n} bp, {len(hits)} CTCF motif hits, " +
              ", ".join(f"{k} {v:.3f}" for k, v in frac.items()), flush=True)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
