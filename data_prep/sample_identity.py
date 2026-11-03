"""
Fiber-seq sample identity check: genotype each sample's CRAM at known cell-line driver mutations and one known
homozygous deletion, so a mislabelled or swapped sample shows up. Like fiber_qc.py, a separate QC step, not part of
setup_data.py. Takes a few minutes on a login node.

Run from the repo root:
    python data_prep/sample_identity.py [--samples Panc1_200U ...] [--out DIR]

Markers (hg38; the reference base at each site is checked against the FASTA first, so a wrong coordinate is
reported, not silently scored). Expected carriers are from the cell lines' published mutation profiles (COSMIC /
CCLE):
  KRAS G12D   chr12:25245350 C>T   Panc1 (heterozygous)
  KRAS G13D   chr12:25245347 C>T   HCT116 (heterozygous)
  TP53 R273H  chr17:7673802  C>T   Panc1 (homozygous)
  PIK3CA H1047R chr3:179234297 A>G HCT116 (heterozygous)
  PIK3CA E545K  chr3:179218303 G>A MCF-7 (heterozygous)
  NRAS Q61L   chr1:114713908 T>A   HepG2 (heterozygous)
  CDKN2A homozygous deletion: reads over CDKN2A (chr9:21,967,000-21,975,000) vs a control window
             (chr9:20,000,000-20,008,000); ~0 in Panc1 (and MCF-7, K562), normal elsewhere.
  Sex: reads on the male-specific part of chrY (chrY:2,800,000-10,000,000, outside the pseudoautosomal regions)
             per 1,000 reads on the same length of chr9 (chr9:100,000,000-107,200,000); ~0 in female lines (GM12878,
             K562, MCF-7, H9), clearly > 0 in male lines (HepG2, HCT116, Panc1, H1) unless the line lost its Y.
Writes <out>/sample_identity.tsv (default out: DATA_FIBER/qc/) and prints the table.
"""
import argparse
import os
import sys

import pandas as pd
import pysam

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

SNVS = [  # name, chrom, 1-based position, ref, alt, expected carriers
    ("KRAS_G12D", "chr12", 25245350, "C", "T", "Panc1"),
    ("KRAS_G13D", "chr12", 25245347, "C", "T", "HCT116"),
    ("TP53_R273H", "chr17", 7673802, "C", "T", "Panc1"),
    ("PIK3CA_H1047R", "chr3", 179234297, "A", "G", "HCT116"),
    ("PIK3CA_E545K", "chr3", 179218303, "G", "A", "MCF-7"),
    ("NRAS_Q61L", "chr1", 114713908, "T", "A", "HepG2"),
]
DELETION = ("CDKN2A", "chr9", 21_967_000, 21_975_000)
CONTROL = ("chr9", 20_000_000, 20_008_000)
CHRY = ("chrY", 2_800_000, 10_000_000)
CHRY_CONTROL = ("chr9", 100_000_000, 107_200_000)


def alt_fraction(cram, chrom, pos, alt):
    """(alt reads, total reads) at a 1-based position. No base-quality filter: the Fiber-seq CRAMs store no
    qualities (pysam reads them as missing), so any threshold would drop every base."""
    counts = cram.count_coverage(chrom, pos - 1, pos, quality_threshold=0)   # A, C, G, T arrays
    by_base = {b: int(c[0]) for b, c in zip("ACGT", counts)}
    return by_base[alt], sum(by_base.values())


def reads_in(cram, chrom, start, end):
    return sum(1 for r in cram.fetch(chrom, start, end) if not (r.is_secondary or r.is_supplementary))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", nargs="+", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.chdir(REPO_ROOT)
    from utils import load_metapaths
    m = load_metapaths("configs/metapaths.yaml")
    samples = [s for s in m["fiber_paths"] if s in (m.get("cell_types") or {})]
    if a.samples:
        samples = [s for s in samples if s in a.samples]
    out = a.out or os.path.join(os.path.dirname(m["regions_root"].rstrip("/")), "qc")

    fasta = pysam.FastaFile(m["fasta_path"])
    for name, chrom, pos, ref, alt, _ in SNVS:
        got = fasta.fetch(chrom, pos - 1, pos).upper()
        if got != ref:
            print(f"[WARN] {name}: reference base at {chrom}:{pos} is {got}, expected {ref}; marker will be misread")

    rows = []
    for s in samples:
        line = os.path.basename(m["cell_types"][s].rstrip("/"))
        cram = pysam.AlignmentFile(m["fiber_paths"][s], "rc", reference_filename=m["fasta_path"])
        row = {"sample": s, "line": line}
        for name, chrom, pos, ref, alt, _ in SNVS:
            k, n = alt_fraction(cram, chrom, pos, alt)
            row[name] = f"{k}/{n}" + (f" ({k / n:.0%})" if n else "")
        d, c = reads_in(cram, *DELETION[1:]), reads_in(cram, *CONTROL)
        row["CDKN2A_reads/control"] = f"{d}/{c}"
        y, yc = reads_in(cram, *CHRY), reads_in(cram, *CHRY_CONTROL)
        row["chrY_per_1k_chr9"] = round(1000 * y / max(yc, 1), 1)
        rows.append(row)
        cram.close()
        print(f"  {s} done", flush=True)

    df = pd.DataFrame(rows).set_index("sample")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "sample_identity.tsv"), sep="\t")
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print("\nAlt reads / total reads at each marker (expected carriers: "
          + ", ".join(f"{n}: {c}" for n, *_, c in SNVS) + "; CDKN2A deleted in Panc1, MCF-7, K562)\n")
    print(df)
    print(f"\nWrote {out}/sample_identity.tsv")


if __name__ == "__main__":
    main()
