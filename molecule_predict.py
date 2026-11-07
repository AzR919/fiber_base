"""
Per-molecule predictions at chosen targets: one row per fiber per target, with the fiber's read name and haplotype
tags, its predicted marks over the target, and summaries of its own inputs. The basis of the single-molecule tests
in reports/TODO.md section 2 (haplotypes, X inactivation, imprinting, CTCF footprints, CpG) and of I2 / I19 in
reports/TODO_extended.md.

Usage:
    python molecule_predict.py (--checkpoint results/<run>/Model_epoch_15.pt | --run <training run name>) \
                               --samples GM12878 [K562_200U ...] --targets <targets.bed> \
                               [--flank 100] [--max_fibers 1000] [--batch_fibers 200] \
                               [--save_arrays 20 | --array_targets name1 name2 ...] [--out_dir DIR] [--device cuda]

Targets: BED with chrom, start, end, name (optional: score, strand; extra columns are carried into the output,
named by a '#chrom start end name score strand <extra ...>' header line if present; data_prep/build_targets.py).
Each target gets one window of the model's context length (default 5000 bp, from the checkpoint) centred on it;
the target must fit inside the window with its flanks. Every fiber whose scored span (fiber_utils.get_fiber_data)
overlaps the window is predicted, up to --max_fibers, in chunks of --batch_fibers: the model processes fibers
independently, so a fiber's prediction does not depend on which other fibers are in the chunk.

Output, in DIR (default results/<timestamp>_molecules_<run>_<targets>/):
  fibers_<sample>.tsv.gz  one row per (target, fiber) with at least 1 bp of the fiber's scored span in the target:
    target, chrom, t_start, t_end, t_strand, [extra BED columns], read, hp, ps, f_start, f_end, f_strand,
    scored_bp (scored span inside the target),
    pred_<assay>  mean per-fiber prediction over the scored target positions (same units as the eval's per-fiber
                  outputs: asinh FC),
    for x in t (target), l (left flank), r (right flank), over the part of the fiber's alignment inside x:
      m6a_rate_x (methylated A/T calls / A/T positions), m6a_n_x (A/T positions), cpg_meth_x (methylated CpGs /
      covered CpGs, a CpG counts as methylated if either strand's call is), cpg_n_x, msp_x, nuc_x, fire_x
      (fraction of bp inside an MSP / nucleosome / FIRE). Inputs are read for all 5 channels whatever the model
      uses, so a no-CpG model still reports the molecule's measured methylation (held-out data, T5a).
  arrays_<sample>.npz     with --save_arrays / --array_targets: full per-bp arrays for those targets (per target:
                          pred (n, K, L) float16, inputs (n, 5, L) int8, mask (n, L) bool, reads, hp, ps, window)
  run.json                checkpoint, input flags, assays, arguments
"""

import argparse
import datetime
import glob
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pysam
import pyft
import torch

from fiber_utils import (get_fiber_data, get_locus_onehot, get_m6a, get_cpg, get_msp, get_nuc, get_fire_msp,
                         suppress_stdout_stderr)
from models import BaseModel
from utils import load_metapaths

ALL_FEATURES = [get_m6a, get_cpg, get_msp, get_nuc, get_fire_msp]   # input_flags order
M6A, CPG, MSP, NUC, FIRE = range(5)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="Model checkpoint (.pt)")
    g.add_argument("--run", help="Training run name; uses the newest results/*_train_<run>/ and its last epoch")
    p.add_argument("--samples", nargs="+", required=True, help="Fiber-seq samples (keys of fiber_paths)")
    p.add_argument("--targets", required=True, help="BED of targets (chrom start end name [score strand ...])")
    p.add_argument("--flank", type=int, default=100, help="Flank size (bp) for the l / r input summaries")
    p.add_argument("--context_length", type=int, default=None, help="Window size (default: from the checkpoint)")
    p.add_argument("--max_fibers", type=int, default=1000, help="Max fibers read per window")
    p.add_argument("--batch_fibers", type=int, default=200, help="Fibers per forward pass")
    p.add_argument("--save_arrays", type=int, default=0, help="Save full arrays for the first N targets")
    p.add_argument("--array_targets", nargs="+", default=None, help="Save full arrays for these target names")
    p.add_argument("--metapaths", default="configs/metapaths.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def resolve_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint
    runs = sorted(glob.glob(f"results/*_train_{args.run}"))
    if not runs:
        sys.exit(f"No results/*_train_{args.run} found.")
    ckpts = glob.glob(os.path.join(runs[-1], "Model_epoch_*.pt"))
    if not ckpts:
        sys.exit(f"No Model_epoch_*.pt in {runs[-1]}.")
    return max(ckpts, key=lambda c: int(re.search(r"Model_epoch_(\d+)\.pt$", c).group(1)))


def read_targets(path):
    """BED6 columns get fixed names; extra columns are named by a '#chrom ...' header line if the file has one
    (data_prep/build_targets.py writes it), else col6, col7, ..."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
        first = f.readline().decode().rstrip("\n")
    header = first.lstrip("#").split("\t") if first.startswith("#") else []
    names = ["chrom", "t_start", "t_end", "target", "score", "t_strand"]
    extra = header[6:] if len(header) == df.shape[1] else [f"col{i}" for i in range(6, df.shape[1])]
    df.columns = names[:min(6, df.shape[1])] + extra[:max(0, df.shape[1] - 6)]
    if "target" not in df:
        df["target"] = [f"{c}:{s}-{e}" for c, s, e in zip(df.chrom, df.t_start, df.t_end)]
    if "t_strand" not in df:
        df["t_strand"] = "."
    return df


def cpg_starts(seq):
    """Window-relative start positions of CpG dinucleotides in the reference."""
    b = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    return np.flatnonzero((b[:-1] == ord("C")) & (b[1:] == ord("G")))


def input_summary(x, aln, cg, lo, hi):
    """Summaries of one fiber's inputs over window positions [lo, hi), restricted to its alignment (aln: (L,) bool).
    x: (5, L) channels; cg: CpG start positions in the window."""
    if hi <= lo:
        return [np.nan, 0, np.nan, 0, np.nan, np.nan, np.nan]
    a = aln[lo:hi]
    nbp = int(a.sum())
    m6 = x[M6A, lo:hi]
    m6_n = int((m6 != 0).sum())
    m6_rate = float((m6 == 1).sum() / m6_n) if m6_n else np.nan
    c = cg[(cg >= lo) & (cg < hi - 1)]
    cpg = x[CPG]
    covered = (cpg[c] != 0) | (cpg[c + 1] != 0)
    meth = (cpg[c] == 1) | (cpg[c + 1] == 1)
    cpg_n = int(covered.sum())
    cpg_meth = float((meth & covered).sum() / cpg_n) if cpg_n else np.nan
    frac = (lambda ch: float(x[ch, lo:hi][a].mean()) if nbp else np.nan)
    return [m6_rate, m6_n, cpg_meth, cpg_n, frac(MSP), frac(NUC), frac(FIRE)]


def tag_lookup(aln_file, chrom, start, end):
    """{(qname, reference_start): (HP, PS)} for the reads overlapping a window (pyft fibers do not expose PS)."""
    out = {}
    for r in aln_file.fetch(chrom, start, end):
        out[(r.query_name, r.reference_start)] = (r.get_tag("HP") if r.has_tag("HP") else None,
                                                   r.get_tag("PS") if r.has_tag("PS") else None)
    return out


@torch.no_grad()
def predict_fibers(model, x, mask, dna, active, device, batch_fibers):
    """Per-fiber predictions (n, K, L) float32 for fibers x (n, 5, L) with scored-span mask (n, L) and the window's
    reference one-hot dna (4, L). active: input_flags indices the model uses. Exposed for reuse (e.g. in-silico
    perturbations, I21): pass modified x."""
    n, L = x.shape[0], x.shape[-1]
    ref = torch.from_numpy(dna.T if dna.shape[0] != 4 else dna).float().unsqueeze(0).to(device)   # (1, 4, L)
    outs = []
    for i in range(0, n, batch_fibers):
        xb = torch.from_numpy(x[i:i + batch_fibers][:, active]).permute(1, 2, 0).unsqueeze(0).to(device)  # 1,C,L,N
        mb = torch.from_numpy(mask[i:i + batch_fibers]).T.unsqueeze(0).to(device)                       # 1,L,N
        with torch.amp.autocast(device.split(":")[0], dtype=torch.float16, enabled=device.startswith("cuda")):
            _, fib = model(xb, ref_dna=ref, fiber_mask=mb, fiber_coverage=mb.sum(-1).float())
        outs.append(fib.float()[0].permute(2, 0, 1).cpu().numpy())                                       # N,K,L
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0, L), dtype=np.float32)


def run_sample(sample, model, assays, active, targets, args, m, context_length, array_names):
    bam = pyft.Fiberbam(os.path.realpath(m["fiber_paths"][sample]))
    aln = pysam.AlignmentFile(m["fiber_paths"][sample], reference_filename=m["fasta_path"])
    fasta = pysam.FastaFile(m["fasta_path"])
    extra = [c for c in targets.columns if c not in ("chrom", "t_start", "t_end", "target", "t_strand")]
    rows, arrays = [], {}
    for k, t in enumerate(targets.itertuples(index=False)):
        t = t._asdict()
        chrom, ts, te = t["chrom"], int(t["t_start"]), int(t["t_end"])
        if te - ts + 2 * args.flank > context_length:
            print(f"  skip {t['target']}: target + flanks longer than the window", flush=True)
            continue
        ws = max(0, (ts + te) // 2 - context_length // 2)
        ws = min(ws, fasta.get_reference_length(chrom) - context_length)
        we = ws + context_length
        with suppress_stdout_stderr():
            x, _, n, _, mask, meta = get_fiber_data(bam, chrom, ws, we, fasta, args.max_fibers, context_length,
                                                    ALL_FEATURES, min_overlap=1, return_meta=True)
        if n == 0:
            continue
        x, mask = x[:n], mask[:n]
        lo, hi = ts - ws, te - ws
        in_target = mask[:, lo:hi].sum(1)
        keep = np.flatnonzero(in_target > 0)
        if len(keep) == 0:
            continue
        seq = fasta.fetch(chrom, ws, we)
        dna = get_locus_onehot(fasta, chrom, ws, we)                     # (L, 4)
        pred = predict_fibers(model, x[keep], mask[keep], dna, active, args.device, args.batch_fibers)
        tags = tag_lookup(aln, chrom, ws, we)
        cg = cpg_starts(seq)
        pos = np.arange(context_length) + ws
        for j, i in enumerate(keep):
            md = meta[i]
            hp, ps = tags.get((md["qname"], md["start"]), (md["hp"], None))
            sm = mask[i, lo:hi]
            p = pred[j][:, lo:hi][:, sm].mean(1)
            a = (pos >= md["start"]) & (pos < md["end"])
            row = {"target": t["target"], "chrom": chrom, "t_start": ts, "t_end": te, "t_strand": t["t_strand"],
                   **{c: t[c] for c in extra}, "read": md["qname"], "hp": hp, "ps": ps,
                   "f_start": md["start"], "f_end": md["end"], "f_strand": md["strand"], "scored_bp": int(sm.sum())}
            row.update({f"pred_{a_}": float(v) for a_, v in zip(assays, p)})
            for tag, (l0, h0) in (("t", (lo, hi)), ("l", (max(0, lo - args.flank), lo)),
                                  ("r", (hi, min(context_length, hi + args.flank)))):
                s = input_summary(x[i], a, cg, l0, h0)
                row.update(dict(zip([f"m6a_rate_{tag}", f"m6a_n_{tag}", f"cpg_meth_{tag}", f"cpg_n_{tag}",
                                     f"msp_{tag}", f"nuc_{tag}", f"fire_{tag}"], s)))
            rows.append(row)
        if t["target"] in array_names or k < args.save_arrays:
            arrays[t["target"]] = {"pred": pred.astype(np.float16), "inputs": x[keep].astype(np.int8),
                                   "mask": mask[keep], "reads": np.array([meta[i]["qname"] for i in keep]),
                                   "hp": np.array([str(tags.get((meta[i]["qname"], meta[i]["start"]),
                                                                (meta[i]["hp"], None))[0]) for i in keep]),
                                   "ps": np.array([str(tags.get((meta[i]["qname"], meta[i]["start"]),
                                                                (None, None))[1]) for i in keep]),
                                   "window": np.array([chrom, ws, we], dtype=object)}
        if (k + 1) % 100 == 0:
            print(f"  {sample}: {k + 1}/{len(targets)} targets, {len(rows)} rows", flush=True)
    return pd.DataFrame(rows), arrays


def main():
    args = parse_args()
    m = load_metapaths(args.metapaths)
    ckpt = resolve_checkpoint(args)
    model, cfg = BaseModel.load_model(ckpt, map_location=args.device)
    model.to(args.device).eval()
    assays = model.init_args["output_assays"]
    flags = model.init_args["input_flags"]
    active = [i for i in range(5) if flags[i]]
    context_length = args.context_length or (cfg.get("context_length") if isinstance(cfg, dict) else None) or 5000
    targets = read_targets(args.targets)
    array_names = set(args.array_targets or [])

    run_dir = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
    run_base = re.sub(r'^\d{2}-\d{2}-\d{2}_T\d{2}-\d{2}-\d{2}_?(train_)?', '', run_dir)
    tname = os.path.splitext(os.path.basename(args.targets))[0]
    now = datetime.datetime.now().strftime("%y-%m-%d_T%H-%M-%S")
    out_dir = args.out_dir or os.path.join("results", f"{now}_molecules_{run_base}_{tname}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Checkpoint {ckpt} | inputs {flags} | {len(targets)} targets | window {context_length} | out {out_dir}",
          flush=True)
    json.dump({"checkpoint": ckpt, "input_flags": flags, "assays": assays, "context_length": context_length,
               "args": vars(args)}, open(os.path.join(out_dir, "run.json"), "w"), indent=2)

    for sample in args.samples:
        print(f"[{sample}]", flush=True)
        df, arrays = run_sample(sample, model, assays, active, targets, args, m, context_length, array_names)
        path = os.path.join(out_dir, f"fibers_{sample}.tsv.gz")
        df.to_csv(path, sep="\t", index=False)
        print(f"  {len(df)} fiber rows over {df.target.nunique() if len(df) else 0} targets -> {path}", flush=True)
        if arrays:
            np.savez_compressed(os.path.join(out_dir, f"arrays_{sample}.npz"),
                                **{f"{name}::{k}": v for name, d in arrays.items() for k, v in d.items()})


if __name__ == "__main__":
    main()
