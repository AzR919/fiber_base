"""Shared setup of the sealed test analysis (reports/sealed_test_protocol.md, frozen 27 Nov).

Two setups with the same code (protocol section 7, step 4: the sealed run needs no code changes):
  sealed  HCT116 + MCF-7 on chr19, the models of protocol section 3; outputs in plots/sealed/
  dryrun  the dev pair A (GM12878 + HepG2) on chr21 with the dev models in the sealed roles (R1a-S = R1a A,
          Pair-S = Pair A, MLP-S = MLP A, Rules-S = Rules A, the other-sample mixture model = R1a B); XI on GM12878
          (trio-phased: one block per chromosome). Outputs in plots/sealed_dryrun/. Only checks that the code runs
          and reads the right files; its numbers are dev results already reported elsewhere.

Model labels are the protocol's. Each model is (abl_common key, training seeds).
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("26_nov_13", "26_nov_20", "26_nov_27", "26_dec_04"):
    sys.path.insert(0, os.path.join(HERE, "..", _sub))
from abl_common import run_name, tag as out_tag  # noqa: E402,F401

S3 = [919, 2, 3]
TARGETS_DIR = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/regions/targets"
ALL_ASSAYS = ["atac", "h3k4me3", "h3k27ac", "ctcf", "h3k4me1", "h3k36me3", "h3k27me3", "h3k9me3"]
ORDER = ["Rules-S", "MLP-S", "R1a-S", "Pair-S", "DEV4NF", "R1a-A", "R1a-B"]
COL = {"Rules-S": "#9e9e9e", "MLP-S": "#4a90d9", "R1a-S": "#eb6834", "Pair-S": "#1baf7a", "DEV4NF": "#606060",
       "R1a-A": "#c9a227", "R1a-B": "#8c3fb8"}
MODEL_UNDER_TEST = "R1a-S"

SETUPS = {
    "sealed": {
        "out": "plots/sealed",
        "chrom": "chr19",
        # fiber sample -> name the molecule outputs are linked under; its part before '_' is the cell's column in
        # the target BEDs (the dev scripts read the cell as sample.split('_')[0], and MCF_7 would give 'MCF')
        "cells": {"HCT116_200U": "HCT116_200U", "MCF_7_200U": "MCF-7_200U"},
        "assays": ALL_ASSAYS,
        "pair_eval": "eval33_sealed_pair_HCT116_MCF_7_chr19",
        "floor_eval": "eval33_sealed_pair_HCT116_MCF_7_chr19",     # R1a-S's eval33 run with --mixture_reference
        "mix_eval": "eval34_sealed_mix_bal_HCT116_MCF_7_chr19",
        "mini_eval": "eval37_sealed_pair_HCT116_MCF_7_chr19_mini",
        "shuffle_eval": "eval38_sealed_shuffle_pair_HCT116_MCF_7_chr19_mini",
        "split_evals": {"HCT116_200U": "eval35_sealed_split_HCT116_chr19_mini",
                        "MCF_7_200U": "eval36_sealed_split_MCF_7_chr19_mini"},
        "ceiling_config": "configs/evals/eval33_sealed_pair_HCT116_MCF_7_chr19.yaml",
        "ceiling_tsv": None,
        "ctcf_targets": "ctcf_sites_chr19",
        "holes": None,                                              # built by sealed_molecules.py --step holes
        "holes_cells": {"HCT116": "HCT116_200U", "MCF-7": "MCF_7_200U"},
        "poised_targets": "poised_tiles_chr19",
        "poised_fractions": f"{TARGETS_DIR}/poised_fractions_chr19.tsv",
        "xi_targets": "tss_chrX_chr21",
        "xi_samples": ["HG008_3E4", "GM25455_LCL", "HEK293T_200U"],
        "models": {"R1a-S": ("sR1a", S3), "DEV4NF": ("noFIRE", [919, 2]), "Pair-S": ("sPair", S3),
                   "MLP-S": ("sMLP", S3), "Rules-S": ("sRules", [919]), "R1a-A": ("mixR1a", S3),
                   "R1a-B": ("mixR1aB", S3)},
    },
    "dryrun": {
        "out": "plots/sealed_dryrun",
        "chrom": "chr21",
        "cells": {"GM12878": "GM12878", "HepG2_200U": "HepG2_200U"},
        "assays": ALL_ASSAYS,
        "pair_eval": "eval24_pair_GM12878_HepG2_200U",
        "floor_eval": "eval24_pair_GM12878_HepG2_200U_mini",       # the only MIXREF run on dev (job _54)
        "mix_eval": "eval31_mix_bal_GM12878_HepG2_200U_mini",
        "mini_eval": "eval24_pair_GM12878_HepG2_200U_mini",
        "shuffle_eval": "eval41_shuffle_pair_GM12878_HepG2_200U_chr21_mini",
        "split_evals": {"HepG2_200U": "eval40_split_HepG2_200U_chr21_mini"},
        "ceiling_config": None,
        "ceiling_tsv": "plots/26_nov_13/ceiling/ceiling_cells.tsv",
        "ctcf_targets": "ctcf_sites_chr21",
        "holes": "plots/26_nov_20/ctcf_noholes/holes.txt",
        "holes_cells": None,
        "poised_targets": "poised_tiles_chr21",
        "poised_fractions": f"{TARGETS_DIR}/poised_fractions_chr21.tsv",
        "xi_targets": "tss_chrX_chr21",
        "xi_samples": ["GM12878"],
        "models": {"R1a-S": ("mixR1a", S3), "DEV4NF": ("noFIRE", [919, 2]), "Pair-S": ("pairNF", S3),
                   "MLP-S": ("mlpA", S3), "Rules-S": ("rulesA", [919]), "R1a-B": ("mixR1aB", S3)},
    },
}


def get_setup(name):
    s = dict(SETUPS[name])
    s["name"] = name
    if s["holes"] is None:
        s["holes"] = os.path.join(s["out"], "ctcf", "holes.txt")
    return s


def runs(S, labels=None):
    """(label, key, seed) for every model run of a setup, in ORDER."""
    for label in ORDER:
        if label in S["models"] and (labels is None or label in labels):
            key, seeds = S["models"][label]
            for s in seeds:
                yield label, key, s


def cell_name(link):
    return link.split("_")[0]


# ---------------------------------------------------------------- comparisons (protocol section 4, "clear")
def clear_vs(a, b):
    """a - b over seeds: delta, SE = sqrt(var_a / n_a + var_b / n_b), clear when |delta| > 2 SE and every seed
    pairing has the sign of delta. One-seed sides (Rules-S): see clear_vs_const."""
    a, b = np.asarray(pd.Series(a).dropna(), float), np.asarray(pd.Series(b).dropna(), float)
    if len(a) == 0 or len(b) == 0:
        return {"delta": np.nan, "se": np.nan, "clear": False, "sign": 0, "n_a": len(a), "n_b": len(b)}
    if len(b) == 1:
        return clear_vs_const(a, b[0])
    if len(a) == 1:
        r = clear_vs_const(b, a[0])
        return {**r, "delta": -r["delta"], "sign": -r["sign"], "n_a": 1, "n_b": len(b)}
    d = a.mean() - b.mean()
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    same = bool(np.all(np.sign(np.subtract.outer(a, b)) == np.sign(d)))
    clear = bool(abs(d) > 2 * se and same and d != 0)
    return {"delta": float(d), "se": se, "clear": clear, "sign": int(np.sign(d)) if clear else 0,
            "n_a": len(a), "n_b": len(b)}


def clear_vs_const(a, c):
    """Seeds a against a fixed value c (Rules-S's one run, a floor, AUC 0.5): seed mean - c > 2 SE of a's seeds
    and every seed on the same side of c."""
    a = np.asarray(pd.Series(a).dropna(), float)
    if len(a) == 0 or c is None or np.isnan(c):
        return {"delta": np.nan, "se": np.nan, "clear": False, "sign": 0, "n_a": len(a), "n_b": 1}
    d = a.mean() - c
    se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan
    same = bool(np.all(np.sign(a - c) == np.sign(d)))
    clear = bool(len(a) > 1 and abs(d) > 2 * se and same and d != 0)
    return {"delta": float(d), "se": se, "clear": clear, "sign": int(np.sign(d)) if clear else 0,
            "n_a": len(a), "n_b": 1}


# ---------------------------------------------------------------- eval outputs
def run_dirs(run):
    """Eval output dirs of a training run, newest first (the run name is followed by nothing or a suffix that is
    not '_seed', so the seed-919 run never picks up a seed run)."""
    return sorted(glob.glob(f"results/*_eval_{run}"), reverse=True)


def eval_entry(run, key, need=None):
    """(entry, summary) from the newest eval_summary.json of a run whose results hold key; need(summary) can
    reject a file (e.g. one scored without the mixture reference)."""
    for d in run_dirs(run):
        f = os.path.join(d, "eval_summary.json")
        if not os.path.exists(f):
            continue
        js = json.load(open(f))
        if key in js["results"] and (need is None or need(js)):
            return js["results"][key], js
    return None, None


def paired_npz(run, cfg):
    """{assay: {auc (oriented, mean over differential bins), dev (mean |AUC - 0.5|), sign, n}} from the newest
    <cfg>_paired.npz of a run (as mix_two_pairs.npz_readouts)."""
    for d in run_dirs(run):
        path = os.path.join(d, f"{cfg}_paired.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        assays = [str(a) for a in z["assays"]]
        auc, td, dp, nb = z["auc"], z["tile_d"], z["tile_dp_bin"], z["tile_n_bin"]
        out = {}
        for k, a in enumerate(assays):
            has = (td[:, k] != 0) & (nb[:, k] > 0)
            ak = auc[auc[:, 1] == k] if len(auc) else auc
            out[a] = {"n": int(has.sum()), "n_auc": int(len(ak)),
                      "sign": float(np.mean(np.sign(dp[has, k]) == np.sign(td[has, k]))) if has.any() else np.nan,
                      "auc": float(ak[:, 3].mean()) if len(ak) else np.nan,
                      "dev": float(np.abs(ak[:, 2] - 0.5).mean()) if len(ak) else np.nan}
        return out, path
    return None, None


def paired_json(run, cfg):
    """differential r per assay from the newest eval_summary.json with <cfg>/paired."""
    e, _ = eval_entry(run, f"{cfg}/paired")
    if e is None:
        return None
    return {a: v.get("diff_pearson") for a, v in e["strata"]["all"]["per_assay"].items()}


def mean_of(d, assays):
    v = [d.get(a) for a in assays] if d else []
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else np.nan


# ---------------------------------------------------------------- molecule outputs
def molecule_dirs(run, targets):
    return sorted(glob.glob(f"results/*_molecules_{run}_{targets}"), reverse=True)


def is_perturb(d):
    try:
        return bool(json.load(open(os.path.join(d, "run.json")))["args"].get("perturb"))
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def link_molecules(S, key, seed, targets, samples, perturb=False):
    """ignore/sealed_molecules/<setup>/<targets>/<key tag>/fibers_<link name>.tsv.gz, linked from the newest
    molecule_predict output of the run that has the sample (with --perturb if asked). Returns the dir, or None
    with a message if a sample is missing."""
    d = os.path.join("ignore", "sealed_molecules", S["name"], targets, out_tag(key, seed))
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "fibers_*.tsv.gz")):
        os.remove(f)
    dirs = [x for x in molecule_dirs(run_name(key, seed), targets) if not perturb or is_perturb(x)]
    for s in samples:
        src = next((os.path.join(x, f"fibers_{s}.tsv.gz") for x in dirs
                    if os.path.exists(os.path.join(x, f"fibers_{s}.tsv.gz"))), None)
        if src is None:
            print(f"!! no {targets}{' --perturb' if perturb else ''} fibers for {s} from {key} seed {seed}",
                  flush=True)
            return None
        os.symlink(os.path.abspath(src), os.path.join(d, f"fibers_{S['cells'].get(s, s)}.tsv.gz"))
    return d


def seed_table(df, keys, value="value"):
    g = df.groupby(keys)[value]
    t = g.agg(["mean", "std", "size"]).reset_index()
    t["seeds"] = g.apply(lambda v: " ".join(f"{x:.3f}" for x in v)).values
    return t


def show(df, **kw):
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_rows", 500,
                           "display.max_colwidth", 40):
        print(df.to_string(**kw), flush=True)


def endpoint_row(pid, claim, what, passed, detail):
    return {"endpoint": pid, "claim": claim, "test": what, "pass": bool(passed), "detail": detail}


def fmt_cmp(r):
    se = "" if r["se"] is None or np.isnan(r["se"]) else f" (SE {r['se']:.3f})"
    return f"delta {r['delta']:+.3f}{se}{', clear' if r['clear'] else ', not clear'}"


__all__ = ["S3", "ALL_ASSAYS", "ORDER", "COL", "MODEL_UNDER_TEST", "SETUPS", "get_setup", "runs", "cell_name",
           "clear_vs", "clear_vs_const", "run_dirs", "eval_entry", "paired_npz", "paired_json", "mean_of",
           "link_molecules", "molecule_dirs", "is_perturb", "seed_table", "show", "endpoint_row", "fmt_cmp",
           "run_name", "out_tag", "itertools"]
