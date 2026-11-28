"""Sealed test, eval readouts (reports/sealed_test_protocol.md): primary endpoints P1-P5 and the secondary bulk / T3
readouts, the random-split control and the shuffle control. Reads eval.py outputs only; nothing is recomputed.

Per model and seed (protocol section 3; setups in common.py):
  own r     Pearson r of the prediction against each cell's own ENCODE track (asinh FC), pair eval (sealed:
            eval33, all chr19 tiles), per assay and the mean over assays; fraction of the replicate ceiling
            (replicate_ceiling.py on the same eval config); MSE
  floor     R1a-S only: the baseline of its pair eval run with --mixture_reference, i.e. the mixture track's r
            with each cell's own track ("bulk on every molecule")
  mix r     r against the balanced 0.5 mixture track (sealed: eval34, all chr19 tiles; key '<cfg>/mixed')
  T3        at the pair's differential bins (loci pairs/<A>__<B>, bulk only): fiber-origin AUC (oriented, mean over
            bins), sign accuracy, differential r
  split     R1a-S only: each cell's library split in two halves by read name (eval35 / eval36), the same AUC at the
            same bins; the halves share one true track, so the AUC should be 0.5
  shuffle   R1a-S only: eval38 (fibers from a deranged partner tile) against eval37 (the same mini tiles,
            unshuffled): AUC, differential r, own r
All model-level numbers are means over assays (assays a cell lacks are left out), as on dev.

Primary endpoints ("clear" = |delta| > 2 SE over seeds and every seed pairing with the same sign; against a fixed
value or the one-seed Rules-S: seed mean - value > 2 SE of R1a-S's seeds and every seed on the same side):
  P1  R1a-S own r clearly above the floor, in both cells
  P2  R1a-S AUC clearly above 0.5; random split: seed-mean AUC within 0.5 +- 0.02 for both cells
  P3  R1a-S clearly above DEV4NF on AUC and on differential r
  P4  R1a-S not clearly below Pair-S on own r (each cell) and on mix r
  P5  R1a-S clearly above MLP-S and above Rules-S on mix r

    python plots_code/sealed/sealed_evals.py [--setup sealed|dryrun]
Writes <out>/evals/: evals_long.tsv, evals_table.tsv (mean, sd, seeds per model x test x readout x assay),
evals_deltas.tsv (every model vs R1a-S), endpoints_evals.tsv, evals_main.png, evals_assays.png; prints everything.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ORDER, COL, MODEL_UNDER_TEST as MUT, get_setup, runs, clear_vs, clear_vs_const,  # noqa: E402
                    eval_entry, paired_npz, paired_json, mean_of, seed_table, show, endpoint_row, fmt_cmp,
                    run_name)

SPLIT_TOL = 0.02


def ceilings(S):
    """{(cell, assay): ceiling r}."""
    if S["ceiling_tsv"]:
        if not os.path.exists(S["ceiling_tsv"]):
            print(f"!! no ceiling table {S['ceiling_tsv']}", flush=True)
            return {}
        t = pd.read_csv(S["ceiling_tsv"], sep="\t")
        return {(c, a): v for c, a, v in zip(t.cell, t.assay, t.ceiling_r) if pd.notna(v)}
    out = {}
    for f in sorted(glob.glob("results/*_replicates_*/replicate_summary.json"), key=os.path.getmtime):
        for c, v in json.load(open(f))["results"].items():
            if v["eval_config"] == S["ceiling_config"]:
                out.update({(c, a): x["ceiling_r"] for a, x in v["per_assay"].items()
                            if x is not None and x.get("ceiling_r") is not None})
    if not out:
        print(f"!! no replicate ceiling for {S['ceiling_config']}", flush=True)
    return out


def collect(S):
    ceil = ceilings(S)
    cells, A = list(S["cells"]), S["assays"]
    rows = []

    def add(label, seed, test, readout, per_assay):
        per_assay = {a: v for a, v in (per_assay or {}).items() if a in A and v is not None and not np.isnan(v)}
        for a, v in per_assay.items():
            rows.append({"model": label, "seed": seed, "test": test, "readout": readout, "assay": a, "value": v})
        if per_assay:
            rows.append({"model": label, "seed": seed, "test": test, "readout": readout, "assay": "mean",
                         "value": float(np.mean(list(per_assay.values())))})

    for label, key, seed in runs(S):
        run = run_name(key, seed)
        # bulk on the pair eval
        for c in cells:
            e, _ = eval_entry(run, f"{S['pair_eval']}/{c}")
            if e is None:
                print(f"!! {label} seed {seed}: no {S['pair_eval']}/{c}", flush=True)
                continue
            pa, pl = e["per_assay_pearson"], e["per_assay_loss"]
            add(label, seed, f"own_{c}", "r", pa)
            add(label, seed, f"own_{c}", "frac_ceiling",
                {a: pa[a] / ceil[(c, a)] for a in pa if pa[a] is not None and (c, a) in ceil})
            add(label, seed, f"own_{c}", "mse", pl)
        # floor: R1a-S's own run with the mixture reference
        if label == MUT:
            for c in cells:
                e, js = eval_entry(run, f"{S['floor_eval']}/{c}", need=lambda j: j.get("baseline_include_own"))
                if e is None or "baseline" not in e:
                    print(f"!! {label} seed {seed}: no mixture-reference run of {S['floor_eval']}/{c}", flush=True)
                    continue
                add(label, seed, f"floor_{c}", "r", {a: v["pearson"] for a, v in e["baseline"]["per_assay"].items()})
        # mixture
        e, _ = eval_entry(run, f"{S['mix_eval']}/mixed")
        if e is None:
            print(f"!! {label} seed {seed}: no {S['mix_eval']}/mixed", flush=True)
        else:
            add(label, seed, "mix", "r", e["per_assay_pearson"])
        # T3
        ro, _ = paired_npz(run, S["pair_eval"])
        if ro is None:
            print(f"!! {label} seed {seed}: no {S['pair_eval']}_paired.npz", flush=True)
        else:
            for r in ("auc", "sign", "dev"):
                add(label, seed, "t3", r, {a: v[r] for a, v in ro.items()})
            add(label, seed, "t3", "diff_r", paired_json(run, S["pair_eval"]))
        if label != MUT:
            continue
        # random split
        for c, cfg in S["split_evals"].items():
            ro, _ = paired_npz(run, cfg)
            if ro is None:
                print(f"!! {label} seed {seed}: no split control {cfg}", flush=True)
                continue
            add(label, seed, f"split_{c}", "auc", {a: v["auc"] for a, v in ro.items()})
            add(label, seed, f"split_{c}", "dev", {a: v["dev"] for a, v in ro.items()})
        # shuffle vs unshuffled on the same mini tiles
        for test, cfg in [("mini", S["mini_eval"]), ("shuffle", S["shuffle_eval"])]:
            ro, _ = paired_npz(run, cfg)
            if ro is None:
                print(f"!! {label} seed {seed}: no {cfg}", flush=True)
                continue
            add(label, seed, test, "auc", {a: v["auc"] for a, v in ro.items()})
            add(label, seed, test, "diff_r", paired_json(run, cfg))
            for c in cells:
                e, _ = eval_entry(run, f"{cfg}/{c}")
                if e is not None:
                    add(label, seed, test, f"r_{c}", e["per_assay_pearson"])
    return pd.DataFrame(rows)


def vals(df, model, test, readout, assay="mean"):
    s = df[(df.model == model) & (df.test == test) & (df.readout == readout) & (df.assay == assay)]
    return s.sort_values("seed").value.values


def endpoints(S, df):
    cells = list(S["cells"])
    rows = []
    # P1
    ok, det = True, []
    for c in cells:
        v, fl = vals(df, MUT, f"own_{c}", "r"), vals(df, MUT, f"floor_{c}", "r")
        floor = float(np.mean(fl)) if len(fl) else np.nan
        r = clear_vs_const(v, floor)
        ok &= r["clear"] and r["delta"] > 0
        det.append(f"{c}: own r {np.mean(v) if len(v) else np.nan:.3f} vs floor {floor:.3f}, {fmt_cmp(r)}")
    rows.append(endpoint_row("P1", "C1", "R1a-S own-track r clearly above the floor (bulk on every molecule), both "
                             "cells", ok, "; ".join(det)))
    # P2
    v = vals(df, MUT, "t3", "auc")
    r = clear_vs_const(v, 0.5)
    ok, det = r["clear"] and r["delta"] > 0, [f"AUC {np.mean(v) if len(v) else np.nan:.3f}, {fmt_cmp(r)}"]
    for c in cells:
        sv = vals(df, MUT, f"split_{c}", "auc")
        m = float(np.mean(sv)) if len(sv) else np.nan
        if c in S["split_evals"]:
            ok &= bool(abs(m - 0.5) <= SPLIT_TOL)
            det.append(f"split {c}: AUC {m:.3f} (seeds {' '.join(f'{x:.3f}' for x in sv)})")
    if len(S["split_evals"]) < len(cells):
        det.append(f"split run for {list(S['split_evals'])} only")
    rows.append(endpoint_row("P2", "C1", "R1a-S fiber-origin AUC clearly > 0.5; random split within 0.5 +- 0.02",
                             ok, "; ".join(det)))
    # P3
    ok, det = True, []
    for ro in ("auc", "diff_r"):
        r = clear_vs(vals(df, MUT, "t3", ro), vals(df, "DEV4NF", "t3", ro))
        ok &= r["clear"] and r["delta"] > 0
        det.append(f"{ro}: {fmt_cmp(r)}")
    rows.append(endpoint_row("P3", "C2", "R1a-S clearly above DEV4NF on AUC and differential r", ok, "; ".join(det)))
    # P4
    ok, det = True, []
    for test in [f"own_{c}" for c in cells] + ["mix"]:
        r = clear_vs(vals(df, MUT, test, "r"), vals(df, "Pair-S", test, "r"))
        ok &= not (r["clear"] and r["delta"] < 0)
        det.append(f"{test} r: {fmt_cmp(r)}")
    rows.append(endpoint_row("P4", "C3", "R1a-S not clearly below Pair-S on own r (each cell) and mix r", ok,
                             "; ".join(det)))
    # P5
    ok, det = True, []
    for m in ("MLP-S", "Rules-S"):
        r = clear_vs(vals(df, MUT, "mix", "r"), vals(df, m, "mix", "r"))
        ok &= r["clear"] and r["delta"] > 0
        det.append(f"vs {m}: {fmt_cmp(r)}")
    rows.append(endpoint_row("P5", "C4", "R1a-S clearly above MLP-S and Rules-S on mix r", ok, "; ".join(det)))
    return pd.DataFrame(rows)


def deltas(df):
    out = []
    for (test, ro, a), g in df.groupby(["test", "readout", "assay"]):
        ref = g[g.model == MUT].sort_values("seed").value.values
        if not len(ref):
            continue
        for m in ORDER:
            if m == MUT or m not in set(g.model):
                continue
            r = clear_vs(g[g.model == m].sort_values("seed").value.values, ref)
            out.append({"test": test, "readout": ro, "assay": a, "model": m, "ref": MUT, **r})
    return pd.DataFrame(out)


def print_tables(S, df, t, d, ep):
    cells = list(S["cells"])
    print(f"\n======== setup {S['name']}: cells {cells}, {S['chrom']}", flush=True)
    print("\n== primary endpoints (evals)")
    for r in ep.itertuples():
        print(f"  {r.endpoint} [{r.claim}] {'PASS' if r[4] else 'FAIL'}: {r.test}\n      {r.detail}", flush=True)
    m = t[t.assay == "mean"].copy()
    m["txt"] = [f"{a:.3f} ± {s:.3f}" if n > 1 else f"{a:.3f}" for a, s, n in zip(m["mean"], m["std"], m["size"])]
    print("\n== means over assays (mean ± sd over seeds)")
    show(m.pivot_table(index=["test", "readout"], columns="model", values="txt", aggfunc="first")
         .reindex(columns=[x for x in ORDER if x in set(m.model)]))
    print("\n== differences to R1a-S, mean over assays: delta (SE), * = clear")
    s = d[d.assay == "mean"].copy()
    s["txt"] = [f"{x:+.3f}" + ("" if np.isnan(e) else f" ({e:.3f})") + ("*" if c else "")
                for x, e, c in zip(s.delta, s.se, s["clear"])]
    show(s.pivot_table(index=["test", "readout"], columns="model", values="txt", aggfunc="first")
         .reindex(columns=[x for x in ORDER if x in set(s.model)]))
    for ro_test, ro in [("t3", "sign"), ("t3", "diff_r")]:
        r = d[(d.test == ro_test) & (d.readout == ro) & (d.model == "Pair-S") & (d.assay == "mean")]
        if len(r):
            print(f"\n   cost of mixture supervision (R1a-S - Pair-S), {ro}: {-r.delta.iloc[0]:+.3f} "
                  f"(SE {r.se.iloc[0]:.3f})", flush=True)
    print("\n== per assay (mean over seeds)")
    for test, ro in [(f"own_{c}", "r") for c in cells] + [(f"own_{c}", "frac_ceiling") for c in cells] + \
                    [("mix", "r"), ("t3", "auc"), ("t3", "sign"), ("t3", "diff_r")]:
        sub = t[(t.test == test) & (t.readout == ro) & (t.assay != "mean")]
        if len(sub):
            print(f"\n   {test} {ro}")
            show(sub.pivot_table(index="assay", columns="model", values="mean")
                 .reindex(columns=[x for x in ORDER if x in set(sub.model)]).round(3))
    print("\n== clear per-assay differences to R1a-S (own r, mix r, AUC, differential r)")
    q = d[(d.assay != "mean") & d["clear"] & d.readout.isin(["r", "auc", "diff_r"])]
    show(q[["test", "readout", "assay", "model", "delta", "se"]].round(3), index=False)
    print("\n== controls (R1a-S, per seed)")
    ctl = df[(df.model == MUT) & (df.assay == "mean") & df.test.str.match(r"(split|mini|shuffle|floor)")]
    show(ctl.pivot_table(index=["test", "readout"], columns="seed", values="value").round(3))
    sh, mi = df[(df.test == "shuffle") & (df.assay == "mean")], df[(df.test == "mini") & (df.assay == "mean")]
    if len(sh) and len(mi):
        j = sh.merge(mi, on=["model", "seed", "readout", "assay"], suffixes=("_shuffle", "_mini"))
        j["change"] = j.value_shuffle - j.value_mini
        print("\n   shuffle - unshuffled (same mini tiles), per seed")
        show(j.pivot_table(index="readout", columns="seed", values="change").round(3))


def figures(S, df, out):
    cells = list(S["cells"])
    panels = [(f"own_{c}", "r", f"{c.split('_')[0]} own r", None) for c in cells] + \
             [(f"floor_{c}", "r", f"{c.split('_')[0]} floor (R1a-S run)", None) for c in cells] + \
             [("mix", "r", "mixture r", None), ("t3", "auc", "fiber-origin AUC", 0.5),
              ("t3", "sign", "sign accuracy", 0.5), ("t3", "diff_r", "differential r", 0.0)]
    models = [m for m in ORDER if m in S["models"]]
    fig, axes = plt.subplots(2, (len(panels) + 1) // 2, figsize=(3.0 * ((len(panels) + 1) // 2), 6.2), squeeze=False)
    for ax, (test, ro, ttl, base) in zip(axes.flat, panels):
        for i, m in enumerate(models):
            v = vals(df, m, test, ro)
            if not len(v):
                continue
            ax.bar(i, v.mean(), 0.6, color=COL[m], alpha=0.35)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=COL[m], s=16, zorder=3)
        if base is not None:
            ax.axhline(base, color="#999999", lw=0.8, ls="--")
        ax.set_xticks(range(len(models)), models, rotation=60, fontsize=7)
        ax.set_title(ttl, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        v = df[(df.test == test) & (df.readout == ro) & (df.assay == "mean")].value
        if len(v):
            pad = (v.max() - v.min()) * 0.3 + 0.01
            ax.set_ylim(min(v.min() - pad, base if base is not None else np.inf), v.max() + pad)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Sealed test ({S['name']}): eval readouts per seed (dots), bar = mean; means over assays")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "evals_main.png"), dpi=130)
    plt.close(fig)

    tests = [(f"own_{c}", "r") for c in cells] + [("mix", "r"), ("t3", "auc"), ("t3", "diff_r")]
    assays = S["assays"]
    fig, axes = plt.subplots(len(tests), 1, figsize=(11, 2.3 * len(tests)), sharex=True)
    for ax, (test, ro) in zip(axes, tests):
        for j, m in enumerate(models):
            sub = df[(df.model == m) & (df.test == test) & (df.readout == ro)]
            q = sub.groupby("assay").value.agg(["mean", "std"])
            x = [i + (j - len(models) / 2) * 0.1 for i, a in enumerate(assays) if a in q.index]
            ax.errorbar(x, [q.loc[a, "mean"] for a in assays if a in q.index],
                        yerr=[0 if np.isnan(q.loc[a, "std"]) else q.loc[a, "std"] for a in assays if a in q.index],
                        fmt="o", ms=3.5, color=COL[m], lw=0.8, label=m)
        ax.set_ylabel(f"{test}\n{ro}", fontsize=8)
        ax.axhline(0.5 if ro == "auc" else 0, color="k", lw=0.5)
    axes[0].legend(fontsize=7, ncol=len(models))
    axes[-1].set_xticks(range(len(assays)), assays)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "evals_assays.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="sealed", choices=["sealed", "dryrun"])
    args = ap.parse_args()
    S = get_setup(args.setup)
    out = os.path.join(S["out"], "evals")
    os.makedirs(out, exist_ok=True)
    df = collect(S)
    if not len(df):
        print("no eval outputs found", flush=True)
        return
    df.to_csv(os.path.join(out, "evals_long.tsv"), sep="\t", index=False, float_format="%.5g")
    t = seed_table(df, ["test", "readout", "assay", "model"])
    t.to_csv(os.path.join(out, "evals_table.tsv"), sep="\t", index=False, float_format="%.5g")
    d = deltas(df)
    d.to_csv(os.path.join(out, "evals_deltas.tsv"), sep="\t", index=False, float_format="%.5g")
    ep = endpoints(S, df)
    ep.to_csv(os.path.join(out, "endpoints_evals.tsv"), sep="\t", index=False)
    print_tables(S, df, t, d, ep)
    figures(S, df, out)
    print(f"\nWrote {out}/", flush=True)


if __name__ == "__main__":
    main()
