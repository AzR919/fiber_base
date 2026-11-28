"""Sealed test, single-molecule readouts (reports/sealed_test_protocol.md): P6 (CTCF footprints), P7 (X inactivation)
and the secondary molecule readouts, with the dev definitions (the dev scripts are imported, not copied, so the
definitions are the same code).

Steps (each reads molecule_predict.py outputs, linked per run into ignore/sealed_molecules/<setup>/):
  holes   bulk-only: CTCF motif sites with no short-read data (CTCF or ATAC fold change exactly 0 over motif +- 1 kb)
          in either cell (26_nov_20/ctcf_holes.py on the setup's sites and cells); dropped before any CTCF statistic,
          as on dev. The dry run uses the dev holes file.
  ctcf    per run: 26_nov_13/ctcf_footprints.py (H1 footprint - open pred CTCF, H2 that minus the ATAC difference, H3
          open - nucleosome, H5 bound > unbound at matched motif score) on the pair's own fibers.
          P6 (R1a-S): per-site H1 at bound sites (sites with >= 3 footprint and >= 3 open fibers), passes when the
          median > 0 with Wilcoxon p < 0.01; on the seed mean (per site, the mean over the 3 seeds; sites scored
          in every seed) and in at least 2 of 3 seeds, in both cells. The same is reported for every model.
  cpg     CTCF x CpG (26_nov_27/ctcf_cpg.py): part C (in-silico flips at bound sites, accessible fibers; expected
          meth < 0, unmeth > 0 on pred CTCF / ATAC / H3K4me3, distal flips ~0), parts A (data) and B
          (associations) as data. Needs the --perturb CTCF runs.
  poised  poised vs Polycomb (26_dec_04/poised.py): D1, D2, M1 with the bulk reference, M2, M3, C.
  xi      X inactivation, block-phased (26_dec_04/xi_blocks.py rule) on the XI cells' fibers: per target the
          majority HiPhase block (PS) among HP-tagged fibers, only that block's fibers; Xi per block from the other
          informative promoters (leave one out, margin 0.2, >= 5 fibers with >= 3 CpGs per haplotype); per gene
          d = mean pred on Xa fibers - Xi fibers over the TSS +- 500 bp.
          P7 (R1a-S, seed mean): per cell and active mark (ATAC, H3K4me3, H3K27ac), silenced CGI promoters: per gene
          the mean of d over seeds, median > 0 with Wilcoxon p < 0.01; passes when each mark passes in >= 2 of 3
          cells. Secondary: H3K9me3 / H3K27me3 (expected < 0), the chr21 null (random orientation per block), the
          escape control (|HP1 - HP2|, silenced > escape + PAR), coverage (genes called), XIST, every model.

    python plots_code/sealed/sealed_molecules.py [--setup sealed|dryrun] [--step holes|ctcf|cpg|poised|xi|all]
Writes <out>/{ctcf,cpg,poised,xi}/ tables and figures and <out>/endpoints_molecules_<step>.tsv; prints the tables.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (ORDER, COL, MODEL_UNDER_TEST as MUT, TARGETS_DIR, get_setup, runs, cell_name,  # noqa: E402
                    link_molecules, out_tag, show, endpoint_row)
from sm_stats import wilcoxon_p  # noqa: E402

P_PASS = 0.01
ACTIVE = ["atac", "h3k4me3", "h3k27ac"]
REPRESSIVE = ["h3k9me3", "h3k27me3"]


def passes(x, sign=1):
    """Protocol 'passes': median with the expected sign and Wilcoxon p < 0.01 over sites / genes."""
    x = pd.Series(x).dropna()
    med = float(x.median()) if len(x) else np.nan
    p = wilcoxon_p(x) if len(x) >= 5 else np.nan
    return {"n": len(x), "median": med, "p": p, "pass": bool(len(x) >= 5 and np.sign(med) == sign and p < P_PASS)}


def seed_mean(per_seed):
    """{seed: Series over sites / genes} -> per-site mean over the seeds, sites present in every seed."""
    w = pd.concat(per_seed, axis=1).dropna()
    return w.mean(axis=1)


# ---------------------------------------------------------------- holes
def step_holes(S):
    if S["holes_cells"] is None:
        print(f"setup {S['name']} uses {S['holes']}", flush=True)
        return
    import ctcf_holes as ch
    ch.BED = f"{TARGETS_DIR}/{S['ctcf_targets']}.bed"
    ch.OUT = os.path.dirname(S["holes"])
    ch.CELLS = S["holes_cells"]
    ch.main()


# ---------------------------------------------------------------- CTCF
def per_site_h1(path, cell, exclude):
    from ctcf_footprints import fiber_state, MIN_PER_CLASS
    df = pd.read_csv(path, sep="\t", usecols=lambda c: not c.startswith("pred_") or c == "pred_ctcf")
    df = df[~df.target.isin(exclude)]
    df = df[(df["set"] != "unbound") & (df[cell] > 0)].copy()
    df["state"] = fiber_state(df)
    g = df[df.state.isin(["footprint", "open"])].groupby(["target", "state"]).pred_ctcf.agg(["mean", "size"])
    g = g[g["size"] >= MIN_PER_CLASS]["mean"].unstack("state")
    if "footprint" not in g or "open" not in g:
        return pd.Series(dtype=float)
    return (g["footprint"] - g["open"]).dropna()


def step_ctcf(S):
    out = os.path.join(S["out"], "ctcf")
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(S["holes"]):
        raise FileNotFoundError(f"{S['holes']} missing: run --step holes first")
    exclude = set(open(S["holes"]).read().split())
    samples = list(S["cells"])
    rows, h1 = [], {}
    for label, key, seed in runs(S):
        d = link_molecules(S, key, seed, S["ctcf_targets"], samples)
        if d is None:
            continue
        t = out_tag(key, seed)
        with open(os.path.join(out, f"log_ctcf_{t}.txt"), "w") as log:
            subprocess.run([sys.executable, "-u", os.path.join(HERE, "..", "26_nov_13", "ctcf_footprints.py"),
                            "--model", key, "--seed", str(seed), "--dir", d, "--exclude", S["holes"], "--out", out],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        cs = pd.read_csv(os.path.join(out, f"ctcf_summary_{t}.tsv"), sep="\t")
        hb = pd.read_csv(os.path.join(out, f"ctcf_bound_vs_unbound_by_score_{t}.tsv"), sep="\t")
        for s in samples:
            link = S["cells"][s]
            b = cs[(cs.site_set == "bound") & (cs["sample"] == link)]
            if len(b):
                b = b.iloc[0]
                for col, ro in [("d_fp_open_ctcf_median", "H1"), ("d_ctcf_minus_atac_median", "H2"),
                                ("d_open_nuc_ctcf_median", "H3"), ("d_fp_open_ctcf_p", "H1_p"),
                                ("d_ctcf_minus_atac_p", "H2_p"), ("d_open_nuc_ctcf_p", "H3_p"),
                                ("d_fp_open_ctcf_n", "H1_sites"), ("footprint_frac_of_accessible", "fp_frac")]:
                    rows.append({"model": label, "seed": seed, "sample": link, "readout": ro, "value": b[col]})
            g = hb[hb["sample"] == link]
            rows.append({"model": label, "seed": seed, "sample": link, "readout": "H5",
                         "value": float((g.bound_mean > g.unbound_mean).mean()) if len(g) else np.nan})
            h1[(label, seed, link)] = per_site_h1(os.path.join(d, f"fibers_{link}.tsv.gz"), cell_name(link), exclude)
        print(f"ctcf {label} seed {seed} done", flush=True)
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(out, "ctcf_long.tsv"), sep="\t", index=False, float_format="%.5g")

    # P6 and the same rule for every model
    prow, ep_ok, det = [], True, []
    for label in [m for m in ORDER if m in S["models"]]:
        for s in samples:
            link = S["cells"][s]
            per = {sd: h1[(label, sd, link)] for (lb, sd, lk) in h1 if lb == label and lk == link}
            if not per:
                continue
            for sd, x in sorted(per.items()):
                prow.append({"model": label, "sample": link, "seed": sd, **passes(x)})
            sm = passes(seed_mean(per)) if len(per) > 1 else passes(next(iter(per.values())))
            n_pass = sum(passes(x)["pass"] for x in per.values())
            prow.append({"model": label, "sample": link, "seed": "mean", **sm, "seeds_passing": n_pass,
                         "seeds": len(per)})
            if label == MUT:
                ok = sm["pass"] and n_pass >= 2
                ep_ok &= ok
                det.append(f"{link}: seed mean median {sm['median']:+.3f} (p {sm['p']:.1e}, {sm['n']} sites), "
                           f"{n_pass} of {len(per)} seeds pass")
    p6 = pd.DataFrame(prow)
    p6.to_csv(os.path.join(out, "ctcf_P6.tsv"), sep="\t", index=False, float_format="%.5g")
    if not det:
        ep_ok, det = False, ["no R1a-S CTCF output"]
    ep = pd.DataFrame([endpoint_row("P6", "C5", "CTCF footprint effect H1 (R1a-S, bound sites) passes on the seed "
                                    "mean and in >= 2 of 3 seeds, both cells", ep_ok, "; ".join(det))])
    ep.to_csv(os.path.join(S["out"], "endpoints_molecules_ctcf.tsv"), sep="\t", index=False)
    print(f"\n======== CTCF ({S['name']})")
    print(f"P6 {'PASS' if ep_ok else 'FAIL'}: {'; '.join(det)}", flush=True)
    print("\n== per-site H1 (footprint - open pred CTCF, bound sites): per seed and seed mean")
    show(p6.round(4), index=False)
    print("\n== H1 / H2 / H3 (per-site medians) and H5 (share of score bins bound > unbound), per seed")
    show(long[long.readout.isin(["H1", "H2", "H3", "H5", "fp_frac"])]
         .pivot_table(index=["sample", "readout"], columns=["model", "seed"], values="value").round(3))


# ---------------------------------------------------------------- CTCF x CpG
def step_cpg(S):
    import ctcf_cpg as cc
    out = os.path.join(S["out"], "cpg")
    os.makedirs(out, exist_ok=True)
    cc.HOLE_SET = set(open(S["holes"]).read().split())
    samples = list(S["cells"])
    brows, crows, tables_a = [], [], {}
    for label, key, seed in runs(S):
        d = link_molecules(S, key, seed, S["ctcf_targets"], samples, perturb=True)
        if d is None:
            continue
        for s in samples:
            link = S["cells"][s]
            df = cc.load(d, link)
            if label == "DEV4NF" and seed == 919:
                tables_a[link] = df
            brows += cc.part_b(df, label, seed, link)
            crows += cc.part_c(df, label, seed, link)
        print(f"cpg {label} seed {seed} done", flush=True)
    B, C = pd.DataFrame(brows), pd.DataFrame(crows)
    if tables_a:
        counts, a1, a2 = cc.part_a(tables_a)
    else:
        print("!! no DEV4NF seed 919 perturb run: part A skipped", flush=True)
        counts = a1 = a2 = pd.DataFrame()
    for name, t in [("cpg_A_counts", counts), ("cpg_A1", a1), ("cpg_A2", a2), ("cpg_B", B), ("cpg_C", C)]:
        t.to_csv(os.path.join(out, f"{name}.tsv"), sep="\t", index=False, float_format="%.4g")
    print(f"\n======== CTCF x CpG ({S['name']})")
    if len(C):
        c = C[(C.sites == "bound")]
        print("\n== C: flips at bound sites, accessible fibers: median per-site change (mean over seeds)")
        show(c.groupby(["mode", "assay", "model", "sample"])["median"].mean().unstack("sample").unstack("model")
             .round(4))
        print("\n== C: share of runs with p < 0.01 and the expected sign (meth < 0, unmeth > 0; distal: any)")
        c = c.assign(expect=np.where(c["mode"] == "meth", -1, np.where(c["mode"] == "unmeth", 1, 0)))
        c = c.assign(ok=(np.sign(c["median"]) == c.expect) & (c.p < P_PASS))
        show(c[c.expect != 0].groupby(["mode", "assay", "model"]).ok.agg(["sum", "size"]).unstack("model"))
    if len(B):
        b = B[(B.sites == "bound + other") & (B.assay == "ctcf")]
        print("\n== B (association): pred CTCF methylated - unmethylated, accessible / footprint / open fibers")
        show(b.groupby(["fibers", "model", "sample"])["median"].mean().unstack("sample").round(4))
    if len(a1):
        print("\n== A (data)")
        show(counts, index=False)
        show(a1, index=False)
        show(a2, index=False)


# ---------------------------------------------------------------- poised
def poised_d1(S):
    f = S["poised_fractions"]
    if not os.path.exists(f):
        print(f"  {f} missing; D1 skipped", flush=True)
        return pd.DataFrame()
    fr = pd.read_csv(f, sep="\t")
    cells = {cell_name(x) for x in S["cells"].values()}
    fr = fr[fr.cell.isin(cells)]
    tot = fr.groupby(["cell", "class"])[["n_tiles", "n_cgi", "n_valid", "n_valid_cgi"]].sum().reset_index()
    tot["genome_frac"] = tot.n_tiles / tot.n_valid
    tot["frac_at_cgi"] = tot.n_cgi / tot.n_tiles.where(tot.n_tiles > 0)
    tot["cgi_frac_in_class"] = tot.n_cgi / tot.n_valid_cgi
    ch = fr[fr.chrom == S["chrom"]].set_index(["cell", "class"])
    tot[f"{S['chrom']}_frac"] = [ch.loc[(c, k), "n_tiles"] / ch.loc[(c, k), "n_valid"] if (c, k) in ch.index
                                 else np.nan for c, k in zip(tot.cell, tot["class"])]
    tot[f"{S['chrom']}_tiles"] = [ch.loc[(c, k), "n_tiles"] if (c, k) in ch.index else np.nan
                                  for c, k in zip(tot.cell, tot["class"])]
    return tot


def step_poised(S):
    import poised as po
    out = os.path.join(S["out"], "poised")
    os.makedirs(out, exist_ok=True)
    links = [S["cells"][s] for s in S["cells"]]
    po.OUT, po.PAIRS = out, {"S": links}
    po.ORDER = [m for m in ORDER if m in S["models"]]
    po.COL = {**COL, "Bulk": "#000000"}
    samples = list(S["cells"])
    d1 = poised_d1(S)
    m1, m2, m3, c = [], [], [], []
    tables = tables3 = None
    for label, key, seed in runs(S):
        d = link_molecules(S, key, seed, S["poised_targets"], samples)
        if d is None:
            continue
        for link in links:
            df = po.load(d, link)
            df3 = po.load(d, link, marked=3.0)
            if label == "DEV4NF" and seed == 919:   # D2 and the bulk reference: the fibers are the same for every model
                tables = {**(tables or {}), link: df}
                tables3 = {**(tables3 or {}), link: df3}
            a, b, cc_ = po.part_m(df, label, seed, link)
            m1 += a; m2 += b; m3 += cc_
            c += po.part_c(df, label, seed, link)
            a3, _, _ = po.part_m(df3, label, seed, link)
            m1 += [{**r, "readout": r["readout"] + "@FC3"} for r in a3]
        print(f"poised {label} seed {seed} done", flush=True)
    if tables is None:
        print("!! no DEV4NF seed 919 poised output: D2 and the bulk reference skipped", flush=True)
        d2 = d2t = pd.DataFrame()
    else:
        d2, d2t = po.part_d2(tables, "FC>=2")
        d2b, _ = po.part_d2(tables3, "FC>=3")
        d2 = pd.concat([d2, d2b], ignore_index=True)
        for s in tables:
            m1 += po.part_bulk(tables[s], s)
            m1 += [{**r, "readout": r["readout"] + "@FC3"} for r in po.part_bulk(tables3[s], s)]
    m1, m2, m3, c = map(pd.DataFrame, (m1, m2, m3, c))
    for name, t in [("D1", d1), ("D2", d2), ("D2_tiles", d2t), ("M1", m1), ("M2", m2), ("M3", m3), ("C", c)]:
        t.to_csv(os.path.join(out, f"poised_{name}.tsv"), sep="\t", index=False, float_format="%.5g")
    s1 = po.seed_summary(m1, ["model", "sample", "readout"], "value") if len(m1) else pd.DataFrame()
    s2 = po.seed_summary(m2, ["model", "sample", "class", "assay", "contrast"], "median") if len(m2) else pd.DataFrame()
    s3 = po.seed_summary(m3, ["model", "sample", "class"], "median_rho") if len(m3) else pd.DataFrame()
    sc = po.seed_summary(c, ["model", "sample", "class", "mode", "assay"], "median") if len(c) else pd.DataFrame()
    pd.concat([s1.assign(part="M1"), s2.assign(part="M2"), s3.assign(part="M3"), sc.assign(part="C")],
              ignore_index=True).to_csv(os.path.join(out, "poised_summary.tsv"), sep="\t", index=False,
                                        float_format="%.5g")
    print(f"\n======== poised vs Polycomb ({S['name']})")
    if len(d1):
        print("\n== D1 genome fractions (all autosomes) and on the scored chromosome")
        show(d1.round(4), index=False)
    if len(d2):
        print("\n== D2 fiber states per class (DEV4NF seed 919 fibers)")
        show(d2.round(3), index=False)
    if len(s1):
        print("\n== M1 (mean over seeds; Bulk = the bulk tracks on the same tiles, the reference)")
        show(s1[~s1.readout.str.startswith("mean_")].pivot_table(index=["sample", "readout"], columns="model",
                                                                   values="mean").round(3))
    if len(m2):
        print("\n== M2 key contrasts: open - closed H3K4me3 at poised tiles (> 0), methylated - unmethylated "
              "H3K27me3 at Polycomb tiles (< 0); median over tiles, mean over seeds; n tiles")
        key = m2[((m2["class"] == "poised") & (m2.assay == "h3k4me3") & (m2.contrast == "open-closed|unmeth")) |
                 ((m2["class"] == "polycomb") & (m2.assay == "h3k27me3") & (m2.contrast == "meth-unmeth|closed"))]
        show(key.groupby(["sample", "class", "assay", "contrast", "model"]).agg(
            median=("median", "mean"), n_tiles=("n_tiles", "min"), p_max=("p", "max")).unstack("model").round(3))
        print("\n== M2 all (median over tiles, mean over seeds)")
        show(s2.pivot_table(index=["sample", "class", "assay", "contrast"], columns="model", values="mean").round(3))
    if len(s3):
        print("\n== M3 (median per-tile rho, mean over seeds)")
        show(s3.pivot_table(index=["sample", "class"], columns="model", values="mean").round(3))
    if len(sc):
        print("\n== C flips (median per-tile change, mean over seeds)")
        show(sc.pivot_table(index=["sample", "class", "mode", "assay"], columns="model", values="mean").round(3))
    if len(m1) and len(d2):
        po.figures(d1, d2[d2.set == "FC>=2"], m1, m2, m3, c)


# ---------------------------------------------------------------- X inactivation
def xi_load(path):
    """Per (target, hp) table on the majority block's fibers, target meta, block per target."""
    import xi_blocks as xb
    df = pd.read_csv(path, sep="\t")
    df = df[df.hp.isin([1, 2]) & df.ps.notna()].copy()
    df["hp"] = df.hp.astype(int)
    df["ps"] = df.chrom.astype(str) + ":" + df.ps.astype(str)
    maj = df.groupby("target").ps.agg(lambda v: v.value_counts().idxmax())
    df = df[df.ps == df.target.map(maj)]
    df["meth"] = np.where(df.cpg_n_t >= xb.MIN_CPG, df.cpg_meth_t, np.nan)
    cols = [f"pred_{a}" for a in xb.MARKS] + ["cpg_meth_t", "msp_t"]
    tab = df.groupby(["target", "hp"]).agg(n=("read", "size"), n_meth=("meth", "count"), meth=("meth", "mean"),
                                           **{c: (c, "mean") for c in cols}).reset_index()
    meta = df.groupby("target").agg(chrom=("chrom", "first"), gene=("gene", "first"), cls=("class", "first"),
                                    cgi=("cgi", "first"))
    return tab, meta, maj.reindex(meta.index)


def step_xi(S):
    import xi_blocks as xb
    out = os.path.join(S["out"], "xi")
    os.makedirs(out, exist_ok=True)
    genes, summ = [], []
    for label, key, seed in runs(S):
        d = link_molecules(S, key, seed, S["xi_targets"], S["xi_samples"])
        if d is None:
            continue
        for sample in S["xi_samples"]:
            tab, meta, block = xi_load(os.path.join(d, f"fibers_{S['cells'].get(sample, sample)}.tsv.gz"))
            groups = xb.group_of(meta)
            xi, score = xb.xi_blocks(tab, meta, block, {})
            dd = xb.deltas(tab, xi)
            rnd = xb.random_orientation(meta, block, np.random.default_rng(919))
            dn = xb.deltas(tab, rnd)
            for kind, x in [("called", dd), ("random_orientation", dn)]:
                x = x.assign(group=groups.reindex(x.index).values, gene=meta.gene.reindex(x.index).values,
                             chrom=meta.chrom.reindex(x.index).values)
                genes.append(x.reset_index().assign(model=label, seed=seed, sample=sample, kind=kind))
            ab = xb.absd_readouts(tab, meta, groups, block)
            chrx = meta.index[meta.chrom == "chrX"]
            summ.append({"model": label, "seed": seed, "sample": sample,
                         "chrX_targets_phased": int(block.reindex(chrx).notna().sum()),
                         "chrX_blocks": int(block.reindex(chrx).nunique()),
                         "silenced_cgi_called": int((groups.reindex(dd.index) == "silenced_cgi").sum()),
                         "silenced_cgi_phased": int(((groups == "silenced_cgi") & block.notna()).sum()),
                         "escape_par_called": int((groups.reindex(dd.index) == "escape_par").sum()),
                         "xist_called": int((groups.reindex(dd.index) == "xist").sum()),
                         **xb.readouts(dd, groups), **ab})
        print(f"xi {label} seed {seed} done", flush=True)
    if not genes:
        print("no XI outputs", flush=True)
        return
    G = pd.concat(genes, ignore_index=True)
    G.to_csv(os.path.join(out, "xi_genes.tsv"), sep="\t", index=False, float_format="%.5g")
    T = pd.DataFrame(summ)
    T.to_csv(os.path.join(out, "xi_runs.tsv"), sep="\t", index=False, float_format="%.5g")

    # per model x sample x group x mark: per seed and seed mean
    rows = []
    marks = {**{a: 1 for a in ACTIVE}, **{a: -1 for a in REPRESSIVE}}
    for (label, sample, kind), g in G.groupby(["model", "sample", "kind"]):
        for grp in ["silenced_cgi", "escape_par", "chr21", "xist"]:
            sub = g[g.group == grp]
            for a, e in marks.items():
                per = {sd: s.set_index("target")[f"pred_{a}"] for sd, s in sub.groupby("seed")}
                if not per:
                    continue
                for sd, x in per.items():
                    rows.append({"model": label, "sample": sample, "kind": kind, "group": grp, "mark": a,
                                 "expected": e, "seed": sd, **passes(x, e), "frac_expected": float((np.sign(x) == e).mean())})
                x = seed_mean(per) if len(per) > 1 else next(iter(per.values()))
                rows.append({"model": label, "sample": sample, "kind": kind, "group": grp, "mark": a, "expected": e,
                             "seed": "mean", **passes(x, e), "frac_expected": float((np.sign(x) == e).mean())})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "xi_tests.tsv"), sep="\t", index=False, float_format="%.5g")

    # P7
    p = R[(R.model == MUT) & (R.kind == "called") & (R.group == "silenced_cgi") & (R.seed == "mean") &
          R.mark.isin(ACTIVE)]
    ok, det = len(p) > 0, []
    for a in ACTIVE:
        q = p[p.mark == a]
        n_pass = int(q["pass"].sum())
        ok &= n_pass >= 2
        det.append(f"{a}: passes in {n_pass} of {len(q)} cells (" + ", ".join(
            f"{r.sample} median {r.median:+.3f} p {r.p:.1e} n {r.n}" for r in q.itertuples()) + ")")
    ep = pd.DataFrame([endpoint_row("P7", "C5", "XI active marks higher on the active X at silenced CGI promoters "
                                    "(R1a-S seed mean, block-phased): each mark passes in >= 2 of 3 cells", ok,
                                    "; ".join(det) if det else "no R1a-S XI output")])
    ep.to_csv(os.path.join(S["out"], "endpoints_molecules_xi.tsv"), sep="\t", index=False)
    print(f"\n======== X inactivation ({S['name']})")
    print(f"P7 {'PASS' if ok else 'FAIL'}: {'; '.join(det)}", flush=True)
    print("\n== coverage and readouts per run (xi_blocks readouts: fraction of genes with the expected sign)")
    cols = ["model", "seed", "sample", "chrX_targets_phased", "chrX_blocks", "silenced_cgi_phased",
            "silenced_cgi_called", "escape_par_called", "xist_called", "silenced_cgi_frac_expected",
            "chr21_frac_expected", "absd_marks_pass"]
    show(T[[c for c in cols if c in T]].round(3), index=False)
    print("\n== seed-mean tests per mark (silenced CGI promoters, called): median, p, pass")
    s = R[(R.kind == "called") & (R.group == "silenced_cgi") & (R.seed == "mean")]
    s = s.assign(txt=[f"{m:+.3f} p{pv:.0e}{'*' if ok_ else ''} n{n}" for m, pv, ok_, n in
                      zip(s["median"], s.p, s["pass"], s.n)])
    show(s.pivot_table(index=["mark", "sample"], columns="model", values="txt", aggfunc="first")
         .reindex(columns=[m for m in ORDER if m in set(s.model)]))
    print("\n== nulls: chr21 promoters (random orientation per block) and escape + PAR (called), seed mean")
    s = R[(R.seed == "mean") & (((R.kind == "random_orientation") & (R.group == "chr21")) |
                                ((R.kind == "called") & (R.group == "escape_par")))]
    show(s.pivot_table(index=["group", "mark", "sample"], columns="model", values="median").round(3))
    print("\n== escape control (|HP1 - HP2|, silenced > escape + PAR, one-sided MWU p) per run")
    pc = [c for c in T.columns if c.startswith("absd_") and c.endswith("_p")]
    show(T[["model", "seed", "sample"] + pc + ["absd_marks_pass"]].round(4), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="sealed", choices=["sealed", "dryrun"])
    ap.add_argument("--step", default="all", choices=["holes", "ctcf", "cpg", "poised", "xi", "all"])
    args = ap.parse_args()
    S = get_setup(args.setup)
    os.makedirs(S["out"], exist_ok=True)
    steps = ["holes", "ctcf", "cpg", "poised", "xi"] if args.step == "all" else [args.step]
    for st in steps:
        print(f"\n################ step {st}", flush=True)
        {"holes": step_holes, "ctcf": step_ctcf, "cpg": step_cpg, "poised": step_poised, "xi": step_xi}[st](S)


if __name__ == "__main__":
    main()
