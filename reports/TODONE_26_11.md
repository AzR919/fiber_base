# Done, November 2026

What was finished, newest first: a new item goes on top. One file per month (`TODONE_YY_MM.md`). Each item says
what was done, the result in a line, and where the details are. Open work stays in `TODO.md`.

## 25 Nov

- **Baseline ladder, rung 1 built and submitted:** per-base annotation tracks (CGI, TSS ± 2 kb, gene body,
  soft-masked repeat, ZNF gene, CTCF motif; chr20 / 21 / 22 / X), the rule model `rules01` (fixed rules + 16
  calibration parameters) and the `ref_annot` loader option; seed 919 on both dev pairs (`2026-11-25_37`–`44`).
  `TODO_extended.md` T11.
- **Baseline ladder, rungs 2 and 4 built and submitted:** `handcrafted.py` (31 fixed per-position features from the
  fiber and the reference DNA, computed inside the model), `mlp01` (rung 2, 6.7 k parameters) and `unet03` with
  `extra_features` (rung 4); configs `model22`, `model23`; seed 919 on both dev pairs running on fir
  (`2026-11-25_17`–`36`). Found on the way: fibertools MSPs include linkers, so "open region" = MSP runs ≥ 151 bp.
  `TODO_extended.md` T11.
- **Hand-crafted baselines designed:** the ladder (floor; rules; a small MLP on simple features; the network; the
  network + features), per-assay rules with the annotations they use, the expected results, and a new CTCF
  specificity test (footprints on other factors' motifs). Confirmed: rung 1 as drafted; rung 2 a small MLP on
  features from the fiber and DNA only, no external annotations. `TODO_extended.md` T11.
- **CTCF × CpG methylation (T5f):** data, model on real fibers, and in-silico CpG flips (new
  `molecule_predict.py --perturb`). Claim kept: CpG methylation lowers active marks locally, in every model and seed,
  mixture-trained included (flips 1.5 kb away do ~nothing). It is not CTCF-specific, and a CTCF-specific effect can't
  be tested: the data show the methylation–binding link between sites but barely within a site. `Nov_27.md` §5,
  `plots/26_nov_27/ctcf_cpg/`.
- **Sealed X-inactivation cells:** checked 19 Fiber-seq lines for an Xi and haplotype tags. HCT116 has no Xi, and
  MCF-7's is partial. HG008_3E4, GM25455 (LCL) and HEK293T have a clear Xi, haplotype blocks and no use so far: they
  become the sealed XI test. `TODO.md` §3, `plots_code/26_nov_27/xi_cells_check.py`.
- **Scope decided:** two cell types at a 0.5 share in this paper; more cells and unequal shares go to a follow-up
  paper. Recipe locked: no more dev pairs, no 3-cell mixture (I28 out of scope). Success-criteria details deferred.
  `TODO.md` §1.
- **Nov_27 report:** R1a recipe, first run, seed noise, the two success-criteria questions with figures
  (`criteria_plots.py`). `reports/Nov_27.md`.

## 24 Nov

- **Seed noise, both dev pairs:** R1a and Pair on GM12878 + HepG2 and K562 + H1, 3 seeds each (10 new runs on fir).
  R1a = Pair on bulk and mixture r; Pair clearly better on sign accuracy (+0.04 / +0.055) and differential r
  (+0.03 / +0.056), where R1a is at DEV4NF's level. The CTCF footprint effect can vanish in a single training, for
  any supervision. `Nov_27.md` §3, `plots/26_nov_27/seeds/`.
- **Moved to fir:** nibi went down. Data copied nibi → fir (regions, annotations, QC, ENCODE replicates); the four
  R1a runs that stopped at the time limit were finished with `RESUME=latest`.

## 23 Nov

- **Two-cluster setup:** `misc/cluster.sh` (explicit sync through the local repo), slurm scripts that run unchanged
  on nibi and fir, CPU / memory overrides for fir on the command line. `CLAUDE.md`.
- **Mixture recipe R1a locked:** first realism run on GM12878 + HepG2 (R0, R1a, Pair, DEV4NF's channels). R1a
  matches R0 on T3 and is level with the pair model on bulk; every biology test keeps its direction. R2 dropped, R1b
  optional. `Nov_27.md` §1–2.
- **Eval fix:** two mixed configs in one eval run were both saved as `mixed`; now keyed `<config>/mixed`
  (`evaluator.py`).

## 18 Nov

- **DEV4NF became the main model** (DEV4 without the FIRE channel): same bulk accuracy, no bulk-assay knowledge in
  the input. `Nov_13_extended.md`.
- **Nov 20 report:** CTCF footprints (rerun without the 75 sites lacking bulk ChIP data) and X inactivation per
  haplotype at silenced CGI promoters, around DEV4NF; locus figures. `Nov_20.md`, `Nov_20_figs.md`.
- **Plan:** the mixture-trained model as the main direction, with a sealed test. `TODO_extended.md` §2.
- **SMIF paper draft** (abstract, related work, setup, methods, evaluation, preliminary results). `reports/paper/`.

## 17 Nov

- **T11 added:** simple per-fiber baselines (does the network learn more than an obvious reading of the input?).

## 11–12 Nov

- **Second training seed** of DEV4 and the four input ablations; seed noise of every reported number.
  `Nov_13_extended.md` §10.
- **ENCODE replicate ceiling (I11):** how close each model gets to what the replicates allow.
  `Nov_13_extended.md` §9.

## 7–10 Nov

- **Input ablations (T10):** no FIRE, m6A + CpG, no accessibility, no CpG. `Nov_13_extended.md` §6.
- **Single-molecule tools:** `molecule_predict.py` (per-fiber predictions at target BEDs), the target builder
  (`data_prep/build_targets.py`), first CTCF footprint and X-inactivation analyses. `Nov_13_extended.md` §7–8.
- **Nov 13 report,** extended and short versions, including the mixture test on two pairs (T3 / T3b).

## 1–4 Nov

- **Split v2 and the main runs:** DEV4 (4 dev cell types), five specialists, pair models A / B, mixture models A / B
  (`train04`). Checkpoint loading and resume logic. `Nov_13.md`.
