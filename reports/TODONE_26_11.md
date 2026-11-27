# Done, November 2026

What was finished, newest first: a new item goes on top. One file per month (`TODONE_YY_MM.md`). Each item says
what was done, the result in a line, and where the details are. Open work stays in `TODO.md`.

## 27 Nov

- **Sealed test started:** protocol frozen (chr19, HCT116 + MCF-7; endpoints P1–P7, decisions D1–D5, Rules-S added,
  the XI test described). New: random-split control (`read_half`, `fiber_utils.ReadHalf`) and shuffle control
  (`shuffle_fibers`) in the eval; chr19 window sets; per-pair tile sets in `build_loci.py` (HCT116__MCF-7 on chr19);
  `setup_replicates.py --cells`; `targets_spec_sealed.yaml`; configs `data38`, `data39`, `eval33`–`eval41`. Data
  prep and the training of R1a-S / Pair-S / MLP-S (3 seeds) and Rules-S submitted on fir (`2026-11-27_32`–`51`).
  `reports/sealed_test_protocol.md`.
- **X inactivation with block phasing:** Xi set per HiPhase block from the other promoters' haplotype methylation,
  tested gene left out. Validated on GM12878 with the block layouts of the sealed XI cells: Xi call right ≥ 0.995,
  silenced-CGI result within 0.05 of chromosome-wide phasing in all 14 runs, ~200 of 425 genes called; escape + PAR
  control read without an Xi call (|HP1 − HP2|), passes for 5 of 5 marks in every run. `Dec_04.md`
  §3, `plots_code/26_dec_04/xi_blocks.py`, `xi_block_maps.py`.
- **TODO "Where we stand" rewritten** for the current state (R1a as the main model, results so far, next steps).

- **Bulk accuracy of every model compared** (DEV4NF, rules, MLP, R1a, R1a+F, Pair; per assay and cell, own track,
  ceiling fraction, MSE, mixture r): R1a = Pair (Pair lower MSE only), R1a slightly above DEV4NF (−0.01 to −0.03 r),
  R1a+F = R1a, MLP −0.13 to −0.18, rules −0.21 to −0.33. `Dec_04.md` §2, `plots_code/26_dec_04/bulk_models.py`.
- **Poised vs Polycomb (T5g) results:** poised tiles are unmethylated CGIs with about half their fibers open; the
  networks give the open fibers H3K4me3 and the closed ones H3K27me3 and lower H3K27me3 on methylated fibers (also
  in silico); they match the bulk track's class separation except at H1's methylated Polycomb domains; the rules
  miss Polycomb-only domains. `Dec_04.md` §1, `plots/26_dec_04/poised/`.
- **Poised vs Polycomb (T5g) defined and started:** 1 kb tiles classed from bulk H3K4me3 / H3K27me3 (FC ≥ 2 marked,
  < 1 unmarked), tests D1–D2, M1–M3, C. Genome fractions: poised 0.77% of autosomal tiles in H1 vs 0.05–0.14% in
  the other dev lines; Polycomb the other way round (1.9% vs 3.8–4.2%). New `build_targets.py --sets poised`,
  `lmsp_*` column (large-MSP fraction) in `molecule_predict.py`; molecule runs `2026-11-27_02`–`25` on fir.
  `TODO_extended.md` T5g.
- **Nov_27 report, definitions:** a "Fiber terms" section (MSP, large MSP = a run of ≥ 151 MSP bases, protected,
  CTCF fiber states, ...) and the exact rules and hand-crafted features of the ladder. `Nov_27.md` §6.

## 26 Nov

- **Baseline ladder compared (rungs 1–4 + Pair, both pairs, 3 seeds except rules):** the network is needed (+0.16
  mixture r over the MLP on hand-crafted features, clearly better cell separation, the only rung with CTCF
  footprints); adding the features to it changes nothing; rules help only where an annotation is the signal. The CpG
  flip effect shows up in the MLP too. `Nov_27.md` §6, `plots/26_nov_27/ladder/`, `plots_code/26_nov_27/ladder.py`.

## 25 Nov

- **Baseline ladder, first results (seed 919) and seeds 2 / 3 submitted:** mixture r (chr21 mini), GM12878 + HepG2 /
  K562 + H1: rules 0.32 / 0.23, MLP on features (running) / 0.36, R1a 0.62–0.63 / 0.52–0.53, network + features
  0.62 / 0.53. Deep learning is needed; the hand-crafted features add nothing to the network. Seeds 2 and 3 of rungs
  2 and 4 on both pairs: `2026-11-25_50`–`89` on fir. `TODO_extended.md` T11.
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
