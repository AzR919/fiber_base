# 13th Nov (extended)

The detailed version: every experiment since 30 Oct with its setup, the hypothesis it tested, the full tables and
figures, and the outcome. The short version is `Nov_13.md`. Plan and test definitions: `TODO.md`,
`TODO_extended.md`.

## Experiments at a glance

| # | Experiment | Hypothesis / question | Outcome |
|---|---|---|---|
| 1 | [DEV5 reference model](#1-dev5-v1-the-all-cells-reference-model) | Does the 5-dev-cell model beat the cross-cell average on chr21? | On MSE yes (7 of 8 cell types); on Pearson only in dev cell types, not held out. Weak on H3K36me3 / H3K9me3 / CTCF |
| 2 | [Panc1 and split v2](#2-panc1-and-the-new-split-v2-panc1-moves-to-held-out) | Is Panc1's sample the problem? Does training on it help? | Sample doubtful, training on it buys nothing → held out; **DEV4** (4 dev) is the main model. Specialists +0.03–0.06 r |
| 3 | [Mixture test on two pairs](#3-mixture-test-on-two-pairs-t3-repeated-on-k562h1) | Mixture-only supervision reaches the own-track ceiling (T3), also on a second pair; how does it transfer to an unseen pair? | Ceiling reached on both pairs (AUC 0.637 vs 0.635, 0.666 vs 0.665). On an unseen pair supervision type doesn't matter; transfer to K562–H1 is poor |
| 4 | [CTCF on K562–H1](#4-ctcf-on-k562h1-a-weak-h1-track-plus-a-model-shift) | (a) H1's CTCF track is weak, or (b) H1's fibers are more accessible | (a) yes, (b) no; plus a constant model shift toward H1 |
| 5 | [Offset breakdown, all assays](#5-the-same-breakdown-for-every-assay-on-both-pairs) | Which failures are track problems, compression or cell-level offsets? | Three failure types; track outliers H1 CTCF, HepG2 / H1 H3K9me3 |
| 6 | [Input ablations](#6-input-ablations-t10-which-fiber-inputs-does-the-model-use) | Which fiber inputs does the model use? Five models: all inputs, no FIRE, m6A + CpG (no fibertools calls), no accessibility (CpG only), no CpG | Fibertools calls add nothing in bulk (no FIRE and m6A + CpG even gain +0.02–0.03 on H3K27ac / H3K4me1). m6A carries ATAC, H3K4me1, H3K27ac, CTCF, H3K9me3; CpG carries H3K36me3 (−0.25 without it) and H3K27me3. All replicate in a second seed |
| 7 | [CTCF footprints on single molecules](#7-ctcf-footprints-on-single-molecules-t5e) | Fibers with a protected CTCF motif get higher predicted CTCF than open fibers at the same site, more than for ATAC | Yes for every model that sees m6A, in both seeds; null clean. The size varies 2–10× between trainings (DEV4 +0.11 / +0.73). **No FIRE is the only model strong and CTCF-specific in both seeds** (+1.14 / +0.95) |
| 8 | [X inactivation per haplotype](#8-x-inactivation-per-haplotype-in-gm12878-t5b) | At silenced chrX promoters, the active X gets more ATAC / H3K4me3 / H3K27ac and less H3K27me3 / H3K9me3 than the inactive X; escape / PAR / autosomal ≈ 0; XIST reversed | Pass for ATAC, H3K4me3, H3K27ac (71–91% of genes, both seeds), nulls flat, XIST reversed. **H3K27me3 fails** (opposite sign). H3K9me3 passes robustly only without CpG; seed-dependent otherwise |
| 9 | [ENCODE replicate ceiling](#9-encode-replicate-ceiling-i11-how-much-of-each-target-can-a-model-get) | How much of each target is reproducible? Which "weak" assays are target noise? | Replicates agree at r ≈ 0.4, so the reachable r is ~0.74. DEV4 gets 66% of it in dev, 58% held out, 46% in Panc1. Dev H3K36me3 / H3K27me3 are as close as the active marks; the held-out gap is cell-specific CTCF, H3K36me3, H3K27me3 |
| 10 | [Second training seed](#10-second-training-seed-which-results-replicate) | Which results of 6–9 survive retraining every model with another seed? | Bulk: yes, one training moves a group mean by ~0.005–0.01. Molecules: directions replicate, sizes vary 2–10×; "10× stronger without FIRE" was mostly a weak DEV4 run |

Experiments 1–5 are bulk or paired readouts; 6–8 are the first tests from the biological-validation plan
(`TODO.md` §2); 9 rescales the bulk numbers by how reproducible each target is; 10 repeats 6–9 with a second
training seed. Decisions and next steps are at the end.

## What changed since 30 Oct

- **Fiber averaging in fold change (V1)** is the default (`Oct_23.md`, section 1), and the **cross-cell baseline**
  now also averages the other cell types' tracks in fold change, then asinh. Baseline numbers and cell-type r
  are therefore not exactly comparable with the October tables.
- **Training budget `train04`:** 15 epochs × 5000 samples (75k samples), LR patience 2 epochs. The old 100 × 1000
  run validated and restarted its data loaders 100 times and plateaued at ~60k samples; 5000-sample epochs cut that
  overhead ~5× (profiling: training is GPU-bound at ~10 samples/s on a 2g.20gb slice).
- **Checkpointing:** training saves a checkpoint after every epoch and stops cleanly before the job's time limit;
  `--resume latest` continues it. Jobs stay under 3 h (fastest partition).
- **Cluster:** runs moved to nibi (shorter queues for our group, same GPUs).
- **Per-molecule predictions (sections 7–8):** `molecule_predict.py` runs a trained model on every fiber over a
  BED of target loci and writes one row per (target, fiber): the predicted assays plus the fiber's input summaries
  (m6A rate, CpG methylation, MSP / nucleosome / FIRE fraction) over the target and its flanks, and the HiPhase
  haplotype tag. Target BEDs come from `data_prep/build_targets.py` (`data_prep/targets_spec.yaml`).
- **Biological-validation plan (7 Nov, `TODO.md` §2):** every new test checks something the model never trained on,
  has its expected direction written down before running, and has a null control.

## 1. DEV5 (V1), the all-cells reference model

> **Question:** with V1 (fold-change averaging) and the longer `train04` budget, does the all-cells model beat the
> cross-cell average on the test chromosome, in dev and in held-out cell types?
>
> **Outcome:** on MSE yes, in 7 of 8 cell types. On Pearson it is ahead or level in 4 of 5 dev cell types and
> behind in Panc1 and the 3 held-out ones. ATAC, H3K4me3, H3K27ac and H3K4me1 carry over to held-out cell types;
> the broad and repressive marks don't, H3K36me3 least of all.

- **Model:** mid 8-assay UNet (`model15`), trained on the 5 dev cell types (`data20`), `train04`. One 3 h job on
  nibi: 2 h 20 min, 10.1 samples/s throughout.
- **Validation** (chr22, 200 windows × 5 cell types): Pearson 0.320 after epoch 1, 0.501 after epoch 15, still
  rising slowly at the end (0.497 → 0.501 over the last two epochs). The old 100 × 1000 run plateaued at 0.49.
- **Test:** all 6,843 valid 5 kb tiles of chr21 (6,806–6,833 per cell type have fibers), for the 5 dev and the 3
  held-out cell types, at positions covered by ≥ 5 fibers. Baseline = fold-change mean of the dev cell types'
  tracks, excluding the evaluated cell's own line. Metrics as in `Oct_23.md` / `metrics/README.md`.

### Summary: mean over assays (model / baseline)

**Bold** = better of model and baseline. Held-out cell types ordered from near to far from the dev set.

| Cell type | MSE | Pearson r | Cell-type r |
|---|---|---|---|
| GM12878 | **0.225** / 0.285 | **0.44** / 0.43 | +0.28 |
| K562 | **0.168** / 0.212 | 0.50 / 0.50 | +0.33 |
| HepG2 | **0.227** / 0.253 | **0.58** / 0.53 | +0.44 |
| Panc1 | **0.207** / 0.224 | 0.40 / **0.48** | +0.22 |
| H1 | **0.271** / 0.305 | **0.40** / 0.39 | +0.29 |
| H9 *(held out)* | **0.186** / 0.198 | 0.38 / **0.46** | +0.19 |
| HCT116 *(held out)* | 0.178 / **0.172** | 0.47 / **0.55** | +0.28 |
| MCF-7 *(held out)* | **0.166** / 0.188 | 0.45 / **0.50** | +0.28 |

- **MSE:** the model beats the baseline in 7 of 8 cell types (all but HCT116), by a wider margin than in October,
  partly because the fold-change baseline has a higher MSE than the old asinh-mean one.
- **Pearson:** ahead or level in 4 of 5 dev cell types (behind in Panc1), behind in the 3 held-out ones, as before.
- **Cell-type r:** positive everywhere, +0.19 to +0.44, as high in HCT116 and MCF-7 (+0.28) as in most dev cell types.

### Summary without ATAC: mean over the other 7 assays (model / baseline)

H1 and H9 have no ATAC, so their means above are over 7 assays and the others over 8. Here every row averages the
same 7 assays.

| Cell type | MSE | Pearson r | Cell-type r |
|---|---|---|---|
| GM12878 | **0.248** / 0.311 | **0.40** / 0.39 | +0.24 |
| K562 | **0.179** / 0.228 | **0.47** / 0.46 | +0.32 |
| HepG2 | **0.251** / 0.279 | **0.55** / 0.49 | +0.41 |
| Panc1 | **0.218** / 0.243 | 0.37 / **0.44** | +0.21 |
| H1 | **0.271** / 0.305 | **0.40** / 0.39 | +0.29 |
| H9 *(held out)* | **0.186** / 0.198 | 0.38 / **0.46** | +0.19 |
| HCT116 *(held out)* | 0.192 / **0.183** | 0.43 / **0.53** | +0.26 |
| MCF-7 *(held out)* | **0.174** / 0.195 | 0.41 / **0.47** | +0.25 |

- Without ATAC (the model's strongest assay) the means drop by 0.03–0.04 in Pearson for the cell types that have
  it, so H1 and H9 are no longer penalised by the missing assay. The picture is unchanged: ahead of or level with
  the baseline on Pearson in 4 of 5 dev cell types, behind in Panc1 and the held-out ones; ahead on MSE in 7 of 8.

### Per assay: mean over cell types (model / baseline)

Dev = the 5 dev cell types, held out = the 3 held-out ones (ATAC: 4 dev, 2 held out).

| Assay | MSE, dev | MSE, held out | Pearson r, dev | Pearson r, held out | Cell-type r, dev | Cell-type r, held out |
|---|---|---|---|---|---|---|
| ATAC | **0.084** / 0.090 | **0.094** / 0.116 | **0.73** / 0.72 | **0.73** / 0.68 | +0.47 | +0.49 |
| H3K4me3 | **0.177** / 0.232 | **0.156** / 0.163 | **0.53** / 0.49 | 0.59 / 0.59 | +0.34 | +0.36 |
| H3K27ac | **0.210** / 0.290 | **0.172** / 0.226 | **0.47** / 0.37 | **0.44** / 0.38 | +0.37 | +0.32 |
| H3K4me1 | **0.223** / 0.293 | **0.145** / 0.180 | **0.50** / 0.42 | **0.54** / 0.52 | +0.38 | +0.38 |
| CTCF | **0.183** / 0.203 | 0.169 / **0.164** | 0.40 / **0.43** | 0.35 / **0.48** | +0.24 | +0.12 |
| H3K27me3 | **0.229** / 0.260 | **0.199** / 0.224 | 0.41 / **0.42** | 0.28 / **0.38** | +0.29 | +0.19 |
| H3K36me3 | 0.290 / **0.246** | 0.244 / **0.176** | 0.39 / **0.52** | 0.23 / **0.54** | +0.23 | +0.07 |
| H3K9me3 | **0.324** / 0.390 | **0.203** / 0.210 | 0.37 / **0.41** | 0.41 / **0.53** | +0.20 | +0.18 |

- **ATAC, H3K4me3, H3K27ac, H3K4me1** (the marks accessibility reflects most directly): ahead of the baseline on
  both metrics, held-out cell types included, with cell-type r +0.32 to +0.49 that does not drop on held-out cell
  types.
- **CTCF:** ahead on MSE in dev only, behind on Pearson; cell-type r halves on held-out cell types (+0.24 → +0.12).
- **H3K27me3, H3K36me3, H3K9me3** (broad / repressive): behind on Pearson, most of all H3K36me3 (0.23 vs 0.54 on
  held-out cell types, cell-type r +0.07). These are the assays for longer context and per-assay output scaling
  (`TODO.md` §3).

### Compared with the October model (V0, 100 × 1000 samples)

| Cell type | MSE (Oct → now) | Pearson r (Oct → now) | Cell-type r (Oct → now) |
|---|---|---|---|
| GM12878 | 0.228 → 0.225 | 0.43 → 0.44 | +0.26 → +0.28 |
| K562 | 0.172 → 0.168 | 0.50 → 0.50 | +0.32 → +0.33 |
| HepG2 | 0.232 → 0.227 | 0.57 → 0.58 | +0.41 → +0.44 |
| Panc1 | 0.194 → 0.207 | 0.40 → 0.40 | +0.21 → +0.22 |
| H1 | 0.261 → 0.271 | 0.40 → 0.40 | +0.28 → +0.29 |
| H9 | 0.177 → 0.186 | 0.38 → 0.38 | +0.19 → +0.19 |
| HCT116 | 0.177 → 0.178 | 0.47 → 0.47 | +0.26 → +0.28 |
| MCF-7 | 0.162 → 0.166 | 0.45 → 0.45 | +0.27 → +0.28 |

The same results with 75% of the training samples in about half the time (2 h 20 min vs 4 h 26 min). Cell-type r
is slightly higher everywhere, but the baseline it is measured against changed, so treat that as level.

### MSE per assay (model / baseline)

| Cell type | ATAC | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|---|
| GM12878 | **.062** / .098 | **.157** / .235 | **.258** / .418 | **.277** / .293 | **.177** / .288 | .321 / **.309** | **.317** / .395 | **.228** / .241 |
| K562 | **.088** / .097 | **.187** / .257 | **.131** / .207 | **.167** / .197 | **.160** / .237 | **.166** / .173 | **.231** / .296 | **.214** / .229 |
| HepG2 | **.060** / .072 | **.217** / .264 | **.168** / .227 | **.147** / .160 | **.225** / .321 | .169 / **.152** | .645 / **.616** | **.188** / .213 |
| Panc1 | .125 / **.093** | **.209** / .253 | **.229** / .251 | **.197** / .256 | **.245** / .249 | .390 / **.279** | **.153** / .292 | **.105** / .119 |
| H1 | – | **.114** / .149 | **.263** / .350 | **.356** / .392 | **.307** / .368 | .402 / **.315** | **.276** / .348 | **.181** / .211 |
| H9 | – | **.139** / .145 | **.185** / .225 | **.242** / .268 | **.141** / .180 | .254 / **.203** | **.204** / .237 | .137 / **.127** |
| HCT116 | **.078** / .099 | **.129** / .151 | **.159** / .220 | .190 / **.188** | **.142** / .175 | .266 / **.184** | .279 / **.195** | .181 / **.165** |
| MCF-7 | **.111** / .133 | .201 / **.194** | **.172** / .232 | **.164** / .217 | **.153** / .186 | .210 / **.141** | **.126** / .199 | **.188** / .199 |

### Pearson r per assay (model / baseline)

| Cell type | ATAC | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|---|
| GM12878 | **.75** / .67 | .42 / **.43** | **.57** / .33 | .32 / **.38** | **.58** / .39 | .44 / **.51** | .21 / **.36** | .26 / **.35** |
| K562 | .71 / **.74** | **.55** / .48 | **.49** / .40 | **.58** / .53 | .38 / **.39** | .65 / .65 | .31 / **.42** | .31 / **.38** |
| HepG2 | **.82** / .77 | **.42** / .41 | **.50** / .43 | **.56** / .55 | **.69** / .50 | .48 / **.63** | **.69** / .48 | **.50** / .47 |
| Panc1 | .62 / **.71** | .56 / .56 | **.47** / .43 | .27 / **.33** | .37 / **.45** | .11 / **.39** | .36 / **.42** | .42 / **.54** |
| H1 | – | **.67** / .58 | **.34** / .28 | **.31** / .30 | **.47** / .35 | .25 / **.43** | .30 / **.37** | **.49** / .42 |
| H9 | – | .58 / **.60** | **.35** / .33 | .34 / **.37** | .48 / **.49** | .16 / **.45** | .30 / **.44** | .40 / **.57** |
| HCT116 | **.74** / .71 | **.59** / .57 | **.56** / .47 | .25 / **.46** | **.55** / .52 | .16 / **.54** | .64 / **.68** | .30 / **.47** |
| MCF-7 | **.73** / .65 | **.61** / .59 | **.40** / .35 | .27 / **.32** | **.59** / .55 | .36 / **.63** | .30 / **.48** | .35 / **.41** |

### Cell-type r per assay

| Cell type | ATAC | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|---|
| GM12878 | +0.54 | +0.22 | +0.50 | +0.15 | +0.49 | +0.26 | −0.01 | +0.10 |
| K562 | +0.43 | +0.39 | +0.38 | +0.46 | +0.26 | +0.51 | +0.10 | +0.16 |
| HepG2 | +0.61 | +0.24 | +0.38 | +0.42 | +0.60 | +0.27 | +0.62 | +0.36 |
| Panc1 | +0.30 | +0.35 | +0.35 | +0.18 | +0.19 | −0.02 | +0.17 | +0.23 |
| H1 | – | +0.51 | +0.26 | +0.22 | +0.37 | +0.15 | +0.15 | +0.35 |
| H9 | – | +0.31 | +0.23 | +0.22 | +0.30 | +0.04 | +0.10 | +0.13 |
| HCT116 | +0.48 | +0.39 | +0.43 | +0.15 | +0.40 | −0.03 | +0.40 | +0.07 |
| MCF-7 | +0.51 | +0.39 | +0.29 | +0.22 | +0.44 | +0.21 | +0.04 | +0.17 |

- **Easy and carried over to held-out cell types:** ATAC, H3K4me3, H3K27ac, H3K4me1. Ahead of the baseline on
  Pearson in most cell types, ahead on MSE almost everywhere, cell-type r +0.2 to +0.6.
- **Hard:** H3K36me3 (Pearson 0.11–0.16 in Panc1, H9, HCT116 against 0.39–0.54 for the baseline; cell-type r ≈ 0
  there), and the repressive marks H3K27me3 / H3K9me3, which mostly trail the baseline on Pearson. Same pattern
  as in October: the broad marks need longer context (`TODO.md` §3).
- **CTCF** trails the baseline on Pearson everywhere except HepG2 and H1, with low cell-type r: it mostly follows
  accessibility.

**Sources:** training log `logs/2026-11-01_04_dev5_v1_train04.out`; eval logs `logs/2026-11-02_{00,02,03}_eval_dev5_v1_*`
and `results/26-11-03_*_eval_DEV5_V1_*/eval_summary.json` on nibi; tables made with `ignore/reports/nov13_tables.py`.

## 2. Panc1 and the new split (v2): Panc1 moves to held out

> **Question:** DEV5 trails the baseline on Panc1. Is the Panc1 sample the problem, and does training on it help
> the other cell types? How much better is a model trained on one cell type only?
>
> **Outcome:** the sample is doubtful (fibers match HCT116's ATAC better than Panc1's own, PANC-1 markers missing,
> half coverage) and training on it buys nothing. Panc1 moves to held out (split v2) and the 4-dev model **DEV4**
> becomes the main model. Specialists beat the multi-cell model by +0.03–0.06 Pearson.

DEV5 trails the baseline on Panc1 (Pearson 0.40 vs 0.48), so we checked the sample and ran two kinds of models
with the same setup and budget as DEV5 (`model15`, `train04`, ~2 h 15 min each on nibi):

- **Leave-Panc1-out:** the other 4 dev cell types (`data24`); evaluated like DEV5 (baseline = the training cell
  types' mean, excluding the evaluated line).
- **Specialists:** one model per dev cell type, trained on its own track only (`data29`–`data33`); evaluated on
  their own cell type, without baseline (it would exclude the only training cell line).

**Sample checks** (`data_prep/fiber_qc.py`, `data_prep/sample_identity.py`): Panc1 has about half the fiber
coverage of the other libraries (~50 fibers per 5 kb window vs 80–160), and its fibers agree better with HCT116's
ATAC (r 0.70) than with Panc1's own (0.62). Of PANC-1's markers it carries KRAS G12D but not the homozygous TP53
R273H, has no CDKN2A deletion and no chrY reads; every other sample matches its markers and sex. Its ENCODE
broad-mark tracks are old with 4–6 NOT_COMPLIANT audits each.

### Mean over assays: DEV5 / leave-Panc1-out / specialist (baseline = DEV5's)

| Cell type | MSE | Pearson r | Baseline r | Cell-type r (DEV5 / LOO) |
|---|---|---|---|---|
| GM12878 | 0.225 / 0.215 / **0.205** | 0.44 / 0.46 / **0.49** | 0.43 | +0.28 / +0.31 |
| K562 | 0.168 / 0.167 / **0.153** | 0.50 / 0.51 / **0.53** | 0.50 | +0.33 / +0.36 |
| HepG2 | 0.227 / 0.228 / **0.195** | 0.58 / 0.59 / **0.63** | 0.53 | +0.44 / +0.46 |
| H1 | 0.271 / 0.261 / **0.241** | 0.40 / 0.42 / **0.46** | 0.39 | +0.29 / +0.32 |
| Panc1 | 0.207 / 0.212 / **0.174** | 0.40 / 0.36 / 0.44 | **0.48** | +0.22 / +0.17 |
| H9 *(held out)* | 0.186 / **0.183** / – | 0.38 / 0.38 / – | **0.46** | +0.19 / +0.21 |
| HCT116 *(held out)* | 0.178 / 0.184 / – | 0.47 / 0.46 / – | **0.55** | +0.28 / +0.29 |
| MCF-7 *(held out)* | **0.166** / 0.169 / – | 0.45 / 0.45 / – | 0.50 | +0.28 / +0.30 |

In the leave-Panc1-out column, Panc1 is a cell type the model never saw. MSE baselines: GM12878 0.285, K562 0.212,
HepG2 0.253, H1 0.305, Panc1 0.224, H9 0.198, HCT116 0.172, MCF-7 0.188.

### Pearson r per assay: DEV5 / leave-Panc1-out / specialist

| Cell type | ATAC | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|---|
| GM12878 | .75/.76/.78 | .42/.45/.48 | .57/.58/.61 | .32/.31/.33 | .58/.59/.62 | .44/.47/.51 | .21/.25/.27 | .26/.25/.32 |
| K562 | .71/.72/.73 | .55/.56/.60 | .49/.50/.52 | .58/.57/.62 | .38/.39/.45 | .65/.68/.70 | .31/.34/.32 | .31/.30/.34 |
| HepG2 | .82/.81/.79 | .42/.44/.50 | .50/.52/.59 | .56/.56/.57 | .69/.70/.73 | .48/.52/.58 | .69/.64/.77 | .50/.50/.50 |
| H1 | – | .67/.68/.69 | .34/.35/.44 | .31/.32/.36 | .47/.48/.52 | .25/.26/.26 | .30/.34/.38 | .49/.51/.57 |
| Panc1 | .62/.60/.67 | .56/.53/.63 | .47/.46/.48 | .27/.16/.30 | .37/.36/.39 | .11/.11/.23 | .36/.25/.41 | .42/.41/.43 |
| H9 | – | .58/.59/– | .35/.36/– | .34/.41/– | .48/.49/– | .16/.15/– | .30/.30/– | .40/.37/– |
| HCT116 | .74/.74/– | .59/.58/– | .56/.57/– | .25/.20/– | .55/.58/– | .16/.18/– | .64/.56/– | .30/.27/– |
| MCF-7 | .73/.73/– | .61/.60/– | .41/.41/– | .27/.23/– | .59/.60/– | .36/.35/– | .30/.32/– | .35/.36/– |

- **Training on Panc1 buys nothing.** Without it the other 4 dev cell types gain slightly (Pearson +0.01–0.02,
  cell-type r +0.02–0.03) and the held-out three are unchanged (±0.01; per assay both ways, e.g. H9 H3K27me3
  +0.07, HCT116 H3K9me3 −0.08).
- **Panc1 is hard even for a model trained only on it:** the specialist reaches 0.44, still below the cross-cell
  average (0.48), the only cell type where that happens. Unseen, the model gets 0.36. This fits fibers and
  tracks that don't come from the same cells.
- **Specialists beat DEV5 on every dev cell type** (Pearson +0.03–0.06, MSE −0.015 to −0.04), most on H3K36me3,
  H3K4me1 and H3K27ac (H1); ATAC barely moves. Part of this is budget: with the same 75k samples a specialist sees
  ~5× more of its own cell type. They are the "told the cell type" reference for the unseen-pair test.

**Decision (4 Nov): split v2.** Panc1 moves to the held-out set as a hard cell type, reported separately from the
other three (we decide later whether to show it at all): 4 dev (GM12878, K562, HepG2, H1), 4 held out (H9,
HCT116, MCF-7, Panc1). The leave-Panc1-out model is the 4-dev model (**DEV4**) and replaces DEV5 as the main
model; its numbers above already use the v2 baselines. Rationale in `configs/README.md`.

**Sources:** logs `logs/2026-11-03_{01..14}_*` on nibi; results `results/26-11-03_T16-1*_eval_{LOO_Panc1,SPEC}_V1_*`;
table script `ignore/reports/nov13_compare.py`; sample checks in `DATA_FIBER/qc/`.

## 3. Mixture test on two pairs (T3 repeated on K562–H1)

> **Hypotheses:** (i) a model trained only on mixtures of two cell types' fibers separates the two cell types' own
> molecules as well as a model trained on their own tracks (October's T3 result, now on a second pair); (ii) on a
> pair it never saw, the mixture model does as well as an own-track model (T3b).
>
> **Outcome:** (i) yes on both pairs (fiber-origin AUC 0.637 vs 0.635, and 0.666 vs 0.665). (ii) yes, supervision
> type makes no difference on an unseen pair; what matters is having seen the cell types. Transfer is lopsided:
> poor toward K562–H1 (sign accuracy 0.59 vs 0.80–0.83). Library effects are small (|AUC − 0.5| 0.04–0.09 vs
> 0.17–0.25 for real pairs).

October's mixture test (T3) used one pair. With split v2 there are two pairs of dev cell types, **A = GM12878 +
HepG2** and **B = K562 + H1**. Four models were trained on them, all with the same `model15` / `train04` setup
as DEV4 (~2 h 15 min each on nibi):

- **Mix A / Mix B** (`data27` / `data35`): trained only on synthetic mixtures of the pair's fibers. The cell
  fraction q is drawn from [0.2, 0.8], and the target is the q-weighted fold-change mixture of the two tracks.
  These models never see a cell type's own track. H1 has no ATAC, so Mix B never trains ATAC (a mixture's assay
  mask is the product over its cells).
- **Pair A / Pair B** (`data28` / `data34`): the same cells, but each fiber is supervised by its own cell type's
  track. This is the ceiling for that pair.
- **DEV4** (`data24`): all 4 dev cell types, own tracks.

Every model is tested on both pairs: its own pair, and the pair it never saw. The readouts come from the paired
evaluation on chr21, at the pair's differential bins. These are 1 kb bins where the two tracks differ by
|asinh Δ| ≥ 0.5 (`loci_spec.yaml` pairs):

- **Fiber-origin AUC:** in each tile, how well a fiber's prediction tells which cell type it came from, oriented by
  the sign of the tracks' difference (0.5 = chance).
- **Sign accuracy:** the fraction of differential bins where the predicted difference between the two cell types
  has the same sign as the tracks' difference.
- **Differential r:** bp-level correlation between the predicted and the true difference (p_A − p_B vs t_A − t_B).

All numbers below and in the figures come from one script, `plots_code/26_nov_13/mix_two_pairs.py`. It also
writes every value per assay to `plots/26_nov_13/mix/mix_results.csv`.

### Overview

![Mean over assays, both pairs](../plots/26_nov_13/mix/mix_summary.png)

| Model | GM12878–HepG2 (8 assays) | K562–H1 (7 assays, no ATAC) |
|---|---|---|
| Mix A (GM12878 + HepG2 mixtures) | **0.637** / 0.761 / 0.329 *(own pair)* | 0.599 / 0.590 / 0.238 *(unseen)* |
| Pair A (GM12878, HepG2 own tracks) | 0.635 / **0.771** / **0.351** *(own pair)* | 0.600 / 0.598 / 0.224 *(unseen)* |
| Mix B (K562 + H1 mixtures) | 0.621 / 0.706 / 0.263 *(unseen)* | **0.666** / 0.802 / 0.291 *(own pair)* |
| Pair B (K562, H1 own tracks) | 0.617 / 0.719 / 0.270 *(unseen)* | 0.665 / **0.828** / **0.335** *(own pair)* |
| DEV4 (all 4 own tracks) | 0.629 / 0.748 / 0.316 | 0.654 / 0.793 / 0.294 |

Each cell gives the fiber-origin AUC / sign accuracy / differential r, as means over the pair's assays.

### Per assay

![Fiber-origin AUC per assay](../plots/26_nov_13/mix/mix_per_assay_auc.png)

| Model | ATAC | H3K4me3 | H3K27ac | CTCF | H3K4me1 | H3K36me3 | H3K27me3 | H3K9me3 |
|---|---|---|---|---|---|---|---|---|
| **GM12878–HepG2** | | | | | | | | |
| Mix A | .713 | .687 | .688 | .643 | .685 | .550 | .644 | .483 |
| Pair A | .705 | .677 | .676 | .642 | .683 | .542 | .653 | .499 |
| Mix B | .705 | .737 | .673 | .659 | .671 | .485 | .575 | .462 |
| Pair B | .710 | .735 | .660 | .623 | .650 | .514 | .575 | .468 |
| DEV4 | .713 | .707 | .674 | .652 | .674 | .525 | .623 | .464 |
| *bins (% higher in GM12878)* | 775 (47%) | 355 (36%) | 702 (60%) | 388 (32%) | 1472 (33%) | 592 (72%) | 605 (40%) | 2952 (13%) |
| **K562–H1** | | | | | | | | |
| Mix A | – | .670 | .564 | .458 | .724 | .476 | .796 | .505 |
| Pair A | – | .664 | .587 | .468 | .724 | .468 | .788 | .496 |
| Mix B | – | .742 | .653 | .469 | .711 | .687 | .786 | .615 |
| Pair B | – | .727 | .678 | .472 | .677 | .687 | .771 | .643 |
| DEV4 | – | .698 | .641 | .479 | .715 | .638 | .790 | .614 |
| *bins (% higher in K562)* | – | 408 (43%) | 509 (58%) | 428 (82%) | 1118 (29%) | 670 (79%) | 925 (78%) | 839 (14%) |

Differential r per assay is in [this figure](../plots/26_nov_13/mix/mix_per_assay_diff_r.png) and in the CSV. It
ranks the models the same way the AUC does.

**Predicted vs true difference per bin.** In each scatter, the top row is the mixture model trained on that pair
and the bottom row is the one trained on the other pair:

![K562–H1 bin scatter](../plots/26_nov_13/mix/mix_bin_scatter_K562_H1.png)

![GM12878–HepG2 bin scatter](../plots/26_nov_13/mix/mix_bin_scatter_GM12878_HepG2.png)

### Library control

The library control compares two Fiber-seq libraries of the **same** cell type (100U vs 200U), scored at the real
pairs' bins. Any separation there comes from the library, not the cell type.

![Library control](../plots/26_nov_13/mix/mix_library_control.png)

Mean |AUC − 0.5| over tiles and assays:

| Model | HepG2 100U vs 200U | K562 100U vs 200U | H1 100U vs 200U | GM12878–HepG2 | K562–H1 |
|---|---|---|---|---|---|
| Mix A | .049 | .089 | .044 | .182 | .245 |
| Pair A | .049 | .090 | .045 | .182 | .246 |
| Mix B | .051 | .075 | .042 | .192 | .232 |
| Pair B | .053 | .083 | .045 | .172 | .221 |
| DEV4 | .051 | .076 | .041 | .184 | .221 |

### Own-track recovery and validation

Recovery r is the Pearson r of each cell type's fibers against its own track, from the same paired evals, averaged
over the pair's assays (so no ATAC on K562–H1). Validation r is taken at the last epoch. Mixture composite r is the
final mini eval on mixtures (`eval26` / `eval30`).

| Model | Recovery r, GM12878–HepG2 | Recovery r, K562–H1 | Val r | Mixture composite r |
|---|---|---|---|---|
| Mix A | GM12878 .466, HepG2 .615 *(own)* | K562 .446, H1 .369 *(unseen)* | .612 | .608 |
| Pair A | GM12878 .469, HepG2 .610 *(own)* | K562 .438, H1 .370 *(unseen)* | .566 | – |
| Mix B | GM12878 .392, HepG2 .513 *(unseen)* | K562 .478, H1 .441 *(own)* | .559 | .518 |
| Pair B | GM12878 .398, HepG2 .554 *(unseen)* | K562 .493, H1 .438 *(own)* | .530 | – |
| DEV4 | GM12878 .458, HepG2 .586 | K562 .478, H1 .419 | – | – |

Mixture models validate on mixtures and pair models on own tracks, so their validation r values don't compare
directly.

### Findings

- **Mixture-only supervision reaches the ceiling on both pairs.**
  - On its own pair, each mixture model's fiber-origin AUC matches the own-track model's: Mix A 0.637 vs Pair A
    0.635, and Mix B 0.666 vs Pair B 0.665.
  - Sign accuracy and differential r are slightly lower (−0.01 to −0.04).
  - Recovery of each cell type's own track is the same within 0.015.
  - This repeats October's T3 result on a second, harder pair.
- **On an unseen pair, the type of supervision makes no difference.** Mix A ≈ Pair A on K562–H1, and Mix B ≈ Pair B
  on GM12878–HepG2, on every readout. The loss comes from never having seen the cell types.
- **The differential readouts transfer lopsidedly.**
  - K562 + H1 → GM12878–HepG2 loses little: AUC 0.62 vs 0.64 for models that saw the pair.
  - GM12878 + HepG2 → K562–H1 loses a lot: AUC 0.60 vs 0.67, and sign accuracy 0.59, close to chance.
  - The K562–H1 loss is in H3K36me3 (0.47 vs 0.69), H3K9me3 (0.50 vs 0.62–0.64) and H3K27ac (0.56–0.59 vs
    0.65–0.68).
  - H3K27me3 and H3K4me1 transfer fully (~0.79, ~0.72).
  - The scatter shows the H3K36me3 failure is a shift, not noise. 79% of those bins are higher in K562, yet the
    models trained on GM12878 + HepG2 predict H1 higher at most of them.
  - Recovery of each cell type's own track drops in both directions (−0.05 to −0.10), so the lopsidedness concerns
    the *difference* between the two cell types, not the per-cell tracks.
- **DEV4 sits just below each pair's own models.** Its AUC is 0.629 vs 0.635–0.637 on GM12878–HepG2, and 0.654 vs
  0.665–0.666 on K562–H1. This matches the specialists beating the multi-cell model (Panc1 section).
- **Some assays fail for every model, including those trained on the pair:**
  - **CTCF on K562–H1** (AUC 0.46–0.48, sign accuracy ~0.4). This has two causes: H1's weak CTCF track, plus a
    constant shift toward H1 in every model. See the next subsection.
  - **H3K9me3 on GM12878–HepG2** (AUC 0.46–0.50). 87% of its 2,952 bins are higher in HepG2, and all models
    predict almost no difference for either group (flat clouds in the scatter). HepG2's H3K9me3 track is the
    outlier (MSE 0.645 in the DEV5 table, the highest of any cell type and assay), so this looks like a
    track-level difference the fibers don't show.
  - **H3K36me3 on GM12878–HepG2** (0.49–0.55).
- **ATAC fiber-origin AUC is a weak readout.** Mix B never trained an ATAC output, yet reaches an AUC of 0.705 on
  GM12878–HepG2, as high as the models that did. Accessibility is in the input, so the two cell types separate on
  ATAC almost for free. Differential r does show the missing training (.37 vs .57–.61).
- **Library effects are small next to cell-type effects.** Same-cell-type library pairs give |AUC − 0.5| ≈ 0.04–0.05
  for HepG2 and H1, and 0.08–0.09 for K562, whose two libraries differ the most. The real pairs give 0.17–0.25,
  including GM12878–HepG2, which crosses Fiber-seq experiments.

## 4. CTCF on K562–H1: a weak H1 track plus a model shift

> **Hypotheses:** every model fails CTCF on K562–H1 because (a) H1's CTCF ChIP track is weakly enriched, or (b) H1's
> fibers are more accessible at those sites.
>
> **Outcome:** (a) holds, (b) doesn't. On top of that, every model adds a constant shift toward H1 at equal
> accessibility. The model trained without CpG (section 6) keeps the shift, so it does not come from the CpG channel.

82% of the 428 CTCF differential bins are higher in K562, yet every model predicts H1 ≥ K562 at most of them. Two
explanations were tested:

- **(a) Track scale.** H1's CTCF ChIP is less enriched everywhere, so most "differential" bins are shared CTCF sites
  at a different scale, which fibers cannot show.
- **(b) Accessibility.** The models' CTCF output follows fiber accessibility, and H1's fibers are more accessible
  at these sites.

![CTCF offset](../plots/26_nov_13/ctcf/ctcf_offset.png)

**(a) holds: H1's CTCF track is weak, not K562's strong.**
- At the 219 chr21 sites with CTCF signal in all 4 dev lines, H1 sits below the 4-line mean at 93% of them (median
  −0.51 asinh). GM12878, K562 and HepG2 sit within ~0.2 of each other (panel a).
- On chr21, H1's 99.9th percentile is 2.40, against 3.1–3.2 for GM12878, K562 and HepG2, and 3.19 for H9. So this is
  not stem-cell biology: H9 is the other ESC line.
- The H1 track comes from an older ENCODE experiment (ENCSR000BNH, no NOT_COMPLIANT audits). MCF-7's
  (ENCSR000AHD) is just as weak (2.31, panel a), which matters for MCF-7's held-out CTCF numbers.
- At the K562-higher bins, H1 is the lowest of the 4 dev lines at 65% of them, while K562 is the highest at only
  57%. Only 41% of them have an H1 peak at all.

**(b) does not hold: K562's fibers are the more accessible ones.**
- At the K562-higher bins, the fraction of the bin inside an MSP is 0.43 for K562 and 0.35 for H1. H1 is more
  accessible at only 8% of them.
- At shared sites the gap is +0.05.

**The models get the accessibility-driven part right but add a constant shift toward H1.**

| | Fit: K562 − H1 vs accessibility difference | Mean K562 − H1 |
|---|---|---|
| Tracks | +0.15 + 3.4 · Δaccess | +0.35 |
| Pair B (trained on K562 + H1) | −0.27 + 1.9 · Δaccess | −0.15 |
| Mix B (K562 + H1 mixtures) | −0.34 + 2.7 · Δaccess | −0.18 |
| Pair A / Mix A (never saw them) | −0.39 / −0.41 + 3.5 / 3.2 · Δaccess | −0.18 / −0.22 |
| DEV4 | −0.31 + 2.6 · Δaccess | −0.15 |

- The predicted difference correlates with the accessibility difference (r 0.60–0.72), with roughly the true slope.
  Almost the whole error is the intercept: at equal accessibility, every model predicts 0.4–0.55 asinh more CTCF
  for H1. That holds even for Pair B, which was trained on H1's weak track.
- It is not a global "H1 looks more open" effect: H1's library has a lower MSP fraction (0.27 vs 0.34) and a lower
  m6A rate (0.092 vs 0.117) than K562's (`fiber_qc.py`). The big global difference is CpG methylation, 0.27 in H1
  against 0.085 in K562, so the model could have read that as a cell-level shift.
- **Tested since: it is not the CpG channel.** The model trained without CpG (section 6) keeps the shift (mean
  K562 − H1 −0.19 vs DEV4 −0.16, true +0.32).

**Consequences:**
- K562–H1 CTCF bins are a poor test of CTCF: most of them measure H1's ChIP enrichment, not differential binding.
  The CTCF row of the K562–H1 tables should be read with that in mind.
- The weak H1 and MCF-7 CTCF tracks also lower those cell types' bulk CTCF numbers.

**Sources:** `plots_code/26_nov_13/ctcf_offset.py`. Fiber part: job `2026-11-07_00_ctcf_offset` (CPU, 2 min).
Outputs are in `plots/26_nov_13/ctcf/`: per-bin tables and the figure.

## 5. The same breakdown for every assay on both pairs

> **Question:** repeating the CTCF checks for every assay and both pairs, which failures are track-scale problems
> that fibers cannot show, which are compression, and which are cell-level offsets?
>
> **Outcome:** three failure types. Track-scale outliers (H1 CTCF, HepG2 and H1 H3K9me3) fail for every model;
> unseen cell types get compressed differences; some marks sit above what a line's accessibility implies (H3K27ac,
> H3K36me3), which training on the pair partly recovers.

The CTCF checks above were repeated for every assay on both pairs, on up to 200 differential bins per assay:
- each line's track scale at shared sites;
- the fibers' accessibility difference;
- each model's mean predicted difference and its slope against accessibility.

The table below covers GM12878–HepG2, where Mix A and Pair A saw the pair and Mix B and Pair B did not.

![Mean predicted vs true difference per assay](../plots/26_nov_13/offsets/pair_offsets.png)

![GM12878–HepG2: predicted difference vs accessibility difference](../plots/26_nov_13/offsets/access_scatter_GM12878__HepG2.png)

| Assay | Shared-site scale GM12878 / HepG2 | True mean | Predicted mean, Mix A / Pair A *(own)* | Mix B / Pair B *(unseen)* | Slope pred/true, Mix A / Mix B |
|---|---|---|---|---|---|
| ATAC | +0.02 / −0.01 | −0.02 | −0.11 / −0.15 | −0.11 / −0.16 | 0.77 / 0.34 |
| H3K4me3 | −0.78 / −0.34 | −0.34 | −0.26 / −0.22 | −0.45 / −0.34 | 0.73 / 1.01 |
| H3K27ac | +0.44 / +0.28 | **+0.31** | −0.06 / −0.05 | −0.13 / −0.11 | 0.66 / 0.28 |
| CTCF | +0.19 / +0.20 | −0.27 | −0.21 / −0.22 | −0.19 / −0.14 | 0.56 / 0.48 |
| H3K4me1 | +0.04 / +0.27 | −0.41 | −0.28 / −0.25 | −0.22 / −0.19 | 0.75 / 0.44 |
| H3K36me3 | −0.04 / −0.15 | **+0.38** | −0.10 / −0.14 | −0.25 / −0.14 | 0.51 / 0.03 |
| H3K27me3 | −0.30 / −0.36 | −0.23 | −0.04 / −0.14 | +0.01 / +0.02 | −0.07 / −0.33 |
| H3K9me3 | −0.35 / **+0.56** | **−1.26** | +0.04 / +0.02 | +0.05 / +0.06 | 0.15 / 0.10 |

Means are of the GM12878 − HepG2 difference over the bins, in asinh units. Bold marks a track outlier or a
true mean that every model gets wrong.

**GM12878–HepG2:**
- **ATAC, H3K4me3, CTCF and H3K4me1 work for every model.** The predicted mean has the right sign and is near the
  true one. The unseen-pair models mainly lose slope (the predictions are compressed): for example ATAC 0.34 vs
  0.77 and H3K4me1 0.44 vs 0.75 (Mix B vs Mix A).
- **CTCF has no offset on this pair**, and both lines' CTCF tracks are normal (+0.19 / +0.20). So the K562–H1 CTCF
  failure is H1's track, not CTCF itself.
- **H3K9me3 is the same kind of track problem as H1's CTCF, in the other direction.** HepG2's H3K9me3 track is
  globally strong (+0.56 at shared sites, GM12878 −0.35). The true mean is −1.26, and every model, trained on HepG2
  or not, predicts ~0. On K562–H1 it repeats: H1's H3K9me3 track is +0.23, K562's −0.40, the true mean is −0.75, and
  the models predict ~0.
- **H3K27ac and H3K36me3 flip sign for every model, including those trained on the pair.**
  - The true difference is GM12878-higher (+0.31, +0.38); the predictions are slightly HepG2-higher.
  - HepG2's fibers are more accessible at these bins (Δ −0.04; its library is also more digested overall, MSP
    fraction 0.275 vs 0.244). The predictions follow accessibility, but GM12878's tracks sit above what its
    accessibility implies.
  - The models trained on the pair pick up part of that, with a steeper slope (0.66 / 0.51). The unseen models pick
    up almost none of it (0.28 / 0.03).

**K562–H1, for comparison:**
- The models that never saw K562 and H1 get H3K27ac and H3K36me3 wrong in sign (−0.50 and −0.34 vs +0.34 and
  +0.54).
- They overshoot H3K4me1 (−0.95 vs −0.46) and H3K27me3 (+0.85 vs +0.54).
- The models that saw them get H3K36me3 right in sign (+0.23 / +0.20) and are close on H3K4me1 and H3K27me3.
- H3K27ac stays near zero for them too (−0.14 / −0.02).

**Summary of the breakdown.** Three distinct failure types:
1. **Track-scale outliers that fibers cannot show:** H1 CTCF, HepG2 H3K9me3, H1 H3K9me3. Every model fails, seen or
   unseen.
2. **Compression on unseen cell types:** the model separates the bins but with flatter differences.
3. **Cell-level offsets:** marks where one line's track sits above what its fibers' accessibility implies.
   Training on the pair recovers part of it (H3K27ac, H3K36me3).

**Sources:** `plots_code/26_nov_13/pair_offsets.py`, run as job `2026-11-07_01_pair_offsets` (CPU, 3 min); the
crashed print step was redone with `--replot`. Outputs, including the K562–H1 scatter, are in
`plots/26_nov_13/offsets/`. The intercept difference against accessibility (the `shift` column) is not reported
here. Where predicted slopes are flatter, it mostly measures compression extrapolated to Δaccess = 0. For CTCF on
K562–H1 the slopes are close, and the intercept and mean error agree (−0.46 vs −0.51 for Mix B).

**Next (not run yet):**
- Routed specialists as the "told the cell type" reference on the unseen pair.
- ~~The CpG-channel ablation for the CTCF shift.~~ Done with the no-CpG model (section 6): not CpG.

**Sources (mixture test):**
- Training and eval logs: `logs/2026-11-04_*` on nibi. Paired evals that timed out were rerun as
  `2026-11-04_{15..18}_eval_*_rerun`.
- Results: `results/26-11-04_*_eval_{MIX_V1,PAIR_V1,LOO_Panc1_V1}_*`, i.e. `<config>_paired.npz` and
  `eval_summary.json`.
- Figures, tables and CSV: `plots_code/26_nov_13/mix_two_pairs.py`, which writes to `plots/26_nov_13/mix/`.

## 6. Input ablations (T10): which fiber inputs does the model use?

> **Question:** the model gets five per-fiber input channels. Which of them does it actually use, and for which
> assays? We retrained DEV4 four times, each time with some channels switched off, and compared all five models.

### Setup: five models, one change each

Everything is the same as DEV4 (data `data24`, architecture `model15`, budget `train04`, reference DNA as input)
except which fiber channels the model sees (`input_flags`):

| Model | m6A | CpG | MSP | Nucleosome | FIRE | What it tests | Config / training job |
|---|---|---|---|---|---|---|---|
| **DEV4** | ✓ | ✓ | ✓ | ✓ | ✓ | reference (all inputs) | `model15_uct_mid_8A` (section 2) |
| **no FIRE** | ✓ | ✓ | ✓ | ✓ | – | is FIRE needed? | `model16_uct_mid_8A_no_fire_msp` / `2026-11-07_03_train_abl_nofire` |
| **m6A + CpG** | ✓ | ✓ | – | – | – | is any fibertools call needed? | `model21_uct_mid_8A_m6a_cpg` / `2026-11-08_15_train_abl_m6acpg` |
| **no accessibility** | – | ✓ | – | – | – | what is left from CpG + DNA alone? | `model20_uct_mid_8A_cpg_only` / `2026-11-07_04_train_abl_noacc` |
| **no CpG** | ✓ | – | ✓ | ✓ | ✓ | what does CpG methylation add? | `model19_uct_mid_8A_no_cpg` / `2026-11-07_02_train_abl_nocpg` |

- **m6A and CpG** are the raw base calls on each fiber.
- **MSP, nucleosome and FIRE** are fibertools calls computed *from the m6A pattern*. FIRE also uses a classifier
  trained on GM12878 DNase and CTCF ChIP peaks, so it carries knowledge of bulk assays.
- The first four models drop inputs step by step: FIRE, then all the calls, then m6A too. No CpG drops the other
  raw channel.
- All models trained for ~2 h 10–15 min, seed 919 (section 10 repeats all five with a second seed). Each was evaluated like DEV4, on every cell type on chr21
  (bulk). This section covers bulk accuracy; the same five models run through the molecule tests in sections 7
  and 8.

### Hypotheses and results

Hypotheses H-A, H-B and H-D were fixed in `TODO.md` §2b–2d before the runs. H-E was set when the m6A + CpG run was
planned, after H-A's result was known.

| | Model | Expected | Result |
|---|---|---|---|
| H-A | no FIRE | Costs little: FIRE is derived from the m6A the model already sees | ✅ costs nothing (mean r +0.01, cell-type r +0.01). The one exception in seed 919, H3K27me3 MSE +0.04 in the dev lines, does not replicate (section 10) |
| H-E | m6A + CpG | Costs little: MSP and nucleosome calls are also derived from the m6A | ✅ costs nothing (mean r +0.01) |
| H-D | no accessibility | No direction fixed: how much is left from CpG + DNA alone | Loses ATAC, H3K4me1, H3K27ac, CTCF, H3K9me3; keeps H3K36me3, H3K27me3, most of H3K4me3 |
| H-B | no CpG | H3K4me3, H3K36me3, H3K9me3, CTCF lose more than ATAC, H3K27ac, H3K4me1 | ◐ H3K36me3 (−0.25) and H3K4me3 (−0.05) yes; CTCF and H3K9me3 no. H3K27me3 (−0.09) loses, not predicted |

### Bulk accuracy on chr21

The three metrics from section 1, each in the same layout (figure, dev table, held-out table, takeaways):
- **Pearson r:** does the prediction rise and fall with the track? Higher is better.
- **MSE:** squared error in asinh units. It also counts level and scale, which r ignores. Lower is better.
- **Cell-type r:** partial correlation of prediction and track given the cross-cell baseline. It scores only
  where this cell type differs from the average cell type. Higher is better; the baseline is 0 by definition.

In every table, *baseline* is the cross-cell average (the mean of the training cell types' tracks), the reference
the model has to beat. Dev = GM12878, K562, HepG2, H1 (ATAC has 3, H1 has none); held out = H9, HCT116, MCF-7
(ATAC has 2). In every table of this section, **bold** = the best value in the row (ties all bold).

In every figure: top left is DEV4 per cell type and assay (brighter = better), the other panels each ablation minus
DEV4 (red = worse, blue = better). Top row: fibertools calls removed (no FIRE, then all calls). Bottom row: a raw
channel removed (m6A with its calls, or CpG).

#### Pearson r

![Bulk Pearson r per assay: DEV4 and the change for each of the four ablations](../plots/26_nov_13/ablations/ablation_bulk.png)

Dev cell types. **Bold** = best in the row (highest; baseline included).

| Assay | DEV4 | no FIRE | m6A + CpG | no accessibility | no CpG | baseline | Read mainly from |
|---|---|---|---|---|---|---|---|
| ATAC | 0.764 | **0.768** | 0.764 | 0.623 | 0.749 | 0.704 | m6A |
| H3K4me3 | 0.532 | **0.537** | 0.536 | 0.488 | 0.485 | 0.443 | both |
| H3K27ac | 0.490 | 0.506 | **0.507** | 0.402 | 0.453 | 0.327 | m6A, some CpG |
| H3K4me1 | 0.538 | **0.568** | 0.566 | 0.405 | 0.513 | 0.376 | m6A |
| CTCF | 0.391 | **0.400** | 0.387 | 0.306 | 0.381 | 0.372 | m6A |
| H3K36me3 | 0.481 | 0.489 | 0.492 | 0.468 | 0.231 | **0.546** | CpG |
| H3K27me3 | 0.443 | 0.451 | **0.455** | 0.410 | 0.353 | 0.422 | CpG |
| H3K9me3 | **0.393** | 0.389 | **0.393** | 0.306 | 0.381 | 0.383 | m6A |
| *Mean* | *0.504* | ***0.514*** | *0.512* | *0.426* | *0.443* | *0.446* |  |

Held-out cell types:

| Assay | DEV4 | no FIRE | m6A + CpG | no accessibility | no CpG | baseline |
|---|---|---|---|---|---|---|
| ATAC | 0.736 | **0.744** | 0.742 | 0.538 | 0.724 | 0.648 |
| H3K4me3 | 0.589 | 0.601 | **0.609** | 0.574 | 0.556 | 0.558 |
| H3K27ac | 0.447 | 0.449 | **0.460** | 0.380 | 0.416 | 0.350 |
| H3K4me1 | 0.552 | **0.585** | 0.577 | 0.369 | 0.528 | 0.489 |
| CTCF | 0.334 | 0.348 | 0.330 | 0.195 | 0.339 | **0.452** |
| H3K36me3 | 0.228 | 0.242 | 0.246 | 0.234 | 0.177 | **0.527** |
| H3K27me3 | 0.279 | 0.309 | 0.302 | 0.290 | 0.248 | **0.357** |
| H3K9me3 | 0.396 | 0.398 | 0.389 | 0.342 | 0.363 | **0.520** |
| *Mean* | *0.445* | *0.460* | *0.457* | *0.365* | *0.419* | ***0.488*** |

Panc1 follows the same pattern (mean 0.360 / 0.382 / 0.364 / 0.313 / 0.331 in the same model order). The one
outlier is no FIRE's Panc1 H3K27me3 (0.27 vs DEV4 0.16), which shrinks to +0.03 in a second seed (section 10).

H3K36me3 per cell type, the clearest single result:

| Cell type | DEV4 | no FIRE | m6A + CpG | no accessibility | no CpG |
|---|---|---|---|---|---|
| GM12878 | .47 | **.48** | .47 | .46 | .19 |
| K562 | .68 | **.70** | **.70** | .67 | .23 |
| HepG2 | .52 | .52 | **.53** | .51 | .28 |
| H1 | **.26** | **.26** | **.26** | .23 | .23 |
| H9 *(held out)* | .15 | **.16** | **.16** | .13 | .14 |
| HCT116 *(held out)* | .18 | .19 | **.20** | .19 | **.20** |
| MCF-7 *(held out)* | .35 | **.38** | **.38** | **.38** | .19 |
| Panc1 *(held out)* | .11 | **.13** | .11 | .07 | .05 |

Without CpG, H3K36me3 collapses wherever DEV4 predicts it well. With CpG and no accessibility it is intact. In H1,
H9 and HCT116, DEV4 is already poor (r 0.15–0.26), so there is little to lose. H3K27me3 behaves the same way, less
strongly: without CpG it drops 0.05–0.12 in the dev lines (K562 .57 → .45, HepG2 .56 → .45).

#### MSE

![Bulk MSE per assay: DEV4 and the change for each of the four ablations](../plots/26_nov_13/ablations/ablation_bulk_mse.png)

Lower is better; in the Δ panels, red = higher MSE than DEV4 (worse). Dev cell types. **Bold** = best in the row
(lowest; baseline included).

| Assay | DEV4 | no FIRE | m6A + CpG | no accessibility | no CpG | baseline |
|---|---|---|---|---|---|---|
| ATAC | **0.069** | 0.072 | 0.070 | 0.101 | 0.072 | 0.101 |
| H3K4me3 | 0.164 | **0.162** | 0.167 | 0.182 | 0.181 | 0.241 |
| H3K27ac | 0.203 | **0.198** | 0.201 | 0.226 | 0.212 | 0.320 |
| H3K4me1 | 0.214 | **0.203** | 0.206 | 0.258 | 0.219 | 0.332 |
| CTCF | 0.201 | **0.200** | 0.202 | 0.215 | 0.203 | 0.249 |
| H3K36me3 | **0.244** | 0.249 | 0.246 | 0.257 | 0.293 | 0.252 |
| H3K27me3 | **0.225** | 0.261 | **0.225** | 0.235 | 0.244 | 0.282 |
| H3K9me3 | **0.377** | 0.392 | 0.383 | 0.401 | 0.385 | 0.466 |
| *Mean* | ***0.212*** | *0.217* | ***0.212*** | *0.234* | *0.226* | *0.280* |

Held-out cell types:

| Assay | DEV4 | no FIRE | m6A + CpG | no accessibility | no CpG | baseline |
|---|---|---|---|---|---|---|
| ATAC | 0.091 | **0.089** | 0.091 | 0.177 | 0.092 | 0.130 |
| H3K4me3 | 0.169 | **0.155** | 0.158 | 0.160 | 0.173 | 0.177 |
| H3K27ac | 0.173 | **0.172** | 0.173 | 0.192 | 0.176 | 0.239 |
| H3K4me1 | 0.151 | **0.149** | 0.155 | 0.177 | 0.158 | 0.199 |
| CTCF | **0.173** | 0.174 | 0.175 | 0.191 | 0.174 | 0.179 |
| H3K36me3 | 0.227 | 0.219 | 0.211 | 0.212 | 0.201 | **0.193** |
| H3K27me3 | 0.205 | **0.193** | 0.200 | 0.199 | 0.202 | 0.244 |
| H3K9me3 | **0.209** | 0.219 | 0.215 | 0.241 | 0.218 | 0.238 |
| *Mean* | *0.175* | ***0.171*** | *0.172* | *0.194* | *0.174* | *0.200* |

Panc1 means: 0.212 / 0.201 / 0.211 / 0.221 / 0.204 (same model order), baseline 0.224.

H3K27me3 per cell type, the one place where MSE and r disagree:

| Cell type | DEV4 | no FIRE | m6A + CpG | no accessibility | no CpG | baseline |
|---|---|---|---|---|---|---|
| GM12878 | **.260** | .310 | .262 | .275 | .267 | .317 |
| K562 | .159 | .209 | **.157** | .167 | .190 | .222 |
| HepG2 | .135 | .173 | **.134** | .143 | .156 | .185 |
| H1 | **.345** | .354 | .346 | .353 | .361 | .403 |
| H9 *(held out)* | .229 | .232 | **.228** | .232 | .232 | .277 |
| HCT116 *(held out)* | .205 | **.189** | .200 | .196 | .195 | .214 |
| MCF-7 *(held out)* | .182 | **.159** | .173 | .170 | .179 | .241 |
| Panc1 *(held out)* | .212 | **.199** | .209 | .211 | .231 | .256 |

- **Same picture as Pearson for most of the table.** No accessibility is worst (dev mean 0.234 vs 0.212), with the
  largest increases on H3K4me1 and ATAC (+0.045, +0.033; held-out ATAC +0.086). No CpG loses on H3K36me3 (+0.049; K562 .162 → .271).
  m6A + CpG is identical to DEV4 (0.212 / 0.212).
- **No FIRE has a higher H3K27me3 MSE in GM12878, K562 and HepG2** (+0.04 to +0.05), although its H3K27me3 r and
  cell-type r are slightly *better* there. So its H3K27me3 predictions follow the track's pattern as well as DEV4's
  but sit at the wrong level or scale in those lines. In the held-out lines it has the lowest H3K27me3 MSE. This is
  the one cost of dropping FIRE that Pearson doesn't show; it isn't in m6A + CpG, which also lacks FIRE.
  **Second seed (section 10): this is seed noise.** With seed 2 no FIRE's H3K27me3 MSE in these lines is 0.003–0.010
  *below* DEV4's.
- **Held-out H3K36me3:** every ablation has a lower MSE than DEV4, and the baseline is lowest (0.193). All models
  are poor here (r 0.18–0.25), and a flatter prediction probably loses less than a confidently wrong one. No CpG
  has the lowest MSE of the models but also the lowest r, so this is not a gain.
- **Against the baseline:** DEV4, no FIRE and m6A + CpG beat it on every assay in both groups except held-out
  H3K36me3. No CpG falls behind it on H3K36me3 in both groups. No accessibility falls behind it on H3K36me3 in both
  groups and on held-out ATAC, CTCF and H3K9me3, and ties on dev ATAC.

#### Cell-type r

![Bulk cell-type r per assay: DEV4 and the change for each of the four ablations](../plots/26_nov_13/ablations/ablation_bulk_ctr.png)

The baseline scores 0 by definition, so every positive value is cell-type-specific signal the model adds. Dev cell
types. **Bold** = best in the row (highest).

| Assay | DEV4 | no FIRE | m6A + CpG | no accessibility | no CpG |
|---|---|---|---|---|---|
| ATAC | 0.564 | **0.572** | 0.567 | 0.357 | 0.543 |
| H3K4me3 | 0.393 | **0.400** | **0.400** | 0.327 | 0.346 |
| H3K27ac | 0.419 | 0.437 | **0.438** | 0.318 | 0.379 |
| H3K4me1 | 0.456 | **0.491** | 0.487 | 0.303 | 0.427 |
| CTCF | 0.271 | **0.281** | 0.267 | 0.200 | 0.259 |
| H3K36me3 | 0.313 | 0.319 | **0.323** | 0.306 | 0.102 |
| H3K27me3 | 0.316 | **0.335** | 0.330 | 0.291 | 0.206 |
| H3K9me3 | 0.252 | 0.249 | **0.253** | 0.165 | 0.247 |
| *Mean* | *0.373* | ***0.386*** | *0.383* | *0.283* | *0.314* |

Held-out cell types:

| Assay | DEV4 | no FIRE | m6A + CpG | no accessibility | no CpG |
|---|---|---|---|---|---|
| ATAC | 0.537 | **0.556** | 0.552 | 0.248 | 0.530 |
| H3K4me3 | 0.390 | 0.403 | **0.418** | 0.358 | 0.362 |
| H3K27ac | 0.355 | 0.355 | **0.370** | 0.282 | 0.325 |
| H3K4me1 | 0.418 | **0.461** | 0.449 | 0.205 | 0.398 |
| CTCF | 0.130 | **0.144** | 0.127 | 0.008 | 0.143 |
| H3K36me3 | 0.057 | 0.066 | 0.069 | **0.073** | 0.035 |
| H3K27me3 | 0.183 | **0.243** | 0.199 | 0.216 | 0.100 |
| H3K9me3 | 0.175 | **0.180** | 0.171 | 0.131 | 0.159 |
| *Mean* | *0.281* | ***0.301*** | *0.294* | *0.190* | *0.256* |

Panc1 means: 0.166 / 0.192 / 0.168 / 0.126 / 0.141 (same model order).

- **The same split as Pearson, and sharper,** because cell-type r removes what all cell types share. Without m6A,
  ATAC's cell-specific signal drops by 0.21 (dev) and 0.29 (held out), H3K4me1's by 0.15 / 0.21, and held-out
  CTCF's to ~0 (0.008). Without CpG, H3K36me3 drops 0.21 (K562 .54 → .07, GM12878 .27 → .04) and H3K27me3 0.11.
- **H3K4me3 uses both inputs:** it loses 0.07 without m6A and 0.05 without CpG.
- **No FIRE has the highest H3K27me3 cell-type r** (+0.02 dev, +0.06 held out, Panc1 .03 → .18). Read together
  with the MSE: its H3K27me3 gets the cell-specific pattern better and the level worse in the dev lines.
  **Second seed:** the held-out gain falls to +0.00 and Panc1's to +0.03, within that assay's seed noise there
  (SD 0.03–0.04).
- **Held-out H3K36me3 is ~0 for every model** (0.04–0.07): no model predicts how a held-out line's H3K36me3 differs
  from the average, with or without CpG.

### What the five models show

1. **FIRE and the other fibertools calls add nothing to bulk accuracy.** Without FIRE, or without any call, every
   assay stays within about ±0.03 of DEV4 in Pearson r and cell-type r (H-A, H-E; the one exception is no FIRE's
   Panc1 H3K27me3, +0.11 r). MSE agrees except for no FIRE's H3K27me3 in the dev lines (+0.04): right pattern, wrong
   level. m6A + CpG doesn't show it. The model gets what it needs from the raw m6A. **Second seed (section 10):**
   both no-FIRE exceptions were seed noise, and no FIRE and m6A + CpG are slightly *better* than DEV4 on H3K27ac
   and H3K4me1 (+0.02–0.03 r in both seeds).
2. **m6A carries the accessibility-type assays.** Without it (no accessibility), ATAC, H3K4me1, H3K27ac, CTCF and
   H3K9me3 drop 0.05–0.20 in r, and their cell-type r drops the most (H-D).
3. **CpG methylation carries H3K36me3 and H3K27me3.** Without it, H3K36me3 drops 0.25 (0.46 in K562) and H3K27me3
   0.09. H3K36me3 is consistent with gene-body methylation following H3K36me3 (DNMT3B). H3K4me3 loses a little
   from either input alone, so it uses both. CTCF and H3K9me3 don't need CpG, against H-B.

In bulk, no FIRE and m6A + CpG are interchangeable. Sections 7 and 8 show where they differ: per molecule.

Seed noise: section 10 retrains all five models with a second seed. One training moves a group mean by
~0.005–0.01 (up to 0.03–0.04 for held-out H3K27me3 / H3K9me3), and points 1–3 above all replicate. Section 9 puts
these numbers on the scale of the ENCODE replicate ceiling.

**Sources:** training logs `logs/2026-11-07_0{2,3,4}_train_abl_*` and `logs/2026-11-08_15_train_abl_m6acpg`;
evals `logs/2026-11-08_*_eval_abl_*_{a,b}` (dev and held-out sets, ~21–24 min each); results
`results/*_eval_ABL_V1_*` on nibi. Tables and figures: `plots_code/26_nov_13/ablations.py` (models and display
order in `abl_common.py`), outputs in `plots/26_nov_13/ablations/` (`bulk_{r,mse,ctr}.tsv`,
`bulk_groups{,_mse,_ctr}.tsv`, `log_ablations.txt`). MSE is `per_assay_loss` and cell-type r is `partial.strata.all` in each
`eval_summary.json`.

## 7. CTCF footprints on single molecules (T5e)

> **Question:** the model was trained on bulk CTCF ChIP only; per-molecule binding was never a label. Does a
> fiber's predicted CTCF follow whether CTCF is bound **on that molecule**, read from the CTCF footprint in its m6A
> pattern?

**Targets** (`data_prep/build_targets.py`, `ctcf_sites_chr21.bed`, chr21 only):
- CTCF motif hits from JASPAR MA0139.2 (15 bp) at p ≤ 1e-5 (score ≥ 13.44 bits, exact threshold from the PWM's
  score distribution).
- **Bound:** the best hit inside each ENCODE CTCF peak of a dev line, merged across lines (330 sites). Per line, a
  site is *bound* if that line has a peak there, and *bound_other_line* if only another dev line does.
- **Unbound:** motif hits more than 500 bp from every dev line's peak, sampled to match the bound sites' motif score
  in 1-bit bins (272; the highest-score bins have too few unbound hits).
- Per line: GM12878 184 bound / 146 other line / 268 unbound; K562 252 / 78 / 265.

**Per-fiber state at the motif** (fixed before looking at results):
- *nucleosome:* ≥ 50% of the motif covered by a nucleosome call;
- *footprint:* ≥ 50% inside an MSP **and** the motif's m6A rate ≤ 0.5 × the mean rate of the 100 bp flanks
  (protected);
- *open:* inside an MSP, not protected;
- *other:* anything else (read ends, < 4 A/T positions in the motif, no flank m6A calls).

**Hypotheses (fixed in advance, in the script header):**
- **H1 (main):** at bound sites, footprint fibers get higher predicted CTCF than open fibers at the same site
  (per-site difference, Wilcoxon signed-rank over sites with ≥ 3 fibers in both states).
- **H2 (specificity):** that difference is larger for CTCF than for ATAC (both states are accessible).
- **H3:** nucleosome-covered fibers get the lowest predicted CTCF.
- **H4 (data, no model):** footprints are more frequent among accessible fibers at bound than at unbound sites.
- **H5 (null):** unbound sites get lower predicted CTCF than bound sites of the same motif score.

Caveat: the model sees m6A (so the footprint) and the reference sequence (so the motif, in both sets). The test is
whether bulk supervision taught it that a protected motif on an accessible molecule means CTCF. The
no-accessibility model sees neither m6A nor MSPs, so it is a built-in null.

### Data check (H4)

Fiber states, measured input (the same for every model):

| Sample | Site set | Sites | Fibers | Nucleosome | Open | Footprint | Other | Footprint / accessible |
|---|---|---|---|---|---|---|---|---|
| GM12878 | bound | 184 | 21,964 | .36 | .29 | .22 | .12 | **.43** |
| | bound, other line | 146 | 18,033 | .67 | .24 | .04 | .05 | .13 |
| | unbound | 268 | 30,482 | .66 | .28 | .04 | .02 | .12 |
| K562 | bound | 252 | 22,742 | .34 | .31 | .21 | .14 | **.41** |
| | bound, other line | 78 | 6,878 | .54 | .36 | .05 | .06 | .12 |
| | unbound | 265 | 21,892 | .52 | .39 | .06 | .03 | .13 |

**H4 holds:** 41–43% of accessible fibers carry the footprint at bound sites, against 12–13% at unbound and
other-line sites (the background rate of our simple call). The literature value is ~30% in K562 (Stergachis 2020),
with a different footprint caller.

### Model results

![CTCF footprints, DEV4](../plots/26_nov_13/molecules/ctcf_footprints_DEV4.png)

![CTCF footprints, no FIRE](../plots/26_nov_13/molecules/ctcf_footprints_noFIRE.png)

![CTCF footprints, m6A + CpG only](../plots/26_nov_13/molecules/ctcf_footprints_m6aCpG.png)

Each figure, per sample: fiber states by site set; per-fiber predicted CTCF by state at bound sites; the per-site
footprint − open difference for CTCF and ATAC; site mean predicted CTCF against the line's CTCF peak signal. The
no-CpG and no-accessibility figures are `ctcf_footprints_noCpG.png` / `_noAcc.png` in the same folder. The fiber
states are computed from the measured MSP / nucleosome calls for every model, including the two that don't see
them as input.

At bound sites (per-site medians; n = 135 GM12878 / 184 K562 sites with both states):

| Model | Sample | Footprint − open, CTCF (fraction of sites > 0, p) | Footprint − open, ATAC | Footprint − open, H3K4me3 | CTCF − ATAC (p) | Open − nucleosome, CTCF |
|---|---|---|---|---|---|---|
| DEV4 | GM12878 | +0.11 (0.64, 7e-5) | +0.09 | −0.10 | +0.02 (0.05) | +1.01 |
| | K562 | +0.04 (0.56, 0.05) | +0.02 | −0.09 | +0.05 (9e-8) | +0.71 |
| No CpG | GM12878 | +0.18 (0.70, 9e-8) | +0.27 | +0.09 | −0.12 (2e-13) | +1.00 |
| | K562 | +0.10 (0.59, 0.003) | +0.11 | +0.02 | −0.02 (0.007) | +0.64 |
| **No FIRE** | GM12878 | **+1.14 (0.94, 6e-23)** | +0.41 | −0.08 | **+0.67 (7e-24)** | +1.11 |
| | K562 | **+0.84 (0.83, 3e-25)** | +0.30 | −0.07 | **+0.54 (2e-31)** | +0.83 |
| No accessibility | GM12878 | 0.00 (0.52, 0.39) | 0.00 | +0.01 | 0.00 (0.90) | −0.01 |
| | K562 | 0.00 (0.55, 0.04) | +0.01 | 0.00 | 0.00 (0.08) | 0.00 |
| m6A + CpG | GM12878 | +0.26 (0.81, 2e-14) | +0.06 | +0.03 | +0.20 (1e-21) | +1.15 |
| | K562 | +0.20 (0.64, 2e-7) | +0.04 | +0.04 | +0.14 (6e-26) | +0.88 |

Unbound and other-line sites (per-site medians):

| Model | Sample | Unbound: footprint − open, CTCF (p) | Unbound: CTCF − ATAC | Other line: footprint − open, CTCF (n sites, p) |
|---|---|---|---|---|
| DEV4 | GM12878 | −0.02 (0.62) | +0.04 | +0.04 (79, 0.002) |
| | K562 | −0.01 (0.95) | +0.04 | +0.21 (43, 0.001) |
| No CpG | GM12878 | −0.07 (0.45) | +0.05 | +0.04 (79, 0.009) |
| | K562 | −0.06 (0.24) | +0.06 | +0.15 (43, 0.01) |
| No FIRE | GM12878 | −0.05 (0.84) | +0.20 | +0.05 (79, 9e-4) |
| | K562 | −0.04 (0.60) | +0.27 | +0.42 (43, 8e-5) |
| No accessibility | both | ≈ 0 | ≈ 0 | ≈ 0 |
| m6A + CpG | GM12878 | −0.04 (0.07) | +0.14 | +0.04 (79, 0.02) |
| | K562 | −0.03 (0.11) | +0.11 | +0.18 (43, 0.004) |

Site level (H5): mean predicted CTCF at bound / unbound sites is 1.25 / 0.58 (GM12878) and 1.09 / 0.60 (K562) for
DEV4, 1.63 / 0.61 and 1.45 / 0.64 without FIRE, 1.44 / 0.62 and 1.28 / 0.66 for m6A + CpG, and 0.71 / 0.57 and
0.66 / 0.58 without accessibility. Within 1-bit motif-score bins, bound sites score higher in every bin except the
20-bit bin, which has only 5 unbound sites (`ctcf_bound_vs_unbound_by_score_<model>.tsv`). Across bound sites, the
site mean follows the line's CTCF peak strength (Spearman 0.81 / 0.80 for DEV4, 0.81 / 0.85 without FIRE, 0.81 /
0.84 for m6A + CpG).

### Outcomes

> **Second seed (section 10):** the tables above are seed 919. Retrained with seed 2, every model that sees m6A
> responds strongly (DEV4 +0.73 / +0.44, m6A + CpG +0.56 / +0.58, no CpG +0.75 / +0.46, no FIRE +0.95 / +0.68),
> unbound sites stay ≈ 0 and the no-accessibility null stays 0. So H1, H3 and H5 replicate, but **the size of the
> response varies 2–10× between trainings, and the comparisons between models below hold only for no FIRE**: it is
> the strongest and the most CTCF-specific model in both seeds. Statements about DEV4, m6A + CpG and no CpG being
> weak are single-seed.

- **H1 holds for every model that sees m6A**, but weakly in DEV4 (+0.11 GM12878, +0.04 K562, p = 0.05). **Without
  FIRE the effect is ~10× larger** (+1.14 / +0.84; 94% / 83% of sites in the expected direction). *Seed 2: DEV4
  +0.73 / +0.44 (p ≤ 6e-20), no FIRE +0.95 / +0.68, so 1.3–1.5×, not 10×.*
- **H2:** weak in DEV4 (CTCF − ATAC +0.02 / +0.05). **Fails without CpG** (−0.12 / −0.02: there, predicted ATAC
  moves more than CTCF, so its footprint response is accessibility-like). **Strong without FIRE** (+0.67 / +0.54).
  H3K4me3 goes slightly *down* on footprint fibers in every model with m6A except no CpG, so the response is
  CTCF-specific, not a general "more signal". *Seed 2: H2 holds for every model with m6A, no CpG included
  (+0.33 / +0.31); no FIRE stays the most specific (+0.85 / +0.63, against DEV4 +0.23 / +0.21).*
- **Caveat on H2:** CTCF − ATAC is also positive at unbound sites (+0.20 / +0.27 without FIRE). There, predicted
  ATAC drops on protected-looking fibers while predicted CTCF doesn't move. At bound sites the no-FIRE CTCF
  increase itself (+1.14 / +0.84) has no counterpart at unbound sites (−0.05 / −0.04), so the main result stands,
  but CTCF − ATAC overstates specificity by about 0.2.
- **H3 holds:** nucleosome-covered fibers are lowest in every model with accessibility (open − nucleosome +0.64 to
  +1.11).
- **H5 holds** for every model with accessibility, including at matched motif scores. The no-accessibility model
  separates bound and unbound sites only slightly (~0.1), from CpG and sequence context.
- **The no-accessibility null is clean:** every per-fiber difference is 0.00. The test does not pick up something
  other than m6A.
- **Other-line sites** (a peak in another dev line only) show a smaller effect, largest in K562 without FIRE (+0.42,
  43 sites): some of these sites may be weakly bound in K562 below the peak caller's threshold.
- **Interpretation (not tested):** with the FIRE channel, the model seems to take "this is a CTCF-type element" from
  FIRE, at the site level, and has little reason to read the m6A pattern inside the MSP. Without FIRE it has to
  read the m6A pattern, and it learns that a protected motif on an accessible molecule means CTCF, from bulk
  supervision alone. Together with section 6 (no bulk cost), this makes the no-FIRE model the better main model.
  *Seed 2 weakens this account: DEV4 can also learn a strong response with FIRE in the input, though a less
  specific one. What replicates is that without FIRE the response is reliably strong and specific.*
- **m6A + CpG (no fibertools calls) sits in between:** footprint − open CTCF +0.26 / +0.20 (81% / 64% of sites),
  about twice DEV4 but a quarter of no FIRE. It is still absent at unbound sites (−0.04 / −0.03), and nucleosome
  fibers are still lowest (+1.15 / +0.88), so H1, H3 and H5 hold. Specificity is weaker: CTCF − ATAC is +0.20 /
  +0.14 at bound sites but +0.14 / +0.11 at unbound ones, so most of it is the generic drop in predicted ATAC on
  protected-looking fibers, and H3K4me3 rises slightly on footprint fibers (+0.03 / +0.04; also +0.05 at unbound
  GM12878 sites) instead of falling as in DEV4 and no FIRE.
- **Interpretation (not tested):** the strongest per-molecule CTCF read needs the MSP / nucleosome calls *without*
  FIRE. The calls give the model the accessible patch, against which a 15 bp protected motif stands out; without
  them it has to find the patch from the m6A density itself, and a short gap inside an MSP is harder to tell apart
  from linker DNA. With FIRE on top, the model takes the site-level answer from FIRE. So the fibertools calls cost
  nothing to drop in bulk (section 6) but help the molecule-level read, and FIRE is the one call to drop.
  *Seed 2: m6A + CpG reaches 60–85% of no FIRE (+0.56 / +0.58, CTCF − ATAC +0.55 / +0.50), so the evidence that the
  MSP / nucleosome calls help is now weak: no FIRE ≥ m6A + CpG in both seeds, but by 1.2–4.4×.*

**Sources:** targets `data_prep/build_targets.py` + `targets_spec.yaml` → `DATA_FIBER/regions/targets/`
(`manifest.json`); per-fiber predictions `molecule_predict.py`, jobs `logs/2026-11-07_05_molecules_ctcf_chr21`,
`logs/2026-11-08_{09,11,13,19}_molecules_ctcf_abl_*` (~4.5 min each); analysis
`plots_code/26_nov_13/ctcf_footprints.py --model <m>` → `plots/26_nov_13/molecules/ctcf_*_<m>.{tsv,png}`.

## 8. X inactivation per haplotype in GM12878 (T5b)

> **Question:** the models were trained on bulk tracks, in which both X copies are mixed, and never saw haplotypes
> or chrX (training is chr20 only). Do their per-fiber predictions differ between the active (Xa) and the inactive
> X (Xi) the way X-inactivation biology says?

**Setup:**
- **Targets** (`tss_chrX_chr21.bed`): TSS ± 500 bp of protein-coding genes plus XIST, on chrX and chr21 (the test
  chromosome, as autosomal null). Classes: *subject* (X-inactivated, 825), *escape* (14), *PAR* (20, hg38
  pseudoautosomal coordinates), *xist* (1), *autosomal* (chr21, 220). CpG-island promoters flagged from the UCSC
  CpG-island track.
- **The escape list is provisional** (15 well-known escape genes from memory; JPX dropped as non-coding). It still
  needs checking against Tukiainen et al. 2017.
- **Haplotypes:** GM12878 is trio-phased (one phase block per chromosome), so HiPhase HP1 / HP2 are whole-chromosome
  haplotypes. **Xi = the haplotype whose XIST promoter is CpG-unmethylated** (XIST is expressed from the Xi). This is
  measured input, not a model output: XIST promoter CpG methylation 0.355 on HP1 (58 fibers) vs 0.085 on HP2 (49
  fibers), so **Xi = HP2**.
- **Per gene:** d = mean over Xa fibers − mean over Xi fibers, for genes with ≥ 5 fibers on each haplotype. 113,151
  phased fiber × TSS rows over 1,073 TSSs; genes kept: subject 787 (425 with a CGI promoter), escape + PAR 33,
  autosomal 211. On chr21, "Xa" / "Xi" are just HP1 / HP2, an arbitrary pairing, so d should be ~0.

**Hypotheses (fixed in advance, in the script header).** Xi promoters are silent, CGI-methylated, depleted of
active marks and enriched for H3K27me3 / H3K9me3; escape and PAR genes are expressed from both copies.
- **H1:** subject genes have d > 0 for ATAC, H3K4me3, H3K27ac and d < 0 for H3K27me3, H3K9me3; strongest at CGI
  promoters.
- **H2:** escape + PAR genes have smaller |d| than subject genes.
- **H3 (null):** chr21 genes have d ~ 0.
- **H4:** XIST is the reverse of H1.

Caveat: the model sees the fibers' CpG methylation and accessibility, which already differ between Xa and Xi. The
test shows that its predictions follow X-inactivation biology on unseen chromosomes and haplotypes; the ablations
show which input carries it.

### Data check (inputs)

Median Xa − Xi per gene, measured input:

| Group | Genes | CpG methylation | MSP fraction |
|---|---|---|---|
| Subject | 787 | −0.063 | +0.040 |
| Subject, CGI promoter | 425 | **−0.234** | **+0.082** |
| Escape + PAR | 33 | 0.000 | −0.002 |
| Autosomal (chr21) | 211 | −0.004 | −0.001 |

The inputs behave as expected: the Xi's CGI promoters are methylated and less accessible, escape / PAR and chr21
genes show no difference. The escape inputs also support the provisional escape list in aggregate.

### Model results

![X inactivation per gene, DEV4](../plots/26_nov_13/molecules/xi_promoters_DEV4.png)

Each panel is one predicted assay or input: d per gene by group (box + points), red line at 0. The other models'
figures are `xi_promoters_{noCpG,noFIRE,noAcc,m6aCpG}.png` in the same folder.

Fraction of genes with the expected sign, subject genes / subject genes with a CGI promoter:

| Assay (expected on Xa) | DEV4 | No CpG | No FIRE | No accessibility | m6A + CpG |
|---|---|---|---|---|---|
| ATAC (higher) | .86 / .91 | .82 / .88 | .83 / .90 | .79 / .88 | .84 / .89 |
| H3K4me3 (higher) | .73 / .84 | .75 / .85 | .79 / .87 | .85 / .88 | .81 / .88 |
| H3K27ac (higher) | .82 / .84 | .82 / .89 | .87 / .90 | .87 / .90 | .87 / .91 |
| H3K9me3 (lower) | .73 / .76 | .80 / .86 | .72 / .71 | **.33 / .21** | .68 / .64 |
| H3K27me3 (lower) | **.43 / .26** | **.28 / .22** | .56 / .46 | **.46 / .29** | **.48 / .35** |

Median d at subject CGI promoters:

| Model | ATAC | H3K4me3 | H3K27ac | H3K27me3 | H3K9me3 |
|---|---|---|---|---|---|
| DEV4 | +0.46 | +0.60 | +0.44 | +0.09 | −0.04 |
| No CpG | +0.41 | +0.47 | +0.47 | +0.04 | −0.08 |
| No FIRE | +0.53 | +0.62 | +0.53 | +0.01 | −0.02 |
| No accessibility | +0.58 | +0.82 | +0.60 | +0.11 | +0.02 |
| m6A + CpG | +0.51 | +0.71 | +0.52 | +0.07 | −0.02 |

DEV4 Wilcoxon p over subject genes: 1e-100 (ATAC) to 1e-12 (H3K27me3); every assay is significant, including
H3K27me3 in the wrong direction.

**Nulls:** over all five models, escape + PAR and chr21 medians are within ±0.06 for every assay, and the fraction
of genes with the expected sign is 0.42–0.61. For DEV4 every null row has Wilcoxon p ≥ 0.06.

**H2 (|d| subject > escape + PAR, Mann–Whitney, DEV4):** significant for ATAC (p 0.016), H3K27me3, H3K4me1,
H3K36me3, H3K9me3 and CTCF (p ≤ 0.006), not for H3K4me3 (0.25) or H3K27ac (0.27). Same pattern for no CpG, no
FIRE and m6A + CpG (H3K4me3 0.36, H3K27ac 0.20); all significant without accessibility. The comparison is diluted: many subject genes are probably not
expressed in GM12878, so they are silent on both copies and have d ≈ 0 (the cloud around 0 in the figure).

**XIST (Xa − Xi):**

| | ATAC | H3K4me3 | H3K27ac | H3K9me3 | CpG methylation (input) | MSP fraction (input) |
|---|---|---|---|---|---|---|
| DEV4 | −0.66 | −1.21 | −1.31 | +0.17 | +0.27 | −0.14 |
| No CpG | −0.60 | −1.18 | −1.16 | +0.11 | | |
| No FIRE | −0.69 | −1.31 | −1.45 | +0.12 | | |
| No accessibility | −0.26 | −0.26 | −0.24 | −0.01 | | |
| m6A + CpG | −0.60 | −1.35 | −1.45 | +0.09 | | |

### Outcomes

> **Second seed (section 10):** ATAC, H3K4me3, H3K27ac, the nulls, XIST and the H3K27me3 failure all replicate in
> every model. H3K9me3 does not, except without CpG: DEV4's falls from 76% to 52% at CGI promoters, and the
> no-accessibility reversal (21%) becomes 67% right. The H3K9me3 statements below are single-seed.

- **H1 holds for ATAC, H3K4me3, H3K27ac and H3K9me3** in DEV4 (73–86% of subject genes, 76–91% at CGI promoters).
  *H3K9me3 is at chance at CGIs in DEV4's second seed (64% / 52%).*
- **H1 fails for H3K27me3:** the models predict *more* H3K27me3 on the Xa, most clearly at CGI promoters (only 26%
  of genes in the expected direction for DEV4; 46% without FIRE, i.e. chance; 17% in no FIRE's second seed). Possible reason: on autosomes,
  promoter H3K27me3 sits at unmethylated CGIs (Polycomb targets), and the models learned that link. On the Xi,
  CGIs are methylated and H3K27me3-enriched at the same time, a combination absent from the training chromosome.
  Also, Xi H3K27me3 is broad and partly in different domains from H3K9me3, so promoter windows may not be the right
  place to test it.
- **H2 partly holds** (6 of 8 assays; see the dilution caveat above).
- **H3 holds:** chr21 and escape / PAR genes show no difference.
- **H4 holds:** XIST is reversed in every model (DEV4 H3K4me3 −1.21, i.e. higher on the Xi), and its input
  confirms it (more accessible and unmethylated on the Xi).
- **What the ablations show:**
  - **Active marks:** every model passes, so either input alone is enough; the signal is redundant. CpG alone gives
    the largest differences (H3K4me3 +0.82 at CGIs), in line with the input: the methylation difference at CGI
    promoters (−0.23) is larger than the accessibility difference (+0.08).
  - **H3K9me3** comes out right only with accessibility (no CpG 80% / 86%); from CpG alone it reverses (33% / 21%).
    So the model's Xi H3K9me3 signal is "closed chromatin → H3K9me3", not methylation. *Seed 2: no CpG again
    79% / 86%, but CpG alone now gets 57% / 67% right, so only the no-CpG result is solid. The per-gene H3K9me3
    differences are small (median |d| ≤ 0.08 at CGIs) and flip between trainings when CpG is an input.*
  - **m6A + CpG** passes every active mark (81–87% of subject genes, 88–91% at CGIs) and reverses XIST as strongly
    as DEV4. Raw m6A restores the H3K9me3 direction that CpG alone gets wrong (68% / 64%, against 33% / 21%), but
    less cleanly than with the MSP / nucleosome calls (72–80%). *Seed 2: m6A + CpG 68% / 61% again, but the CpG-only
    comparison it rests on did not replicate.*
  - **H3K27me3** is wrong for every model, worst without CpG (22% at CGIs), closest to chance without FIRE.
    *Seed 2: wrong for every model, 17–24% at CGIs; no FIRE's "closest to chance" was seed noise.*

**Sources:** targets as in section 7 (`tss_chrX_chr21.bed`); per-fiber predictions `molecule_predict.py`, jobs
`logs/2026-11-07_07_molecules_tss_chrX` (DEV4; job 06 crashed on a column-name bug, fixed in 55c22a8) and
`logs/2026-11-08_{10,12,14,20}_molecules_tss_abl_*`; analysis `plots_code/26_nov_13/xi_promoters.py --model <m>` →
`plots/26_nov_13/molecules/xi_*_<m>.{tsv,png}`.

## 9. ENCODE replicate ceiling (I11): how much of each target can a model get?

> **Question:** ENCODE tracks are themselves noisy. How high could any prediction score on each assay, given the
> noise in the target? This separates "hard for the model" from "noisy target", and puts sections 1 and 6 on one
> scale: the fraction of the reachable score each model gets.

### Setup

- **Data:** for each of the 62 pooled targets we train and evaluate on (8 cell types × 8 assays, minus ATAC for H1
  and H9), the per-replicate fold-change bigWigs of the *same* ENCODE experiment: the two lowest-numbered biological
  replicates, cropped to chr21 (`data_prep/setup_replicates.py` → `data_prep/encode_replicates_lock.yaml`;
  124 files, ~1 GB). Every experiment had two.
- **Scoring** (`replicate_ceiling.py`): replicate A against replicate B at exactly the positions `eval.py` scores
  (all chr21 tiles, ≥ 5 fibers, asinh), with the same three metrics and the same cross-cell baseline.
- **From replicate agreement to a ceiling.** The pooled target averages k replicates (k = 2 for 50 targets, 3 for
  12), so it is less noisy than one replicate. With replicate agreement ρ:
  - the pooled track's reliability is the Spearman–Brown value kρ / (1 + (k − 1)ρ), and the best r any prediction
    can reach against it is the square root of that: the **ceiling r**;
  - the same for cell-type r (**ceiling cell-type r**);
  - **MSE floor** = replicate MSE / (2k), the pooled track's own noise if replicate noise is independent.
- **Fraction of the ceiling** = model r / ceiling r per cell type × assay, then averaged. 1 = as good as the
  replicates allow. Cell-type groups as in section 6: dev, held out (H9, HCT116, MCF-7), Panc1.

| | Question | Result |
|---|---|---|
| Q1 | Is the low r on H3K36me3, H3K27me3, H3K9me3 and CTCF target noise? | In dev cell types mostly yes for H3K36me3 and H3K27me3 (67% of the ceiling, like the active marks); H3K9me3 stays furthest from its ceiling |
| Q2 | Is the dev → held-out drop caused by noisier held-out targets? | No: the ceilings are the same (0.74 / 0.74 / 0.75). The drop is the model's, and only in CTCF, H3K36me3, H3K27me3 (and H3K9me3 in Panc1) |
| Q3 | Do the ablation conclusions of section 6 hold on this scale? | Yes: fibertools calls add nothing, accessibility is worth ~10 points of the ceiling, CpG half of dev H3K36me3 |

### The ceiling per assay (all 8 cell types)

| Assay | n | Replicate r | Ceiling r | DEV4 r | Baseline r | Replicate cell-type r | Ceiling cell-type r | MSE floor | DEV4 MSE | Baseline MSE |
|---|---|---|---|---|---|---|---|---|---|---|
| ATAC | 6 | 0.76 | 0.93 | 0.73 | 0.69 | 0.59 | 0.86 | 0.021 | 0.086 | 0.109 |
| H3K4me3 | 8 | 0.44 | 0.77 | 0.55 | 0.50 | 0.31 | 0.67 | 0.082 | 0.176 | 0.218 |
| H3K27ac | 8 | 0.38 | 0.75 | 0.47 | 0.35 | 0.33 | 0.70 | 0.099 | 0.196 | 0.281 |
| H3K4me1 | 8 | 0.50 | 0.80 | 0.52 | 0.43 | 0.42 | 0.75 | 0.084 | 0.196 | 0.272 |
| CTCF | 8 | 0.30 | 0.68 | 0.37 | 0.42 | 0.21 | 0.58 | 0.100 | 0.179 | 0.206 |
| H3K36me3 | 8 | 0.30 | 0.69 | 0.34 | 0.52 | 0.17 | 0.55 | 0.131 | 0.249 | 0.233 |
| H3K27me3 | 8 | 0.26 | 0.66 | 0.35 | 0.39 | 0.20 | 0.59 | 0.133 | 0.216 | 0.264 |
| H3K9me3 | 8 | 0.34 | 0.69 | 0.38 | 0.44 | 0.26 | 0.61 | 0.120 | 0.287 | 0.359 |
| **Mean** | 62 | 0.40 | 0.74 | 0.45 | 0.46 | 0.30 | 0.66 | 0.099 | 0.202 | 0.247 |

![Replicate agreement, ceiling, DEV4 and baseline](../plots/26_nov_13/ceiling/ceiling_overview.png)

- **Two replicates of the same histone or CTCF experiment agree at only r 0.26–0.50** at base-pair resolution on
  chr21. Only ATAC is reproducible (0.76; 0.85 in the dev cell types).
- So **the reachable r against our targets is ~0.65–0.80**, not 1, and DEV4's mean r of 0.45 is 60% of it.
- **Predictions often beat a single replicate** (DEV4 r > replicate r in 42 of 62): a smooth prediction is compared
  with the pooled track, while a replicate carries its own sampling noise. That is why the ceiling is scaled to the
  pooled track rather than read off replicate r.
- **Held-out and Panc1 targets are as reproducible as dev ones** (mean ceiling 0.74 dev, 0.74 held out, 0.75
  Panc1). Panc1's ENCODE tracks are fine; if its data is a problem (section 2), it is on the fiber side.

### DEV4 against the ceiling, per cell-type group

Fraction of the ceiling (model / ceiling), with the baseline's fraction of the r ceiling for comparison.

| Assay | r: dev | r: held out | r: Panc1 | cell-type r: dev | cell-type r: held out | cell-type r: Panc1 | baseline r: dev | baseline r: held out | baseline r: Panc1 |
|---|---|---|---|---|---|---|---|---|---|
| ATAC | 79% | 81% | 70% | 60% | 65% | 40% | 73% | 72% | 82% |
| H3K4me3 | 71% | 75% | 63% | 57% | 61% | 42% | 60% | 72% | 66% |
| H3K27ac | 65% | 63% | 54% | 58% | 55% | 42% | 43% | 49% | 51% |
| H3K4me1 | 66% | 68% | 51% | 58% | 56% | 31% | 46% | 61% | 64% |
| CTCF | 61% | 48% | 55% | 49% | 24% | 32% | 58% | 64% | 72% |
| H3K36me3 | 67% | 32% | 18% | 53% | 8% | −10% | 77% | 75% | 67% |
| H3K27me3 | 67% | 45% | 21% | 54% | 34% | 5% | 64% | 56% | 44% |
| H3K9me3 | 55% | 57% | 39% | 35% | 30% | 4% | 57% | 75% | 64% |
| **Mean** | 66% | 58% | 46% | 53% | 40% | 23% | 59% | 65% | 64% |

- **In dev cell types every assay reaches 55–79% of its ceiling.** H3K36me3 and H3K27me3 (67%) are as close as
  H3K27ac and H3K4me1, so their low raw r is mostly target noise (Q1). H3K9me3 is furthest (55%; cell-type r 35%).
- **ATAC and the active marks transfer to held-out cell types unchanged** (63–81%, within 2 points of dev).
- **The held-out drop comes from CTCF, H3K36me3 and H3K27me3** (Q2). Their cell-type r falls to 8–34% of the
  ceiling. The cell-specific H3K36me3 is almost entirely missed (8%), although the replicates show it is there
  (cell-type ceiling 0.41–0.70). Panc1 adds H3K9me3 and is lowest overall (46%).
- **On held-out cell types the cross-cell baseline is closer to the ceiling than DEV4 for CTCF, H3K36me3, H3K27me3
  and H3K9me3** (56–75% against 32–57%), and over all cell types its r beats DEV4's in 28 of 62 pairs. Most of
  what is reproducible in these marks is shared across cell types, so the average gets it for free; on held-out
  cell types DEV4 predicts them worse than copying the average.

### All five models against the ceiling

Top: fraction of the r ceiling (with the baseline in grey). Bottom: fraction of the cell-type r ceiling.

![Fraction of the ceiling, five models](../plots/26_nov_13/ceiling/ceiling_models.png)

#### Pearson r / ceiling, dev

| Assay | DEV4 | no FIRE | m6A + CpG | no acc. | no CpG | baseline |
|---|---|---|---|---|---|---|
| ATAC | **0.79** | **0.79** | **0.79** | 0.64 | 0.77 | 0.73 |
| H3K4me3 | 0.71 | **0.72** | 0.71 | 0.65 | 0.65 | 0.60 |
| H3K27ac | 0.65 | **0.67** | **0.67** | 0.54 | 0.60 | 0.43 |
| H3K4me1 | 0.66 | **0.69** | **0.69** | 0.49 | 0.63 | 0.46 |
| CTCF | 0.61 | **0.62** | 0.60 | 0.48 | 0.59 | 0.58 |
| H3K36me3 | 0.67 | 0.68 | 0.68 | 0.65 | 0.33 | **0.77** |
| H3K27me3 | 0.67 | 0.68 | **0.69** | 0.62 | 0.54 | 0.65 |
| H3K9me3 | 0.55 | 0.54 | 0.55 | 0.44 | 0.54 | **0.57** |
| **Mean** | 0.66 | **0.67** | **0.67** | 0.56 | 0.57 | 0.59 |

#### Pearson r / ceiling, held out (H9, HCT116, MCF-7)

| Assay | DEV4 | no FIRE | m6A + CpG | no acc. | no CpG | baseline |
|---|---|---|---|---|---|---|
| ATAC | 0.81 | **0.82** | **0.82** | 0.60 | 0.80 | 0.72 |
| H3K4me3 | 0.75 | 0.77 | **0.78** | 0.73 | 0.71 | 0.72 |
| H3K27ac | 0.63 | 0.63 | **0.65** | 0.54 | 0.59 | 0.49 |
| H3K4me1 | 0.68 | **0.73** | 0.71 | 0.45 | 0.66 | 0.61 |
| CTCF | 0.48 | 0.50 | 0.48 | 0.30 | 0.49 | **0.64** |
| H3K36me3 | 0.32 | 0.33 | 0.34 | 0.32 | 0.25 | **0.75** |
| H3K27me3 | 0.45 | 0.50 | 0.49 | 0.47 | 0.40 | **0.56** |
| H3K9me3 | 0.57 | 0.57 | 0.56 | 0.49 | 0.52 | **0.75** |
| **Mean** | 0.58 | 0.60 | 0.59 | 0.48 | 0.54 | **0.65** |

#### Cell-type r / ceiling, dev

| Assay | DEV4 | no FIRE | m6A + CpG | no acc. | no CpG |
|---|---|---|---|---|---|
| ATAC | 0.60 | **0.61** | 0.60 | 0.38 | 0.58 |
| H3K4me3 | 0.57 | **0.58** | **0.58** | 0.47 | 0.50 |
| H3K27ac | 0.58 | 0.60 | **0.61** | 0.45 | 0.52 |
| H3K4me1 | 0.58 | **0.62** | **0.62** | 0.38 | 0.54 |
| CTCF | 0.49 | **0.51** | 0.48 | 0.36 | 0.47 |
| H3K36me3 | 0.53 | 0.54 | **0.55** | 0.52 | 0.19 |
| H3K27me3 | 0.54 | **0.57** | 0.56 | 0.49 | 0.35 |
| H3K9me3 | 0.35 | 0.35 | **0.36** | 0.25 | **0.36** |
| **Mean** | 0.53 | **0.55** | 0.54 | 0.41 | 0.43 |

#### Cell-type r / ceiling, held out (H9, HCT116, MCF-7)

| Assay | DEV4 | no FIRE | m6A + CpG | no acc. | no CpG |
|---|---|---|---|---|---|
| ATAC | 0.65 | **0.68** | 0.67 | 0.30 | 0.64 |
| H3K4me3 | 0.61 | 0.63 | **0.65** | 0.55 | 0.57 |
| H3K27ac | 0.55 | 0.54 | **0.57** | 0.43 | 0.50 |
| H3K4me1 | 0.56 | **0.62** | 0.60 | 0.27 | 0.53 |
| CTCF | 0.24 | **0.26** | 0.24 | 0.05 | 0.25 |
| H3K36me3 | 0.08 | 0.09 | **0.10** | **0.10** | 0.06 |
| H3K27me3 | 0.34 | **0.44** | 0.37 | 0.40 | 0.20 |
| H3K9me3 | 0.30 | **0.31** | 0.30 | 0.23 | 0.28 |
| **Mean** | 0.41 | **0.44** | 0.43 | 0.29 | 0.37 |

Panc1 is in the figure (third column) and in `ceiling_groups.tsv`.

#### MSE / floor, mean over assays (lower is better)

| Group | DEV4 | no FIRE | m6A + CpG | no acc. | no CpG | baseline |
|---|---|---|---|---|---|---|
| Dev | **2.77** | 2.84 | 2.78 | 3.40 | 2.96 | 3.87 |
| Held out | 2.20 | **2.15** | 2.17 | 2.54 | 2.20 | 2.57 |
| Panc1 | 2.69 | **2.59** | 2.68 | 2.89 | 2.64 | 2.81 |

Per assay in `ceiling_groups.tsv`. Dev ATAC (7.8×) and dev H3K9me3 (3.8×) are the highest because their floors are
small. ATAC is very reproducible. HepG2's H3K9me3 replicates agree at r 0.83, while DEV4's MSE there is 0.69
against a floor of 0.07: the HepG2 H3K9me3 scale problem from section 5 is a real, reproducible track. On held-out
H3K36me3 the baseline is closer to the floor than every model (2.04 against 2.25–2.46).

### What the ceiling shows

1. **The targets cap r at ~0.74, and DEV4 gets 60% of that** (66% in dev, 58% held out, 46% in Panc1). Most of the
   "weak broad marks" in dev cell types is target noise: H3K36me3 and H3K27me3 are as close to their ceiling as the
   active marks.
2. **The real gaps are the cell-type-specific signal in held-out cell types** for CTCF, H3K36me3 and H3K27me3 (and
   H3K9me3 in Panc1), plus H3K9me3 everywhere. There, the reproducible part is mostly shared across cell types,
   and the cross-cell average is closer to the ceiling than any model.
3. **The ablation conclusions hold on this scale.** No FIRE and m6A + CpG are level with or above DEV4 in r and
   cell-type r in every group (0 to +0.04 of the ceiling), and within 0.1 of it in MSE / floor. Without accessibility the models lose ~10 points of the ceiling
   (cell-type r of ATAC 0.60 → 0.38 dev, 0.65 → 0.30 held out). Without CpG, dev H3K36me3 halves (0.67 → 0.33).
4. **No input fixes held-out H3K36me3:** every model gets 6–10% of its cell-specific ceiling. That is a
   generalization failure, not a missing input.
5. **No FIRE's H3K27me3 gain is the largest difference between DEV4 and the models that keep both raw inputs**
   (held-out cell-type r 0.44 against 0.34; Panc1 0.26 against 0.05). With one seed it could be noise; the second
   seeds will tell. *They did (section 10): it is mostly noise. Averaged over both seeds no FIRE's held-out
   H3K27me3 cell-type r is 0.37 of the ceiling against DEV4's 0.32.*

### Caveats

- **Unequal replicates.** In 14 of 62 experiments one replicate matches the pooled track much better than the
  other (difference > 0.15, e.g. H1 CTCF 0.55 / 0.92, MCF-7 H3K27me3 0.52 / 0.94). The pooled track is then
  mostly one replicate, the true ceiling is higher, and the fractions there are too generous (H1 CTCF's 86% of the
  ceiling is one of them). The list is in `log_ceiling.txt`.
- **Spearman–Brown assumes equally good replicates**, and the pooled ENCODE fold change is computed from pooled
  reads, not as the mean of the replicate tracks. The MSE floor (replicate MSE / 2k) is the rougher of the two
  estimates.
- **The 12 targets pooled from three replicates** are scaled from two of the three.
- **Base-pair resolution.** Ceilings would be higher at 200 bp or 1 kb, most of all for the broad marks.
- **Seed noise:** the tables are seed 919. With both seeds averaged the fractions move by ≤ 0.01 on the mean
  (DEV4 65 / 58 / 47%; section 10, `seeds_ceiling.tsv`).

**Sources:** download job `logs/2026-11-11_00_setup_replicates` (124 / 124 files); scoring jobs
`logs/2026-11-11_0{2,3}_eval_replicate_ceiling*` (`slurm_batch_command_replicates.sh`, CPU only, ~18 min per cell
type); results `results/*_replicates_data24_dev_loo_Panc1_200U/replicate_summary.json` on nibi. Tables and
figures: `plots_code/26_nov_13/ceiling.py` (uses `ablations.py`'s `bulk_{r,mse,ctr}.tsv`), outputs in
`plots/26_nov_13/ceiling/` (`ceiling_cells.tsv`, `ceiling_groups.tsv`, `log_ceiling.txt`).

## 10. Second training seed: which results replicate?

> **Question:** sections 6–9 rest on one training per model. How much do the numbers move when the same model is
> trained again with a different seed, and which model differences survive that?

### Setup

- **Runs:** all five models of section 6 retrained with `--seed 2` instead of 919 (`SEED=2` in
  `slurm_batch_command.sh`; run prefix `*_seed2`). The configs, data, budget and slice are unchanged. The seed sets
  weight initialization, training-region order and fiber sampling, and the validation windows.
- **Evals:** each run was chained to the same jobs as seed 919: dev and held-out bulk evals, pair evals, and both
  molecule jobs (`2026-11-11_04`–`33`, all completed, trainings 2 h 11–19 min). The test evals take their seed from
  the eval config, so both seeds are scored on the same fibers.
- **Seed noise of a table entry:** for each group mean (one model, one assay, one cell-type group), d = seed 2 −
  seed 919. Pooling d over the five models gives the SD of one training, sd = √(mean d² / 2). A difference between
  two models, each averaged over its two seeds, has the same SD. Below, an ablation effect counts as **clear** if
  its two-seed mean is beyond 2 SD and both seeds give the same sign.

### Bulk: the numbers barely move

![Seed 2 against seed 919, every cell type × assay](../plots/26_nov_13/seeds/seeds_scatter.png)

- **Per cell type × assay, the two seeds differ by a median of 0.010 in r** (90th percentile 0.036, largest 0.09:
  no CpG, HCT116 H3K9me3) and 0.012 in cell-type r (largest 0.11: no FIRE, Panc1 H3K27me3).
- **Group means move less.** The SD of one training is 0.005–0.009 for the mean over assays, and per assay mostly
  0.003–0.013:

| Assay | r: dev | r: held out | r: Panc1 | cell-type r: dev | cell-type r: held out | cell-type r: Panc1 | MSE: dev | MSE: held out | MSE: Panc1 |
|---|---|---|---|---|---|---|---|---|---|
| ATAC | 0.006 | 0.007 | 0.005 | 0.010 | 0.007 | 0.007 | 0.002 | 0.002 | 0.010 |
| H3K4me3 | 0.013 | 0.020 | 0.021 | 0.014 | 0.022 | 0.021 | 0.007 | 0.014 | 0.024 |
| H3K27ac | 0.005 | 0.006 | 0.007 | 0.005 | 0.007 | 0.007 | 0.004 | 0.003 | 0.005 |
| H3K4me1 | 0.008 | 0.011 | 0.004 | 0.009 | 0.013 | 0.004 | 0.004 | 0.006 | 0.009 |
| CTCF | 0.003 | 0.004 | 0.004 | 0.003 | 0.004 | 0.006 | 0.001 | 0.002 | 0.003 |
| H3K36me3 | 0.009 | 0.005 | 0.010 | 0.009 | 0.007 | 0.009 | 0.003 | 0.009 | 0.021 |
| H3K27me3 | 0.004 | **0.028** | **0.028** | 0.005 | **0.034** | **0.037** | 0.012 | 0.008 | 0.004 |
| H3K9me3 | 0.007 | 0.012 | **0.032** | 0.009 | 0.008 | **0.039** | 0.017 | 0.011 | 0.004 |
| **Mean** | 0.005 | 0.008 | 0.007 | 0.006 | 0.009 | 0.008 | 0.003 | 0.002 | 0.003 |

  H3K4me3 is the noisiest assay everywhere (~0.02), and H3K27me3 / H3K9me3 outside the dev lines are the least
  stable (0.03–0.04).
- **The ceiling fractions of section 9 hold.** Averaged over both seeds, DEV4 gets 65% of the r ceiling in dev,
  58% held out and 47% in Panc1 (seed 919: 66 / 58 / 46%). No FIRE 66 / 59 / 49%, m6A + CpG 67 / 58 / 47%
  (`seeds_ceiling.tsv`).

### Bulk: ablation effects in both seeds

Ablation − DEV4, each seed paired with the DEV4 of the same seed (seed 919 / seed 2); **bold** = clear; last
column = SD of one training.

![Ablation effects on Pearson r, both seeds](../plots/26_nov_13/seeds/seeds_effects_r.png)

#### Pearson r, dev

| Assay | no FIRE | m6A + CpG | no acc. | no CpG | 1 SD |
|---|---|---|---|---|---|
| ATAC | +0.004 / +0.009 | +0.001 / +0.009 | **−0.140 / −0.147** | −0.015 / +0.004 | 0.006 |
| H3K4me3 | +0.006 / −0.011 | +0.004 / −0.006 | **−0.043 / −0.047** | **−0.047 / −0.032** | 0.013 |
| H3K27ac | **+0.016 / +0.018** | **+0.017 / +0.022** | **−0.088 / −0.101** | **−0.037 / −0.033** | 0.005 |
| H3K4me1 | **+0.030 / +0.019** | **+0.028 / +0.033** | **−0.134 / −0.135** | −0.025 / −0.003 | 0.008 |
| CTCF | **+0.009 / +0.007** | −0.003 / −0.000 | **−0.085 / −0.085** | −0.009 / −0.001 | 0.003 |
| H3K36me3 | +0.009 / −0.012 | +0.011 / −0.003 | −0.013 / −0.020 | **−0.250 / −0.234** | 0.009 |
| H3K27me3 | **+0.009 / +0.008** | **+0.012 / +0.014** | **−0.032 / −0.034** | **−0.090 / −0.076** | 0.004 |
| H3K9me3 | −0.003 / +0.010 | +0.001 / +0.011 | **−0.087 / −0.080** | −0.012 / +0.015 | 0.007 |
| **Mean** | +0.010 / +0.006 | +0.009 / +0.010 | **−0.076 / −0.079** | **−0.062 / −0.046** | 0.005 |

#### Pearson r, held out (H9, HCT116, MCF-7)

| Assay | no FIRE | m6A + CpG | no acc. | no CpG | 1 SD |
|---|---|---|---|---|---|
| ATAC | +0.009 / +0.005 | +0.006 / +0.003 | **−0.197 / −0.217** | −0.012 / −0.004 | 0.007 |
| H3K4me3 | +0.012 / −0.036 | +0.021 / −0.027 | −0.015 / −0.033 | −0.033 / −0.024 | 0.020 |
| H3K27ac | +0.003 / +0.019 | **+0.014 / +0.021** | **−0.067 / −0.065** | **−0.031 / −0.009** | 0.006 |
| H3K4me1 | **+0.033 / +0.021** | **+0.024 / +0.030** | **−0.183 / −0.192** | −0.024 / +0.005 | 0.011 |
| CTCF | **+0.014 / +0.006** | **−0.004 / −0.014** | **−0.139 / −0.140** | +0.005 / +0.006 | 0.004 |
| H3K36me3 | +0.014 / +0.001 | **+0.018 / +0.004** | +0.006 / −0.015 | **−0.051 / −0.066** | 0.005 |
| H3K27me3 | +0.030 / +0.008 | +0.023 / −0.010 | +0.011 / +0.010 | −0.031 / +0.005 | 0.028 |
| H3K9me3 | +0.002 / +0.001 | −0.007 / −0.014 | **−0.054 / −0.077** | −0.033 / −0.009 | 0.012 |
| **Mean** | +0.015 / +0.003 | +0.012 / −0.001 | **−0.075 / −0.086** | **−0.027 / −0.012** | 0.008 |

#### Cell-type r, held out (H9, HCT116, MCF-7)

| Assay | no FIRE | m6A + CpG | no acc. | no CpG | 1 SD |
|---|---|---|---|---|---|
| ATAC | **+0.019 / +0.018** | +0.015 / +0.005 | **−0.289 / −0.308** | −0.007 / +0.001 | 0.007 |
| H3K4me3 | +0.013 / −0.047 | +0.028 / −0.025 | −0.032 / −0.053 | −0.028 / −0.025 | 0.022 |
| H3K27ac | −0.000 / +0.019 | **+0.014 / +0.025** | **−0.073 / −0.071** | **−0.030 / −0.007** | 0.007 |
| H3K4me1 | **+0.043 / +0.030** | **+0.030 / +0.039** | **−0.213 / −0.221** | −0.020 / +0.013 | 0.013 |
| CTCF | **+0.014 / +0.004** | **−0.003 / −0.016** | **−0.123 / −0.127** | **+0.012 / +0.009** | 0.004 |
| H3K36me3 | +0.008 / −0.013 | +0.012 / −0.006 | +0.016 / −0.009 | **−0.022 / −0.051** | 0.007 |
| H3K27me3 | +0.060 / +0.001 | +0.016 / −0.040 | +0.033 / +0.032 | **−0.083 / −0.061** | 0.034 |
| H3K9me3 | +0.005 / +0.004 | −0.004 / −0.016 | **−0.044 / −0.059** | −0.017 / +0.001 | 0.008 |
| **Mean** | +0.020 / +0.002 | +0.013 / −0.005 | **−0.082 / −0.093** | **−0.025 / −0.016** | 0.009 |

Cell-type r in the dev lines and MSE: `seeds_effects.tsv` and `seeds_effects_ctr.png`.

- **Every large effect of section 6 replicates** at the same size: no accessibility (−0.08 mean r; ATAC, H3K4me1,
  H3K27ac, CTCF, H3K9me3) and no CpG (dev H3K36me3 −0.25 / −0.23, H3K27me3 −0.09 / −0.08, H3K4me3 −0.05 / −0.03).
  CTCF and H3K9me3 are again unaffected without CpG.
- **No FIRE and m6A + CpG are not just "level" with DEV4: both are slightly better on H3K27ac and H3K4me1**
  (+0.02–0.03 r in both seeds: H3K4me1 in dev and held out, H3K27ac in dev, and held out for m6A + CpG). Over all
  assays their gain (−0.001 to +0.015) is within 2 SD.
- **No FIRE's H3K27me3 gain mostly does not replicate.** Held-out cell-type r +0.060 → +0.001, Panc1 +0.148 →
  +0.032 (r +0.113 → +0.031). Per cell type its r stays above DEV4's in 7 of 8 lines in seed 2, but by +0.00–0.03.
- **No FIRE's dev H3K27me3 MSE exception is seed noise:** +0.050 / +0.050 / +0.038 in GM12878 / K562 / HepG2 with
  seed 919, −0.009 / −0.010 / −0.003 with seed 2.

### Molecules: much more seed-dependent

![Single-molecule readouts, both seeds](../plots/26_nov_13/seeds/seeds_molecules.png)

**CTCF footprints (section 7)**, bound sites, per-site medians (seed 919 / seed 2):

| Model | Footprint − open, CTCF: GM12878 | K562 | CTCF − ATAC: GM12878 | K562 | Unbound sites: footprint − open, CTCF |
|---|---|---|---|---|---|
| DEV4 | +0.11 / **+0.73** | +0.04 / **+0.44** | +0.02 / +0.23 | +0.05 / +0.21 | −0.06 to −0.01 |
| No FIRE | **+1.14 / +0.95** | **+0.84 / +0.68** | **+0.67 / +0.85** | **+0.54 / +0.63** | −0.05 to −0.02 |
| m6A + CpG | +0.26 / +0.56 | +0.20 / +0.58 | +0.20 / +0.55 | +0.14 / +0.50 | −0.04 to +0.03 |
| No accessibility | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | ≈ 0 |
| No CpG | +0.18 / +0.75 | +0.10 / +0.46 | −0.12 / +0.33 | −0.02 / +0.31 | −0.07 to −0.03 |

(CTCF − ATAC is also positive at unbound sites, +0.04–0.27 depending on model and seed; see the caveat in section 7.)

- **What replicates:** H1 (footprint fibers get more predicted CTCF than open fibers at bound sites) for every
  model that sees m6A, p ≤ 3e-17 in seed 2; no effect at unbound sites; the no-accessibility null.
- **What does not:** the size of the response. It differs 2–10× between the two trainings of DEV4 (+0.11 → +0.73 in
  GM12878, +0.04 → +0.44 in K562), m6A + CpG (+0.26 → +0.56) and no CpG (+0.18 → +0.75). So **"10× stronger without FIRE" was mostly DEV4's weak
  seed-919 run**; in seed 2 no FIRE is 1.3–1.5× DEV4.
- **No FIRE is the only model that is strong and CTCF-specific in both trainings**, and it is the highest in both
  seeds and both lines. In seed 2, where DEV4 responds too, no FIRE's CTCF − ATAC is still 3–4× DEV4's (+0.85 / +0.63 against
  +0.23 / +0.21).
- No CpG's H2 failure (CTCF − ATAC < 0) and m6A + CpG's "a quarter of no FIRE" do not replicate either (seed 2:
  +0.33 / +0.31, and 60–85% of no FIRE).

**X inactivation (section 8)**, fraction of subject genes in the expected direction, all / CGI promoters (seed 919
→ seed 2):

| Assay (expected on Xa) | DEV4 | No CpG | No FIRE | No accessibility | m6A + CpG |
|---|---|---|---|---|---|
| ATAC (higher) | .86 / .91 → .86 / .91 | .82 / .88 → .84 / .89 | .83 / .90 → .85 / .90 | .79 / .88 → .81 / .88 | .84 / .89 → .87 / .90 |
| H3K4me3 (higher) | .73 / .84 → .78 / .86 | .75 / .85 → .74 / .84 | .79 / .87 → .73 / .86 | .85 / .88 → .82 / .86 | .81 / .88 → .71 / .82 |
| H3K27ac (higher) | .82 / .84 → .84 / .87 | .82 / .89 → .83 / .88 | .87 / .90 → .84 / .86 | .87 / .90 → .85 / .87 | .87 / .91 → .85 / .88 |
| H3K9me3 (lower) | .73 / .76 → .64 / **.52** | .80 / .86 → .79 / .86 | .72 / .71 → .71 / .65 | **.33 / .21 → .57 / .67** | .68 / .64 → .68 / .61 |
| H3K27me3 (lower) | .43 / .26 → .40 / .24 | .28 / .22 → .28 / .23 | .56 / .46 → **.29 / .17** | .46 / .29 → .41 / .23 | .48 / .35 → .28 / .17 |

- **What replicates:** the active marks in every model (71–91%), XIST reversed in all ten runs, the H3K27me3
  failure (now in every run but one), and no CpG's H3K9me3 (79–86% in both seeds).
- **What does not:** H3K9me3 in the models that see CpG. It drops to chance at DEV4's CGI promoters (52%), and the
  no-accessibility model's reversal (21% at CGIs) becomes 67% right. The per-gene H3K9me3 differences are small
  (median |d| ≤ 0.08 at CGIs), so the fraction flips easily. No FIRE's "closest to chance" H3K27me3 (46%) is
  17% in seed 2.

### What the second seed shows

1. **Bulk results are robust.** One training moves a reported group mean by ~0.005–0.01 (up to 0.03–0.04 for
   held-out H3K27me3 / H3K9me3 and H3K4me3). Every ablation conclusion of section 6 holds; the ceiling fractions of
   section 9 hold.
2. **Small differences of ±0.02 need two seeds.** Two single-seed claims fell: no FIRE's H3K27me3 gain (section 9,
   point 5) and its dev H3K27me3 MSE exception (section 6). A small real one appeared: no FIRE and m6A + CpG gain
   +0.02–0.03 on H3K27ac and H3K4me1.
3. **Molecule-level readouts vary far more between trainings than bulk does.** The direction of the CTCF footprint
   response replicates; its size varies 2–10× for DEV4, m6A + CpG and no CpG. The H3K9me3 direction on the Xi
   is unstable for every model that sees CpG.
4. **The case for no FIRE stands, on a different basis:** it is the only model whose CTCF footprint response is
   strong and specific in both trainings, with no bulk cost (a small gain on H3K27ac / H3K4me1). The ranking of
   the other models' molecule responses cannot be read from one seed.

**Sources:** training `logs/2026-11-11_{04,10,16,22,28}_train_*_seed2`, evals and molecule jobs
`logs/2026-11-11_{05..33}_*_seed2*`; results `results/*_seed2_*` on nibi. Molecule analyses
`ctcf_footprints.py` / `xi_promoters.py --model <m> --seed 2` → `plots/26_nov_13/molecules/*_<m>_seed2.*`. Tables and
figures `plots_code/26_nov_13/seeds.py` → `plots/26_nov_13/seeds/` (`seeds_bulk.tsv`, `seeds_groups.tsv`,
`seeds_noise.tsv`, `seeds_effects.tsv`, `seeds_ceiling.tsv`, `seeds_molecules.tsv`, `log_seeds.txt`). The pair
evals of seed 2 ran too (`*_eval_*_seed2*/` with the pair keys) but are not analysed here.

## Decisions and next steps

- **Make the no-FIRE model the main model** (the decision rule fixed in `TODO.md` §2c). After the second seed
  (section 10) the case is: the same bulk accuracy or slightly better (H3K27ac, H3K4me1 +0.02–0.03 r in both
  seeds), no bulk-assay knowledge in the input, and **the only model whose molecule-level CTCF response is strong
  and CTCF-specific in both trainings**. It is not "10× stronger" than DEV4: that was DEV4's weak seed-919 run.
  Whether to keep the MSP / nucleosome calls is open (no FIRE ≥ m6A + CpG in both seeds, by 1.2–4.4×); keeping
  them costs nothing in bulk.
- **Molecule-level claims need several seeds.** Bulk numbers move ~0.01 between trainings, molecule readouts 2–10×.
  Report molecule tests over at least two seeds (or an ensemble) from now on; for the main model, consider 3 seeds.
  The seed-2 pair evals are done but not analysed yet.
- **What to improve, from the ceiling (section 9):** in dev cell types the model is at 55–79% of every ceiling, so
  the room is in held-out cell types, in the cell-specific part of CTCF, H3K36me3 and H3K27me3, where the
  cross-cell average currently does better. H3K9me3 is the one assay far from its ceiling everywhere. Score future
  models as a fraction of the ceiling; rerun the ceiling at 200 bp / 1 kb if we change output resolution.
- **X inactivation:** restrict the subject / escape comparison to genes expressed in GM12878 (RNA-seq), check the
  escape list against Tukiainen 2017, and test H3K27me3 over broader windows (gene bodies, Xi domains) as well as
  promoters.
- **Remaining tests from `TODO.md` §2:**
  - in-silico CpG flips (I21);
  - held-out methylation with the no-CpG model (T5a): that model now exists, so a molecule's measured methylation is
    held-out data;
  - imprinted control regions and allele-specific ChIP;
  - the search for external single-molecule histone data (§2f).
- **Bulk:** H3K36me3 is read from CpG, which suggests its weakness in held-out cell types comes from methylation
  patterns that differ between lines, not from context length alone. Worth keeping in mind for the longer-context
  runs (`TODO.md` §3).
