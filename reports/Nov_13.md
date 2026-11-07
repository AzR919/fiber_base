# 13th Nov

Work in progress. This report covers results on the current setup: the reference models and the unseen-pair test
(`TODO.md`).

## Summary

- **Reference model (DEV5, 5 dev cell types, `train04`).**
  - It beats the cross-cell baseline on MSE in 7 of 8 cell types, and is ahead of or level with it on Pearson in 4
    of 5 dev cell types.
  - It trails the baseline on Pearson for Panc1 and the 3 held-out cell types.
  - The marks accessibility reflects most directly (ATAC, H3K4me3, H3K27ac, H3K4me1) carry over to held-out cell
    types. The broad and repressive marks don't, H3K36me3 least of all.
- **Panc1 and split v2.**
  - Panc1's sample is doubtful: its fibers match HCT116's ATAC better than Panc1's own, it lacks some of PANC-1's
    markers, and it has half coverage.
  - Training on it buys nothing.
  - It moves to the held-out set as a "hard" cell type. The 4-dev model **DEV4** replaces DEV5 as the main model.
  - Specialists, one cell type each, beat the multi-cell model by +0.03–0.06 Pearson.
- **Mixture test on two pairs** (GM12878–HepG2 and K562–H1).
  - Training only on mixtures reaches the own-track ceiling on both pairs (fiber-origin AUC 0.637 vs 0.635, and
    0.666 vs 0.665).
  - On a pair it never saw, a model does the same whether it was trained on mixtures or on own tracks. What
    matters is having seen the cell types.
  - Transfer toward K562–H1 is poor: sign accuracy 0.59 vs 0.80–0.83.
  - Library effects are small (|AUC − 0.5| 0.04–0.09, against 0.17–0.25 for real pairs).
  - Weak spots: CTCF on K562–H1 and H3K9me3 on GM12878–HepG2 fail even for models trained on the pair.
    - For CTCF, H1's ChIP track is weakly enriched (it is the outlier at shared sites; MCF-7's is too).
    - On top of that, every model adds a constant shift toward H1 that accessibility doesn't explain.

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

## DEV5 (V1), the all-cells reference model

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

## Panc1 and the new split (v2): Panc1 moves to held out

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

## Mixture test on two pairs (T3 repeated on K562–H1)

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

### CTCF on K562–H1: a weak H1 track plus a model shift

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
- The cause is not tested. It is not a global "H1 looks more open" effect: H1's library has a lower MSP fraction
  (0.27 vs 0.34) and a lower m6A rate (0.092 vs 0.117) than K562's (`fiber_qc.py`). The big global difference is
  CpG methylation, 0.27 in H1 against 0.085 in K562. The model could read that as a cell-level shift.
- The planned input ablations (`TODO.md` §2, dropping the CpG channel) would test this.

**Consequences:**
- K562–H1 CTCF bins are a poor test of CTCF: most of them measure H1's ChIP enrichment, not differential binding.
  The CTCF row of the K562–H1 tables should be read with that in mind.
- The weak H1 and MCF-7 CTCF tracks also lower those cell types' bulk CTCF numbers.

**Sources:** `plots_code/26_nov_13/ctcf_offset.py`. Fiber part: job `2026-11-07_00_ctcf_offset` (CPU, 2 min).
Outputs are in `plots/26_nov_13/ctcf/`: per-bin tables and the figure.

### The same breakdown for every assay on both pairs

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
- The CpG-channel ablation for the CTCF shift.

**Sources (mixture test):**
- Training and eval logs: `logs/2026-11-04_*` on nibi. Paired evals that timed out were rerun as
  `2026-11-04_{15..18}_eval_*_rerun`.
- Results: `results/26-11-04_*_eval_{MIX_V1,PAIR_V1,LOO_Panc1_V1}_*`, i.e. `<config>_paired.npz` and
  `eval_summary.json`.
- Figures, tables and CSV: `plots_code/26_nov_13/mix_two_pairs.py`, which writes to `plots/26_nov_13/mix/`.
