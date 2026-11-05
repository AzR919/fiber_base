# 13th Nov

Work in progress. Results on the current setup: the reference model bank and the unseen-pair test (`TODO.md`).

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

October's mixture test (T3) used one pair. With split v2 we have two pairs of dev cell types, **A = GM12878 +
HepG2** and **B = K562 + H1**, and train four models on them (same `model15` / `train04` setup as DEV4, ~2 h 15 min
each on nibi):

- **Mix A / Mix B** (`data27` / `data35`): trained only on synthetic mixtures of the pair's fibers (cell fraction
  q drawn from [0.2, 0.8]), supervised by the q-weighted fold-change mixture of the two tracks. They never see a
  cell type's own track. H1 has no ATAC, so Mix B never trains ATAC (the mixture's assay mask is the product over
  its cells).
- **Pair A / Pair B** (`data28` / `data34`): the same cells, each fiber supervised by its own cell type's track
  (the ceiling for that pair).
- **DEV4** (`data24`): all 4 dev cell types, own tracks.

Every model is tested on both pairs: its own pair, and the pair it never saw. The readouts come from paired
evaluation at the pair's differential bins (`loci_spec.yaml` pairs: 1 kb bins where the two tracks differ by
|asinh Δ| ≥ 0.5) on chr21:

- **Fiber-origin AUC:** in each tile, how well a fiber's prediction tells which cell type it came from, oriented by
  the tracks' difference (0.5 = chance).
- **Sign accuracy:** the fraction of differential bins where the predicted difference between the two cell types
  has the tracks' sign.
- **Differential r:** bp-level correlation of predicted vs track difference (p_A − p_B vs t_A − t_B).

### Mean over assays

GM12878–HepG2 has 8 assays; K562–H1 has 7 (no H1 ATAC). Values are AUC / sign accuracy / differential r.

| Model | GM12878–HepG2 | K562–H1 |
|---|---|---|
| Mix A (GM12878 + HepG2 mixtures) | **0.637** / 0.761 / 0.329 *(own pair)* | 0.599 / 0.590 / 0.238 *(unseen)* |
| Pair A (GM12878, HepG2 own tracks) | 0.635 / **0.771** / **0.351** *(own pair)* | 0.600 / 0.598 / 0.224 *(unseen)* |
| Mix B (K562 + H1 mixtures) | 0.621 / 0.706 / 0.263 *(unseen)* | **0.666** / 0.802 / 0.291 *(own pair)* |
| Pair B (K562, H1 own tracks) | 0.617 / 0.719 / 0.270 *(unseen)* | 0.665 / **0.828** / **0.335** *(own pair)* |
| DEV4 (all 4 own tracks) | 0.629 / 0.748 / 0.316 | 0.654 / 0.793 / 0.294 |

### Fiber-origin AUC per assay

GM12878–HepG2:

| Model | ATAC | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|---|
| Mix A | .713 | .687 | .688 | .644 | .685 | .550 | .483 | .643 |
| Pair A | .705 | .677 | .676 | .653 | .683 | .542 | .499 | .642 |
| Mix B | .705 | .737 | .673 | .575 | .671 | .485 | .462 | .659 |
| Pair B | .710 | .735 | .660 | .575 | .650 | .514 | .468 | .623 |
| DEV4 | .713 | .707 | .674 | .623 | .674 | .525 | .464 | .652 |
| *bins* | 775 | 355 | 702 | 605 | 1472 | 592 | 2952 | 388 |

K562–H1:

| Model | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|
| Mix A | .670 | .564 | .796 | .724 | .476 | .505 | .458 |
| Pair A | .664 | .587 | .788 | .724 | .468 | .496 | .468 |
| Mix B | .742 | .653 | .786 | .711 | .687 | .615 | .469 |
| Pair B | .727 | .678 | .771 | .677 | .687 | .643 | .472 |
| DEV4 | .698 | .641 | .790 | .715 | .638 | .614 | .479 |
| *bins* | 408 | 509 | 925 | 1118 | 670 | 839 | 428 |

### Differential r per assay

GM12878–HepG2:

| Model | ATAC | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|---|
| Mix A | .60 | .22 | .45 | .27 | .60 | .13 | .19 | .16 |
| Pair A | .61 | .26 | .46 | .30 | .60 | .13 | .28 | .17 |
| Mix B | .37 | .20 | .36 | .18 | .54 | .07 | .23 | .16 |
| Pair B | .57 | .20 | .32 | .17 | .49 | .06 | .24 | .11 |
| DEV4 | .58 | .23 | .43 | .24 | .58 | .11 | .19 | .17 |

K562–H1:

| Model | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|
| Mix A | .28 | .25 | .25 | .35 | .25 | .13 | .15 |
| Pair A | .27 | .25 | .23 | .35 | .21 | .12 | .15 |
| Mix B | .36 | .31 | .26 | .36 | .33 | .20 | .21 |
| Pair B | .39 | .37 | .31 | .39 | .33 | .22 | .34 |
| DEV4 | .38 | .30 | .27 | .37 | .32 | .20 | .23 |

### Library control

The control compares two Fiber-seq libraries of the **same** cell type (100U vs 200U; both libraries scored at the
real pairs' bins), so any separation comes from the library, not the cell type. The readout is the mean over tiles
of |AUC − 0.5|; for the real pairs the same measure is 0.17–0.19 (GM12878–HepG2) and 0.22–0.25 (K562–H1).

| Model | HepG2 | K562 | H1 |
|---|---|---|---|
| Mix A | .049 | .089 | .044 |
| Pair A | .049 | .090 | .045 |
| Mix B | .051 | .075 | .042 |
| Pair B | .053 | .083 | .045 |
| DEV4 | .051 | .076 | .041 |

### Own-track recovery and validation

Recovery r is the Pearson r of each cell type's fibers against its own track, from the same paired evals. The
other columns give validation r at the last epoch and the r on mixture composites (`eval26` / `eval30`).

| Model | Recovery r, own pair | Recovery r, unseen pair | Val r | Mixture composite r |
|---|---|---|---|---|
| Mix A | GM12878 .466, HepG2 .615 | K562 .479, H1 .369 | .612 | .608 |
| Pair A | GM12878 .469, HepG2 .610 | K562 .473, H1 .370 | .566 | – |
| Mix B | K562 .467, H1 .441 | GM12878 .392, HepG2 .514 | .559 | .518 |
| Pair B | K562 .522, H1 .438 | GM12878 .398, HepG2 .554 | .530 | – |
| DEV4 | GM12878 .458, HepG2 .586, K562 .509, H1 .419 | – | – | – |

Mixture models validate on mixtures and pair models on own tracks, so their validation r values don't compare
directly.

### Findings

- **Mixture-only supervision reaches the ceiling on both pairs.**
  - On its own pair, each mixture model's fiber-origin AUC matches the own-track model's: Mix A 0.637 vs Pair A
    0.635, and Mix B 0.666 vs Pair B 0.665.
  - Sign accuracy and differential r are slightly lower (−0.01 to −0.04).
  - Recovery of each cell's own track is the same, except K562 under Mix B (.467 vs .522).
  - This repeats October's T3 result on a second, harder pair.
- **On the unseen pair, mixture and own-track models trained on the same cells perform the same** (Mix A ≈ Pair A on
  K562–H1, Mix B ≈ Pair B on GM12878–HepG2). The loss comes from never having seen the cell types, not from the
  kind of supervision.
- **The drop on the unseen pair is lopsided.**
  - K562 + H1 → GM12878–HepG2 loses little: AUC 0.62 vs 0.64.
  - GM12878 + HepG2 → K562–H1 loses a lot: AUC 0.60 vs 0.67, and sign accuracy 0.59, close to chance.
  - The K562–H1 loss is in H3K36me3 (0.47 vs 0.69), H3K9me3 (0.50 vs 0.62–0.64) and H3K27ac (0.56–0.59 vs
    0.65–0.68).
  - H3K27me3 and H3K4me1 transfer fully (~0.79, ~0.72).
  - A plausible reason is that GM12878 and HepG2 offer nothing like H1's stem-cell chromatin to learn from.
- **DEV4 sits just below each pair's own models** (AUC 0.629 vs 0.635–0.637; 0.654 vs 0.665–0.666). This matches
  the specialists beating the multi-cell model above.
- **Some assays fail for every model, including those trained on the pair:**
  - **H3K9me3 on GM12878–HepG2** (AUC 0.46–0.50). Almost all of its 2,952 bins are higher in HepG2: this is the
    HepG2 H3K9me3 track's scale issue, not single-molecule signal.
  - **CTCF on K562–H1** (0.46–0.48).
  - **H3K36me3 on GM12878–HepG2** (0.49–0.55).
- **ATAC fiber-origin AUC is a weak readout.** Mix B never trained an ATAC output, yet reaches AUC 0.705 on
  GM12878–HepG2, as high as the models that did. Accessibility is in the input, so the cell types separate on ATAC
  almost for free. Differential r does show the missing training (.37 vs .57–.61).
- **Library effects are small next to cell-type effects.** Same-cell-type library pairs give |AUC − 0.5| ≈ 0.04–0.05
  (HepG2, H1) and 0.08–0.09 (K562, whose two libraries differ most), against 0.17–0.25 for the real pairs. The
  GM12878–HepG2 pair crosses Fiber-seq experiments and still sits well above the controls.

**Next (not run yet):** routed specialists as the "told the cell type" reference on the unseen pair.

**Sources:** training and eval logs `logs/2026-11-04_*` on nibi (paired evals that timed out were rerun as
`2026-11-04_{15..18}_eval_*_rerun`); results `results/26-11-04_*_eval_{MIX_V1,PAIR_V1,LOO_Panc1_V1}_*`
(`<config>_paired.npz`, `eval_summary.json`); table script `ignore/reports/mix_compare.py`.
