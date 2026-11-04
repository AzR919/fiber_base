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
