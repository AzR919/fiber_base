# 30th Oct

Notes for the presentation: each section below is one or two slides.

## Setup

- **Regions:** train on chr20, validate on chr22 during training, test on chr21. No chromosome is used for two
  purposes.
  - Training samples are drawn half from cCREs and half from anywhere (background).
  - Test: all 6,843 valid 5 kb tiles of chr21 (cCREs + background). Results can be sliced by region type
    (promoter, enhancer, CTCF, non-cCRE).
- **Targets:** ENCODE fold change over control. Fold change is linear in the number of marked molecules, which
  matches how the model builds the bulk prediction (an average over molecules).
- **Cell types and assays:** 5 development cell types, 3 held out; 8 assays (see "Bulk recovery" under Progress).

## Previous results

Just show a few previous results.

## Progress

### FIRE-MSP contribution

- From `Oct_16.md`: two models identical except for the fire_msp input (old 4-assay setup, trained on GM12878 +
  K562 + HepG2, scored on chr21 cCREs). Dropping fire_msp changes the loss very little, slightly in its favour
  (average MSE −0.014 GM12878, −0.017 K562, +0.013 HepG2).

  Average MSE over the 4 assays (ATAC, H3K4me3, H3K27ac, H3K27me3), chr21 cCREs; **bold** = lower.

  | Model | GM12878 | K562 | HepG2 |
  |---|---|---|---|
  | With fire_msp | 0.344 | 0.587 | **0.412** |
  | Without fire_msp | **0.330** | **0.570** | 0.425 |
  | Δ (without − with) | −0.014 | −0.017 | +0.013 |

- Single-molecule figures (H3K27ac and H3K4me3; ENCODE track, with fire_msp in red, without in blue; 7 fibers with
  their inputs and both models' per-fiber predictions):
  - `figures/oct30/fire_msp_GM12878_03_chr21_34580155-34585155_h3k27ac-h3k4me3.png`
  - `figures/oct30/fire_msp_GM12878_07_chr21_45108980-45113980_h3k27ac-h3k4me3.png`
  - `figures/oct30/fire_msp_K562_02_chr21_33400612-33405612_h3k27ac-h3k4me3.png`
- Takeaway:
  - Both models agree on the FIRE-bearing fibers: they put the marks around the called FIRE elements, even the
    model that never sees fire_msp (GM 03). Fully closed fibers get nothing from either.
  - Where they differ is open fibers with no FIRE call. The model with fire_msp treats the FIRE call as a switch:
    no call, (almost) no signal. The model without it reads accessibility directly and marks open stretches the
    FIRE caller missed. Clearest case: GM 07 fiber #30, a long accessible patch under the bulk peak with no FIRE
    call, gets ~5–6 without fire_msp and ~0 with it. The same happens next to called elements (#59, #91).
  - FIRE calls are a strict, thresholded summary of the m6A / accessible-patch channels, so the model can get the
    same information from the other inputs, and without the shortcut it learns to.
  - The bulk cannot tell the two apart: "more signal on fewer fibers" and "less on more" fit it about equally,
    which matches the tiny loss difference. Which one is right per molecule needs a test with known answers (T3
    mixture), not the bulk loss.

### Bulk recovery on the dev split

(100-epoch dev model, all chr21 tiles; numbers from `Oct_23.md` and the `2026-10-18_04` eval log.)

**Cell types:** trained on the 5 dev cell types only; scored on all 8. The held-out three are new cell types the
model has never seen.

| Group | Cell type | Tissue | ENCODE assays | Role |
|---|---|---|---|---|
| Dev | GM12878 | B cell (blood) | all 8 | comparability with earlier runs |
| Dev | K562 | myeloid leukemia (blood) | all 8 | comparability with earlier runs |
| Dev | HepG2 | liver carcinoma | all 8 | comparability with earlier runs |
| Dev | Panc1 | pancreatic carcinoma | all 8 | noisier broad-mark tracks: fine to train on, poor to test on |
| Dev | H1 | embryonic stem cell | no ATAC | brings stem-cell chromatin into training |
| Held out | H9 | embryonic stem cell | no ATAC | **near**: twin of H1 |
| Held out | HCT116 | colon carcinoma | all 8 | **medium**: related tissue (gut-derived, like HepG2 / Panc1) |
| Held out | MCF-7 | breast carcinoma | all 8 | **far**: no close relative in dev |

- Held-out cell types range from near to far, so we see how performance falls off as a new cell type gets less
  familiar, not just one number.
- Held-out cell types are never used for training or for any decision (architecture, hyperparameters, checkpoint).
- HCT116 and MCF-7 have the cleanest ENCODE tracks (newest, few audit flags), and only one Fiber-seq sample each,
  so holding them out costs the least training data.
- Same rule for the cell-specific loci: dev loci compare each dev cell type with the other 4 dev ones; test loci
  (all 8) are only used for milestone reports.

**Assays:** 8 ENCODE tracks per cell type. Sharp / broad follows ENCODE's peak calling (narrow vs broad peaks).

| Assay | What it marks | Shape |
|---|---|---|
| ATAC-seq | open (accessible) DNA | sharp |
| CTCF | CTCF binding: insulators, loop anchors | sharp |
| H3K4me3 | active promoters | sharp |
| H3K27ac | active promoters and enhancers | sharp |
| H3K4me1 | enhancers (active and primed) | broad |
| H3K36me3 | bodies of transcribed genes | broad |
| H3K27me3 | Polycomb-repressed regions | broad |
| H3K9me3 | heterochromatin (constitutive repression) | broad |

- ENCODE has no ATAC-seq for H1 and H9, so their ATAC is left out of the loss and the scores: their averages are
  over 7 assays instead of 8.

**Metrics** (all at base-pair resolution on asinh(fold change), positions covered by ≥ 5 fibers)
- **Pearson r:** is the *shape* along the genome right? Insensitive to scale.
- **MSE:** are the signal *levels* right?
- **Baseline:** for each cell type, the average of the *other* dev cell types' tracks. It knows nothing about the
  cell, but it is strong because most of every track is shared across cell types.
- **Cell-type r:** where this cell type differs from the average cell type, does the prediction differ the same
  way? Partial correlation of prediction and target after regressing out the baseline. 0 = nothing beyond the
  average (the baseline scores exactly 0). The DNA input is identical across cell types, so a positive value
  means the model reads cell-specific information from the fibers.

**Table 1: mean over assays.** Model / baseline; **bold** = better of the two.

| Cell type | Pearson r | MSE | Cell-type r |
|---|---|---|---|
| GM12878 | 0.43 / **0.44** | **0.228** / 0.258 | +0.26 |
| K562 | 0.50 / **0.51** | **0.172** / 0.191 | +0.32 |
| HepG2 | **0.57** / 0.54 | **0.232** / 0.244 | +0.41 |
| Panc1 | 0.40 / **0.49** | **0.194** / 0.195 | +0.21 |
| H1 | **0.40** / 0.39 | **0.261** / 0.283 | +0.28 |
| H9 *(near)* | 0.38 / **0.47** | **0.177** / 0.182 | +0.19 |
| HCT116 *(medium)* | 0.47 / **0.56** | 0.177 / **0.163** | +0.26 |
| MCF-7 *(far)* | 0.45 / **0.51** | **0.162** / 0.166 | +0.27 |
| **Mean** | 0.45 / **0.49** | **0.200** / 0.210 | +0.27 |

- **MSE:** the model beats the baseline in 7 of 8 cell types (all but HCT116).
- **Pearson:** the baseline wins in 6 of 8, including all three held-out cell types.
- **Cell-type r:** positive everywhere, and as high in the held-out cell types (+0.19 to +0.27) as in several dev
  ones (+0.21 to +0.41). No clear near-to-far trend (H9 is not better than MCF-7).

**Table 2: sharp vs broad marks.** Same numbers, averaged within each group of assays (sharp / broad as in the
assay table). H1 and H9 have 3 sharp assays (no ATAC). Model / baseline; **bold** = better of the two.

| Cell type | Sharp: Pearson | Sharp: MSE | Sharp: cell-type r | Broad: Pearson | Broad: MSE | Broad: cell-type r |
|---|---|---|---|---|---|---|
| GM12878 | **0.50** / 0.46 | **0.178** / 0.231 | +0.31 | 0.36 / **0.42** | **0.277** / 0.285 | +0.21 |
| K562 | 0.52 / 0.52 | **0.157** / 0.179 | +0.32 | 0.47 / **0.50** | **0.186** / 0.203 | +0.32 |
| HepG2 | **0.56** / 0.53 | **0.159** / 0.177 | +0.38 | **0.59** / 0.54 | **0.304** / 0.310 | +0.45 |
| Panc1 | 0.53 / **0.57** | 0.163 / **0.157** | +0.31 | 0.27 / **0.40** | **0.224** / 0.232 | +0.11 |
| H1 | **0.51** / 0.43 | **0.186** / 0.220 | +0.38 | 0.33 / **0.36** | **0.318** / 0.331 | +0.22 |
| H9 *(near)* | 0.45 / **0.51** | 0.155 / **0.154** | +0.22 | 0.32 / **0.44** | **0.194** / 0.204 | +0.16 |
| HCT116 *(medium)* | 0.55 / **0.57** | **0.137** / 0.144 | +0.32 | 0.39 / **0.55** | 0.217 / **0.182** | +0.21 |
| MCF-7 *(far)* | **0.53** / 0.51 | **0.165** / 0.170 | +0.33 | 0.37 / **0.50** | **0.160** / 0.162 | +0.21 |
| **Mean** | **0.52** / 0.51 | **0.162** / 0.179 | +0.32 | 0.39 / **0.47** | **0.235** / 0.239 | +0.23 |

- **Sharp marks:** the model is level with the baseline on Pearson (0.52 vs 0.51 on average; ahead or tied in
  5 of 8) and clearly better on MSE (0.162 vs 0.179; 6 of 8, H9 a tie). Cell-type r is +0.22 to +0.38, and the
  held-out cell types (+0.22 to +0.33) look like the dev ones.
- **Broad marks:** almost the whole Pearson gap is here (0.39 vs 0.47; the baseline wins in 7 of 8). MSE is still
  slightly better than the baseline (7 of 8), and cell-type r is lower (+0.23 on average) but still positive.
- **Message:** on the marks a molecule's accessibility actually reflects, the model matches the baseline's shape,
  gets the levels closer and adds cell-specific signal in new cell types. The weak spot is broad domains,
  especially H3K36me3 (gene bodies) and H3K27me3 / H3K9me3, which are wider than the 5 kb window; longer DNA
  context is the likely fix. H3K4me1 is the exception among broad marks (it tracks enhancers and does well).
- **Figures** (code: `plots_code/26_oct_30/sharp_broad_bars.py`, from the run's `eval_summary.json`), one panel per
  group, one bar group per cell type plus the mean:
  - `figures/oct30/sharp_broad_pearson.png`: model vs baseline. Sharp bars sit level with the baseline; broad bars
    show the gap.
  - `figures/oct30/sharp_broad_mse.png`: model vs baseline. The model is lower almost everywhere, in both groups.
  - `figures/oct30/sharp_broad_cell_r.png`: model only (the baseline is 0 by definition). Held-out cell types sit
    with the dev ones on sharp marks; broad marks are lower but positive everywhere.
- Caveat: these numbers use the earlier fiber averaging and baseline (both since updated); a rerun is pending.
  The picture is not expected to change.

### Mixture test (T3): are single molecules resolved?

(Details: `Oct_23_T3.md`; design: `TODO.md`, T3.)

**The question.** The model only ever sees bulk averages as targets. It could fit those while giving every molecule
roughly the same bulk-like value ("bulk on every molecule", the null hypothesis H0). No bulk metric can tell the
two apart, so we need a test where the answer per group of molecules is known.

**Setup**
- **Mixture model:** trained only on mixtures of GM12878 and HepG2 (B cell vs liver, very different chromatin).
  Each training sample pools 100 fibers from both cell types at the same locus, GM12878's share drawn between 20%
  and 80%; the target is the mixture's track (the two tracks mixed in fold change, weighted by the fibers present).
  Neither cell type's own track is ever used, in training or validation.
- **Test:** give the model one cell type's fibers alone and ask whether it returns *that* cell type's track. Under
  H0 both cell types get the mixture; a model that reads each molecule can split them, because fibers are processed
  independently (no pooling across fibers).
- **Upper bound:** the same model trained the usual way on the same two cell types, each on its own track.
- Same architecture as the bulk results (mid 8-assay UNet), chr20 training, 25 epochs, fibers averaged in fold
  change (V1). Test on all chr21 tiles (~6,800 per cell type).
- **Library control:** HepG2 has two Fiber-seq libraries (100U and 200U) with one true track, so any difference the
  model predicts between them is a library effect, not biology.

**Metrics** (each has a known value under H0)
- **Differential r:** correlation between the predicted difference (GM12878 prediction − HepG2 prediction) and the
  true difference of the tracks, per base pair or per 5 kb window. Range −1 to 1; **H0: 0** (both get the same
  prediction, so the predicted difference is flat).
- **Sign accuracy:** at the differential 1 kb bins (where the two tracks differ by ≥ 0.5 in asinh FC), how often the
  predicted difference has the right sign. Range 0 to 1; **H0: 0.5**.
- **Fiber-origin AUC:** at those bins, pick one GM12878 fiber and one HepG2 fiber; how often does the fiber from the
  cell type with the mark score higher (each fiber's mean prediction over the bin)? Range 0 to 1; **H0: 0.5**. The
  most direct single-molecule readout: it is about individual fibers, not averages.
- **Library control:** the same AUC between HepG2 100U and 200U fibers, reported as |AUC − 0.5|. Should stay at
  sampling noise (≈ 0.05).

**Table 3: mean over the 8 assays.** Mixture model / upper bound; **bold** = better of the two.

| Readout | H0 | Mixture model / upper bound |
|---|---|---|
| Differential r, per 5 kb window | 0 | **0.49** / 0.47 |
| Differential r, per base pair | 0 | **0.31** / 0.30 |
| Sign accuracy at differential bins | 0.50 | **0.78** / 0.74 |
| Fiber-origin AUC | 0.50 | **0.64** / 0.60 |
| Library control, \|AUC − 0.5\| | ≈ 0.05 | 0.044 / 0.049 |

**Table 4: sharp vs broad marks.** Same readouts, averaged within each group (as in the assay table).

| Readout | H0 | Sharp: mixture / upper | Broad: mixture / upper |
|---|---|---|---|
| Differential r, per 5 kb window | 0 | 0.50 / **0.55** | **0.47** / 0.39 |
| Differential r, per base pair | 0 | 0.33 / **0.35** | **0.29** / 0.26 |
| Sign accuracy | 0.50 | 0.87 / 0.87 | **0.70** / 0.62 |
| Fiber-origin AUC | 0.50 | **0.68** / 0.66 | **0.59** / 0.55 |
| Library control, \|AUC − 0.5\| | ≈ 0.05 | 0.043 / 0.046 | 0.045 / 0.052 |

- **Single molecules are resolved without single-molecule supervision.** Every readout is well clear of H0: the
  predicted difference follows the true one (window r 0.49), has the right sign at 78% of differential bins (87%
  on sharp marks, 96% for ATAC), and one fiber's prediction tells its cell of origin (AUC 0.64; 0.68 on sharp
  marks).
- **Mixture supervision loses nothing:** the mixture model matches the upper bound trained on each cell type's own
  track (sharp marks level, broad marks slightly ahead; within run-to-run noise).
- **Sharp marks work best, broad marks are weaker but not at H0:** broad AUC 0.59 is carried by H3K4me1 (0.68) and
  H3K27me3 (0.61); H3K36me3 and H3K9me3 sit at 0.54.
- **It is biology, not the library:** the separation flips with the direction of the difference (e.g. ATAC: the chance a
  GM12878 fiber scores above a HepG2 fiber is 0.65 where GM12878's track is higher, 0.23 where HepG2's is), and two libraries of one cell type are
  not told apart (|AUC − 0.5| 0.044 vs 0.175 for GM12878 vs HepG2).
- **Bar charts** (`figures/oct30/`, code: `plots_code/26_oct_30/t3_bars.py`, from both runs' eval outputs):
  - `t3_summary_bars.png`: Table 4 as bars (differential r per window, sign accuracy, fiber-origin AUC; sharp,
    broad, all 8; mixture vs upper bound; H0 as a dashed line).
  - `t3_per_assay_bars.png`: the same three readouts per assay. The mixture model is above H0 on every assay and
    readout; the upper bound's H3K9me3 sign accuracy (0.35) and AUC (0.47) fall below it.
  - `t3_library_bars.png`: fiber-origin |AUC − 0.5| per assay, GM12878 vs HepG2 (0.10–0.23) next to the library
    control (0.04–0.05 on every assay).
- **Figures** (copied from `figures/oct23_t3/` to `figures/oct30/` with a `t3_` prefix):
  - `t3_mix_paired_locus_0.png`: a GM12878-specific locus. Each cell type's prediction from its own fibers, the
    mixture (what H0 gives both), and every fiber's prediction: most GM12878 fibers light up at the H3K27ac /
    H3K4me3 peak, a minority stay dark, HepG2 fibers stay dark.
  - `t3_mix_differential_scatter.png`: per chr21 window, predicted vs true difference (H0: all on y = 0).
  - `t3_mix_fiber_auc.png` and `t3_mix_library_fiber_auc.png`: fiber-origin AUC per differential bin, pair vs library
    control.
- Caveats: one training run per model (second seed pending); the AUC checks separation by cell of origin, a proxy
  for "this molecule carries the mark"; GM12878 and HepG2 come from different Fiber-seq experiments (a same-series
  pair, e.g. K562 + HepG2, would remove that concern).
