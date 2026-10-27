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
- **Figure:** `figures/oct30/sharp_broad_pearson.png` (code: `plots_code/26_oct_30/sharp_broad_bars.py`). Pearson,
  model vs baseline, one panel per group: the sharp bars sit level with the baseline, the broad bars show the gap.
- Caveat: these numbers use the earlier fiber averaging and baseline (both since updated); a rerun is pending.
  The picture is not expected to change.
