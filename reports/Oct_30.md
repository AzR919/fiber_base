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
  - `../plots/26_oct_30/rest/fire_msp_GM12878_03_chr21_34580155-34585155_h3k27ac-h3k4me3.png`
  - `../plots/26_oct_30/rest/fire_msp_GM12878_07_chr21_45108980-45113980_h3k27ac-h3k4me3.png`
  - `../plots/26_oct_30/rest/fire_msp_K562_02_chr21_33400612-33405612_h3k27ac-h3k4me3.png`
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
- **Figures** (`../plots/26_oct_30/rest/`, code: `plots_code/26_oct_30/sharp_broad_bars.py`, from the run's
  `eval_summary.json`), one panel per group, one bar group per cell type plus the mean:
  - `sharp_broad_pearson.png`: model vs baseline. Sharp bars sit level with the baseline; broad bars show the gap.
  - `sharp_broad_mse.png`: model vs baseline. The model is lower almost everywhere, in both groups.
  - `sharp_broad_cell_r.png`: model only (the baseline is 0 by definition). Held-out cell types sit with the dev
    ones on sharp marks; broad marks are lower but positive everywhere.
- Caveat: these numbers use the earlier fiber averaging and baseline (both since updated); a rerun is pending.
  The picture is not expected to change.

#### Single-molecule examples (dev and held-out)

- **How loci were found:**
  - Data: chr21 test tiles. Each 1 kb bin is scored as the cell's ENCODE signal minus the cross-cell average
    (asinh FC).
  - Categories:
    - **gain**: the cell is ≥ 0.5 above the average;
    - **loss**: the cell is ≥ 0.5 below the average;
    - **agree**: within 0.25 of the average, at a peak ≥ 1.5.
  - The top candidates per cell type were run through the model and ranked by:
    - how much of the gap to the average the model recovers;
    - the model's r over the 5 kb tile.
  - Plots: per cell type and group, the best few plus random picks. 224 figures in total, all on fir in
    `results/26-10-28_T21-22-02_single_molecule_dev5/` (latest version; ranking from `..._T06-06-09_...`).
  - Code: `plots_code/26_oct_30/single_molecule_dev5.py`.
- **Each figure:**
  - Two panels: the flagged assay and its partner (ATAC↔CTCF, H3K4me3↔H3K27ac, H3K4me1↔H3K36me3,
    H3K27me3↔H3K9me3; for H1/H9, which have no ATAC, CTCF is paired with H3K27ac).
  - Top: the ENCODE track, the model's average over fibers, and the baseline, with the tile MSE of each (scored
    positions, ≥ 5 fibers).
  - The 5 kb window is centred on the flagged bin.
  - Then a heatmap of every fiber's prediction, sorted by the flagged bin.
  - Then five fibers across that ranking (preferring fibers that cover the window on both sides of the bin), with
    their inputs (m6A, MSP, nucleosomes, FIRE). Each fiber's prediction
    is drawn against the baseline (dashed): a bulk baseline gives every fiber that same curve, while the model gives
    each fiber its own.
- **Over all 799 ranked candidates** (not just the plotted ones):

  | | Model r > baseline r | Gap recovered (median) |
  |---|---|---|
  | Sharp gain (dev / held-out) | 84% / 81% | 29% / 28% |
  | Broad gain (dev / held-out) | 71% / 72% | 14% / 14% |
  | Loss, sharp and broad | 45–70% | 81–95% |
  | Agree | 13–23% | — |

  - At gain loci the model beats the baseline on shape in most cases, held-out cell types included, but recovers
    only part of the height (more for sharp marks).
  - At loss loci it mostly removes the peak the baseline wrongly predicts.
  - Where the cell matches the average, the baseline is already right and is hard to beat (r 0.88 vs 0.92 for
    sharp marks), but the model is close.
- **Figures to show** (`../plots/26_oct_30/sm/`; tile MSE is model vs baseline, flagged assay first, then the
  partner):
  - **Gain (the cell has signal the average lacks):**
    - `sm_test_mcf-7_atac_gain.png` (MCF-7, held out; ATAC 0.14 vs 1.07):
      - The model finds an ATAC peak that the average does not have.
      - The heatmap shows which fibers are open; the closed bottom fiber predicts ~0.
      - CTCF is over-called at the same site (0.43 vs 0.26): CTCF follows accessibility.
    - `sm_test_mcf-7_h3k4me3_gain.png` (MCF-7; H3K4me3 3.69 vs 9.56, H3K27ac 0.61 vs 2.78):
      - A wide active domain the average misses. The model has the extent but about half the height.
      - The fibers split into marked and unmarked; the bottom fiber is ~0.
    - `sm_test_hct116_h3k4me3_gain.png` (HCT116, held out; 0.15 vs 1.30, H3K27ac 1.42 vs 3.30):
      - The promoter is called on most fibers.
      - The model over-calls the downstream half; H3K27ac is under-called but still closer than the baseline.
    - `sm_dev_hepg2_h3k4me3_gain.png` (HepG2, dev; 0.63 vs 3.44, H3K27ac 1.42 vs 2.65):
      - The model follows the promoter domain, below its full height (~2.5 vs ~3.5).
      - The bottom fiber is almost fully nucleosome-covered and predicts nothing: a silent copy.
    - `sm_dev_k562_h3k4me1_gain.png` (K562, dev; broad mark, 0.67 vs 2.36):
      - The model rises over the domain, below its full height.
      - Smooth gradient across fibers, down to a fully nucleosome-covered fiber at ~0.
  - **Loss (the average has signal the cell lacks):**
    - `sm_dev_gm12878_atac_loss.png` (GM12878, dev; ATAC 0.02 vs 0.82):
      - The model removes the ATAC peak the average puts there.
      - A few fibers with a FIRE element at the site still light up: rare open copies that average out in the bulk.
      - It also drops the CTCF peak, which GM12878 partly has (0.25 vs 0.79).
    - `sm_test_h9_h3k4me3_loss.png` (H9, held out; 0.11 vs 2.65, H3K27ac 0.23 vs 1.15):
      - A promoter active in the other cell types but weak in H9; the model stays low on both marks.
      - Only the top few fibers (with FIRE elements) carry signal.
    - `sm_test_h9_h3k4me1_loss.png` (H9; broad, 0.09 vs 2.27):
      - The model stays at H9's low level instead of the average's ~2.
      - H3K36me3 is over-called (0.32 vs 0.14).
  - **Agree (the cell matches the average):**
    - `sm_test_mcf-7_atac_agree.png` (MCF-7; ATAC 0.145 vs 0.057, CTCF 0.28 vs 0.53):
      - Peak shape and height match.
      - The baseline is slightly closer on ATAC; the model is closer on CTCF.
      - The closed bottom fiber predicts ~0 where the bulk peak is.
    - `sm_dev_h1_h3k4me3_agree.png` (H1, dev; 0.22 vs 0.15, H3K27ac 0.99 vs 1.59):
      - The bulk is matched, yet single fibers differ: the bottom fiber is marked only in patches.
    - `sm_test_h9_h3k4me1_agree.png` (H9; broad, 0.11 vs 0.11):
      - The bulk is matched, and the fibers again vary.
      - A fiber with a long accessible stretch (FIRE) at the bin predicts a dip there: no nucleosome, no histone
        mark.
      - A fully nucleosome-covered fiber predicts ~0.
  - **Failure:** `sm_test_hct116_h3k27me3_gain.png` (HCT116, a random pick; 0.80 vs 0.72):
    - The H3K27me3 domain is missed.
    - The prediction is flat and uniform across fibers: the broad-mark weakness from Table 2.
  - **Pattern in the "agree" figures:**
    - The bulk is no better than the baseline (expected: the baseline is already right there).
    - The fibers still differ, which is information the baseline cannot give.

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
- **Bar charts** (`../plots/26_oct_30/t3/`, code: `plots_code/26_oct_30/t3_bars.py`, from both runs' eval outputs):
  - `t3_summary_bars.png`: Table 4 as bars (differential r per window, sign accuracy, fiber-origin AUC; sharp,
    broad, all 8; mixture vs upper bound; H0 as a dashed line).
  - `t3_per_assay_bars.png`: the same three readouts per assay. The mixture model is above H0 on every assay and
    readout; the upper bound's H3K9me3 sign accuracy (0.35) and AUC (0.47) fall below it.
  - `t3_library_bars.png`: fiber-origin |AUC − 0.5| per assay, GM12878 vs HepG2 (0.10–0.23) next to the library
    control (0.04–0.05 on every assay).
- **Figures** (copied from `../plots/26_oct_23_t3/` to `../plots/26_oct_30/t3/` with a `t3_` prefix):
  - `t3_mix_paired_locus_0.png`: a GM12878-specific locus. Each cell type's prediction from its own fibers, the
    mixture (what H0 gives both), and every fiber's prediction: most GM12878 fibers light up at the H3K27ac /
    H3K4me3 peak, a minority stay dark, HepG2 fibers stay dark.
  - `t3_mix_differential_scatter.png`: per chr21 window, predicted vs true difference (H0: all on y = 0).
  - `t3_mix_fiber_auc.png` and `t3_mix_library_fiber_auc.png`: fiber-origin AUC per differential bin, pair vs library
    control.
- **Single-molecule examples** (`../plots/26_oct_30/t3/t3_sm_*.png`, code: `plots_code/26_oct_30/t3_single_molecule.py`).
  - What each figure shows:
    - The mixture model is run on each cell type's fibers alone, at chr21 bins where GM12878 and HepG2 differ.
    - Each figure has two panels: the differential assay and its partner.
    - Top of each panel: both ENCODE tracks, the 50:50 mixture (what H0 gives both cell types), and each cell type's
      predicted average.
    - Bottom: a heatmap of every fiber's prediction, GM12878 fibers above the cyan line and HepG2 below.
  - How they were picked:
    - "Best" is the top bin by fiber-origin AUC for each assay and direction.
    - "Random" is drawn from the 7,838 eligible bins (≥ 10 fibers per cell type).
    - All 24 are on fir in `results/26-10-27_T04-55-25_t3_single_molecule/`.
  - `t3_sm_best_atac_hepg2_high.png` (chr21:45.41 Mb, AUC 0.98):
    - HepG2 fibers open at the ATAC peak; GM12878 fibers stay closed.
    - CTCF (partner) follows the same split.
  - `t3_sm_best_h3k27ac_gm_high.png` (chr21:14.76 Mb, AUC 0.95):
    - The GM12878-only H3K27ac / H3K4me3 domain is predicted on GM12878 fibers only.
    - Within GM12878, about half the fibers carry it, which is the per-molecule heterogeneity we want.
  - `t3_sm_best_h3k4me1_hepg2_high.png` (AUC 0.98) is a broad mark that works: a 5 kb HepG2 H3K4me1 domain, with
    the GM12878 average flat near 0.
  - `t3_sm_best_h3k27me3_hepg2_high.png` (AUC 0.94):
    - This is a repressive mark that works: HepG2 H3K27me3 is separated by fiber.
    - The predicted average undershoots the track, at ~1 vs ~2.
  - `t3_sm_random_h3k27ac_gm_high.png` (random pick, AUC 0.80) is a typical case: same pattern as the best H3K27ac
    example, a little noisier.
  - `t3_sm_random_h3k9me3_fail.png` (random pick, AUC 0.30) is a failure:
    - Both cell types get the same flat H3K9me3 prediction.
    - The fiber heatmap is driven by the shared sequence (vertical stripes), not by the fiber.
    - This matches H3K9me3 being the weakest assay in Table 4.
- Caveats: one training run per model (second seed pending); the AUC checks separation by cell of origin, a proxy
  for "this molecule carries the mark"; GM12878 and HepG2 come from different Fiber-seq experiments (a same-series
  pair, e.g. K562 + HepG2, would remove that concern).

## Future directions

Full plan and IDs (T = test, I = implementation, M = model change) in `reports/TODO.md`.

### Immediate (next 1–2 weeks: firm up what is on these slides)

1. **Rerun the bulk numbers with the current setup:**
   - fiber averaging in FC (V1) and the FC-mean baseline;
   - this removes the caveat under Bulk recovery;
   - add the gain / loss loci breakdown to the eval (I3), which turns the single-molecule candidate table into a
     proper metric.
2. **Reference values for r**, so "0.5" can be read as good or bad:
   - replicate ceiling: ENCODE replicate 1 vs 2 per assay, with plain r and cell-type r (I11);
   - related-cell reference: H1's real track as the prediction for H9 (I12);
   - report our numbers as a fraction of the ceiling.
3. **Wrong-cell control for the cell-type r** (I5): score cell A's target with cell B's prediction. If that is also
   positive, the cell-type r is not cell-specific. Cheap, eval only.
4. **T3 robustness:**
   - a second seed for the mixture model and the upper bound;
   - the random-split noise floor;
   - a second pair from the same Fiber-seq series (K562 + HepG2), which rules out batch differences between
     GM12878 and HepG2.
5. **T3b, the mixture test on unseen cell types** (I15):
   - setup: the GM12878 / HepG2 mixture model, evaluated on fibers of other pairs (K562 / Panc1, H1 / K562);
   - why: rules out the shortcut T3 cannot, a model that only recognises the cell type and returns its bulk;
   - cost: eval only, on existing checkpoints.

### Next (model and test changes)

- **Broad marks** (the weak spot in Tables 2 and 4 and in the failure examples):
  - Use the whole read as context (M3): fibers are ~18 kb but cropped to the 5 kb window.
  - H3K36me3 sits on gene bodies, and H3K27me3 / H3K9me3 domains are wider than the window.
- **Outputs in physical units** (T4: V2 per-assay scale, then V3 / M1):
  - Predict per nucleosome and model the ChIP blurring once per assay, rather than painting a bulk-shaped peak
    onto every molecule.
- **Separating marks that travel together.** The examples show CTCF following accessibility and H3K27ac following
  H3K4me3. Possible fixes:
  - supervision that distinguishes molecules (M2): more cell types at the same loci, mixtures alongside single
    cell types, both libraries;
  - DNA as modulation rather than an extra input (M5);
  - same-molecule constraints (M6).
- **FIRE as an input** (M4): the fire_msp model treats FIRE as a switch. Feed continuous FIRE scores and m6A / CpG
  call probabilities instead of thresholded channels.
- **Single-molecule ground truth within a cell type** (T5):
  - T5a: a no-CpG model whose per-fiber predictions are checked against the fiber's measured CpG methylation;
  - T5b: X inactivation in female lines (two populations of molecules at X-linked promoters);
  - T5c: imprinted loci.
- **Comparison ladder:**
  - single-molecule references (I10, after saving predictions, I2): a rule-based "accessible = active" imputer and
    "bulk on every molecule" (H0);
  - bulk references: a DNA-only model and CANDI (on ATAC and on aggregated Fiber-seq accessibility).
- **Engineering:** training throughput (bigWig cache, less frequent validation, more training chromosomes), so the
  runs above are cheaper.
