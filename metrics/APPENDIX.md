# Metrics appendix

Definitions, formulas and caveats for each metric in [README.md](README.md). Notation, per assay:
**t** = target (the evaluated cell's ENCODE track), **p** = model prediction (bulk: mean of the per-fiber
predictions), **b** = cross-cell-average baseline, all in asinh(fold change), at scored positions.

---

## A1. Scored positions

`score_weights(assay_mask, fiber_coverage, min_coverage)` → (B, K, L) weights:

- **assay_mask** (B, K): 1 if the sample's cell type has ENCODE data for the assay (H1/H9 have no ATAC).
- **fiber_coverage** (B, L): number of fibers whose *scored span* covers the position. A fiber's scored span is
  its alignment trimmed to [first nucleosome start, last nucleosome end] (read ends beyond them have neither
  MSP nor nucleosome calls, so their msp/nuc channels are uninformative). Padding rows cover nothing.
- A position counts when coverage ≥ `min_coverage` (default 5; `--min_coverage`): averages over 1–4 fibers are
  too noisy to score.

The model also zeroes per-fiber outputs outside each fiber's span, so those positions get no gradient and do not
enter the average. On chr21 tiles with 100 fibers per sample, well under 1% of positions fall below 5.

## A2. Training and validation loss

`masked_mse(pred, target, weights)` = Σ w·(p − t)² / Σ w over the batch. This is what backpropagates; assays a
cell lacks and low-coverage positions contribute nothing. Validation uses the same loss on the fixed
`val_chr22_n200` set every epoch (it drives the LR scheduler).

## A3. MSE

Pooled over every scored base pair of the window set, per assay: Σ w·(p − t)² / Σ w (`PearsonMeter.mse`; the
eval's `per_assay_loss` is the same quantity accumulated per batch). The composite is the mean over assays the
cell has.

- Measures **levels**: an offset or wrong scale costs MSE even if the shape is right.
- **Not comparable across assays or cells with different signal ranges.** E.g. HepG2's H3K9me3 track runs much
  higher than other cells', so every method has a large MSE there. Compare models on the same cell and assay.

## A4. Pearson r

`PearsonMeter`: one Pearson correlation per assay, **pooled over all scored base pairs of all windows**
(streaming, merged per batch with Chan's update in float64, so it is exact over tens of millions of positions).
Composite r = mean over the assays the cell has.

- Measures **shape**: invariant to scale and offset of the prediction.
- Pooled across windows, so it rewards getting **between-window** differences right (which windows are high)
  as well as the profile within a window; regions with large signal swings weigh more than quiet ones.
- `val_pearson` is the same quantity on the validation set (mean over assays).

## A5. Breakdown by region type

The same MSE and r, restricted to bases of one type (per base pair, from `strata/ccre_classes.bed`):

| Type | Bases |
|---|---|
| `all` | every scored base |
| `ccre` | inside any cCRE |
| `PLS` / `ELS` / `CTCF` / `other` | inside a promoter-like / enhancer-like (pELS+dELS) / CTCF-only / DNase-H3K4me3 cCRE |
| `non_ccre` | outside all cCREs |

A tile labelled "PLS" is still mostly non-cCRE bases; the per-bp masks, not tile labels, define these sets.
Types with few bases (CTCF, other: ~100k bases per cell on chr21) are noisier.

## A6. Per-window loss and r (dashboards)

Each dashboard panel shows the MSE and Pearson r of that one 5 kb window over its scored positions (shaded
regions excluded). Within-window r ignores the window's overall level and is noisy (one window, strongly
autocorrelated signal); use it to read plots, never to compare models.

## A7. Cross-cell-average baseline

For an eval sample (cell *c*, window), per assay: **b = mean over the training cell lines except c's own line
of asinh(their track)** at the same positions (`SingleCellFiberDataset.get_baseline`).

- Training cells come from the run being compared (`--baseline`: from the checkpoint; `--baseline_only`: from
  `--train_data_config`). One vote per cell line (K562_100U and K562_200U share a track). Lines without the
  assay are skipped; if none has it, the assay is not scored for the baseline.
- Excluding the evaluated line matters for training cells evaluated on held-out chromosomes: including it would
  hand the baseline part of the answer.
- Scored like a model, on the same samples and positions.
- It is strong because much of every track is shared across cell types; averaging more lines estimates that
  shared part better (the 5-cell baseline is clearly stronger than the 2-cell one).
- It depends only on the data, so it is computed once per training set (`--baseline_only`, CPU job) and reused.

## A8. Cell-specific r (partial correlation given the baseline)

**Question:** where this cell deviates from the typical cell, does the prediction deviate the same way?

**Definition.** Regress the baseline out of both target and prediction, then correlate what is left:

    t_res = t − (α + β·b)        (this cell's signal that the cross-cell average does not explain)
    p_res = p − (γ + δ·b)        (the part of the prediction that is not a rescaled average)
    r(p, t | b) = corr(p_res, t_res)

Computed without storing tracks, from three pooled correlations on the same positions
(`partial_correlation`):

    r(p, t | b) = (r_pt − r_bt · r_pb) / sqrt((1 − r_bt²) (1 − r_pb²))

**Worked picture.**

- Housekeeping promoter (b high, t high, p high): explained by the baseline, both residuals ≈ 0 → does not count.
- Enhancer active only in this cell (b low, t high → t_res > 0): if p is high there, p_res > 0 → counts *for* the model.
- Element active elsewhere but silent here (b high, t low → t_res < 0): predicting low → also counts for the model.
- The baseline itself, or any rescaled version of it: p_res = 0 everywhere → exactly 0.

**Reading the number.** 0 = nothing beyond the average; 1 = the cell-specific part predicted perfectly. Its
square is about the share of the variation left unexplained by the baseline that the model explains (the R² gain
from adding the model to the baseline in one linear fit): r = 0.36 → ~13%, r = 0.56 → ~31%.

**Why regression, not subtraction.** corr(p − b, t − b) shares the −b term on both sides, so even a constant
prediction scores high (−0.71 in a synthetic test; "baseline + noise" scored +0.15 instead of 0). Regression
fits the best scale and offset of b separately; on synthetic data the formula matches an explicit regression to
four decimals.

**What it shows.** The reference DNA is identical for every cell type; the fibers are the only cell-specific
input. A clearly positive value, especially for cells the model never saw, means the model reads cell-specific
chromatin information from the fibers.

**Caveats.**

- Population-level: scores the averaged prediction, not individual molecules.
- Part of a positive value can come from estimating the *shared* signal better than a 4–5-track average does
  (the baseline is a noisy estimate of the typical cell). Control: score one cell's target with *another* cell's
  prediction; if that also scores well, the gain is shared, not cell-specific. The pairwise test (A9) builds this
  control in.
- Linear and pooled over all positions.

## A9. Planned: pairwise differential tests (test 1)

For two cells A and B on the same windows, per assay and per base pair:

- **Differential correlation:** corr(p_A − p_B, t_A − t_B). The baseline (same value for both cells) scores 0.
- **Sign accuracy** at strongly differential windows (top 10% |t_A − t_B| per pair and assay).
- **Fiber-origin AUC:** at those windows, how well a single fiber's predicted mark level identifies which cell
  it came from (0.5 = chance). The per-molecule part of the test.

Stratified per pair and assay into A-high / B-high / shared-high / shared-low windows from the two targets.
Controls: the same cell line from two libraries (e.g. K562_100U vs K562_200U; true difference 0) and a random
split of one library (noise floor). Because the current model processes every fiber independently, a mixed
stack gives the same group averages as running each cell alone, so mixtures need not be built for it.
