# 23rd Oct: mixture test (T3)

Are single molecules resolved? The model is trained only on bulk averages, so the question is whether its
per-molecule output carries molecule-specific information. Setup of the pipeline: `Oct_19.md`; bulk results and
the fiber-averaging change: `Oct_23.md`. Design of the test: `TODO.md`, T3.

**Summary.** A model trained only on mixtures of GM12878 and HepG2 (never on either cell type's own track) gives
each cell type's molecules their own signal: the predicted difference between the two follows the true one
(window-level r 0.49), has the right sign at 78% of differential bins, and one fiber's prediction tells its cell
of origin (AUC 0.64). It matches a model trained on each cell type's own track, and the separation is
locus-specific, not a library artefact.

**The question.** The model is only ever trained on bulk averages. It could get those right while giving every
molecule roughly the same, bulk-like value ("bulk on every molecule", our null hypothesis H0). Then its
single-molecule output would be worthless, and no bulk metric would show it.

**The test.** Train a model on **mixtures only**: every training sample pools fibers from two cell types at the
same locus, and the target is the mixture's track. Neither cell type's own track is ever used, in training or in
validation. Then give the model one cell type's fibers alone and ask whether it returns *that* cell type's track.
A model that paints the bulk on every molecule cannot do this: both cell types get the mixture. A model that reads
each molecule can, because fibers are processed independently (no pooling across fibers).

**Setup**
 - Cell types: GM12878 (B cell) and HepG2 (liver), both dev cells with very different chromatin.
 - **Mixture model:** GM12878's share of each sample's 100 fibers drawn uniformly from 20–80% (varying the share
   is what makes each population's signal identifiable). Target per position: the two tracks mixed in fold change,
   weighted by each cell type's fibers present there.
 - **Upper bound:** the same model trained on the same two cell types the usual way (each on its own track).
 - Mid 8-assay UNet (`model15`), chr20 training regions, 25 × 1000 samples, fibers averaged in fold change (V1,
   `Oct_23.md` section 1). Test on all chr21 tiles (~6,800 per cell type), each cell type's fibers alone.
 - **Library control:** HepG2 has two Fiber-seq libraries (100U, 200U) with one true track, so any difference the
   model predicts between them is a library effect, not biology.

**Readouts** (each has a known value under H0)

| Readout | Question | H0 |
|---|---|---|
| Differential r | Where GM12878's track is above HepG2's, is the GM12878 prediction above the HepG2 one? Correlation of the two differences (per base pair, or per 5 kb window) | 0 |
| Sign accuracy | At the GM12878 vs HepG2 differential 1 kb bins (\|Δ asinh\| ≥ 0.5), does the predicted difference have the right sign? | 0.5 |
| Fiber-origin AUC | At those bins: take one GM12878 fiber and one HepG2 fiber; how often does the fiber from the cell type with the mark score higher? (per fiber: its mean prediction over the bin) | 0.5 |

The fiber-origin AUC is the most direct single-molecule readout: it is about individual fibers, not averages.

## Results

Mean over the 8 assays:

| | H0 | **Mixture model** | Upper bound |
|---|---|---|---|
| Differential r, per base pair | 0 | **0.31** | 0.30 |
| Differential r, per 5 kb window | 0 | **0.49** | 0.47 |
| Sign accuracy at differential bins | 0.50 | **0.78** | 0.74 |
| Fiber-origin AUC | 0.50 | **0.64** | 0.60 |
| Fiber-origin \|AUC − 0.5\|, library control (HepG2 100U vs 200U) | ≈ 0.05 (sampling) | 0.044 | 0.049 |
| Recovery of own track, Pearson r: GM12878 / HepG2 | | 0.42 / 0.57 | 0.40 / 0.53 |

Per assay, mixture model / upper bound:

| | ATAC | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|---|
| Differential r (window) | .55/.57 | .38/.46 | .63/.67 | .52/.36 | .77/.72 | .28/.10 | .32/.39 | .45/.50 |
| Sign accuracy | .96/.95 | .87/.86 | .85/.86 | .77/.67 | .89/.90 | .55/.55 | .59/.35 | .80/.81 |
| Fiber-origin AUC | .72/.70 | .67/.65 | .67/.66 | .61/.55 | .68/.65 | .54/.52 | .54/.47 | .64/.63 |
| Differential bins | 776 | 355 | 702 | 605 | 1,472 | 592 | 2,952 | 388 |

![Per-assay readouts: mixture model vs upper bound](../plots/26_oct_23_t3/t3_report.png)

**Window-level differences** (mixture model): each dot is a chr21 window, x = true GM12878 − HepG2 difference,
y = predicted difference from each cell type's own fibers. H0 would put every dot on y = 0.

![Differential scatter](../plots/26_oct_23_t3/mix_differential_scatter.png)

**Single molecules at a GM12878-specific locus** (mixture model): tracks of both cell types (light filled areas),
their 50:50 mixture (dashed: what H0 gives both), each cell type's prediction from its own fibers (bold lines,
darker shade of the same colour), and below, every fiber's
prediction, GM12878 fibers above the cyan line, HepG2 below, each block sorted by its value in the shaded bin.
At the H3K27ac / H3K4me3 peak most GM12878 fibers light up, a minority stay dark (molecules without the mark), and
HepG2 fibers stay dark.

![Paired locus](../plots/26_oct_23_t3/mix_paired_locus_0.png)

The four panels are the assays that differ most between the two cell types at this window. All 8 assays at the
same locus: [paired locus, all assays](../plots/26_oct_23_t3/mix_paired_locus_0_all.png).

A second locus shows the weak case: the H3K27ac and H3K4me1 differences are right, but for H3K36me3 (a broad
gene-body mark) the predicted difference has the wrong sign: [paired locus 2](../plots/26_oct_23_t3/mix_paired_locus_1.png).

**Fiber-origin AUC per differential bin** (mixture model) and the library control:

![Fiber-origin AUC](../plots/26_oct_23_t3/mix_fiber_auc.png)
![Library control AUC](../plots/26_oct_23_t3/mix_library_fiber_auc.png)

## Is the separation biology or library?

GM12878's Fiber-seq comes from a different experiment than HepG2's, so the model could in principle tell the two
apart by library-wide features (labeling density, read length) rather than by reading each locus. Two checks:

 1. **The separation follows the locus, not the library.** A library-wide effect would push all GM12878 fibers the
    same way everywhere. Instead the AUC flips with the direction of the difference. P(GM12878 fiber scores higher
    than HepG2 fiber), at bins where GM12878's track is higher | where HepG2's is higher:

    | | ATAC | H3K4me3 | H3K27ac | H3K4me1 | CTCF | H3K27me3 |
    |---|---|---|---|---|---|---|
    | GM12878 vs HepG2 | 0.65 \| 0.23 | 0.64 \| 0.31 | 0.60 \| 0.22 | 0.60 \| 0.28 | 0.54 \| 0.31 | 0.59 \| 0.37 |
    | HepG2 100U vs 200U | 0.49 \| 0.49 | 0.52 \| 0.51 | 0.52 \| 0.51 | 0.50 \| 0.51 | 0.50 \| 0.51 | 0.48 \| 0.47 |

 2. **Two libraries of the same cell type are not told apart.** HepG2 100U vs 200U fibers give AUC 0.47–0.53, about
    what sampling noise alone gives (mean |AUC − 0.5| 0.044 vs 0.175 for GM12878 vs HepG2). The one visible
    library effect is a small global offset on repressive marks (H3K27me3 predicted lower and H3K9me3 higher in
    100U).

## Findings

 - **Single molecules are resolved, without single-molecule supervision.** Trained only on mixture averages, the
   model gives each cell type's molecules their own signal: the predicted difference between the two cell types
   tracks the true one (window-level r 0.49), has the right sign at 78% of differential bins (96% for ATAC), and a
   single fiber's prediction tells its cell of origin (AUC 0.64; 0.67–0.72 for ATAC and the active marks). H0
   predicts 0, 0.5 and 0.5.
 - **Mixture supervision loses nothing.** The mixture model matches or slightly exceeds the upper bound trained on
   each cell type's own track on every mean readout (per assay the two trade places, mostly within a few
   hundredths). Supervision on mixtures with varying proportions is enough for the model to learn which molecules
   carry the signal.
 - **The separation is locus-specific, not a library artefact:** it reverses with the direction of the difference,
   and two libraries of one cell type are not separated.
 - **Active marks and accessibility work best** (ATAC, H3K4me1, H3K27ac, H3K4me3). **Broad and repressive marks are
   weak** (H3K36me3 AUC 0.54, H3K9me3 0.54), as in bulk; for H3K9me3 most "differential" bins reflect HepG2's
   globally higher track rather than locus-specific differences.

## Caveats and next steps
 - One training run per model; the mixture vs upper-bound gap is within run-to-run noise. Repeat with a second
   seed.
 - The AUC measures separation by cell of origin, a proxy: it does not check that a given molecule's mark is
   right, only that molecules from the cell type with the mark score higher.
 - Noise floor: predicted differences between two libraries (RMS 0.09 vs 0.20 for GM12878 vs HepG2) mix library
   effects with sampling different molecules. A random split of one library would separate the two.
 - Second pair from the same Fiber-seq series (e.g. K562 + HepG2) to remove the cross-experiment concern entirely.
 - The within-cell tests (T5: held-out CpG methylation, X inactivation) and the per-assay output scale (V2) are next
   on the plan (`reports/TODO.md`).
