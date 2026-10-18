# Research plan and TODO

What we want to show, how each test bears on it, and what to implement. Update the status lines as things run.
Metric definitions: `metrics/README.md` / `metrics/APPENDIX.md`. Background and motivation: the directed-reading
report, Chapter 4 (Fiber Imputation: Setup and Future Directions).

---

## 1. Research question and main hypothesis

**Question.** Does the chromatin accessibility of a single DNA molecule (Fiber-seq), together with its DNA
sequence, carry enough information to predict the histone modifications and other assays on that molecule?

**Main hypothesis (H).** A model that imputes each fiber independently, supervised only through the average over
fibers (weak, multiple-instance supervision), learns a mapping from a molecule's own accessibility to its own
histone state, rather than predicting the locus-level bulk on every molecule. The mapping needs no knowledge of
the cell type beyond its fibers, so it works out of the box for new cell types.

**Null (H0).** The per-fiber outputs are effectively a function of the locus (DNA, position) and the cell's
population-level signal; individual molecules at a locus get (nearly) the same prediction. Under H0 the bulk can
still be fit well, because the loss only constrains the average.

**Why this needs dedicated tests.** The training loss cannot distinguish H from H0: any change to the per-fiber
predictions that keeps every locus average fixed fits equally well. There is no molecule-level ground truth, so
each test below checks a consequence of H at the finest resolution where measurements exist, and is designed so
that a "bulk on every molecule" model fails it.

**Evaluation levels** (from coarse to fine):

1. Bulk recovery on training cell types (T1).
2. Bulk recovery on unseen cell types, and the cell-specific part of it (T2).
3. Recovery of individual cell types from a model supervised only on their mixture (T3), the main test of H.
4. Within-population structure: molecules of one cell type that should differ (T5).

Levels 1–3 test differences *between* populations of molecules; only level 4 tests differences *within* one.
None verifies an individual molecule; that would need single-molecule histone measurements (T8).

---

## 2. Where we stand (2026-10-18)

Setup for all numbers: split v1 (dev = GM12878, K562, HepG2, Panc1, H1; held out = HCT116, MCF-7, H9), 200U
samples, train on chr20, validate on chr22, test on all 6,843 valid chr21 tiles; mid 8-assay UNet (model15),
fibers trimmed to their nucleosome span, scoring at positions with ≥ 5 fibers. Results on fir under `results/`.

| Run | What | Key numbers (chr21, mean over assays) |
|---|---|---|
| `2026-10-17_06/07` GM_K5_ccre | GM12878 + K562, cCRE-only training, 25 epochs | Pearson 0.31–0.49 per cell; beats its 2-cell baseline in 4/8 cells |
| `2026-10-18_00/01` baselines | Cross-cell average of data26 / data20 training cells | data20 baseline Pearson 0.39–0.56 |
| `2026-10-17_08/09` DEV5 | data20 (5 dev cells, half cCRE / half random), 25 epochs | Pearson 0.36–0.51 (below the 5-cell baseline in every cell); MSE at or below the baseline in 6/8 cells |
| `2026-10-18_02` DEV5 + baseline | Same model, cell-specific r | **Partial r +0.18 to +0.36 in every cell, held-out cells +0.18 to +0.25**; strongest for ATAC, H3K4me1, H3K27ac, H3K4me3; H3K36me3 ≈ +0.05 |
| `2026-10-18_03/04` DEV5_long | Same, 100 epochs (train03), eval with baseline | running; val Pearson 0.48 at epoch 39 vs 0.44 final for 25 epochs |

Reading so far: bulk recovery works but does not beat the cross-cell average on Pearson (expected, see T1); the
model does carry cell-specific information that generalizes to unseen cells (T2, supports H over H0 at the
population level). Broad marks, especially H3K36me3, are the weak point.

---

## 3. Tests

Each test: hypothesis, design, readouts, controls, predictions (what supports H, what supports H0), status.

### T1. Bulk recovery

- **Hypothesis.** Aggregating imputed molecules recovers the bulk track, as the bulk is by definition the sum over
  molecules.
- **Design.** Coverage-normalized mean of per-fiber predictions vs ENCODE bulk, on test tiles of training and
  held-out cell types.
- **Readouts.** MSE, Pearson (genome-wide, by region type); vs the cross-cell-average baseline; vs a bulk model with
  the same information (T6).
- **Prediction.** Clearly positive recovery, but not necessarily above bulk methods: imputing one molecule at a
  time is a constraint, and the cross-cell average is hard to beat genome-wide (as in the ENCODE imputation
  challenge). Matching T6's bulk model would show the per-molecule constraint costs little.
- **Status.** Done for GM_K5_ccre and DEV5; DEV5_long pending. TODO: add the cell-type-specific-loci breakdown
  (I3) and the T6 reference.

### T2. Held-out cell types and the cell-specific part of the signal

- **Hypothesis.** The fiber → histone mapping generalizes to cell types never seen in training, including the part
  of their signal that differs from other cells.
- **Design.** Train on the 5 dev cells, evaluate on HCT116, MCF-7, H9 (and the dev cells on chr21).
- **Readouts.** Partial correlation of prediction and target given the cross-cell average (cell-specific r);
  plain r and MSE at cell-type-specific loci (I3), where the baseline is weakest.
- **Controls.** "Wrong cell" control: score cell A's target with cell B's prediction (given the baseline). If that
  scores as well, the gain is a better estimate of shared signal, not cell-specific information.
- **Prediction (H).** Cell-specific r clearly > 0 on held-out cells and close to the dev cells; the wrong-cell
  control near 0. **(H0)** cell-specific r ≈ 0, or the wrong-cell control as high as the right one.
- **Status.** Cell-specific r done for DEV5 (supports H); wrong-cell control and cell-specific-loci breakdown TODO.

### T3. Mixture-only supervision (main test of H)

- **Hypothesis.** Supervised only on the bulk of a mixture of two cell types, the model recovers each cell type's
  own track from that cell type's fibers, i.e. it resolves structure finer than its supervision.
- **Why it tests H vs H0.** Neither cell type's own track is ever used in training. A model predicting the
  (mixture) bulk on every molecule gives both populations the same output and cannot recover either track.
  Fibers are processed independently, so the model cannot pool across fibers to cheat.
- **Design.**
  - Train: in-silico mixtures of two cell types (e.g. GM12878 + HepG2; later more pairs and >2 cell types). Each
    sample pools fibers from both at the same locus; target = mixture track. **Mixing proportions vary across
    samples** (e.g. uniform in [0.2, 0.8]); with one fixed ratio, many splits of the signal between the two
    populations fit equally well, so varying proportions is what makes per-population signal identifiable.
  - Train on chr20, evaluate on chr21 (unseen loci).
  - Evaluate: apply the model to each cell type's fibers alone; compare with that cell type's own track.
  - First version with the current target scale (mixture = weighted mean of asinh tracks, as the code does
    now); then repeat with the scale fixes (T4) and compare.
- **Readouts.**
  - Per-cell recovery: r and MSE of each cell's prediction vs its own track, **compared with the mixture track's
    r and MSE against that cell** (the "bulk on every molecule" reference).
  - Differential correlation: corr(p_A − p_B, t_A − t_B); the reference scores 0.
  - Sign accuracy at strongly differential windows (top 10% |t_A − t_B| per assay).
  - Fiber-origin AUC at differential windows: can one fiber's predicted mark identify its cell type?
  - Reference points: the same readouts for a model trained on the two cell types separately (upper bound).
- **Controls.**
  - Same cell line, two libraries (K562_100U + K562_200U, HepG2, Panc1, H1): true tracks are identical, so any
    recovered difference is the model keying on library artefacts (e.g. m6A labeling density).
  - Random split of one library: noise floor for the differential readouts.
- **Prediction (H).** Each cell's prediction is closer to its own track than the mixture track is; differential
  correlation clearly > 0 and well above the controls; fiber-origin AUC > 0.5 at differential windows.
  **(H0)** predictions for both cell types ≈ the mixture track; differential correlation and controls ≈ 0.
- **Caveat.** The model may separate populations by library-level features (global m6A density, fiber length,
  global CpG methylation) and output a cell-typical pattern, without reading local accessibility. The
  two-library control quantifies this; pairs of cell types with similar library statistics reduce it.
- **Status.** Not started. Needs I1, I2 (and I4 for the second round).

### T4. Target scale and calibration (methodology comparison)

- **Hypothesis.** Aggregating per-fiber predictions in a space linear in the fraction of marked molecules
  (fold change), with cell-agnostic calibration, gives per-fiber outputs that resolve molecules better than the
  current setup (mean of per-fiber outputs compared directly with asinh(FC)).
- **Background.** The bulk is linear in the fraction of marked molecules; asinh(bulk) is not, so the current
  per-fiber outputs are not "this molecule's share of the bulk". The current mixture target (weighted mean of
  asinh tracks) has the same problem. Per-cell calibration is ruled out: prediction must need nothing about the
  cell beyond its fibers.
- **Variants to compare.**
  - V0 (current): softplus outputs, mean compared with asinh(FC), mixtures mixed in asinh space.
  - V1: aggregate in linear FC space, asinh of prediction and target only inside the loss; mixtures mixed in FC
    space, then transformed.
  - V2: V1 + sigmoid per-fiber outputs (probability of the mark) + one affine calibration per **assay** (shared by
    all cells), and/or per-track linear target normalization in data setup (a single scale per ENCODE track, from
    the track alone; choose carefully so real global differences are not removed).
- **Readouts.** T3's fiber-level readouts (main), plus T1/T2 bulk readouts (should not get worse).
- **Prediction.** V1/V2 improve T3's separation at equal or better bulk recovery.
- **Status.** Deferred until after T3 V0. Needs I4.

### T5. Within-population structure

Only these probe correctness *within* a cell type. Each uses a group of molecules defined independently of the
model with a known expected difference.

- **T5a. Held-out CpG methylation (same molecule).** Train a model without the CpG channel; the fiber's measured
  CpG methylation is then ground truth about that molecule the model never saw.
  - Hypotheses: per-molecule predicted H3K4me3 anti-correlates with the molecule's methylation at CpG-island
    promoters; predicted H3K36me3 correlates positively with gene-body methylation; H3K27me3 sits on unmethylated
    CpG islands of silent genes.
  - Readout: correlation across molecules at the same locus between predicted mark and held-out methylation.
    The baseline and any "bulk on every molecule" model score 0.
- **T5b. X inactivation.** In female lines (GM12878, MCF-7, H9; K562 less clear), molecules at X-linked promoters
  come from an active and an inactive X, so active-mark predictions should be **bimodal across molecules**; in
  male lines (HepG2, HCT116, H1, Panc1) unimodal. Needs no phasing. Needs a chrX window set (regions exclude chrX).
- **T5c. Imprinted loci** (H19/IGF2, SNRPN, PEG3, ...). Split fibers by haplotype (heterozygous SNPs, or HP tags
  if the CRAMs carry them): active marks should concentrate on one allele's molecules, repressive on the other's.
- Caveat for T5b/c: CpG methylation in the input partly encodes allele state; strongest with the no-CpG model.
- **Status.** Not started. T5a needs one extra training run; T5b a window set; T5c a phasing check.

### T6. Bulk reference model with the same information

- **Hypothesis.** Modelling molecules individually costs little at the bulk level relative to a model that sees
  the same accessibility aggregated over fibers.
- **Design.** Same UNet, input = per-position means of the fiber channels over fibers (+ DNA), output = bulk
  directly; same data, splits and metrics. Retraining published hybrid models (Enformer Celltyping, EPCOT) on
  aggregated Fiber-seq is the stronger reference but expensive; keep for later.
- **Status.** Not started.

### T7. Controls that the model reads each molecule

- Shuffle each fiber's inputs across loci: per-fiber predictions should break down while the bulk changes less.
- Permute which input belongs to which molecule within a window: predictions should follow the inputs.
- These do not prove correctness, but a model ignoring each molecule's data would fail them.
- **Status.** Not started; cheap once predictions can be saved (I2).

### T8. External single-molecule data (opportunistic)

- DiMeLo-seq or similar antibody-directed single-molecule histone data for one of our cell lines would be the
  closest to ground truth; single-cell CUT&Tag (e.g. K562) gives per-locus distributions to compare against the
  spread of per-molecule predictions. **TODO:** check availability.

---

## 4. Implementation TODO

Do not change the code base until the current runs (`2026-10-18_03_dev5_long`, `_04_..._eval_baseline`) finish.

| ID | Change | For | Notes |
|---|---|---|---|
| I1 | Mixture training with **variable proportions** per sample; mixture evaluator reporting per-cell-type group averages (apply to each cell's fibers alone) and the T3 readouts | T3 | `MixedCellFiberDataset` exists (fixed ratios, composite target); per-group reporting is not wired up in the current evaluator |
| I2 | `eval.py --save_predictions`: per sample bulk prediction, target, scored-position mask (float16), plus a per-fiber summary (mean predicted mark per assay over its span); offline analysis script for pairwise / mixture readouts | T3, T5, T7 | Full per-fiber tracks are too large to save |
| I3 | Cell-type-specific-loci breakdown in eval: windows where the target departs strongly from the cross-cell average (large \|t − b\|) | T1, T2 | Needs the baseline in the same pass |
| I4 | Target scale: aggregate in linear FC, asinh only in the loss; mix targets in FC space; optional sigmoid outputs + per-assay calibration; optional per-track target normalization in data setup | T4 | Changes training: rerun references after |
| I5 | Wrong-cell control for the cell-specific r | T2 | Score cell A's target with cell B's prediction |
| I6 | No-CpG model run (`input_flags` with cpg = 0) + molecule-level correlation with held-out methylation | T5a | One training run |
| I7 | chrX window set (regions spec) + bimodality analysis | T5b | Spec entry only for the windows |
| I8 | Check CRAMs for haplotype (HP) tags; phasing otherwise | T5c | |
| I9 | Aggregated-input bulk model | T6 | New model variant |

## 5. Engineering backlog (performance and quality)

- **bigWig cache:** training is limited by cold bigWig reads (~9 samples/s train vs ~23 val). Cache asinh target
  tracks for the chromosomes in use as float16 memory-mapped arrays (data setup); also makes the baseline cheap
  enough to run with every eval.
- **Fiber selection bias:** the first 100 fibers by start position are kept, so coverage is lower on the right of
  each window. Random (seeded) selection decodes all overlapping fibers (+50–90% fetch time); separate step.
- **Context and broad marks:** fibers are cropped to the 5 kb window before the model. Process longer input
  (full fiber or up to 20 kb; regions already allow it) and score the central region (centre-only scoring is
  not implemented yet). Likely needed for H3K36me3 / H3K27me3 domains.
- **Longer training / LR schedule:** 25 epochs was undertrained; check where DEV5_long saturates.
- **Eval summary keyed by cell type:** two eval configs for the same cell type in one run overwrite each other.
- **Mixed-mode evaluation:** composite target currently mixes asinh tracks (see T4).
- **Housekeeping:** October logs exist only on nibi (the fir sync's initial pull replaced local `logs/`); nibi
  test job 24330075 may still be queued there.
