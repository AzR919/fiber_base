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

## 2. Where we stand

Update this section as the work moves on. Detailed numbers go in the dated reports (`reports/`).

**Done**
- Pipeline: reproducible data setup (ENCODE fold-change tracks, 8 assays × 8 cell lines; region files), cell-type
  split (5 dev, 3 held out), train/val/test regions (chr20 / chr22 / chr21 tiles), fiber masks trimmed to the
  nucleosome span, scoring at positions with ≥ 5 fibers, cross-cell-average baseline and cell-specific r.
- **T1 (bulk recovery):** works. With enough training the dev-split model is close to the cross-cell average on
  Pearson (ahead in some dev cells, behind in Panc1 and the held-out cells) and better on MSE in most cells.
- **T2 (held-out cell types):** the cell-specific r is positive in every cell type, held-out ones included, so the
  model carries cell-specific information from the fibers that generalizes to new cell types (supports H over
  H0 at the population level). Wrong-cell control and cell-type-specific loci still to do.

**Known weak points**
- Broad marks, H3K36me3 above all (far below the baseline in several cell types); likely needs longer context.
- Training plateaus after ~60 × 1000 samples with the current LR schedule.

**Next**
- T3 (mixture-only supervision), the main test of H, with the current target scale; then T4 (scale variants).
- Single-molecule references for the ladder (4a).

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

### T4. Output parameterization and assay model (methodology comparison)

- **What varies.** What a molecule's output represents and how outputs map to the bulk track (the "assay model"
  between molecules and ENCODE). All variants stay cell-agnostic: nothing about the cell type beyond its fibers.
- **Hypothesis.** Outputs in physical units (a probability per molecule, aggregated linearly in the fraction of
  marked molecules, with assay effects such as scale and fragment smoothing modelled once per assay) resolve
  molecules better than the current setup (an unbounded per-bp value whose mean is compared with asinh(FC)).
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
  - V3: V2 with the probability defined on **nucleosomes**: for histone marks, one probability per called
    nucleosome, painted over its span, 0 on linkers / accessible patches (no histone there); bulk = a per-assay
    smoothing kernel (ChIP fragment blurring, learned once per assay) applied to the average over molecules.
    Unlike V1/V2 this changes the *support* of the prediction, not only its scale, and it differs by assay type:
    nucleosomes for histone marks, the molecule's own accessible patches for ATAC, the motif footprint
    (bound / unbound) for CTCF.
- **Readouts.** T3's fiber-level readouts (main), plus T1/T2 bulk readouts (should not get worse).
- **Prediction.** V1–V3 improve T3/T5 readouts at equal or better bulk recovery; V3 most, because it restricts
  per-molecule outputs to physically meaningful units and lets them be sharp while the bulk stays smooth.
- **Status.** Deferred until after T3 with V0. Needs I4 (V1/V2) and I14 (V3).

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

### T9. A pretrained sequence-to-activity model as the DNA encoder

- **Idea.** Replace the small DNA encoder (a DoubleConv on the window's one-hot sequence) with the trunk of a
  sequence-to-activity model (Enformer, Borzoi; ChromBPNet-style for base-pair resolution), whose purpose is
  exactly to encode DNA.
- **For.** Long-range context: Enformer/Borzoi embeddings summarize ~100–500 kb around each position, which is what
  the weak broad marks (H3K36me3 over gene bodies, H3K27me3/H3K9me3 domains) probably need and a 5 kb window
  cannot provide.
- **Against / risks, besides compute.**
  1. **Leakage.** Enformer and Borzoi were trained on ENCODE tracks that include our cell lines and assays
     (GM12878, K562, HepG2, ... histone ChIP), with held-out sets of genomic regions, not whole chromosomes. Their
     embeddings may partly encode the very chr21 tracks we test on. Use only a model/fold whose training excluded
     our test regions, or train our own larger encoder on our training chromosomes.
  2. **It strengthens the shortcut we test against.** The DNA embedding is identical for every fiber (and every
     cell type) at a locus. A strong one lets the model fit the bulk from DNA alone and ignore the fibers: bulk
     metrics may improve while per-molecule informativeness drops (H0). Judge it on the cell-specific r and T3/T5,
     not on bulk r.
  3. **Resolution mismatch.** Enformer outputs 128 bp bins, Borzoi 32 bp; our per-fiber outputs are per base pair,
     so embeddings must be upsampled (coarse) and combined with base-pair features.
  4. **Context mismatch.** These models need 196–524 kb of input around each 5 kb window.
  5. **Engineering.** Large models (Enformer ~250M parameters) from TensorFlow/Keras, with PyTorch ports; keep the
     trunk frozen and **precompute embeddings once per locus** (DNA is the same for all cells and fibers), e.g.
     chr20–22 at 128 bp × ~3k dims in float16 is a few GB per chromosome. This also makes the compute cost small.
- **Design.** Frozen trunk, cached embeddings, injected where the DNA encoder is now. Compare with the current
  model on bulk r/MSE, cell-specific r (T2), and T3. Include "the seq2act model alone" as the DNA-only rung of
  the ladder.
- **Prediction.** Gains concentrated in broad marks; per-molecule readouts (cell-specific r, T3) should not drop.
  If bulk improves but they drop, the model is copying DNA-derived bulk onto molecules.
- **Status.** Idea; after the single-molecule tests.

---

## 4. Comparison ladder

Reference points from best possible to worst possible, so results can be read qualitatively. **The single-molecule
part (4a) comes first**; the bulk ladder (4b) later. Our claim is not to top the bulk ladder but: comparable bulk
accuracy to methods with the same information, plus molecule-level output that none of them provides.

### 4a. Single-molecule references (focus)

Scored with the molecule-level tests (T3 mixtures, T5 within-population), not bulk tracks.

| Reference | What it represents | Expected on T3 / T5 |
|---|---|---|
| Model trained with each cell type's own track (not only the mixture) | Upper bound for T3 | Best separation |
| **fiberCANDI (us)** | | Should beat the two below |
| **Rule-based single-molecule imputer**, e.g. "molecules accessible at an element carry the active mark", scaled to match the bulk | The obvious objection: did the model learn more than accessible = active? | Some separation for active marks, little for repressive ones |
| **Bulk on every molecule** (the mixture / cross-cell bulk assigned to each molecule) | H0 | No separation, by construction |
| Random per-molecule values with the right mean | Noise floor | None |

### 4b. Bulk references (later)

Two axes matter: what the method sees about the target cell, and its output resolution. Where we land is an
empirical result, not a fixed position.

| Rung | Sees from the target cell | Output | Notes |
|---|---|---|---|
| Replicate of the target assay | the assay itself | bulk | **Noise ceiling; priority.** ENCODE per-replicate FC tracks exist for most experiments; compare replicate vs replicate (not vs pooled, which contains it) |
| CANDI, all of the target cell's assays (minus the target) | other assays | bulk | Richer input than ours. **Check CANDI's training split**: if it saw the target (cell, assay), it is partly an oracle |
| Model trained on the target cell type | fibers + target track in training | molecule | Cost of generalizing to a new cell; exists for dev cells, missing for the held-out 3 |
| **CANDI, ATAC only** and **CANDI on aggregated Fiber-seq accessibility** | bulk accessibility | bulk | **The head-to-head peer.** The Fiber-seq version is the like-for-like one; H1/H9 have no ATAC |
| Bulk model on aggregated fiber channels (T6) | aggregated fibers | bulk | In-house "same information" reference |
| fiberCANDI + bulk ChIP of a similar-cell cluster | fibers + other cells' histone bulk | molecule | A model variant, not a reference: likely better bulk but risk of copying the prior onto every molecule (H0); needs cell knowledge |
| **fiberCANDI (us)** | fibers only | molecule | |
| Average over similar cell types | nothing | bulk | "Similar" defined without the target's histone data (accessibility or lineage); tiny groups with 8 cells |
| Average over all other (training) cell types | nothing | bulk | Done (cross-cell-average baseline) |
| DNA-only model (ours without fibers, or a seq2act model, T9) | nothing | bulk | Isolates what the fibers add; `setup_wandb_run` skips all-zero `input_flags` runs, adjust |
| Genome-wide constant per assay | nothing | bulk | MSE floor only (Pearson undefined) |

Fair comparison: same chr21 tiles, same scored positions (target cell's fiber coverage ≥ 5 applied to every
method), same metrics and transform (asinh FC); external predictions at coarser resolution (CANDI bins) need a
common binning for the ladder.

Priority within 4b: replicate ceiling → DNA-only model → similar-cell average → CANDI (after the leakage check)
→ target-cell-trained model for held-out cells → CANDI all-assays and fiberCANDI + cluster.

---

## 5. Implementation TODO

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
| I10 | Rule-based single-molecule imputer + "bulk on every molecule" reference, scored with T3/T5 | 4a | Analysis only, on saved predictions (I2) |
| I11 | Replicate-level ENCODE tracks in data setup (noise ceiling) | 4b | Extend the spec/lock to per-replicate FC files |
| I12 | DNA-only model run; similar-cell average baseline | 4b | Allow all-zero `input_flags`; baseline with a chosen cell subset |
| I13 | Pretrained seq2act DNA encoder (frozen, cached embeddings) | T9 | Leakage check first |
| I14 | Nucleosome-level outputs + per-assay smoothing kernel (output variant V3) | T4, §7 | Model + loss change |

## 6. Engineering backlog (performance and quality)

**Step 0: find the bottleneck.** Training runs at ~9 samples/s and validation (forward only) at ~23/s. If
validation is GPU-bound, a training step (forward + backward + checkpoint recomputation, ~3× forward) predicts
~8 samples/s, so training may already be GPU-bound on the 2g.20gb slice (2/7 of an H100), not loader-bound.
Time a training step on synthetic batches (GPU) and the loader alone (CPU) before choosing fixes.

**Speed**
- **Validate less often:** validation is ~42 s per 107 s of training (~28% of time) at 1000 samples per epoch;
  use ~5000 samples per epoch.
- **Worker restarts:** `forkserver` workers restart every epoch (~15–20 s per loader); larger epochs fix most of
  it, persistent workers with a shared epoch counter the rest.
- **Gradient checkpointing** in every `DoubleConv` recomputes activations (~+30% compute) while memory has
  headroom (11.9 of 19.6 GB): drop it where memory allows.
- **DNA encoded once per window, not per fiber:** the DNA features are concatenated onto every fiber, so the first
  UNet level redoes DNA work 100×; a separate DNA branch run once per window and added at each scale is cheaper
  (and can be deeper or longer-range).
- **Fewer fibers per training sample** (e.g. 50 random): halves compute; the bulk average is noisier but unbiased;
  eval keeps 100.
- `torch.compile`, a larger MIG slice: if GPU-bound.
- **Eval batch size > 1** (now 1, GPU mostly idle).

- **bigWig cache:** training is limited by cold bigWig reads (~9 samples/s train vs ~23 val). Cache asinh target
  tracks for the chromosomes in use as float16 memory-mapped arrays (data setup); also makes the baseline cheap
  enough to run with every eval.
**Training and data (better results)**
- **Fiber selection bias / augmentation:** the first 100 fibers by start position are kept, so coverage is lower
  on the right of each window and a locus always shows the same fibers. Random (seeded) subsets per visit fix
  the bias and act as augmentation; they decode all overlapping fibers (+50–90% fetch time).
- **Reverse-complement augmentation:** flip the window (fiber channels and targets) and reverse-complement the
  DNA; marks are strand-agnostic, so this doubles the effective data.
- **More training chromosomes:** training uses chr20 only (64 Mb); drawing from all chromosomes except val/test
  costs nothing per sample (config change) and likely helps generalization.
- **Context and broad marks:** fibers are cropped to the 5 kb window before the model. Process longer input
  (full fiber or up to 20 kb; regions already allow it) and score the central region (centre-only scoring is
  not implemented yet). Likely needed for H3K36me3 / H3K27me3 domains.
- **Longer training / LR schedule:** 25 epochs was undertrained; with train03 val Pearson plateaus at ~0.49 by
  epoch ~60, after which ReduceLROnPlateau (patience 5) has cut the LR to ~0. ~60 epochs captures the gain; a
  longer patience or a cosine schedule with warmup (possibly a higher peak LR) and gradient clipping may get more.
- **Comparison protocol for these changes:** one dev config, ~30k samples, mini eval + cell-specific r; one
  change per run so gains are attributable.
**Other**
- **Eval summary keyed by cell type:** two eval configs for the same cell type in one run overwrite each other.
- **Mixed-mode evaluation:** composite target currently mixes asinh tracks (see T4).
- **Housekeeping:** October logs exist only on nibi (the fir sync's initial pull replaced local `logs/`); nibi
  test job 24330075 may still be queued there.

## 7. Model improvements for single-molecule imputation

Aimed at more accurate, more molecule-specific outputs rather than bulk. Several will barely move bulk metrics, so
judge them with the single-molecule readouts (cell-specific r, T3, T5), with bulk metrics as a guard.

| # | Change | Why it should help single molecules | Effort |
|---|---|---|---|
| M1 | **Nucleosome-level outputs + per-assay assay model** (T4 variant V3) | Histone marks sit on nucleosomes; the ChIP peak shape comes from fragment blurring and averaging, not from each molecule. Predicting a probability per nucleosome (0 on linkers) and modelling the blurring once per assay restricts outputs to physical units and stops the model painting a bulk-like shape onto every molecule | Medium–large |
| M2 | **Supervision that distinguishes molecules:** mixture training with varying proportions (in addition to single-cell training), more cell types at the same loci, both libraries (100U + 200U) of a cell | The loss only constrains averages; seeing the same DNA with different molecules and different targets forces the model to read the fibers. Mixtures make per-subpopulation signal identifiable; two libraries with one target teach robustness to library effects | Medium (I1) |
| M3 | **Whole read as context** (process the full fiber, crop the output; the report's design) | A molecule's state depends on its surroundings (inside a transcribed gene body for H3K36me3, nucleosome arrays, nearby elements); fibers are ~18 kb but cropped to 5 kb. Likely also the fix for broad marks | Medium–large |
| M4 | **Continuous call confidences as inputs:** m6A/CpG ML probabilities instead of ±1 at ML ≥ 200; FIRE scores of MSPs instead of a thresholded channel | Keeps per-molecule information currently discarded | Small |
| M5 | **DNA as modulation, not an additive path:** gate or scale (FiLM) fiber features by DNA instead of concatenating DNA as a full input | The DNA path is identical for every fiber at a locus, the channel through which the bulk-on-every-molecule shortcut flows; with modulation a fiber's output needs fiber evidence | Medium |
| M6 | **Same-molecule biological constraints** (soft penalties): H3K27ac vs H3K27me3 on the same nucleosome (same lysine); check H3K4me3 vs CpG methylation at CpG islands | Independent, molecule-level supervision that is meaningless on bulk tracks; small weights, after M1–M2 | Small–medium |
| M7 | **Per-molecule consistency:** per-position (or fiber-masked) normalization instead of `GroupNorm(1, C)` over the whole window; optionally consistency of a molecule's prediction across overlapping windows | Now a molecule's prediction can depend on how much of it falls in the window, which a single-molecule model should not | Small |
| M8 | **Model selection on single-molecule readouts** | Bulk metrics cannot detect drift towards H0 | Process only |

M1–M3 fit together as a "v2" model (nucleosome-level outputs on full reads, trained on single cells plus
mixtures), to compare with the current model on T3/T5 and on bulk as a guard.

