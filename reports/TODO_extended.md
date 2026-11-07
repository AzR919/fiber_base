# Research plan: extended TODO

The full plan and every floating idea: the hypothesis, the test catalogue (T1–T10), the comparison ladder, the
implementation list, the engineering backlog, model ideas, and the registry of reference models. The **current**
plan (what we run now, max two pages) is `TODO.md`; ideas move from here to there when they become current.
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
4. Within-population structure and biological validation: molecules of one cell type that should differ for a
   known biological reason (alleles, X inactivation, imprinting, CpG state, CTCF footprints; T5), and input
   ablations read through those tests (T10).

Levels 1–3 test differences *between* populations of molecules; only level 4 tests differences *within* one.
None verifies an individual molecule; that would need single-molecule histone measurements (T8).

**Validation strategy (since 7 Nov 2026).** Levels 1–3 use proxies. Fiber-origin AUC, for example, assumes every
molecule of the cell with the higher track should score higher, which molecular heterogeneity contradicts. The main
evidence is therefore level 4: biological backing, as in the single-molecule literature.

What the literature does:
- **SMFNet** (Schwämmle, Diamant et al. 2026) predicts *measured* SMF states. Beyond read reconstruction, it backs
  its biology with:
  - allele differences in an F1 hybrid (B6 × SPRET) at motif-disrupting variants;
  - reporter motif deletions (mCHIRA, r = 0.83);
  - in-silico motif insertions that match known TF behaviour.
- **The Fiber-seq papers** (Stergachis 2020; FIRE, Vollger et al.; fibertools, Jha et al. 2024) use:
  - held-out orthogonal assays (scATAC);
  - haplotype tests (GM12878 Xi is maternal, imprinting at GNAS / MAGEL2 / ZDBF2, heterozygous variants);
  - CTCF footprints and nucleosome phasing;
  - co-actuation of neighbouring elements;
  - null controls (shuffled reads, untreated / WGA DNA).

None of them validates histone marks on single molecules, so our claim is new.

Rule for every level-4 test:
1. It checks something the model was never trained on.
2. The expected direction comes from the literature and is written down before running.
3. It has a null control.

The current plan is in `TODO.md` §2.

---

## 2. Status snapshot (1 Nov 2026)

Frozen when the current plan moved to `TODO.md` (which now keeps the up-to-date status). Detailed numbers go in the
dated reports (`reports/`).

**Done**
- Pipeline: reproducible data setup (ENCODE fold-change tracks, 8 assays × 8 cell lines; region files), cell-type
  split (5 dev, 3 held out), train/val/test regions (chr20 / chr22 / chr21 tiles), fiber masks trimmed to the
  nucleosome span, scoring at positions with ≥ 5 fibers, cross-cell-average baseline and cell-specific r.
- **V1 output aggregation is the default:** per-fiber outputs stay in asinh units, the bulk averages them in fold
  change (asinh(mean sinh)); mixture targets and the baseline also average in FC. Bulk within noise of V0. The
  bulk results in `Oct_23.md` are V0, with the old baseline (mean of asinh tracks).
- **Cell-specific loci** (dev: 5 dev lines, leave-one-out; test: all 8) built on the chr21 tiles by data setup.
- **T1 (bulk recovery):** works. With enough training the dev-split model is close to the cross-cell average on
  Pearson (ahead in some dev cells, behind in Panc1 and the held-out cells) and better on MSE in most cells.
- **T2 (held-out cell types):** the cell-specific r is positive in every cell type, held-out ones included, so the
  model carries cell-specific information from the fibers that generalizes to new cell types (supports H over
  H0 at the population level). Wrong-cell control and the cell-specific-loci breakdown still to do.
- **T3 (mixture-only supervision), first run (GM12878 + HepG2):** supports H at the single-molecule level. Trained
  only on mixtures, the model separates the two cell types' molecules (fiber-origin AUC 0.64, sign accuracy 0.78)
  as well as the single-cell-trained upper bound, locus-specifically, with no separation of two HepG2 libraries
  (`Oct_23_T3.md`).

**Known weak points**
- Broad marks, H3K36me3 above all (far below the baseline in several cell types); likely needs longer context.
- Training plateaus after ~60 × 1000 samples with the current LR schedule.

**Next** (priority: single-molecule correctness)
- Rerun the baseline numbers with the FC mean.
- T3 follow-ups: a second seed for both models, the random-split noise floor, a second pair from the same
  Fiber-seq series (e.g. K562 + HepG2).
- T3b (mixture test on unseen cell types): rules out the "cell-type classifier" shortcut that T3 cannot.
- Reference values for r (what counts as a good Pearson / cell-type r): neither can reach 1, because the ENCODE
  target is itself noisy (worst at bp resolution, on broad marks and H3K9me3), and the cell-specific residual is
  mostly noise. Measure per assay (1) the replicate ceiling, replicate 1 vs replicate 2, with both plain r and
  cell-type r given the baseline (I11), and (2) a related-cell reference, a sibling cell type's real track used as
  the prediction, e.g. H1 for H9 (I12). Then report our numbers as a fraction of the ceiling, and note that r
  rises with coarser bins, so bp-level values are not comparable with 25 bp imputation papers.
- Later: per-assay scale (V2), T4 V3, single-molecule references for the ladder (4a).

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
- **Status.** Done with the old averaging (V0): DEV5 short and long (`Oct_23.md`, `Oct_30.md`). Rerun with a V1
  DEV5 and the fold-change baseline (`TODO.md`). TODO: report the loci breakdown (I3, implemented) and the T6
  reference.

### T2. Held-out cell types and the cell-specific part of the signal

- **Hypothesis.** The fiber → histone mapping generalizes to cell types never seen in training, including the part
  of their signal that differs from other cells.
- **Design.** Train on the 5 dev cells, evaluate on HCT116, MCF-7, H9 (and the dev cells on chr21).
- **Readouts.** Partial correlation of prediction and target given the cross-cell average (cell-specific r);
  plain r and MSE at cell-type-specific loci (I3), where the baseline is weakest.
- **Controls.** "Wrong cell" control: score cell A's target with cell B's prediction (given the baseline). If that
  scores as well, the gain is a better estimate of shared signal, not cell-specific information.
- **Reference values.** Replicate ceiling of the cell-specific r (replicate 1 vs 2, given the baseline; I11) and a
  related-cell reference (H1's real track as the prediction for H9; I12), so the cell-specific r reads as a fraction
  of what is achievable rather than as r² of the total.
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
  - With V1 (default): the model averages fibers in FC and the mixture target is mixed in FC, so one fixed
    per-fiber output fits every proportion. (Under V0 with FC-mixed targets no per-fiber output would fit every
    proportion, since the right output would depend on the proportion.) Weight the composite by each cell's
    realized fiber coverage per position, not the nominal ratio (part of I1).
- **Readouts.**
  - Per-cell recovery: r and MSE of each cell's prediction vs its own track, **compared with the mixture track's
    r and MSE against that cell** (the "bulk on every molecule" reference).
  - Differential correlation: corr(p_A − p_B, t_A − t_B); the reference scores 0.
  - Sign accuracy at strongly differential windows (top 10% |t_A − t_B| per assay).
  - Fiber-origin AUC at differential windows: can one fiber's predicted mark identify its cell type?
  - Reference points: the same readouts for a model trained on the two cell types separately (single-cell-
    supervised reference; not a ceiling: same fibers and model, different supervision).
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
- **Status.** Implemented (I1, I3): `data27` (mixtures, share [0.2, 0.8]), `data28` (single-cell reference), paired eval
  `eval24` (GM12878 vs HepG2_200U) and library control `eval25` (HepG2_100U vs 200U), mixture eval `eval26`,
  report `t3_report.py`. Differential bins from `loci/pairs/GM12878__HepG2`. The fiber-origin AUC is computed
  during eval (per fiber, the mean prediction over the bin), so I2 is not needed for it. First run done
  (`Oct_23_T3.md`): supports H, mixture model on par with the single-cell reference. Random-split noise floor not
  implemented yet.

### T3b. Mixture test on unseen cell types

> Done on two pairs (GM12878–HepG2, K562–H1) with pair-model and DEV4 references; results in `Nov_13.md`
> ("Mixture test on two pairs"). The channel ablations that were planned with it are now hypothesis-driven (T10).
> The text below is the original motivation.

- **Hypothesis.** A model trained on mixtures of two cell types separates the molecules of two *other* cell types
  it never saw, at the loci where those two differ.
- **Why it adds to T3.** T3 cannot rule out a shortcut: the model infers each molecule's cell type from global
  features (overall CpG methylation, m6A pattern) and predicts that cell type's typical chromatin at the locus from
  DNA. With two training cell types this separates molecules locus by locus, passes the direction-split check and
  generalizes to new loci. For unseen cell types there is no profile to recall: at best the shortcut pushes each
  new cell type towards the training cell type it resembles. Fibers are processed independently, so testing on a
  mixture of unseen cell types is the paired eval of the two cell types (no composite needed).
- **Design.**
  - Pairs: K562 vs Panc1 (dev cell types the T3 models never saw; both from the same Fiber-seq series, so no
    cross-experiment library concern); H1 vs K562. Held-out pair (HCT116 vs MCF-7) at milestones only.
  - Models: the T3 mixture model, the T3 single-cell reference (GM12878 + HepG2, also unseen), and DEV5 (trained
    on K562 / Panc1 / H1's own tracks) as the reference that has seen the pair.
  - Readouts: as T3 (differential r, sign accuracy, fiber-origin AUC at the pair's differential bins).
  - **Shortcut check:** correlate the predicted difference with the unseen pair's true difference (reading) and
    with the training pair's difference at the same windows (shortcut). Reading predicts the first.
- **Prediction (H).** Clearly positive readouts, close to DEV5's, following the unseen pair's difference, not the
  training pair's. **(Shortcut)** readouts near chance, or tracking the GM12878 - HepG2 axis.
- **Caveat.** A failure is ambiguous: no molecule reading, or reading learned from two cell types does not transfer
  (compare with DEV5 and with mixture training on more cell types, M2). A success is clear. Does not replace T5
  (within one cell type no identity shortcut exists at all).
- **Status.** Done (Nov 2026, `Nov_13.md`):
  - Mixture-only training reaches the own-track ceiling on both pairs.
  - On an unseen pair, mixture and own-track models do the same.
  - Transfer is lopsided: poor toward K562–H1.
  - Routed specialists (the third reference) deferred.

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
  - V1 (**default since 2026-10-20**, `fiber_aggregation: fc`): per-fiber outputs z_f stay in asinh units (the
    range the network learns in); bulk = asinh(mean_f sinh(z_f)), i.e. averaged in FC, equal to z when fibers
    agree. Mixture targets and the baseline are averaged in FC too.
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
- **Status.** V1 implemented and default (V0 = `fiber_aggregation: asinh`, also how old checkpoints load).
  V2 (per-assay scale) and V3 after T3. Needs I4 (V2) and I14 (V3).

### T5. Within-population structure and biological validation (current plan, `TODO.md` §2)

Only these probe correctness *within* a cell type. Each uses a group of molecules defined independently of the
model, with a known expected difference written down before running, and a null.

**Why haplotypes are clean for us:**
- All Fiber-seq CRAMs carry HiPhase HP/PS tags (I8).
- The DNA input is the reference sequence (`dna_type: ref`), and the ENCODE targets are unphased.
- So a difference between the two copies can only come from the molecules' chromatin signals, and was never
  supervised.
- **Exception:** SNPs that create or destroy a CpG (the CpG channel sees them). Exclude them.

- **T5a. CpG methylation, same molecule.** The hypotheses (literature directions):
  - H3K4me3 is low on molecules with methylated CGI promoters (CFP1 binds unmethylated CGIs).
  - H3K36me3 goes with gene-body methylation (DNMT3B is recruited by H3K36me3).
  - H3K9me3 goes with methylation at ICRs and repeats.
  - H3K27me3 is largely antagonistic to methylation at CGIs.
  - CTCF is blocked by methylation at its motif.

  Three readouts:
  1. **Held-out methylation:** with the no-CpG model (T10), correlate predicted marks with the molecule's measured
     methylation across molecules at the same locus. The baseline and any "bulk on every molecule" model score 0.
  2. **In-silico flips:** on the full model, methylate or unmethylate a molecule's CGI CpGs and check the
     direction of change per assay (cf. SMFNet's in-silico motif insertions). No retraining (I21).
  3. **Ablation loss per assay:** see T10.
- **T5b. X inactivation**, phased instead of only bimodality:
  - Cell line: GM12878, where the Xi is maternal (99% of XIST transcripts, FIRE paper). Assign HP blocks to Xa / Xi
    by trio phasing, or by XIST-promoter state measured away from the test genes.
  - Expected: Xi molecules gain H3K27me3 / H3K9me3 and lose H3K4me3 / H3K27ac at TSSs active only on the Xa.
    Escape genes (UBA1, KDM5C, KDM6A, ...) are active on both. CTCF sits on both copies at most sites, and is
    Xi-specific at DXZ4 / FIRRE / ICCE.
  - Nulls: autosomes; male lines (HepG2, H1).
  - The unphased bimodality version still applies to the other female lines (MCF-7, H9; XCI often eroded in hESC).
  - Needs a chrX window set (I7).
- **T5c. Imprinted control regions** (H19/IGF2, KvDMR1, SNRPN, PEG3, MEG3, GNAS; MAGEL2, ZDBF2 from FIRE):
  - Unmethylated allele: H3K4me3 + accessibility (CTCF at the H19 ICR). Methylated allele: H3K9me3.
  - Showcase figures across the phased lines.
  - Some loci sit on training chromosomes (GNAS on chr20). That is acceptable, since haplotypes were never
    supervised, but say so.
- **T5d. Allele-specific ChIP at heterozygous SNPs** (strongest quantitative haplotype test):
  - GM12878 AlleleDB / ADASTRA calls for CTCF, H3K27ac, H3K4me3 (I20).
  - The predicted per-haplotype difference should have the sign of the ChIP allelic imbalance, and be ~0 at
    balanced SNPs.
- **T5e. CTCF footprints:**
  - Per-fiber CTCF predictions at CTCF motifs, against footprint calls on the same fiber (I22).
  - Expected ranking: bound > accessible-unbound > closed. Stergachis 2020: about 30% of accessible fibers at K562
    CTCF sites are bound.
  - Footprints are a measured single-molecule readout of CTCF occupancy.
- **T5f. Same-molecule co-occurrence where accessibility and marks disagree:**
  - Bivalent promoters in H1: H3K4me3 + H3K27me3 on the same molecules, but not in differentiated lines.
  - H3K36me3 and H3K27me3 mutually exclusive (PRC2 is inhibited by H3K36me3).
  - Repressive marks not simply inverted accessibility.
  - Co-actuation of neighbouring elements on the same molecule, with distance decay (Stergachis 2020).
- **Status.**
  - Planned (`TODO.md` §2); HP tags checked (I8).
  - Phase 1 (no retraining): T5b–T5e, plus the T5a in-silico flips.
  - T5a held-out needs the no-CpG model (T10).

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

- Antibody-directed single-molecule histone data for one of our cell lines would be the closest to ground truth:
  - DiMeLo-seq (Altemose 2022);
  - ChromSMF (Palamin & Krebs 2026, histone modifications + protein–DNA interactions + methylation on multi-kb
    molecules; cited by SMFNet);
  - Yue et al. 2022 (nanopore profiling of histone marks + methylation).
- Single-cell CUT&Tag (e.g. K562) gives per-locus distributions to compare against the spread of per-molecule
  predictions.
- **TODO (first step of `TODO.md` §2):** check which cell lines and marks have public data.

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

### T10. Input ablations with biological hypotheses

Replaces the channel-ablation table of the old T3b plan. Each ablation has a hypothesis and is read mainly through
the T5 tests (allele, X, imprinting, footprints), with bulk and T3 readouts as guards. DEV4 (`data24`) is retrained
per variant; `input_flags` order is [m6A, CpG, MSP, nuc, FIRE].

| Variant | `input_flags` | Hypothesis | Main readouts |
|---|---|---|---|
| No CpG | [1, 0, 1, 1, 1] | CpG matters for the CGI / methylation-linked assays: H3K4me3, H3K36me3, H3K9me3, CTCF lose more than ATAC, H3K27ac, H3K4me1 | Per-assay loss; T5a held-out methylation; imprinting / Xi differences in H3K9me3 shrink; whether the H1 CTCF shift disappears |
| No FIRE (`model16`) | [1, 1, 1, 1, 0] | FIRE adds little beyond raw m6A / MSP. FIRE was trained on GM12878 DNase + CTCF ChIP peaks, so a large drop means a bulk-derived shortcut | CTCF and GM12878 first; T5e footprints; if the cost is small, no-FIRE becomes the main model |
| No accessibility | [0, 1, 0, 0, 0] (CpG + DNA) | Accessibility carries most of the active marks; what remains shows what CpG + DNA alone give | Per-assay loss; compare with the accessibility-only baseline (§4a) from the other side |

Not current (later, if the above call for them):
- FIRE only [0, 0, 0, 0, 1];
- CpG only [0, 1, 0, 0, 0] without DNA;
- no DNA (all fiber channels);
- no fibertools calls ([1, 1, 0, 0, 0]: whether MSP / nuc / FIRE are needed or learnt from raw m6A);
- continuous ML / FIRE scores instead of ternary calls.

**Cost.** About 2.2 h per variant on a 2g.20gb slice, plus evals. Differences of a few hundredths need a second
seed.

**Status.** Planned (`TODO.md` §2, step 2).

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
| I1 | ~~Mixture training with variable proportions; paired eval with the T3 readouts~~ (done 2026-10-21: `mix_range`, target weighted by the fibers present, `paired` eval configs, `t3_report.py`); random-split noise floor still to do | T3 | |
| I2 | `eval.py --save_predictions`: per sample bulk prediction, target, scored-position mask (float16), plus a per-fiber summary (mean predicted mark per assay over its span); offline analysis script for pairwise / mixture readouts | T3, T5, T7 | Full per-fiber tracks are too large to save |
| I3 | Cell-type-specific-loci breakdown in eval: report MSE, sign accuracy and cell-specific r on each cell's loci tiles (gain / loss) | T1, T2, T3 | Done: loci files (`data_prep/build_loci.py`) and eval config key `loci` (region types `loci:...`); gain / loss split not reported separately yet. Plain r is depressed there (range restricted on the target), so don't compare it with genome-wide r |
| I4 | Output scale: ~~average fibers in FC (V1), mix targets and baseline in FC~~ (done); per-assay scale (V2); optional sigmoid outputs + per-assay calibration; optional per-track target normalization in data setup | T4 | V2 changes training: rerun references after |
| I5 | Wrong-cell control for the cell-specific r | T2 | Score cell A's target with cell B's prediction |
| I6 | No-CpG model run (`input_flags` [1, 0, 1, 1, 1], new model config) + molecule-level correlation with held-out methylation | T5a, T10 | |
| I7 | chrX window set (regions spec) + bimodality analysis | T5b | Spec entry only for the windows; regions currently exclude chrX |
| I8 | ~~Check CRAMs for haplotype (HP) tags~~ (done 7 Nov: HiPhase HP/PS in all CRAMs; tagged share in a chr21 sample GM12878 93%, K562 100%, HepG2 100%, H1 98%, H9 55%, MCF-7 18%) | T5b–d | Tags are consistent within a phase block (PS) only; parental assignment needs trio phasing (GM12878) or a per-block label |
| I9 | Aggregated-input bulk model | T6 | New model variant |
| I10 | Rule-based single-molecule imputer + "bulk on every molecule" reference, scored with T3/T5 | 4a | Analysis only, on saved predictions (I2) |
| I11 | Replicate-level ENCODE tracks in data setup (noise ceiling) | 4b, T2 | Extend the spec/lock to per-replicate FC files. Score replicate vs replicate per assay and cell type with plain r, MSE and cell-specific r (given the baseline), on the same scored positions |
| I12 | DNA-only model run; similar-cell average baseline; related-cell reference | 4b, T2 | Allow all-zero `input_flags`; baseline with a chosen cell subset; related-cell reference = one sibling cell's real track as the prediction (H1 → H9), scored with cell-specific r |
| I13 | Pretrained seq2act DNA encoder (frozen, cached embeddings) | T9 | Leakage check first |
| I14 | Nucleosome-level outputs + per-assay smoothing kernel (output variant V3) | T4, §7 | Model + loss change |
| I15 | *(Done, Nov 2026: pairs GM12878–HepG2 and K562–H1, `Nov_13.md`)* Unseen-pair test: `pairs` in `loci_spec.yaml` (K562–Panc1, K562–H1, Panc1–H1, GM12878–H1, HepG2–H1), paired eval configs + library controls (K562, Panc1, H1), single-cell and pair data configs, ablation model configs, routing for specialists, pair × assay summary (`t3_report.py`); optional shortcut check (correlate the predicted difference with the training pair's difference at the same windows) | T3b | |
| I16 | *(Deferred; specialists trained, routing not built)* Oracle-identity reference for T3: one model per cell type (trained on its own track), each fiber routed through its own cell type's model. Quantifies how much separation knowing the cell type alone buys; not a ceiling for molecule reading (origin is given, so it scores high even without reading fibers) | T3 | Two extra training runs + routing in the paired eval |
| I17 | Multi-resolution r: Pearson / MSE also on 200 bp and 1 kb bin means of prediction and target (same scored positions) | assay tiers (`TODO.md` §3) | Evaluator only |
| I18 | Assay-specific r: partial correlation of predicted and true mark given the true ATAC track, per assay | assay tiers (`TODO.md` §3) | Evaluator only; cells without ATAC (H1, H9) skipped |
| I19 | Haplotype-split analysis: per-fiber predictions at chosen loci split by HP within each phase block (PS); Xa / Xi block labels for GM12878 (trio phasing or XIST-promoter state); exclusion of CpG-creating / destroying SNPs | T5b–d | Builds on I2 (saved per-fiber summaries) |
| I20 | Allele-specific ChIP truth set: heterozygous SNPs with allelic-imbalance calls (AlleleDB / ADASTRA) for GM12878 CTCF, H3K27ac, H3K4me3, matched to the fibers' haplotypes (GIAB / Platinum NA12878 phased VCF) | T5d | Data download + matching only |
| I21 | In-silico CpG flips: set a molecule's CpG calls in a region (e.g. a CGI) to methylated / unmethylated and rerun the model on the same fibers | T5a | Inference-only tool |
| I22 | CTCF footprint calls per fiber at CTCF motifs (fibertools `ft footprint` or an m6A-gap caller), joined to per-fiber CTCF predictions | T5e | |
| I23 | Ablation model configs: no CpG [1, 0, 1, 1, 1], no accessibility [0, 1, 0, 0, 0]; no FIRE = `model16` | T10 | Next free model numbers |

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

## 8. Reference models (registry)

Trained models we compare against repeatedly. Add a row when a run finishes: the checkpoint is on the cluster
under `results/`. All V1 unless noted. Same setup unless noted: chr20, half cCRE / half random regions, `model15`,
`train02` (25 × 1000 samples), 100 fibers, 5 kb.

| Model | Data config | Trained on | Cluster | Job | Checkpoint (`results/...`) | Notes |
|---|---|---|---|---|---|---|
| DEV5 short (V0) | `data20` | 5 dev cell types | fir | `2026-10-17_08_dev5_ccre_random` | `*_DEV5_*_train02/Model_epoch_25.pt` | old averaging; superseded by DEV5 (V1) |
| DEV5 long (V0) | `data20` | 5 dev cell types | fir | `2026-10-18_03_dev5_long` | `26-10-18_T10-14-23_DEV5_long_..._train03/Model_epoch_100.pt` | old averaging, 100 epochs |
| DEV5 short (V1 check) | `data20` | 5 dev cell types | fir | `2026-10-20_02_dev5_v1_short` | `*_DEV5_V1_short_*/Model_epoch_25.pt` | V1 check run; can serve as DEV5 (V1) |
| T3 mixture | `data27` | GM12878 + HepG2 mixtures | fir | `2026-10-21_01_t3_mix` | `26-10-21_T00-55-17_T3_mix_..._train02/Model_epoch_25.pt` | mixture-only supervision |
| GM12878 + HepG2 | `data28` | GM12878, HepG2 (own tracks) | fir | `2026-10-21_02_t3_upper` | `*_T3_upper_data28_*_train02/Model_epoch_25.pt` | T3 single-cell-supervised reference; main model A of T3b |
| DEV5 (V1, train04) | `data20` | 5 dev cell types | nibi | `2026-11-01_04_dev5_v1_train04` | `26-11-02_T02-37-45_train_DEV5_V1_data20_dev5_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | 15 × 5000 samples, 2 h 20 min; val r 0.501, HepG2 mini r 0.588; split v1 (with Panc1); superseded by DEV4 |
| **DEV4 (V1, train04)** | `data24` | 4 dev cell types (split v2: GM12878, K562, HepG2, H1) | nibi | `2026-11-03_01_train_loo_panc1_v1` | `*_train_LOO_Panc1_V1_data24_dev_loo_Panc1_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | **main dev model from 4 Nov** (trained as the leave-Panc1-out fold); 2 h 19 min; full evals 3 Nov (`Nov_13.md`) |
| Specialists × 5 | `data29`–`data33` | one cell type each | nibi | `2026-11-03_05`–`09_train_spec_*` | `*_train_SPEC_V1_<data>_model15_uct_mid_8A_train04/Model_epoch_15.pt` | done 3 Nov, ~2 h 12 min each; full own-cell evals without baseline; Panc1's is a held-out-cell oracle under split v2 |
| Pair model A | `data28` | GM12878 + HepG2 (own tracks) | nibi | `2026-11-04_01_train_pair_a_gm12878_hepg2` | `*_train_PAIR_V1_data28_GM12878_HepG2_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | T3b ceiling A (train04 rerun of the T3 upper bound); done 4 Nov, val r .566; paired evals on both pairs + controls (`Nov_13.md`) |
| Pair model B | `data34` | K562 + H1 (own tracks) | nibi | `2026-11-04_02_train_pair_b_k562_h1` | `*_train_PAIR_V1_data34_K562_H1_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | T3b ceiling B; done 4 Nov, val r .530; paired evals on both pairs + controls (`Nov_13.md`) |
| Mixture model A | `data27` | GM12878 + HepG2 mixtures only | nibi | `2026-11-04_06_train_mix_a_gm12878_hepg2` | `*_train_MIX_V1_data27_mix_GM12878_HepG2_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | train04 rerun of the T3 mixture model; done 4 Nov, val r .612; paired evals on both pairs + controls (`Nov_13.md`) |
| Mixture model B | `data35` | K562 + H1 mixtures only | nibi | `2026-11-04_07_train_mix_b_k562_h1` | `*_train_MIX_V1_data35_mix_K562_H1_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | no ATAC; done 4 Nov, val r .559; paired evals on both pairs + controls (`Nov_13.md`) |

## 9. Ideas from the 30 Oct presentation review

- **Drop fibers with very low m6A?** Poorly labelled fibers carry little information. But compact heterochromatin
  also has less accessible DNA and so less m6A, so a hard filter could bias against H3K9me3 / H3K27me3 molecules.
  First look at m6A density per fiber per library; then drop only the failed-labelling tail, or give the density
  to the model as an input.
- **H3K36me3 should be catchable** (Christine): larger windows. Longer input with centre-only scoring, then the
  whole read as context (M3).
- **Other derived features:** FiberHMM nucleosome / accessibility calls; per-molecule global numbers (m6A density,
  read length, mean MSP length, CpG methylation fraction) as broadcast input channels. Global features make it
  easier to recognise the cell type, so check them on unseen pairs (T3b).
- **Arabidopsis data from Christine:** a new organism and genome; a strong generalisation test, but needs a new
  data setup (genome, plant ChIP tracks).

