# Research plan: extended TODO

The full plan and every floating idea: the hypothesis, the main plan (§2), the test catalogue (T1–T11), the
comparison ladder, the implementation list, the engineering backlog, model ideas, and the registry of reference
models. The **current**
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

*Update (18 Nov):* the mixture test showed that what transfers is having seen the cell types, not the type of
supervision. The main claim is now that a model trained on a sample's own (mixture) bulk resolves that sample's
molecules (§2). "Out of the box for new cell types" stays a secondary question.

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

The open biological tests are in `TODO.md` §3; the main plan is §2 here and `TODO.md` §1.

---

## 2. Main plan (18 Nov): the mixture-trained model and a sealed test

### 2a. The claim and why

- **The use case.** A tissue is a mixture of cell types. You can run Fiber-seq and ChIP on the same mixed sample,
  but you can't separate it into pure cell types to get each one's own tracks. The selling point: train on the
  mixture's own bulk, and still impute each molecule correctly.
- **What we know (`Nov_13_extended.md` §3, one seed):**
  - on its own pair, the mixture-only model matches the pure-track pair model (fiber-origin AUC 0.637 vs 0.635 on
    GM12878–HepG2, 0.666 vs 0.665 on K562–H1; own-track recovery within 0.015);
  - on a pair it never saw, the type of supervision makes no difference: what matters is having seen the cell
    types. Transfer is lopsided (poor toward K562–H1).
- **So the main model to sell is one trained on the sample itself,** compared with DEV4NF as "a model pretrained on
  other cells". The unseen-pair result predicts the mixture model wins on new cell types.
- **This shifts the main hypothesis (§1)** from "works out of the box for new cell types" to "trained on a
  sample's own bulk, it resolves that sample's molecules". The out-of-the-box version stays a secondary question
  (DEV4NF on held-out cells).

### 2b. Realism gaps in the current mixtures

Current mixtures (`mix_range`, `data27` / `data35`) are easier than a real sample in two ways:
1. **The cell fraction is redrawn for every training window** (uniform in [0.2, 0.8]). A real sample has one
   fraction. T3's design note says this variation is what makes each population's signal identifiable ("with one
   fixed ratio, many splits of the signal fit equally well"), so the realistic case may be much harder.
2. **The target is weighted by the fibers actually sampled** at each position (q_c(x) = cell c's share of the
   fibers covering x). Real ChIP reflects the cell population, not the particular molecules Fiber-seq happened to
   read.

Variants to compare (2 cells first):

| Variant | Cell fraction | Target | Realistic? |
|---|---|---|---|
| **R0** (current) | redrawn per window, [0.2, 0.8] | weighted by the fibers sampled | no |
| **R1** | one fixed fraction q per mixture (e.g. 0.3 and 0.5 as two separate runs) | population: asinh(q sinh t_A + (1 − q) sinh t_B), independent of the fibers sampled | yes: one sample |
| **R2** | 3–4 samples, each with its own fixed fraction (e.g. 0.25 / 0.5 / 0.75); each window comes from one sample | population target of that sample | yes: a few donors, tissues or time points |

- If R1 ≈ R0: the realistic claim holds as is.
- If R1 fails but R2 works: the claim becomes "a few samples with different proportions are enough".
- If both fail: mixture training only works with information a real experiment doesn't have; report that and
  sell the pure-track model plus the biology tests instead.
- Fiber sampling within a window stays random (Fiber-seq depth noise is real). Optional later: lower fiber
  coverage, closer to a real experiment's depth.

### 2c. Test hygiene

- **What has been touched:** the held-out cells (H9, HCT116, MCF-7) entered the ceiling fractions, the ablation
  tables, the seed analysis and the DEV4NF choice; Panc1 moved to held out because of its own results; chr21 is
  the evaluation chromosome of every bulk and molecule test so far; chrX was used for the X-inactivation test.
- **From now on:**
  - **development:** the 4 dev cell types; train chr20, validation chr22, development evaluation chr21 (plus chrX
    for X inactivation);
  - **sealed test:** the held-out cell types on a fresh chromosome, never evaluated before. Pick it now by
    criteria fixed in advance (size and gene density similar to chr21, ENCODE coverage in every held-out cell,
    enough CTCF sites and CGI promoters), e.g. chr18 or chr19;
  - the test is run once with the locked recipe and reported whatever it shows. The earlier use of the held-out
    cells on chr21 is stated in the paper.
- **Held-out mixtures:** HCT116 + MCF-7 (the cleanest pair: different tissues, same Fiber-seq series); a 3-cell
  mixture with H9. H9 and H1 are both embryonic stem cells, so H9 is less of a new cell type.
- **Panc1** stays out (its sample identity is in doubt).

### 2d. Comparisons

| Comparison | Question | Where |
|---|---|---|
| Mixture model vs **pair model** (same cells, own tracks, same channels) | Does mixture supervision lose anything? (identifiability ceiling) | dev pairs |
| Mixture model vs **DEV4NF** applied without retraining | Train on your sample, or use a pretrained model? | dev pairs (DEV4NF saw the cells) and the sealed test (it didn't) |
| Mixture model vs **simple per-fiber models** under the same supervision (T11) | Is the network needed? | dev pairs, then test |
| Floors: **bulk on every molecule**, library control, random split of one library | What does chance / H0 score? | everywhere |

All mixture and pair models use DEV4NF's architecture and channels (`model16`, no FIRE), so supervision is the
only difference.

### 2e. Readouts

- **T3 readouts:** fiber-origin AUC, sign accuracy and differential r at the pair's differential bins; each cell's
  recovery of its own track, against the mixture track's recovery of it ("bulk on every molecule").
- **More than 2 cells:** one-vs-rest fiber-origin AUC per cell, and each cell's own-track recovery.
- **Biology tests on mixture-trained models:** CTCF footprints on each cell's fibers (without the hole sites), and
  X inactivation when GM12878 is in the mixture. "Trained on a mixture, the molecules still show the right
  biology" is a stronger claim than separating cell types.
- **Bulk guard:** mixture bulk r on the evaluation chromosome.

### 2f. Seeds and success criteria

- **3 seeds for every claim;** report mean ± sd and the seed-to-seed spread. Measure seed noise first (stage 2).
- **Draft success criteria** (finalize after stage 2, before stage 3):
  1. on the dev pairs, the realistic mixture model is within 2 seed SDs of the pair model on fiber-origin AUC and
     own-track recovery;
  2. it is clearly above the floors (bulk on every molecule, library control);
  3. the CTCF footprint and X-inactivation active-mark tests pass with mixture-only supervision, in all 3 seeds;
  4. on the sealed test, the model trained on the held-out mixture beats DEV4NF applied without retraining on
     fiber-origin AUC and differential r.

### 2g. Stages and cost

Training runs take ~2.2 h each on a 2g.20gb slice (under the 3 h limit), plus evals.

| Stage | What | Runs |
|---|---|---|
| 1 | Protocol, success criteria and test chromosome written down; data / eval configs | — |
| 2 | Seed noise: Mix A, Mix B, Pair A, Pair B with `model16`, 3 seeds each | 12 (~27 GPU-h) |
| 3 | Realism: R1 and R2 on both dev pairs × 3 seeds (R0 from stage 2) | 12 (+ 6 if R1 runs at two fractions) |
| 4 | Lock: more dev pairs (spread across pairs), one 3-cell dev mixture, biology tests on the mixture models | ~10–15 |
| 5 | Sealed test: locked recipe on HCT116 + MCF-7 and a 3-cell held-out mixture, 3 seeds; DEV4NF without retraining | ~6 |

Implementation: I26–I31 (§5).

### 2h. Done so far (details in the dated reports)

- Pipeline: ENCODE fold-change tracks (8 assays × 8 lines), region files, cell-specific and pairwise loci, V1
  fiber averaging in fold change (default since 20 Oct), split v2 (4 Nov).
- T1 / T2 bulk and cell-specific r: `Oct_23.md`, `Nov_13.md`. Replicate ceiling (I11): `Nov_13_extended.md` §9.
- T3 / T3b mixture test on two pairs: `Oct_23_T3.md`, `Nov_13_extended.md` §3.
- T10 input ablations, 2 seeds: `Nov_13_extended.md` §6, §10. DEV4NF became the main model (18 Nov).
- T5e CTCF footprints and T5b X inactivation: `Nov_20.md`, with locus figures in `Nov_20_figs.md`. CTCF rerun
  without the 75 motif sites lacking bulk short-read data (`ctcf_holes.py`).

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
- **Status.** Done (V1, DEV5 / DEV4 / DEV4NF; `Nov_13.md`), including the replicate ceiling. Open: the loci
  breakdown by gain / loss (I3) and the T6 reference.

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
- **Status.** Cell-specific r done (supports H), with the replicate ceiling of it (`Nov_13_extended.md` §9).
  Open: wrong-cell control (I5), related-cell reference (I12). Held-out cells are sealed from 18 Nov (§2c): new
  T2 numbers only on the sealed test chromosome at the end.

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
- **Status.** Done on two pairs, one seed (`Oct_23_T3.md`, `Nov_13_extended.md` §3): the mixture model matches
  the pair model on its own pair. Configs: `data27` / `data35` (mixtures), `data28` / `data34` (pairs), paired
  evals with library controls, `t3_report.py`, `plots_code/26_nov_13/mix_two_pairs.py`. **Now the base of the
  main plan (§2):** 3 seeds, realistic mixtures (R1 / R2), more than 2 cells, sealed test. Random-split noise floor
  still to implement (I31).

### T3b. Mixture test on unseen cell types

- **Question.** Does a model trained on mixtures of two cell types separate the molecules of two *other* cell
  types it never saw? (Rules out the shortcut "infer the cell type, recall its typical profile".)
- **Result** (`Nov_13_extended.md` §3, one seed):
  - on an unseen pair, mixture and own-track models do the same: what matters is having seen the cell types;
  - transfer is lopsided: fine toward GM12878–HepG2, poor toward K562–H1 (H3K36me3, H3K9me3, H3K27ac);
  - routed specialists (I16) deferred.
- **Consequence:** the main model is trained on the sample itself (§2); DEV4NF applied to new cells is the
  "pretrained" comparison.

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

### T5. Within-population structure and biological validation (open tests: `TODO.md` §3)

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
  3. **Ablation loss per assay:** done (T10). Open: readouts 1 (the no-CpG model now exists) and 2 (I21).
- **T5b. X inactivation:** done at promoters (`Nov_20.md` §2; GM12878, Xi = HP2 from the measured XIST promoter
  methylation). At silenced CGI promoters the active marks pass in every model and seed, H3K9me3 weakly, H3K27me3
  fails (more on the Xa). Open:
  - restrict to genes expressed in GM12878 (RNA-seq). PTCHD1 (`Nov_20_figs.md`) suggests part of the H3K27me3
    "failure" is non-expressed genes, where Polycomb on the Xa copy may be correct;
  - check the escape list against Tukiainen 2017;
  - H3K27me3 and H3K36me3 over gene bodies and Xi domains;
  - CTCF per haplotype at chrX motifs (footprint frequency on Xa vs Xi; DXZ4 / FIRRE as Xi-specific controls);
  - the unphased bimodality version for the other female lines (MCF-7, H9; XCI often eroded in hESC). Held-out
    lines are sealed (§2c), so only at the end.
- **T5c. Imprinted control regions** (H19/IGF2, KvDMR1, SNRPN, PEG3, MEG3, GNAS; MAGEL2, ZDBF2 from FIRE):
  - Unmethylated allele: H3K4me3 + accessibility (CTCF at the H19 ICR). Methylated allele: H3K9me3.
  - Showcase figures across the phased lines.
  - Some loci sit on training chromosomes (GNAS on chr20). That is acceptable, since haplotypes were never
    supervised, but say so.
- **T5d. Allele-specific ChIP at heterozygous SNPs** (strongest quantitative haplotype test):
  - GM12878 AlleleDB / ADASTRA calls for CTCF, H3K27ac, H3K4me3 (I20).
  - The predicted per-haplotype difference should have the sign of the ChIP allelic imbalance, and be ~0 at
    balanced SNPs.
- **T5e. CTCF footprints:** done (`Nov_20.md` §1, chr21, GM12878 and K562; our own m6A footprint call). Footprint
  fibers get more predicted CTCF than open fibers at bound sites, in every model that sees m6A; DEV4NF strongest
  and most specific in both seeds; unbound sites ≈ 0 after removing the motif sites without bulk data. Open: run
  it on the mixture-trained models (§2e) and the simple baselines (T11).
- **T5f. Same-molecule co-occurrence where accessibility and marks disagree:**
  - Bivalent promoters in H1: H3K4me3 + H3K27me3 on the same molecules, but not in differentiated lines.
  - H3K36me3 and H3K27me3 mutually exclusive (PRC2 is inhibited by H3K36me3).
  - Repressive marks not simply inverted accessibility.
  - Co-actuation of neighbouring elements on the same molecule, with distance decay (Stergachis 2020).
- **Status.** T5b and T5e done (`Nov_20.md`). Open: T5a (in-silico flips, held-out methylation), T5c, T5d, T5f,
  and the T5b follow-ups above (`TODO.md` §3).

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
- **Status.** Not started; cheap now that `molecule_predict.py` exists (inference only). Planned with T11
  (`TODO.md` §2).

### T8. External single-molecule data (opportunistic)

- Antibody-directed single-molecule histone data for one of our cell lines would be the closest to ground truth:
  - DiMeLo-seq (Altemose 2022);
  - ChromSMF (Palamin & Krebs 2026, histone modifications + protein–DNA interactions + methylation on multi-kb
    molecules; cited by SMFNet);
  - Yue et al. 2022 (nanopore profiling of histone marks + methylation).
- Single-cell CUT&Tag (e.g. K562) gives per-locus distributions to compare against the spread of per-molecule
  predictions.
- **TODO (`TODO.md` §3):** check which cell lines and marks have public data.

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

**Status.** Done for no CpG, no FIRE, no accessibility and m6A + CpG (no fibertools calls), 2 seeds each
(`Nov_13_extended.md` §6, §10; molecule tests in `Nov_20.md`). No FIRE (DEV4NF) became the main model. Open:
T5a held-out methylation with the no-CpG model.

### T11. Simple per-fiber baselines: does the network learn more than an obvious reading of the input?

- **Why.** The standing objection to every T5 result: "the model just reads the fiber data, so of course
  footprint fibers get more CTCF". That is what the model is supposed to do, so it can't be argued away; it can
  be measured. A simple per-fiber model run through the same tests says how much the network adds over an
  obvious reading of the same input.
- **Hypothesis.** The network beats a simple per-fiber model on the readouts the simple model was not built
  from, most of all where no clean rule is known (H3K27ac, H3K4me1) and on held-out cell types.
- **How to read the outcome** (all three are reportable):
  - network ≈ simple model: it learned the known biology from bulk alone, with no rule written in (a sanity
    result, a weaker claim);
  - network > simple model: it uses information an obvious reading misses (the stronger claim);
  - both fail (e.g. H3K27me3 on the Xi, T5b): the information is likely not in those fiber features at all, not a
    model-capacity problem.
- **Two kinds of baseline:**
  1. **Hand-written rules, one per assay** (§4a rule-based imputer). Easy to explain, but the comparison partly
     measures how well the rule was written. Fix every rule before looking at results. Rough rules:

     | Assay | Per-fiber rule |
     |---|---|
     | ATAC | fraction of the window inside an MSP |
     | CTCF | footprint at a motif |
     | H3K4me3 | accessible + unmethylated CpG near a CGI / TSS |
     | H3K27ac / H3K4me1 | accessible, away from TSS (no clean rule) |
     | H3K36me3 | high CpG methylation over a gene body |
     | H3K27me3 | unmethylated CGI + closed |
     | H3K9me3 | closed + methylated (+ repeats) |

     Per-fiber scores need a per-assay scale to compare with bulk (fit one linear map per assay on chr20).
  2. **A simple learned model, trained like ours (preferred, the fair comparison).** Per fiber: summary features
     at a few window sizes (MSP fraction, nucleosome fraction, m6A rate, mean CpG methylation) plus reference
     annotations (motif score, CGI / TSS distance); a linear or logistic map to the 8 outputs; the same Softplus +
     coverage-normalized fiber average; trained on the same bulk targets, loss, data and budget. The only
     differences from the network are capacity and fine spatial patterns, which answers "is a deep network
     needed, or do local summary statistics do?" directly. **Leave out any hand-made motif-protection feature**,
     so that a CTCF footprint response in the network and not here is evidence it found the pattern unaided.
- **Trap: the CTCF footprint test is circular for a hand rule.** T5e labels fibers with a rule (motif m6A ≤ 0.5 ×
  flanks, inside an MSP); a rule "fewer m6A at the motif → CTCF" is the same rule and scores near-perfectly by
  construction. There it is a reference ceiling for that test ("what reading the footprint directly gives"), not
  a competitor. Compare on what the rule was not built from: bulk CTCF r (fiber-averaged rule vs ChIP), bound vs
  unbound sites at matched motif score and the site-mean vs peak-strength Spearman (T5e H5), and the pair /
  held-out-cell differences (T2, T3).
- **Readouts:** every existing test, unchanged: bulk r and fraction of the replicate ceiling (I11), mixture
  fiber-origin AUC and differential readouts (T3), CTCF footprints (T5e), X inactivation (T5b), later T5a–d.
  Molecule tests over 3 seeds for the learned baseline too (`Nov_13_extended.md` §10).
- **Expected** (write the final version down before running):
  - ATAC: simple model close; not where the claim lives.
  - CTCF: learned summary model without a motif-protection feature has little footprint response; the hand rule
    has it by construction.
  - H3K36me3: windowed CpG methylation may carry most of it, so the simple model could be close (consistent with
    the no-CpG ablation, T10).
  - H3K27ac, H3K4me1, held-out cell types: the largest gap, if there is one.
- **If the gap is small almost everywhere,** know it before writing: the pitch shifts from "deep model" to
  "bulk-supervised per-fiber imputation works, and this is what is learnable from fiber features".
- **Update (18 Nov), after `Nov_20.md`:**
  - **Compare against DEV4NF** (the main model since 18 Nov), in both seeds. Give the baselines the same channels:
    m6A, CpG, MSP, nucleosome, no FIRE.
  - **Once the mixture recipe is settled (§2, stage 3), train the learned baseline the same way** (same mixtures,
    same target), so it answers the same question as the main model. 3 seeds, like everything we claim.
  - **Run the tightened tests:** CTCF footprints without the 75 sites lacking bulk data (`ctcf_holes.py`), with
    height and shape against the bulk track (`Nov_20_figs.md`); X inactivation scored on silenced CGI promoters.
  - **H3K27me3 on the Xi is the clearest "both fail?" case.** If the simple model also puts more H3K27me3 on the
    Xa, the "open, unmethylated CGI → H3K27me3" link is in the fiber features, not a network quirk. Check expressed
    genes separately (PTCHD1, `Nov_20_figs.md`).
  - **Order:**
    1. Fix the designs and write the expected results down before running (the list above, updated for DEV4NF).
    2. Hand rules on the molecule tests only (I25). The per-fiber TSVs already hold the summaries; no new
       predictions needed.
    3. Shuffle controls for DEV4NF (T7).
    4. Learned summary model (I24), 2 seeds, scored with every test as above. Add a baseline column to the locus
       figures.
    5. Later: the bulk model on fiber-averaged inputs (T6); bulk scoring for the hand rules.
  - **Open choices for I24:**
    - Window sizes: start at ≈ 50 bp, so the baseline cannot see footprints. Also run a second arm with ≈ 25 bp,
      to test whether a linear map already gets part of the footprint response (it cannot tie protection to the
      motif, since a linear map cannot multiply them).
    - DNA input: local GC / CpG density (a CGI proxy), or explicit annotations (motif score, CGI, TSS distance).
      The annotations make I24 and I25 comparable but need annotation tracks.
    - Linear, or also a one-hidden-layer MLP on the same summaries, to separate "nonlinearity" from "fine spatial
      pattern".
- **Status.** Idea (17 Nov); plan updated 18 Nov. Implementation I24 (learned), I25 (rules).

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
| **Mixture-trained model on the sample itself (main, §2)** | Realistic use: trained on the mixture's own bulk | Close to the own-track model; should beat the rows below |
| **DEV4NF applied without retraining** | A model pretrained on other (pure) cell types | Good on cells it saw; worse on new cell types (T3b) |
| **Simple learned per-fiber model** (T11): summary features per fiber → linear map, same fiber averaging, same bulk training | Same information and supervision, no deep network: is the network needed? | Close on ATAC (and maybe H3K36me3); weaker on the T5e footprint response and H3K27ac / H3K4me1 |
| **Rule-based single-molecule imputer** (T11), e.g. "molecules accessible at an element carry the active mark", scaled to match the bulk | The obvious objection: did the model learn more than accessible = active? | Some separation for active marks, little for repressive ones. Circular on T5e (the footprint rule is the test's label): a ceiling there, not a competitor |
| **Bulk on every molecule** (the mixture / cross-cell bulk assigned to each molecule) | H0 | No separation, by construction |
| Random per-molecule values with the right mean | Noise floor | None |

### 4b. Bulk references (later)

Two axes matter: what the method sees about the target cell, and its output resolution. Where we land is an
empirical result, not a fixed position.

| Rung | Sees from the target cell | Output | Notes |
|---|---|---|---|
| Replicate of the target assay | the assay itself | bulk | **Noise ceiling. Done** (I11, `Nov_13_extended.md` §9): replicate r ≈ 0.4, reachable r ≈ 0.74 |
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

Priority within 4b: DNA-only model → similar-cell average → CANDI (after the leakage check) → target-cell-trained
model for held-out cells → CANDI all-assays and fiberCANDI + cluster.

---

## 5. Implementation TODO

Open items only. **Done** (details in the reports and the code): I1 mixture training and paired eval (`mix_range`,
`t3_report.py`); I3 cell-type-specific loci; I4 V1 fiber averaging; I6 / I23 ablation models (no CpG, no FIRE,
no accessibility, m6A + CpG); I8 haplotype tags; I11 replicate ceiling; I15 unseen-pair test; I22 per-fiber CTCF
footprint call (`ctcf_footprints.py`); per-fiber predictions at target BEDs (I2, molecule part,
`molecule_predict.py`); Xa / Xi split for GM12878 (I19, partly).

**For the main plan (§2):**

| ID | Change | For | Notes |
|---|---|---|---|
| I26 | **Population mixture target:** option `mix_target: population` = asinh(Σ_c q_c sinh t_c) with the nominal fractions q_c, independent of the fibers sampled (now: weighted by the realized fiber share per position) | §2b R1 / R2 | `data_utils.py` mixed dataset; the current behaviour stays the default for old configs. Fixed fraction already possible with `mix_range: [q, q]` |
| I27 | **Several samples with fixed fractions:** option `mix_samples: [0.25, 0.5, 0.75]`; each window draws one sample, and fibers and target follow that sample's fraction | §2b R2 | Same place as I26 |
| I28 | **Mixtures of more than 2 cell types:** fractions per cell (fixed or per sample), target over all cells; mixed eval and paired readouts generalized (one-vs-rest fiber-origin AUC, per-cell own-track recovery) | §2b, §2e | `mix_range` now needs exactly 2 cell types. Differential bins for n cells: per cell vs the rest |
| I29 | **Sealed test chromosome:** pick it by fixed criteria, add its tiles / mini set to `regions_spec.yaml`, loci for the held-out pairs (and a held-out 3-cell set) on it, eval configs. Build, never evaluate until stage 5 | §2c | `build_regions.py`, `build_loci.py`, `loci_spec.yaml`. Also the CTCF-site and TSS target BEDs on it for the biology tests (held-out cells only at the end) |
| I30 | **3-seed runs and aggregation:** mixture / pair data configs with `model16`, `SEED` per run (exists in `slurm_batch_command.sh`), one script that collects the T3 readouts per pair × model × seed (mean ± sd) | §2f | Extend `mix_two_pairs.py` |
| I31 | **Random-split noise floor:** one library's fibers split at random into two halves, scored like a pair | §2d floors, T3 | Left over from I1 |

**Other open items:**

| ID | Change | For | Notes |
|---|---|---|---|
| I2 | Bulk prediction / mask saving in `eval.py` for the pairwise readouts | T3, T7 | Molecule part done (`molecule_predict.py`) |
| I3 | Report the loci breakdown by gain / loss separately | T1, T2 | Plain r is depressed at loci (range restricted on the target); don't compare it with genome-wide r |
| I4 | Per-assay scale (V2); optional sigmoid outputs + per-assay calibration; optional per-track target normalization in data setup | T4 | V2 changes training: rerun references after |
| I5 | Wrong-cell control for the cell-specific r | T2 | Score cell A's target with cell B's prediction |
| I7 | chrX window set (regions spec) for bulk evaluation on chrX | T5b | The X-inactivation test used target BEDs instead |
| I9 | Aggregated-input bulk model | T6 | New model variant |
| I10 | "Bulk on every molecule" and random references, scored with T3 / T5 | 4a, §2d | Analysis only, on saved predictions |
| I12 | DNA-only model run; similar-cell average baseline; related-cell reference | 4b, T2 | Allow all-zero `input_flags`; related-cell reference = one sibling cell's real track as the prediction (H1 → H9) |
| I13 | Pretrained seq2act DNA encoder (frozen, cached embeddings) | T9 | Leakage check first |
| I14 | Nucleosome-level outputs + per-assay smoothing kernel (output variant V3) | T4, §7 | Model + loss change |
| I16 | *(Deferred)* Routed specialists: each fiber through its own cell type's model | T3 | Specialists trained, routing not built |
| I17 | Multi-resolution r: Pearson / MSE on 200 bp and 1 kb bin means | assay tiers | Evaluator only |
| I18 | Assay-specific r: partial correlation of predicted and true mark given the true ATAC track | assay tiers | Evaluator only; cells without ATAC (H1, H9) skipped |
| I19 | Haplotype split for the other phased lines (per-block labels), and exclusion of CpG-creating / destroying SNPs | T5c, T5d | GM12878 done (one phase block per chromosome; HP2 = Xi) |
| I20 | Allele-specific ChIP truth set: AlleleDB / ADASTRA calls for GM12878 CTCF, H3K27ac, H3K4me3, matched to the fibers' haplotypes (phased NA12878 VCF) | T5d | Data download + matching only |
| I21 | In-silico CpG flips: set a molecule's CpG calls in a region (e.g. a CGI) to methylated / unmethylated and rerun the model on the same fibers | T5a | Inference-only tool on top of `molecule_predict.py` |
| I24 | Learned summary-feature baseline: a new model in `models.py` (per-fiber window summaries at a few scales + reference annotations → linear / 1×1 layer → Softplus → coverage-normalized average), trained with the usual `main.py` pipeline | T11 | No motif-protection feature; DEV4NF's channels; two window arms (smallest ≈ 50 bp, and ≈ 25 bp); 3 seeds; trained like the main model once §2 stage 3 settles it |
| I25 | Hand-written per-assay rules, fixed before running, as a scorer over `molecule_predict.py` outputs, plus a per-assay linear scale fit on chr20 for bulk | T11 | Rules in the T11 table. First: molecule tests only, from the existing per-fiber TSVs (CTCF without holes, X inactivation at CGI promoters) |

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
- **Housekeeping:** October logs exist only on nibi (the fir sync's initial pull replaced local `logs/`).

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
| **DEV4 (V1, train04)** | `data24` | 4 dev cell types (split v2: GM12878, K562, HepG2, H1) | nibi | `2026-11-03_01_train_loo_panc1_v1` | `*_train_LOO_Panc1_V1_data24_dev_loo_Panc1_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | main dev model 4–18 Nov (trained as the leave-Panc1-out fold), replaced by DEV4NF; 2 h 19 min; full evals 3 Nov (`Nov_13.md`) |
| Specialists × 5 | `data29`–`data33` | one cell type each | nibi | `2026-11-03_05`–`09_train_spec_*` | `*_train_SPEC_V1_<data>_model15_uct_mid_8A_train04/Model_epoch_15.pt` | done 3 Nov, ~2 h 12 min each; full own-cell evals without baseline; Panc1's is a held-out-cell oracle under split v2 |
| Pair model A | `data28` | GM12878 + HepG2 (own tracks) | nibi | `2026-11-04_01_train_pair_a_gm12878_hepg2` | `*_train_PAIR_V1_data28_GM12878_HepG2_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | T3b ceiling A (train04 rerun of the T3 upper bound); done 4 Nov, val r .566; paired evals on both pairs + controls (`Nov_13.md`) |
| Pair model B | `data34` | K562 + H1 (own tracks) | nibi | `2026-11-04_02_train_pair_b_k562_h1` | `*_train_PAIR_V1_data34_K562_H1_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | T3b ceiling B; done 4 Nov, val r .530; paired evals on both pairs + controls (`Nov_13.md`) |
| Mixture model A | `data27` | GM12878 + HepG2 mixtures only | nibi | `2026-11-04_06_train_mix_a_gm12878_hepg2` | `*_train_MIX_V1_data27_mix_GM12878_HepG2_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | train04 rerun of the T3 mixture model; done 4 Nov, val r .612; paired evals on both pairs + controls (`Nov_13.md`) |
| Mixture model B | `data35` | K562 + H1 mixtures only | nibi | `2026-11-04_07_train_mix_b_k562_h1` | `*_train_MIX_V1_data35_mix_K562_H1_200U_model15_uct_mid_8A_train04/Model_epoch_15.pt` | no ATAC; done 4 Nov, val r .559; paired evals on both pairs + controls (`Nov_13.md`) |
| **DEV4NF** (no FIRE) | `data24` | 4 dev cell types | nibi | `2026-11-07_03_train_abl_nofire` | `*_train_ABL_V1_data24_dev_loo_Panc1_200U_model16_uct_mid_8A_no_fire_msp_train04/Model_epoch_15.pt` | **main model from 18 Nov**; `model16`, `input_flags` [1, 1, 1, 1, 0]; molecule tests `Nov_20.md` |
| Ablations | `data24` | 4 dev cell types | nibi | `2026-11-07_02_train_abl_nocpg`, `2026-11-07_04_train_abl_noacc`, `2026-11-08_15_train_abl_m6acpg` | `*_train_ABL_V1_data24_dev_loo_Panc1_200U_model{19_..._no_cpg, 20_..._cpg_only, 21_..._m6a_cpg}_train04/` | no CpG, no accessibility (CpG only), m6A + CpG; `Nov_13_extended.md` §6 |
| Seed 2 of DEV4, DEV4NF and the ablations | `data24` | 4 dev cell types | nibi | `2026-11-11_{04,10,16,22,28}_train_*_seed2` | run names with `_seed2` after the prefix (`*_train_LOO_Panc1_V1_seed2_...`, `*_train_ABL_V1_seed2_...`) | `slurm_batch_command.sh` with `SEED=2`; `Nov_13_extended.md` §10 |

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

