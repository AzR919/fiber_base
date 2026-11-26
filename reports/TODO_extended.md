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

**Scope (user, 25 Nov):** this paper uses two cell types at a 0.5 share only. Real samples have more cell types at
unequal, unknown shares. That is harder, goes to a follow-up paper, and the paper says so. No 3-cell mixtures (I28),
no other shares, no more dev pairs.

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

#### Addendum (23 Nov): separating the gaps, and how R1 / R2 sample fibers

The text above is unchanged; this refines it.

**Setting q first or drawing fibers first is equivalent when the target follows the fibers.** In both cases the
target is *consistent*: exactly what the fiber average can produce if every fiber carried its cell type's value.
The original design (fix q, trim the larger cell type's fibers until the stack matches q, target from q) and R0
(draw q, take the fibers, target from the fibers' realized share per position) are both consistent. Two details of
R0 as coded (`MixedCellFiberDataset._sample`): it does not rebalance when a cell type runs short (GM12878 70,
K562 20, p = 0.5 → 50 + 20 fibers, realized share 0.71, and the target uses 0.71), and the share is per position,
because fibers cover different spans of the window.

**So there are three gaps, not one:**
1. **Fixed vs varying q** (gap 1 above): identifiability. The original fixed-q design has this gap too.
2. **Consistent vs population target** (gap 2 above): in a real sample the ChIP reflects the population at
   fraction π, while the fibers at a locus are a separate random sample whose share scatters around π (binomial
   noise), so target and fibers never match exactly. R0 and the original design both lack this noise; it matters
   most at low coverage.
3. **One library vs two (new).** A real mixture goes through one Fiber-seq prep. Synthetic mixtures combine
   separately prepared libraries that differ in digestion level and MSP fraction (GM12878–HepG2 even crosses
   Fiber-seq experiments), so the model can partly separate cell types by library. The library control puts this
   at |AUC − 0.5| ≈ 0.04–0.09 against 0.17–0.25 for real pairs (`Nov_13_extended.md` §3): small, but it flatters
   every synthetic-mixture result. Only a real mixed-sample library removes it; until then it is a stated
   limitation.

**Fiber counts must follow q in R1 / R2.** With a population target at fixed q, a stack that doesn't match q
(e.g. 70 GM12878 + 20 K562 fibers against a 50/50 target) is a systematic mismatch a real library never has. The
output is the fiber average, so the model would have to distort per-fiber values (e.g. inflate the K562 fibers)
to hit the target, corrupting exactly the per-fiber predictions we care about. Sampling rule:
- **Stack size:** N = min(budget, min_c floor(available_c / q_c)). GM12878 70, K562 20, q = 0.5 → N = 40 (20 + 20,
  as in the original design); q_GM = 0.3 → N = 28 (≈ 8 + 20). A smaller stack is just lower local depth, which is
  realistic.
- **Per-cell counts:** n ~ Multinomial(N, q), capped at what is available, rather than exact rounding. This adds
  the binomial scatter of a real library (gap 2).
- **Target:** the population target at fixed q, the same at every position (R1 / R2 rows above); per-position
  differences in fiber share become realistic noise rather than part of the target.
- **Minimum:** skip a window if N < ~10 and log how often; skipping is fine if rare but biases training toward
  well-covered loci.
- **Eval** uses the same rule.
- *Optional later:* scale each cell type's expected share by its local coverage relative to its genome-wide mean,
  to mimic copy number (K562 is near-triploid), since fiber share ≠ cell share where copy number differs.

**Split R1 so the two gaps can be told apart.** R1 changes gaps 1 and 2 at once; if it fails, the cause is
unclear. One extra variant separates them:

| Variant | Cell fraction | Target | Measures |
|---|---|---|---|
| R0 (current) | redrawn per window | consistent (fibers sampled) | reference |
| **R1a** (original design) | fixed | consistent (fibers forced to q) | R0 → R1a: cost of a fixed fraction (identifiability) |
| **R1b** | fixed | population, multinomial fiber counts (rule above) | R1a → R1b: cost of target–fiber noise |
| R2 | 3–4 fixed fractions | population, same rule | whether several fractions restore identifiability |

Gap 3 is not tested by any variant.

#### First run (23 Nov): R1a on one pair, one seed

- **R1a as implemented** (`mix_balance` in `MixedCellFiberDataset._sample`, config `data36`): the share is fixed
  (`mix_range: [0.5, 0.5]`); the stack size is N = min(100, min_c floor(available_c / share_c)) and each cell type
  gives round(share_c × N), so the stack always matches the share; windows with N < 10 are retried (train) or
  skipped (eval), with the count printed per worker. **The target is unchanged** (per-position share of the fibers
  present), so R0 → R1a changes only the fraction. A target set from q instead would add part of gap 2, because
  fibers cover different spans of the window. Old configs behave exactly as before.
- **Smoke test** (40 validation windows): R1a stacks are 50 / 50 at 100 fibers in every window (none below 10),
  so for this pair R1a is a clean "one fixed fraction" test. R0 unchanged (shares 0.26–0.79).
- **Runs** (`2026-11-23_00`–`13`, seed 919, `model16`): R0 (`data27`, prefix `MIXR0_NF`), R1a (`data36`,
  `MIXR1A_NF`), Pair (`data28`, `PAIR_NF`); evals `eval24` (paired: T3 readouts and per-cell bulk), `eval25`
  (library control), `eval26` (composites, varying share), `eval31` (composites, as R1a trains); the same evals for
  DEV4NF; molecule predictions at the CTCF sites (GM12878, HepG2 fibers) and chrX / chr21 TSSs (GM12878).
- **Scored with every test:** T3 readouts; bulk per cell and for the mixture, with replicate-ceiling fractions;
  CTCF footprints without the hole sites; X inactivation at silenced CGI promoters.
- **Decision:** R1a passes if it matches R0 within seed noise (judged against the 2-seed spread of DEV4NF until
  the mixture models have their own seeds) on the T3 readouts and on bulk, and the biology tests keep their
  direction. Then 3 seeds and R1b. If R1a falls clearly below R0, go to R2. (Outcome: R1a passed and is locked;
  R1b later, R2 dropped. See "Result and decision" below.)
- **To settle before R1b, if it is run** (review of the addendum, 23 Nov):
  - draw the multinomial counts again when one exceeds what is available, instead of capping (capping always
    trims the scarce cell, a systematic bias);
  - the population target should weight each cell by its share of the DNA at the position, π_c × CN_c(x) /
    Σ π × CN, not by the constant π: ENCODE tracks are fold change over each cell's own input, so a mixed sample's
    fold change is this weighted mean. Estimate CN_c(x) from local coverage relative to the cell's genome-wide
    mean (K562 is near-triploid; HepG2 has CNVs too);
  - realistic depth: a synthetic stack pools two libraries, about twice one real library's depth. Set the budget
    to one library's typical coverage per window;
  - **gap 4, one ChIP vs separate ENCODE experiments:** a real sample has one ChIP experiment; synthetic targets
    mix experiments with different scales (track outliers: H1 CTCF, HepG2 H3K9me3). Exclude flagged tracks from
    mixture scoring, or normalize track scale (I4);
  - gap 3 mitigations: a pair from one Fiber-seq series (K562, HepG2, H1) next to a cross-series one; match m6A
    density between libraries. A real mixing experiment (cells of two lines mixed before Fiber-seq, each read
    assigned to its line by SNP genotype) would remove gap 3 and give per-fiber ground truth: ask the wet lab, or
    look for public data.

#### Result and decision (23 Nov): R1a is the mixture recipe

Scripts `plots_code/26_nov_27/realism_t3.py` and `realism_bio.py`, outputs in `plots/26_nov_27/realism/`. One pair,
one seed; all evals use the final checkpoint (epoch 15).

| | Fiber-origin AUC | Sign accuracy | Differential r | Bulk r GM12878 / HepG2 (fraction of ceiling) | Mixture r (`eval26` / `eval31`) | CTCF H1 GM / HepG2 | CTCF H2 GM / HepG2 |
|---|---|---|---|---|---|---|---|
| DEV4NF | 0.637 | 0.743 | 0.328 | 0.471 (0.64) / 0.596 (0.71) | 0.593 / 0.604 | 1.14 / 0.97 | 0.67 / 0.79 |
| R0 | 0.654 | 0.779 | 0.330 | 0.468 (0.63) / 0.616 (0.74) | 0.610 / 0.623 | 0.99 / 0.81 | 0.92 / 0.98 |
| **R1a** | 0.653 | 0.770 | 0.352 | 0.482 (0.65) / 0.629 (0.75) | 0.622 / 0.633 | 0.81 / 0.58 | 0.69 / 0.70 |
| Pair | 0.646 | 0.795 | 0.368 | 0.482 (0.65) / 0.620 (0.74) | 0.615 / 0.625 | 1.12 / 0.86 | 0.59 / 0.66 |

- **T3 and bulk:** R1a matches R0 on the T3 readouts and is equal or better on bulk, level with the pair model.
  The library control stays at chance (|AUC − 0.5| 0.050–0.053) for every model. The pair model's edge on sign
  accuracy is mostly H3K9me3 (0.63 vs 0.38–0.43).
- **Biology tests:** every hypothesis keeps its direction in all four models. CTCF H1–H3 and H5 pass, the
  smallest p is 2e-14 (R1a, HepG2), and bound sites score above unbound ones in all 9 motif-score bins. In X
  inactivation, the active marks are at 0.87–0.92 for every model, the nulls are flat and XIST is reversed.
  H3K9me3 / H3K27me3 move as much as DEV4NF does between its seeds.
- **Watch across seeds:** R1a's CTCF footprint effect is the smallest (about 30–40% below DEV4NF; ~10 points of
  that is R1a's ~10% lower overall CTCF scale). DEV4NF moved 1.14 → 0.95 on GM12878 between its two seeds.
- **Unexpected:** the mixture models' pred ATAC barely separates footprint from open fibers, so their CTCF
  specificity (H2) is level with or above the pair model's.
- **Decision (user, 23 Nov):** R1a is the mixture recipe for the rest of the plan. The story is "trained on the
  kind of mixture you meet in practice (one sample, one fixed composition), the model still imputes each molecule
  well", and R1a covers that.
  - **R2 is dropped.** It was the fallback in case a single fixed fraction made the cell-specific signal
    unidentifiable; R1a shows it doesn't.
  - **R1b moves to later**, as an optional robustness check (population target, multinomial counts,
    copy-number weights). At 100 fibers and a 0.5 share the counts would swing by about ±5 fibers, so we expect a
    small effect; it is one run if a reviewer asks.
  - **Gaps 3 and 4** (one library vs two; one ChIP vs separate ENCODE experiments) are not addressed by any
    variant. They go in the paper's limitations.
- **Eval fix found here:** two mixed configs in one eval run were both saved as `mixed` in `eval_summary.json`, so
  the second overwrote the first. Mixed configs are now keyed `<config>/mixed` (`evaluator.py`); for these runs,
  `eval26` was read from the eval logs.

#### Seed noise (24 Nov): R1a and Pair, both dev pairs, 3 seeds

Script `plots_code/26_nov_27/realism_seeds.py` (job `realism_seeds.sh`), outputs in `plots/26_nov_27/seeds/`
(`seeds_table.tsv`, `seeds_deltas.tsv`, `seeds_t3.png`, `seeds_bulk.png`, `seeds_bio.png`). Seeds 919, 2 and 3 for R1a
and Pair, 919 and 2 for DEV4NF, seed 919 only for R0. Trained on fir (`2026-11-23_14`–`57`, `2026-11-24_00`–`26`).
Pair B means are over 7 assays (H1 has no ATAC). "Clear" = |Δ| > 2 SE and every pairing of seeds has the same sign.

| Pair | Model | AUC | Sign accuracy | Differential r | Own-track r (cell 1 / cell 2) | Mixture r | CTCF H1 (cell 1 / cell 2) |
|---|---|---|---|---|---|---|---|
| GM12878 + HepG2 | DEV4NF | 0.631 ± .008 | 0.739 ± .006 | 0.333 ± .007 | 0.462 / 0.598 | 0.600 | 1.04 / 0.90 |
| | R1a | 0.642 ± .010 | 0.754 ± .015 | 0.338 ± .014 | 0.474 / 0.624 | 0.628 | 0.87 / 0.62 |
| | Pair | 0.648 ± .004 | 0.797 ± .011 | 0.367 ± .001 | 0.481 / 0.625 | 0.628 | 0.77 / 0.58 |
| K562 + H1 | DEV4NF | 0.653 ± .010 | 0.785 ± .006 | 0.290 ± .012 | 0.483 / 0.419 | 0.512 | 0.76 / 0.66 |
| | R1a | 0.664 ± .004 | 0.789 ± .023 | 0.272 ± .004 | 0.480 / 0.431 | 0.526 | 0.51 / 0.70 |
| | Pair | 0.676 ± .006 | 0.843 ± .010 | 0.328 ± .002 | 0.486 / 0.440 | 0.525 | 0.83 / 0.68 |

- **Bulk:** R1a = Pair within noise on every own-track r and on mixture r, on both pairs. R1a is clearly above
  DEV4NF on HepG2, H1 and the GM12878 + HepG2 mixture. Pair has a slightly lower MSE (GM12878, H1).
- **T3:** Pair is clearly above R1a on sign accuracy (+0.04 / +0.055) and differential r (+0.03 / +0.056) on both
  pairs, and on AUC for K562 + H1 (+0.012). R1a sits at DEV4NF's level here, or below it (differential r on
  K562 + H1, −0.018). The first run's "R1a ≈ Pair" came partly from seed 919 being R1a's best seed (AUC 0.653 vs
  mean 0.642). The library control is at chance for every model (0.050–0.064).
- **CTCF:** the footprint effect (H1) varies a lot between trainings. Pair A seed 3 has none (0.01 on GM12878,
  −0.11 on HepG2), and R1a B seed 919 is at 0.02 on K562 (p 0.06). Otherwise H1 passes everywhere (p < 1e-7).
  It passes in 11 of 12 seed × cell cases for R1a, 10 of 12 for Pair and 8 of 8 for DEV4NF. The first run's ranking
  (DEV4NF > Pair > R1a) is within seed noise. H3 passes in every run. H5 is 1.0 in every run, so it can't tell
  models apart.
- **XI (GM12878 fibers):** on silenced CGI genes, 0.69–0.79 of genes have the expected direction, and both nulls
  stay at ~0.5. No model difference is clear. H3K27me3 is near or below 0.5 for every model, so it is unusable as an
  XI mark.
- **Caveat:** R1a K562 + H1 has no ATAC target (data37), so its ATAC head is untrained. Its XI ATAC (0.43) and CTCF
  H2 (which subtracts pred ATAC) mean nothing. Leave both out for that model.
- **Reading for the story:** trained only on mixtures, R1a matches own-track supervision on bulk accuracy,
  including on mixtures. The cost is in single-fiber differential signals, where it reaches DEV4NF's level but not
  the pair model's.

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
- **Held-out mixture:** HCT116 + MCF-7 at a 0.5 share (the cleanest pair: different tissues, same Fiber-seq
  series). ~~A 3-cell mixture with H9~~: dropped 25 Nov (scope: two cells, 0.5 share; §2, "Scope").
- **Panc1** stays out (its sample identity is in doubt).
- **Sealed X-inactivation cells (25 Nov).** Neither held-out cell works for XI: HCT116 has no Xi, and MCF-7's is
  partial. Three other Fiber-seq lines in `DATA_FIBER/fiber_multi_cell/` have a clear Xi, haplotype tags (HiPhase
  blocks) and no use so far: HG008_3E4, GM25455 (LCL) and HEK293T (`plots_code/26_nov_27/xi_cells_check.py`,
  `plots/26_nov_27/xi_cells/`; numbers in `TODO.md` §3). They have no ENCODE tracks, but XI needs only fibers and
  predictions. Score them once, in stage 5, with the HCT116 + MCF-7 model and DEV4NF. Neither has seen these cells.
  Look at nothing in them before then beyond this Xi check.

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
- ~~More than 2 cells~~ (out of scope, 25 Nov).
- **Biology tests on mixture-trained models:** CTCF footprints on each cell's fibers (without the hole sites), and
  X inactivation. "Trained on a mixture, the molecules still show the right biology" is a stronger claim than
  separating cell types.
- **Which biology tests the sealed test can run (25 Nov):**
  - on HCT116 and MCF-7 fibers (no phasing needed): CTCF footprints, CTCF × CpG methylation, poised vs Polycomb,
    and nucleosome positioning at CTCF;
  - X inactivation on the sealed XI cells (HG008_3E4, GM25455, HEK293T; §2c). Their phasing is in blocks, so the
    Xi haplotype is set per block from the other genes' methylation in the block, leaving out the tested gene
    (a change to the XI script);
  - development-only showcases on GM12878: imprinted control regions per haplotype, Xi-specific CTCF
    (DXZ4, FIRRE).
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
- **Against stage 2 (24 Nov; deferred 25 Nov, one problem run per set; figures in `Nov_27.md` §4):**
  - Criterion 1 holds for own-track recovery. On AUC it holds for GM12878 + HepG2 and is borderline for K562 + H1
    (Δ 0.012, 2 SD of Pair = 0.012).
  - Sign accuracy and differential r fail it on both pairs. Decide whether they enter the criteria, or are
    reported as the cost of mixture supervision.
  - Criterion 3 fails "all 3 seeds" in one case: R1a K562 + H1 seed 919 (CTCF H1 p 0.06). The pair model fails it
    too (seed 3). Consider a criterion over seed means, or a majority of seeds.

### 2g. Stages and cost

Training runs take ~2.2 h each on a 2g.20gb slice (under the 3 h limit), plus evals. Exception: R1a loads slowly
on fir's 3 CPUs (~7 samples/s, 13–14 min per epoch on GM12878 + HepG2), so 15 epochs need about 3.5 h. Plan a
chained `RESUME=latest` job from the start. The last epoch also needs the final-eval reserve (~15 min).

| Stage | What | Runs |
|---|---|---|
| 1 | Protocol, success criteria and test chromosome written down; data / eval configs | — |
| 2–3a | **Done (23 Nov):** R0, R1a, Pair on GM12878 + HepG2, seed 919, `model16`. R1a locked as the recipe (§2b, "Result and decision") | 3 |
| 2 | **Done (24 Nov):** seed noise. R1a and Pair on both dev pairs, 3 seeds each; R1a = Pair on bulk, below it on sign accuracy and differential r (§2b, "Seed noise") | 10 new |
| 3 | ~~Realism: R1a, R1b and R2~~ settled by the first run; R2 dropped, R1b optional later (one run per pair if needed) | 0–2 |
| 4 | **Changed 25 Nov:** no more dev pairs or 3-cell mixture (recipe locked; more pairs would not change it). **Next:** hand-crafted baselines (T11) and the biology tests that also run on the held-out cells (§2e), on the dev models | 0 new main-model runs; baselines |
| 5 | Sealed test: R1a on HCT116 + MCF-7, 3 seeds; DEV4NF without retraining; XI on HG008_3E4, GM25455, HEK293T | 3 |

Implementation: I26–I31 (§5).

### 2h. Done so far (details in the dated reports)

- Pipeline: ENCODE fold-change tracks (8 assays × 8 lines), region files, cell-specific and pairwise loci, V1
  fiber averaging in fold change (default since 20 Oct), split v2 (4 Nov).
- T1 / T2 bulk and cell-specific r: `Oct_23.md`, `Nov_13.md`. Replicate ceiling (I11): `Nov_13_extended.md` §9.
- T3 / T3b mixture test on two pairs: `Oct_23_T3.md`, `Nov_13_extended.md` §3.
- T10 input ablations, 2 seeds: `Nov_13_extended.md` §6, §10. DEV4NF became the main model (18 Nov).
- T5e CTCF footprints and T5b X inactivation: `Nov_20.md`, with locus figures in `Nov_20_figs.md`. CTCF rerun
  without the 75 motif sites lacking bulk short-read data (`ctcf_holes.py`).
- Mixture realism, first run (23 Nov): R0, R1a, Pair on GM12878 + HepG2, one seed, every test; R1a locked as the
  mixture recipe (§2b).
- Seed noise (24 Nov): R1a and Pair on both dev pairs × 3 seeds, every test (§2b, "Seed noise").

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
- **Draft (25 Nov, to confirm before running): the ladder, the rules and the expected results.**
  - **The ladder** (user, 25 Nov). Rungs 2–4 use the same supervision (R1a mixtures), the same channels as DEV4NF
    (m6A, CpG, MSP, nucleosome; no FIRE), the same Softplus + coverage-normalized fiber averaging, loss and budget, and
    3 seeds on both dev pairs:

    | Rung | Model | What it answers |
    |---|---|---|
    | 0 | Floor: bulk on every molecule (and the library control) | What no per-fiber reading gives |
    | 1 | **Rules:** one fixed rule per assay → per-position fiber score in [0, 1] → one fitted affine map per assay | Can an obvious reading of the fiber do it? |
    | 2 | **Feature model, small MLP** (user, 25 Nov), **no external annotations** (user: keep it comparable to the network, which sees only the fiber and the DNA): per-position features from the fiber and the reference sequence only → MLP (2 hidden layers, ~64 units, same output head) | 1 → 2: does learning how to combine simple features help? 2 → 3: is deep learning needed? |
    | 3 | **The network:** R1a (exists) | — |
    | 4 | **Network + rules:** the rung-2 features as extra input channels (annotation-free, for the same reason) | 3 → 4: does the network already capture what the simple features carry? |

    **Rung 2 features,** per position, from the fiber and the DNA only:
    - fiber summaries over ~50, 200 and 1000 bp windows centred on the position: MSP fraction, nucleosome fraction, m6A
      rate, mean CpG methylation, fraction of CpGs covered;
    - the annotation-free parts of the rules: inside an MSP; distance to the nearest MSP edge (capped); is a flanking
      nucleosome (≤ 300 bp / ≤ 1 kb of an MSP); motif-free footprint (a protected 10–40 bp patch inside an MSP);
    - sequence summaries from the reference over the same windows: GC fraction, CpG density and CpG observed/expected
      (a CpG-island proxy computed from the DNA, which the network also sees).

    Not in rung 2: CpG-island, TSS, gene-body and repeat annotations, and the JASPAR CTCF motif. So the motif and gene
    structure, which the network has to learn from the DNA, are not handed to rung 2 either. Control arm: rung 2
    without the footprint feature, since with it the CTCF footprint test is partly circular.
  - **Scale:** a rule gives 0 / 1 per position. Calibrate it with an affine map per assay, fitted on the training
    chromosome by the normal training loop (2 parameters per assay), so it is scored on bulk exactly like the network.
    Per molecule, an output is best read as a probability (a nucleosome is marked or not); the molecule tests are
    rank-based (AUC, sign accuracy, per-site differences), so they don't depend on scale.
  - **Annotations rung 1 may use** (only rung 1; the network and rung 2 get the fiber and DNA only, and that rung 1
    needs them is an argument in itself): CpG islands (UCSC),
    TSSs and gene bodies (GENCODE protein-coding), RepeatMasker, the JASPAR CTCF motif (MA0139.2, p ≤ 1e-5, as in the
    footprint test). Per-position annotation tracks per window, like the DNA one-hot.
  - **"MSP" in the rules means a large MSP (a run of ≥ 151 bp, a nucleosome-free region)** (found 25 Nov while
    building rung 2: fibertools MSPs include the short linkers between nucleosomes, so "an MSP within 1 kb" is true
    almost everywhere). Same for rung 2's rule parts.
  - **The rules,** per fiber and position. "Flanking" = nucleosome calls within the given distance of an MSP edge;
    methylation = the fiber's own CpG calls, mean over ± 500 bp; FIRE is not used (its classifier was trained on DNase
    and CTCF peaks).

    | Assay | Rule: score 1 where | Biology behind it |
    |---|---|---|
    | ATAC | the position is inside an MSP | accessibility is what MSPs measure |
    | CTCF | a CTCF motif hit inside or next to an MSP is protected (motif m6A rate ≤ 0.5 × flanks; same rule as the footprint test), over the motif ± 100 bp | bound CTCF protects its motif from the methyltransferase |
    | H3K4me3 | nucleosomes flanking an MSP (≤ 300 bp; the ± 1 nucleosomes) at a CpG island, fiber unmethylated (< 0.3) | H3K4me3 sits on the nucleosomes around active CGI promoters, which stay unmethylated |
    | H3K27ac | nucleosomes flanking an MSP (≤ 1 kb), fiber not methylated (< 0.5), any CpG context | active promoters and enhancers |
    | H3K4me1 | as H3K27ac, but > 2 kb from any TSS and up to ± 2 kb around the MSP | enhancers and their flanks; depleted at the TSS itself |
    | H3K36me3 | inside a gene body (TSS + 2 kb to TES), fiber methylated (> 0.7), nucleosome-covered | co-transcriptional mark; gene-body methylation follows it (DNMT3B reads H3K36me3) |
    | H3K27me3 | CpG island or CpG-rich (± 1 kb), fiber unmethylated (< 0.3), no MSP within ± 500 bp | Polycomb targets: closed, unmethylated CGIs |
    | H3K9me3 | in an LTR / satellite repeat or a chr19 ZNF cluster, fiber methylated (> 0.7), no MSP within ± 1 kb | constitutive heterochromatin over repeats and ZNF genes |

    Motif-free footprint rule (for the specificity test below): any protected patch (10–40 bp with an m6A rate
    ≤ 0.5 × its flanks) inside an MSP, whatever the sequence.
  - **Expected results** (fixed now; the outcome is reported whatever it shows):

    | Assay | Bulk r, rules (1) vs network (3) | MLP on features (2) vs network (3) | Why |
    |---|---|---|---|
    | ATAC | close (gap < 0.05) | close | MSPs are accessibility |
    | CTCF | network clearly better | network clearly better | the rule fires only at strong motif hits with a footprint; the MLP sees footprints but not the motif, so it can't tell CTCF from other factors |
    | H3K4me3 | network better, modest gap | close | open, unmethylated, CpG-rich: the MLP has all three (CpG density from the DNA) |
    | H3K27ac, H3K4me1 | largest gap | network better | same fiber signature; rung 1 separates them by TSS distance, the MLP has no TSS at all |
    | H3K36me3 | moderate gap | network better | rung 1 gets gene bodies from the annotation; the MLP has only gene-body methylation over ≤ 1 kb |
    | H3K27me3, H3K9me3 | network better, both low | network better | no clean rule; the network can combine closedness, methylation and sequence (repeats) |

    - **Marks with the same fiber signature:** the rule tracks of H3K27ac and H3K4me1 correlate with each other more
      than the true tracks do; the network's tracks come closer to the true correlation.
    - **T3 (pair separation):** rules lower on sign accuracy and differential r, except ATAC.
    - **Biology tests:**
      - CTCF footprints: the rule is the test's own label, so it is a ceiling there, not a competitor;
      - X inactivation: the active-mark rules use the fiber's CpG methylation, so they pass almost by construction
        (no difference from the network expected);
      - H3K27me3 on the Xi: the rule puts more H3K27me3 on the Xa (unmethylated CGIs), like the network. If both fail,
        the information is not in the fiber features;
      - CpG flips: only the rules that use CpG respond, and abruptly (thresholds), while the network responds
        gradually.
    - **Rung 4:** within seed noise of the network. If it is better, the network misses something the simple
      features make explicit (most likely the footprint, on CTCF).
  - **New specificity test for CTCF** (the strongest argument for the network over rules):
    - at footprints on other factors' motifs (NRF1, YY1, REST, ...; JASPAR), the motif-free footprint rule fires,
      but the network's pred CTCF should stay low: it ties the footprint to the CTCF sequence;
    - at bound CTCF peaks without a strong motif hit, does the network still respond to a footprint? The motif rule
      can't.
  - **Implementation:**
    - rules and annotations as per-position input channels computed per window (`fiber_utils`, like the DNA one-hot),
      from annotation tracks built once (`data_prep/`);
    - rung 1 as a model class with no trainable weights besides the per-assay affine map, so the evaluator and
      `molecule_predict.py` run on it unchanged;
    - rung 2 as a small MLP on the annotation-free feature channels (computed per window from the fiber and the
      DNA), with the same output head, fiber averaging and loss;
    - rung 4 as `model16` with the rung-2 feature channels added (`input_flags` extended).
- **Status.** Idea (17 Nov); plan updated 18 Nov; ladder, rules and expected results drafted 25 Nov. Confirmed
  25 Nov: rung 1 as drafted (with annotations); rung 2 a small MLP without external annotations.
  - **Rungs 2 and 4 built (25 Nov):** `handcrafted.py` (`FiberFeatures`: 31 fixed features from the fiber channels and
    the reference DNA, computed inside the model, so the evaluator, `molecule_predict.py` and `--perturb` work
    unchanged; `FeatureMLP`), `models.py` (`mlp01` = `MLPBaseline`; `unet03` with `extra_features: true`), configs
    `model22_mlp_handcrafted` (6.7 k parameters) and `model23_uct_mid_8A_no_fire_msp_handcrafted`, slurm cases
    `r1a_mlp`, `r1a_mlp_b`, `r1a_hc`, `r1a_hc_b`.
  - **Seed 919 submitted on fir (25 Nov),** both dev pairs: jobs `2026-11-25_17`–`36` (training, chained resume,
    eval with library controls and the balanced mixture, CTCF molecules with CpG flips, TSS molecules). Seeds 2 and 3
    after the first epochs look sane.
    `_17` died (code pushed while it ran; see memory) and was resubmitted as `_45`–`49`.
  - **Rung 1 built and submitted (25 Nov):** `data_prep/build_annotation_tracks.py` (uint8 bits per base in
    `data/misc/annotation_tracks/`: CGI, TSS ± 2 kb, gene body, soft-masked repeat, ZNF gene, CTCF motif hit; built for
    chr20, chr21, chr22, chrX), `dna_type: ref_annot` (the loader adds `batch['annot']`; `molecule_predict.py` too),
    `handcrafted.rule_scores` and `models.RuleBaseline` (`rules01`, 16 parameters: Softplus(a s + b) per assay),
    config `model24_rules`, `train05` (3 epochs, lr 1e-2), slurm cases `r1a_rules`, `r1a_rules_b` (`TRAIN=train05`;
    evals with `EPOCH=3`). Seed 919 submitted: `2026-11-25_37`–`44`. Repeats are the soft-mask for now (user,
    25 Nov: skip the RepeatMasker download, decide after the results). Implementation I24 (learned), I25 (rules).
  - **First results, seed 919 (25 Nov, end-of-training mixture r on the chr21 mini balanced mixture, eval31 /
    eval32):**

    | Rung | GM12878 + HepG2 | K562 + H1 |
    |---|---|---|
    | 1 rules + calibration | 0.316 | 0.232 |
    | 2 MLP on features | running (val r 0.465 at epoch 12) | 0.364 |
    | 3 R1a (seeds 919 / 2 / 3) | 0.633 / 0.633 / 0.617 | 0.515 / 0.534 / 0.531 |
    | 4 network + features | 0.620 | 0.529 |

    Big steps up to the network, nothing from adding the features to it. Rung 1's calibration barely moves (val r
    0.310 → 0.311): its score is the rules'. Evals and molecules done for rungs 1 and 4 (both pairs) and rung 2 B;
    rung 2 A chain (`_46`–`49`) still running.
  - **Seeds 2 and 3 of rungs 2 and 4 submitted (25 Nov):** both pairs, same chains, jobs `2026-11-25_50`–`89` on fir
    (run prefixes `MIXR1A_MLP_seed<n>`, `MIXR1A_HC_seed<n>`). Rung 1 stays at one seed (16 parameters). Next: pull,
    then the ladder side by side in the seed analysis (T3, bulk + ceiling, mixture r, CTCF footprints, CpG flips).

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
`molecule_predict.py`); Xa / Xi split for GM12878 (I19, partly); balanced stacks at a fixed share for R1a
(`mix_balance`, `data36`, `eval31`; 23 Nov).

**For the main plan (§2):**

| ID | Change | For | Notes |
|---|---|---|---|
| I26 | **(Later, optional; R1a locked 23 Nov)** **Population mixture target (R1b):** option `mix_target: population`, weights = each cell's expected share of the DNA at the position (π_c × local coverage / genome-wide coverage, renormalized), independent of the fibers sampled; fiber counts multinomial at the fixed share, redrawn (not capped) when one exceeds what is available | §2b R1b / R2 | `data_utils.py` mixed dataset, next to `mix_balance`; the current behaviour stays the default. Needs a per-cell coverage track (CN proxy) |
| I27 | **(Dropped 23 Nov: R1a passed, R2 not needed)** **Several samples with fixed fractions:** option `mix_samples: [0.25, 0.5, 0.75]`; each window draws one sample, and fibers and target follow that sample's fraction | §2b R2 | Same place as I26 |
| I28 | **(Out of scope 25 Nov: two cells, 0.5 share in this paper; follow-up paper)** **Mixtures of more than 2 cell types:** fractions per cell (fixed or per sample), target over all cells; mixed eval and paired readouts generalized (one-vs-rest fiber-origin AUC, per-cell own-track recovery) | §2b, §2e | `mix_range` now needs exactly 2 cell types. Differential bins for n cells: per cell vs the rest |
| I29 | **Sealed test chromosome:** pick it by fixed criteria, add its tiles / mini set to `regions_spec.yaml`, loci for the held-out pair (HCT116 + MCF-7) on it, eval configs. Build, never evaluate until stage 5 | §2c | `build_regions.py`, `build_loci.py`, `loci_spec.yaml`. Also the CTCF-site and TSS target BEDs on it for the biology tests (held-out cells only at the end) |
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
| R0 (mixture, `model16`) | `data27` | GM12878 + HepG2 mixtures, share redrawn per window | nibi | `2026-11-23_00_train_r0_nf` | `26-11-23_T17-51-43_train_MIXR0_NF_data27_mix_GM12878_HepG2_200U_model16_uct_mid_8A_no_fire_msp_train04/Model_epoch_15.pt` | 2 h 16 min; realism first run (§2b) |
| **R1a** (mixture, `model16`) | `data36` | GM12878 + HepG2 mixtures, share fixed at 0.5, balanced stack | nibi | `2026-11-23_01_train_r1a_nf` | `26-11-23_T17-52-13_train_MIXR1A_NF_data36_mix_bal_GM12878_HepG2_200U_model16_uct_mid_8A_no_fire_msp_train04/Model_epoch_15.pt` | **the mixture recipe from 23 Nov**; 2 h 20 min (§2b) |
| Pair (`model16`) | `data28` | GM12878 + HepG2 (own tracks) | nibi | `2026-11-23_02_train_pair_nf` | `26-11-23_T17-54-40_train_PAIR_NF_data28_GM12878_HepG2_200U_model16_uct_mid_8A_no_fire_msp_train04/Model_epoch_15.pt` | ceiling for mixture supervision with DEV4NF's channels; 2 h 18 min (§2b) |
| R1a and Pair seeds, both dev pairs (`model16`) | `data36`, `data28`, `data37`, `data34` | GM12878 + HepG2 / K562 + H1: R1a mixtures and own tracks | fir | `2026-11-23_14`–`23` (R1a resumed in `2026-11-24_00`–`03`, `17`–`18`) | `*_train_{MIXR1A_NF,PAIR_NF}[_seed{2,3}]_{data36_mix_bal_GM12878_HepG2,data28_GM12878_HepG2,data37_mix_bal_K562_H1,data34_K562_H1}_200U_model16_uct_mid_8A_no_fire_msp_train04/Model_epoch_15.pt` | seeds 2, 3 (and 919 for K562 + H1); seed analysis §2b. R1a K562 + H1 has no ATAC target |

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

