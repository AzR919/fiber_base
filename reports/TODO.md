# TODO

Current plan only. Details, the test catalogue (T1–T11), the comparison ladder, the implementation list, the
backlog, ideas and the registry of trained models are in `TODO_extended.md`. Results go in the dated reports.

## Where we stand (24 Nov)

- **Pipeline:** ENCODE fold-change tracks, region files, cell-specific and pairwise loci, per-fiber predictions at
  target BEDs (`molecule_predict.py`). Split v2: 4 dev cell types (GM12878, K562, HepG2, H1), 3 held out (H9,
  HCT116, MCF-7), Panc1 separately. Train chr20, validation chr22, evaluation chr21.
- **Main model: DEV4NF** (DEV4 without the FIRE channel, `model16`). Same bulk accuracy as DEV4, no bulk-assay
  knowledge in the input, and the only model whose CTCF footprint response is strong and specific in both seeds
  (`Nov_13.md`, `Nov_20.md`).
- **Done:**
  - input ablations (no FIRE, m6A + CpG, no accessibility, no CpG), 2 seeds each;
  - ENCODE replicate ceiling;
  - CTCF footprints (rerun without the motif sites that have no bulk data);
  - X inactivation per haplotype at silenced CGI promoters;
  - locus figures for both (`Nov_20_figs.md`);
  - the mixture test on two pairs, one seed;
  - the first realism run (23 Nov): mixtures at one fixed share (R1a) lose nothing against the current ones (R0).
    R1a is now the mixture recipe.
  - seed noise (24 Nov): R1a and pair models on both dev pairs × 3 seeds, every test.
- **What the mixture test showed:** trained only on a pair's mixtures, the model separates that pair's molecules as
  well as a model trained on their own tracks. On a pair it never saw, the type of supervision doesn't matter;
  having seen the cell types does.
- **Two problems to fix before claiming more:**
  - Molecule readouts move 2–10× between training seeds. Claims need 3 seeds. Measured on 24 Nov: T3 and bulk
    move ±0.005–0.02, but the CTCF footprint effect can vanish in a single seed.
  - The test set is no longer clean. The held-out cells and chr21 both entered decisions (ceiling fractions,
    ablation tables, the DEV4NF choice).

## 1. Main direction: the mixture-trained model (details: `TODO_extended.md` §2)

**The claim to sell.** In practice you can run Fiber-seq and ChIP on the same mixed sample (a tissue), but you
can't separate it into pure cell types. A model trained on that mixture's own bulk should still impute each
molecule correctly. Our unseen-pair result predicts it beats a model pretrained on other cells (DEV4NF), because
it has seen the sample's cell types.

**Test hygiene, from now on:**
- Development: the 4 dev cell types; train chr20, validation chr22, development evaluation chr21 (and chrX for the
  X-inactivation test).
- Sealed test: the held-out cell types (HCT116, MCF-7, H9) on a fresh chromosome, never evaluated before (to
  pick, e.g. chr19 or chr18).
- The test is run once, with the locked recipe, and reported whatever it shows. The earlier use of the held-out
  cells on chr21 is stated in the paper.

**The mixture recipe: R1a (decided 23 Nov,** `TODO_extended.md` §2b, "Result and decision").
- **What it is:** one fixed share for the whole sample (0.5 for a pair), each window's fiber stack balanced to
  exactly that share (up to 100 fibers; stacks under 10 are redrawn), the target unchanged
  (`mix_range: [0.5, 0.5]`, `mix_balance: true`; `data36`, `eval31`). It mirrors one real sample with one composition.
- **First run** (GM12878 + HepG2, seed 919, `model16`):
  - R1a matches R0 on the T3 readouts (AUC 0.653 vs 0.654, differential r 0.352 vs 0.330) and is equal or better
    on bulk, level with the pair model;
  - the CTCF and X-inactivation tests keep their direction in every model;
  - R1a's CTCF footprint effect is the smallest of the four (about 30–40% below DEV4NF). Across seeds, this ranking
    is within noise.
- **Seed noise** (24 Nov; 3 seeds, both dev pairs; `TODO_extended.md` §2b, "Seed noise"):
  - **bulk:** R1a = Pair on own-track r and on mixture r for both pairs; R1a is above DEV4NF on HepG2, H1 and the
    GM12878 + HepG2 mixture;
  - **T3:** Pair is clearly above R1a on sign accuracy (+0.04 / +0.055) and differential r (+0.03 / +0.056).
    There, R1a is at DEV4NF's level. The first run's "R1a ≈ Pair" was partly a lucky seed;
  - **biology:** CTCF H1 passes in 11 of 12 R1a seed × cell cases (Pair 10 of 12; one Pair seed has no
    footprint effect at all). XI holds in every model, with no clear differences;
  - **story:** mixture supervision costs nothing on bulk accuracy. On the single-fiber differential signals it
    reaches DEV4NF's level, not the pair model's;
  - R1a K562 + H1 has no ATAC target (H1 has no ATAC), so leave out its ATAC-based readouts.
- **R2 dropped** (one fixed share is enough). **R1b later, optional:** population target with multinomial counts,
  as a robustness check if a reviewer asks.
- **Limitations, not runs:** synthetic mixtures combine two Fiber-seq libraries and two separate ChIP experiments;
  a real sample has one of each.

**Stages:**
1. **Write the protocol and success criteria down before running** (draft in `TODO_extended.md` §2).
2. **Seed noise: done (24 Nov),** see above. Still to settle: the success criteria against these numbers
   (`TODO_extended.md` §2f). Do sign accuracy and differential r count, or are they reported as the cost? Is CTCF
   judged per seed or on the seed mean?
3. **Lock the recipe on the dev cells (next):**
   - more of the 6 dev pairs, for the spread across pairs;
   - one 3-cell mixture;
   - the CTCF and X-inactivation tests on mixture-trained models (GM12878 in the mix for X inactivation).
4. **Final test, once:**
   - train with the locked recipe on held-out mixtures (HCT116 + MCF-7; a 3-cell one), 3 seeds;
   - score against each held-out cell's own track (used for scoring only) on the sealed chromosome;
   - compare with DEV4NF applied without retraining.

**Comparisons:**
- mixture vs pair model with the same cells and channels: does mixture supervision lose anything?
- mixture vs DEV4NF: train on your sample, or use a pretrained model?
- floors: "bulk on every molecule", the library control, a random split of one library.

## 2. Comparison against simpler models (T11, details: `TODO_extended.md` T11)

"The model just reads the fiber data" can't be argued away, but it can be measured. Run simple per-fiber models
through the same tests, with the same supervision as the main model (R1a mixtures):
1. Write the expected results down.
2. Hand-written rules on the molecule tests, from the existing per-fiber tables (no new predictions).
3. Shuffle controls (T7): shuffled inputs across fibers or loci should break the per-fiber predictions.
4. A learned summary model (per-fiber window averages → linear map, same fiber averaging and training), with and
   without ≈ 25 bp windows, 3 seeds.

Careful: on the CTCF footprint test a footprint rule is the test's own label, so it is a ceiling there, not a
competitor.

## 3. Biological tests still open (T5)

- **X inactivation follow-ups:**
  - restrict to genes expressed in GM12878 (RNA-seq): PTCHD1 suggests the H3K27me3 "failure" is partly
    non-expressed genes;
  - check the escape list against Tukiainen 2017;
  - H3K27me3 over gene bodies and Xi domains, not only promoters.
- **CTCF on chrX per haplotype:** footprint frequency on the Xa vs the Xi at chrX motifs, with DXZ4 / FIRRE as
  Xi-specific controls. H3K36me3 over gene bodies of silenced genes.
- **CpG:**
  - in-silico flips of a molecule's CGI CpGs (I21);
  - held-out methylation with the no-CpG model (T5a), which now exists.
- **Imprinted control regions** (showcase) and **allele-specific ChIP at heterozygous SNPs** (GM12878; I20).
- **External single-molecule histone data** (DiMeLo-seq, ChromSMF, Yue 2022): search before building more (T8).

## 4. Later

- **Bulk metrics:**
  - r at several resolutions (bp, 200 bp, 1 kb; I17);
  - assay-specific r given ATAC (I18);
  - report assays by difficulty.
- **Model:**
  - per-assay learned scale, then a per-assay smoothing kernel (T4 V2 / V3);
  - longer context for broad marks;
  - m6A density per fiber.
- **Bulk reference on fiber-averaged inputs (T6).**
- **Arabidopsis data.**
