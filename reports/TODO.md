# TODO

Current plan only. Details, the test catalogue (T1–T11), the comparison ladder, the implementation list, the
backlog, ideas and the registry of trained models are in `TODO_extended.md`. Results go in the dated reports. What
has been finished, newest first: `TODONE_26_11.md` (one file per month).

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
- **Scope of this paper (user, 25 Nov): two cell types at a 0.5 share only.** Real samples have more cell types at
  unequal, unknown shares. That is harder and goes to a follow-up paper; the paper says so. So no 3-cell mixtures
  (I28) and no other shares here.
- **Limitations, not runs:** synthetic mixtures combine two Fiber-seq libraries and two separate ChIP experiments;
  a real sample has one of each. Fixed 0.5 share of two cells (above).

**Stages:**
1. **Write the protocol and success criteria down before running** (draft in `TODO_extended.md` §2).
2. **Seed noise: done (24 Nov),** see above. The success-criteria details (sign accuracy / differential r; CTCF
   per seed) are deferred: one problem run per set (user, 25 Nov).
3. **Recipe locked on the dev cells (25 Nov):** R1a, two cells, 0.5 share. No more dev pairs (user: they would not
   change the scheme). Before the final test: the hand-crafted baselines (§2) and the biology tests that also run
   on the held-out cells (§3).
4. **Final test, once:**
   - train with the locked recipe on the held-out mixture (HCT116 + MCF-7), 3 seeds;
   - score against each held-out cell's own track (used for scoring only) on the sealed chromosome;
   - compare with DEV4NF applied without retraining;
   - biology tests: CTCF footprints, CTCF × CpG and poised vs Polycomb on HCT116 / MCF-7 fibers; X inactivation
     on HG008_3E4, GM25455 and HEK293T (unseen lines with a clear Xi, §3).

**Comparisons:**
- mixture vs pair model with the same cells and channels: does mixture supervision lose anything?
- mixture vs DEV4NF: train on your sample, or use a pretrained model?
- floors: "bulk on every molecule", the library control, a random split of one library.

## 2. Comparison against simpler models (T11, details: `TODO_extended.md` T11)

**The ladder (25 Nov; rules, features and expected results in `TODO_extended.md` T11):**
0. floor: bulk on every molecule;
1. fixed rules per assay, calibrated; these may use annotations (CGIs, TSSs, gene bodies, repeats, CTCF motif);
2. a small MLP on simple per-position features from the fiber and the DNA only, no external annotations (user:
   comparable to the network);
3. the network (R1a);
4. the network + the rung-2 feature channels.

1 → 2 asks whether learning to combine the rules helps, 2 → 3 whether deep learning is needed, 3 → 4 whether the
network already captures what the rules know.

**Status (26 Nov): done, `Nov_27.md` §6** (rungs 2–4 with 3 seeds, rules 1 seed, both pairs; `ladder.py`).
- **2 → 3:** the network is needed: +0.16 mixture r on both pairs, clearly better AUC and differential r, and the only
  rung with CTCF footprints (the MLP has none, though a footprint feature is an input).
- **3 → 4:** the features add nothing; CTCF footprint effect slightly weaker with them.
- Rules help only where the annotation is the signal (H3K36me3 / gene bodies, H3K4me3 / CGIs), still below the
  network. The MLP shows the CpG-flip effect too, so section 5's claim doesn't separate the network from simple models.
- Open: rerun `ladder.py --step summary` and `plots_code/26_dec_04/bulk_models.py` (`Dec_04.md` §2) when the rung-4
  seed-3 evals finish (`2026-11-25_82`, `_87`); the CTCF
  specificity test (at footprints on other factors' motifs a motif-free rule fires, but the network's CTCF should
  not); shuffle controls (T7: shuffled inputs across fibers or loci should break the per-fiber predictions);
  RepeatMasker for the H3K9me3 rule (user: decide after the results).
- On the CTCF footprint test the rules' CTCF is the footprint, the test's own label: a ceiling there, not a competitor.
- H3K36me3 with larger windows: not now (25 Nov); its poor pair-A result is likely a data limit, check the
  differential ceiling first.

## 3. Biological tests still open (T5)

**Which tests can run on the sealed test (25 Nov).** Without phasing: CTCF footprints, CTCF × CpG methylation,
poised vs Polycomb, nucleosome positioning at CTCF. X inactivation needs a female line with an Xi and haplotype
tags. Check over 19 Fiber-seq lines (`plots_code/26_nov_27/xi_cells_check.py`, `plots/26_nov_27/xi_cells/`):
- **Clear Xi, unseen by every model:** HG008_3E4 (chrX CGI promoters: 0.46 of fibers methylated, 76% of
  promoters bimodal, XIST unmethylated on one haplotype only, 1.00 vs 0.05), GM25455 LCL (0.34, 67%, 0.83 vs 0.27),
  HEK293T (0.28, 50%, 0.96 vs 0.28); GM25456 fibroblast weaker (0.25, 40%). These lines have Fiber-seq only
  (no ENCODE tracks), but XI needs only fibers and predictions, so they can be the sealed XI test.
- **Held-out pair:** HCT116 has no Xi (male; its chrX and chr21 are both hypermethylated). MCF-7 is partial and
  messy (0.27, 33% bimodal; XIST unmethylated on one haplotype, but few fibers). H9 has the Xi methylation but has
  mostly lost the unmethylated XIST (eroded Xi). Dev males / no Xi: K562, HepG2, H1.
- **Phasing:** the other lines have HiPhase blocks (PS), ~2–3 CGI promoters per block on chrX, not chromosome-wide
  like GM12878. The Xi haplotype has to be decided per block, e.g. from the other genes' methylation in the same
  block (leave the tested gene out).

**New tests to build on the dev cells, in order (25 Nov; the first three also run on the sealed test):**
1. **CTCF × CpG methylation** (prof): **done 25 Nov** (`Nov_27.md` §5). **Claim kept (user, 25 Nov): CpG
   methylation lowers active marks locally.** In-silico methylation of the motif ± 100 bp lowers pred CTCF, ATAC
   and H3K4me3 in every model and seed, mixture-trained R1a included; the same flips 1.5 kb away do ~nothing. A
   CTCF-specific effect can't be tested: the data show the methylation–binding link between sites but barely within
   a site, molecule by molecule (K562 only). Rerun on HCT116 / MCF-7 in the sealed test
   (`molecule_predict.py --perturb`).
2. **X inactivation with block phasing:** adapt the XI script to HiPhase blocks (Xi set per block, tested gene
   left out). Check it on GM12878 by pretending its phasing is in blocks, so the sealed run needs no new code.
3. **Poised vs Polycomb** (prof): genome fractions from bulk H3K4me3 / H3K27me3 (poised = +K4me3 +K27me3,
   Polycomb = −K4me3 +K27me3), then per fiber split by CpG and MSP; do pred K4me3 / K27me3 follow? H1 is the
   strongest dev case. **Defined 27 Nov** (`TODO_extended.md` T5g): 1 kb tiles, FC ≥ 2 marked / < 1 unmarked,
   tests D1–D2 (data), M1–M3 (model), C (CpG flips); DEV4NF, R1a, Pair, rules, MLP. **Done 27 Nov**
   (`Dec_04.md` §1): poised tiles are unmethylated CGIs with about half their fibers open; the networks give the open
   fibers H3K4me3 and the closed ones H3K27me3, and lower H3K27me3 on methylated fibers. Weak spot: H1's methylated
   Polycomb domains. Rerun on HCT116 / MCF-7 in the sealed test.
4. **Nucleosome positioning at CTCF** (prof): phased flanking nucleosomes on footprinted fibers; does pred CTCF
   follow? Needs per-position arrays (`--save_arrays`).
5. GM12878-only showcases: imprinted control regions per haplotype (H19/IGF2: CTCF on the unmethylated allele),
   Xi-specific CTCF (DXZ4, FIRRE).
6. Later / if time: m6A ↔ CpG prediction from each other (prof); training for seed-stable per-fiber predictions
   (prof's "identifiable models"; first, measure the agreement of per-fiber predictions between seeds); an adapter
   on a pilot chromosome for experiment-specific artifacts.

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
