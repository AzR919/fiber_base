# TODO

Current plan only. Everything else (tests T1–T10, comparison ladder, implementation list, backlog, ideas, the
registry of trained reference models) is in `TODO_extended.md`. Results go in the dated reports.

## Where we stand

- **Pipeline:** data setup (ENCODE fold-change tracks, region files, cell-specific and pairwise loci), 4 dev /
  4 held-out cell types (split v2, 4 Nov: Panc1 moved to held out as a hard cell, its sample identity is in
  doubt; `configs/README.md`), fibers averaged in fold change (V1, default since 20 Oct).
- **Bulk (`Nov_13.md`, V1):** beats the cross-cell average on MSE, mostly not on Pearson; cell-type r
  positive everywhere, held-out included. Strong on ATAC and active marks, weak on H3K36me3 / H3K9me3 / CTCF.
  Dropping Panc1 from training costs nothing; specialists beat the dev model by 0.03–0.06 r.
- **Single molecules (`Oct_23_T3.md`, `Nov_13.md`), mixture test (T3, T3b) done on both pairs:**
  - Trained only on a pair's mixtures, the model separates that pair's molecules as well as a model trained on
    their own tracks.
  - On the unseen pair, mixture and own-track models do the same. Transfer is lopsided: fine toward GM12878–HepG2,
    poor toward K562–H1.
  - Failures sort into track-scale outliers (H1 CTCF, HepG2 / H1 H3K9me3), compression on unseen cell types, and
    cell-level offsets.
- **Why the plan changes (7 Nov).**
  - Without single-molecule ground truth, the current readouts are proxies. Fiber-origin AUC assumes every
    molecule of the cell with the higher track should score higher, which molecular heterogeneity contradicts.
  - So the main evidence should be biological, as in the single-molecule literature: SMFNet (F1-hybrid allele
    tests, motif perturbations) and the Fiber-seq / FIRE / fibertools papers (haplotypes, X inactivation,
    imprinting, CTCF footprints, null controls).
  - These papers validate measured accessibility. None validates histone marks on single molecules, so the claim is
    new and the tests must be designed for it.
  - The current metrics stay as sanity checks.

## 1. Reference models

All V1, dev-config setup (chr20, half cCRE / half random, `model15`, `train04` = 15 × 5000 samples). Checkpoints
in `TODO_extended.md` §8.

- **DEV4** (`data24`, done): all 4 dev cell types; the main dev model and the default for the tests below.
- **Specialists** (`data29`–`data31`, `data33`, done): one per dev cell type.
- **Pair models A / B** (`data28`, `data34`) and **mixture models A / B** (`data27`, `data35`): done.

## 2. Biological validation of single molecules (T5, T10 in `TODO_extended.md`)

**Rule for every test:**
1. It checks something the model was never trained on.
2. The expected direction comes from the literature and is written down before running.
3. It has a null control.

A "bulk on every molecule" model must fail it.

### 2a. Haplotype tests (headline)

Why these are clean for us:
- Every Fiber-seq CRAM carries HiPhase haplotype tags (HP/PS). In a chr21 sample the share of tagged reads is
  GM12878 93%, K562 100%, HepG2 100%, H1 98%, H9 55%, MCF-7 18%.
- The model's DNA input is the *reference* sequence (`dna_type: ref`), and the training targets are unphased. So
  any difference it predicts between the two copies comes from the molecules' own chromatin signals, and was never
  supervised.
- **Caveat:** at SNPs that create or destroy a CpG, the CpG channel sees the genotype. Exclude those SNPs.

| Test | Cell line, loci | Expected (literature) | Null |
|---|---|---|---|
| **X inactivation** | GM12878. The Xi is maternal: 99% of XIST transcripts are maternal (FIRE paper). Needs trio phasing or a per-phase-block Xi label | Xi molecules: H3K27me3 / H3K9me3 up, H3K4me3 / H3K27ac down at TSSs active only on the active X. Escape genes (UBA1, KDM5C, KDM6A, ...) active on both. CTCF on both copies at most sites; Xi-specific at DXZ4 / FIRRE / ICCE | Autosomal genes, and male lines (HepG2, H1) where the X has one copy |
| **Imprinted control regions** | H19/IGF2, KvDMR1, SNRPN, PEG3, MEG3, GNAS (plus MAGEL2, ZDBF2) in all phased lines | Unmethylated allele: H3K4me3 + accessibility (CTCF at the H19 ICR). Methylated allele: H3K9me3 | Non-imprinted genes with matched expression. Showcase figures, not statistics |
| **Allele-specific ChIP at heterozygous SNPs** | GM12878 first: AlleleDB / ADASTRA calls for CTCF, H3K27ac, H3K4me3. Other lines if available | Predicted per-haplotype difference has the sign of the ChIP allelic imbalance | Balanced SNPs: predicted difference ~0 |

### 2b. CpG methylation, with directions fixed in advance

- H3K4me3 is low on molecules with methylated CpG-island promoters (CFP1 binds unmethylated CGIs).
- H3K36me3 goes with gene-body methylation (DNMT3B is recruited by H3K36me3).
- H3K9me3 goes with methylation at ICRs and repeats.
- H3K27me3 is largely antagonistic to methylation at CGIs.
- CTCF is blocked by methylation at its motif (H19 ICR).

Three experiments:
1. **In silico (no retraining):** flip the CpG calls on real molecules (methylate / unmethylate the CGI CpGs) and
   check that each assay moves in the stated direction.
2. **Retrain without CpG** (`input_flags` [1, 0, 1, 1, 1]): H3K4me3, H3K36me3, H3K9me3 and CTCF should lose more
   than ATAC, H3K27ac and H3K4me1. This also tests whether the H1 CTCF shift comes from CpG (H1 0.27 vs K562
   0.085 methylation).
3. **Held-out methylation:** with the no-CpG model, the molecule's measured methylation becomes held-out data.
   The predicted marks should correlate with it in the directions above (T5a).

### 2c. FIRE dependence

- The FIRE classifier was trained on GM12878 using DNase **and CTCF ChIP** peaks as positives. So the FIRE channel
  carries bulk-assay knowledge, CTCF above all, and it is derived from the m6A calls we already input.
- **Hypothesis:** dropping FIRE (`model16`, [1, 1, 1, 1, 0]) costs little. If it costs little, the no-FIRE model
  becomes the main one, which is a cleaner claim. A large drop means a shortcut; check CTCF and GM12878 first.

### 2d. More than accessibility?

- **Accessibility-only baseline:** a per-assay monotone function of each fiber's accessibility, fitted to the bulk
  (`TODO_extended.md` §4a, rule-based imputer). Our offset breakdown (`Nov_13.md`) suggests the models largely
  follow accessibility.
- **Same-molecule cases where accessibility and the mark disagree:**
  - bivalent promoters in H1 (H3K4me3 and H3K27me3 together, unlike in the differentiated lines);
  - H3K36me3 and H3K27me3 rarely on the same molecule (PRC2 is inhibited by H3K36me3);
  - repressive marks that are not just inverted accessibility.
- **Retrain with accessibility removed** (CpG + DNA only, [0, 1, 0, 0, 0]): how much is left per assay.

### 2e. CTCF footprints (structural check, from the Fiber-seq papers)

- Per-fiber CTCF predictions at CTCF motifs, against footprint calls on the same fiber: should rank bound >
  accessible-unbound > closed.
- Reference point: in K562, about 30% of accessible fibers at CTCF sites are bound (Stergachis 2020).

### 2f. External single-molecule histone data

- Before building heavily, search for public data in our cell lines from DiMeLo-seq, ChromSMF (Krebs lab, cited by
  SMFNet) or Yue et al. 2022 (nanopore histone marks + methylation). It would be the only direct ground truth (T8).

### Order and cost

1. **No retraining** (DEV4, plus specialists / pair models where useful): 2f search, haplotype tests (GM12878 X
   inactivation first), CTCF footprints, in-silico CpG flips, accessibility-only baseline.
   - Needs saved per-fiber predictions (I2), a chrX window set (regions exclude chrX, I7), haplotype-split eval
     (I19) and the allele-specific ChIP calls (I20).
2. **Three DEV4 retrains:** no CpG, no FIRE, no accessibility. About 2.2 h each on a 2g.20gb slice, plus evals.
3. **Score the ablations mainly through the step-1 tests,** not bulk Pearson. For example, the no-CpG model should
   lose the H3K9me3 allele differences but keep the accessibility-driven ones.

**Later / deferred:**
- routed specialists as the "told the cell type" reference for T3b;
- the other channel ablations (FIRE only, CpG only, no DNA, continuous scores; `TODO_extended.md` T10);
- a second mixture model;
- A549 / split v3.

## 3. Right after

- **Bulk tables for DEV4** in the split v2 layout (dev 4, held out 3, Panc1 separately).
- **Assays: keep the 8, report them by difficulty** so the model's limits are stated, not hidden in averages:
  easy (ATAC, H3K4me3), easy–medium (H3K27ac, H3K4me1), medium (CTCF), hard (H3K27me3, H3K36me3, H3K9me3: broad,
  repressive or repeat-rich). Assay-specific metrics:
  - **Replicate ceiling:** ENCODE replicate 1 vs 2 per assay (plain r and cell-type r); report each assay as a
    fraction of it, separating "hard for the model" from "noisy target" (I11).
  - **r at several resolutions** (bp, 200 bp, 1 kb): bp-level r penalises broad marks.
  - **Assay-specific r:** partial correlation of predicted and true mark given the true ATAC track: is each mark
    more than accessibility (H3K27ac vs H3K4me1 vs CTCF)?
  - **Track-scale outliers:** flag (cell, assay) tracks that sit far from the other lines at shared sites (H1 and
    MCF-7 CTCF, HepG2 H3K9me3; `plots_code/26_nov_13/pair_offsets.py`) and report them separately.
- **Per-assay learned parameters** (test, V2/V3 in `TODO_extended.md` T4): a learned scale (and offset) per assay
  mapping per-fiber outputs to the ENCODE track, the same for every cell type; then a per-assay smoothing kernel for
  ChIP fragment blurring (V3). Should fix signal height (predictions ~half the true size) without cell-type
  knowledge.
- **Longer context for broad marks** (H3K36me3, H3K27me3, H3K9me3): longer input window with only the central 5 kb
  scored (regions allow up to 20 kb); then the whole read as context.
- **m6A density per fiber:** check per library; drop only the failed-labelling tail or use it as an input (a hard
  filter would bias against compact heterochromatin molecules).
- **Later:** Arabidopsis data.
