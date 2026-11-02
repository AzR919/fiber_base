# TODO

Current plan only. Everything else (tests T1–T9, comparison ladder, implementation list, backlog, ideas, the
registry of trained reference models) is in `TODO_extended.md`. Results go in the dated reports.

## Where we stand

- **Pipeline:** data setup (ENCODE fold-change tracks, region files, cell-specific and pairwise loci), 5 dev /
  3 held-out cell types, fibers averaged in fold change (V1, default since 20 Oct).
- **Bulk (`Oct_23.md`, `Oct_30.md`):** beats the cross-cell average on MSE, mostly not on Pearson; cell-type r
  positive everywhere, held-out included. Strong on ATAC and active marks, weak on H3K36me3 / H3K9me3. Still V0
  numbers with the old baseline.
- **Single molecules (`Oct_23_T3.md`):** trained only on GM12878 + HepG2 mixtures, the model separates their
  molecules (fiber-origin AUC 0.64) as well as a model trained on their own tracks.
- **Open:** does it hold for cell types the model never saw, and which inputs drive the predictions?

## 1. Reference models (train first)

All V1, dev-config setup (chr20, half cCRE / half random, `model15`, `train02` = 25 × 1000 samples). Log
checkpoints in `TODO_extended.md` §8.

- **DEV5 (V1):** all 5 dev cell types. The existing DEV5 runs are V0; also gives the updated bulk numbers.
- **5 specialists:** one per dev cell type. "Told the cell type" reference: fibers routed to their own cell
  type's model.
- **GM12878 + HepG2** (`data28`, done): main model A below.

## 2. Unseen-pair test (T3b) and input ablations

**Question / hypothesis.** A model trained on two cell types separates the molecules of two *other* cell types at
the loci where they differ, close to a model that saw them, because it maps a molecule's own accessibility (+ DNA)
to its marks. Accessibility channels should carry most of it; CpG adds to the CpG-island marks.

**Design.** Two main pairs, each tested on the three pairs of the other dev cell types (B checks that results do
not depend on the training pair and doubles as a replicate):

| | Trained on | Unseen test pairs |
|---|---|---|
| A | GM12878 + HepG2 | K562–Panc1, K562–H1, Panc1–H1 |
| B | K562 + Panc1 | GM12878–HepG2, GM12878–H1, HepG2–H1 |

H1 has no ATAC; B's GM12878–HepG2 pair crosses Fiber-seq experiments.

**References per test pair:** (1) a pair model trained on that pair's own tracks (matched: differs only in having
seen the pair; 5 new runs, K562+Panc1 doubling as main B); (2) DEV5; (3) routed specialists.

**Readouts per test pair × assay, at the pair's differential bins:** fiber-origin AUC (0.5 = no separation; no
upper bound claimed, the references show where good models land), differential r and sign accuracy; library
control (100U vs 200U of K562, Panc1, HepG2, H1: AUC ≈ 0.5); bulk recovery as a guardrail; paired dashboards.

**Ablations** (the two main models retrained per variant; references stay at full inputs):

| Variant | Inputs (all with DNA unless noted) | Tells us |
|---|---|---|
| All | m6A, CpG, MSP, nuc, FIRE | reference |
| No FIRE pipeline | m6A, CpG, MSP, nuc (no FIRE calls) | what the FIRE calls add (= `model16`) |
| FIRE only | FIRE | what the FIRE calls alone carry |
| No fibertools | m6A, CpG (no MSP, nuc, FIRE) | whether the derived calls are needed or learnt from raw m6A |
| No CpG | m6A, MSP, nuc, FIRE | what CpG methylation adds |
| CpG only | CpG | what methylation alone carries |
| No DNA | all fiber channels, no DNA | what the DNA adds |
| Continuous scores (later) | ML / FIRE scores instead of ternary calls | whether finer inputs help |

Order: all, then the FIRE / fibertools variants, then CpG and DNA.

**Which model is better:** higher AUC and differential r on the unseen pairs, per assay and over the 6 test pairs,
in both designs; library control ≈ 0.5; bulk not clearly worse; dashboards agree. Differences of a few hundredths
need a second seed.

**Cost / build:** 11 reference runs + 2 per ablation variant (12 for the 6 variants), ~70 min each; eval on 6
pairs + 4 library controls per model, minis first. Build list: I15 in `TODO_extended.md`.

## 3. Right after

- **Rerun the bulk tables** with V1 DEV5 and the fold-change baseline.
- **Reference values for r:** ENCODE replicate 1 vs 2 per assay.
- **H3K36me3:** longer input window, central 5 kb scored (regions allow up to 20 kb); then the whole read.
- **m6A density per fiber:** check per library; drop only the failed-labelling tail or use it as an input (a hard
  filter would bias against compact heterochromatin molecules).
- **Later:** T5 (CpG on the same molecule, X inactivation), physical output units (V2/V3), Arabidopsis data.
