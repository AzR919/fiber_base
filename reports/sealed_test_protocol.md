# Sealed test protocol

**Status: FROZEN, 27 Nov 2026.** The decisions are in section 8.

Rules once frozen:
- the test runs once, and everything in it is reported, pass or fail;
- any later change is logged in section 9 with its date and reason, before any chr19 model output exists;
- a change made after seeing chr19 results is reported as such.

Background and dev results:
- `TODO.md` §1 (stage 4) and `TODO_extended.md` §2;
- dev results in `Nov_27.md` and `Dec_04.md`;
- fiber terms in `Nov_27.md` ("Fiber terms").

## 1. What the test is for

The paper's claim: on a sample that is a mixture of cell types, a model trained only on that sample's own bulk ChIP
imputes each molecule correctly. On the dev cells this held. The sealed test asks it once more, on cells and a
chromosome that never entered a decision. It tests five claims:

| | Claim | Shown on the dev pairs by |
|---|---|---|
| C1 | Trained on the mixture's bulk only, the model recovers each cell's own track and tells the two cells' molecules apart | `Nov_27.md` §1–3 |
| C2 | Training on your own sample beats a model pretrained on other cells (DEV4NF, no retraining) | `Nov_13_extended.md` §3 (T3b) |
| C3 | Mixture supervision costs little against training on each cell's own tracks (Pair) | `Nov_27.md` §3, `Dec_04.md` §2 |
| C4 | The network is needed: a small model on hand-crafted features, and fixed rules, do clearly worse | `Nov_27.md` §6 |
| C5 | Single-molecule biology holds: CTCF footprints, CpG methylation lowering active marks locally, poised regions split into open / closed molecules, X inactivation | `Nov_20.md`, `Nov_27.md` §5, `Dec_04.md` §1, §3 |

## 2. Data

- **Sealed chromosome: chr19** (user, 27 Nov).
  - Never used for training, validation or evaluation.
  - The only contact so far is the dev cells' bulk-only genome fractions in the poised test.
  - It is denser in genes and CpG islands than chr21, and holds the KRAB-ZNF clusters (H3K9me3). So expect
    different absolute numbers from the dev results; the comparisons between models are what carry over.
- **Sealed pair: HCT116 + MCF-7,** 200U libraries, at a 0.5 share with balanced stacks (the R1a recipe). Both have
  all 8 ENCODE assays.
- **What each source is used for:**

  | Source | Used for |
  |---|---|
  | HCT116 / MCF-7 fibers, chr20 | training (in mixtures; each cell alone for Pair-S) |
  | HCT116 / MCF-7 fibers, chr22 | validation (as in training) |
  | HCT116 / MCF-7 bulk tracks, chr20 / chr22 | the mixture target (R1a-S, MLP-S, Rules-S); own targets (Pair-S only) |
  | HCT116 / MCF-7 bulk tracks and CTCF peaks, chr19 | scoring only, and defining target sets (CTCF bound sites, poised classes, differential bins) |
  | ENCODE replicates, chr19 | the replicate ceiling (scoring only) |
  | HG008_3E4, GM25455 (LCL), HEK293T fibers, chrX + chr21 | the X-inactivation test only (section 4, "How the X-inactivation test works") |
  | HCT116 / MCF-7 fibers and tracks, chr21 | the end-of-training check only (not sealed; used before) |

- **H9 and Panc1 stay out.** Scope is two cells (user, 25 Nov), and Panc1's identity is in doubt.
- **The earlier use of the held-out cells on chr21** (ceiling fractions, ablation tables, the DEV4NF choice) is
  stated in the paper.

## 3. Models

Every model uses DEV4NF's channels (m6A, CpG, MSP, nucleosome; no FIRE), trains on chr20 and validates on chr22. The
run config is the last epoch, as on dev. Run prefixes in `slurm_batch_command.sh` (`CELL=s_*`).

| Model | Supervision | Config | Seeds | Role |
|---|---|---|---|---|
| **R1a-S** | balanced 0.5 mixtures of HCT116 + MCF-7 | `data38`, `model16`, `train04` | 3 | the model under test |
| DEV4NF | own tracks of the 4 dev cells; applied without retraining | existing | 2 (existing) | pretrained reference (C2) |
| Pair-S | own tracks of HCT116 and MCF-7 | `data39`, `model16`, `train04` | 3 | ceiling (C3) |
| MLP-S | the R1a-S mixtures; MLP on the 31 hand-crafted features | `data38`, `model22`, `train04` | 3 | simple model (C4) |
| Rules-S | the R1a-S mixtures; fixed rules, only the calibration trained | `data38`, `model24`, `train05` | 1 (as on dev) | simplest model (C4) |
| R1a-A, R1a-B | R1a on the dev pairs (GM12878 + HepG2, `data36`; K562 + H1, `data37`); applied without retraining | existing | 3 + 3 (existing) | a mixture model trained on another sample (secondary) |

Seeds are 919, 2, 3. The network + features model is not rerun: on dev it equalled R1a.

R1a-S is the R1a recipe trained on the sealed pair; R1a-A / R1a-B are the same recipe trained on the dev pairs. R1a-S
vs R1a-A / R1a-B differ only in the cells trained on, so they show whether a mixture model must be trained on the sample
itself (DEV4NF also differs in supervision: own tracks of four cells).

## 4. Primary endpoints (pass / fail)

Definitions are as on dev unless stated.
- **"Clear"** means |Δ| > 2 SE over seeds and every seed pairing has the same sign. Against Rules-S (one run):
  R1a-S's seed mean minus Rules-S > 2 SE of R1a-S's seeds, and every R1a-S seed above Rules-S.
- **"Passes"** for a per-site / per-gene test means the median effect has the expected sign with Wilcoxon p < 0.01
  over sites or genes.

| # | Claim | Endpoint | Pass |
|---|---|---|---|
| P1 | C1 | Own-track r of R1a-S, per cell (HCT116, MCF-7), against the floor "bulk on every molecule" (the mixture track's r with the cell's own track) | R1a-S clearly above the floor in both cells |
| P2 | C1 | Fiber-origin AUC of R1a-S at the pair's differential bins; random-split control (each cell's library split in two halves at random by read name, AUC at the same bins; the sealed cells have no second digestion level, so this replaces the library control) | AUC clearly > 0.5; random split within 0.5 ± 0.02 for both cells |
| P3 | C2 | R1a-S vs DEV4NF on fiber-origin AUC and on differential r | R1a-S clearly higher on both |
| P4 | C3 | R1a-S vs Pair-S on own-track r (both cells) and on mixture r | R1a-S not clearly below Pair-S on any of the three |
| P5 | C4 | R1a-S vs MLP-S and vs Rules-S on mixture r | R1a-S clearly higher than both |
| P6 | C5 | CTCF footprint effect (H1, footprint − open pred CTCF, bound sites) on HCT116 and MCF-7 fibers, R1a-S | passes on the seed mean, and in at least 2 of 3 seeds, in both cells |
| P7 | C5 | X inactivation of the active marks (ATAC, H3K4me3, H3K27ac) at silenced CGI promoters, block-phased, on HG008_3E4, GM25455, HEK293T, model R1a-S (seed mean) | each mark passes in at least 2 of the 3 cells |

**Expected** (from dev): P1–P5 and P7 pass. P6 passes on the seed mean but may fail in single seeds (2 of 24 dev runs
lost the footprint effect). If P3 fails, the "train on your own sample" claim falls back to "as good as a pretrained
model".

### How the X-inactivation test works

The XI cells are three female lines with a clear inactive X: HG008_3E4, GM25455 (LCL) and HEK293T. No model has seen
them, and they need no ENCODE tracks: the test compares each model's predictions on the two X copies of the same cell.
The models trained on HCT116 + MCF-7 are applied to these cells' fibers as they are (the cells are outside the training
pair, like DEV4NF on any new cell).

1. **Which fibers come from which copy.** HiPhase tags each fiber with a haplotype (HP1 / HP2) inside a phase block.
   The haplotype labels are consistent only within a block, so which of HP1 / HP2 is the inactive X is decided per
   block.
2. **Which copy is the inactive X, per block.** On the inactive X, CGI promoters of silenced genes are methylated
   and the XIST promoter is unmethylated. For a promoter under test, the inactive copy is called from the *other*
   informative promoters of the same block (the tested gene is left out, so its own methylation never sets its call):
   the mean of (methylation HP2 − methylation HP1) over the subject CGI promoters, with XIST's sign flipped. Mean
   > +0.2: HP2 is the inactive X; < −0.2: HP1; otherwise no call and the gene is not tested. A promoter needs ≥ 5
   fibers with ≥ 3 CpGs per haplotype to count. Validated on GM12878 with these cells' block layouts: the call was
   right ≥ 99.5% of the time (`Dec_04.md` §3).
3. **The readout.** For each called silenced CGI promoter (TSS ± 500 bp), the model's mean prediction per mark on the
   active-copy fibers minus the inactive-copy fibers, over the scored positions. Active marks (ATAC, H3K4me3, H3K27ac)
   should be higher on the active copy: the median over genes > 0 with Wilcoxon p < 0.01 (P7).
4. **Controls (secondary).**
   - **chr21 null:** autosomal promoters with a random orientation per block should show no difference.
   - **Escape genes and PAR**, read without an inactive-X call: |HP1 − HP2| should be smaller than at silenced genes.
5. **What it shows and what it doesn't.** The model never sees the haplotype tag. Whatever separates the copies comes
   from the fibers themselves: accessibility, nucleosomes and CpG methylation. The copy call also uses CpG methylation,
   though of the other promoters. So the test asks whether the model reads the inactive copy's molecules as silent;
   the CpG channel is one way it can, and the dev CpG-flip results (`Nov_27.md` §5) say it uses it. H3K27me3 and
   H3K9me3 are reported as secondary readouts; H3K27me3 is expected to come out wrong (section 6).

## 5. Secondary readouts (reported, no pass / fail)

The bulk and T3 readouts are given for each of R1a-S, Pair-S, MLP-S, Rules-S, DEV4NF and R1a-A / R1a-B (the
molecule readouts below too):
- **R1a-S vs R1a-A / R1a-B:** the same comparisons as P3 (fiber-origin AUC, differential r), plus own-track and
  mixture r; reported, no pass / fail.
- **Bulk:** own-track r and fraction of the replicate ceiling per cell and assay; MSE; mixture r per assay.
- **T3:**
  - sign accuracy and differential r per assay;
  - the gap to Pair-S, reported as the cost of mixture supervision, not as a criterion (D1).
- **CTCF footprints** (every model):
  - H1 per seed;
  - H2 (CTCF effect beyond the ATAC effect);
  - H3 (open > nucleosome);
  - H5 (bound > unbound at matched motif score).
- **CTCF × CpG (part C only, in-silico flips at bound sites):**
  - methylating the motif ± 100 bp lowers pred CTCF, ATAC and H3K4me3 (expected < 0);
  - unmethylating raises them;
  - flips 1.5 kb away do ~nothing.
  - Parts A and B (associations) are reported as data.
- **Poised vs Polycomb** (tiles classed from HCT116 / MCF-7 bulk on chr19):
  - D1, D2;
  - M1 with the bulk reference;
  - M2 (open − closed H3K4me3 at poised tiles > 0; methylated − unmethylated H3K27me3 at Polycomb tiles < 0);
  - M3;
  - C.
  - On dev the differentiated lines had 17–34 poised tiles on chr21, so the poised readouts may rest on few tiles.
    Polycomb readouts should have hundreds.
- **X inactivation, rest:**
  - H3K9me3 and H3K27me3 at silenced CGI promoters;
  - the escape control;
  - the chr21 null;
  - coverage (genes called per cell);
  - the same for every other model.
- **Shuffle control (T7, D5):**
  - each chr19 mini tile's fibers are replaced by the same cell's fibers from another tile of chr19 (a seeded
    derangement, the same partner for both cells); target and reference DNA stay the tile's own;
  - readouts: fiber-origin AUC, differential r and own-track r, against the unshuffled run on the same tiles;
  - expected: AUC and differential r fall to near their floors, own-track r falls less (the DNA still carries the
    locus);
  - run on R1a-S (3 seeds).

## 6. Expected failures, written now

These follow from the dev results. They are not a reason to change anything:
- **H3K27me3 on the inactive X:** predicted higher on the active copy (every dev model).
- **H3K27me3 where DNA methylation and Polycomb coincide:** low (H1 on dev). HCT116's genome is heavily methylated
  (its chrX and chr21 promoters were hypermethylated in the 25 Nov check), so the poised / Polycomb H3K27me3 readouts
  in HCT116 may fail.
- **H3K9me3** is the hardest assay on dev (r ≤ 0.35 outside HepG2). chr19's ZNF clusters make it larger and maybe
  easier.
- **Seed noise:** single seeds may lose the CTCF footprint effect.

## 7. Procedure

1. **Freeze this protocol** (done 27 Nov).
2. **Train** R1a-S, Pair-S, MLP-S (3 seeds each) and Rules-S (1). Training touches chr20 / chr22 only, so it runs
   while the chr19 inputs are built. Watch training and validation loss, and the end-of-training check on chr21
   (`eval39`, the sealed pair's mixture on chr21 mini tiles): it only confirms that training ran, and decides nothing.
3. **Build everything without any model on chr19:**
   - **Windows:** `test_chr19_tiles`, `test_chr19_mini2000` (`data_prep/regions_spec.yaml`).
   - **Differential bins:** `loci/pairs/HCT116__MCF-7` on chr19 (`data_prep/loci_spec.yaml`).
   - **Eval configs:** `eval33` (pair, all chr19 tiles), `eval34` (mixture, all tiles), `eval35` / `eval36` (random
     split, mini), `eval37` (pair, mini), `eval38` (shuffle, mini).
   - **Molecule targets** (`data_prep/targets_spec_sealed.yaml`): chr19 CTCF sites with HCT116 / MCF-7 peaks; poised
     tiles from HCT116 / MCF-7 tracks; XI promoters are the existing chrX + chr21 set.
   - **Annotation tracks** on chr19 (Rules-S).
   - **Ceiling:** chr19 replicate tracks of HCT116 and MCF-7.
   - **Downloads (each asked for first):** the two CTCF peak files; the replicate tracks (cropped to chr19 + chr21).
   - **Checks allowed now:** counts and sanity only (number of sites, tiles, bins, fibers); no readout.
4. **Dry run of the whole pipeline on a dev pair on chr21** with the dev models: the random-split and shuffle configs
   (`eval40`, `eval41`), and the analysis scripts with the sealed paths swapped for dev ones, so that the sealed run
   needs no code changes.
5. **Score once:**
   - evals on chr19 (every model; R1a-S's pair eval with the mixture reference for P1);
   - molecule runs on HCT116 / MCF-7 fibers on chr19 (CTCF sites with flips, poised tiles), every model;
   - molecule runs on the XI cells (chrX + chr21), every model.
6. **Analyze with the scripts as frozen;** report every endpoint in a dated report. Bugs found afterwards are fixed
   and the fix and its effect reported.

## 8. Decisions (user, 27 Nov)

- **D1. Sign accuracy and differential r against Pair:** reported as the cost of mixture supervision (secondary), not
  a pass / fail criterion. The primary comparison with Pair is bulk (P4).
- **D2. CTCF footprint test:** passes on the seed mean plus at least 2 of 3 seeds individually (P6). Pair also lost
  the effect in one dev seed, so a per-seed rule would fail for any supervision.
- **D3. Pair-S:** trained (3 seeds).
- **D4. MLP-S:** trained (3 seeds). **Rules-S** added too (1 run, as on dev), so the whole ladder is on the sealed
  pair; P5 covers both.
- **D5. Shuffle control:** in, as a secondary readout (section 5). Nucleosome positioning at CTCF and the CTCF
  specificity test are not in the sealed test (deferred; nucleosome positioning to be discussed with the professor).

## 9. Change log

- 27 Nov: draft.
- 27 Nov: frozen with D1–D5 as above; Rules-S added; XI test described; shuffle control scoped to the eval readouts
  (AUC, differential r, own-track r); the end-of-training check on chr21 allowed.
- 27 Nov (user, before any chr19 output): the dev R1a models (R1a-A, R1a-B, 3 seeds each) added as a secondary
  reference, applied without retraining. No primary endpoint changed.
