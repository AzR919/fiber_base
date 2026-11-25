# 27th Nov

The mixture-trained model, from the choice of mixture recipe to its seed noise. Section 1 is the recipe (R1a),
section 2 its first run on one pair, section 3 three training seeds on both dev pairs, section 4 the two decisions
on the success criteria that the seed results raise. Plan and open items: `TODO.md` §1, `TODO_extended.md` §2.

## Models

All models use DEV4NF's architecture and channels (`model16_uct_mid_8A_no_fire_msp`, no FIRE), training budget
(`train04`, 15 × 5000 samples) and region set, so the **supervision** is the only difference.

| Model | Trained on | Data config (pair A / pair B) | Seeds |
|---|---|---|---|
| **DEV4NF** | own tracks of the 4 dev cell types (GM12878, K562, HepG2, H1) | `data24` | 919, 2 |
| R0 | mixtures of the pair, share redrawn per window in [0.2, 0.8] | `data27` / – | 919 (pair A) |
| **R1a** | mixtures of the pair, one fixed share (0.5), stack balanced to it | `data36` / `data37` | 919, 2, 3 |
| Pair | own tracks of the two cells of the pair | `data28` / `data34` | 919, 2, 3 |

Pair A = GM12878 + HepG2, pair B = K562 + H1. Every model is scored on chr21 (test chromosome), with the eval
configs' own seed, so every training seed sees the same fibers. H1 has no ATAC track, so pair B readouts are over the
7 other assays, and R1a on pair B was trained without an ATAC target.

## 1. The mixture recipe: R1a

> **Question:** the claim is that a model trained on the bulk of the mixture you actually have (one sample, one
> composition) still imputes each molecule well. The first mixture models (R0) redrew the share for every window,
> which no real sample does. Does training on one fixed composition lose anything?

**R1a** (`mix_range: [0.5, 0.5]`, `mix_balance: true`, `mix_min_fibers: 10`):
- **Share:** fixed at 0.5 for the whole sample, as in one real mixture.
- **Fiber stack:** balanced to exactly that share, N = min(100, min over cells of available / share). Windows with
  under 10 fibers are redrawn in training and skipped in evaluation (0% of windows in practice).
- **Target:** unchanged, the per-position share-weighted mix of the two cells' tracks.

Left out on purpose: R1b (population target, multinomial fiber counts, copy-number weights; a robustness check if a
reviewer asks) and R2 (several samples with different fixed shares; dropped once R1a passed). Two gaps no variant
addresses go to the paper's limitations: a synthetic mixture combines two Fiber-seq libraries and two separate ChIP
experiments, where a real sample has one of each.

## 2. First run: GM12878 + HepG2, one seed (23 Nov)

Scripts `plots_code/26_nov_27/realism_t3.py` and `realism_bio.py`; outputs in `plots/26_nov_27/realism/`.

| | Fiber-origin AUC | Sign accuracy | Differential r | Bulk r GM12878 / HepG2 (fraction of ceiling) | Mixture r, share redrawn / 0.5 balanced | CTCF H1 GM12878 / HepG2 |
|---|---|---|---|---|---|---|
| DEV4NF | 0.637 | 0.743 | 0.328 | 0.471 (0.64) / 0.596 (0.71) | 0.593 / 0.604 | 1.14 / 0.97 |
| R0 | 0.654 | 0.779 | 0.330 | 0.468 (0.63) / 0.616 (0.74) | 0.610 / 0.623 | 0.99 / 0.81 |
| **R1a** | 0.653 | 0.770 | 0.352 | 0.482 (0.65) / 0.629 (0.75) | 0.622 / 0.633 | 0.81 / 0.58 |
| Pair | 0.646 | 0.795 | 0.368 | 0.482 (0.65) / 0.620 (0.74) | 0.615 / 0.625 | 1.12 / 0.86 |

![T3 readouts, first run](../plots/26_nov_27/realism/t3_summary.png)

![Bulk accuracy per assay, first run](../plots/26_nov_27/realism/t3_bulk.png)

![CTCF footprints, first run](../plots/26_nov_27/realism/bio_ctcf.png)

![X inactivation, first run](../plots/26_nov_27/realism/bio_xi.png)

- R1a matched R0 on T3 and was equal or better on bulk, level with the pair model. The library control stayed at
  chance for every model.
- The biology tests kept their direction in all four models: CTCF H1–H3 and H5 pass; X inactivation's active marks
  were at 0.87–0.92, with flat nulls and XIST reversed.
- **Decision (23 Nov): R1a is the mixture recipe.**
- Found on the way: two mixed eval configs in one run were both saved as `mixed`, so the second overwrote the
  first. They are now keyed `<config>/mixed` (`evaluator.py`).

## 3. Seed noise: both dev pairs, 3 seeds (24 Nov)

> **Question:** which of the first run's differences survive training noise, and do they hold on a second pair?

Script `plots_code/26_nov_27/realism_seeds.py` (job `realism_seeds.sh`); outputs in `plots/26_nov_27/seeds/`
(`seeds_table.tsv` has every readout as mean, sd and the per-seed values; `seeds_deltas.tsv` the model differences).
A difference counts as **clear** when |Δ| > 2 SE of the difference of seed means and every pairing of seeds has the
same sign. Values are mean ± sd over seeds.

### T3 and bulk

| Pair | Model | AUC | Sign accuracy | Differential r | Own-track r, cell 1 / cell 2 | Mixture r (0.5 balanced) | Library \|AUC − 0.5\| |
|---|---|---|---|---|---|---|---|
| A | DEV4NF | 0.631 ± .008 | 0.739 ± .006 | 0.333 ± .007 | 0.462 / 0.598 | 0.600 ± .006 | 0.052 |
| | R0 (1 seed) | 0.654 | 0.779 | 0.330 | 0.468 / 0.616 | 0.623 | 0.051 |
| | **R1a** | 0.642 ± .010 | 0.754 ± .015 | 0.338 ± .014 | 0.474 / 0.624 | 0.628 ± .009 | 0.051 |
| | Pair | 0.648 ± .004 | 0.797 ± .011 | 0.367 ± .001 | 0.481 / 0.625 | 0.628 ± .003 | 0.050 |
| B | DEV4NF | 0.653 ± .010 | 0.785 ± .006 | 0.290 ± .012 | 0.483 / 0.419 | 0.512 ± .004 | 0.060 |
| | **R1a** | 0.664 ± .004 | 0.789 ± .023 | 0.272 ± .004 | 0.480 / 0.431 | 0.526 ± .010 | 0.060 |
| | Pair | 0.676 ± .006 | 0.843 ± .010 | 0.328 ± .002 | 0.486 / 0.440 | 0.525 ± .008 | 0.064 |

Cells: pair A GM12878 / HepG2, pair B K562 / H1. As fractions of the ENCODE replicate ceiling, R1a is at 0.64 / 0.75
(pair A) and 0.65 / 0.67 (pair B), within 0.015 of Pair everywhere.

![T3 readouts per seed](../plots/26_nov_27/seeds/seeds_t3.png)

![Bulk accuracy per seed](../plots/26_nov_27/seeds/seeds_bulk.png)

- **Bulk: mixture supervision costs nothing.** R1a = Pair within noise on every own-track r and on mixture r, on
  both pairs. R1a is clearly above DEV4NF on HepG2, H1 and the pair A mixture.
- **Single-fiber differential signals: it does.** Pair is clearly above R1a on sign accuracy (+0.042 / +0.055) and
  differential r (+0.029 / +0.056) on both pairs, and on AUC for pair B (+0.012). On these readouts R1a sits at
  DEV4NF's level, or below it (differential r on pair B, −0.018).
- The first run's "R1a ≈ Pair on T3" came partly from seed 919 being R1a's best seed on pair A (AUC 0.653 against a
  mean of 0.642).
- The library control is at chance for every model and seed.

### Biology tests

![CTCF and X inactivation per seed](../plots/26_nov_27/seeds/seeds_bio.png)

- **CTCF:** the footprint effect (H1, footprint − open pred CTCF at bound sites) changes a lot between trainings of
  the same model. One Pair seed on pair A has none at all (0.01 on GM12878, −0.11 on HepG2), and R1a seed 919 on
  pair B is at 0.02 on K562 (p 0.06). Every other run passes with p < 1e-7. The first run's ranking
  (DEV4NF > Pair > R1a) is within this noise. H3 (open > nucleosome) passes in every run. H5 (bound > unbound at
  matched motif score) is 1.0 everywhere, so it no longer tells models apart.
- **X inactivation** (GM12878 fibers, for every model, so GM12878 is an unseen cell for the pair B models): on
  silenced CGI genes, 0.69–0.79 of genes have the expected direction (mean of 5 marks), and both nulls stay at
  ~0.5. No model difference is clear. ATAC, H3K4me3 and H3K27ac are at 0.85–0.91. H3K27me3 is near or below 0.5 for
  every model, so it is unusable as an X-inactivation mark.
- **Caveat:** R1a on pair B has an untrained ATAC head (no ATAC target), so its XI ATAC (0.43) and its CTCF H2 (which
  subtracts the pred ATAC effect) mean nothing and are left out of the reading.

## 4. Two decisions on the success criteria

Draft criteria (`TODO_extended.md` §2f), written before the seed runs:
1. on the dev pairs, the mixture model is within 2 seed SDs of the pair model on fiber-origin AUC and own-track
   recovery;
2. it is clearly above the floors (library control, bulk on every molecule);
3. the CTCF footprint and X-inactivation active-mark tests pass with mixture-only supervision, in all 3 seeds;
4. on the sealed test, the mixture model beats DEV4NF applied without retraining on AUC and differential r.

Script `plots_code/26_nov_27/criteria_plots.py`; outputs in `plots/26_nov_27/criteria/`.

### Decision 1: do sign accuracy and differential r count as criteria?

![Gap to the pair model per readout](../plots/26_nov_27/criteria/criteria_gaps.png)

- Above the dotted line (criterion 1 as written, plus mixture r), R1a is inside or at the edge of the tolerance
  band, except GM12878 own-track r (next point). Below it, R1a is clearly below Pair on both readouts and both pairs, by more than the band.
- The band (2 SDs of Pair's 3 seeds) is unstable: Pair's GM12878 own-track r has sd 0.001, so the band is ±0.003 and
  R1a's −0.007 falls outside it, although the difference itself is not clear (2 SE = 0.010). A tolerance based on
  the SE of the difference, or on the seed SD pooled over models, would be steadier.

![Sign accuracy and differential r per assay](../plots/26_nov_27/criteria/criteria_per_assay.png)

- **Where the gap is:** on pair A the sign-accuracy gap is mostly two marks, H3K9me3 (R1a 0.33 vs Pair 0.58) and
  H3K36me3 (0.49 vs 0.55). There DEV4NF and R1a are at or below chance, and Pair is barely above it. On pair B the
  gap is spread over most marks (−0.05 to −0.10 on H3K27ac, H3K27me3, H3K36me3, H3K9me3). On differential r it is
  largest for CTCF on pair B (0.13 vs 0.30; DEV4NF 0.21).
- **Options:**
  - (a) include them: R1a fails, and the claim becomes "mixtures lose part of the single-fiber difference";
  - (b) report them as the cost of mixture supervision, with the criteria on AUC, own-track and mixture recovery,
    and the biology tests.
- **Recommendation: (b), with the tolerance redefined on the SE of the difference.** The claim to sell is "train on
  your own sample". Its comparator in practice is DEV4NF (no own-track data for your sample), and R1a matches or beats
  DEV4NF on every T3 and bulk readout except differential r on pair B. (Its CTCF footprint effect on HepG2 is
  clearly smaller than DEV4NF's, 0.62 vs 0.90, though still clearly positive in every seed.) The pair model is an upper bound that needs data a real
  sample doesn't have. Reporting the gap to it is honest and doesn't hurt the claim.

### Decision 2: the CTCF footprint test per seed, or on the seed mean?

![CTCF footprint effect per seed](../plots/26_nov_27/criteria/criteria_ctcf.png)

| Rule | DEV4NF (8 cases) | R1a (12) | Pair (12) |
|---|---|---|---|
| passes in every seed (criterion 3 as written) | yes | no (K562, seed 919) | no (GM12878 and HepG2, seed 3) |
| passes in at least 2 of 3 seeds, in every cell | yes | yes | yes |
| seed mean − 2 SE > 0, in every cell | yes | yes (K562 only just: 0.015) | no (HepG2: −0.11) |

- The per-seed failures are not specific to mixtures: own-track supervision (Pair) loses the footprint effect in a
  whole training, on both cells. So "every seed" tests training stability, which no criterion on supervision should
  depend on.
- **Recommendation:** pass if the test holds in at least 2 of 3 seeds in every cell, and report every seed's effect and
  p (as in the figure). The seed-mean rule is weaker with 3 seeds, since one bad seed inflates the SE.

## Infrastructure (23–24 Nov)

- **Two clusters:** work moves between nibi and fir, whichever is up. `misc/cluster.sh` does every transfer
  explicitly through the local repo; the slurm scripts and venv are the same on both (fir: CPU / memory overrides on
  the command line). Data is the same on both since 24 Nov. nibi went down on 24 Nov; the seed runs ran on fir.
- **R1a on fir** loads slowly on 3 CPUs (~7 samples/s on pair A, 13–14 min per epoch), so 15 epochs need about
  3.5 h. Four R1a seed runs were finished with chained `RESUME=latest` jobs.

## Next steps

- Settle the two decisions above and finalize the success criteria (`TODO_extended.md` §2f).
- Stage 4, lock the recipe on the dev cells: more dev pairs (the spread across pairs), one 3-cell mixture (needs
  cells with every assay, or a per-assay mask, since H1 has no ATAC), the biology tests on those models.
- Then the sealed test, once: R1a on held-out mixtures (HCT116 + MCF-7, a 3-cell one), 3 seeds, against DEV4NF
  without retraining.
