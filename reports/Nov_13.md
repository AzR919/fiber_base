# 13th Nov

Short version. Every experiment with its setup, full tables, figures and sources is in `Nov_13_extended.md`
(section numbers below refer to it).

## In one paragraph

The main model is now **DEV4** (4 dev cell types; Panc1 moved to held out). It beats the cross-cell average on MSE
but on Pearson only in dev cell types, and it is weak on the broad marks. Training on mixtures alone works as
well as training on each cell type's own track. The first biological tests on single molecules pass for the
most part, on chromosomes the model never trained on:
- per-fiber predicted CTCF follows the CTCF footprint on the molecule;
- per-haplotype predictions follow X inactivation.

Dropping the FIRE input costs nothing in bulk (slightly better on H3K27ac / H3K4me1), and the no-FIRE model is the
only one whose CTCF footprint response is strong and CTCF-specific in both of two trainings, so it should become
the main model. A second training seed of every model confirms all bulk results; the molecule-level responses
vary 2–10× between trainings, so the earlier "10× stronger without FIRE" was mostly a weak DEV4 run. ENCODE
replicates agree at only r ≈ 0.4, so the best reachable r is ~0.74;
DEV4 gets two thirds of it in dev cell types, and the remaining gap is mostly cell-specific CTCF, H3K36me3 and
H3K27me3 in held-out cell types.

## Results

**1–2. Reference model and split v2** (sections 1–2)
- The all-cells model (DEV5) beats the cross-cell average on MSE in 7 of 8 cell types. On Pearson it is ahead only
  in dev cell types.
- Strong on ATAC, H3K4me3, H3K27ac, H3K4me1 (including held-out cell types); weak on H3K36me3, H3K27me3, H3K9me3,
  CTCF.
- Panc1's sample is doubtful and training on it buys nothing, so it moved to held out, and the 4-dev **DEV4** is
  the main model.
- Single-cell-type specialists beat the multi-cell model by +0.03–0.06 r.

**3. Mixture test on two pairs** (section 3)
- A model trained only on mixtures of two cell types separates their molecules as well as one trained on their
  own tracks, on both pairs (fiber-origin AUC 0.637 vs 0.635, and 0.666 vs 0.665).
- On a pair it never saw, the type of supervision makes no difference; having seen the cell types is what
  matters. Transfer toward K562–H1 is poor.

![Mixture test summary](../plots/26_nov_13/mix/mix_summary.png)

**4–5. Why some assays fail on the pairs** (sections 4–5). Three failure types:
1. **Track-scale outliers fibers cannot show:** H1's CTCF ChIP is weakly enriched everywhere; HepG2's and H1's
   H3K9me3 tracks are globally strong. Every model fails these, trained on the pair or not. The models also
   predict more CTCF for H1 at equal accessibility; the no-CpG model (6) keeps this shift, so it isn't from CpG.
2. **Compression** on cell types the model never saw.
3. **Cell-level offsets:** a line's track sits above what its fibers' accessibility implies (H3K27ac, H3K36me3).

**6. Input ablations** (section 6). Four DEV4 retrains: no CpG, no FIRE, no accessibility (CpG + DNA only), and
m6A + CpG only (no fibertools calls: no MSP, nucleosome or FIRE).

![Bulk r per assay for DEV4 and the ablations](../plots/26_nov_13/ablations/ablation_bulk.png)

| Assay | DEV4 r (dev) | Δ no FIRE | Δ m6A + CpG | Δ no accessibility | Δ no CpG |
|---|---|---|---|---|---|
| ATAC | 0.76 | 0.00 | 0.00 | **−0.14** | −0.01 |
| H3K4me3 | 0.53 | +0.01 | 0.00 | −0.04 | −0.05 |
| H3K27ac | 0.49 | +0.02 | +0.02 | −0.09 | −0.04 |
| H3K4me1 | 0.54 | +0.03 | +0.03 | **−0.13** | −0.03 |
| CTCF | 0.39 | +0.01 | 0.00 | −0.08 | −0.01 |
| H3K36me3 | 0.48 | +0.01 | +0.01 | −0.01 | **−0.25** |
| H3K27me3 | 0.44 | +0.01 | +0.01 | −0.03 | **−0.09** |
| H3K9me3 | 0.39 | 0.00 | 0.00 | −0.09 | −0.01 |

- **No FIRE costs nothing** (slightly better almost everywhere). This confirms the hypothesis.
- **Dropping all fibertools calls costs nothing either** (m6A + CpG: +0.009 dev, +0.012 held out, same as no FIRE).
  In bulk, the model gets everything it needs from raw m6A and CpG.
- **All of this replicates in a second seed** (section 10). The one real gain: no FIRE and m6A + CpG are +0.02–0.03
  on H3K27ac and H3K4me1 in both seeds.
- **CpG carries H3K36me3** (−0.25; −0.46 in K562), consistent with gene-body methylation, and H3K27me3. Without
  accessibility, H3K36me3 stays intact. I had predicted CTCF and H3K9me3 would lose too; they don't.
- **Accessibility carries** ATAC, H3K4me1, H3K27ac and CTCF.

**7. CTCF footprints on single molecules** (section 7)
- At bound CTCF sites on chr21, fibers whose motif is protected from m6A (a footprint) were compared with open
  fibers at the same site.
- In the data, 41–43% of accessible fibers carry the footprint at bound sites vs 12–13% at unbound ones.

Footprint − open, predicted CTCF at bound sites (GM12878 / K562), in two trainings of each model:

| Model | Seed 919 | Seed 2 | CTCF − ATAC, seed 919 | CTCF − ATAC, seed 2 | Unbound sites |
|---|---|---|---|---|---|
| DEV4 | +0.11 / +0.04 | +0.73 / +0.44 | +0.02 / +0.05 | +0.23 / +0.21 | ≈ 0 |
| **No FIRE** | **+1.14 / +0.84** | **+0.95 / +0.68** | **+0.67 / +0.54** | **+0.85 / +0.63** | ≈ 0 |
| m6A + CpG (no fibertools calls) | +0.26 / +0.20 | +0.56 / +0.58 | +0.20 / +0.14 | +0.55 / +0.50 | ≈ 0 |
| No accessibility (null: sees no m6A) | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | 0 |
| No CpG | +0.18 / +0.10 | +0.75 / +0.46 | −0.12 / −0.02 | +0.33 / +0.31 | ≈ 0 |

![Single-molecule readouts, both seeds](../plots/26_nov_13/seeds/seeds_molecules.png)

- **In both seeds, every model that sees m6A responds**, and nothing happens at unbound motifs of the same score or
  in the no-accessibility null.
- **The size of the response varies 2–10× between trainings** (DEV4 +0.11 → +0.73). In seed 919 DEV4 was weak,
  which made no FIRE look ~10× stronger; in seed 2 the ratio is 1.3–1.5×.
- **No FIRE is the only model that is strong and CTCF-specific in both trainings** (CTCF − ATAC 3–4× DEV4's in
  seed 2). FIRE's classifier was trained on CTCF ChIP, so with FIRE in the input the model may take "CTCF site" from
  that channel instead of the molecule; seed 2 shows DEV4 can still learn the read, less reliably.
- **m6A + CpG** reaches 60–85% of no FIRE in seed 2 (a quarter in seed 919), so whether the MSP / nucleosome calls
  help the molecule read is open.

**8. X inactivation per haplotype in GM12878** (section 8)
- The inactive X (Xi) is identified from measured XIST promoter methylation (0.09 vs 0.36), so Xi = HP2.
- For each silenced chrX gene, active X minus inactive X was compared, at chrX promoters the model never trained on.

| Expected on the active X | DEV4: fraction of genes right (all / CpG-island promoters) |
|---|---|
| ATAC higher | 86% / 91% |
| H3K27ac higher | 82% / 84% |
| H3K4me3 higher | 73% / 84% |
| H3K9me3 lower | 73% / 76% |
| H3K27me3 lower | **43% / 26%: fails** |

![X inactivation per gene, DEV4](../plots/26_nov_13/molecules/xi_promoters_DEV4.png)

- **Nulls are flat:** escape and PAR genes and chr21 genes show no difference. **XIST is reversed** (higher on the Xi).
- **H3K27me3 has the wrong sign.** The model probably applies the autosomal rule that Polycomb sits at unmethylated
  CpG islands; on the Xi, methylation and H3K27me3 occur together.
- **Ablations:** the active marks pass from CpG alone or from accessibility alone. H3K9me3 needs accessibility;
  from CpG alone it reverses, and raw m6A without the fibertools calls brings it back partly (68% of genes).
- **Second seed:** the active marks, nulls, XIST and the H3K27me3 failure replicate in every model. H3K9me3 does
  not, except without CpG (79–86% in both seeds): DEV4's drops to chance at CGI promoters (52%) and CpG alone gets
  it 67% right instead of reversed. So the H3K9me3 results above are single-seed.

**9. ENCODE replicate ceiling** (section 9)
- Two replicates of the same ENCODE experiment were scored against each other at our eval positions on chr21. That
  gives the best r any prediction could reach against the pooled target (the ceiling).
- **Histone and CTCF replicates agree at only r 0.26–0.50**, so the ceiling is ~0.65–0.80 (ATAC 0.93). It is the
  same in dev, held-out and Panc1 targets.

| Fraction of the ceiling (DEV4) | Dev | Held out | Panc1 |
|---|---|---|---|
| Pearson r | 66% | 58% | 46% |
| Cell-type r | 53% | 40% | 23% |

![Fraction of the ceiling, five models](../plots/26_nov_13/ceiling/ceiling_models.png)

- **In dev cell types every assay is at 55–79% of its ceiling.** H3K36me3 and H3K27me3 (67%) are as close as the
  active marks: their low raw r was mostly target noise. H3K9me3 is furthest (55%).
- **ATAC and the active marks transfer to held-out cell types unchanged.** The held-out drop is the cell-specific
  part of CTCF, H3K36me3 (8% of its cell-type ceiling) and H3K27me3. For these, the cross-cell average is closer to
  the ceiling than any model.
- **The ablation results hold on this scale:** no FIRE and m6A + CpG ≥ DEV4; no accessibility −10 points; no CpG
  halves dev H3K36me3. No model fixes held-out H3K36me3.

**10. Second training seed** (section 10)
- All five models were retrained with seed 2 and run through the same evals and molecule tests.
- **Bulk is robust:** per cell type × assay the seeds differ by a median of 0.01 r. A reported group mean moves by
  ~0.005–0.01 (up to 0.03–0.04 for held-out H3K27me3 / H3K9me3), and every ablation result and ceiling fraction
  holds.
- **Two single-seed claims fell:** no FIRE's H3K27me3 gain (held-out cell-type r +0.06 → +0.00) and its dev
  H3K27me3 MSE cost (+0.04 → −0.01).
- **Molecule readouts vary much more** (2–10× in size; H3K9me3 on the Xi flips). Their directions and nulls hold.

![Ablation effects in both seeds](../plots/26_nov_13/seeds/seeds_effects_r.png)

## Scorecard

| Test | Hypothesis | Result |
|---|---|---|
| Mixture supervision (T3, T3b) | Mixtures reach the own-track ceiling | ✅ both pairs |
| No FIRE | Costs little | ✅ costs nothing (both seeds; +0.02–0.03 on H3K27ac / H3K4me1) |
| m6A + CpG only (no fibertools calls) | Costs little | ✅ costs nothing in bulk (both seeds); CTCF footprint response 1.2–4.4× weaker than no FIRE |
| No CpG | CpG-linked marks lose most | ◐ H3K36me3 / H3K4me3 yes; CTCF / H3K9me3 no; H3K27me3 unexpectedly (both seeds) |
| CTCF footprints | Footprint fibers get more CTCF than open fibers, more than ATAC | ✅ every model with m6A in seed 2, size varies 2–10× between trainings; strong and specific in both seeds only without FIRE; null clean |
| X inactivation | Xa > Xi for active marks, Xa < Xi for repressive; nulls ≈ 0; XIST reversed | ✅ ATAC, H3K4me3, H3K27ac, nulls, XIST (both seeds); ◐ H3K9me3 (robust only without CpG); ❌ H3K27me3 |
| Second seed | Results survive retraining | ✅ bulk; ◐ molecules (directions yes, sizes no) |
| Replicate ceiling | Weak broad marks are target noise; held-out gap is not | ◐ H3K36me3 / H3K27me3 in dev yes, H3K9me3 no; ✅ held-out targets as reproducible as dev |

## Next

- **Switch to the no-FIRE model.** The second seed confirms it is level or slightly better in bulk and the only
  model with a reliably strong, CTCF-specific footprint response. Keeping the MSP / nucleosome inputs costs
  nothing in bulk; whether they help per molecule is open.
- **Report molecule tests over at least two seeds** (or an ensemble): they vary 2–10× between trainings, bulk
  only ~0.01. Analyse the seed-2 pair evals (done, not yet looked at).
- **Where to improve (ceiling):** the cell-specific CTCF, H3K36me3 and H3K27me3 in held-out cell types, and
  H3K9me3 everywhere. Report future models as a fraction of the ceiling.
- **X inactivation:** restrict to genes expressed in GM12878, check the escape list against Tukiainen 2017, and test
  H3K27me3 over broader windows.
- **Remaining single-molecule tests:** in-silico CpG flips, measured methylation as held-out data for the no-CpG
  model, imprinting, allele-specific ChIP, and the search for external single-molecule histone data.
