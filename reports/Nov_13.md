# 13th Nov

Short version. Every experiment with its setup, full tables, figures and sources is in `Nov_13_extended.md`
(section numbers below refer to it; its sections 7 and 8, the molecule tests, are now in `Nov_20.md`).

## Summary

We retrained the main model (DEV4) four times, each time with some fiber input channels switched off, and trained
all five models twice (seeds 919 and 2). **In bulk, the fibertools calls (FIRE, MSP, nucleosome) add nothing:**
the model gets what it needs from raw m6A and CpG, and without FIRE it is even slightly better on H3K27ac and
H3K4me1. **m6A carries the accessibility-type assays; CpG carries H3K36me3 and H3K27me3.** All of this holds in
both seeds. **On single molecules, every model that sees m6A reads the CTCF footprint**, but the size of that
response varies 2–10× between trainings. The model without FIRE is the only one that is strong and CTCF-specific
in both trainings, so it should become the main model.

## Context (sections 1–5, 9)

- **Main model:** DEV4, trained on 4 dev cell types (Panc1 moved to held out, section 2). It beats the cross-cell
  average on MSE everywhere and on Pearson only in dev cell types.
- **Mixture supervision** (section 3): a model trained only on two-cell-type mixtures separates their molecules as
  well as one trained on each cell type's own track, on two pairs.
- **Failures on the pairs** (sections 4–5) come from track-scale outliers (H1 CTCF, HepG2 / H1 H3K9me3),
  compression on unseen cell types, and cell-level offsets.
- **Replicate ceiling** (section 9): ENCODE replicates agree at only r ≈ 0.4, so the best reachable r is ~0.74.
  DEV4 gets 66% of it in dev cell types, 58% held out, 46% in Panc1. The gap is mostly the cell-specific part of
  CTCF, H3K36me3 and H3K27me3 in held-out cell types.

## Input ablations (sections 6, 7, 8, 10)

Five models, the same in everything (data, architecture, budget, reference DNA) except the fiber channels they see.
Each was trained twice (seed 919, seed 2).

| Model | m6A | CpG | MSP | Nucleosome | FIRE | What it tests |
|---|---|---|---|---|---|---|
| **DEV4** | ✓ | ✓ | ✓ | ✓ | ✓ | reference (all inputs) |
| **No FIRE** | ✓ | ✓ | ✓ | ✓ | – | is FIRE needed? |
| **m6A + CpG** | ✓ | ✓ | – | – | – | is any fibertools call needed? |
| **No accessibility** | – | ✓ | – | – | – | what is left from CpG + DNA alone? |
| **No CpG** | ✓ | – | ✓ | ✓ | ✓ | what does CpG methylation add? |

MSP, nucleosome and FIRE are fibertools calls computed from the m6A pattern. FIRE's classifier was also trained on
GM12878 DNase and CTCF ChIP, so it carries knowledge of bulk assays into the input.

### 1. Bulk accuracy on chr21

Pearson r minus DEV4's, dev cell types, mean of the two seeds. **Bold** = beyond 2 SD of seed noise with the same
sign in both seeds (one training moves these entries by ~0.003–0.013).

| Assay | DEV4 r | No FIRE | m6A + CpG | No accessibility | No CpG |
|---|---|---|---|---|---|
| ATAC | 0.76 | +0.01 | +0.01 | **−0.14** | −0.01 |
| H3K4me3 | 0.53 | 0.00 | 0.00 | **−0.05** | **−0.04** |
| H3K27ac | 0.49 | **+0.02** | **+0.02** | **−0.09** | **−0.04** |
| H3K4me1 | 0.54 | **+0.03** | **+0.03** | **−0.13** | −0.01 |
| CTCF | 0.39 | **+0.01** | 0.00 | **−0.09** | −0.01 |
| H3K36me3 | 0.48 | 0.00 | 0.00 | −0.02 | **−0.24** |
| H3K27me3 | 0.44 | **+0.01** | **+0.01** | **−0.03** | **−0.08** |
| H3K9me3 | 0.39 | 0.00 | +0.01 | **−0.08** | 0.00 |
| **Mean** | 0.50 | +0.01 | +0.01 | **−0.08** | **−0.05** |

![Ablation effects in both seeds](../plots/26_nov_13/seeds/seeds_effects_r.png)

- **No FIRE and m6A + CpG cost nothing, in both seeds**, and gain +0.02–0.03 on H3K27ac and H3K4me1. In bulk the
  model gets everything it needs from raw m6A and CpG.
- **m6A carries ATAC, H3K4me1, H3K27ac, CTCF and H3K9me3.** Without it they drop 0.08–0.14; held out the drop is
  larger (ATAC −0.21, H3K4me1 −0.19, CTCF −0.14).
- **CpG carries H3K36me3** (−0.24; −0.46 in K562), consistent with gene-body methylation, and H3K27me3. H3K4me3
  uses both inputs. CTCF and H3K9me3 don't need CpG, against what I predicted.
- **On the ceiling scale** (section 9): no accessibility costs ~10 points of the reachable r, no CpG halves dev
  H3K36me3 (67% → 33%). No model fixes held-out H3K36me3 (5–10% of its cell-specific ceiling).
- **Two single-seed details fell** with the second seed: no FIRE's H3K27me3 gain in held-out cell types and its
  higher H3K27me3 MSE in the dev lines.

### 2. CTCF footprints on single molecules

At bound CTCF sites on chr21, fibers whose motif is protected from m6A (a footprint) were compared with open fibers
at the same site. In the data, 41–43% of accessible fibers carry the footprint at bound sites, 12–13% at unbound
ones. The model only ever saw bulk CTCF ChIP.

The models differ only in their input channels (table above). DEV4 → no FIRE → m6A + CpG removes the fibertools
calls one step at a time: first FIRE, the call trained on CTCF ChIP, then MSP and nucleosome too. No accessibility
sees no m6A, so it cannot see the footprint (the null). The fiber states are measured, so they are the same for every
model.

Footprint − open, predicted CTCF at bound sites (GM12878 / K562). CTCF − ATAC = the same difference for CTCF minus
that for ATAC (specificity: both fiber states are accessible).

| Model | Seed 919 | Seed 2 | CTCF − ATAC, seed 919 | CTCF − ATAC, seed 2 | Unbound sites |
|---|---|---|---|---|---|
| DEV4 | +0.11 / +0.04 | +0.73 / +0.44 | +0.02 / +0.05 | +0.23 / +0.21 | ≈ 0 |
| **No FIRE** | **+1.14 / +0.84** | **+0.95 / +0.68** | **+0.67 / +0.54** | **+0.85 / +0.63** | ≈ 0 |
| m6A + CpG | +0.26 / +0.20 | +0.56 / +0.58 | +0.20 / +0.14 | +0.55 / +0.50 | ≈ 0 |
| No accessibility (null) | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | 0 |
| No CpG | +0.18 / +0.10 | +0.75 / +0.46 | −0.12 / −0.02 | +0.33 / +0.31 | ≈ 0 |

![Single-molecule readouts, both seeds](../plots/26_nov_13/seeds/seeds_molecules.png)

- **Every model that sees m6A reads the footprint**, and nothing happens at unbound motifs of the same score or in
  the no-accessibility null.
- **The size varies 2–10× between trainings** (DEV4 +0.11 → +0.73). Seed 919's weak DEV4 made no FIRE look ~10×
  stronger; in seed 2 the ratio is 1.3–1.5×.
- **No FIRE is the only model strong and CTCF-specific in both trainings** (CTCF − ATAC 3–4× DEV4's even in seed
  2). A likely reason: with FIRE, a CTCF-ChIP-trained call, in the input, the model can take "CTCF site" from that
  channel instead of the molecule.
- **m6A + CpG** reaches a quarter of no FIRE in seed 919 and 60–85% in seed 2, so whether the MSP / nucleosome calls
  help the molecule read is open.

### 3. X inactivation per haplotype (GM12878)

At silenced chrX promoters (never seen in training), active X minus inactive X per gene; Xi identified from
measured XIST promoter methylation. Fraction of CpG-island promoters in the expected direction, seed 919 → seed 2:

| Expected on the active X | DEV4 | No FIRE | m6A + CpG | No accessibility | No CpG |
|---|---|---|---|---|---|
| ATAC higher | .91 → .91 | .90 → .90 | .89 → .90 | .88 → .88 | .88 → .89 |
| H3K4me3 higher | .84 → .86 | .87 → .86 | .88 → .82 | .88 → .86 | .85 → .84 |
| H3K27ac higher | .84 → .87 | .90 → .86 | .91 → .88 | .90 → .87 | .89 → .88 |
| H3K9me3 lower | .76 → **.52** | .71 → .65 | .64 → .61 | **.21 → .67** | **.86 → .86** |
| H3K27me3 lower | .26 → .24 | .46 → .17 | .35 → .17 | .29 → .23 | .22 → .23 |

- **The active marks pass in every model and seed** (82–91%), from CpG alone or accessibility alone: the signal is
  redundant across inputs. Nulls (escape / PAR genes, chr21) are flat and XIST is reversed in all ten runs.
- **H3K27me3 fails in every model** (17–46%, below half in all ten runs): the models predict more H3K27me3 on the active X, probably applying
  the autosomal rule that Polycomb sits at unmethylated CpG islands.
- **H3K9me3 is reliable only without CpG** (86% in both seeds). With CpG in the input it varies between trainings
  (DEV4 76% → 52%; CpG alone 21% → 67%): the per-gene differences are small and flip easily.

### What each channel does

| Channel | Bulk | Single molecule |
|---|---|---|
| **FIRE** | Adds nothing; including it costs 0.02–0.03 on H3K27ac / H3K4me1 | With it, the CTCF footprint read is weaker or less specific than without (both seeds) |
| **MSP / nucleosome** | Add nothing | Open: no FIRE ≥ m6A + CpG in both seeds, by 1.2–4.4× |
| **m6A** | ATAC, H3K4me1, H3K27ac, CTCF, H3K9me3 | Required for the footprint read; Xi active marks also come from CpG alone |
| **CpG** | H3K36me3, H3K27me3, part of H3K4me3 | Not needed for the footprint read; with it, Xi H3K9me3 varies between trainings |

**Seed noise** (section 10): bulk numbers move ~0.01 between trainings and every bulk conclusion above holds in
both seeds. Molecule readouts keep their direction but vary 2–10× in size, so molecule claims need several seeds.

## Scorecard

| Test | Hypothesis | Result (both seeds) |
|---|---|---|
| No FIRE | Costs little | ✅ costs nothing; +0.02–0.03 on H3K27ac / H3K4me1 |
| m6A + CpG (no fibertools calls) | Costs little | ✅ costs nothing in bulk; footprint response 1.2–4.4× weaker than no FIRE |
| No CpG | CpG-linked marks lose most | ◐ H3K36me3 / H3K4me3 yes; CTCF / H3K9me3 no; H3K27me3 unexpectedly |
| CTCF footprints | Footprint fibers get more CTCF than open fibers, more than ATAC | ✅ every model with m6A; size varies 2–10×; strong and specific in both seeds only without FIRE; null clean |
| X inactivation | Xa > Xi for active marks, Xa < Xi for repressive; nulls ≈ 0; XIST reversed | ✅ active marks, nulls, XIST; ◐ H3K9me3 (robust only without CpG); ❌ H3K27me3 |
| Mixture supervision (section 3) | Mixtures reach the own-track ceiling | ✅ both pairs |
| Replicate ceiling (section 9) | Weak broad marks are target noise; held-out gap is not | ◐ H3K36me3 / H3K27me3 in dev yes, H3K9me3 no; ✅ held-out targets as reproducible as dev |

## Next

- **Switch to the no-FIRE model:** level or slightly better in bulk, no bulk-assay knowledge in the input, and the
  only model with a reliably strong, CTCF-specific footprint read. Keep the MSP / nucleosome inputs (free in
  bulk; their per-molecule value is open).
- **Run molecule tests over at least two seeds** (or an ensemble); consider 3 seeds of the no-FIRE model.
  Analyse the seed-2 pair evals (done, not yet looked at).
- **H3K27me3 on the Xi:** test over broader windows (gene bodies, Xi domains), and restrict X inactivation to genes
  expressed in GM12878.
- **Remaining single-molecule tests:** in-silico CpG flips, measured methylation as held-out data for the no-CpG
  model, imprinting, allele-specific ChIP.
- **Bulk:** the room is in the cell-specific CTCF, H3K36me3 and H3K27me3 of held-out cell types, and H3K9me3
  everywhere (section 9).
