# 20th Nov

The two single-molecule tests (CTCF footprints, X inactivation), moved here from `Nov_13_extended.md` (sections 7
and 8). Section 1 is unchanged apart from the model name. Section 2 is rewritten around the main model and the one
gene group the test is about. Bulk results, the replicate ceiling and the second-seed analysis stay in
`Nov_13_extended.md` (sections 6, 9, 10).

## Models

**DEV4NF** ("DEV4, no FIRE") is the main model from now on (decision in `Nov_13_extended.md`, "Decisions and next
steps"). It is DEV4 with the FIRE channel removed. Older reports, figures and file names call it *no FIRE* /
`noFIRE`.

All five models share data (`data24`), architecture (`model15`), training budget (`train04`) and the reference DNA
input. They differ only in which fiber channels they see, and each was trained twice (seeds 919 and 2):

| Model | m6A | CpG | MSP | Nucleosome | FIRE | Config |
|---|---|---|---|---|---|---|
| **DEV4NF** (main) | ✓ | ✓ | ✓ | ✓ | – | `model16_uct_mid_8A_no_fire_msp` |
| DEV4 | ✓ | ✓ | ✓ | ✓ | ✓ | `model15_uct_mid_8A` |
| m6A + CpG | ✓ | ✓ | – | – | – | `model21_uct_mid_8A_m6a_cpg` |
| No accessibility | – | ✓ | – | – | – | `model20_uct_mid_8A_cpg_only` |
| No CpG | ✓ | – | ✓ | ✓ | ✓ | `model19_uct_mid_8A_no_cpg` |

MSP, nucleosome and FIRE are fibertools calls computed from the m6A pattern. FIRE's classifier was also trained on
GM12878 DNase and CTCF ChIP peaks, so it carries bulk-assay knowledge into the input.

## 1. CTCF footprints on single molecules (T5e)

> **Question:** the model was trained on bulk CTCF ChIP only; per-molecule binding was never a label. Does a
> fiber's predicted CTCF follow whether CTCF is bound **on that molecule**, read from the CTCF footprint in its m6A
> pattern?

**Targets** (`data_prep/build_targets.py`, `ctcf_sites_chr21.bed`, chr21 only):
- CTCF motif hits from JASPAR MA0139.2 (15 bp) at p ≤ 1e-5 (score ≥ 13.44 bits, exact threshold from the PWM's
  score distribution).
- **Bound:** the best hit inside each ENCODE CTCF peak of a dev line, merged across lines (330 sites). Per line, a
  site is *bound* if that line has a peak there, and *bound_other_line* if only another dev line does.
- **Unbound:** motif hits more than 500 bp from every dev line's peak, sampled to match the bound sites' motif score
  in 1-bit bins (272; the highest-score bins have too few unbound hits).
- Per line: GM12878 184 bound / 146 other line / 268 unbound; K562 252 / 78 / 265.

**Per-fiber state at the motif** (fixed before looking at results):
- *nucleosome:* ≥ 50% of the motif covered by a nucleosome call;
- *footprint:* ≥ 50% inside an MSP **and** the motif's m6A rate ≤ 0.5 × the mean rate of the 100 bp flanks
  (protected);
- *open:* inside an MSP, not protected;
- *other:* anything else (read ends, < 4 A/T positions in the motif, no flank m6A calls).

**Hypotheses (fixed in advance, in the script header):**
- **H1 (main):** at bound sites, footprint fibers get higher predicted CTCF than open fibers at the same site
  (per-site difference, Wilcoxon signed-rank over sites with ≥ 3 fibers in both states).
- **H2 (specificity):** that difference is larger for CTCF than for ATAC (both states are accessible).
- **H3:** nucleosome-covered fibers get the lowest predicted CTCF.
- **H4 (data, no model):** footprints are more frequent among accessible fibers at bound than at unbound sites.
- **H5 (null):** unbound sites get lower predicted CTCF than bound sites of the same motif score.

Caveat: the model sees m6A (so the footprint) and the reference sequence (so the motif, in both sets). The test is
whether bulk supervision taught it that a protected motif on an accessible molecule means CTCF. The
no-accessibility model sees neither m6A nor MSPs, so it is a built-in null.

### Data check (H4)

Fiber states, measured input (the same for every model):

| Sample | Site set | Sites | Fibers | Nucleosome | Open | Footprint | Other | Footprint / accessible |
|---|---|---|---|---|---|---|---|---|
| GM12878 | bound | 184 | 21,964 | .36 | .29 | .22 | .12 | **.43** |
| | bound, other line | 146 | 18,033 | .67 | .24 | .04 | .05 | .13 |
| | unbound | 268 | 30,482 | .66 | .28 | .04 | .02 | .12 |
| K562 | bound | 252 | 22,742 | .34 | .31 | .21 | .14 | **.41** |
| | bound, other line | 78 | 6,878 | .54 | .36 | .05 | .06 | .12 |
| | unbound | 265 | 21,892 | .52 | .39 | .06 | .03 | .13 |

**H4 holds:** 41–43% of accessible fibers carry the footprint at bound sites, against 12–13% at unbound and
other-line sites (the background rate of our simple call). The literature value is ~30% in K562 (Stergachis 2020),
with a different footprint caller.

### Model results

The three figures below are the same model trained with different input channels (Nov 13 extended, section 6), everything else
equal. They remove the fibertools calls, which are all derived from the m6A pattern, one step at a time. All three
are seed 919; both seeds are compared in the last table of this subsection.

![CTCF footprints, DEV4](../plots/26_nov_13/molecules/ctcf_footprints_DEV4.png)

*DEV4: all inputs (m6A, CpG, MSP, nucleosome, FIRE). The reference.*

![CTCF footprints, DEV4NF](../plots/26_nov_13/molecules/ctcf_footprints_noFIRE.png)

*DEV4NF: m6A, CpG, MSP, nucleosome. FIRE removed, the call whose classifier was trained on DNase and CTCF ChIP.
Tests whether that bulk CTCF knowledge in the input changes how the model reads the molecule.*

![CTCF footprints, m6A + CpG only](../plots/26_nov_13/molecules/ctcf_footprints_m6aCpG.png)

*m6A + CpG: all three fibertools calls removed (MSP, nucleosome, FIRE). Tests whether the model can find the
footprint from raw m6A alone, without the calls that mark the accessible patch.*

Each figure, per sample: fiber states by site set; per-fiber predicted CTCF by state at bound sites; the per-site
footprint − open difference for CTCF and ATAC; site mean predicted CTCF against the line's CTCF peak signal. The
no-CpG (CpG removed) and no-accessibility (CpG only: the null, which cannot see the footprint) figures are
`ctcf_footprints_noCpG.png` / `_noAcc.png` in the same folder. The fiber
states are computed from the measured MSP / nucleosome calls for every model, including the two that don't see
them as input.

At bound sites (per-site medians; n = 135 GM12878 / 184 K562 sites with both states):

| Model | Sample | Footprint − open, CTCF (fraction of sites > 0, p) | Footprint − open, ATAC | Footprint − open, H3K4me3 | CTCF − ATAC (p) | Open − nucleosome, CTCF |
|---|---|---|---|---|---|---|
| DEV4 | GM12878 | +0.11 (0.64, 7e-5) | +0.09 | −0.10 | +0.02 (0.05) | +1.01 |
| | K562 | +0.04 (0.56, 0.05) | +0.02 | −0.09 | +0.05 (9e-8) | +0.71 |
| No CpG | GM12878 | +0.18 (0.70, 9e-8) | +0.27 | +0.09 | −0.12 (2e-13) | +1.00 |
| | K562 | +0.10 (0.59, 0.003) | +0.11 | +0.02 | −0.02 (0.007) | +0.64 |
| **DEV4NF** | GM12878 | **+1.14 (0.94, 6e-23)** | +0.41 | −0.08 | **+0.67 (7e-24)** | +1.11 |
| | K562 | **+0.84 (0.83, 3e-25)** | +0.30 | −0.07 | **+0.54 (2e-31)** | +0.83 |
| No accessibility | GM12878 | 0.00 (0.52, 0.39) | 0.00 | +0.01 | 0.00 (0.90) | −0.01 |
| | K562 | 0.00 (0.55, 0.04) | +0.01 | 0.00 | 0.00 (0.08) | 0.00 |
| m6A + CpG | GM12878 | +0.26 (0.81, 2e-14) | +0.06 | +0.03 | +0.20 (1e-21) | +1.15 |
| | K562 | +0.20 (0.64, 2e-7) | +0.04 | +0.04 | +0.14 (6e-26) | +0.88 |

Unbound and other-line sites (per-site medians):

| Model | Sample | Unbound: footprint − open, CTCF (p) | Unbound: CTCF − ATAC | Other line: footprint − open, CTCF (n sites, p) |
|---|---|---|---|---|
| DEV4 | GM12878 | −0.02 (0.62) | +0.04 | +0.04 (79, 0.002) |
| | K562 | −0.01 (0.95) | +0.04 | +0.21 (43, 0.001) |
| No CpG | GM12878 | −0.07 (0.45) | +0.05 | +0.04 (79, 0.009) |
| | K562 | −0.06 (0.24) | +0.06 | +0.15 (43, 0.01) |
| DEV4NF | GM12878 | −0.05 (0.84) | +0.20 | +0.05 (79, 9e-4) |
| | K562 | −0.04 (0.60) | +0.27 | +0.42 (43, 8e-5) |
| No accessibility | both | ≈ 0 | ≈ 0 | ≈ 0 |
| m6A + CpG | GM12878 | −0.04 (0.07) | +0.14 | +0.04 (79, 0.02) |
| | K562 | −0.03 (0.11) | +0.11 | +0.18 (43, 0.004) |

Site level (H5): mean predicted CTCF at bound / unbound sites is 1.25 / 0.58 (GM12878) and 1.09 / 0.60 (K562) for
DEV4, 1.63 / 0.61 and 1.45 / 0.64 for DEV4NF, 1.44 / 0.62 and 1.28 / 0.66 for m6A + CpG, and 0.71 / 0.57 and
0.66 / 0.58 without accessibility. Within 1-bit motif-score bins, bound sites score higher in every bin except the
20-bit bin, which has only 5 unbound sites (`ctcf_bound_vs_unbound_by_score_<model>.tsv`). Across bound sites, the
site mean follows the line's CTCF peak strength (Spearman 0.81 / 0.80 for DEV4, 0.81 / 0.85 for DEV4NF, 0.81 /
0.84 for m6A + CpG).

**Both training seeds** (bound sites, per-site medians, seed 919 / seed 2; from Nov 13 extended, section 10):

| Model | Footprint − open, CTCF: GM12878 | K562 | CTCF − ATAC: GM12878 | K562 | Unbound sites: footprint − open, CTCF |
|---|---|---|---|---|---|
| DEV4 | +0.11 / **+0.73** | +0.04 / **+0.44** | +0.02 / +0.23 | +0.05 / +0.21 | −0.06 to −0.01 |
| **DEV4NF** | **+1.14 / +0.95** | **+0.84 / +0.68** | **+0.67 / +0.85** | **+0.54 / +0.63** | −0.05 to −0.02 |
| m6A + CpG | +0.26 / +0.56 | +0.20 / +0.58 | +0.20 / +0.55 | +0.14 / +0.50 | −0.04 to +0.03 |
| No accessibility | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | ≈ 0 |
| No CpG | +0.18 / +0.75 | +0.10 / +0.46 | −0.12 / +0.33 | −0.02 / +0.31 | −0.07 to −0.03 |

![Single-molecule readouts, both seeds](../plots/26_nov_13/seeds/seeds_molecules.png)

### Outcomes

> **Second seed (Nov 13 extended, section 10):** the tables above are seed 919. Retrained with seed 2, every model that sees m6A
> responds strongly (DEV4 +0.73 / +0.44, m6A + CpG +0.56 / +0.58, no CpG +0.75 / +0.46, DEV4NF +0.95 / +0.68),
> unbound sites stay ≈ 0 and the no-accessibility null stays 0. So H1, H3 and H5 replicate, but **the size of the
> response varies 2–10× between trainings, and the comparisons between models below hold only for DEV4NF**: it is
> the strongest and the most CTCF-specific model in both seeds. Statements about DEV4, m6A + CpG and no CpG being
> weak are single-seed.

- **H1 holds for every model that sees m6A**, but weakly in DEV4 (+0.11 GM12878, +0.04 K562, p = 0.05). **In DEV4NF
  the effect is ~10× larger** (+1.14 / +0.84; 94% / 83% of sites in the expected direction). *Seed 2: DEV4
  +0.73 / +0.44 (p ≤ 6e-20), DEV4NF +0.95 / +0.68, so 1.3–1.5×, not 10×.*
- **H2:** weak in DEV4 (CTCF − ATAC +0.02 / +0.05). **Fails without CpG** (−0.12 / −0.02: there, predicted ATAC
  moves more than CTCF, so its footprint response is accessibility-like). **Strong in DEV4NF** (+0.67 / +0.54).
  H3K4me3 goes slightly *down* on footprint fibers in every model with m6A except no CpG, so the response is
  CTCF-specific, not a general "more signal". *Seed 2: H2 holds for every model with m6A, no CpG included
  (+0.33 / +0.31); DEV4NF stays the most specific (+0.85 / +0.63, against DEV4 +0.23 / +0.21).*
- **Caveat on H2:** CTCF − ATAC is also positive at unbound sites (+0.20 / +0.27 in DEV4NF). There, predicted
  ATAC drops on protected-looking fibers while predicted CTCF doesn't move. At bound sites the DEV4NF CTCF
  increase itself (+1.14 / +0.84) has no counterpart at unbound sites (−0.05 / −0.04), so the main result stands,
  but CTCF − ATAC overstates specificity by about 0.2.
- **H3 holds:** nucleosome-covered fibers are lowest in every model with accessibility (open − nucleosome +0.64 to
  +1.11).
- **H5 holds** for every model with accessibility, including at matched motif scores. The no-accessibility model
  separates bound and unbound sites only slightly (~0.1), from CpG and sequence context.
- **The no-accessibility null is clean:** every per-fiber difference is 0.00. The test does not pick up something
  other than m6A.
- **Other-line sites** (a peak in another dev line only) show a smaller effect, largest in K562 for DEV4NF (+0.42,
  43 sites): some of these sites may be weakly bound in K562 below the peak caller's threshold.
- **Interpretation (not tested):** with the FIRE channel, the model seems to take "this is a CTCF-type element" from
  FIRE, at the site level, and has little reason to read the m6A pattern inside the MSP. Without FIRE it has to
  read the m6A pattern, and it learns that a protected motif on an accessible molecule means CTCF, from bulk
  supervision alone. Together with Nov 13 extended section 6 (no bulk cost), this makes DEV4NF the better main model.
  *Seed 2 weakens this account: DEV4 can also learn a strong response with FIRE in the input, though a less
  specific one. What replicates is that without FIRE the response is reliably strong and specific.*
- **m6A + CpG (no fibertools calls) sits in between:** footprint − open CTCF +0.26 / +0.20 (81% / 64% of sites),
  about twice DEV4 but a quarter of DEV4NF. It is still absent at unbound sites (−0.04 / −0.03), and nucleosome
  fibers are still lowest (+1.15 / +0.88), so H1, H3 and H5 hold. Specificity is weaker: CTCF − ATAC is +0.20 /
  +0.14 at bound sites but +0.14 / +0.11 at unbound ones, so most of it is the generic drop in predicted ATAC on
  protected-looking fibers, and H3K4me3 rises slightly on footprint fibers (+0.03 / +0.04; also +0.05 at unbound
  GM12878 sites) instead of falling as in DEV4 and DEV4NF.
- **Interpretation (not tested):** the strongest per-molecule CTCF read needs the MSP / nucleosome calls *without*
  FIRE. The calls give the model the accessible patch, against which a 15 bp protected motif stands out; without
  them it has to find the patch from the m6A density itself, and a short gap inside an MSP is harder to tell apart
  from linker DNA. With FIRE on top, the model takes the site-level answer from FIRE. So the fibertools calls cost
  nothing to drop in bulk (Nov 13 extended, section 6) but help the molecule-level read, and FIRE is the one call to drop.
  *Seed 2: m6A + CpG reaches 60–85% of DEV4NF (+0.56 / +0.58, CTCF − ATAC +0.55 / +0.50), so the evidence that the
  MSP / nucleosome calls help is now weak: DEV4NF ≥ m6A + CpG in both seeds, but by 1.2–4.4×.*

**Sources:** targets `data_prep/build_targets.py` + `targets_spec.yaml` → `DATA_FIBER/regions/targets/`
(`manifest.json`); per-fiber predictions `molecule_predict.py`, jobs `logs/2026-11-07_05_molecules_ctcf_chr21`,
`logs/2026-11-08_{09,11,13,19}_molecules_ctcf_abl_*` (~4.5 min each); analysis
`plots_code/26_nov_13/ctcf_footprints.py --model <m>` → `plots/26_nov_13/molecules/ctcf_*_<m>.{tsv,png}`.

## 2. X inactivation per haplotype in GM12878 (T5b)

> **Question:** the models were trained on bulk tracks, in which both X copies are mixed, and never saw haplotypes
> or chrX (training is chr20 only). At promoters the inactive X has silenced, do the per-fiber predictions differ
> between the active X (Xa) and the inactive X (Xi) the way X-inactivation biology says?

**Biology.** In female cells one X is silenced early in development and stays silenced. At a silenced gene, the
Xi copy's promoter has its **CpG island (CGI)** methylated: the CpG sites inside the island, which stay
unmethylated on an active promoter, carry 5mC. The promoter is closed, lacks the active marks (ATAC, H3K4me3,
H3K27ac) and is enriched for the repressive marks H3K27me3 and H3K9me3. Escape genes and PAR genes are expressed
from both copies. XIST is the one gene expressed only from the Xi.

### Setup

- **Haplotypes:** GM12878 is trio-phased (one phase block per chromosome), so HiPhase HP1 / HP2 are
  whole-chromosome haplotypes. **Xi = the haplotype whose XIST promoter is CpG-unmethylated** (XIST is expressed
  from the Xi). This is measured input, not a model output: XIST promoter CpG methylation 0.355 on HP1 (58 fibers)
  vs 0.085 on HP2 (49 fibers), so **Xi = HP2**.
- **Targets** (`tss_chrX_chr21.bed`): TSS ± 500 bp of every protein-coding gene plus XIST on chrX and chr21 (the
  test chromosome).
- **Per gene:** d = mean prediction over Xa fibers − mean over Xi fibers, for genes with ≥ 5 fibers on each
  haplotype (113,151 phased fiber × TSS rows over 1,073 TSSs).
- **CGI promoter:** the TSS ± 500 bp window overlaps an island of the UCSC CpG-island track.

**Gene groups:**

| Group | Definition | Genes kept | Expected Xa − Xi |
|---|---|---|---|
| **Silenced, CGI promoter** (test group) | chrX gene outside the PAR and not on the escape list, with a CGI promoter | 425 | > 0 for ATAC, H3K4me3, H3K27ac; < 0 for H3K27me3, H3K9me3 |
| Escape + PAR (null) | 14 escape genes (provisional list) and the 20 genes in the hg38 pseudoautosomal regions; expressed from both copies | 33 (24 with a CGI) | ≈ 0 |
| chr21 (null) | chr21 genes; "Xa" / "Xi" are HP1 / HP2, an arbitrary pairing on an autosome | 211 (118 with a CGI) | ≈ 0 |
| XIST | expressed only from the Xi | 1 | reversed: < 0 for the active marks, > 0 for the repressive ones |

- **"Silenced" is an annotation, not a measurement.** It means "subject to X inactivation" (called *subject* in
  `Nov_13_extended.md`). Some of these genes are not expressed in GM12878 at all, so they are quiet on both copies
  and d ≈ 0 is correct for them. Restricting to genes expressed in GM12878 (RNA-seq) is still to do.
- **The escape list is provisional** (15 well-known escape genes from memory; JPX dropped as non-coding). It still
  needs checking against Tukiainen et al. 2017.
- **Why only CGI promoters.** X inactivation silences a promoter by methylating its CpG island. A promoter without
  an island is usually methylated on both copies whatever its activity, and the inputs confirm there is no Xa / Xi
  difference to follow there (data check below). The 362 silenced genes without a CGI are therefore not a test of
  anything, and pooling them with the CGI genes (the *subject* numbers in `Nov_13_extended.md`) only diluted the
  result. They are left out from here on.

**Which assays the test uses.** The models predict eight assays. The test window is the promoter (TSS ± 500 bp),
and only five assays have a clear expected Xa − Xi direction there:

| Assay | What it marks | Expected at a silenced promoter | Used |
|---|---|---|---|
| ATAC | open chromatin | Xa higher (the Xi promoter is closed) | ✓ |
| H3K4me3 | active promoters | Xa higher | ✓ |
| H3K27ac | active promoters and enhancers | Xa higher | ✓ |
| H3K9me3 | heterochromatin, enriched on the Xi | Xa lower | ✓ |
| H3K27me3 | Polycomb repression, enriched on the Xi | Xa lower | ✓ |
| H3K4me1 | enhancers and the flanks of active promoters; depleted at the TSS itself, where H3K4me3 takes over | unclear: a ± 500 bp window mixes the depleted core with the enriched flanks | – |
| H3K36me3 | bodies of transcribed genes; low in the first ~1 kb | none at the promoter: the window is the wrong place (needs gene-body windows) | – |
| CTCF | specific motif sites; most promoters have none, and the Xi keeps CTCF only at a few loci (DXZ4, FIRRE) | none per promoter (needs chrX CTCF motif sites with a footprint per haplotype, as in section 1) | – |

All the tables and figures below show only the five assays used.

**Hypotheses** (fixed in advance, in the `xi_promoters.py` header; H1 already named CGI promoters as the strongest
case, H2's restriction to CGI genes is new here):
- **H1:** at silenced CGI promoters, d > 0 for ATAC, H3K4me3, H3K27ac and d < 0 for H3K27me3, H3K9me3.
- **H2:** escape + PAR genes have smaller |d| than silenced CGI genes.
- **H3 (null):** chr21 genes have d ≈ 0.
- **H4:** XIST is the reverse of H1.

Caveat: the model sees the fibers' CpG methylation and accessibility, which already differ between Xa and Xi at
silenced CGI promoters. So the test does not show the model discovering the Xi. It shows whether the model turns
those input differences into the right direction for each mark, on a chromosome and haplotypes it never saw. The
ablations show which input carries each mark.

### Data check (inputs)

Median Xa − Xi per gene, measured input (the same for every model):

| Group | Genes | CpG methylation | MSP fraction |
|---|---|---|---|
| **Silenced, CGI promoter** | 425 | **−0.234** | **+0.082** |
| Silenced, no CGI (left out) | 362 | +0.018 | +0.015 |
| Escape + PAR | 33 | 0.000 | −0.002 |
| chr21 | 211 | −0.004 | −0.001 |
| XIST | 1 | +0.270 | −0.137 |

The Xi's silenced CGI promoters are methylated and less accessible; the other groups show no difference, and XIST
is reversed (unmethylated and more accessible on the Xi). The flat escape + PAR row also supports the provisional
escape list in aggregate.

### DEV4NF, both seeds

![X inactivation per gene, DEV4NF](../plots/26_nov_20/xi/xi_DEV4NF.png)

*d per gene for the three groups, seed 919 (blue) and seed 2 (orange); XIST as a star; grey line at 0. The two
input panels are measured data, the same for both seeds.*

At silenced CGI promoters (425 genes), seed 919 / seed 2:

| Mark (expected Xa − Xi) | Genes in the expected direction | Median d | Wilcoxon p | XIST d (expected) |
|---|---|---|---|---|
| ATAC (> 0) | **.90 / .90** | +0.53 / +0.45 | 2e-63 / 8e-63 | −0.69 / −0.51 (< 0) ✓ |
| H3K4me3 (> 0) | **.87 / .86** | +0.62 / +0.54 | 1e-59 / 2e-57 | −1.31 / −0.87 (< 0) ✓ |
| H3K27ac (> 0) | **.90 / .86** | +0.53 / +0.29 | 1e-61 / 6e-56 | −1.45 / −0.86 (< 0) ✓ |
| H3K9me3 (< 0) | .71 / .65 | −0.02 / −0.03 | 2e-28 / 2e-20 | +0.12 / +0.15 (> 0) ✓ |
| H3K27me3 (< 0) | **.46 / .17** ✗ | +0.01 / +0.25 | 2e-4 / 1e-53, wrong direction | +0.10 / −0.17 (> 0) ◐ |

**H2**, median |d|, silenced CGI vs escape + PAR (Mann–Whitney, one-sided), seed 919 / seed 2:

| | ATAC | H3K4me3 | H3K27ac | H3K9me3 | H3K27me3 |
|---|---|---|---|---|---|
| Silenced CGI | 0.54 / 0.45 | 0.64 / 0.57 | 0.54 / 0.31 | 0.04 / 0.05 | 0.11 / 0.26 |
| Escape + PAR | 0.09 / 0.10 | 0.17 / 0.10 | 0.14 / 0.07 | 0.02 / 0.02 | 0.04 / 0.07 |
| p | 1e-6 / 5e-6 | 2e-4 / 3e-5 | 5e-4 / 2e-3 | 8e-3 / 7e-5 | 2e-5 / 7e-9 |

**Nulls (H3):** escape + PAR genes are in the "expected" direction 48–64% of the time, with medians within ±0.03
and every p ≥ 0.2. chr21 genes: 43–59%, medians within ±0.01; the active marks in seed 919 reach p = 0.02–0.05,
at a median of +0.01, against +0.5 at silenced CGI promoters.

### All five models

![Fraction of silenced CGI genes in the expected direction](../plots/26_nov_20/xi/xi_cgi_fractions.png)

*One colour per model; seed 919 (circle, left) joined to seed 2 (square, right); dashed line at chance. A long
line means the result changed between trainings (H3K9me3 for DEV4 and no accessibility, H3K27me3 for DEV4NF and
m6A + CpG).*

Fraction of the 425 silenced CGI genes in the expected direction, seed 919 → seed 2:

| Mark (expected Xa − Xi) | **DEV4NF** | DEV4 | m6A + CpG | No accessibility | No CpG |
|---|---|---|---|---|---|
| ATAC (> 0) | **.90 → .90** | .91 → .91 | .89 → .90 | .88 → .88 | .88 → .89 |
| H3K4me3 (> 0) | **.87 → .86** | .84 → .86 | .88 → .82 | .88 → .86 | .85 → .84 |
| H3K27ac (> 0) | **.90 → .86** | .84 → .87 | .91 → .88 | .90 → .87 | .89 → .88 |
| H3K9me3 (< 0) | **.71 → .65** | .76 → **.52** | .64 → .61 | **.21 → .67** | .86 → .86 |
| H3K27me3 (< 0) | **.46 → .17** | .26 → .24 | .35 → .17 | .29 → .23 | .22 → .23 |

Median d at silenced CGI promoters is +0.29 to +0.82 for the active marks in every run, −0.08 to +0.02 for
H3K9me3, and +0.01 to +0.25 for H3K27me3 (the wrong sign in all ten runs).

**XIST (Xa − Xi)**, seed 919 / seed 2:

| Model | ATAC | H3K4me3 | H3K27ac | H3K9me3 | H3K27me3 |
|---|---|---|---|---|---|
| **DEV4NF** | −0.69 / −0.51 | −1.31 / −0.87 | −1.45 / −0.86 | +0.12 / +0.15 | +0.10 / −0.17 |
| DEV4 | −0.66 / −0.61 | −1.21 / −1.15 | −1.31 / −1.20 | +0.17 / +0.09 | −0.09 / −0.14 |
| m6A + CpG | −0.60 / −0.59 | −1.35 / −1.00 | −1.45 / −1.08 | +0.09 / +0.09 | −0.03 / −0.18 |
| No accessibility | −0.26 / −0.23 | −0.26 / −0.38 | −0.24 / −0.20 | −0.01 / −0.01 | −0.25 / −0.27 |
| No CpG | −0.60 / −0.55 | −1.18 / −0.77 | −1.16 / −1.13 | +0.11 / +0.15 | −0.13 / −0.10 |

Across all models and seeds: escape + PAR medians are within ±0.05 (every p ≥ 0.06), chr21 medians within ±0.02.
H2 holds for every mark, model and seed (p ≤ 0.008).

### Outcomes

- **H1, active marks: holds.** DEV4NF puts ATAC, H3K4me3 and H3K27ac higher on the Xa at 86–90% of silenced CGI
  promoters in both seeds, with median differences of +0.3 to +0.6. Every other model and seed does the same
  (82–91%), so either input alone is enough. CpG alone (no accessibility) gives the largest differences (H3K4me3
  +0.82 / +0.74): the methylation difference at these promoters (−0.23) is larger than the accessibility one
  (+0.08).
- **H1, H3K9me3: holds weakly in DEV4NF.** 71% / 65% of genes, in the right direction in both seeds, but the
  differences are small (median −0.02 / −0.03, against +0.5 for the active marks). Only no CpG is strong and stable
  (86% in both seeds, median −0.08). In the models that see CpG the fraction moves between trainings: DEV4 76% →
  52%, CpG only 21% → 67%. So the Xi H3K9me3 signal comes from accessibility ("closed chromatin → H3K9me3"), and CpG
  in the input makes it unstable.
- **H1, H3K27me3: fails.** DEV4NF is at chance in seed 919 (46%) and clearly wrong in seed 2 (17%, median +0.25);
  the other models are wrong in both seeds (17–35%). The predicted H3K27me3 is higher on whichever copy's promoter
  is open and unmethylated: on the Xa at silenced genes, and on the Xi at XIST (9 of 10 runs; there too the expected
  sign is the opposite). It does this from either input alone: no CpG (22–23%) sees only accessibility, no
  accessibility (23–29%) sees only CpG.
  - **Possible reason:** on the training chromosome, promoter H3K27me3 sits at Polycomb-target CGIs, which are
    unmethylated and nucleosome-depleted, the same features that mark the Xa copy here. On the Xi, H3K27me3 covers
    methylated, closed CGIs, a combination the model never saw.
  - Xi H3K27me3 also forms broad domains, so a ± 500 bp promoter window may be the wrong scale to test it.
- **H2 holds** for every mark, model and seed (p ≤ 0.008): silenced CGI genes differ more than escape + PAR genes,
  by 2–6× in DEV4NF (median |d|) and up to 15× in the CpG-only model. In `Nov_13_extended.md` it held for only 6 of 8 assays; that was dilution by the non-CGI genes. This
  restriction was made after seeing the data. For H3K27me3, H2 holds in size but not in direction.
- **H3 holds:** escape + PAR and chr21 genes show no Xa / Xi difference in any model or seed.
- **H4 holds for the active marks and H3K9me3:** XIST is more active on the Xi in all ten runs (DEV4NF H3K4me3 −1.31
  / −0.87), and H3K9me3 is higher on the Xa in the eight runs with accessibility. Its H3K27me3 follows the
  "open copy" pattern above instead. XIST is one gene, so this is a check, not a statistic.

**What it shows:** at the promoters X inactivation acts on, DEV4NF's per-fiber predictions separate the two X
copies in the right direction for ATAC, H3K4me3, H3K27ac and (weakly) H3K9me3, in both trainings, on a chromosome
and haplotypes it never saw, with clean nulls. H3K27me3 is the exception: the model applies a rule learned on
autosomal Polycomb promoters that is backwards on the Xi.

**Next:**
- restrict the silenced group to genes expressed in GM12878 (RNA-seq);
- check the escape list against Tukiainen et al. 2017;
- test H3K27me3 over broader windows (gene bodies, Xi H3K27me3 domains) as well as promoters.

**Sources:** targets `data_prep/build_targets.py` (`tss_chrX_chr21.bed`); per-fiber predictions
`molecule_predict.py`, jobs `logs/2026-11-07_07_molecules_tss_chrX` (DEV4; job 06 crashed on a column-name bug,
fixed in 55c22a8), `logs/2026-11-08_{10,12,14,20}_molecules_tss_abl_*` (seed 919) and the seed-2 molecule jobs in
`logs/2026-11-11_*_seed2*`; per-gene d `plots_code/26_nov_13/xi_promoters.py --model <m> [--seed 2]` →
`plots/26_nov_13/molecules/xi_per_gene_<m>[_seed2].tsv`; this section's tables and figures
`plots_code/26_nov_20/xi_cgi.py` → `plots/26_nov_20/xi/` (`xi_cgi_summary.tsv`, `xi_cgi_inputs.tsv`,
`xi_cgi_h2.tsv`, `xi_cgi_xist.tsv`, `xi_DEV4NF.png`, `xi_cgi_fractions.png`, `log_xi_cgi.txt`).
