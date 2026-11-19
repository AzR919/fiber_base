# 20th Nov: single-molecule locus figures

These figures accompany `Nov_20.md`. They show individual GM12878 fibers at a few loci from the two biology tests,
with the model's per-bp predictions on each fiber. The comparison is DEV4 (all inputs) against DEV4NF (no FIRE, the
main model), each in both training seeds (919 and 2).

The loci in each section were chosen from measured data only (fiber states and inputs, never model predictions), so
the choice cannot favour a model (`select_loci.py`).

**How to read the per-locus figures:**
- Each row is one fiber. The left strip gives its group: the fiber state at the motif (CTCF), or its haplotype
  (X inactivation).
- The first panel is the measured input. MSP is blue, nucleosome is grey, and black marks are m6A calls (CTCF) or
  CpG calls (X inactivation: black = methylated, white = unmethylated).
- The other panels are per-bp predictions on the same fibers, in asinh fold change. Panels of the same assay share
  one colour scale.
- The bottom row is the mean over the fibers in each group. In the CTCF figures it also shows the ENCODE GM12878
  bulk track of the same assay (black dashed), in asinh fold change, the units of the training target and of the
  predictions.
- The per-fiber values in the tables are means over the motif (15 bp) or over the test window (TSS ± 500 bp), the
  same windows as the statistics in `Nov_20.md`.

## 1. CTCF footprints

**The test** (`Nov_20.md` section 1). Does a fiber's predicted CTCF follow whether CTCF is bound *on that
molecule*? Each fiber at a CTCF motif is classed by its own m6A pattern:
- *footprint:* the motif is protected inside an accessible patch, the mark of a bound CTCF;
- *open:* accessible but unprotected;
- *nucleosome:* covered by a nucleosome.

Expected at bound sites: footprint > open > nucleosome. At unbound motifs: no footprint effect.

**Loci:**

| Locus | Rule |
|---|---|
| 3 bound sites (`ctcf_bound_179`, `_319`, `_306`) | GM12878 CTCF peak; most fibers in the smaller of the footprint / open states |
| 1 unbound motif (`ctcf_unbound_76`) | no peak in any dev line, with bulk CTCF and ATAC data (not a hole, see below); most accessible fibers |

![CTCF loci, all four runs](../plots/26_nov_20/loci/ctcf_loci_summary.png)

*Mean predicted CTCF over the motif, per fiber state, for every run; dashed line = ENCODE GM12878 bulk CTCF at the
motif.*

Mean predicted CTCF over the motif, footprint / open fibers, and the bulk value at the motif:

| Site (fibers: footprint / open / nucleosome) | Bulk | DEV4 | **DEV4NF** | DEV4 seed 2 | DEV4NF seed 2 |
|---|---|---|---|---|---|
| bound_179 (56 / 61 / 17) | 4.47 | 1.60 / 1.69 | **1.98 / 1.79** | 2.46 / 1.83 | 2.31 / 2.11 |
| bound_319 (54 / 60 / 19) | 4.01 | 1.42 / 1.46 | **2.92 / 1.48** | 2.01 / 1.45 | 3.20 / 1.87 |
| bound_306 (72 / 51 / 14) | 4.19 | 1.86 / 1.98 | **3.74 / 2.00** | 2.89 / 2.06 | 3.42 / 1.93 |
| unbound_76 (7 / 74 / 79) | 1.32 | 0.41 / 0.64 | **0.46 / 0.71** | 0.40 / 0.67 | 0.56 / 0.80 |

Nucleosome-covered fibers get 0.3–1.0 in every run.

![CTCF bound site 306](../plots/26_nov_20/loci/ctcf_ctcf_bound_306.png)

![CTCF bound site 319](../plots/26_nov_20/loci/ctcf_ctcf_bound_319.png)

![CTCF bound site 179](../plots/26_nov_20/loci/ctcf_ctcf_bound_179.png)

- **The measured footprint is clear.** On footprint fibers, the m6A rate drops to ~0 over the motif and stays at
  0.6–0.7 just around it. Open fibers stay high across the motif, and nucleosome fibers are low throughout.
- **DEV4 (seed 919) ignores it:** footprint and open fibers get the same predicted CTCF at all three bound sites.
  The heatmap is one broad, uniform band over the accessible patch.
- **DEV4NF reads it in both seeds.** Footprint fibers get a ~200 bp block of high CTCF centred on the motif
  (+1.4 to +1.7 above open fibers at sites 319 and 306; +0.2 at 179). In seed 919, open fibers get a lower,
  two-humped profile with a dip right at the motif (sites 319 and 306), so the model lowers CTCF where the motif is
  visibly unprotected. In seed 2 their profile is flatter.
- **DEV4 seed 2 is in between** (+0.6 to +0.8). This is the seed dependence from `Nov_20.md` section 1: DEV4 can
  learn the response, but not reliably. DEV4NF has it in both trainings.
- **Specificity:** DEV4NF's predicted ATAC also rises on footprint fibers, but less, and over a wider patch. The
  CTCF separation is the larger one.

**Against the bulk CTCF track.** The idea: the ChIP signal comes from molecules where CTCF is bound, so the
predictions on footprint fibers should look more like the bulk track than those on open fibers. This holds for the
height, not for the shape.

- **Height: yes.** At the motif, DEV4NF's footprint fibers reach 73–89% of the bulk level at sites 319 and 306, in
  both seeds (44–52% at 179). Its open fibers stay at 37–48%. DEV4 seed 919 gives both states 35–47% of the bulk.
- **Shape: no, but the bulk is not a per-molecule shape reference.** Pearson r between each state's mean profile and
  the bulk profile over ± 300 bp (`ctcf_loci_shape.png`):

  | Run | Footprint | Open |
  |---|---|---|
  | DEV4 | 0.84–0.96 | 0.82–0.95 |
  | **DEV4NF** | **0.75–0.84** | **0.84–0.93** |
  | DEV4 seed 2 | 0.85–0.93 | 0.86–0.96 |
  | DEV4NF seed 2 | 0.77–0.82 | 0.85–0.94 |

  DEV4NF's footprint profile matches the bulk *less* well than its open profile (sites 319 and 306; equal at 179).
  The figures show why. The bulk peak is broad (about 400 bp wide at half height), because ChIP-seq smears every binding event over its
  fragment length. DEV4NF's footprint block is about 200 bp, bounded by the accessible patch on the molecule, with
  steep edges. The broader open-fiber profile (and DEV4's broad band) correlates better with a broad peak. A fair
  shape test needs a sharper reference (ChIP-exo or CUT&RUN), or the predictions smoothed by the ChIP fragment
  distribution first.

![CTCF bulk-shape match](../plots/26_nov_20/loci/ctcf_loci_shape.png)

*Pearson r of each state's mean predicted CTCF profile with the bulk CTCF profile (± 300 bp), per site and run.*

**The unbound control.** The first two picks for this site sat where short reads don't map, so I replaced them.
- **`ctcf_unbound_26` (7.1 Mb):** GM12878 and K562 CTCF fold change exactly 0 over ± 20 kb.
- **`ctcf_unbound_52` (10.0 Mb):** ATAC exactly 0 over ± 1 kb despite 105 accessible fibers.

At such sites "no peak in any dev line" means "not measured", not "unbound". The rule is now: no unbound site where
the bulk CTCF or ATAC fold change is exactly 0 over the motif ± 1 kb in any dev line (`ctcf_holes.py`).

![CTCF unbound motif 76](../plots/26_nov_20/loci/ctcf_ctcf_unbound_76.png)

- **`ctcf_unbound_76` (14.7 Mb) is a real negative.** It is the most accessible unbound motif with data in both
  tracks: a strong motif (18.3 bits) in weakly open chromatin. Bulk CTCF at the motif is 1.3 (fold change ≈ 1.7,
  background), against 4.0–4.5 at the bound sites. Only 7 of its 81 accessible fibers carry a footprint, the
  background rate of the call.
- **No run predicts a CTCF block.** Predictions stay low (0.4–0.8) in every run. The few footprint fibers get
  *less* predicted CTCF than open fibers (0.40–0.56 against 0.64–0.80), about as much as nucleosome fibers.
- **This changed the unbound set of `Nov_20.md` section 1.** 74 of its 272 unbound sites (27%) are such holes, 46
  of them in the repetitive 5–9 Mb of chr21; only 1 of 330 bound sites is
  (`plots/26_nov_20/ctcf_noholes/ctcf_sites_bulk_holes.tsv`). Rerun without them (`Nov_20.md` section 1, last
  subsection): the conclusions hold and the null is cleaner. Unbound footprint − open stays ≈ 0 or slightly
  negative, and bound sites score higher than unbound ones in every motif-score bin.

## 2. X inactivation per haplotype

**The test** (`Nov_20.md` section 2). GM12878 is phased, so each fiber belongs to one X copy. The inactive X
(Xi = HP2) is identified from the measured CpG methylation of the XIST promoter. Per gene, the score is Xa − Xi:
mean prediction over Xa fibers minus mean over Xi fibers, over TSS ± 500 bp. Expected:
- at silenced genes with a CpG-island promoter, ATAC, H3K4me3 and H3K27ac higher on the Xa, H3K9me3 and H3K27me3
  lower;
- at escape genes and chr21 genes (an arbitrary HP1 / HP2 pair), no difference;
- at XIST, the reverse of silenced genes.

**Loci:**

| Locus | Rule |
|---|---|
| PTCHD1, CASK, PIGA | silenced CGI genes with the largest measured CpG-methylation difference (Xa − Xi −0.59, −0.55, −0.51); ≥ 15 fibers per haplotype |
| TXLNG | escape gene with the most fibers on both haplotypes |
| PRMT2 | chr21 CGI-promoter gene with the most fibers on both haplotypes |
| XIST | the only XIST TSS |

![X inactivation loci, all four runs](../plots/26_nov_20/loci/xi_loci_summary.png)

*Xa − Xi over TSS ± 500 bp, per gene and mark, for every run. Expected at silenced genes: ATAC, H3K4me3 and H3K27ac
above 0; H3K9me3 and H3K27me3 below. Expected for TXLNG and PRMT2: ≈ 0. Expected for XIST: reversed.*

Xa − Xi over TSS ± 500 bp, DEV4NF seed 919 (range over all four runs in brackets):

| Gene | ATAC | H3K4me3 | H3K27ac | H3K9me3 | H3K27me3 |
|---|---|---|---|---|---|
| PIGA (silenced) | +1.76 (1.44 to 1.76) | +2.34 (2.27 to 2.56) | +2.08 (1.44 to 2.08) | −0.04 (−0.24 to −0.04) | −0.05 (−0.05 to +0.21) |
| CASK (silenced) | +1.20 (0.78 to 1.20) | +1.94 (1.61 to 1.94) | +1.51 (0.93 to 1.51) | +0.05 (−0.03 to +0.14) | +0.14 (0.14 to 0.41) |
| PTCHD1 (silenced) | +0.40 (0.20 to 0.40) | +0.17 (0.05 to 0.22) | +0.07 (−0.05 to +0.07) | +0.07 (−0.03 to +0.13) | **+0.74 (0.70 to 1.03)** |
| TXLNG (escape) | +0.19 (0.17 to 0.19) | +0.28 (0.28 to 0.44) | +0.24 (0.24 to 0.31) | −0.02 (−0.03 to −0.02) | −0.09 (−0.10 to −0.05) |
| PRMT2 (chr21) | −0.04 (−0.08 to −0.02) | +0.10 (−0.10 to +0.10) | +0.02 (−0.11 to +0.02) | +0.02 (0.01 to 0.02) | +0.03 (0.02 to 0.06) |
| XIST | −0.69 (−0.69 to −0.51) | −1.31 (−1.31 to −0.87) | −1.45 (−1.45 to −0.86) | +0.12 (0.09 to 0.17) | +0.10 (−0.17 to +0.10) |

![PIGA](../plots/26_nov_20/loci/xi_PIGA.png)

![CASK](../plots/26_nov_20/loci/xi_CASK.png)

- **PIGA and CASK show the textbook picture.** On the Xi, the promoter CpGs are methylated and the fibers mostly
  nucleosome-covered. On the Xa they are unmethylated and open. DEV4NF puts a strong ATAC / H3K4me3 / H3K27ac
  signal on the Xa fibers (+1.2 to +2.3) and almost nothing on the Xi fibers. It does this fiber by fiber, not as a
  smooth average.
- **The repressive marks are small at the promoter.** At PIGA, predicted H3K27me3 and H3K9me3 are visibly higher on
  the Xi fibers over the flanks (outside ± 500 bp), but within the test window the difference is near 0. This is
  the "promoter window may be the wrong scale" point in `Nov_20.md`.

![PTCHD1](../plots/26_nov_20/loci/xi_PTCHD1.png)

- **PTCHD1 has the largest measured methylation difference, but almost no active marks on either copy.** It is a
  neuronal gene, probably not expressed in GM12878 (to check with RNA-seq). The models predict much more H3K27me3
  on the Xa (+0.70 to +1.03 in all four runs). This is the largest case of the H3K27me3 "failure" in `Nov_20.md`.
  - For a silent gene, it may not be a failure. On an active X, a silent gene's unmethylated CpG island is a typical
    Polycomb (H3K27me3) target, while the methylated Xi copy is not.
  - If that holds, part of the H3K27me3 result comes from non-expressed genes, where our expected direction is
    wrong. Restricting to expressed genes (the planned next step) would separate the two.

![TXLNG](../plots/26_nov_20/loci/xi_TXLNG.png)

![PRMT2](../plots/26_nov_20/loci/xi_PRMT2.png)

- **The controls look the same on both copies.** At the escape gene TXLNG, the two haplotypes are open and
  unmethylated, and the predictions overlap (a small Xa excess of +0.2 to +0.4 on the active marks). On chr21
  (PRMT2), every difference is within ± 0.11.

![XIST](../plots/26_nov_20/loci/xi_XIST.png)

- **XIST is reversed.** Here the Xi fibers are the open, active ones (H3K4me3 −1.31 for Xa − Xi). Predicted H3K9me3
  is higher on the Xa in all four runs (+0.09 to +0.17), as expected for the silent copy. H3K27me3 is mixed.

**DEV4 vs DEV4NF at these loci:** the dashed (DEV4) and solid (DEV4NF) profiles almost overlap. Removing FIRE
changes the X-inactivation predictions very little. Where the runs differ (H3K9me3, H3K27me3), the two seeds of
one model differ as much as the two models.

## Takeaways

- **CTCF:** the change from DEV4 to DEV4NF is visible on single molecules. DEV4NF gives a motif-protected fiber a
  CTCF block, clearly above an unprotected but open fiber, in both trainings. DEV4 does this only in one of its two
  trainings, and more weakly. Footprint fibers come closer to the bulk CTCF level than open fibers do. Their profile
  shape is sharper than the bulk peak, which ChIP-seq broadens.
- **Unbound controls:** a quarter of the "unbound" CTCF motifs sat where short reads don't map. Without them the
  section 1 null is cleaner, and a real unbound motif (site 76) gets no CTCF block in any run.
- **X inactivation:** both models split the two X copies fiber by fiber at expressed silenced genes, and leave the
  escape and chr21 controls alone.
- **H3K27me3:** the clearest "wrong" case (PTCHD1) is a likely non-expressed gene, where H3K27me3 on the active
  copy may be correct.

**Sources:** loci `plots_code/26_nov_20/select_loci.py` → `plots/26_nov_20/loci/{loci_ctcf,loci_tss}.bed`,
`selection.tsv`; per-bp predictions `molecule_predict.py --save_arrays` for the four runs, job
`plots_code/26_nov_20/molecules_loci.sh` (`logs/2026-11-18_00_molecules_loci`, ~2 min; CTCF loci rerun after
replacing the unbound site in `2026-11-18_01_molecules_loci_ctcf` (site 52) and `_02_` (site 76)) →
`results/*_molecules_<run>_loci_{ctcf,tss}/`; figures `plots_code/26_nov_20/locus_plots.py` →
`plots/26_nov_20/loci/` (`ctcf_<site>.png`, `xi_<gene>.png`, `ctcf_loci_summary.{png,tsv}`,
`ctcf_loci_shape.{png,tsv}`, `xi_loci_summary.{png,tsv}`, `log_locus_plots.txt`). Bulk tracks: the ENCODE GM12878
fold-change bigWigs of the training data (`configs/metapaths.yaml`). Sites without bulk short-read data:
`plots_code/26_nov_20/ctcf_holes.py` → `plots/26_nov_20/ctcf_noholes/` (`ctcf_sites_bulk_holes.tsv`, `holes.txt`).
