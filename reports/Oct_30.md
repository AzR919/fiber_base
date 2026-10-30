# 30th Oct

Notes for the presentation: each section below is one or two slides.

## Setup

- **Regions:** train on chr20, validate on chr22 during training, test on chr21. No chromosome is used for two
  purposes.
  - Training samples are drawn half from cCREs and half from anywhere (background).
  - Test: all 6,843 valid 5 kb tiles of chr21 (cCREs + background). Results can be sliced by region type
    (promoter, enhancer, CTCF, non-cCRE).
- **Targets:** ENCODE fold change over control. Fold change is linear in the number of marked molecules, which
  matches how the model builds the bulk prediction (an average over molecules).
- **Cell types and assays:** 5 development cell types, 3 held out; 8 assays (see the dev-test split below).

### Dev-test split: does the model work on cell types it has never seen?

#### High-level idea

- Does the model's performance carry over to new cell types (cell-type agnostic)?
- Train on one group of cell types; test on a group the model has never seen.
- **Hypothesis (H):** the model learns the transformation from Fiber-seq (+ DNA) to histone modifications, and
  that transformation does not depend on the cell type.
- **Null (H0):** it does not learn that transformation. Two ways this can happen:
  - **H0a, memorisation:** it learns each training cell type's behaviour, so it is good on dev and falls off on
    test.
  - **H0b, average from DNA:** it predicts the average cell type from the sequence and ignores the fibers. This
    looks fine on both dev and test, but adds nothing cell-specific.

#### Division of cell types

- **Dev set (5):** GM12878 (B cell), K562 (myeloid leukemia), HepG2 (liver), Panc1 (pancreas), H1 (embryonic stem
  cell).
- **Test set (3), from near to far:**
  - H9: embryonic stem cell, a twin of H1;
  - HCT116: colon, gut-derived like HepG2 and Panc1;
  - MCF-7: breast, no close relative in dev.
- Near to far lets us see how performance falls off as a new cell type gets less familiar, not just one number.
- Held-out cell types are never used for training or for any decision (architecture, hyperparameters,
  checkpoint).

#### Assays and baseline

- **8 assays:**
  - Narrow peaks: ATAC, H3K4me3, H3K27ac, CTCF.
  - Broad peaks: H3K4me1, H3K36me3, H3K27me3, H3K9me3.
  - ENCODE has no ATAC for H1 and H9, so they are scored on the other 7.
- **Baseline: average activity.**
  - For each cell type, the average of the other dev cell types' tracks.
  - It knows nothing about the cell, but it is strong because most of every track is shared across cell types.

#### Metrics

- **MSE** [lower is better], range [0, ∞): are the signal levels right?
- **Pearson r** [higher is better], range [−1, 1]: is the shape of the signal right? It ignores magnitude.
- Both are computed on the bulk signal (asinh fold change, base-pair resolution, positions with ≥ 5 fibers). They
  say nothing about whether individual molecules are right.
- Recovering the bulk is necessary, not sufficient: the baseline already recovers it well (r 0.49) without
  knowing anything about the cell.
- **Predictions:**
  - Under H: the bulk is recovered about as well on test as on dev.
  - Under H0a: good on dev, worse on test.
  - Under H0b: close to the baseline on both.
- Bulk metrics alone cannot rule out H0b, which is why we need the next metric.

#### Cell-specific metric

- **Cell-type r** [higher is better]: where this cell type differs from the average cell type, does the prediction
  differ the same way?
  - Range [−1, 1]; the baseline scores exactly 0.
  - r(p, t | b) = corr(p_res, t_res), with t_res = t − (α + β·b) and p_res = p − (γ + δ·b).
  - p = prediction, t = target (ENCODE), b = baseline. α, β, γ and δ are fitted by least squares.
- The DNA input is identical across cell types, so a positive value means the model reads cell-specific
  information from the fibers.
- **Predictions:**
  - Under H: positive on both dev and test, at similar levels.
  - Under H0a: positive on dev, near 0 on test.
  - Under H0b: near 0 on both.

#### Summary of metrics

| Metric | What it asks | Baseline | H (learns the transformation) | H0a (memorises dev) | H0b (average from DNA) |
|---|---|---|---|---|---|
| MSE [lower is better] | Are the signal levels right? | strong | at or below the baseline, on dev and test | below on dev, not on test | ≈ baseline on both |
| Pearson r [higher is better] | Is the shape right? | strong | at the baseline, on dev and test | at the baseline on dev, below on test | ≈ baseline on both |
| Cell-type r [higher is better] | Is the cell-specific part right? | 0 by definition | positive on dev and test | positive on dev, ≈ 0 on test | ≈ 0 on both |

- MSE and Pearson score the whole track. The baseline is hard to beat on them, because most of every track is
  shared across cell types.
- Cell-type r scores only the part where this cell type differs from the average. It is the one metric that
  separates H from H0b.
- What to look for in the results:
  - Does each metric hold up from dev to held-out cell types (H vs H0a)?
  - Is cell-type r above 0 (H vs H0b)?
- None of the three checks individual molecules. That is the mixture test's job.

#### Results

(100-epoch dev model, all chr21 tiles; numbers from `Oct_23.md` and the `2026-10-18_04` eval log.)

**Table 1: mean over assays.** Model / baseline; **bold** = better of the two.

| Cell type | MSE | Pearson r | Cell-type r |
|---|---|---|---|
| GM12878 | **0.228** / 0.258 | 0.43 / **0.44** | +0.26 |
| K562 | **0.172** / 0.191 | 0.50 / **0.51** | +0.32 |
| HepG2 | **0.232** / 0.244 | **0.57** / 0.54 | +0.41 |
| Panc1 | **0.194** / 0.195 | 0.40 / **0.49** | +0.21 |
| H1 | **0.261** / 0.283 | **0.40** / 0.39 | +0.28 |
| H9 *(near)* | **0.177** / 0.182 | 0.38 / **0.47** | +0.19 |
| HCT116 *(medium)* | 0.177 / **0.163** | 0.47 / **0.56** | +0.26 |
| MCF-7 *(far)* | **0.162** / 0.166 | 0.45 / **0.51** | +0.27 |
| **Mean** | **0.200** / 0.210 | 0.45 / **0.49** | +0.27 |

- **MSE:** the model beats the baseline in 7 of 8 cell types (all but HCT116).
- **Pearson:** the baseline wins in 6 of 8, including all three held-out cell types.
- **Cell-type r:** positive everywhere, and as high in the held-out cell types (+0.19 to +0.27) as in several dev
  ones (+0.21 to +0.41). No clear near-to-far trend (H9 is not better than MCF-7).

**Table 2: MSE per assay, dev vs held-out.**
- Each cell is model / baseline, averaged over the cell types in the group; **bold** = lower.
- "Model better" counts the cell types where the model's MSE is below the baseline's.
- Chart: `../plots/26_oct_30/rest/per_assay_mse.png` (code: `plots_code/26_oct_30/per_assay_bars.py`).
- H1 and H9 have no ATAC, so ATAC averages over 4 dev and 2 held-out cell types.

| Assay | Dev (5): MSE | Dev: model better | Held out (3): MSE | Held out: model better |
|---|---|---|---|---|
| ATAC | **0.082** / 0.083 | 3 / 4 | **0.094** / 0.107 | 2 / 2 |
| H3K4me3 | **0.179** / 0.214 | 5 / 5 | **0.149** / 0.152 | 2 / 3 |
| H3K27ac | **0.211** / 0.254 | 4 / 5 | **0.175** / 0.190 | 3 / 3 |
| CTCF | **0.181** / 0.192 | 5 / 5 | 0.170 / **0.160** | 0 / 3 |
| H3K4me1 | **0.227** / 0.264 | 4 / 5 | **0.153** / 0.161 | 3 / 3 |
| H3K36me3 | 0.265 / **0.233** | 0 / 5 | 0.201 / **0.171** | 1 / 3 |
| H3K27me3 | **0.227** / 0.243 | 4 / 5 | **0.197** / 0.206 | 2 / 3 |
| H3K9me3 | **0.329** / 0.349 | 4 / 5 | 0.210 / **0.191** | 1 / 3 |
| **All assays** | **0.217** / 0.234 | 29 / 39 | 0.172 / **0.170** | 14 / 23 |

- **Dev:**
  - The model is below the baseline on 7 of 8 assays (all but H3K36me3), and wins in 29 of 39 cell-type × assay
    pairs.
  - Overall MSE is 7% lower than the baseline's.
- **Held out:**
  - The model is still below the baseline on 5 of 8 assays: ATAC, H3K4me3, H3K27ac, H3K4me1 and H3K27me3.
  - It wins in 14 of 23 pairs.
  - CTCF, H3K36me3 and H3K9me3 go the other way, and their losses cancel the gains: overall the model and the
    baseline tie (0.172 vs 0.170).
- **Assay by assay:**
  - The active marks carry over to new cell types: ATAC, H3K4me3, H3K27ac and H3K4me1 win in 10 of 11 held-out
    pairs. These are the marks a molecule's accessibility reflects most directly.
  - CTCF flips: the model wins in 5 of 5 dev cell types and in none of the 3 held-out ones. This fits the
    single-molecule examples, where the model calls CTCF wherever the DNA is open: binding sites differ between
    cell types more than accessibility does.
  - H3K36me3 loses everywhere, dev included: gene-body domains are wider than the 5 kb window.
  - H3K9me3 wins on dev and loses on held-out, so it likely needs more than accessibility to carry over.
- In terms of the hypotheses:
  - The held-out advantage is smaller than the dev one, assay by assay, so there is some H0a (dev-specific
    fitting).
  - It is concentrated in CTCF, H3K36me3 and H3K9me3, not spread over all marks.
- Compare model with baseline within a group, not absolute MSE across groups: the held-out cell types' tracks are
  simply easier (lower MSE for both).

**Table 3: Pearson r per assay, dev vs held-out.**
- Each cell is model / baseline, averaged over the cell types in the group; **bold** = higher.
- "Model better" counts the cell types where the model's r is above the baseline's.
- Chart: `../plots/26_oct_30/rest/per_assay_pearson.png`.

| Assay | Dev (5): Pearson r | Dev: model better | Held out (3): Pearson r | Held out: model better |
|---|---|---|---|---|
| ATAC | 0.73 / **0.74** | 2 / 4 | **0.73** / 0.69 | 2 / 2 |
| H3K4me3 | **0.53** / 0.50 | 3 / 5 | **0.61** / 0.59 | 2 / 3 |
| H3K27ac | **0.47** / 0.39 | 5 / 5 | **0.43** / 0.41 | 2 / 3 |
| CTCF | 0.40 / **0.44** | 2 / 5 | 0.35 / **0.49** | 0 / 3 |
| H3K4me1 | **0.48** / 0.43 | 3 / 5 | 0.52 / **0.53** | 2 / 3 |
| H3K36me3 | 0.37 / **0.53** | 0 / 5 | 0.23 / **0.54** | 0 / 3 |
| H3K27me3 | 0.40 / **0.42** | 3 / 5 | 0.30 / **0.39** | 1 / 3 |
| H3K9me3 | 0.37 / **0.41** | 1 / 5 | 0.39 / **0.53** | 0 / 3 |
| **All assays** | 0.46 / **0.47** | 19 / 39 | 0.43 / **0.51** | 9 / 23 |

- **Shape is the hardest metric to win.**
  - The baseline's shape is the part every cell type shares, and that is most of every track.
  - The model can only pull ahead where the cell-specific part is large.
  - Overall the model is level with the baseline on dev (0.46 vs 0.47) and behind on held-out (0.43 vs 0.51).
- **The active marks are level or ahead, held-out cell types included:**
  - ATAC (0.73 vs 0.69), H3K4me3 (0.61 vs 0.59) and H3K27ac (0.43 vs 0.41) are ahead on held-out.
  - H3K4me1 is level (0.52 vs 0.53).
- **The held-out gap comes from four assays:**
  - H3K36me3 (0.23 vs 0.54) is behind on dev too.
  - CTCF (0.35 vs 0.49), H3K9me3 (0.39 vs 0.53) and H3K27me3 (0.30 vs 0.39) are close on dev but fall behind on
    held-out cell types.

**Table 4: cell-type r per assay, dev vs held-out.**
- Each cell is the mean over the cell types in the group, with the range across them in brackets.
- The baseline scores 0 by definition, so there is no baseline column.
- Chart: `../plots/26_oct_30/rest/per_assay_cell_r.png` (bars: mean over cell types; dots: each cell type).

| Assay | Dev (5): cell-type r | Held out (3): cell-type r |
|---|---|---|
| ATAC | +0.44 (+0.28 to +0.59) | +0.46 (+0.44 to +0.48) |
| H3K4me3 | +0.34 (+0.20 to +0.51) | +0.37 (+0.32 to +0.41) |
| H3K27ac | +0.35 (+0.25 to +0.48) | +0.29 (+0.22 to +0.40) |
| CTCF | +0.24 (+0.09 to +0.36) | +0.11 (+0.06 to +0.17) |
| H3K4me1 | +0.36 (+0.18 to +0.57) | +0.34 (+0.27 to +0.39) |
| H3K36me3 | +0.21 (−0.03 to +0.48) | +0.07 (−0.00 to +0.18) |
| H3K27me3 | +0.27 (+0.14 to +0.46) | +0.19 (+0.12 to +0.27) |
| H3K9me3 | +0.20 (+0.03 to +0.54) | +0.16 (+0.07 to +0.34) |
| **All assays** | **+0.30**, positive in 38 / 39 | **+0.24**, positive in 22 / 23 |

- **Positive almost everywhere:**
  - 38 of 39 dev and 22 of 23 held-out cell-type × assay pairs.
  - The only exceptions are H3K36me3, one in each group, at about 0.
  - The fibers carry cell-specific information in new cell types too, which rules out H0b.
- **The active marks carry over fully:** ATAC (+0.44 → +0.46), H3K4me3 (+0.34 → +0.37), H3K4me1 (+0.36 → +0.34)
  and H3K27ac (+0.35 → +0.29) are about as high on held-out cell types as on dev.
- **The drop is in the same assays as in Tables 2 and 3:**
  - CTCF (+0.24 → +0.11) and H3K36me3 (+0.21 → +0.07) lose more than half.
  - H3K27me3 and H3K9me3 drop a little.
  - So the H0a part (dev-specific fitting) is confined to these marks.
- **Across Tables 2–4:**
  - On the marks accessibility reflects most directly, the model generalizes to new cell types: lower MSE than the
    baseline and the same cell-type r as on dev.
  - CTCF and H3K36me3 do not carry over, on any of the three metrics.
- Caveat: these numbers use the earlier fiber averaging and baseline (both since updated); a rerun is pending.
  The picture is not expected to change.

#### Single-molecule examples (dev and held-out)

- **How loci were found:**
  - Data: chr21 test tiles. Each 1 kb bin is scored as the cell's ENCODE signal minus the cross-cell average
    (asinh FC).
  - Categories:
    - **gain**: the cell is ≥ 0.5 above the average;
    - **loss**: the cell is ≥ 0.5 below the average;
    - **agree**: within 0.25 of the average, at a peak ≥ 1.5.
  - The top candidates per cell type were run through the model and ranked by:
    - how much of the gap to the average the model recovers;
    - the model's r over the 5 kb tile.
  - Plots: per cell type and group, the best few plus random picks. 224 figures in total, all on fir in
    `results/26-10-28_T21-22-02_single_molecule_dev5/` (latest version; ranking from `..._T06-06-09_...`).
  - Code: `plots_code/26_oct_30/single_molecule_dev5.py`.
- **Each figure:**
  - Two panels: the flagged assay and its partner (ATAC↔CTCF, H3K4me3↔H3K27ac, H3K4me1↔H3K36me3,
    H3K27me3↔H3K9me3; for H1/H9, which have no ATAC, CTCF is paired with H3K27ac).
  - Top: the ENCODE track, the model's average over fibers, and the baseline, with the tile MSE of each (scored
    positions, ≥ 5 fibers).
  - The 5 kb window is centred on the flagged bin.
  - Then a heatmap of every fiber's prediction, sorted by the flagged bin.
  - Then five fibers across that ranking (preferring fibers that cover the window on both sides of the bin), with
    their inputs (m6A, MSP, nucleosomes, FIRE). Each fiber's prediction
    is drawn against the baseline (dashed): a bulk baseline gives every fiber that same curve, while the model gives
    each fiber its own.
- **Over all 799 ranked candidates** (not just the plotted ones):

  | | Model r > baseline r | Gap recovered (median) |
  |---|---|---|
  | Sharp gain (dev / held-out) | 84% / 81% | 29% / 28% |
  | Broad gain (dev / held-out) | 71% / 72% | 14% / 14% |
  | Loss, sharp and broad | 45–70% | 81–95% |
  | Agree | 13–23% | — |

  - At gain loci the model beats the baseline on shape in most cases, held-out cell types included, but recovers
    only part of the height (more for sharp marks).
  - At loss loci it mostly removes the peak the baseline wrongly predicts.
  - Where the cell matches the average, the baseline is already right and is hard to beat (r 0.88 vs 0.92 for
    sharp marks), but the model is close.
- **Figures to show** (`../plots/26_oct_30/sm/`; tile MSE is model vs baseline, flagged assay first, then the
  partner):
  - **Gain (the cell has signal the average lacks):**
    - `sm_test_mcf-7_atac_gain.png` (MCF-7, held out; ATAC 0.14 vs 1.07):
      - The model finds an ATAC peak that the average does not have.
      - The heatmap shows which fibers are open; the closed bottom fiber predicts ~0.
      - CTCF is over-called at the same site (0.43 vs 0.26): CTCF follows accessibility.
    - `sm_test_mcf-7_h3k4me3_gain.png` (MCF-7; H3K4me3 3.69 vs 9.56, H3K27ac 0.61 vs 2.78):
      - A wide active domain the average misses. The model has the extent but about half the height.
      - The fibers split into marked and unmarked; the bottom fiber is ~0.
    - `sm_test_hct116_h3k4me3_gain.png` (HCT116, held out; 0.15 vs 1.30, H3K27ac 1.42 vs 3.30):
      - The promoter is called on most fibers.
      - The model over-calls the downstream half; H3K27ac is under-called but still closer than the baseline.
    - `sm_dev_hepg2_h3k4me3_gain.png` (HepG2, dev; 0.63 vs 3.44, H3K27ac 1.42 vs 2.65):
      - The model follows the promoter domain, below its full height (~2.5 vs ~3.5).
      - The bottom fiber is almost fully nucleosome-covered and predicts nothing: a silent copy.
    - `sm_dev_k562_h3k4me1_gain.png` (K562, dev; broad mark, 0.67 vs 2.36):
      - The model rises over the domain, below its full height.
      - Smooth gradient across fibers, down to a fully nucleosome-covered fiber at ~0.
  - **Loss (the average has signal the cell lacks):**
    - `sm_dev_gm12878_atac_loss.png` (GM12878, dev; ATAC 0.02 vs 0.82):
      - The model removes the ATAC peak the average puts there.
      - A few fibers with a FIRE element at the site still light up: rare open copies that average out in the bulk.
      - It also drops the CTCF peak, which GM12878 partly has (0.25 vs 0.79).
    - `sm_test_h9_h3k4me3_loss.png` (H9, held out; 0.11 vs 2.65, H3K27ac 0.23 vs 1.15):
      - A promoter active in the other cell types but weak in H9; the model stays low on both marks.
      - Only the top few fibers (with FIRE elements) carry signal.
    - `sm_test_h9_h3k4me1_loss.png` (H9; broad, 0.09 vs 2.27):
      - The model stays at H9's low level instead of the average's ~2.
      - H3K36me3 is over-called (0.32 vs 0.14).
  - **Agree (the cell matches the average):**
    - `sm_test_mcf-7_atac_agree.png` (MCF-7; ATAC 0.145 vs 0.057, CTCF 0.28 vs 0.53):
      - Peak shape and height match.
      - The baseline is slightly closer on ATAC; the model is closer on CTCF.
      - The closed bottom fiber predicts ~0 where the bulk peak is.
    - `sm_dev_h1_h3k4me3_agree.png` (H1, dev; 0.22 vs 0.15, H3K27ac 0.99 vs 1.59):
      - The bulk is matched, yet single fibers differ: the bottom fiber is marked only in patches.
    - `sm_test_h9_h3k4me1_agree.png` (H9; broad, 0.11 vs 0.11):
      - The bulk is matched, and the fibers again vary.
      - A fiber with a long accessible stretch (FIRE) at the bin predicts a dip there: no nucleosome, no histone
        mark.
      - A fully nucleosome-covered fiber predicts ~0.
  - **Failure:** `sm_test_hct116_h3k27me3_gain.png` (HCT116, a random pick; 0.80 vs 0.72):
    - The H3K27me3 domain is missed.
    - The prediction is flat and uniform across fibers. H3K27me3 is one of the marks that carries over less
      (Table 4).
  - **Pattern in the "agree" figures:**
    - The bulk is no better than the baseline (expected: the baseline is already right there).
    - The fibers still differ, which is information the baseline cannot give.

#### Conclusion

- **H0b (average from DNA) is ruled out.**
  - Cell-type r is positive in 60 of 62 cell-type × assay pairs: +0.30 on dev and +0.24 on held-out cell types.
  - The DNA is the same for every cell type, so this signal comes from the fibers.
- **H holds for the marks accessibility reflects most directly:** ATAC, H3K4me3, H3K27ac and H3K4me1. On held-out
  cell types the model:
  - beats the baseline on MSE in 10 of 11 cases;
  - is level with or ahead of it on Pearson;
  - keeps its dev-level cell-type r.
- **Some H0a, confined to a few marks.**
  - CTCF and H3K36me3 lose ground on held-out cell types on all three metrics; H3K27me3 and H3K9me3 lose a
    little.
  - Overall, held-out MSE ties the baseline (0.172 vs 0.170) and Pearson trails it (0.43 vs 0.51). Most of that is
    these marks.
  - Likely causes:
    - H3K36me3 domains are wider than the 5 kb window;
    - CTCF is called wherever the DNA is open, but binding differs between cell types more than accessibility
      does.
- **No near-to-far trend** across the three held-out cell types: H9 is nearest but lowest on cell-type r. It also
  has no ATAC, so this is not a clean comparison.
- **What the single-molecule figures add:**
  - At one locus, fibers differ: open vs closed, marked vs unmarked. Nucleosome-covered fibers get ~0.
  - This holds even where the bulk matches the baseline, and a bulk track cannot show it.
  - Where the cell gains signal, the model finds where but under-calls how high.
- **Limits:**
  - Bulk tracks only test the average over molecules. Whether each molecule's call is right is what the mixture
    test (next) checks.
  - One training run, three held-out cell types, and the earlier fiber averaging and baseline (rerun pending).
  - No ceiling for r yet: the replicate reference is in Future directions.

### Mixture test (T3): are single molecules resolved?

(Details: `Oct_23_T3.md`; design: `TODO.md`, T3.)

#### High-level idea

- The dev-test split showed that the fibers carry cell-specific information, but only at the level of the bulk (the
  average over molecules). It cannot tell whether each molecule gets the right value.
- The mixture test asks exactly that: a model trained only on a mixture of two cell types should still learn the
  transformation from Fiber-seq to histone modifications.
  - If it does, it can recover each cell type's own bulk signal from that cell type's fibers alone.
  - Every fiber is imputed independently, so this works only if individual molecules are read correctly. That makes
    it a test of correctness at single-molecule resolution.

#### Question

- The model only ever sees bulk averages as targets.
- **Hypothesis (H):** the model tells the fibers apart and gives each one a value that matches its own cell type's
  bulk.
- **Null (H0):** the model fits the bulk by giving every molecule roughly the same bulk-like value. Both cell types
  then get the mixture track.
- The mixture track (the average of the two cell types' bulk) plays the role the cross-cell-average baseline played
  in the dev-test split.

#### Setup

- **Mixture model:** trained only on mixtures of GM12878 and HepG2 (B cell vs liver, very different chromatin).
  - Each training sample pools 100 fibers from both cell types at the same locus, with GM12878's share drawn
    between 20% and 80%.
  - The target is the matching mixture of the two tracks (mixed in fold change, weighted by the fibers present).
  - Neither cell type's own track is ever used, in training or validation.
- **Test: imputation per fiber.**
  - Give the model one cell type's fibers alone and average their predictions.
  - Compare the result with that cell type's own track and with the mixture (H0).
  - A model that reads each molecule can split the two cell types; under H0 both get the mixture.
- **Upper bound:** the same model and training regime, but each step sees one cell type's fibers with that cell
  type's true track.
- Same architecture as the dev-test model (mid 8-assay UNet), trained on chr20 for 25 epochs; tested on all chr21
  tiles (~6,800 per cell type).

#### Metrics: bulk level (difference between the two cell types)

- **Differential r** [higher is better], range [−1, 1], **H0: 0**:
  - The correlation between the predicted difference (GM12878 prediction − HepG2 prediction) and the true
    difference between the two tracks, per 5 kb window.
  - Like the dev-test metrics, but on the difference between the cell types, which is exactly what the mixture
    hides.
- **Sign accuracy** [higher is better], range [0, 1], **H0: 0.5**:
  - Differential bins are 1 kb bins where the two tracks differ by ≥ 0.5 in asinh fold change.
  - Within those bins, how often does the predicted difference have the right sign?
- **Predictions:**
  - Under H, both are clearly above H0 and close to the upper bound.
  - Under H0 the predicted difference is flat: r 0, sign accuracy 0.5.

#### Metrics: molecule level

- **Fiber-origin AUC** [higher is better], range [0, 1], **H0: 0.5**:
  - At each differential bin, pick one GM12878 fiber and one HepG2 fiber at random.
  - How often does the fiber from the cell type with the higher track get the higher prediction (its mean over the
    bin)?
  - The most direct single-molecule readout: it is about individual fibers, not averages.
- **Library control** (a control study), reported as |AUC − 0.5|; should stay at sampling noise (≈ 0.05):
  - The same AUC between two Fiber-seq libraries of one cell type (HepG2 100U vs 200U), which share one true
    track.
  - The same cell should not be told apart by library. If it were, the model would be separating fibers by
    library features (e.g. m6A density), not biology.

#### Summary of metrics

| Metric | Level | What it asks | H0 (bulk on every molecule) | H (reads each molecule) | Library offset |
|---|---|---|---|---|---|
| Differential r [higher is better] | bulk | Does the predicted difference between the cell types follow the true one? | 0 | clearly > 0, near the upper bound | ≈ 0 |
| Sign accuracy [higher is better] | bulk | Where the cell types differ, is the direction right? | 0.5 | clearly > 0.5 | ≈ 0.5 (right in one direction only) |
| Fiber-origin AUC [higher is better] | molecule | Does a single fiber's prediction follow its own cell type? | 0.5 | > 0.5, flipping with the direction of the difference | one cell type's fibers score higher everywhere |
| Library control, \|AUC − 0.5\| [lower is better] | molecule | Are two libraries of one cell type told apart? | ≈ 0.05 (noise) | ≈ 0.05 (noise) | clearly > 0.05 |

- The upper bound (trained on each cell type's own track) shows how high each metric can get with this
  architecture. The mixture model gets no cell-specific target at all.
- The two bulk metrics rule out H0: under it, both cell types get the mixture, so the predicted difference is flat.
- The two molecule metrics check that the separation is per fiber and follows the biology.
  - "Library offset" is a model that reacts to library features (e.g. overall m6A density) and shifts one sample's
    fibers up everywhere.
  - That would show up in the library control, and the fiber-origin AUC would not follow the direction of the
    difference.
- What this test cannot rule out:
  - A model that recognises which cell type a fiber comes from and returns that cell type's bulk would pass all four
    metrics.
  - The mixture test on unseen cell types (T3b, Future directions) checks for that.
- Compared with the dev-test split:
  - The mixture track plays the role of the cross-cell-average baseline.
  - The difference between the two cell types plays the role of cell-type r.
  - The fiber-origin AUC is new: the first metric scored on single molecules.

#### Results

**Table 5: mean over the 8 assays.** Mixture model / upper bound; **bold** = better of the two.

| Readout | H0 | Mixture model / upper bound |
|---|---|---|
| Differential r (per 5 kb window) | 0 | **0.49** / 0.47 |
| Sign accuracy at differential bins | 0.50 | **0.78** / 0.74 |
| Fiber-origin AUC | 0.50 | **0.64** / 0.60 |
| Library control, \|AUC − 0.5\| | ≈ 0.05 | 0.044 / 0.049 |

**Table 6: per assay.**
- Each cell is mixture model / upper bound; **bold** = better of the two (for the library control, lower).
- H0: differential r 0; sign accuracy and AUC 0.5; library control ≈ 0.05.
- Charts (`../plots/26_oct_30/t3/`, code: `plots_code/26_oct_30/t3_bars.py`): `t3_per_assay_bulk.png` (differential r
  and sign accuracy) and `t3_per_assay_fiber_auc.png` (fiber-origin AUC and library control).

| Assay | Differential r | Sign accuracy | Fiber-origin AUC | Library control |
|---|---|---|---|---|
| ATAC | 0.55 / **0.57** | **0.96** / 0.95 | **0.72** / 0.70 | **0.042** / 0.048 |
| H3K4me3 | 0.38 / **0.46** | **0.87** / 0.86 | **0.67** / 0.65 | **0.045** / 0.049 |
| H3K27ac | 0.63 / **0.67** | 0.85 / **0.86** | **0.67** / 0.66 | **0.043** / 0.047 |
| CTCF | 0.45 / **0.50** | 0.80 / **0.81** | **0.64** / 0.63 | **0.041** / 0.043 |
| H3K4me1 | **0.77** / 0.72 | 0.89 / **0.90** | **0.68** / 0.65 | **0.043** / 0.047 |
| H3K36me3 | **0.28** / 0.10 | 0.55 / 0.55 | **0.54** / 0.52 | **0.043** / 0.051 |
| H3K27me3 | **0.52** / 0.36 | **0.77** / 0.67 | **0.61** / 0.55 | **0.047** / 0.062 |
| H3K9me3 | 0.32 / **0.39** | **0.59** / 0.35 | **0.54** / 0.47 | **0.046** / 0.048 |
| **All** | **0.49** / 0.47 | **0.78** / 0.74 | **0.64** / 0.60 | **0.044** / 0.049 |

- **Single molecules are resolved without single-molecule supervision.**
  - The mixture model is above H0 on every assay and every readout.
  - The predicted difference follows the true one (differential r 0.49).
  - It has the right sign at 78% of differential bins: 80–96% on ATAC, H3K4me3, H3K27ac, CTCF and H3K4me1.
  - One fiber's prediction tells its cell of origin (AUC 0.64; 0.64–0.72 on those same five assays).
- **Mixture supervision costs little:** the mixture model is close to the upper bound, which is trained on each
  cell type's own track.
  - The upper bound has a slightly higher differential r on the four narrow-peak assays.
  - The mixture model is ahead on H3K4me1, H3K36me3 and H3K27me3.
  - With one training run each, read this as "matches", not "beats".
- **Same ranking of assays as in the dev-test split:**
  - ATAC, H3K4me3, H3K27ac, CTCF and H3K4me1 separate well.
  - H3K36me3 and H3K9me3 are weakest: sign accuracy 0.55–0.59 and AUC 0.54.
  - They are still above H0 for the mixture model; the upper bound falls below H0 on H3K9me3 (sign accuracy 0.35,
    AUC 0.47).
- **It is biology, not the library:**
  - The separation flips with the direction of the difference. For ATAC, the chance a GM12878 fiber scores above a
    HepG2 fiber is 0.65 where GM12878's track is higher and 0.23 where HepG2's is.
  - Two libraries of one cell type are not told apart: |AUC − 0.5| is 0.041–0.047 on every assay, at sampling
    noise. That compares with 0.175 for GM12878 vs HepG2.
- **Other charts** (`../plots/26_oct_30/t3/`, same script):
  - `t3_library_bars.png`: fiber-origin |AUC − 0.5| per assay, GM12878 vs HepG2 (0.10–0.23) next to the library
    control (0.04–0.05 on every assay).
- **Figures** (copied from `../plots/26_oct_23_t3/` to `../plots/26_oct_30/t3/` with a `t3_` prefix):
  - `t3_mix_paired_locus_0.png`: a GM12878-specific locus. Each cell type's prediction from its own fibers, the
    mixture (what H0 gives both), and every fiber's prediction: most GM12878 fibers light up at the H3K27ac /
    H3K4me3 peak, a minority stay dark, HepG2 fibers stay dark.
  - `t3_mix_differential_scatter.png`: per chr21 window, predicted vs true difference (H0: all on y = 0).
    - One dot per 5 kb tile, placed at its 5 kb mean difference.
    - Coloured tiles contain a differential 1 kb bin. A narrow peak averaged over 5 kb shrinks to about a fifth, so
      many coloured dots of sharp marks sit near the origin (43% for ATAC, 58% for CTCF).
  - `t3_mix_differential_scatter_bin.png` (code: `plots_code/26_oct_30/t3_bin_scatter.py`): the same at bin level.
    - One dot per differential 1 kb bin (the bins sign accuracy is scored on), with y = x dashed.
    - Why some dots still sit inside ±0.5:
      - Bins are flagged on asinh of the bin's mean fold change (the loci rule), but plotted as the mean of the
        per-bp asinh differences, which shrinks narrow peaks.
      - Checked against the bigWigs: both values match exactly and the bins align. The gap is the definition.
      - It affects 29% of CTCF bins and 2% of H3K27ac bins; the sign disagrees in only 4 of 1,090 bins.
    - Bin-level r is 0.83–0.89 for ATAC, H3K27ac and H3K4me1; 0.79 H3K4me3, 0.69 CTCF.
    - H3K27me3 (0.64) and H3K36me3 (0.62) mostly have the right sign but are flattened towards 0.
    - H3K9me3 (0.16) is flat on 0 in both directions: no information.
    - The dots sit below the diagonal in magnitude: the direction is right but the size is under-predicted, about
      half.
  - `t3_mix_fiber_auc.png` and `t3_mix_library_fiber_auc.png`: fiber-origin AUC per differential bin, pair vs library
    control.
- **Single-molecule examples** (`../plots/26_oct_30/t3/t3_fib_*.png`, code:
  `plots_code/26_oct_30/t3_single_molecule_fibers.py`).
  - What each figure shows (one assay, window centred on a differential 1 kb bin), top to bottom:
    - Bulk: both ENCODE tracks, the 50:50 mixture (what H0 gives every molecule), and each cell type's prediction
      from its own fibers.
    - Every fiber's prediction, GM12878 above the cyan line and HepG2 below, sorted by the bin within each cell
      type.
    - The same fibers in the same order, drawn from their Fiber-seq inputs: nucleosome, MSP (accessible), FIRE.
      This shows which input pattern goes with a high prediction.
    - Three example fibers per cell type (top, middle and bottom of the ranking), with all input channels and
      their prediction against the mixture.
  - How they were picked:
    - "Best" is the top bin by fiber-origin AUC for each assay and direction.
    - "Random" is drawn from the 7,838 eligible bins (≥ 10 fibers per cell type).
    - All 24 are on fir in `results/26-10-29_T21-20-16_t3_single_molecule_fibers/`.
    - The AUC in each title is recomputed from the fibers shown, since the centred window can hold a slightly
      different set of fibers; it is within 0.06 of the eval's in all but one.
  - `t3_fib_best_atac_hepg2_high.png` (AUC 0.99):
    - HepG2 fibers carry FIRE elements across the bin and are predicted open.
    - GM12878 fibers are nucleosome-covered there and stay closed.
    - A HepG2 fiber with FIRE only next to the bin stays low inside it.
  - `t3_fib_best_h3k27ac_gm_high.png` (AUC 0.93):
    - The GM12878 fibers that light up are the ones with a FIRE block at the bin.
    - About half the GM12878 fibers lack it and look like HepG2 fibers, in inputs and prediction: per-molecule
      variation.
  - `t3_fib_best_h3k4me1_hepg2_high.png` (AUC 0.98): a broad mark that works. Most HepG2 fibers are marked across
    the domain; GM12878 fibers are flat near 0.
  - `t3_fib_best_h3k27me3_hepg2_high.png` (AUC 0.91):
    - A repressive mark that works: HepG2 fibers are marked, GM12878 fibers mostly not.
    - The inputs look alike between the cell types, so the model is reading more than open vs closed.
    - The predicted average undershoots the track (~1 vs ~2).
  - `t3_fib_best_atac_gm_high.png` (AUC 0.97): the ATAC example in the other direction.
    - GM12878 fibers with a FIRE block across the bin are predicted open; HepG2 fibers are nucleosome-covered there
      and stay low.
    - The lowest-ranked GM12878 fiber has no FIRE at the bin and stays low too.
  - `t3_fib_best_h3k36me3_hepg2_high.png` (AUC 0.93): a broad gene-body mark.
    - HepG2 fibers are predicted high across the whole 5 kb window, GM12878 fibers low, matching the tracks.
    - There is no FIRE block to explain it: the inputs look alike between the cell types, as with H3K27me3.
  - `t3_fib_random_h3k27ac_gm_high.png` (random pick, AUC 0.79): a typical case. Same pattern as the best H3K27ac
    example, noisier.
  - `t3_fib_random_h3k4me3_hepg2_high.png` (random pick, AUC 0.66): a weak case.
    - Both cell types have a FIRE block in the bin (a shared open promoter), and the fibers that carry it are
      predicted high in both.
    - The tracks differ mostly in the left half of the bin, which the model largely misses, so the fibers separate
      only partly.
  - `t3_fib_random_h3k9me3_fail.png` (random pick, AUC 0.29): a failure.
    - Both cell types get the same prediction, in vertical stripes: it follows the sequence and each fiber's
      overall openness, not the cell type.
    - This matches H3K9me3 being one of the two weakest assays in Table 6.
- Caveats: one training run per model (second seed pending); the AUC checks separation by cell of origin, a proxy
  for "this molecule carries the mark"; GM12878 and HepG2 come from different Fiber-seq experiments (a same-series
  pair, e.g. K562 + HepG2, would remove that concern).

#### Conclusion

- **H0 (bulk on every molecule) is ruled out.** Trained only on mixtures, the model still gives each cell type its
  own signal:
  - The predicted difference follows the true one: differential r 0.49 per window, and 0.83–0.89 at the
    differential 1 kb bins of ATAC, H3K27ac and H3K4me1.
  - The sign is right at 78% of differential bins.
  - A single fiber's prediction tells its cell of origin: AUC 0.64, against 0.5 under H0.
  - Every assay is above H0 on every readout.
- **The separation is per molecule and biological, not a library offset:**
  - It follows the direction of the difference. For ATAC, GM12878 fibers score higher where GM12878's track is
    higher (0.65) and lower where HepG2's is (0.23).
  - Two libraries of one cell type are not told apart: |AUC − 0.5| is 0.044, at sampling noise.
- **Mixture supervision costs little:** the model matches the upper bound trained on each cell type's own track
  (0.49 vs 0.47 differential r, 0.64 vs 0.60 AUC). Learning from mixtures is enough.
- **Same assay ranking as the dev-test split:**
  - ATAC, H3K4me3, H3K27ac, CTCF and H3K4me1 separate well.
  - H3K27me3 gets the direction but little of the size.
  - H3K36me3 and H3K9me3 are close to H0. At bin level, H3K9me3 gives no difference at all (r 0.16).
  - Across both tests, the marks that accessibility reflects are the ones resolved; gene-body and heterochromatin
    marks are not.
- **Direction right, size about half:**
  - At differential bins the predicted difference is roughly half the true one.
  - The single-molecule figures show the same: domains are found but under-called in height.
- **What the single-molecule figures add:**
  - At a differential bin, most fibers of the marked cell type light up and the other cell type's stay dark.
  - Within the marked cell type, a share of fibers stay dark too (about half at the best H3K27ac locus). That is
    the per-molecule variation a bulk track averages away.
- **Limits:**
  - One training run per model.
  - The two cell types come from different Fiber-seq experiments.
  - A model that recognises which cell type a fiber comes from and returns its bulk would also pass. The mixture
    test on unseen cell types (T3b) checks for that, using existing checkpoints.

## Future directions

(Full plan in `reports/TODO.md`.)

### Immediate: firm up these results

- **Rerun the dev-test numbers** with the current fiber averaging and baseline.
- **Reference values for r:** ENCODE replicate 1 vs 2 as the ceiling; H1's track as a prediction for H9.
- **Mixture test:** a second training seed, a pair from the same Fiber-seq series (K562 + HepG2), and the test on
  unseen cell types (T3b). T3b rules out a model that only recognises the cell type; it is eval only.

### Next: improve the model

- **Broad marks (H3K36me3, H3K9me3):** use the whole ~18 kb read as context instead of the 5 kb window.
- **Signal height** (about half of the true size in both tests): outputs in physical units, i.e. per-nucleosome
  marks with a per-assay ChIP model.
- **CTCF following accessibility:** supervision that separates molecules more (more cell types, mixtures), and DNA
  as a modulator rather than an extra input.
- **Inputs:** continuous FIRE and m6A / CpG scores instead of thresholded calls.

### Later: more single-molecule tests

- **Ground truth within one cell type:** held-out CpG methylation on the same molecule, X inactivation, imprinted
  loci.
- **References:** an "accessible = active" rule and "bulk on every molecule" scored the same way.

## Previous results

Just show a few previous results.

## Progress

### FIRE-MSP contribution

- From `Oct_16.md`: two models identical except for the fire_msp input (old 4-assay setup, trained on GM12878 +
  K562 + HepG2, scored on chr21 cCREs). Dropping fire_msp changes the loss very little, slightly in its favour
  (average MSE −0.014 GM12878, −0.017 K562, +0.013 HepG2).

  Average MSE over the 4 assays (ATAC, H3K4me3, H3K27ac, H3K27me3), chr21 cCREs; **bold** = lower.

  | Model | GM12878 | K562 | HepG2 |
  |---|---|---|---|
  | With fire_msp | 0.344 | 0.587 | **0.412** |
  | Without fire_msp | **0.330** | **0.570** | 0.425 |
  | Δ (without − with) | −0.014 | −0.017 | +0.013 |

- Single-molecule figures (H3K27ac and H3K4me3; ENCODE track, with fire_msp in red, without in blue; 7 fibers with
  their inputs and both models' per-fiber predictions):
  - `../plots/26_oct_30/rest/fire_msp_GM12878_03_chr21_34580155-34585155_h3k27ac-h3k4me3.png`
  - `../plots/26_oct_30/rest/fire_msp_GM12878_07_chr21_45108980-45113980_h3k27ac-h3k4me3.png`
  - `../plots/26_oct_30/rest/fire_msp_K562_02_chr21_33400612-33405612_h3k27ac-h3k4me3.png`
- Takeaway:
  - Both models agree on the FIRE-bearing fibers: they put the marks around the called FIRE elements, even the
    model that never sees fire_msp (GM 03). Fully closed fibers get nothing from either.
  - Where they differ is open fibers with no FIRE call. The model with fire_msp treats the FIRE call as a switch:
    no call, (almost) no signal. The model without it reads accessibility directly and marks open stretches the
    FIRE caller missed. Clearest case: GM 07 fiber #30, a long accessible patch under the bulk peak with no FIRE
    call, gets ~5–6 without fire_msp and ~0 with it. The same happens next to called elements (#59, #91).
  - FIRE calls are a strict, thresholded summary of the m6A / accessible-patch channels, so the model can get the
    same information from the other inputs, and without the shortcut it learns to.
  - The bulk cannot tell the two apart: "more signal on fewer fibers" and "less on more" fit it about equally,
    which matches the tiny loss difference. Which one is right per molecule needs a test with known answers (T3
    mixture), not the bulk loss.
