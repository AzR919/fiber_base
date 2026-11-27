# 4th Dec

Section 1 is a new biology test, poised vs Polycomb chromatin on single molecules (the professor's test, T5g).
Section 2 compares the bulk accuracy of every model so far: DEV4NF, the R1a mixture model, the Pair model and the
baseline ladder. Section 3 adapts the X-inactivation test to block phasing for the sealed cells and validates it on
GM12878. Models and fiber terms (MSP, large MSP, protected, fiber methylation) are as in `Nov_27.md`
("Models", "Fiber terms"). Everything is scored on chr21, the test chromosome. Plan and open items: `TODO.md` §3.

## 1. Poised vs Polycomb on single molecules (T5g, 27 Nov)

> **Question:** bulk ChIP calls some regions poised (H3K4me3 and H3K27me3 together, typical of developmental genes
> in stem cells) and others Polycomb (H3K27me3 only).
> - What do single fibers in each class look like?
> - Is bulk "poised" a mix of open and closed molecules?
> - Does the model, trained on bulk tracks only, separate the classes?
> - Do its per-fiber H3K4me3 / H3K27me3 predictions follow each fiber's own state?

**Setup** (defined before looking; `TODO_extended.md` T5g):
- **Tiles:** 1 kb tiles, minus the blacklist and N runs.
  - Per cell, take the mean fold change (FC) of its ENCODE H3K4me3 and H3K27me3 tracks over the tile.
  - A mark is **marked** at FC ≥ 2 and **unmarked** at FC < 1. The thresholds are absolute, so the fractions compare
    across cells.
  - Classes: **poised** = both marks marked; **Polycomb** = H3K27me3 marked, H3K4me3 unmarked; **active** = H3K4me3
    marked, H3K27me3 unmarked; **neither** = both unmarked. Tiles in between are left out.
  - Robustness check: the same with FC ≥ 3 as "marked".
- **Targets:** chr21, up to 250 tiles per cell and class, the union over the 4 dev cells: 2,755 tiles
  (`data_prep/build_targets.py --sets poised`).
- **Fiber states** over the tile:
  - methylated: CpG methylation > 0.5 from ≥ 3 covered CpGs; unmethylated < 0.5;
  - open: at least one tile base inside a large MSP (≥ 151 consecutive MSP bases);
  - four groups: unmethylated + open (UO), unmethylated + closed (UC), methylated + open (MO), methylated + closed
    (MC).
- **Models**, each on its own pair's fibers:
  - DEV4NF (2 seeds), R1a and Pair (3 seeds each);
  - the ladder's MLP (3 seeds) and rules (1 seed) as references. The rules contain "closed, unmethylated CpG island
    → H3K27me3".
- **Clear** means every seed has the same sign and p < 0.01 over tiles (Wilcoxon).
- **Code and outputs:** script `plots_code/26_dec_04/poised.py` (job `poised.sh`), outputs in `plots/26_dec_04/poised/`.
  Molecule runs: `molecule_predict.py --perturb meth unmeth meth_distal`, jobs `2026-11-27_02`–`25`.

### D1. How much of the genome is poised or Polycomb? (bulk only)

Share of autosomal 1 kb tiles in each class, and the share of each class at CpG islands:

| Cell | Poised | Polycomb | Active | Poised at CGIs | Polycomb at CGIs | Active at CGIs |
|---|---|---|---|---|---|---|
| H1 | **0.77%** | 1.9% | 1.5% | 66% | 2% | 37% |
| GM12878 | 0.11% | 3.8% | 1.9% | 86% | 5% | 33% |
| HepG2 | 0.14% | 4.2% | 1.5% | 74% | 5% | 32% |
| K562 | 0.05% | 4.1% | 1.7% | 86% | 5% | 45% |

- **As expected:** the stem cells (H1) have 5–15× more poised chromatin than the differentiated lines, and about half
  as much Polycomb-only chromatin.
- **Poised regions are CpG islands; Polycomb-only regions mostly are not.** At this threshold, Polycomb-only means
  broad H3K27me3 domains.
- **Power:** chr21 has 230 poised tiles in H1 but only 17–34 in each other line. With FC ≥ 3, only 2–8 poised tiles
  are left outside H1, so the poised results below rest mainly on H1.

### D2. What do the fibers look like? (data, no model)

![Genome fractions and fiber groups per class](../plots/26_dec_04/poised/poised_D.png)

| Cell | Class | Tiles | Fibers methylated | Fibers open | Tiles with 20–80% open fibers | SD of per-tile open share |
|---|---|---|---|---|---|---|
| H1 | active | 317 | 19% | 56% | 59% | 0.28 |
| H1 | **poised** | 230 | 9% | 53% | **87%** | 0.19 |
| H1 | Polycomb | 287 | **59%** | 23% | 54% | 0.11 |
| GM12878 | active | 430 | 8% | 47% | 64% | 0.27 |
| GM12878 | **poised** | 19 | 1% | 39% | **95%** | 0.16 |
| GM12878 | Polycomb | 364 | 5% | 16% | 25% | 0.07 |
| HepG2 | active | 334 | 14% | 61% | 54% | 0.28 |
| HepG2 | **poised** | 34 | 4% | 54% | **91%** | 0.17 |
| HepG2 | Polycomb | 444 | 13% | 32% | 81% | 0.13 |
| K562 | active | 380 | 10% | 63% | 65% | 0.24 |
| K562 | **poised** | 17 | 2% | 42% | **100%** | 0.11 |
| K562 | Polycomb | 677 | 1% | 40% | 97% | 0.11 |

- **Poised tiles are unmethylated** (1–9% of fibers) and **less often open than active tiles** (39–54% vs 47–63%;
  about level in H1).
- **Each poised tile is about half open.**
  - Nearly every poised tile has between 20% and 80% open fibers, and the per-tile open share varies little between
    tiles (SD 0.11–0.19).
  - Active tiles range from mostly closed to mostly open (SD 0.24–0.28).
  - So bulk "poised" is consistently a mix of open and closed molecules at every site. This is what you would see if
    some molecules carry an active promoter and others a closed, Polycomb-bound one. It doesn't rule out both marks
    on one molecule (Fiber-seq can't see the marks).
- **Polycomb tiles are closed** in GM12878 (16% open) and HepG2 (32%), and unmethylated in every line except H1.
  - In H1 59% of Polycomb-tile fibers are methylated: H1's genome is highly methylated outside CpG islands, and its
    Polycomb-only tiles are almost never at CGIs (2%).
  - K562 fibers are more open everywhere (even "neither" tiles 26%), so open shares there sit higher across the
    board.

### M1. Do tile-average predictions separate the classes?

![M1 AUCs per model, with the bulk track as reference](../plots/26_dec_04/poised/poised_M1.png)

Every prediction is averaged over a tile's fibers (tiles with ≥ 5 fibers). AUC means the chance that a tile of the
first class scores higher than one of the second. "Bulk" is the same AUC from the true ENCODE track on the same tiles,
the reference. Mean over seeds:

| Cell | Readout | Bulk | Rules | MLP | R1a | Pair | DEV4NF |
|---|---|---|---|---|---|---|---|
| GM12878 | H3K4me3 poised > Polycomb | 1.00 | 0.95 | 0.99 | 0.97 | 0.98 | 0.95 |
| | H3K4me3 active > poised | 0.62 | 0.36 | 0.44 | 0.56 | 0.55 | 0.59 |
| | H3K27me3 Polycomb > active | 1.00 | **0.37** | 0.64 | 0.83 | 0.86 | 0.67 |
| | H3K27me3 poised > Polycomb | 0.60 | 0.89 | 0.86 | 0.70 | 0.66 | 0.74 |
| HepG2 | H3K4me3 poised > Polycomb | 1.00 | 0.87 | 0.98 | 0.98 | 0.98 | 0.98 |
| | H3K4me3 active > poised | 0.64 | 0.42 | 0.49 | 0.70 | 0.66 | 0.64 |
| | H3K27me3 Polycomb > active | 1.00 | **0.33** | 0.80 | 0.98 | 0.98 | 0.87 |
| | H3K27me3 poised > Polycomb | 0.54 | 0.82 | 0.67 | 0.51 | 0.40 | 0.62 |
| K562 | H3K4me3 poised > Polycomb | 1.00 | 0.97 | 0.98 | 0.99 | 0.97 | 0.93 |
| | H3K4me3 active > poised | 0.71 | 0.43 | 0.41 | 0.77 | 0.85 | 0.90 |
| | H3K27me3 Polycomb > active | 1.00 | **0.28** | 0.56 | 0.97 | 0.92 | 0.89 |
| | H3K27me3 poised > Polycomb | 0.52 | 0.95 | 0.96 | 0.77 | 0.66 | 0.60 |
| H1 | H3K4me3 poised > Polycomb | 1.00 | 0.90 | 0.99 | 0.99 | 0.98 | 0.97 |
| | H3K4me3 active > poised | 0.51 | 0.42 | 0.38 | 0.41 | 0.47 | 0.48 |
| | H3K27me3 Polycomb > active | 1.00 | **0.24** | **0.11** | 0.61 | 0.61 | **0.45** |
| | H3K27me3 poised > Polycomb | 0.82 | 0.89 | 0.99 | 0.81 | 0.81 | 0.86 |

- **AUCs that equal 1.0 for the bulk track** follow from the class definitions (e.g. poised tiles are H3K4me3-marked,
  Polycomb tiles are not). There the question is how close a model gets to 1.0. **The other two rows** (active vs
  poised on H3K4me3, poised vs Polycomb on H3K27me3) are where the bulk track itself is between 0.5 and 0.8, and a
  model should match it rather than reach 1.0.
- **H3K4me3 poised > Polycomb** (the planned main readout): every model reaches 0.93–0.99, rules included, so it
  doesn't tell models apart. Poised tiles are open, unmethylated CpG islands; that alone predicts H3K4me3.
- **H3K4me3 active vs poised:** the networks are close to the bulk (within ±0.1) in GM12878, HepG2 and H1. They match
  bulk H1's finding that poised promoters carry about as much H3K4me3 as active ones. In K562 they overshoot
  (0.77–0.90 vs 0.71), from only 17 poised tiles. The MLP and rules put poised tiles above active ones (0.36–0.49),
  the wrong way round.
- **H3K27me3 Polycomb > active** separates the models:
  - the rules fail (0.24–0.37): their H3K27me3 rule needs a CpG island, and Polycomb-only tiles mostly aren't at one;
  - outside H1, R1a and Pair reach 0.83–0.98 and DEV4NF 0.67–0.89;
  - in H1 the networks reach only 0.45–0.61: Polycomb tiles there are methylated (D2), and every model has learned
    "methylated → little H3K27me3" (M2, C);
  - the MLP drops to 0.11 in H1.
- **H3K27me3 poised vs Polycomb:** the bulk says they're about equal outside H1 (0.52–0.60). The networks stay within
  0.15 of it (0.25 in K562); the MLP and rules put poised above (0.67–0.99), again because poised tiles are CpG
  islands. With 17–34 poised tiles, this AUC varies a lot between seeds outside H1 (e.g. Pair on HepG2: 0.21–0.59).
- **Predicted levels are compressed:** at poised tiles, the bulk mean is 1.7–2.8 (asinh FC) for H3K27me3 and the
  predictions 1.0–1.9. That is the usual shrinkage of a per-fiber average, not specific to this test.

### M2. Within a tile, do the predictions follow each fiber?

![Per-tile fiber-group differences](../plots/26_dec_04/poised/poised_M2.png)

Per tile, the difference of mean predictions between two fiber groups (≥ 3 fibers each); median over tiles, mean over
seeds:

| Contrast | Tiles | R1a | Pair | DEV4NF | MLP |
|---|---|---|---|---|---|
| H3K4me3, open − closed (unmethylated fibers), poised | all cells | +0.53 to +0.69 | +0.49 to +0.69 | +0.33 to +1.01 | +0.87 to +1.38 |
| H3K27me3, open − closed (unmethylated fibers), poised | all cells | −0.32 to −0.42 | −0.17 to −0.23 (K562 −0.05) | H1 −0.16; others mixed in sign | −0.10 to −0.74 |
| H3K4me3, open − closed (unmethylated fibers), Polycomb | all cells | +0.02 to +0.30 | −0.05 to +0.18 | −0.02 to +0.13 | +0.58 to +0.79 |
| H3K27me3, methylated − unmethylated (closed fibers), Polycomb | all cells | −0.13 to −0.45 | −0.03 to −0.40 | −0.19 to −0.31 | −0.37 to −0.52 |

- **At poised tiles, the model splits the molecules.**
  - Open unmethylated fibers get more H3K4me3 than closed ones: clear for every network, cell and seed.
  - The same open fibers get less H3K27me3: clear for R1a in all four cells, for Pair in three (not K562), for DEV4NF
    in H1.
  - So the network reads a poised tile as open "active-like" molecules plus closed "Polycomb-like" ones, which
    matches D2.
- **At Polycomb tiles, opening changes predicted H3K4me3 little** (networks ≤ +0.3). The networks tie accessibility to
  H3K4me3 only where the tile looks like a promoter (CpG island). The MLP raises H3K4me3 on every open fiber wherever
  it is (+0.6 to +0.8).
- **Methylated closed fibers get less H3K27me3**, at Polycomb tiles in every network, cell and seed (clear). At poised
  tiles too, though only H1 has enough methylated fibers (73 tiles). This matches the known biology: DNA methylation
  and Polycomb exclude each other at CpG islands.

### M3. Does the model put both marks on the same molecule?

Per tile, Spearman across fibers of predicted H3K4me3 vs predicted H3K27me3; median over tiles (figure
`poised_M3_C.png`, left):

| Cell | Class | R1a | Pair | DEV4NF |
|---|---|---|---|---|
| GM12878 | poised / Polycomb / active | −0.75 / −0.60 / −0.59 | −0.71 / −0.69 / −0.48 | −0.25 / −0.40 / −0.18 |
| HepG2 | poised / Polycomb / active | −0.75 / −0.58 / −0.46 | −0.76 / −0.69 / −0.40 | −0.31 / −0.27 / −0.06 |
| K562 | poised / Polycomb / active | +0.16 / +0.23 / −0.24 | +0.19 / +0.08 / −0.09 | +0.19 / −0.04 / −0.15 |
| H1 | poised / Polycomb / active | **−0.44** / +0.58 / −0.06 | **−0.34** / +0.29 / −0.10 | **−0.30** / +0.27 / −0.07 |

- **The networks predict the two marks on different molecules at poised tiles** in GM12878, HepG2 and H1. In H1 this
  is specific to poised tiles; at Polycomb and "neither" tiles the two predictions go together.
- In GM12878 and HepG2 the correlation is negative at every class, so there it is a general "open → H3K4me3 up,
  H3K27me3 down" pattern, not a poised-specific one. K562 (17 poised tiles) is slightly positive.
- There is no per-molecule truth here: this is what the model claims. It fits D2 (poised tiles as a mix of open and
  closed molecules), but true bivalent nucleosomes exist (both marks on one nucleosome) and Fiber-seq can't see them.
- The rules' values (up to ≈ 1.0, `poised_M3.tsv`) are an artefact of predictions that are constant within a tile.

### C. In-silico CpG flips: does methylation lower predicted H3K27me3?

The CpGs over the tile ± 100 bp are set to methylated on unmethylated fibers (meth), or to unmethylated on methylated
fibers (unmeth). As a control, the same width is methylated 1.5 kb away (meth_distal). Per tile, the median per-fiber
change of predicted H3K27me3; mean over seeds (figure `poised_M3_C.png`, right):

| Cell | Class | R1a meth | R1a meth_distal | Pair meth | DEV4NF meth | MLP meth |
|---|---|---|---|---|---|---|
| GM12878 | Polycomb | −0.46 | −0.22 | −0.30 | −0.22 | −0.66 |
| HepG2 | Polycomb | −0.56 | −0.25 | −0.45 | −0.39 | −0.71 |
| K562 | Polycomb | −0.68 | −0.37 | −0.43 | −0.47 | −0.69 |
| H1 | Polycomb | −0.06 | −0.03 | −0.01 | −0.10 | −0.47 |
| GM12878 | poised | −0.69 | −0.10 | −0.47 | −0.66 | −0.82 |
| HepG2 | poised | −0.51 | −0.12 | −0.35 | −0.56 | −0.60 |
| K562 | poised | −1.10 | −0.17 | −0.61 | −0.81 | −0.73 |
| H1 | poised | −0.05 | −0.05 | −0.14 | −0.32 | −0.58 |

- **Methylating unmethylated fibers lowers predicted H3K27me3**, and unmethylating methylated fibers raises it, at
  Polycomb and poised tiles. This is clear everywhere except in H1 for R1a and Pair at poised tiles and Pair at
  Polycomb tiles. R1a at H1 Polycomb tiles is clear but small (−0.06). In H1, most Polycomb fibers are already
  methylated, and the models trained with H1 give methylation less weight for H3K27me3 there.
- **Unlike the active marks, H3K27me3 also responds to flips 1.5 kb away:** −0.2 to −0.4 at Polycomb tiles outside
  H1, about half the local effect. For H3K4me3 and ATAC the distal flips do ~nothing (as for CTCF, `Nov_27.md` §5). The model
  reads methylation over a wider stretch for the broad mark.
- **Methylating poised tiles also lowers predicted H3K4me3** (networks −0.07 to −1.70), as at CTCF sites.
- **The MLP shows the same direction and is larger** (−0.47 to −0.82, even in H1). As for CTCF, the rule "methylation
  → less H3K27me3" is in the fiber features; a small model learns it too.

### Outcome

- **The data:** H1 has 5–15× more poised chromatin than the differentiated lines. Poised regions are unmethylated CpG
  islands with about half their molecules open, consistently at every site: bulk "poised" looks like a mix of open
  and closed molecules.
- **The model's reading of that mix:**
  - networks trained on bulk tracks only give the open molecules of a poised tile H3K4me3 and the closed ones
    H3K27me3 (M2, M3), in every network, mixture-trained R1a included;
  - they predict less H3K27me3 on methylated molecules, locally and over ±1.5 kb (M2, C);
  - between classes they match the bulk track where it is informative (M1), except for H3K27me3 at H1's methylated
    Polycomb domains.
- **Against the simple models:**
  - the rules miss Polycomb-only domains entirely;
  - the MLP over-assigns H3K27me3 to CpG islands and H3K4me3 to any open fiber;
  - "methylation lowers H3K27me3" is learned by the MLP too, so it is not evidence that the network learned more.
- **Limits:**
  - outside H1 the poised results rest on 17–34 tiles (2–8 at FC ≥ 3);
  - nothing here can tell "two kinds of molecule" from "both marks on one molecule": that is the model's claim, read
    beside D2.
- **On the sealed test:** the same script runs on HCT116 + MCF-7; their bulk tracks only define the classes.

## 2. Bulk accuracy of every model (27 Nov)

> **Question:** how do the models compare on the bulk metrics, per cell and per assay? This puts the molecule tests
> above beside plain accuracy against ENCODE.

**Setup:**
- **Readouts:**
  - r: Pearson r against each cell's own ENCODE track (asinh FC), from the pair's T3 eval (`eval24` / `eval27`);
  - fraction of the ceiling: r / the ENCODE replicate ceiling of that cell and assay (`Nov_13_extended.md` §9);
  - MSE: on the cell's own track;
  - mixture r: against the balanced 0.5 mixture track (`eval31` / `eval32`, chr21 mini).
- **Models:** DEV4NF (2 seeds), Rules (1), MLP (3), R1a (3), R1a+F (2 so far; seed-3 evals `2026-11-25_82` / `_87`
  still pending), Pair (3).
- **What each model is trained on:**
  - R1a and the ladder models (rules, MLP, R1a+F): mixtures of the pair only;
  - Pair: each cell's own tracks;
  - DEV4NF: own tracks of all 4 dev cells.
- **Pair B** has no ATAC: H1 has no ATAC track.
- **Differences to R1a:** SE from the seeds; clear when |Δ| > 2 SE and every seed pairing agrees.
- **Code and outputs:** script `plots_code/26_dec_04/bulk_models.py` (job `bulk_models.sh`), outputs in
  `plots/26_dec_04/bulk/`.

![Bulk r per assay and model](../plots/26_dec_04/bulk/bulk_assays.png)

Assay means, mean over seeds:

| Readout | Cell | Rules | MLP | R1a | R1a+F | Pair | DEV4NF |
|---|---|---|---|---|---|---|---|
| r | GM12878 | 0.26 | 0.34 | 0.47 | 0.47 | 0.48 | 0.46 |
| | HepG2 | 0.30 | 0.47 | 0.62 | 0.61 | 0.63 | 0.60 |
| | K562 | 0.21 | 0.30 | 0.48 | 0.48 | 0.49 | 0.48 |
| | H1 | 0.20 | 0.31 | 0.43 | 0.42 | 0.44 | 0.42 |
| fraction of ceiling | GM12878 | 0.34 | 0.45 | 0.64 | 0.64 | 0.65 | 0.63 |
| | HepG2 | 0.36 | 0.55 | 0.75 | 0.73 | 0.75 | 0.72 |
| | K562 | 0.28 | 0.41 | 0.65 | 0.64 | 0.66 | 0.66 |
| | H1 | 0.30 | 0.46 | 0.67 | 0.65 | 0.68 | 0.65 |
| MSE | GM12878 | 0.273 | 0.247 | 0.216 | 0.216 | 0.205 | 0.221 |
| | HepG2 | 0.321 | 0.270 | 0.213 | 0.214 | 0.219 | 0.224 |
| | K562 | 0.218 | 0.225 | 0.179 | 0.182 | 0.172 | 0.191 |
| | H1 | 0.306 | 0.283 | 0.257 | 0.261 | 0.246 | 0.259 |
| mixture r | pair A | 0.32 | 0.47 | 0.63 | 0.62 | 0.63 | 0.60 |
| | pair B | 0.23 | 0.36 | 0.53 | 0.53 | 0.53 | 0.51 |

Bulk r per assay (own track), mean over seeds:

| Cell | Model | ATAC | CTCF | H3K4me3 | H3K27ac | H3K4me1 | H3K36me3 | H3K27me3 | H3K9me3 |
|---|---|---|---|---|---|---|---|---|---|
| GM12878 | Rules | 0.58 | 0.16 | 0.25 | 0.35 | 0.29 | 0.37 | 0.08 | −0.02 |
| | MLP | 0.69 | 0.20 | 0.28 | 0.47 | 0.47 | 0.33 | 0.20 | 0.08 |
| | R1a | 0.76 | 0.27 | 0.45 | 0.61 | 0.63 | 0.51 | 0.34 | 0.22 |
| | Pair | 0.78 | 0.27 | 0.47 | 0.62 | 0.63 | 0.51 | 0.35 | 0.23 |
| | DEV4NF | 0.76 | 0.27 | 0.44 | 0.60 | 0.61 | 0.45 | 0.32 | 0.24 |
| HepG2 | Rules | 0.68 | 0.20 | 0.28 | 0.20 | 0.24 | 0.47 | 0.11 | 0.20 |
| | MLP | 0.76 | 0.46 | 0.27 | 0.37 | 0.58 | 0.40 | 0.29 | 0.60 |
| | R1a | 0.82 | 0.51 | 0.46 | 0.57 | 0.75 | 0.57 | 0.57 | 0.75 |
| | Pair | 0.82 | 0.50 | 0.48 | 0.57 | 0.76 | 0.58 | 0.59 | 0.71 |
| | DEV4NF | 0.81 | 0.50 | 0.45 | 0.55 | 0.73 | 0.52 | 0.57 | 0.65 |
| K562 | Rules | – | 0.19 | 0.41 | 0.09 | 0.07 | 0.56 | 0.08 | 0.04 |
| | MLP | – | 0.18 | 0.37 | 0.19 | 0.19 | 0.56 | 0.44 | 0.18 |
| | R1a | – | 0.29 | 0.59 | 0.50 | 0.43 | 0.65 | 0.59 | 0.32 |
| | Pair | – | 0.31 | 0.60 | 0.50 | 0.45 | 0.64 | 0.58 | 0.33 |
| | DEV4NF | – | 0.32 | 0.53 | 0.52 | 0.41 | 0.68 | 0.59 | 0.33 |
| H1 | Rules | – | 0.17 | 0.55 | 0.12 | 0.17 | 0.20 | 0.20 | −0.02 |
| | MLP | – | 0.40 | 0.59 | 0.27 | 0.41 | 0.11 | 0.20 | 0.17 |
| | R1a | – | 0.50 | 0.67 | 0.40 | 0.51 | 0.26 | 0.34 | 0.34 |
| | Pair | – | 0.54 | 0.67 | 0.41 | 0.51 | 0.26 | 0.34 | 0.36 |
| | DEV4NF | – | 0.50 | 0.67 | 0.36 | 0.49 | 0.26 | 0.32 | 0.34 |

- **R1a = Pair on bulk.** No assay-mean difference is clear on r, fraction of the ceiling or mixture r (|Δ| ≤ 0.015).
  Only MSE is lower for Pair (−0.006 to −0.011, clear on GM12878 and H1): Pair is trained on these exact tracks, so it
  gets their scale right, while R1a only sees mixtures. Per assay, Pair is clearly better only on CTCF in pair B
  (+0.03 / +0.04) and H3K4me3 in GM12878 (+0.01); R1a is clearly better on H3K9me3 in HepG2 (+0.04).
- **R1a is slightly above DEV4NF** in 3 of 4 cells:
  - r −0.01 to −0.03 for DEV4NF (clear on HepG2 and H1), mixture r −0.03 on pair A (clear);
  - the largest gaps are H3K9me3 in HepG2 (−0.10) and H3K36me3 in pair A (−0.05 / −0.06);
  - DEV4NF is clearly better on CTCF in K562 (+0.03) and H3K9me3 in GM12878 (+0.02);
  - K562 is level on the assay mean.

  A possible reason: DEV4NF trains on four cells' own tracks with the same budget, so the two cells of a pair get
  half of its training instead of all of it.
- **R1a+F = R1a** (Δ ≤ 0.01 on the assay means). Its few clear per-assay differences go both ways (H3K4me3 GM12878
  +0.02, H3K36me3 K562 +0.03; H3K4me1 and H3K27ac pair B −0.02 to −0.03). This is still on 2 seeds.
- **The simple rungs are far below:**
  - MLP −0.13 to −0.18 r and about −0.2 of the ceiling, clear on every assay and cell;
  - Rules −0.21 to −0.33.

  The MLP comes closest on ATAC (−0.07), CTCF in HepG2 (−0.05) and H3K4me3 in H1 (−0.08).
- **Where the rules beat the MLP,** an annotation carries the signal: H3K36me3 (gene bodies) in GM12878, HepG2 and H1;
  H3K4me3 (CpG islands) in HepG2 and K562. The MLP has no annotation, and the networks do better without one.
- **Hardest assays for every model:** H3K9me3 (except in HepG2), CTCF in GM12878 / K562, H3K36me3 in H1 and H3K27me3
  in GM12878 (R1a r ≤ 0.35, ≤ 0.6 of the ceiling).

## 3. X inactivation with block phasing (27 Nov)

> **Question:** the sealed X-inactivation cells (HG008_3E4, GM25455, HEK293T) are phased in short HiPhase blocks,
> not across the whole chromosome like GM12878. Can the XI test decide the inactive X block by block, from the fibers
> alone, without biasing the result? Validated on GM12878 before the sealed cells are touched.

**Rule** (fixed before looking; `plots_code/26_dec_04/xi_blocks.py`):
- **Signal per promoter:** the mean fiber CpG methylation on HP2 minus on HP1. It uses promoters with ≥ 5 fibers
  with a methylation call (≥ 3 covered CpGs) on each haplotype. Silenced CGI promoters count as they are (methylated
  on the Xi); XIST counts negated (unmethylated on the Xi).
- **Per tested gene:** average the signal over the other informative promoters in its block, leaving the tested gene
  out so its own methylation never sets its Xi.
  - average ≥ 0.2: Xi = HP2;
  - average ≤ −0.2: Xi = HP1;
  - otherwise no call, and the gene is dropped.
- **Then the usual test:** per gene, Xa − Xi of each prediction; per group, the fraction of genes with the expected
  sign (`Nov_20.md` §2).

**Validation on GM12878:**
- **Simulated blocks:** each GM12878 promoter is given the phase block it has in a sealed cell. Only the block layout
  is read from the sealed cells, from their reads' HP / PS tags; not their methylation or predictions.
- **Random swaps:** each block's HP labels are swapped at random, so the rule can't use GM12878's chromosome-wide
  phase.
- **Truth:** the chromosome-wide answer (Xi = HP2, from XIST).
- **Runs:** all 14 model × seed runs of the XI test (DEV4NF, R1a and Pair on both pairs).
- **Block layouts** (`xi_block_maps.py`): 177–241 blocks on chrX, median 2 promoters per block. 81–87% of the silenced
  CGI promoters share a block with another one.

![Block phasing vs chromosome-wide phasing on GM12878](../plots/26_dec_04/xi_blocks/xi_blocks.png)

| Readout | Expected (fixed in advance) | Result (HG008_3E4 / GM25455 / HEK293T layouts) |
|---|---|---|
| Accuracy of the Xi call | ≥ 0.95 | 0.995 / 1.000 / 0.995 (every run) |
| Same calls whatever the random swaps | yes | yes (no change over 5 draws) |
| Coverage: silenced CGI genes called / scored chromosome-wide (425) | ≥ 0.5 | **0.45** / 0.52 / **0.49** (193 / 221 / 210 genes) |
| Silenced CGI, fraction with the expected sign: block − chromosome-wide | within ± 0.05 | +0.006 to +0.046, every run inside |
| chr21 null, random orientation per block | ≈ 0.5 | 0.46–0.56 |
| chr21 by the rule | no calls (no allele-specific methylation) | 0 genes called |
| Escape control without an Xi call: silenced > escape + PAR on \|HP1 − HP2\| (added after the first run) | as chromosome-wide (5 of 5 marks) | 5 of 5 marks, p < 0.05, in every run and layout |

- **The Xi call is right** whenever the rule makes one (≥ 0.995), and it doesn't depend on how the labels were
  swapped.
- **The test result carries over.** The silenced-CGI readout is within 0.05 of the chromosome-wide one for every model
  and seed (left panel); per mark the differences are ≤ 0.05 too.
- **But it reads slightly higher (+0.02 on average, never lower).** The called genes sit in blocks whose neighbours
  are clearly methylated on one haplotype, i.e. in well-silenced regions, so the selection favours cleaner genes. Read
  the sealed-cell numbers with this in mind; the comparison between models on the same genes is unaffected.
- **Coverage is about half, just under the 0.5 expected for two of three layouts:**
  - 55–74 genes are alone in their block, so they have no neighbour to call from;
  - 130–159 genes are in blocks whose other silenced promoters show no methylation difference between haplotypes
    (average signal mostly within ±0.1). Those can't be called by any margin; a looser margin would only add
    errors.

  That still leaves about 200 silenced CGI genes per cell for the sealed test.
- **The escape + PAR control needs a version without an Xi call.** By the rule, only 1–8 escape + PAR genes get a call
  (33 chromosome-wide): they cluster in blocks without silenced neighbours.
  - Its question (do escape genes differ less between the two copies?) doesn't need to know which copy is the Xi.
  - Added (27 Nov, after the first run): per gene |HP1 − HP2| of each prediction, silenced CGI vs escape + PAR, one-sided
    Mann-Whitney per mark, on every phased gene (H2 of the original test).
  - **Result:** it keeps 412–419 silenced and 32–33 escape + PAR genes, and passes for all 5 marks (p < 0.05) in all
    14 runs and all three layouts, as chromosome-wide. Median |HP1 − HP2| is larger for silenced genes by 1.5–9×,
    least for H3K9me3 (e.g. H3K4me3 0.44–1.06 vs 0.09–0.17).

  `xi_blocks_escape.tsv`.

**Ready for the sealed cells:**
- **Inputs:** `xi_blocks.py` needs only those cells' molecule outputs at `tss_chrX_chr21` (`molecule_predict.py`, no
  new code) and their block maps (already made).
- **Readouts there:**
  - silenced CGI genes with the block rule;
  - escape + PAR vs silenced by |HP1 − HP2|;
  - chr21 with a random orientation per block as the null.

## Next steps

- When `2026-11-25_82` / `_87` finish: rerun `ladder.py --step summary` and `bulk_models.py` (R1a+F at 3 seeds).
- Poised vs Polycomb on the sealed test (HCT116 + MCF-7) with the same script.
- Remaining biology tests on the dev models (`TODO.md` §3): nucleosome positioning at CTCF, the CTCF specificity
  test.
