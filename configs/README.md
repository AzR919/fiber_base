# configs/

How experiments are set up: which files exist, what is generated, and the logic behind the
cell-type split. Update this file whenever that logic changes.

## Layout

| Path | What | Edited by |
|---|---|---|
| `metapaths.yaml` | Fiber sample → CRAM, sample → ENCODE cell folder, per-cell assay status | **Generated** by data setup (`data_prep/setup_data.py`, step 1) from `data_prep/encode_spec.yaml`; don't edit by hand |
| `reference.yaml` | Reference genome, `regions_root`, training chromosomes | Hand-edited (merged into metapaths by `load_metapaths`) |
| *(cluster)* `DATA_FIBER/regions/` | Region files: training pools, val/test window sets (see Regions below) | **Generated** by data setup (step 2) from `data_prep/regions_spec.yaml`; referenced from configs by name |
| `data/dataNN_*.yaml` | Training data: which fiber samples, context length, fibers per sample, training regions, validation window set | Hand |
| `evals/evalNN_*.yaml` | Evaluation: one sample + a test window set each (`_mini` = `test_chr21_mini2000`) | Hand |
| `models/modelNN_*.yaml` | Architecture, inputs, output assays | Hand |
| `training/trainNN.yaml` | Epochs, iterations, batch size, lr, seed | Hand |
| `old_configs/` | Retired configs, kept for reproducing old runs | — |

Conventions:
- Configs are numbered per folder. Add a new config with the next number; don't edit one that has
  already been used in a run.
- Every config starts with a header comment saying what it is.
- `NN = 00` configs are tiny debug templates for smoke tests.
- Output assays come from the model config, never the data config. Assays a cell type has no ENCODE
  data for are masked: no loss, no gradient, and eval reports them as `null`.

## Fiber samples: 200U only

Several cell lines have Fiber-seq at two concentrations (100U and 200U). **For now we use only the 200U
samples, wherever a cell line has one.** GM12878 has a single sample (`GM12878`), so it is used as-is.
The 100U samples stay in `metapaths.yaml` for later.

## Cell-type split (v1)

The 8 cell lines with ENCODE data are split into a dev group of 5 and a held-out group of 3.

| Group | Sample | Cell line | ENCODE assays |
|---|---|---|---|
| Dev | `GM12878` | Lymphoblastoid (B cell) | all 8 |
| Dev | `K562_200U` | Erythroleukemia (myeloid) | all 8 |
| Dev | `HepG2_200U` | Liver carcinoma | all 8 |
| Dev | `Panc1_200U` | Pancreatic carcinoma | all 8 |
| Dev | `H1_200U` | Embryonic stem cell (male) | no ATAC |
| **Held-out** | `HCT116_200U` | Colon carcinoma | all 8 |
| **Held-out** | `MCF_7_200U` | Breast carcinoma | all 8 |
| **Held-out** | `H9_200U` | Embryonic stem cell (female) | no ATAC |

**Dev (5):** used for everything we tune: ablations, architecture choices, and leave-one-cell-out
checks.

**Held-out (3):** used only to test finished models, as an out-of-distribution test. Never train on
them, and never use their numbers to pick a model or a setting. If we do, they stop being held out.

Why this split:
- **The held-out cells range from near to far from the dev set,** so we see how performance falls off
  as the new cell gets less familiar, not just one number:
  - H9 is a near-twin of H1;
  - HCT116 is a new tissue of a related kind (gut-derived epithelial carcinoma, like HepG2 and Panc1);
  - MCF-7 has no close relative in dev.
- **The held-out cells have clean ENCODE targets.** HCT116 and MCF-7 have newer tracks (2016–17) with
  almost no NOT_COMPLIANT audits, whereas Panc1's broad marks carry 4–6 each (see
  `data_prep/encode_lock.yaml`). Noisy targets hurt evaluation more than training, so Panc1 trains.
- **It costs the least training data.** HCT116 and MCF-7 have only one fiber sample each.
- **GM12878, K562 and HepG2 stay in dev,** so results remain comparable with earlier runs.
- **H1 brings stem-cell chromatin into training**, for example bivalent promoters (H3K4me3 + H3K27me3).

Known gaps (accepted):
- **H1 and H9 have no ATAC.** The leave-one-out fold that holds out H1 gives no ATAC number, so report
  ATAC averaged over the other 4 folds. On the held-out side, ATAC is tested only on HCT116 and MCF-7.
- **No held-out cell is an entirely new kind of cell.** The alternative split (hold out H1 + H9 + Panc1)
  would test that, at the cost of weaker ATAC testing and less training data.

## Workflow

1. **Ablations / model selection (dev only).** For each setting, train a leave-one-out fold
   (`data21`–`data25`) and evaluate on the dev cell it left out (`eval21`–`eval23`, `eval16`, `eval17`). To keep cost down,
   screen settings with one fold first (e.g. `data23`, HepG2 held out). Run all 5 folds only for the
   finalists.
2. **Final models.** Train on all 5 dev cells (`data20`).
3. **Out-of-distribution test.** Evaluate final models on the held-out 3 (`eval18`–`eval20`).

| Config | Use |
|---|---|
| `data20_dev5_200U` | All 5 dev cells: final models |
| `data21_dev_loo_GM12878` | Dev minus GM12878 → eval with `eval21_GM12878` |
| `data22_dev_loo_K562_200U` | Dev minus K562 → eval with `eval22_K562_200U` |
| `data23_dev_loo_HepG2_200U` | Dev minus HepG2 → eval with `eval23_HepG2_200U` |
| `data24_dev_loo_Panc1_200U` | Dev minus Panc1 → eval with `eval16_Panc1_200U` |
| `data25_dev_loo_H1_200U` | Dev minus H1 → eval with `eval17_H1_200U` (no ATAC) |
| `eval18_HCT116_200U`, `eval19_MCF_7_200U`, `eval20_H9_200U` | Held-out test (final models only) |
| `data26_GM12878_K562_200U_ccre_only` | Temporary, outside the split: GM12878 + K562 with cCRE-only training (the old `data16` setup on region files), tested on HepG2 (`eval23`) to compare with old results. Retire when done |

Every eval config scores `test_chr21_tiles` (all 6,843 valid chr21 tiles) and has a `_mini` twin on
`test_chr21_mini2000` (2,000 of those tiles) for quick checks. The cCRE-list evals `eval13`–`eval15`
are retired to `old_configs/evals/`.

Model configs for the 8-assay setup (all UNet03 conv-transformer + reference DNA; each odd number has
an even-numbered `no_fire_msp` ablation twin):

| Config | Size | emb_dims | MIG slice |
|---|---|---|---|
| `model13_uct_small_8A` / `model14_…_no_fire_msp` | small | 8/16/32/64 | 1g.10gb |
| `model15_uct_mid_8A` / `model16_…_no_fire_msp` | mid | 8/24/48/96 | 2g.20gb |
| `model17_uct_large_8A` / `model18_…_no_fire_msp` | large | 8/32/64/128 | 2g.20gb |

## Regions

Where we train, validate and test is defined by region files, built by data setup step 2
(`data_prep/build_regions.py`) from `data_prep/regions_spec.yaml` into `DATA_FIBER/regions/`, and read
at runtime by `regions.py`. To change them, edit the spec and re-run data setup.

**Chromosomes (dev phase):** train on chr20 (`train_chrs` in `reference.yaml`), validate per epoch on
chr22, test on chr21. Val and test chromosomes are fixed by the window sets, not by chromosome lists,
so no chromosome is used for two purposes. Final runs add bigger sets (see below).

**Region vs window.** The files store **regions**: the interval we score (`R = 5000` bp). The runtime
builds the model's input **window** (`context_length = L`) centred on each region. Today `L = R`, so they
coincide. Centre-only scoring (`L > R`, score just the central `R` bp) is not implemented yet: with
`L ≠ R` the runtime prints a note and scores the whole window. Keep `context_length: 5000` for real runs.

**Region rule.** Every region (val, test, and training draws) satisfies:
1. It does not overlap the ENCODE hg38 blacklist v2.
2. An input window of up to `max_window = 20 kb` centred on it is inside the chromosome and has no N
   bases. So any `context_length` up to 20 kb works without rebuilding, and windows never need clamping.
3. It is on chr1–22. chrX/chrY/chrM are left out (the cell lines differ in sex and X copy number).

`allowed_centres.bed` is exactly the set of centres that satisfy 1–2.

**Window class** (used only to stratify sampling): the first of `PLS` > `ELS` (pELS/dELS) > `CTCF`
(CTCF-only) > `other` (DNase-H3K4me3) with a cCRE overlapping the region, else `background`.

### Files

| File | Contents |
|---|---|
| `exclude.bed` | Blacklist v2 + N runs (`chrom start end reason`) |
| `allowed_centres.bed` | Where a region centre may lie (region rule above) |
| `strata/ccre_classes.bed` | Every cCRE with its class, for per-bp scoring by class |
| `pools/ccre_centres.bed` | cCRE centres that satisfy the region rule: training pool |
| `windows/<set>.bed` | Fixed val/test sets: `chrom start end region_id class`, 0-based half-open, sorted; opens in IGV |
| `manifest.json` | How everything was built: rule parameters, seeds, class counts, md5 of the inputs |

### Val/test window sets

Val and test regions are tiles of one fixed grid: `start = k × 5000` from each chromosome start, kept
if the tile's centre is allowed. Tiles never overlap and a sampled set is always a subset of the full
tiling, so no region is ever counted twice.

| Set | Use | How it's built |
|---|---|---|
| `val_chr22_n200` | Per-epoch validation (200 regions × each training cell) | Proportional stratified sample of chr22 tiles, seed 919 |
| `test_chr21_tiles` | Reported dev-phase test numbers | Every valid chr21 tile |
| `test_chr21_mini2000` | Quick checks after training | Proportional stratified sample of the chr21 tiles, seed 919 |
| `test_chr21_ccre_centred` | Legacy: compare with old runs | 5 kb centred on every chr21 cCRE, overlaps kept (the old protocol) |

- **Proportional stratified:** each class gets the same share of the sample as it has among all tiles
  (largest-remainder rounding), sampled with a fixed seed. The mini set keeps the real composition, so
  its numbers estimate the full-tiling numbers.
- **Same regions for every cell type:** the eval config picks the cells; every region is run for each.
  Leave-one-out folds, held-out cells and baselines are all scored on identical regions.
- **Scoring by region type** happens per base pair: eval samples carry a mask of which bases fall in a
  cCRE of each class (from `strata/ccre_classes.bed`), and the evaluator reports MSE and Pearson r over
  `all` bases, `ccre` (any cCRE), `PLS` / `ELS` / `CTCF` / `other`, and `non_ccre`. The window class only
  steers sampling.
- **Regions without fibers** for a cell are skipped; eval prints "N of M samples evaluated".
- **Low-coverage positions are not scored:** loss and metrics only use positions covered by at least
  `min_coverage` fibers (default 5), counting each fiber only inside its scored span (alignment trimmed
  to its first..last nucleosome).
- **Final runs** add new sets to the spec (e.g. test tiles on chr10 + chr21); no code changes.

### Training (sampled at runtime)

Per sample: pick a source by the data config's weights (e.g. `{ccre: 0.5, random: 0.5}`), then draw a
region centre from `pools/ccre_centres.bed` (+ uniform jitter) or uniformly from `allowed_centres.bed`,
restricted to `train_chrs` (`reference.yaml`; dev phase `[chr20]`). Jittered centres must still be allowed.
The dev configs (`data20`–`data25`) use `{sources: {ccre: 0.5, random: 0.5}, jitter: 200}`: half the
regions on regulatory elements, half anywhere, so training covers background and broad domains like
the chr21 tiles we test on. Without `train_windows` the default is `{sources: {ccre: 1.0}, jitter: 200}`,
matching the old cCRE-centred training, so the retired `data12`–`data19` still run.

### Config keys

```yaml
# data config
train_windows: {sources: {ccre: 0.5, random: 0.5}, jitter: 200}   # weights are normalised
val_windows: val_chr22_n200                          # per-epoch validation: 200 regions x each training cell

# eval config
windows: test_chr21_mini2000                         # any set in regions/windows/
max_regions: 5                                       # optional: seeded subset, for debugging (eval00)
```

Old keys `num_sample_ccres` and `val_chrs` are rejected with an error pointing here.

### Outputs

- **Console:** per cell type, a table of Pearson r per region type × assay, plus how many samples were
  evaluated out of how many expected.
- **`eval_summary.json`:** per cell type, the window set, sample counts, the usual loss / Pearson per
  assay, and `strata`: bases scored, loss and r per region type, per assay.
- **wandb:** the usual `test/<cell>/loss…` and `pearson…` keys, plus `test/<cell>/pearson@<region type>`
  and `loss@<region type>` (means over assays).
