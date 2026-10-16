# configs/

How experiments are set up: which files exist, what is generated, and the logic behind the
cell-type split. Update this file whenever that logic changes.

## Layout

| Path | What | Edited by |
|---|---|---|
| `metapaths.yaml` | Fiber sample → CRAM, sample → ENCODE cell folder, per-cell assay status | **Generated** by `data_prep/setup_data.py` from `data_prep/encode_spec.yaml`; don't edit by hand |
| `reference.yaml` | Reference genome, cCRE file, train/val chromosomes | Hand-edited (merged into metapaths by `load_metapaths`) |
| `data/dataNN_*.yaml` | Training data: which fiber samples, context length, fibers per sample | Hand |
| `evals/evalNN_*.yaml` | Evaluation sets (one sample each; `_mini` = 500 seeded cCREs) | Hand |
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
   (`data21`–`data25`) and evaluate on the dev cell it left out (`eval13`–`eval17`). To keep cost down,
   screen settings with one fold first (e.g. `data23`, HepG2 held out). Run all 5 folds only for the
   finalists.
2. **Final models.** Train on all 5 dev cells (`data20`).
3. **Out-of-distribution test.** Evaluate final models on the held-out 3 (`eval18`–`eval20`).

| Config | Use |
|---|---|
| `data20_dev5_200U` | All 5 dev cells: final models |
| `data21_dev_loo_GM12878` | Dev minus GM12878 → eval with `eval13_GM12878` |
| `data22_dev_loo_K562_200U` | Dev minus K562 → eval with `eval14_K562_200U` |
| `data23_dev_loo_HepG2_200U` | Dev minus HepG2 → eval with `eval15_HepG2_200U` |
| `data24_dev_loo_Panc1_200U` | Dev minus Panc1 → eval with `eval16_Panc1_200U` |
| `data25_dev_loo_H1_200U` | Dev minus H1 → eval with `eval17_H1_200U` (no ATAC) |
| `eval18_HCT116_200U`, `eval19_MCF_7_200U`, `eval20_H9_200U` | Held-out test (final models only) |

All eval configs have a `_mini` twin (500 seeded cCREs) for quick checks.

Model configs for the 8-assay setup (all UNet03 conv-transformer + reference DNA; each odd number has
an even-numbered `no_fire_msp` ablation twin):

| Config | Size | emb_dims | MIG slice |
|---|---|---|---|
| `model13_uct_small_8A` / `model14_…_no_fire_msp` | small | 8/16/32/64 | 1g.10gb |
| `model15_uct_mid_8A` / `model16_…_no_fire_msp` | mid | 8/24/48/96 | 2g.20gb |
| `model17_uct_large_8A` / `model18_…_no_fire_msp` | large | 8/32/64/128 | 2g.20gb |

## Chromosomes and windows

- **Chromosomes:** train on `chr20`, validate on `chr21` (set in `reference.yaml`).
  - Eval configs run on the validation chromosome, so every eval number is on chromosomes and cell
    types the model never trained on, unless stated otherwise.
  - An eval config can override this with its own `val_chrs:` key, e.g. to score a held-out cell on
    chr20 and separate "new cell type" from "new region of the genome".
  - Caveat: `run_final_eval` keys results by cell type, so two configs for the same cell type in one
    eval run overwrite each other. Run them as separate eval jobs.
- **Windows:** samples are 5 kb windows centred on cCREs (5000 bp context, 100 fibers). Random
  windows beyond cCREs are planned.
