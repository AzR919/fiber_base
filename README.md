# fiber_base

Base repository for single-molecule imputation using Fiber-seq.

**End goal: impute histone modifications at single-molecule resolution** — Given chromatin accessibility on an individual molecule (Fiber-seq), impute histone modifications on that molecule.

No ground truth exists at that resolution, so the model is trained with **weak supervision**: it takes a stack of single-molecule fiber reads at a genomic locus, predicts a per-fiber signal for each assay, and only the average of those per-fiber predictions is compared against bulk tracks (currently ATAC-seq, H3K4me3, H3K27ac and H3K27me3, all at once). The bulk prediction is the training signal; the per-fiber predictions are the actual product.

---

## Overview

**What it learns**: given N fibers overlapping a window of length L around a cCRE, predict per-fiber, per-assay signal whose coverage-normalized average matches the bulk signal track(s) over that window.

**How the current model does it** (`UNet03ConvTransformerWithDNA`): each fiber is encoded independently (fused with a reference-DNA encoding), passed through a 1D U-Net with a transformer bottleneck, and turned into a non-negative per-fiber prediction for each assay. The bulk prediction is the sum over fibers divided by fiber coverage at each position — i.e. "predict each fiber's contribution, then average". The model returns both: `y` (bulk, `(B, K, L)`, used for the loss) and `processed_fibers` (single-molecule predictions, `(B, K, L, N)`, the imputation we ultimately care about).

**Validating the single-molecule output**: low bulk MSE is necessary but not sufficient — a model can match the bulk average while spreading signal uniformly across fibers. Mixed-cell deconvolution (below) is one indirect check that per-fiber predictions carry molecule-specific information.

**Two dataset modes** (`dataset_type`):
- **Single-cell** (`single`): each sample is fibers from one cell type predicting that cell type's bulk signal. With several cell types listed, each training sample picks one cell type at random.
- **Mixed-cell** (`mixed`): each sample is a composite fiber stack from several cell types at the same locus, with a ratio-weighted bulk target. Eval in this mode also reports per-cell-type deconvolution MSE.

**Train / validation split**: by chromosome — train on `chr20` cCREs (~25k), validate and evaluate on `chr21` cCREs (~11.4k). Set in `configs/metapaths.yaml`.

Eval is **fully deterministic**: same checkpoint + same eval config → identical results.

---

## Repository Structure

| Path | Description |
|---|---|
| `main.py` | Training entry point — parses args, builds datasets and model, trains, optionally runs eval |
| `eval.py` | Standalone evaluation of a saved checkpoint |
| `args.py` | Argument parsing and config loading (CLI > config files > defaults) |
| `trainer.py` | Training loop, validation, checkpointing, wandb logging |
| `evaluator.py` | `Evaluator` (inference + metrics) and `run_final_eval` (loops over eval configs) |
| `data_utils.py` | `SingleCellFiberDataset`, `MixedCellFiberDataset`, `make_fiber_dataset()` |
| `fiber_utils.py` | Low-level per-fiber feature extraction from CRAM files |
| `models.py` | `BaseModel` (save/load), `Base01DebugModel`, `UNet03ConvTransformerWithDNA`, `model_selector()` |
| `metrics.py` | Loss functions (MSE) |
| `utils.py` | Seeding, config + metapaths resolution, run naming, wandb setup, meters |
| `vis_utils.py` | Fiber-stack and evaluation dashboard plots |
| `configs/` | YAML configs: `data/`, `models/`, `training/`, `evals/`, `metapaths.yaml`; retired ones in `old_configs/` |
| `slurm_batch_command.sh` | Slurm script for training runs |
| `slurm_batch_command_eval.sh` | Slurm script for evaluating a checkpoint |
| `logs/` | Slurm `.out` / `.err` files, named by job name |
| `reports/` | Dated write-ups of results |
| `plots_code/` | Figure scripts (e.g. `ceehrc_26/`) |
| `misc/` | Sync scripts (macOS + Windows), handy commands (`misc.txt`), data-prep and sweep scripts |

---

## Tensor Shape Contract

- Fiber features: `(C, L, N)` — Channels × context Length × N fibers; batched `(B, C, L, N)`
- Bulk target: `(K, L)` per sample, `(B, K, L)` batched — one row per output assay, `asinh`-transformed bigWig signal (NaN → 0)
- Fiber coverage: `(L,)` per sample, `(B, L)` batched
- Reference DNA one-hot: `(4, L)` per sample

Channel order (selected by `input_flags`): `[m6a, cpg, msp, nuc, fire_msp]`
Assay order follows the model's `output_assays`.

---

## Configs

A run combines one data config, one model config, one training config and (optionally) one or more eval configs. File paths are not written in these configs — they are resolved from `configs/metapaths.yaml` using the cell type names and the model's `output_assays`.

### Metapaths (`configs/metapaths.yaml`)

Single source of truth for paths on the cluster: hg38 FASTA, cCRE BED, train/val chromosomes, a base directory per assay, and per-cell-type filenames for the fiber CRAM and each bulk bigWig. `null` means that file isn't available yet. To add a cell type, add a block here — no code changes needed.

Cell types with fiber + all four bulk assays: **GM12878, K562_200U, HepG2_200U**. Others (A549, Caco-2, H1, H9, several LCL/fibroblast/iPSC lines) have fiber data only so far.

### Data Config (`configs/data/`)

```yaml
context_length: 5000       # window size in bp
fibers_per_entry: 100      # fibers per sample
dataset_type: single       # 'single' or 'mixed'

cell_types:
  GM12878:                 # single: no ratio needed
  K562_200U:               # mixed: optionally add  ratio: 0.5
  HepG2_200U:
```

### Model Config (`configs/models/`)

```yaml
model: "unet03"                    # 'unet03' or 'base01' (debug)
input_flags: [1, 1, 1, 1, 1]       # [m6a, cpg, msp, nuc, fire_msp]
dna_type: "ref"                    # 'none' | 'ref'
kernel_size: 15
tf_heads: 4
tf_layers: 2
emb_dims: [8, 24, 48, 96]
output_assays: ["atac", "h3k4me3", "h3k27ac", "h3k27me3"]   # keys from metapaths
```

### Training Config (`configs/training/`)

```yaml
epochs: 25
iters_per_epoch: 1000
batch_size: 8
lr: 0.0001
res_dir: "./results"
seed: 919
```

### Eval Config (`configs/evals/`)

One config per cell type. `_mini` variants use a 500-cCRE subset for quick checks; full variants use every val cCRE.

```yaml
context_length: 5000
fibers_per_entry: 100
num_sample_ccres: -1       # -1 = all val cCREs; otherwise a seeded subset
seed: 919
dataset_type: single

cell_types:
  GM12878:
```

> `input_flags`, `dna_type` and `output_assays` are **not** read from eval configs — they always come from the checkpoint (`model.init_args`).

### Current configs

| Config | What it is |
|---|---|
| `data00`, `model00`, `train00`, `eval00` | Tiny debug/smoke-test setup |
| `data13` / `data14` / `data15` | Single cell type: GM12878 / K562_200U / HepG2_200U |
| `data16` / `data18` / `data19` | Two cell types (one held out): GM+K562 / GM+HepG2 / K562+HepG2 |
| `data17` | All three cell types |
| `model10` | UNet03 mid, ATAC only |
| `model11` | UNet03 mid, all 4 assays, all 5 inputs |
| `model12` | Same as model11 without `fire_msp` (ablation) |
| `model13` / `model14` | UNet03 small (8/16/32/64), all 8 assays, with / without `fire_msp`; 1g.10gb slice (tight) |
| `model15` / `model16` | UNet03 mid (8/24/48/96), all 8 assays, with / without `fire_msp`; 2g.20gb slice |
| `model17` / `model18` | UNet03 large (8/32/64/128), all 8 assays, with / without `fire_msp`; 2g.20gb slice |
| `train01` | Standard run (25 × 1000 iters, bs 8) |
| `eval13` / `eval14` / `eval15` (+ `_mini`) | Eval on GM12878 / K562_200U / HepG2_200U |

---

## How to Train

```bash
python main.py \
  --data_config  configs/data/data17_GM12878_K562_200U_HepG2_200U.yaml \
  --model_config configs/models/model11_uct_mid_all_A.yaml \
  --train_config configs/training/train01.yaml \
  [--eval_configs configs/evals/eval13_GM12878_mini.yaml configs/evals/eval15_HepG2_200U_mini.yaml] \
  [--metapaths configs/metapaths.yaml] \
  [--name_prefix myrun] [--name_suffix v1] [-D]
```

- Results go to `results/<YY-MM-DD_THH-MM-SS>_<prefix>_<data>_<model>_<train>_<suffix>/` (checkpoints `Model_epoch_N.pt`).
- If `--eval_configs` is given, each eval config runs after training and is logged to wandb.
- Any config value can be overridden on the CLI (e.g. `--lr 3e-4`).

**Reproducibility**: the same `seed` + same epoch always gives the same training data order (`set_epoch(epoch)`); validation uses a fixed cCRE iteration.

---

## How to Evaluate a Saved Checkpoint

```bash
python eval.py \
  --checkpoint results/<run_dir>/Model_epoch_25.pt \
  --eval_configs configs/evals/eval13_GM12878.yaml configs/evals/eval14_K562_200U.yaml configs/evals/eval15_HepG2_200U.yaml \
  [--metapaths configs/metapaths.yaml] [--device cuda] \
  [--output_dir ./results] [--save_plots] [--wandb]
```

---

## Cluster Setup (DRAC nibi)

- **Code**: `/project/def-maxwl/azr/code/fiber_base`
- **Venv**: `source /project/def-maxwl/azr/misc/menv/bin/activate`
- **WandB**: entity `liblab`, project `fiber`. Run `wandb login` once before submitting GPU jobs.
- **Submit**: `sbatch slurm_batch_command.sh` (training) or `sbatch slurm_batch_command_eval.sh` (eval). Logs land in `logs/<job-name>.out/.err`.
- **Job naming**: `YYYY-MM-DD_NN_description`, e.g. `2026-10-14_00_no_fire`. `NN` is a two-digit index that increments with each job submitted that day.
- **GPUs**: H100 MIG slices — `nvidia_h100_80gb_hbm3_{1g.10gb,2g.20gb,3g.40gb}`. The mid-size UNet03 fits in `2g.20gb`; full evals fit in `1g.10gb`.

### Syncing local ↔ nibi

Code is edited locally and synced to nibi over an SSH ControlMaster socket (2FA when the socket opens). `wandb/`, `results/`, `ignore/`, caches and `.git/` are never synced.

| | macOS | Windows (WSL) |
|---|---|---|
| Start | `misc/sync_fiber.sh [nibi\|fir]` — rsync + watchdog, pushes each saved file | `misc\sync_fiber.bat [nibi\|fir] [--no-pull]` — pulls remote first, then Mutagen two-way-safe |
| Stop | `misc/sync_fiber_stop.sh` | `misc\sync_fiber_stop.bat` |

Run the stop script before closing the laptop to avoid fail2ban bans from dangling connections. See `misc/misc.txt` for other cluster commands.

---

## Dataset Modes: When to Use Which

| Training mode | Eval mode | Scientific question |
|---|---|---|
| `single` | `single` | Standard fiber → bulk learning |
| `single` | `mixed` | Does a single-cell-trained model generalize to mixed inputs? (zero-shot) |
| `mixed` | `mixed` | Does training on mixed inputs improve mixed-input eval? |
| `mixed` | `single` | Can a mixed-trained model handle single-cell inference? |
