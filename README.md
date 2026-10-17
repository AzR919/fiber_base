# fiber_base

Base repository for single-molecule imputation using Fiber-seq.

**End goal: impute histone modifications at single-molecule resolution** — Given chromatin accessibility on an individual molecule (Fiber-seq), impute histone modifications on that molecule.

No ground truth exists at that resolution, so the model is trained with **weak supervision**: it takes a stack of single-molecule fiber reads at a genomic locus, predicts a per-fiber signal for each assay, and only the average of those per-fiber predictions is compared against bulk tracks (up to 8 ENCODE assays at once: ATAC-seq, CTCF and six histone marks). The bulk prediction is the training signal; the per-fiber predictions are the actual product.

---

## Overview

**What it learns**: given N fibers overlapping a window of length L centred on a genomic region, predict per-fiber, per-assay signal whose coverage-normalized average matches the bulk signal track(s) over that window.

**How the current model does it** (`UNet03ConvTransformerWithDNA`): each fiber is encoded independently (fused with a reference-DNA encoding), passed through a 1D U-Net with a transformer bottleneck, and turned into a non-negative per-fiber prediction for each assay. The bulk prediction is the mean of the per-fiber predictions over the fibers present at each position — i.e. "predict each fiber's contribution, then average".

**Fiber mask**: each fiber only counts inside its *scored span*: its alignment trimmed to its first..last nucleosome (read ends beyond the outermost nucleosomes are neither MSP nor nucleosome, so their msp/nuc channels are uninformative). The loader returns this as `fiber_mask` `(L, N)`; the model zeroes per-fiber predictions outside it, so empty parts of a fiber row and padding rows get no output, no gradient and do not count in the average. The trimmed read ends are still part of the input, as context. Fibers with no nucleosome are skipped.

**Scored positions**: loss and every metric only use positions covered by at least `min_coverage` fibers (default 5; `--min_coverage`), since averages over fewer fibers are noisy. Dashboards shade the unscored positions. The model returns both: `y` (bulk, `(B, K, L)`, used for the loss) and `processed_fibers` (single-molecule predictions, `(B, K, L, N)`, the imputation we ultimately care about).

**Validating the single-molecule output**: low bulk MSE is necessary but not sufficient — a model can match the bulk average while spreading signal uniformly across fibers. Mixed-cell deconvolution (below) is one indirect check that per-fiber predictions carry molecule-specific information.

**Two dataset modes** (`dataset_type`):
- **Single-cell** (`single`): each sample is fibers from one cell type predicting that cell type's bulk signal. With several cell types listed, each training sample picks one cell type at random.
- **Mixed-cell** (`mixed`): each sample is a composite fiber stack from several cell types at the same locus, with a ratio-weighted bulk target. Eval in this mode also reports per-cell-type deconvolution MSE.

**Where we train, validate and test** comes from **region files** built during data setup (`configs/README.md` → Regions):
- **Training** draws regions at runtime from pools on `chr20` (cCRE centres with jitter, optionally mixed with uniform random regions).
- **Validation** each epoch iterates a fixed window set on `chr22` (`val_chr22_n200`: 200 regions per training cell).
- **Testing** iterates fixed sets on `chr21`: `test_chr21_tiles` (all 6,843 valid non-overlapping 5 kb tiles) or `test_chr21_mini2000`.
- Metrics are also reported per region type (bases in promoter / enhancer / CTCF cCREs, outside cCREs).

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
| `data_utils.py` | `SingleCellFiberDataset`, `MixedCellFiberDataset`, `make_fiber_dataset()`: build windows around regions, load fibers + targets |
| `regions.py` | Reads the region files at runtime: window sets, allowed centres, training pools (`TrainSampler`), per-bp cCRE-class masks |
| `fiber_utils.py` | Low-level per-fiber feature extraction from CRAM files |
| `models.py` | `BaseModel` (save/load), `Base01DebugModel`, `UNet03ConvTransformerWithDNA`, `model_selector()` |
| `metrics.py` | Masked MSE (missing assays excluded), per-assay MSE, pooled + per-window Pearson |
| `utils.py` | Seeding, config + metapaths resolution, run naming, wandb setup, meters |
| `vis_utils.py` | Fiber-stack and evaluation dashboard plots |
| `configs/` | YAML configs: `data/`, `models/`, `training/`, `evals/`, `metapaths.yaml`, `reference.yaml`; retired ones in `old_configs/`. `configs/README.md` explains the cell-type split and the regions |
| `data_prep/` | Data setup: one command downloads ENCODE tracks, regenerates `metapaths.yaml` and builds the region files (see Data Setup) |
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
- Fiber mask: `(L, N)` bool per sample, `(B, L, N)` batched — each fiber's scored span (padding rows all False)
- Fiber coverage: `(L,)` per sample, `(B, L)` batched — `fiber_mask.sum(-1)`, the number of fibers scored at each position
- Reference DNA one-hot: `(4, L)` per sample

Channel order (selected by `input_flags`): `[m6a, cpg, msp, nuc, fire_msp]`
Assay order follows the model's `output_assays`.

---

## Configs

A run combines one data config, one model config, one training config and (optionally) one or more eval configs. File paths are not written in these configs — they are resolved from `configs/metapaths.yaml` using the cell type names and the model's `output_assays`.

### Metapaths (`configs/metapaths.yaml` + `configs/reference.yaml`)

`metapaths.yaml` is **generated** by data setup (step 1); don't edit it. It has:
- `fiber_paths`: fiber sample → CRAM.
- `cell_types`: fiber sample → its cell line's ENCODE folder. At runtime the folder's `manifest.json` lists which assays exist; requested assays a cell lacks are masked (no loss, no gradient, `null` in eval).
- `assay_status`: informational summary of which assays each cell line has.

`reference.yaml` is hand-edited: FASTA, `regions_root` (the region files) and `train_chrs`. `load_metapaths()` merges it in.

Cell lines with ENCODE data: **GM12878, K562, HepG2, HCT116, MCF-7, Panc1** (all 8 assays) and **H1, H9** (no ATAC). The dev / held-out split and the "200U samples only" rule are in `configs/README.md`.

### Data Config (`configs/data/`)

```yaml
context_length: 5000       # window size in bp
fibers_per_entry: 100      # fibers per sample
dataset_type: single       # 'single' or 'mixed'

train_windows: {sources: {ccre: 0.5, random: 0.5}, jitter: 200}   # training regions (default without the key: ccre 1.0)
val_windows: val_chr22_n200                          # per-epoch validation window set (default)

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
samples_per_epoch: 1000     # training samples per epoch (steps = samples / batch_size)
batch_size: 8
lr: 0.0001
res_dir: "./results"
seed: 919
# num_workers: 3           # optional DataLoader workers (default 0); datasets split their work across them
# min_coverage: 5          # optional (default 5): loss/metrics only at positions with >= this many fibers
```

### Eval Config (`configs/evals/`)

One config per cell type, each naming a test window set. Full variants use `test_chr21_tiles` (6,843 regions); `_mini` variants use `test_chr21_mini2000`.

```yaml
context_length: 5000
fibers_per_entry: 100
windows: test_chr21_tiles  # window set in <regions_root>/windows/
# max_regions: 5           # optional seeded subset (debugging)
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
| `model10` | UNet03 mid, ATAC only |
| `model11` | UNet03 mid, all 4 assays, all 5 inputs |
| `model12` | Same as model11 without `fire_msp` (ablation) |
| `model13` / `model14` | UNet03 small (8/16/32/64), all 8 assays, with / without `fire_msp`; 1g.10gb slice (tight) |
| `model15` / `model16` | UNet03 mid (8/24/48/96), all 8 assays, with / without `fire_msp`; 2g.20gb slice |
| `model17` / `model18` | UNet03 large (8/32/64/128), all 8 assays, with / without `fire_msp`; 2g.20gb slice |
| `train02` | Standard run (25 epochs × 1000 samples, bs 8); `train01` is retired (old key `iters_per_epoch`) |
| `data20` | All 5 dev cell types (split v1, 200U); training regions half cCRE, half random |
| `data21`–`data25` | Dev leave-one-out folds (dev set minus one cell type); same regions as `data20` |
| `data26` | Temporary: GM12878 + K562_200U, cCRE-only training, to compare with the old setup (test on HepG2, `eval23`) |
| `old_configs/data/data12`–`data19` | Retired pre-split configs (still run: no `train_windows` means cCRE-only) |
| `eval21` / `eval22` / `eval23` (+ `_mini`) | Eval on dev cells GM12878 / K562_200U / HepG2_200U |
| `eval16` / `eval17` (+ `_mini`) | Eval on dev cells Panc1_200U / H1_200U |
| `eval18` / `eval19` / `eval20` (+ `_mini`) | Held-out test: HCT116_200U / MCF_7_200U / H9_200U |
| `old_configs/evals/eval13`–`eval15` | Retired cCRE-list evals (do not run with the current code) |

---

## Data Setup (`data_prep/`)

One command sets up all data. Every step skips work that is already done, so it is cheap and safe to re-run:

```bash
sbatch data_prep/slurm_setup_data.sh          # as a CPU job (needs internet: ENCODE + GitHub)
python data_prep/setup_data.py --dry-run      # report what each step would do, write nothing
python data_prep/setup_data.py                # run interactively
```

| Step | Script | Hand-edited input | Generated output |
|---|---|---|---|
| 1. ENCODE tracks | `setup_encode.py` | `encode_spec.yaml` (cell lines, assays, fiber samples) | `<encode_root>/<cell>/*_fc.bigWig` + `manifest.json`, `encode_lock.yaml`, `configs/metapaths.yaml` |
| 2. Regions | `build_regions.py` | `regions_spec.yaml` (chromosomes, region rule, val/test window sets) | `<regions_root>/`: `exclude.bed`, `allowed_centres.bed`, `strata/`, `pools/`, `windows/`, `manifest.json` |

- **Skipping:** ENCODE files already on disk with the right md5 are skipped, and `encode_lock.yaml` pins which file each cell + assay uses. Region files are rebuilt only when `regions_spec.yaml`, `build_regions.py` or an input (FASTA index, cCRE file, blacklist) changes; `manifest.json` stores a fingerprint of all of them.
- **Flags:** `--dry-run`, `--workers N` (parallel downloads), `--verify` (re-hash ENCODE files), `--force-regions`.
- **To change the data:** edit a spec and re-run. Never edit generated files. Each step can also run on its own (`python data_prep/setup_encode.py`, `python data_prep/build_regions.py`).
- **ENCODE file selection** (`encode_api.py`): released, unperturbed GRCh38 "fold change over control" bigWigs; per experiment the file with the most biological replicates; experiments ranked by audit severity, then replicates, then recency. Audits rank candidates, never exclude them.

---

## How to Train

```bash
python main.py \
  --data_config  configs/data/data23_dev_loo_HepG2_200U.yaml \
  --model_config configs/models/model15_uct_mid_8A.yaml \
  --train_config configs/training/train02.yaml \
  [--eval_configs configs/evals/eval21_GM12878_mini.yaml configs/evals/eval23_HepG2_200U_mini.yaml] \
  [--metapaths configs/metapaths.yaml] \
  [--num_workers 3] [--name_prefix myrun] [--name_suffix v1] [-D]
```

- Results go to `results/<YY-MM-DD_THH-MM-SS>_<prefix>_<data>_<model>_<train>_<suffix>/` (checkpoints `Model_epoch_N.pt`).
- If `--eval_configs` is given, each eval config runs after training and is logged to wandb.
- Any config value can be overridden on the CLI (e.g. `--lr 3e-4`).
- `--num_workers N` loads data in N background processes (the job's CPUs minus one is a good start). Workers are started with `forkserver`, because pyft hangs in workers forked from a process that has already read fibers; any script that uses workers needs an `if __name__ == "__main__":` guard (`main.py` and `eval.py` have one).

**Reproducibility**: the same `seed` + same epoch always gives the same training data order (`set_epoch(epoch)`); validation and eval iterate fixed window sets. With `num_workers > 0` the data are identical for a given number of workers (each worker has its own seed and share of the regions).

---

## How to Evaluate a Saved Checkpoint

```bash
python eval.py \
  --checkpoint results/<run_dir>/Model_epoch_25.pt \
  --eval_configs configs/evals/eval21_GM12878.yaml configs/evals/eval22_K562_200U.yaml configs/evals/eval23_HepG2_200U.yaml \
  [--metapaths configs/metapaths.yaml] [--device cuda] \
  [--output_dir ./results] [--save_plots] [--wandb] [--num_workers 3]
```

Prints, per cell type, how many samples were evaluated (regions without fibers are skipped) and a table of Pearson r per region type × assay. `eval_summary.json` has the full breakdown (`strata`: bases, loss and r per region type, per assay).

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
