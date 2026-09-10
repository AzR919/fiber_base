# fiber_base

Base repository for single-molecule epigenomics imputation using Fiber-seq data. The model takes single-molecule fiber reads (m6A, CpG, MSP, NUC, FIRE signals) as input and predicts bulk chromatin accessibility or histone modification profiles.

---

## Overview

**What it learns**: given a stack of single-molecule fiber reads at a genomic locus, predict the aggregate bulk signal (e.g., ATAC-seq, H3K4me3) at that locus.

**Two training modes**:
- **Single-cell** (`dataset_type: single`): each training sample is fibers from one cell type predicting the bulk signal of that same cell type. Tests whether the model can learn the fiber→bulk relationship within a cell type.
- **Mixed-cell** (`dataset_type: mixed`): each training sample is a composite fiber stack from multiple cell types at a shared locus, with a ratio-weighted bulk target. Closer to the evaluation distribution when mixed eval is used.

**Two evaluation modes** (same `dataset_type` flag):
- **Single-cell eval**: iterates loci deterministically, yields one (locus, cell_type) sample per cell type per locus. Computes composite MSE only.
- **Mixed-cell eval**: iterates loci deterministically, builds a composite fiber stack from all cell types, computes composite MSE and per-cell-type deconvolution MSE.

Eval is **fully deterministic**: same model + same eval config → identical results every run.

---

## Repository Structure

| File | Description |
|---|---|
| `main.py` | Entry point — parses args, builds datasets, trains model |
| `args.py` | Argument parsing and config file loading (CLI > config > defaults) |
| `trainer.py` | Training loop, validation, wandb logging, post-training eval trigger |
| `evaluator.py` | Independent evaluator: takes model + dataset, runs inference, returns metrics |
| `eval.py` | Standalone CLI for evaluating a saved checkpoint |
| `data_utils.py` | `SingleCellFiberDataset`, `MixedCellFiberDataset`, `make_fiber_dataset()` |
| `models.py` | `BaseModel` and all concrete model classes |
| `metrics.py` | Loss functions (MSE only for now) |
| `utils.py` | Shared utilities: seeding, config loading, metapaths resolution, meters |
| `vis_utils.py` | Visualization functions for fiber stacks and evaluation dashboards |
| `fiber_utils.py` | Low-level fiber data extraction from CRAM/BAM files |
| `args.py` | Argument parsing and config file loading |
| `configs/` | YAML config files for data, models, training, and evaluation |

---

## Tensor Shape Contract

- Fiber features: `(C, L, N)` — Channels × context_Length × N_fibers
- Batched fiber features: `(B, C, L, N)`
- Bulk target: `(L,)` per sample, `(B, L)` batched
- Fiber coverage: `(L,)` per sample, `(B, L)` batched
- DNA one-hot: `(4, L)` per sample

Channel order (controlled by `input_flags`): `[m6a, cpg, msp, nuc, fire_msp]`

---

## Config Schema

### Data Config (`configs/data/`)

```yaml
dataset_type: single       # 'single' or 'mixed'
context_length: 2048       # window size in base pairs
fibers_per_entry: 200      # total fibers per sample
bulk_name: ATAC_seq_fcc    # label for target bulk signal
assay: atac                # assay key — resolves paths via metapaths.yaml

cell_types:
  GM12878:                 # single-cell: no ratio needed
  K562:                    # for mixed: optionally add ratio:
    # ratio: 0.5
```

### Eval Config (`configs/evals/`)

```yaml
dataset_type: mixed        # 'single' or 'mixed'
context_length: 2048
fibers_per_entry: 200
num_sample_ccres: 1000     # number of cCREs to evaluate (-1 = all)
bulk_name: ATAC_seq_fcc
seed: 919
assay: atac

cell_types:
  GM12878:
    ratio: 0.5
  K562:
    ratio: 0.5
```

> `input_flags` and `dna_type` are **not** read from eval configs — they are always loaded from the model checkpoint (`model.init_args`).

### Model Config (`configs/models/`)

```yaml
model: base
d_model: 64
decoder_type: avg_n
kernel_size: 15
input_flags: [1, 1, 1, 1, 1]   # [m6a, cpg, msp, nuc, fire_msp]
dna_type: none                  # 'none' | 'ref' | 'fiber' | 'both'
```

### Training Config (`configs/training/`)

```yaml
epochs: 50
iters_per_epoch: 1000
batch_size: 16
lr: 1e-4
seed: 919
```

---

## How to Train

```bash
python main.py \
  --data_config configs/data/data00.yaml \
  --model_config configs/models/model00.yaml \
  --train_config configs/training/train00.yaml \
  [--eval_config_path configs/evals/eval00.yaml] \
  [--metapaths configs/metapaths.yaml] \
  [--name_prefix myrun] \
  [--name_suffix v1]
```

If `--eval_config_path` is provided, the evaluator runs automatically after training and logs results to wandb.

**Reproducibility**: set `seed` in the training config. The same seed + same epoch always produces the same data ordering via `set_epoch(epoch)`.

---

## How to Evaluate a Saved Checkpoint

```bash
python eval.py \
  --checkpoint results/26-09-09_T20-12-27_.../Model_epoch_50.pt \
  --eval_config configs/evals/eval00.yaml \
  [--metapaths configs/metapaths.yaml] \
  [--device cuda] \
  [--output_dir ./eval_results] \
  [--save_plots]
```

`input_flags` and `dna_type` are extracted from the checkpoint automatically — do not put them in the eval config.

---

## Cluster Setup (DRAC Nibi)

**Code**: `/project/def-maxwl/azr/code/fiber_base`

**Virtualenv**: `source /project/def-maxwl/azr/misc/menv/bin/activate`

**Slurm job naming**: `YYYY-MM-DD_NN_description` — increment `NN` for each new run on the same day.

**Slurm template** (`slurm_batch_command.sh`):

```bash
#SBATCH --job-name=2026-09-09_00_description
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-4:00:00

source /project/def-maxwl/azr/misc/menv/bin/activate
python main.py --data_config configs/data/data00.yaml ...
```

**WandB**: project `fiber` under entity `liblab`. Login with `wandb login` before submitting GPU jobs.

---

## Reproducibility Guarantees

| Context | Guarantee |
|---|---|
| Eval (any mode) | Fully deterministic: same checkpoint + same eval config = identical results |
| Training data ordering | Seeded-deterministic: same `seed` + same `epoch` = same data sequence |
| Val loss during training | Consistent across epochs (uses `mode='eval'` with fixed cCRE iteration) |

---

## Dataset Modes: When to Use Which

| Training mode | Eval mode | Scientific question |
|---|---|---|
| `single` | `mixed` | Can the model generalize from single-cell training to mixed inputs? (zero-shot test) |
| `mixed` | `mixed` | Does training on mixed inputs improve mixed-input eval? |
| `single` | `single` | Standard single-cell fiber→bulk learning |
| `mixed` | `single` | Can a mixed-trained model handle single-cell inference? |
