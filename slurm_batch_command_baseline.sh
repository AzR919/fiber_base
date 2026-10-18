#!/bin/bash

#SBATCH --job-name=2026-10-18_01_baseline_data20
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

# Cross-cell-average baseline (CPU only, no model): for every eval sample, the mean of the training cells' tracks
# (excluding the evaluated cell line), scored like a model on the full chr21 tiles of all 8 cell types.
# It depends only on the data, so it runs once per training set; the summary JSON is saved under results/.
# Set TRAIN_DATA_CONFIG to the data config whose cell_types are the training cells.

set -euo pipefail  # Exit on any error, undefined variables, and pipe failures

TRAIN_DATA_CONFIG=configs/data/data20_dev5_200U.yaml

# Move to the directory where your script should run
SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

# Print some information about the job
echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

# Activate virtual environment
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

python -u eval.py \
  --baseline_only --train_data_config "$TRAIN_DATA_CONFIG" \
  --eval_configs configs/evals/eval21_GM12878.yaml configs/evals/eval22_K562_200U.yaml configs/evals/eval23_HepG2_200U.yaml \
                 configs/evals/eval16_Panc1_200U.yaml configs/evals/eval17_H1_200U.yaml \
                 configs/evals/eval18_HCT116_200U.yaml configs/evals/eval19_MCF_7_200U.yaml configs/evals/eval20_H9_200U.yaml \
  --device cpu \
  --num_workers 7

# Print job completion time
echo "Job finished on $(date)"
