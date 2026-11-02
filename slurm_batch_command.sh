#!/bin/bash

#SBATCH --job-name=2026-11-01_02_dev5_v1_train04
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-5:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail  # Exit on any error, undefined variables, and pipe failures

# Move to the directory where your script should run
SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

# Print some information about the job
echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

# Activate virtual environment
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# Reference model bank (reports/TODO.md, section 1): DEV5 with V1 fiber averaging, all 5 dev cell types (data20), mid
# 8-assay model, train04 (15 x 5000 samples, LR patience 2). Replaces the V0 DEV5 runs. HepG2 mini eval at the end;
# the full 8-cell eval with the fold-change baseline runs separately.
python -u main.py \
  --data_config configs/data/data20_dev5_200U.yaml \
  --model_config configs/models/model15_uct_mid_8A.yaml \
  --train_config configs/training/train04.yaml \
  --eval_configs configs/evals/eval23_HepG2_200U_mini.yaml \
  --num_workers 3 \
  --name_prefix DEV5_V1

# Print job completion time
echo "Job finished on $(date)"
