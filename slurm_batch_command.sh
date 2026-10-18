#!/bin/bash

#SBATCH --job-name=2026-10-18_03_dev5_long
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-8:00:00
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

# Dev-split model trained 4x longer: same as 2026-10-17_08_dev5_ccre_random (data20, mid 8-assay model) but
# train03 (100 x 1000 samples); HepG2 mini eval at the end (full 8-cell eval with baseline runs separately)
python -u main.py \
  --data_config configs/data/data20_dev5_200U.yaml \
  --model_config configs/models/model15_uct_mid_8A.yaml \
  --train_config configs/training/train03.yaml \
  --eval_configs configs/evals/eval23_HepG2_200U_mini.yaml \
  --num_workers 3 \
  --name_prefix DEV5_long

# Print job completion time
echo "Job finished on $(date)"
