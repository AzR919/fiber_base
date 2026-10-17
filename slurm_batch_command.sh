#!/bin/bash

#SBATCH --job-name=2026-10-17_01_test00_debug
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-0:30:00
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

# Test 1: tiny debug setup (00 configs) end to end: train 2 x 10 samples, validate, eval on 5 regions
python -u main.py \
  --data_config configs/data/data00.yaml \
  --model_config configs/models/model00.yaml \
  --train_config configs/training/train00.yaml \
  --eval_configs configs/evals/eval00.yaml \
  --num_workers 3 \
  --name_prefix TEST00

# Print job completion time
echo "Job finished on $(date)"
