#!/bin/bash

#SBATCH --job-name=2026-08-17_00_dna_test
#SBATCH --account=def-maxwl
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-3:00:00
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

# # Load required base modules
# module load python/3.11
# module load StdEnv/2023
# module load cudacore/.12.2.2
# module load scipy-stack/2024a
# module load gcc
# module load arrow/17.0.0

# Activate virtual environment
source /home/azr/lab/misc/menv/bin/activate

# Run the training
python main.py \
  --data_config configs/data/data00.yaml \
  --model_config configs/models/model00.yaml \
  --train_config configs/training/train00.yaml \
  --eval_config configs/evals/eval00.yaml \
  --dna_type both \
  --name_prefix dna_test

# Print job completion time
echo "Job finished on $(date)"
