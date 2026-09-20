#!/bin/bash

#SBATCH --job-name=2026-09-19_01_gm12878_all_assays
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-4:00:00
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

# Run the training
python main.py \
  --data_config configs/data/data12_gm_atac.yaml \
  --model_config configs/models/model11_uct_mid_all_A.yaml \
  --train_config configs/training/train01.yaml \
  --eval_configs configs/evals/eval12_gm_atac_mini.yaml \
  --name_prefix All_A_GM_run

# Print job completion time
echo "Job finished on $(date)"
