#!/bin/bash

#SBATCH --job-name=2026-09-21_00_GM_K5_eval_full
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

# Activate virtual environment
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# Run evaluation — fill in --checkpoint before submitting
python eval.py \
  --checkpoint results/26-09-21_T05-57-36_All_A_GM_K5_run_data16_GM12878_K562_200U_model11_uct_mid_all_A_train01/Model_epoch_25.pt \
  --eval_configs configs/evals/eval13_GM12878.yaml configs/evals/eval14_K562_200U.yaml configs/evals/eval15_HepG2_200U.yaml \
  --wandb

# Print job completion time
echo "Job finished on $(date)"
