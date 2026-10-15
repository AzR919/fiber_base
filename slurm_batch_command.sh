#!/bin/bash

#SBATCH --job-name=2026-10-14_00_no_fire
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

# Run the training (fire_msp ablation: same as the all-three model11 run, minus fire_msp input)
python main.py \
  --data_config configs/data/data17_GM12878_K562_200U_HepG2_200U.yaml \
  --model_config configs/models/model12_uct_mid_all_A_no_fire_msp.yaml \
  --train_config configs/training/train01.yaml \
  --eval_configs configs/evals/eval15_HepG2_200U_mini.yaml \
  --name_prefix All_A_All_T_no_fire

# Print job completion time
echo "Job finished on $(date)"
