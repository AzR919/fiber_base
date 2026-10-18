#!/bin/bash

#SBATCH --job-name=2026-10-17_07_gm_k5_ccre_eval_full
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-4:00:00
#SBATCH --dependency=afterok:65690544
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

# Full evaluation (all valid chr21 tiles) of the GM12878 + K562 cCRE-only replication run (job 65690544,
# 2026-10-17_06_gm_k5_ccre) on all 8 cell types. Starts only if that training job succeeds; the checkpoint is
# the newest matching results folder.
CKPT=$(ls -td results/*_GM_K5_ccre_data26_GM12878_K562_200U_ccre_only_model15_uct_mid_8A_train02/Model_epoch_25.pt | head -1)
echo "Checkpoint: $CKPT"
python -u eval.py \
  --checkpoint "$CKPT" \
  --eval_configs configs/evals/eval21_GM12878.yaml configs/evals/eval22_K562_200U.yaml configs/evals/eval23_HepG2_200U.yaml \
                 configs/evals/eval16_Panc1_200U.yaml configs/evals/eval17_H1_200U.yaml \
                 configs/evals/eval18_HCT116_200U.yaml configs/evals/eval19_MCF_7_200U.yaml configs/evals/eval20_H9_200U.yaml \
  --num_workers 3 \
  --wandb

# Print job completion time
echo "Job finished on $(date)"
