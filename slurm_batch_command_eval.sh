#!/bin/bash

#SBATCH --job-name=2026-10-18_04_dev5_long_eval_baseline
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-10:00:00
#SBATCH --dependency=afterok:65777536
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

# Full evaluation of the 100-epoch dev-split model (2026-10-18_03_dev5_long, job 65777536; starts only if it succeeds)
# on all 8 cell types, with the
# cross-cell-average baseline of its training cells and the cell-specific score (partial correlation of
# prediction and target given the baseline). The baseline reads every training cell's tracks, hence 7 workers.
CKPT=$(ls -td results/*_DEV5_long_data20_dev5_200U_model15_uct_mid_8A_train03/Model_epoch_100.pt | head -1)
echo "Checkpoint: $CKPT"
python -u eval.py \
  --checkpoint "$CKPT" \
  --baseline \
  --eval_configs configs/evals/eval21_GM12878.yaml configs/evals/eval22_K562_200U.yaml configs/evals/eval23_HepG2_200U.yaml \
                 configs/evals/eval16_Panc1_200U.yaml configs/evals/eval17_H1_200U.yaml \
                 configs/evals/eval18_HCT116_200U.yaml configs/evals/eval19_MCF_7_200U.yaml configs/evals/eval20_H9_200U.yaml \
  --num_workers 7 \
  --wandb

# Print job completion time
echo "Job finished on $(date)"
