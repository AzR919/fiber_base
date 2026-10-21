#!/bin/bash

#SBATCH --job-name=2026-10-21_04_t3_upper_eval
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-12:00:00
#SBATCH --dependency=afterok:66297802
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

# Full mixture-test (T3) eval of the single-cell upper bound (2026-10-21_02_t3_upper, job 66297802; starts only if it
# succeeds), on exactly the configs of 2026-10-21_03_t3_mix_eval: paired GM12878 vs HepG2_200U (eval24), library
# control (eval25), mixture composites (eval26_mini), with the mixture reference as baseline.
CKPT=$(ls -td results/*_T3_upper_data28_GM12878_HepG2_200U_model15_uct_mid_8A_train02/Model_epoch_25.pt | head -1)
echo "Checkpoint: $CKPT"
python -u eval.py \
  --checkpoint "$CKPT" \
  --baseline --mixture_reference \
  --eval_configs configs/evals/eval24_pair_GM12878_HepG2_200U.yaml configs/evals/eval25_pair_HepG2_100U_200U.yaml \
                 configs/evals/eval26_mix_GM12878_HepG2_200U_mini.yaml \
  --num_workers 7 \
  --wandb

# Print job completion time
echo "Job finished on $(date)"
