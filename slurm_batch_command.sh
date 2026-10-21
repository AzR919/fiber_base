#!/bin/bash

#SBATCH --job-name=2026-10-21_02_t3_upper
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-4:00:00
#SBATCH --dependency=afterok:66297589
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

# Upper bound for the mixture test (T3): GM12878 + HepG2_200U trained as single cell types on their own tracks
# (data28), same model and budget as 2026-10-21_01_t3_mix. Starts only if the smoke test (66297589) succeeds. Also
# the V1 single-cell reference. Quick paired mini eval at the end; full eval separately (2026-10-21_04).
python -u main.py \
  --data_config configs/data/data28_GM12878_HepG2_200U.yaml \
  --model_config configs/models/model15_uct_mid_8A.yaml \
  --train_config configs/training/train02.yaml \
  --eval_configs configs/evals/eval24_pair_GM12878_HepG2_200U_mini.yaml \
  --num_workers 3 \
  --name_prefix T3_upper

# Print job completion time
echo "Job finished on $(date)"
