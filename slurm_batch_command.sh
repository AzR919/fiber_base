#!/bin/bash

#SBATCH --job-name=2026-10-17_02_test_dataload
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-1:00:00
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

# Test 2: real-size data (5 kb, 100 fibers, 4 dev cells incl. H1 without ATAC, cCRE + random training regions,
# chr22 validation) with the mid 8-assay model, kept short: 2 epochs x 48 samples, then the HepG2 mini eval
python -u main.py   --data_config configs/data/data23_dev_loo_HepG2_200U.yaml   --model_config configs/models/model15_uct_mid_8A.yaml   --train_config configs/training/train02.yaml   --eval_configs configs/evals/eval23_HepG2_200U_mini.yaml   --epochs 2 --samples_per_epoch 48   --num_workers 3   --name_prefix TEST_dataload

# Print job completion time
echo "Job finished on $(date)"
