#!/bin/bash

#SBATCH --job-name=2025-12-27_04_ATAC_seq_nan_bug_fix
#SBATCH --account=def-maxwl
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-3:00:00  # 0 days, 23 hours, 59 minutes
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail  # Exit on any error, undefined variables, and pipe failures

# Move to the directory where your script should run
SCRIPT_DIR='/home/azr/lab/base/fiber_base/'
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
    --fiber_data_path /home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/GM12878-fire-v0.1-filtered.cram \
    --other_data_path /home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/ENCFF603BJO_ATAC_seq.bigWig \
    --batch_size 8 --epochs 100 --model simple --fibers_per_entry 200 \
    --res_dir ./results --name_suffix atac_seq_nan_bug_fix

# Print job completion time
echo "Job finished on $(date)"
