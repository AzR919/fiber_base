#!/bin/bash

#SBATCH --job-name=2026-07-14_00_gm_h3k4me3_avg_n_fibers
#SBATCH --account=def-maxwl
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-3:00:00  # 0 days, 3 hours, 0 minutes
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
    --other_data_path /home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/ENCFF287HAO_H3K4me3.bigWig \
    --batch_size 16 --epochs 25 --model deep01 --fibers_per_entry 200 --input_flags 1 1 1 1 1 \
    --res_dir ./results --decoder_type avg_n --kernel_size 15 \
    --name_suffix gm_h3k4me3_avg_n_fibers

# /home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/GM12878-fire-v0.1-filtered.cram
# GM12878/ENCFF798KYP_H3K27ac.bigWig
# GM12878/ENCFF287HAO_H3K4me3.bigWig
# GM12878/ENCFF012DMX_H3K4me3_signal.bigWig
# GM12878/ENCFF603BJO_ATAC_seq.bigWig
# GM12878/ENCFF667MDI_ATAC_seq_signal.bigWig

# /home/azr/projects/def-maxwl/azr/data/DATA_FIBER/K562/K562-fire-v0.1-filtered.cram
# K562/ENCFF911JVK_H3K4me3.bigWig
# K562/ENCFF071GML_H3K4me3_signal.bigWig
# K562/ENCFF102ARJ_ATAC_seq.bigWig
# K562/ENCFF357GNC_ATAC_seq_signal.bigWig

# Print job completion time
echo "Job finished on $(date)"
