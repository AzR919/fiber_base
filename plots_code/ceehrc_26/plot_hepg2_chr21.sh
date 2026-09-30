#!/bin/bash

#SBATCH --job-name=2026-09-29_00_ceehrc26_hepg2_chr21_plots
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

set -euo pipefail

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# One-off plotting job: pre-filters HepG2_200U chr21 cCREs to those with
# ATAC & H3K4me3 & H3K27ac signal > 1.5, then plots that set for 3 models.
# Rerunning after a timeout is safe — already-plotted loci are skipped.
python plots_code/ceehrc_26/plotter.py

echo "Job finished on $(date)"
