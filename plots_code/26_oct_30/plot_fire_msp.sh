#!/bin/bash

#SBATCH --job-name=2026-10-26_00_fire_msp_plots
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# One-off job (Oct 30 presentation): single-molecule figures of the fire_msp ablation (with vs without fire_msp,
# reports/Oct_16.md) for H3K27ac and H3K4me3 on GM12878 and K562, 10 chr21 cCRE windows each, 7 fibers per
# figure. Writes results/26_oct_30_fire_msp/<cell>/. Edit the EDIT ME block in fire_msp_single_molecule.py.
python -u plots_code/26_oct_30/fire_msp_single_molecule.py

echo "Job finished on $(date)"
