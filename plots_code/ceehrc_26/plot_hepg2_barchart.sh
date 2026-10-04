#!/bin/bash

#SBATCH --job-name=2026-09-30_02_ceehrc26_hepg2_barchart
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-00:15:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# One-off job: publication-quality HepG2_200U MSE bar chart from reports/Sep_25.md.
python plots_code/ceehrc_26/hepg2_mse_barchart.py

echo "Job finished on $(date)"
