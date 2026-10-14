#!/bin/bash

#SBATCH --job-name=2026-10-04_03_ceehrc26_single_molecule_large
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
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

# One-off job: for a hand-picked list of indices (edit SELECTED_INDICES in
# single_molecule_plotter.py), renders large-font versions into
# plots/ceehrc_26/<label>/: the all-assay dashboard, plus single-molecule
# figures with 2 assay columns each (5 fibers per column).
python plots_code/ceehrc_26/single_molecule_plotter.py

echo "Job finished on $(date)"
