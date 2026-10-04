#!/bin/bash

#SBATCH --job-name=2026-09-30_01_ceehrc26_single_molecule
#SBATCH --account=def-maxwl_cpu
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

#SBATCH### --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# One-off job: promotes a hand-picked list of indices (edit SELECTED_INDICES
# in single_molecule_plotter.py) from ignore/ceehrc_26/<label>/ to
# plots/ceehrc_26/<label>/, and renders a per-molecule / per-assay grid
# (one column per assay, 10 fibers per column) for each.
python plots_code/ceehrc_26/single_molecule_plotter.py

echo "Job finished on $(date)"
