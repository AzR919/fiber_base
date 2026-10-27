#!/bin/bash

#SBATCH --job-name=2026-10-27_02_single_molecule_dev5_baseline_ref
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# One-off job (Oct 30 presentation): single-molecule figures of the DEV5 long model on chr21 test tiles, dev and
# held-out cell types, sharp and broad marks, tiles that differ from / line up with the cross-cell average.
# Screens with the ENCODE tracks, ranks candidates with the model, plots best + random picks.
# Writes results/<timestamp>_single_molecule_dev5/. Edit the EDIT ME block in single_molecule_dev5.py.
python -u plots_code/26_oct_30/single_molecule_dev5.py

echo "Job finished on $(date)"
