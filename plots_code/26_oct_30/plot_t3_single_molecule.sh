#!/bin/bash

#SBATCH --job-name=2026-10-27_01_t3_single_molecule
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
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

# One-off job (Oct 30 presentation): single-molecule figures of the mixture test (T3): the mixture model on each
# cell type's fibers alone at GM12878 / HepG2 differential bins, best per assay and direction + random picks, chosen
# from the mixture eval's per-bin fiber-origin AUC table. Writes results/<timestamp>_t3_single_molecule/.
# Edit the EDIT ME block in t3_single_molecule.py.
python -u plots_code/26_oct_30/t3_single_molecule.py

echo "Job finished on $(date)"
