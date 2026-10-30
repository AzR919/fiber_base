#!/bin/bash

#SBATCH --job-name=2026-10-29_03_t3_single_molecule_fibers
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail
umask 027

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# One-off job (Oct 30 presentation): mixture-test (T3) single-molecule figures, one assay each, with the Fiber-seq
# inputs (input heatmap + example fibers), on CPU. Writes results/<timestamp>_t3_single_molecule_fibers/.
export OMP_NUM_THREADS=4   # the other 4 CPUs feed the data loader workers
python -u plots_code/26_oct_30/t3_single_molecule_fibers.py

echo "Job finished on $(date)"
