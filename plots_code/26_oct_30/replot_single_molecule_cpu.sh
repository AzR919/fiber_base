#!/bin/bash

#SBATCH --job-name=2026-10-27_04_replot_single_molecule_dev5_mse
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
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

# One-off job (Oct 30 presentation): re-plot the DEV5 single-molecule figures of an earlier run on CPU (same loci,
# no screening or ranking), e.g. after a plotting change. Writes results/<timestamp>_single_molecule_dev5/.
export REPLOT_FROM="results/26-10-27_T06-06-09_single_molecule_dev5"
export OMP_NUM_THREADS=4   # the other 4 CPUs feed the data loader workers
python -u plots_code/26_oct_30/single_molecule_dev5.py

echo "Job finished on $(date)"
