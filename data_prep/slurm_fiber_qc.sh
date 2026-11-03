#!/bin/bash

#SBATCH --job-name=2026-11-03_00_fiber_qc
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-1:00:00

# Fiber-seq data QC (data_prep/fiber_qc.py): library stats and agreement with ENCODE for every sample; outputs in
# DATA_FIBER/qc/. Extra arguments are passed on, e.g. sbatch data_prep/slurm_fiber_qc.sh --samples Panc1_200U.
set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
echo "Job started on $(date) on $(hostname)"
python -u data_prep/fiber_qc.py --windows 300 --workers 8 "$@"
echo "Job finished on $(date)"
