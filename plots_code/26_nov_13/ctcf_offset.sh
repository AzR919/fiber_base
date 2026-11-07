#!/bin/bash

#SBATCH --job-name=2026-11-07_00_ctcf_offset
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-0:30:00

# One-off job (Nov 13 report): the K562-H1 CTCF offset. Part 1 (tracks, model predictions) plus part 2 (fiber
# accessibility of K562_200U / H1_200U at the CTCF differential bins and at constitutive sites). Writes
# plots/26_nov_13/ctcf/. See plots_code/26_nov_13/ctcf_offset.py.
set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
echo "Job started on $(date) on $(hostname)"
python -u plots_code/26_nov_13/ctcf_offset.py --fibers --workers 4
echo "Job finished on $(date)"
