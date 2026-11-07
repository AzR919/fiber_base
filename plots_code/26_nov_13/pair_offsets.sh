#!/bin/bash

#SBATCH --job-name=2026-11-07_01_pair_offsets
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-0:25:00

# One-off job (Nov 13 report): the CTCF decomposition (track scale / fiber accessibility / constant model shift) for
# every assay on both pairs. Writes plots/26_nov_13/offsets/. See plots_code/26_nov_13/pair_offsets.py.
set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
echo "Job started on $(date) on $(hostname)"
python -u plots_code/26_nov_13/pair_offsets.py --workers 8 --max_bins 200
echo "Job finished on $(date)"
