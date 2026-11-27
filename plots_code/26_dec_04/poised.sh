#!/bin/bash
# Poised vs Polycomb (poised.py, T5g): reads the poised molecule outputs. CPU only.
#   sbatch --job-name=<YYYY-MM-DD_NN_poised> plots_code/26_dec_04/poised.sh

#SBATCH --job-name=poised
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=0-0:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
echo "Job started on $(date) on $(hostname)"
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=$SLURM_TMPDIR
mkdir -p plots/26_dec_04/poised
python -u plots_code/26_dec_04/poised.py 2>&1 | tee plots/26_dec_04/poised/log_poised.txt
echo "Job finished on $(date)"
