#!/bin/bash
# Bulk metrics of every model, per assay and cell (bulk_models.py). CPU only.
#   sbatch --job-name=<YYYY-MM-DD_NN_bulk_models> plots_code/26_dec_04/bulk_models.sh

#SBATCH --job-name=bulk_models
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-0:20:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
echo "Job started on $(date) on $(hostname)"
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=$SLURM_TMPDIR
mkdir -p plots/26_dec_04/bulk
python -u plots_code/26_dec_04/bulk_models.py 2>&1 | tee plots/26_dec_04/bulk/log_bulk.txt
echo "Job finished on $(date)"
