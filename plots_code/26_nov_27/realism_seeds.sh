#!/bin/bash
# Seed analysis of the mixture realism test (realism_seeds.py): CTCF / XI per run, then the summary. CPU only.
#   sbatch --job-name=<YYYY-MM-DD_NN_realism_seeds> plots_code/26_nov_27/realism_seeds.sh

#SBATCH --job-name=realism_seeds
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=0-1:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
echo "Job started on $(date) on $(hostname)"
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=$SLURM_TMPDIR
mkdir -p plots/26_nov_27/seeds
python -u plots_code/26_nov_27/realism_seeds.py --step "${STEP:-all}" 2>&1 | tee plots/26_nov_27/seeds/log_seeds.txt
echo "Job finished on $(date)"
