#!/bin/bash
# Baseline ladder (ladder.py): CTCF / XI for the new rungs, then the summary with CpG flips and per-assay tables. CPU only.
#   sbatch --job-name=<YYYY-MM-DD_NN_ladder> plots_code/26_nov_27/ladder.sh

#SBATCH --job-name=ladder
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=0-0:45:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
echo "Job started on $(date) on $(hostname)"
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=$SLURM_TMPDIR
mkdir -p plots/26_nov_27/ladder
python -u plots_code/26_nov_27/ladder.py --step "${STEP:-all}" 2>&1 | tee plots/26_nov_27/ladder/log_ladder.txt
echo "Job finished on $(date)"
