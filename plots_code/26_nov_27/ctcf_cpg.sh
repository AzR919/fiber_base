#!/bin/bash
# CTCF x CpG methylation analysis (ctcf_cpg.py). CPU only; run after the --perturb molecule jobs.
#   sbatch --job-name=<YYYY-MM-DD_NN_ctcf_cpg> [--dependency=afterok:<ids>] plots_code/26_nov_27/ctcf_cpg.sh

#SBATCH --job-name=ctcf_cpg
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=0-1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
echo "Job started on $(date) on $(hostname)"
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=$SLURM_TMPDIR
mkdir -p plots/26_nov_27/ctcf_cpg
python -u plots_code/26_nov_27/ctcf_cpg.py 2>&1 | tee plots/26_nov_27/ctcf_cpg/log_ctcf_cpg.txt
echo "Job finished on $(date)"
