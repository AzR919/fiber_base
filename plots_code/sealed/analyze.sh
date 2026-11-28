#!/bin/bash
# Sealed test analysis (sealed_evals.py, sealed_molecules.py; reports/sealed_test_protocol.md). CPU only.
#   sbatch --job-name=<YYYY-MM-DD_NN_sealed_analysis> --export=ALL,SETUP=sealed|dryrun[,STEP=all] plots_code/sealed/analyze.sh
# score_sealed.sh submits it after the scoring jobs (SETUP=sealed). Logs: <out>/log_evals.txt, log_molecules.txt.

#SBATCH --job-name=sealed_analysis
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-1:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
echo "Job started on $(date) on $(hostname)"
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}
SETUP=${SETUP:?set SETUP=sealed or dryrun}
OUT=$([ "$SETUP" = sealed ] && echo plots/sealed || echo plots/sealed_dryrun)
mkdir -p "$OUT"
python -u plots_code/sealed/sealed_evals.py --setup "$SETUP" 2>&1 | tee "$OUT/log_evals.txt"
python -u plots_code/sealed/sealed_molecules.py --setup "$SETUP" --step "${STEP:-all}" 2>&1 | tee "$OUT/log_molecules.txt"
echo "Job finished on $(date)"
