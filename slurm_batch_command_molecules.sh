#!/bin/bash

#SBATCH --job-name=2026-11-07_05_molecules
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=36G
#SBATCH --time=0-1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"
echo "Job started on $(date) on $(hostname)"
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# Per-molecule predictions at a target BED (molecule_predict.py; reports/TODO.md section 2). Pick the model, the
# samples and the targets with
#   sbatch --job-name=<name> --time=<close to expected> \
#          --export=ALL,TARGETS=<bed>,SAMPLES="GM12878 K562_200U"[,RUN=<training run name>][,EXTRA="--save_arrays 20"] \
#          slurm_batch_command_molecules.sh
# RUN defaults to DEV4. Fiber reading dominates (~0.5-1 s per target and sample); a 1g.10gb slice is enough.
RUN="${RUN:-LOO_Panc1_V1_data24_dev_loo_Panc1_200U_model15_uct_mid_8A_train04}"
: "${TARGETS:?set TARGETS=<bed>}"
: "${SAMPLES:?set SAMPLES=\"<sample> ...\"}"
python -u molecule_predict.py --run "$RUN" --samples $SAMPLES --targets "$TARGETS" ${EXTRA:-}

echo "Job finished on $(date)"
