#!/bin/bash

#SBATCH --job-name=2026-11-11_02_eval_replicate_ceiling
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-2:45:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

echo "Job started on $(date)"
echo "Running on host $(hostname)"

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# Replicate noise ceiling (replicate_ceiling.py): ENCODE replicate A vs B per cell type and assay, on the positions
# eval.py scores, with the DEV4 training cells (data24) as the cross-cell baseline. CPU only; fiber reading
# dominates (~18 min per cell type with 7 workers, ~2.4 h for all 8). Needs data_prep/setup_replicates.py first.
# CONFIGS overrides the eval configs (e.g. a mini config for a quick test).
CONFIGS="${CONFIGS:-configs/evals/eval21_GM12878.yaml configs/evals/eval22_K562_200U.yaml configs/evals/eval23_HepG2_200U.yaml configs/evals/eval17_H1_200U.yaml configs/evals/eval20_H9_200U.yaml configs/evals/eval18_HCT116_200U.yaml configs/evals/eval19_MCF_7_200U.yaml configs/evals/eval16_Panc1_200U.yaml}"
python -u replicate_ceiling.py \
  --train_data_config configs/data/data24_dev_loo_Panc1_200U.yaml \
  --eval_configs $CONFIGS \
  --num_workers 7

echo "Job finished on $(date)"
