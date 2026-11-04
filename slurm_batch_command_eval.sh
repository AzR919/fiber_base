#!/bin/bash

#SBATCH --job-name=2026-11-03_02_eval_loo_panc1_v1_a
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail  # Exit on any error, undefined variables, and pipe failures

# Move to the directory where your script should run
SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

# Print some information about the job
echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

# Activate virtual environment
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

# Full eval on all chr21 tiles with the cross-cell baseline (fold-change mean of the training cells except the
# evaluated line) and the cell-type r, split v2 (configs/README.md). Two jobs of ~25-30 min each (the baseline reads
# every training cell's tracks per sample); pick one with
#   sbatch --job-name=<name> --export=ALL,EVAL_SET=<a|b>[,RUN=<training run name>] slurm_batch_command_eval.sh
#   a: dev (GM12878, K562, HepG2, H1)   b: held out (H9, HCT116, MCF-7, and Panc1 = hard)
# RUN (default: the DEV4 model, data24 = the 4 dev cells) picks the newest results/*_train_<RUN> checkpoint,
# e.g. RUN=DEV5_V1_data20_dev5_200U_model15_uct_mid_8A_train04 for the split v1 model.
# EVAL_SET=one with CONFIGS="<eval config paths>" evaluates any list; BASELINE=0 skips the baseline (needed for a
# single-cell specialist: its baseline would exclude its only training cell line, leaving nothing to average).
# MIXREF=1 adds --mixture_reference (baseline = the mixture track; for a mixture model on its own pair).
RUN="${RUN:-LOO_Panc1_V1_data24_dev_loo_Panc1_200U_model15_uct_mid_8A_train04}"
case "${EVAL_SET:-a}" in
  a) CONFIGS="configs/evals/eval21_GM12878.yaml configs/evals/eval22_K562_200U.yaml configs/evals/eval23_HepG2_200U.yaml configs/evals/eval17_H1_200U.yaml" ;;
  b) CONFIGS="configs/evals/eval20_H9_200U.yaml configs/evals/eval18_HCT116_200U.yaml configs/evals/eval19_MCF_7_200U.yaml configs/evals/eval16_Panc1_200U.yaml" ;;
  one) : "${CONFIGS:?EVAL_SET=one needs CONFIGS}" ;;
  *) echo "Unknown EVAL_SET=${EVAL_SET}"; exit 1 ;;
esac
CKPT=$(ls -td results/*_train_${RUN}/Model_epoch_15.pt | head -1)
echo "Checkpoint: $CKPT | eval set ${EVAL_SET:-a}: $CONFIGS"
python -u eval.py \
  --checkpoint "$CKPT" \
  $([ "${BASELINE:-1}" = 1 ] && echo --baseline) \
  $([ "${MIXREF:-0}" = 1 ] && echo --mixture_reference) \
  --eval_configs $CONFIGS \
  --num_workers 3 \
  --wandb

# Print job completion time
echo "Job finished on $(date)"
