#!/bin/bash

#SBATCH --job-name=2026-11-03_05_train_spec_gm12878
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-2:45:00
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

# Specialists (reports/TODO.md, section 1): one model per dev cell type, trained on its own track only (data29-33),
# same model and budget as DEV5 V1 (model15, train04: 15 x 5000 samples, ~2.2 h). Pick the cell type with
#   sbatch --job-name=<name> --export=ALL,CELL=<gm12878|k562|hepg2|panc1|h1> slurm_batch_command.sh
# Each ends with its own cell type's mini eval; the full eval runs separately (slurm_batch_command_eval.sh).
# Unseen-pair test (T3b, reports/TODO.md section 2), same budget: CELL=pairA (GM12878 + HepG2, data28) and
# CELL=pairB (K562 + H1, data34), the two main models, each the other's matched reference.
# Mixture-only models of the same pairs: CELL=mixA (data27) and CELL=mixB (data35, no ATAC: H1 has none).
# If a job stops at the time limit it keeps last.pt: continue with --export=ALL,CELL=...,RESUME=latest.
PREFIX=SPEC_V1
case "${CELL:-gm12878}" in
  gm12878) DATA=data29_GM12878;    EVAL=eval21_GM12878_mini ;;
  k562)    DATA=data30_K562_200U;  EVAL=eval22_K562_200U_mini ;;
  hepg2)   DATA=data31_HepG2_200U; EVAL=eval23_HepG2_200U_mini ;;
  panc1)   DATA=data32_Panc1_200U; EVAL=eval16_Panc1_200U_mini ;;
  h1)      DATA=data33_H1_200U;    EVAL=eval17_H1_200U_mini ;;
  pairA)   DATA=data28_GM12878_HepG2_200U; EVAL=eval24_pair_GM12878_HepG2_200U_mini; PREFIX=PAIR_V1 ;;
  pairB)   DATA=data34_K562_H1_200U;       EVAL=eval22_K562_200U_mini;              PREFIX=PAIR_V1 ;;
  mixA)    DATA=data27_mix_GM12878_HepG2_200U; EVAL=eval26_mix_GM12878_HepG2_200U_mini; PREFIX=MIX_V1 ;;
  mixB)    DATA=data35_mix_K562_H1_200U;       EVAL=eval30_mix_K562_H1_200U_mini;       PREFIX=MIX_V1 ;;
  *) echo "Unknown CELL=${CELL}"; exit 1 ;;
esac
echo "Model: $PREFIX $DATA"
python -u main.py \
  --data_config configs/data/$DATA.yaml \
  --model_config configs/models/model15_uct_mid_8A.yaml \
  --train_config configs/training/train04.yaml \
  --eval_configs configs/evals/$EVAL.yaml \
  --num_workers 3 \
  --name_prefix $PREFIX \
  ${RESUME:+--resume "$RESUME"}

# Print job completion time
echo "Job finished on $(date)"
