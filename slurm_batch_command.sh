#!/bin/bash

#SBATCH --job-name=2026-11-11_04_train_dev4_seed2
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
# Input ablations of DEV4 (reports/TODO_extended.md T10), same data (data24) and budget: CELL=nocpg (model19),
# CELL=nofire (model16), CELL=noacc (model20: CpG + DNA only), CELL=m6acpg (model21: m6A + CpG, no fibertools calls).
# If a job stops at the time limit it keeps last.pt: continue with --export=ALL,CELL=...,RESUME=latest.
# DEV4 itself: CELL=dev4 (model15 on data24, run prefix LOO_Panc1_V1). Another training seed of any of these: add
# SEED=<n> (default 919); the run prefix gets _seed<n>, so evals and analyses never pick it up as the seed-919 run.
# Mixture realism test (reports/TODO_extended.md §2b), GM12878 + HepG2 with DEV4NF's channels (model16):
# CELL=r0_nf (data27, share redrawn per window), CELL=r1a_nf (data36, one fixed share, balanced stacks),
# CELL=pair_nf (data28, own tracks; the ceiling). Second pair, K562 + H1: CELL=r1a_nf_b (data37), CELL=pair_nf_b
# (data34). R1a is the mixture recipe from 23 Nov; 3 seeds each (SEED=2, SEED=3 next to the default 919).
PREFIX=SPEC_V1
MODEL=model15_uct_mid_8A
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
  dev4)    DATA=data24_dev_loo_Panc1_200U; EVAL=eval23_HepG2_200U_mini; PREFIX=LOO_Panc1_V1 ;;
  nocpg)   DATA=data24_dev_loo_Panc1_200U; EVAL=eval23_HepG2_200U_mini; PREFIX=ABL_V1; MODEL=model19_uct_mid_8A_no_cpg ;;
  nofire)  DATA=data24_dev_loo_Panc1_200U; EVAL=eval23_HepG2_200U_mini; PREFIX=ABL_V1; MODEL=model16_uct_mid_8A_no_fire_msp ;;
  noacc)   DATA=data24_dev_loo_Panc1_200U; EVAL=eval23_HepG2_200U_mini; PREFIX=ABL_V1; MODEL=model20_uct_mid_8A_cpg_only ;;
  m6acpg)  DATA=data24_dev_loo_Panc1_200U; EVAL=eval23_HepG2_200U_mini; PREFIX=ABL_V1; MODEL=model21_uct_mid_8A_m6a_cpg ;;
  r0_nf)   DATA=data27_mix_GM12878_HepG2_200U;     EVAL=eval26_mix_GM12878_HepG2_200U_mini;     PREFIX=MIXR0_NF; MODEL=model16_uct_mid_8A_no_fire_msp ;;
  r1a_nf)  DATA=data36_mix_bal_GM12878_HepG2_200U; EVAL=eval31_mix_bal_GM12878_HepG2_200U_mini; PREFIX=MIXR1A_NF; MODEL=model16_uct_mid_8A_no_fire_msp ;;
  pair_nf) DATA=data28_GM12878_HepG2_200U;         EVAL=eval24_pair_GM12878_HepG2_200U_mini;    PREFIX=PAIR_NF; MODEL=model16_uct_mid_8A_no_fire_msp ;;
  r1a_nf_b)  DATA=data37_mix_bal_K562_H1_200U; EVAL=eval32_mix_bal_K562_H1_200U_mini; PREFIX=MIXR1A_NF; MODEL=model16_uct_mid_8A_no_fire_msp ;;
  pair_nf_b) DATA=data34_K562_H1_200U;         EVAL=eval22_K562_200U_mini;            PREFIX=PAIR_NF;   MODEL=model16_uct_mid_8A_no_fire_msp ;;
  # Baseline ladder (reports/TODO_extended.md T11), R1a supervision: rung 2 = small MLP on hand-crafted features
  # (model22), rung 4 = DEV4NF's network + those features (model23); _b = K562 + H1
  r1a_mlp)    DATA=data36_mix_bal_GM12878_HepG2_200U; EVAL=eval31_mix_bal_GM12878_HepG2_200U_mini; PREFIX=MIXR1A_MLP; MODEL=model22_mlp_handcrafted ;;
  r1a_mlp_b)  DATA=data37_mix_bal_K562_H1_200U;       EVAL=eval32_mix_bal_K562_H1_200U_mini;       PREFIX=MIXR1A_MLP; MODEL=model22_mlp_handcrafted ;;
  r1a_hc)     DATA=data36_mix_bal_GM12878_HepG2_200U; EVAL=eval31_mix_bal_GM12878_HepG2_200U_mini; PREFIX=MIXR1A_HC;  MODEL=model23_uct_mid_8A_no_fire_msp_handcrafted ;;
  r1a_hc_b)   DATA=data37_mix_bal_K562_H1_200U;       EVAL=eval32_mix_bal_K562_H1_200U_mini;       PREFIX=MIXR1A_HC;  MODEL=model23_uct_mid_8A_no_fire_msp_handcrafted ;;
  # rung 1 = fixed rules, only a per-assay calibration trained (model24, train05: 3 epochs; eval with EPOCH=3)
  r1a_rules)   DATA=data36_mix_bal_GM12878_HepG2_200U; EVAL=eval31_mix_bal_GM12878_HepG2_200U_mini; PREFIX=MIXR1A_RULES; MODEL=model24_rules; TRAIN=train05 ;;
  r1a_rules_b) DATA=data37_mix_bal_K562_H1_200U;       EVAL=eval32_mix_bal_K562_H1_200U_mini;       PREFIX=MIXR1A_RULES; MODEL=model24_rules; TRAIN=train05 ;;
  *) echo "Unknown CELL=${CELL}"; exit 1 ;;
esac
if [ -n "${SEED:-}" ]; then PREFIX="${PREFIX}_seed${SEED}"; fi
echo "Model: $PREFIX $DATA $MODEL seed ${SEED:-919}"
python -u main.py \
  --data_config configs/data/$DATA.yaml \
  --model_config configs/models/$MODEL.yaml \
  --train_config configs/training/${TRAIN:-train04}.yaml \
  --eval_configs configs/evals/$EVAL.yaml \
  --num_workers $(( ${SLURM_CPUS_PER_TASK:-4} - 1 )) \
  --name_prefix $PREFIX \
  --seed ${SEED:-919} \
  ${RESUME:+--resume "$RESUME"}

# Print job completion time
echo "Job finished on $(date)"
