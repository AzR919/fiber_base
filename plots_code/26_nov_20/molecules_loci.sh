#!/bin/bash

#SBATCH --job-name=2026-11-18_00_molecules_loci
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=36G
#SBATCH --time=0-0:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

# Per-bp arrays for the Nov_20_figs.md locus plots: molecule_predict.py on the loci picked by select_loci.py, for
# DEV4 and DEV4NF in both training seeds, GM12878 only. ~1 min per run (model load dominates).
#   sbatch plots_code/26_nov_20/molecules_loci.sh   (from the repo root, after select_loci.py)
# TARGETS="loci_ctcf" (export) limits it to one target set; 18 Nov rerun of the CTCF loci after the unbound site
# was replaced (job 2026-11-18_01_molecules_loci_ctcf).

set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
echo "Job started on $(date) on $(hostname)"
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

RUNS=(
  LOO_Panc1_V1_data24_dev_loo_Panc1_200U_model15_uct_mid_8A_train04                 # DEV4
  LOO_Panc1_V1_seed2_data24_dev_loo_Panc1_200U_model15_uct_mid_8A_train04           # DEV4 seed 2
  ABL_V1_data24_dev_loo_Panc1_200U_model16_uct_mid_8A_no_fire_msp_train04           # DEV4NF
  ABL_V1_seed2_data24_dev_loo_Panc1_200U_model16_uct_mid_8A_no_fire_msp_train04     # DEV4NF seed 2
)
for RUN in "${RUNS[@]}"; do
  for T in ${TARGETS:-loci_ctcf loci_tss}; do
    python -u molecule_predict.py --run "$RUN" --samples GM12878 \
      --targets /home/azr/projects/def-maxwl/azr/code/fiber_base/plots/26_nov_20/loci/$T.bed --save_arrays 100
  done
done

echo "Job finished on $(date)"
