#!/bin/bash
# Sealed test molecule runs of one model (reports/sealed_test_protocol.md, section 7 step 5), one or more parts in a row:
#   ctcf    HCT116 / MCF-7 fibers at the chr19 CTCF sites, CpG flips meth, unmeth, meth_distal, unmeth_distal
#   poised  HCT116 / MCF-7 fibers at the chr19 poised tiles, CpG flips meth, unmeth, meth_distal
#   xi      the XI cells' fibers (HG008_3E4, GM25455_LCL, HEK293T_200U) at the chrX + chr21 promoters
#   sbatch --job-name=<name> --time=<...> --export=ALL,RUN=<training run name>,PARTS="ctcf poised" \
#          plots_code/sealed/molecules_sealed.sh
# score_sealed.sh submits it. On dev, one part took 10-45 min on a 1g.10gb slice (fiber reading dominates).

#SBATCH --job-name=molecules_sealed
#SBATCH --account=def-maxwl_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=36G
#SBATCH --time=0-2:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
echo "Job started on $(date) on $(hostname)"
source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
: "${RUN:?set RUN=<training run name>}"
T=/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/regions/targets
for part in ${PARTS:?set PARTS=\"ctcf poised xi\"}; do
  echo "== $part $(date)"
  case $part in
    ctcf)   python -u molecule_predict.py --run "$RUN" --samples HCT116_200U MCF_7_200U --targets "$T/ctcf_sites_chr19.bed" \
              --perturb meth unmeth meth_distal unmeth_distal ;;
    poised) python -u molecule_predict.py --run "$RUN" --samples HCT116_200U MCF_7_200U --targets "$T/poised_tiles_chr19.bed" \
              --perturb meth unmeth meth_distal ;;
    xi)     python -u molecule_predict.py --run "$RUN" --samples HG008_3E4 GM25455_LCL HEK293T_200U \
              --targets "$T/tss_chrX_chr21.bed" ;;
    *) echo "unknown part $part"; exit 1 ;;
  esac
done
echo "Job finished on $(date)"
