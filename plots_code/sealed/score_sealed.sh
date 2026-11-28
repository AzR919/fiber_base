#!/bin/bash
# Score the sealed test once (reports/sealed_test_protocol.md, section 7 step 5). Run on the fir login node from the
# repo root, after every training run has finished and the dry run is done:
#   START=<NN> [DATE=YYYY-MM-DD] [DRY=1] bash plots_code/sealed/score_sealed.sh
# START = the first free job index of the day (count across both clusters). DRY=1 prints the jobs without submitting.
#
# Per model run (common.py SETUPS['sealed']: R1a-S, Pair-S, MLP-S x 3 seeds, Rules-S, DEV4NF x 2, R1a-A, R1a-B x 3):
#   eval_sealed_pair_<run>   eval33 (pair, all chr19 tiles); R1a-S with --mixture_reference (the P1 floor)
#   eval_sealed_mix_<run>    eval34 (mixture, all chr19 tiles); R1a-S also eval35 / eval36 (random split), eval37 /
#                            eval38 (unshuffled / shuffled, mini tiles). No baseline (no readout needs it).
#   molecules_sealed_ctcf_poised_<run>, molecules_sealed_xi_<run>   molecules_sealed.sh: molecule_predict.py on
#                            HCT116 / MCF-7 fibers at the chr19 CTCF sites (CpG flips meth, unmeth, meth_distal,
#                            unmeth_distal) and poised tiles (flips meth, unmeth, meth_distal); on the XI cells' fibers
#                            at the chrX + chr21 promoters
# Once: sealed_ceiling (replicate_ceiling.py on eval33), then sealed_analysis (analyze.sh, SETUP=sealed) after all.
# fir sizes: 2g.20gb evals with 3 CPUs, 1g.10gb molecules with 1 CPU / 35 GB (CLAUDE.md).
set -euo pipefail
cd /home/azr/projects/def-maxwl/azr/code/fiber_base
: "${START:?set START=<first free job index of the day>}"
DATE=${DATE:-$(date +%F)}
E=configs/evals
T_PAIR=0-1:45:00; T_MIX=0-2:00:00; T_MIX_R1A=0-2:45:00; T_CTCF_POISED=0-2:30:00; T_XI=0-1:30:00

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate
RUNS=$(python - <<'PY'
import sys; sys.path.insert(0, "plots_code/sealed")
from common import get_setup, runs, run_name
S = get_setup("sealed")
short = {"R1a-S": "r1a", "Pair-S": "pair", "MLP-S": "mlp", "Rules-S": "rules", "DEV4NF": "dev4nf", "R1a-A": "r1a_a",
         "R1a-B": "r1a_b"}
for label, key, seed in runs(S):
    print(f"{label}|{short[label]}_s{seed}|{run_name(key, seed)}")
PY
)

i=$START
IDS=()
sub() {   # sub <name> <sbatch args...>
  local name=$(printf "%s_%02d_%s" "$DATE" "$i" "$1"); shift
  i=$((i + 1))
  if [ "${DRY:-0}" = 1 ]; then echo "DRY $name: $*"; return; fi
  local j
  j=$(sbatch --parsable --job-name="$name" "$@")
  IDS+=("$j")
  echo "$name $j"
}

while IFS="|" read -r label tag run; do
  if [ "$label" = "R1a-S" ]; then
    sub "eval_sealed_pair_$tag" --cpus-per-task=3 --time=$T_PAIR \
      --export=ALL,EVAL_SET=one,RUN="$run",MIXREF=1,CONFIGS="$E/eval33_sealed_pair_HCT116_MCF_7_chr19.yaml" \
      slurm_batch_command_eval.sh
    sub "eval_sealed_mix_$tag" --cpus-per-task=3 --time=$T_MIX_R1A \
      --export=ALL,EVAL_SET=one,RUN="$run",BASELINE=0,CONFIGS="$E/eval34_sealed_mix_bal_HCT116_MCF_7_chr19.yaml $E/eval35_sealed_split_HCT116_chr19_mini.yaml $E/eval36_sealed_split_MCF_7_chr19_mini.yaml $E/eval37_sealed_pair_HCT116_MCF_7_chr19_mini.yaml $E/eval38_sealed_shuffle_pair_HCT116_MCF_7_chr19_mini.yaml" \
      slurm_batch_command_eval.sh
  else
    sub "eval_sealed_pair_$tag" --cpus-per-task=3 --time=$T_PAIR \
      --export=ALL,EVAL_SET=one,RUN="$run",CONFIGS="$E/eval33_sealed_pair_HCT116_MCF_7_chr19.yaml" \
      slurm_batch_command_eval.sh
    sub "eval_sealed_mix_$tag" --cpus-per-task=3 --time=$T_MIX \
      --export=ALL,EVAL_SET=one,RUN="$run",BASELINE=0,CONFIGS="$E/eval34_sealed_mix_bal_HCT116_MCF_7_chr19.yaml" \
      slurm_batch_command_eval.sh
  fi
  sub "molecules_sealed_ctcf_poised_$tag" --cpus-per-task=1 --mem=35G --time=$T_CTCF_POISED \
    --export=ALL,RUN="$run",PARTS="ctcf poised" plots_code/sealed/molecules_sealed.sh
  sub "molecules_sealed_xi_$tag" --cpus-per-task=1 --mem=35G --time=$T_XI \
    --export=ALL,RUN="$run",PARTS="xi" plots_code/sealed/molecules_sealed.sh
done <<< "$RUNS"

sub "sealed_ceiling" --time=0-1:30:00 \
  --export=ALL,CONFIGS="$E/eval33_sealed_pair_HCT116_MCF_7_chr19.yaml" slurm_batch_command_replicates.sh

if [ "${DRY:-0}" = 1 ]; then
  sub "sealed_analysis" --export=ALL,SETUP=sealed plots_code/sealed/analyze.sh
else
  DEP=$(IFS=:; echo "${IDS[*]}")
  sub "sealed_analysis" --dependency=afterany:$DEP --export=ALL,SETUP=sealed plots_code/sealed/analyze.sh
fi
echo "submitted indices $START..$((i - 1)) for $DATE"
