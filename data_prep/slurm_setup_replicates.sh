#!/bin/bash

#SBATCH --job-name=2026-11-11_00_setup_replicates
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-2:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

# Replicate-level ENCODE tracks for the noise ceiling (data_prep/setup_replicates.py): two per-replicate
# fold-change bigWigs per locked track, downloaded, cropped to chr21, full files deleted (~100 GB transferred,
# ~10 GB peak disk, ~2 GB kept). Submit from the repo root: sbatch data_prep/slurm_setup_replicates.sh
# Safe to resubmit: finished files are skipped and partial downloads resume.

set -euo pipefail

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

echo "Job started on $(date)"
echo "Running on host $(hostname)"

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

if ! curl -s -o /dev/null --max-time 30 https://www.encodeproject.org/; then
    echo "ERROR: $(hostname) cannot reach encodeproject.org (no internet on this node?)" >&2
    exit 1
fi

# REP_ARGS overrides the arguments, e.g. the sealed test (reports/sealed_test_protocol.md):
#   sbatch --export=ALL,REP_ARGS="--cells HCT116 MCF-7 --chroms chr19 chr21 --workers 4" data_prep/slurm_setup_replicates.sh
python -u data_prep/setup_replicates.py ${REP_ARGS:---workers 4}

echo "Job finished on $(date)"
