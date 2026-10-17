#!/bin/bash

#SBATCH --job-name=2026-10-17_00_data_setup
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ara199@sfu.ca

# Sets up all data: ENCODE tracks + configs/metapaths.yaml (data_prep/encode_spec.yaml), then the region files
# (data_prep/regions_spec.yaml). Submit from the repo root: sbatch data_prep/slurm_setup_data.sh
# Safe to resubmit: finished work is skipped and partial downloads resume.

set -euo pipefail

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

for url in https://www.encodeproject.org/ https://github.com/; do
    if ! curl -s -o /dev/null --max-time 30 "$url"; then
        echo "ERROR: $(hostname) cannot reach $url (no internet on this node?)" >&2
        exit 1
    fi
done

python -u data_prep/setup_data.py --workers 4

echo "Job finished on $(date)"
