#!/bin/bash

#SBATCH --job-name=2026-10-16_00_encode_data_setup
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

# Downloads the ENCODE tracks in data_prep/encode_spec.yaml and regenerates configs/metapaths.yaml.
# Submit from the repo root: sbatch data_prep/slurm_setup_data.sh
# Safe to resubmit: verified files are skipped and partial downloads resume.

set -euo pipefail

SCRIPT_DIR="/home/azr/projects/def-maxwl/azr/code/fiber_base"
cd "$SCRIPT_DIR"

echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Working directory is $(pwd)"

source /home/azr/projects/def-maxwl/azr/misc/menv/bin/activate

if ! curl -s -o /dev/null --max-time 30 https://www.encodeproject.org/; then
    echo "ERROR: $(hostname) cannot reach www.encodeproject.org (no internet on this node?)" >&2
    exit 1
fi

python data_prep/setup_data.py --spec data_prep/encode_spec.yaml --workers 4

echo "Job finished on $(date)"
