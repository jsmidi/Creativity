#!/usr/bin/env bash
#SBATCH --job-name=pm-head-acts
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pm-head-acts-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project root}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PYTHON="${PYTHON:-.venv/bin/python}"

srun "$PYTHON" -u scripts_new/mech_interp/extract_response_activations.py "$@"
