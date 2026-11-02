#!/usr/bin/env bash
#SBATCH --job-name=pm-evaluate
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pm-evaluate-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project root}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PYTHON="${PYTHON:-.venv/bin/python}"

srun "$PYTHON" -u scripts_new/evaluation/evaluate_pm_tasks.py "$@"
