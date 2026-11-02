#!/usr/bin/env bash
#SBATCH --job-name=pm-resid-plots
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pm-resid-plots-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project root}"
export MPLBACKEND=Agg OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
PYTHON="${PYTHON:-.venv/bin/python}"
srun "$PYTHON" -u scripts_new/mech_interp/plot_response_residual_differences.py "$@"
