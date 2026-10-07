#!/bin/bash
#SBATCH --job-name=temp-mi
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/temp-mi-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON="${PYTHON:-.venv/bin/python}"
srun "$PYTHON" -u "${RUN_ROOT:?}/code/run_temperature_mechanistic.py" \
    --phase "${PHASE:?}" --index "${SLURM_ARRAY_TASK_ID:-0}" \
    --model "${MODEL:?}" --run-root "$RUN_ROOT" --analysis-root "${ANALYSIS_ROOT:?}" --repeats 60
