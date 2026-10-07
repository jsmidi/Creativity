#!/bin/bash
#SBATCH --job-name=greedy-aut
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/greedy-aut-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${1:-models/Llama-3.1-8B-Instruct}"
if [[ "$MODEL" == llama ]]; then MODEL=models/Llama-3.1-8B-Instruct; fi
RUN_DIR="outputs/greedy_aut_${SLURM_JOB_ID}"
ANALYSIS_DIR="analysis/greedy_aut_${SLURM_JOB_ID}"
# Freeze executable sources at job start, including uncommitted changes.
mkdir -p "$RUN_DIR/code"
cp scripts/*.py "$RUN_DIR/code/"
srun "$PYTHON" -u "$RUN_DIR/code/run_greedy_task.py" --task aut --model "$MODEL" \
    --repeats "${REPEATS:-60}" --mi-repeats "${MI_REPEATS:-1}" \
    --run-dir "$RUN_DIR" --analysis-dir "$ANALYSIS_DIR"
