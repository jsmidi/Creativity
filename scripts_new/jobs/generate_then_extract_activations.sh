#!/usr/bin/env bash
#SBATCH --job-name=pm-gen-acts
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pm-gen-acts-%j.log

# End-to-end convenience job. Configure via environment variables; see jobs/README.md.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project root}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PYTHON="${PYTHON:-.venv/bin/python}"

: "${MODEL:?Set MODEL to the local model checkpoint path}"
: "${OUTPUT:?Set OUTPUT to the generated response CSV path}"
: "${TASKS:?Set TASKS to space-separated task codes, such as 'aut'}"
read -r -a task_args <<< "$TASKS"
read -r -a condition_args <<< "${CONDITIONS:-creative non_creative}"
REPEATS="${REPEATS:-1}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
POSITIONS="${POSITIONS:-response last-prompt}"
read -r -a position_args <<< "$POSITIONS"
HEAD_OUTPUT="${HEAD_OUTPUT:-${OUTPUT%.csv}_head_activations.pt}"
RESIDUAL_OUTPUT="${RESIDUAL_OUTPUT:-${OUTPUT%.csv}_residual_activations.pt}"

srun "$PYTHON" -u scripts_new/generation/generate_responses.py \
  --model "$MODEL" --tasks "${task_args[@]}" --conditions "${condition_args[@]}" \
  --repeats "$REPEATS" --seed "$SEED" --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --output "$OUTPUT"

# Extraction replays the saved prompt and response token IDs. Generation itself
# does not currently cache the requested intermediate activations.
srun "$PYTHON" -u scripts_new/mech_interp/extract_response_activations.py \
  "$OUTPUT" --model "$MODEL" --output "$HEAD_OUTPUT" --positions "${position_args[@]}"
srun "$PYTHON" -u scripts_new/mech_interp/extract_response_residuals.py \
  "$OUTPUT" --model "$MODEL" --output "$RESIDUAL_OUTPUT" --positions "${position_args[@]}"
