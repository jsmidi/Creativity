#!/bin/bash
#SBATCH --job-name=creativity
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/creativity-%A_%a.log

set -euo pipefail
if (( $# > 1 )) || [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: sbatch scripts/snellius_behavioral.sh [llama|gpt-oss|LOCAL_MODEL_PATH]"
    echo "Optional environment settings: REPEATS (per item), GREEDY_REPEATS, TEMPERATURES, TOP_P, TOP_K, MAX_TOKENS, PYTHON"
    if (( $# > 1 )); then exit 1; else exit 0; fi
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the project root with sbatch}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Positional model takes precedence over the legacy MODEL environment setting.
MODEL="${1:-${MODEL:-llama}}"
case "$MODEL" in
    llama) MODEL="/scratch-shared/jsmidi/models/Llama-3.1-70B-Instruct" ;;
    gpt-oss) MODEL="models/gpt-oss-20b" ;;
esac
if [[ ! -f "$MODEL/config.json" ]]; then
    echo "Local model config not found: $MODEL/config.json" >&2
    exit 1
fi
REPEATS="${REPEATS:-30}"
read -r -a TEMPERATURE_GRID <<< "${TEMPERATURES:-0 0.7 1.0 1.3}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX >= ${#TEMPERATURE_GRID[@]} )); then echo "Array index exceeds temperature grid" >&2; exit 1; fi
TEMPERATURE="${TEMPERATURE_GRID[$INDEX]}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/temperature_sweep_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/t${INDEX}}"
PYTHON="${PYTHON:-.venv/bin/python}"
if "$PYTHON" -c 'import sys; sys.exit(float(sys.argv[1]) != 0)' "$TEMPERATURE"; then
    REPEATS="${GREEDY_REPEATS:-1}"
fi
MODEL_TYPE=$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["model_type"])' "$MODEL/config.json")
case "$MODEL_TYPE" in
    gpt_oss) DEFAULT_MAX_TOKENS=4096 ;;
    *) DEFAULT_MAX_TOKENS=800 ;;
esac
MAX_TOKENS="${MAX_TOKENS:-$DEFAULT_MAX_TOKENS}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
echo "Temperature=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K output=$OUTPUT_ROOT"
echo "Model: $MODEL; repeats: $REPEATS; max tokens: $MAX_TOKENS; reasoning effort (GPT-OSS): $REASONING_EFFORT"

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
TASKS=("Divergent Association Task" "Alternative Uses Task")
if [[ "${INCLUDE_ASSOCIATION:-0}" == 1 ]]; then
    TASKS+=("Conditional Divergent Association Task" "Divergent Remote Association Test")
fi
for task in "${TASKS[@]}"; do
    srun "$PYTHON" -u scripts/generate.py \
        --provider local --model "$MODEL" --task "$task" \
        --conditions Standard Conventional Effective Boring Creative \
        --repeats "$REPEATS" --paraphrases 0 --randomize \
        --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
        --output-root "$OUTPUT_ROOT" --max-tokens "$MAX_TOKENS" \
        --reasoning-effort "$REASONING_EFFORT"
done
