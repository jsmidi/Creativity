#!/bin/bash
#SBATCH --job-name=temp-qwen7b
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-11%4
#SBATCH --output=logs/temp-qwen7b-%A_%a.log

# One task/temperature per array entry; behavioral generation and scoring.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="models/Qwen2.5-7B-Instruct"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to a prepared directory containing frozen code}"
CODE="$RUN_ROOT/code"
INDEX="${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
if (( INDEX < 0 || INDEX > 11 )); then echo 'Array index must be 0 through 11' >&2; exit 1; fi
TASK_NAMES=('Alternative Uses Task' 'Divergent Association Task' 'Conditional Divergent Association Task' 'Divergent Remote Association Test')
TASK_KEYS=(aut dat cdat drat)
TEMPERATURES=(0 0.7 1)
TASK_INDEX=$((INDEX / 3))
TEMP_INDEX=$((INDEX % 3))
TASK="${TASK_NAMES[$TASK_INDEX]}"
KEY="${TASK_KEYS[$TASK_INDEX]}"
TEMP="${TEMPERATURES[$TEMP_INDEX]}"
REPEATS=60
if [[ "$TEMP" == 0 ]]; then REPEATS=1; fi
RAW_ROOT="$RUN_ROOT/t${TEMP}/$KEY"
ANALYSIS_ROOT="analysis/${RUN_ROOT##*/}/t${TEMP}/$KEY"
mkdir -p "$RAW_ROOT"
echo "Task=$TASK temperature=$TEMP repeats=$REPEATS model=$MODEL"
echo "Frozen sources=$CODE raw=$RAW_ROOT analysis=$ANALYSIS_ROOT"
srun "$PYTHON" -u "$CODE/generate.py"     --provider local --model "$MODEL" --task "$TASK"     --conditions Standard Conventional Effective Boring Creative     --repeats "$REPEATS" --paraphrases 0 --randomize --seed 42     --temperature "$TEMP" --top-p 1 --top-k 0 --max-tokens 800     --output-root "$RAW_ROOT"
if [[ "$KEY" == aut ]]; then
    srun "$PYTHON" -u "$CODE/evaluate_aut.py" --inputs "$RAW_ROOT"         --norm-db aut_quality_scored_all.csv --output-dir "$ANALYSIS_ROOT"
    METRIC=Semantic_Distance_Proxy
elif [[ "$KEY" == dat ]]; then
    srun "$PYTHON" -u "$CODE/evaluate_dat.py" --inputs "$RAW_ROOT"         --scorer official --official-code databases/dat.py         --dictionary databases/words.txt --glove databases/glove.840B.300d.txt         --output-dir "$ANALYSIS_ROOT"
    METRIC=DAT_GloVe_Score
else
    srun "$PYTHON" -u "$CODE/evaluate_association.py" --task "$KEY"         --inputs "$RAW_ROOT" --output-dir "$ANALYSIS_ROOT" --device cuda
    if [[ "$KEY" == cdat ]]; then METRIC=CDAT_Novelty; else METRIC=DRAT_Score; fi
fi
srun "$PYTHON" -u "$CODE/analyze_effects.py" "$ANALYSIS_ROOT/${KEY}_responses.csv"     --metric "$METRIC" --treatment Creative     --controls Standard Effective Conventional Boring     --output "$ANALYSIS_ROOT/creative_vs_controls.csv"
echo "Complete: $TASK T=$TEMP. CDAT effects require checking cdat_gates.csv."
