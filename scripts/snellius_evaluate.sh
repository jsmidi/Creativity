#!/bin/bash
#SBATCH --job-name=llama-evaluation
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/evaluation-%j.log

# Submit from the project root; create logs/ before calling sbatch.
# Optional arguments: DAT_CSV AUT_CSV. Otherwise select the latest of each.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the project root}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

PYTHON="${PYTHON:-.venv/bin/python}"
INPUT_DIR="${INPUT_DIR:-outputs/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/llama_evaluation_${SLURM_JOB_ID}}"

if [[ "${1:-}" == --sweep ]]; then
    if (( $# != 2 )); then echo 'Usage: sbatch scripts/snellius_evaluate.sh --sweep OUTPUT_ROOT' >&2; exit 1; fi
    srun "$PYTHON" -u scripts/evaluate_sweep.py --inputs "$2" --output-dir "$OUTPUT_DIR"
    exit 0
fi

if (( $# != 0 && $# != 2 )); then
    echo "Usage: sbatch scripts/snellius_evaluate.sh [DAT_CSV AUT_CSV]" >&2
    exit 1
fi

latest_csv() {
    local task="$1"
    local -a files
    # Filenames contain UTC timestamps, so lexical order gives the latest run.
    local LC_ALL=C
    shopt -s nullglob
    files=("$INPUT_DIR"/results_"$task"_*.csv)
    if (( ${#files[@]} == 0 )); then
        echo "No $task CSV found in $INPUT_DIR" >&2
        return 1
    fi
    printf '%s\n' "${files[${#files[@]}-1]}"
}

if (( $# == 2 )); then
    DAT_CSV="$1"
    AUT_CSV="$2"
else
    DAT_CSV="$(latest_csv divergent_association_task)"
    AUT_CSV="$(latest_csv alternative_uses_task)"
fi

for file in "$DAT_CSV" "$AUT_CSV" databases/dat.py databases/words.txt \
    databases/glove.840B.300d.txt aut_quality_scored_all.csv; do
    if [[ ! -r "$file" ]]; then
        echo "Required input is missing or unreadable: $file" >&2
        exit 1
    fi
done

echo "DAT input: $DAT_CSV"
echo "AUT input: $AUT_CSV"
echo "Output directory: $OUTPUT_DIR"
srun "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: AUT requires a GPU for this job"; print("GPU:", torch.cuda.get_device_name(0))'

echo "Evaluating DAT with official GloVe scoring (CPU)..."
srun "$PYTHON" -u scripts/evaluate_dat.py \
    --inputs "$DAT_CSV" \
    --scorer official \
    --official-code databases/dat.py \
    --dictionary databases/words.txt \
    --glove databases/glove.840B.300d.txt \
    --output-dir "$OUTPUT_DIR/dat"

echo "Evaluating AUT with MPNet embeddings (GPU)..."
srun "$PYTHON" -u scripts/evaluate_aut.py \
    --inputs "$AUT_CSV" \
    --norm-db aut_quality_scored_all.csv \
    --output-dir "$OUTPUT_DIR/aut"

echo "DAT and AUT evaluation complete: $OUTPUT_DIR"
