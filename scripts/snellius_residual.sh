#!/bin/bash
#SBATCH --job-name=residual-pilot
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/residual-%j.log

# A validation pilot, not a frozen confirmatory test. Submit from project root.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: sbatch scripts/snellius_residual.sh [llama|LOCAL_LLAMA_PATH]'
    echo 'Settings: LAYER=16 ALPHA=1 REPEATS=3 RANDOM_VECTORS=2 SCOPE=prefill SCORE=1'
    exit 0
fi
if (( $# > 1 )); then echo 'Expected at most one model argument.' >&2; exit 1; fi
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${1:-llama}"
if [[ "$MODEL" == llama ]]; then MODEL=models/Llama-3.1-8B-Instruct; fi
LAYER="${LAYER:-16}"
ALPHA="${ALPHA:-1}"
ALPHA_LABEL=$("$PYTHON" -c 'import sys; print(format(float(sys.argv[1]), "g"))' "$ALPHA")
SCOPE="${SCOPE:-prefill}"
RUN_DIR="${RUN_DIR:-outputs/residual_pilot_${SLURM_JOB_ID}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis/residual_pilot_${SLURM_JOB_ID}}"
ARTIFACT="$RUN_DIR/aut_l${LAYER}.pt"

"$PYTHON" -c 'import json, sys, torch; c=json.load(open(sys.argv[1]+"/config.json")); assert c["model_type"] == "llama", "This pilot adapter is validated for Llama only"; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' "$MODEL"
echo "Model=$MODEL layer=$LAYER alpha=$ALPHA scope=$SCOPE"
echo "Artifacts/responses: $RUN_DIR; analysis: $ANALYSIS_DIR"

srun "$PYTHON" -u scripts/activation_steer.py extract \
    --model "$MODEL" --artifact "$ARTIFACT" --layer "$LAYER" \
    --tasks 'Alternative Uses Task' --items brick rope bottle spoon \
    --paraphrases 1 --baseline Standard

srun "$PYTHON" -u scripts/activation_steer.py generate \
    --model "$MODEL" --artifact "$ARTIFACT" --task 'Alternative Uses Task' \
    --items book fork paperclip towel can --paraphrases 0 --split validation \
    --alphas "$ALPHA" --repeats "${REPEATS:-3}" \
    --random-vectors "${RANDOM_VECTORS:-2}" --scope "$SCOPE" \
    --max-tokens 800 --output-dir "$RUN_DIR/responses"

# Always export format audits and blinded sheets, without an embedding download.
srun "$PYTHON" -u scripts/evaluate_aut.py --inputs "$RUN_DIR/responses" \
    --ratings-only --output-dir "$ANALYSIS_DIR"
if [[ "${SCORE:-1}" == 1 ]]; then
    # Requires all-mpnet-base-v2 cached, as in the existing AUT scoring job.
    srun "$PYTHON" -u scripts/evaluate_aut.py --inputs "$RUN_DIR/responses" \
        --output-dir "$ANALYSIS_DIR"
    srun "$PYTHON" -u scripts/analyze_effects.py "$ANALYSIS_DIR/aut_responses.csv" \
        --metric Semantic_Distance_Proxy --treatment "Standard_add_${ALPHA_LABEL}" \
        --control Standard --output "$ANALYSIS_DIR/addition_effect.csv" \
        > "$ANALYSIS_DIR/addition_effect.txt"
    srun "$PYTHON" scripts/analyze_effects.py "$ANALYSIS_DIR/aut_responses.csv" \
        --metric Semantic_Distance_Proxy --treatment Creative \
        --controls Standard Effective Conventional Boring \
        --output "$ANALYSIS_DIR/creative_vs_controls.csv"
fi
echo "Pilot complete. Inspect validity, truncation, random controls and blinded ratings before interpreting scores."
