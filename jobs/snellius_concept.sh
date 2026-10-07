#!/bin/bash
#SBATCH --job-name=concept-heads
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/concept-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
# Bound CPU thread pools for small RSA matrices.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${1:-llama}"
if (( $# > 1 )); then echo 'Expected one model argument.' >&2; exit 1; fi
if [[ "$MODEL" == llama ]]; then MODEL=/scratch-shared/jsmidi/models/Llama-3.1-70B-Instruct; fi
RUN_DIR="${RUN_DIR:-outputs/concept_pilot_${SLURM_JOB_ID}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis/concept_pilot_${SLURM_JOB_ID}}"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
srun "$PYTHON" -u scripts/concept_vectors.py --model "$MODEL" \
    --output-dir "$RUN_DIR/discovery" --top-k "${TOP_K:-5}" --layer "${LAYER:-16}" \
    --permutations "${PERMUTATIONS:-199}" --items brick rope bottle spoon \
    --paraphrases 1 2 --audit-paraphrase 0

# Paper-style mean, contrastive adaptation, then a direct intervention on the
# top discovery-ranked head. Audit scores never re-rank heads.
HEAD_ARTIFACT=$("$PYTHON" -c 'import csv,sys; r=next(csv.DictReader(open(sys.argv[1]))); print("head_l"+r["Layer"]+"_h"+r["Head"])' "$RUN_DIR/discovery/head_ranking.csv")
for candidate in creative_mean contrast "$HEAD_ARTIFACT"; do
    srun "$PYTHON" -u scripts/activation_steer.py generate \
        --model "$MODEL" --artifact "$RUN_DIR/discovery/$candidate.pt" \
        --task 'Alternative Uses Task' --items book fork paperclip towel can --paraphrases 0 \
        --split validation --alphas "${ALPHA:-1}" --repeats "${REPEATS:-2}" \
        --random-vectors 2 --scope "${SCOPE:-prefill}" --max-tokens 800 \
        --output-dir "$RUN_DIR/$candidate"
    srun "$PYTHON" -u scripts/evaluate_aut.py --inputs "$RUN_DIR/$candidate" \
        --ratings-only --output-dir "$ANALYSIS_DIR/$candidate"
    if [[ "${SCORE:-1}" == 1 ]]; then
        srun "$PYTHON" -u scripts/evaluate_aut.py --inputs "$RUN_DIR/$candidate" \
            --output-dir "$ANALYSIS_DIR/$candidate"
        ALPHA_LABEL=$("$PYTHON" -c 'import sys; print(format(float(sys.argv[1]),"g"))' "${ALPHA:-1}")
        srun "$PYTHON" scripts/analyze_effects.py "$ANALYSIS_DIR/$candidate/aut_responses.csv" \
            --metric Semantic_Distance_Proxy --treatment "Standard_add_$ALPHA_LABEL" \
            --control Standard --output "$ANALYSIS_DIR/$candidate/addition_effect.csv"
        srun "$PYTHON" scripts/analyze_effects.py "$ANALYSIS_DIR/$candidate/aut_responses.csv" \
            --metric Semantic_Distance_Proxy --treatment Creative \
            --controls Standard Effective Conventional Boring \
            --output "$ANALYSIS_DIR/$candidate/creative_vs_controls.csv"
    fi
done
echo "Concept-head pilot complete: $RUN_DIR and $ANALYSIS_DIR"
