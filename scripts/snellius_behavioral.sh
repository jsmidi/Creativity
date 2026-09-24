#!/bin/bash
#SBATCH --job-name=llama-creativity
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=creativity-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Submit from the project root. Override MODEL or REPEATS before sbatch if needed.
MODEL="${MODEL:-models/Llama-3.1-8B-Instruct}"
REPEATS="${REPEATS:-5}"
PYTHON="${PYTHON:-.venv-snellius/bin/python}"

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
for task in "Divergent Association Task" "Alternative Uses Task"; do
    srun "$PYTHON" -u scripts/generate.py \
        --provider local --model "$MODEL" --task "$task" \
        --conditions Standard Conventional Effective Boring Creative \
        --repeats "$REPEATS" --paraphrases 0 \
        --temperature 0.7 --max-tokens 800
done
