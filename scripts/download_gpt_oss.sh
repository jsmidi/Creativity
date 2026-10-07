#!/bin/bash
# Run from the project root on a node with internet access.
set -euo pipefail
"${HF:-.venv/bin/hf}" download openai/gpt-oss-20b \
    --local-dir "${MODEL:-models/gpt-oss-20b}" \
    --include '*.json' --include '*.safetensors' --include '*.jinja' \
    --include '*.txt' --include '*.tiktoken' --exclude 'original/*'
