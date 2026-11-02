# Slurm jobs for `scripts_new`

Submit these jobs from the repository root so that relative model, data, and output paths resolve correctly. Logs go to the existing `logs/` directory. The generation and activation jobs request one A100; CPU evaluation and plotting use the `rome` partition. Change the `#SBATCH --partition` line if your cluster account uses different partition names.

All scripts pass their arguments through to the corresponding Python CLI. The Python environment defaults to `.venv/bin/python`; override it with `PYTHON=/path/to/python sbatch ...` if needed. Keep model checkpoints and embedding resources available locally because the jobs set Hugging Face offline mode.

## 1. Generate responses

```bash
sbatch scripts_new/jobs/generate_responses.sh \
  --model models/Meta-Llama-3.1-8B-Instruct \
  --tasks aut --conditions creative non_creative \
  --repeats 5 --seed 42 --output outputs/aut_creative_responses.csv
```

Use a local model path for `--model`. `--conditions` accepts the configured labels, including `creative` and `non_creative`. The generated CSV retains prompt and response text, their token IDs, run/response IDs, model and generation settings, and completion reason for later replay and auditing.

## 2. Behavioral metrics and plots

```bash
sbatch scripts_new/jobs/evaluate_responses.sh outputs/aut_creative_responses.csv \
  --tasks aut --output-dir analysis/aut_behavior
```

The evaluator supports `aut`, `dat`, `cdat`, and `drat`. It writes scored response CSVs, run-level aggregates, and plots under `<output-dir>/<task>/`. It loads `sentence-transformers/all-mpnet-base-v2` by default, so make that model available locally or pass `--embedding-model /local/path`. DAT additionally requires `--glove` and `--dictionary`; CDAT and DRAT require `--noun-vocabulary`. For example, pass multiple inputs before the options to aggregate files together.

## 3–4. Extract attention-head and residual-stream activations

```bash
sbatch scripts_new/jobs/extract_attention_activations.sh \
  outputs/aut_creative_responses.csv --model models/Meta-Llama-3.1-8B-Instruct \
  --output outputs/aut_heads.pt --positions response last-prompt

sbatch scripts_new/jobs/extract_residual_activations.sh \
  outputs/aut_creative_responses.csv --model models/Meta-Llama-3.1-8B-Instruct \
  --output outputs/aut_residuals.pt --positions response last-prompt
```

`response` pools generated answer-token activations; `last-prompt` captures the final formatted prompt token. Extraction uses the saved token IDs when available. Generate only the conditions you plan to compare to avoid extracting unneeded rows. For an older CSV with many conditions, the current extractors process every row; the plotting scripts then select the two condition labels.

Use `--positions response-predictor` to instead pool the causal states that predict each answer token. This sequence includes the last prompt position for the first answer token and the preceding answer-token positions for later tokens. The extractor also removes any configured EOS token IDs before pooling.

## 5. Optional single-allocation generation plus extraction

This job generates the CSV and then extracts both activation types within the same GPU allocation. It replays the saved token sequences after generation; it does not capture activations inline during decoding. Separate jobs are easier to rerun if a later analysis step fails.

```bash
MODEL=models/Meta-Llama-3.1-8B-Instruct \
TASKS=aut CONDITIONS="creative non_creative" REPEATS=5 SEED=42 \
OUTPUT=outputs/aut_creative_responses.csv \
POSITIONS="response last-prompt" \
sbatch scripts_new/jobs/generate_then_extract_activations.sh
```

Optional variables: `HEAD_OUTPUT`, `RESIDUAL_OUTPUT`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_P`, and `PYTHON`. `TASKS` and `CONDITIONS` are space-separated lists.

## 6. Compare and plot activations

```bash
sbatch scripts_new/jobs/plot_attention_head_differences.sh outputs/aut_heads.pt \
  --output-dir analysis/aut_head_differences --creative creative \
  --non-creative non_creative --position response

sbatch scripts_new/jobs/plot_residual_differences.sh outputs/aut_residuals.pt \
  --output-dir analysis/aut_residual_differences --creative creative \
  --non-creative non_creative --position response
```

Repeat with `--position last-prompt` to plot the prompt-end state. Difference plots average repeats within each item, pair items that have both conditions, and then average the creative-minus-non-creative differences across items. For DAT, which has a blank item field, the scripts pair matching repeats within a run instead. Attention-head and residual outputs include absolute and relative shift summaries. For the optional RSA analysis, use `plot_attention_head_rsa.sh` with the head artifact; it defaults to within-item centering and compares creative with non_creative. RSA answers a different question from the direct condition-difference plots.
