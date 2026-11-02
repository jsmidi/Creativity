# New generation pipeline

Generation and activation extraction are separate. The generation CSV stores
the plain prompt text, its exact formatted prompt token IDs, and the exact
generated response token IDs. It also records `top_p`, maximum new tokens,
sampling mode, finish reason, the resolved model revision when available, and
a SHA-256 fingerprint of local weight shards plus model/tokenizer configuration
files. Computing the fingerprint reads those local files once at model load.
The activation command
loads the model and CSV later, runs a teacher-forced forward pass, and mean-pools
per-head outputs over answer tokens. It includes the first answer token and
excludes prompt positions and configured terminal EOS tokens. `response`
captures states at answer-token positions. `response-predictor` instead pools
the causal positions that predict each answer token, from the final prompt
token through the position before the final answer token. `last-prompt` is the
first of those predictor positions on its own.

```bash
python -m scripts_new.generation.generate_responses \
  --model models/Llama-3.1-8B-Instruct \
  --tasks aut \
  --conditions standard creative \
  --repeats 1 \
  --output outputs/aut_responses.csv

python scripts_new/mech_interp/extract_response_activations.py \
  outputs/aut_responses.csv \
  --output outputs/aut_response_heads.pt

python scripts_new/mech_interp/plot_response_head_rsa.py \
  outputs/aut_response_heads.pt \
  --output-dir analysis/aut_response_rsa \
  --positive creative --negative non_creative --position response
```

Use `--position response-predictor` to pool the positions that predict each
answer token, or `--position last-prompt` to use only the final formatted
prompt token.

The plot script averages repeat activations within each task/item/condition,
then computes a Spearman RSA score for each layer/head. By default, it first
subtracts each item's mean across the two conditions, so item-specific
activation baselines do not dominate the condition comparison. This makes the
RSA score ask whether condition shifts are consistent across items; pass
`--no-within-item-center` to inspect uncentered representational geometry.
Select `response` for
mean-pooled answer-token activations, `response-predictor` for the causal
states that predict answer tokens, or `last-prompt` for the final prompt token.
It compares cosine similarities for the selected representations from the same versus different
conditions, using pairs from different items only. The resulting heatmap is
condition-related activation geometry; it does not establish that a head causes
creative behavior. For a cross-item analysis, select a task with at least two
items and collect both requested conditions for every item. DAT has no item
cues, so its RSA uses paired repeat units when those are available; it does
not support conclusions about generalization across different items.

The existing `scripts/run_attention_heads.py` and `scripts/plot_head_rsa.py`
remain a separate prompt-processing analysis: they capture the final prompt
token and run RSA over prompt condition/format attributes.

For older response CSVs without `Prompt_Token_IDs` or `Response_Token_IDs`, the
extraction scripts reconstruct prompt and answer tokenization from saved text.
This gives a useful approximation, but exact token-level correspondence to the
original generation cannot be guaranteed. New generation CSVs preserve both
formatted prompt IDs and generated response IDs, and the extractors use them
when present.

## Attention-head activation differences

The head activation artifact can also be plotted as a direct condition
contrast, without RSA:

```bash
python scripts_new/mech_interp/plot_response_head_differences.py \
  outputs/aut_response_heads.pt \
  --output-dir analysis/aut_head_differences \
  --creative creative --non-creative non_creative --position response
```

The plotter averages repeats per item and condition, subtracts the
non-creative head vectors from the creative vectors for matched items, then
averages those item differences equally. It saves signed head-dimension maps
and layer-by-head absolute, per-item, and relative shift heatmaps, along with
CSV and tensor outputs. Use `--position response-predictor` to
compare states that predict answer tokens, or `--position last-prompt` to
compare the final prompt-token activations.

## Residual-stream activation differences

Extract post-block residual states from the same saved responses. The default
captures both the mean across answer-token residuals and the final prompt-token
residual for every layer:

```bash
python scripts_new/mech_interp/extract_response_residuals.py \
  outputs/aut_responses.csv \
  --output outputs/aut_residuals.pt

python scripts_new/mech_interp/plot_response_residual_differences.py \
  outputs/aut_residuals.pt \
  --output-dir analysis/aut_residual_differences \
  --creative creative --non-creative non_creative --position response
```

The plotter averages repeats within each item and condition, subtracts the
non-creative vector from the creative vector for matched items, then averages
the item differences equally. It saves a signed layer-by-residual-dimension
heatmap, per-layer absolute and relative L2 shift plots and CSVs, and the
difference tensors.
Use `--position response-predictor` to compare states that predict answer
tokens, or `--position last-prompt` to compare the final prompt-token residuals.

## Psychometric task scoring

For AUT, the semantic proxies remove the target object and common answer
framing (for example, “Use a book as …”) before embedding each proposed use.
For CDAT and DRAT, the noun vocabulary is used only to construct reference
baselines and thresholds; generated response nouns are not filtered against
that vocabulary.

The evaluator still counts only numbered single-word entries as valid nouns
for DAT/CDAT/DRAT scoring. Phrases such as “the ocean” are therefore not
silently converted into single nouns.

Generate each model-seed replicate to its own CSV, then pass all replicates to
the evaluator. For example, for AUT:

```bash
python -m scripts_new.generation.generate_responses --model models/Llama-3.1-8B-Instruct \
  --tasks aut --conditions standard creative --seed 42 --output outputs/aut_seed42.csv
python -m scripts_new.generation.generate_responses --model models/Llama-3.1-8B-Instruct \
  --tasks aut --conditions standard creative --seed 43 --output outputs/aut_seed43.csv
python -m scripts_new.evaluation.evaluate_pm_tasks outputs/aut_seed42.csv outputs/aut_seed43.csv \
  --tasks aut --output-dir analysis/aut_eval
```

The evaluator writes response scores and per-run plots, plus `run_means.csv`,
`run_average.csv`, and `run_average.png/.pdf`. It averages repeats within
items, then items within each `RUN_ID`, and weights run means equally. Thus,
separate executions with the same model seed remain distinct replicates. Error
bars show one sample SD across run means. New generation CSVs record both
`RUN_ID` and the base `Model_Seed`; older CSVs without `RUN_ID` fall back to
`Run_ID` (alternate capitalization) or `Model_Seed` when available.
