# Creativity experiments

Run commands from the project root with `.venv/bin/python`.
The main scripts follow three steps: run a task, analyze model activations,
and plot the results. Shared modules contain prompts, scoring and model hooks.

## Task runs

| Script | Purpose |
|---|---|
| `run_tasks.py` | Generate AUT, DAT, CDAT or DRAT responses, locally or through an API |
| `run_creative_writing.py` | Generate paired stories; extract vectors from their response tokens |
| `evaluate_aut.py` | Score AUT uses and export blinded human rating sheets |
| `evaluate_dat.py` | Score DAT with official GloVe or the MPNet proxy |
| `evaluate_association.py` | Score CDAT and DRAT |
| `evaluate_sweep.py` | Score several task runs separately |

```bash
.venv/bin/python scripts/run_tasks.py --task aut \
  --model models/Llama-3.1-8B-Instruct --temperature 0.7 \
  --repeats 60 --randomize --dry-run
```

Use `aut`, `dat`, `cdat` or `drat`; full task names also work. Remove
`--dry-run` to generate. Use `--provider together`, `groq` or `openrouter`
for API runs; credentials come from the existing local `scripts/ATT05522.env`.
API seeds require `--send-seed`. Paired conditions share a generation seed.
At temperature zero, one repetition is normally sufficient.

```bash
.venv/bin/python scripts/run_creative_writing.py generate \
  --model models/Qwen2.5-7B-Instruct --split discovery \
  --output-dir outputs/writing_example --dry-run

.venv/bin/python scripts/run_creative_writing.py extract \
  --run-dir outputs/writing_example --output-dir outputs/writing_vectors
```

Stories use cue triplets and Standard/Creative conditions. The bundled ten-cue
pilot is not the paper's 50-triplet selection. Extraction replays saved token IDs,
checks complete pairs and averages response-token residuals equally across cues.
See [the story protocol](../writing/creative_writing_protocol.md).

Score raw runs with `--inputs PATH --output-dir PATH`:

```bash
.venv/bin/python scripts/evaluate_aut.py --inputs outputs/RAW.csv \
  --output-dir analysis/aut_example

.venv/bin/python scripts/evaluate_dat.py --inputs outputs/RAW.csv \
  --scorer official --official-code databases/dat.py \
  --dictionary databases/words.txt --glove databases/glove.840B.300d.txt \
  --output-dir analysis/dat_example

.venv/bin/python scripts/evaluate_association.py --task cdat \
  --inputs outputs/RAW.csv --output-dir analysis/cdat_example --device cuda
```

Use `--task drat` for DRAT. Prepare the association noun vocabulary once with
`prepare_association_resources.py`. Association scoring requires a new output
directory to preserve its calibration. Missing scores remain missing.
CDAT novelty needs its condition-level appropriateness gate in `cdat_gates.csv`;
DRAT scores zero when fewer than three nouns pass its relevance threshold.
AUT automated scores are proxies; cluster novelty depends on the scoring sample.
The primary AUT metric, `Pairwise_Semantic_Distance_Proxy`, is the mean cosine
distance across all distinct pairs of ideas within each generated response
(45 pairs for a ten-idea list). Higher scores indicate more semantically diverse
uses. Duplicate ideas remain in the calculation; fewer than two parsed ideas
produce a missing score. `Semantic_Distance_Proxy` retains the mean distance
from each idea to the target object. Both use `all-mpnet-base-v2` embeddings.

## Mechanistic analysis

| Script | Purpose |
|---|---|
| `run_residual.py extract` | Extract Creative-minus-baseline prompt vectors |
| `run_residual.py generate` | Inject residual or head vectors; test addition, suppression and patching |
| `run_attention_heads.py` | Rank heads with RSA, check held-out prompts and export head vectors |
| `run_temperature_mechanistic.py` | Run extraction, intervention or summary phases for temperature sweeps |
| `analyze_effects.py` | Compare paired response scores with bootstrap intervals |

Extraction and injection share a runner so an existing vector can be reused.
Attention discovery is separate because it ranks and selects many heads.
Its held-out audit checks whether selected heads generalize to new prompts.
That scientific check remains part of the analysis.

**Where vectors are collected:**

- `run_residual.py`: `collect_residual_activations(...)` collects one Creative and
  one baseline state per example. `extract(...)` visibly runs collect, average,
  then save. The vector is the mean Creative-minus-baseline difference, with
  equal weighting across tasks.
- `interventions.py`: `capture_prompt_activation(...)` installs `read_residual`
  on a decoder block. It reads `residual_stream[:, -1, :]`: the residual at the
  final formatted prompt token, after that block and before the final model norm.
- `run_attention_heads.py`: `collect_head_activations(...)` calls
  `capture_attention_heads(...)`. Its `read_head_outputs` hook reads the input
  to each attention `o_proj`, takes the final prompt token and reshapes it into
  separate query heads. The tensor is `[prompts, layers, heads, head_dim]`.
  `main()` explicitly collects discovery states, selects heads with RSA,
  collects held-out/Standard states, then averages and saves the selected vectors.
- `run_creative_writing.py`: `pool_stories(...)` uses
  `interventions.py::capture_response_means(...)` to collect residuals over saved
  response tokens. This is a different token position from prompt-vector extraction.


```bash
.venv/bin/python scripts/run_residual.py extract --help
.venv/bin/python scripts/run_residual.py generate --help
.venv/bin/python scripts/run_attention_heads.py --help

.venv/bin/python scripts/analyze_effects.py analysis/aut_example/aut_responses.csv \
  --metric Semantic_Distance_Proxy --treatment Creative \
  --controls Standard Effective Conventional Boring \
  --output analysis/aut_example/effects.csv
```

Head values are captured before `o_proj`; residual vectors use post-block states.
Suppression uses the matching Standard center. Random controls match vector norms.
Held-out items and instruction wordings are checked before intervention runs.
DAT head discovery transfers from AUT. Temperature sweeps use layer 16, alpha 1,
prefill scope, four candidates and temperatures 0, 0.7 and 1. These are exploratory
settings. RSA measures instruction geometry, not creativity scores.
Bootstrap intervals are exploratory and unadjusted for multiple comparisons;
greedy decoding produces descriptive differences without sampling intervals.

## Plots

`plot_tasks.py` plots any scored task table or paired effects. `plot_head_rsa.py`
plots attention-head rankings and held-out scores. Scorers also save a quick
response plot automatically.

```bash
.venv/bin/python scripts/plot_tasks.py analysis/aut_example/aut_responses.csv \
  --metrics Pairwise_Semantic_Distance_Proxy --output analysis/aut_example/scores.pdf

.venv/bin/python scripts/plot_tasks.py analysis/aut_example/effects.csv \
  --effects --output analysis/aut_example/effects.png

.venv/bin/python scripts/plot_head_rsa.py \
  --run-root outputs/MECHANISTIC_RUN --analysis-root analysis/COMPARISON_RUN
```

Response bar plots show means and one sample SD, not confidence intervals.

## Outputs and supporting code

New task runs write one self-contained CSV, with these columns:

```text
Run_ID, Response_ID, Model, Task, Instruction, Item, Condition,
Generation_Seed, Temperature, Max_Tokens, Top_P, Top_K, Timestamp_UTC,
Prompt, Response, Generated_Tokens, Prompt_Tokens, Error, Model_Dtype
```

`Item` identifies the AUT object or association cue. Scorers infer request status
from `Response`/`Error` and reconstruct paired blocks from `Item`/`Generation_Seed`.
Repeat counters, request order, protocol and block IDs are not saved in new raw
CSVs. Scored tables keep the fields needed for analysis and the score columns;
full request details remain in the raw CSV.

Task runs no longer create JSON manifests or automatic source snapshots.
Intervention runs use the same columns plus `Artifact` and `Scope`, which identify
the vector file and injection timing; strength/direction are in `Condition`.
Residual/head vector metadata is stored inside each `.pt` file, without a duplicate
JSON file. API dtype and top-k are recorded as `unknown`; token counts stay blank
if the provider omits usage. API seeds are sent only with `--send-seed`.
At temperature zero, top-p/top-k are configured values; greedy decoding ignores them.

Story CSVs remain a ten-column reading table; their JSONL keeps exact prompts and
token IDs. Story replay still uses a small manifest to validate complete pairs,
cue splits and model/tokenizer compatibility. Association scoring manifests retain
its calibration resources. These serve a different purpose from the removed
behavioral settings sidecar. File fingerprints and automatic code snapshots are
no longer saved. Exact token IDs and complete-pair checks still validate story replay.

Shared modules:

- `task_config.py`, `creative_writing.py`, `experiment.py`: tasks, prompts and paired schedules.
- `interventions.py`, `local_generation.py`, `sampling.py`: model execution and hooks.
- `scoring_common.py`: CSV loading, list parsing and response plots.

`jobs/` contains Slurm launchers. Create `logs/` before submission and check model
and resource settings. All maintained launchers execute the main `scripts/` directory.
`run_greedy_task.py` remains for the four existing standalone greedy launchers.
GPT-OSS supports behavioral generation, not intervention hooks.

Removed redundant utilities: `audit_sampling.py`, `check_providers.py`, `run_all.py`,
`plot_greedy_results.py`, and `plot_greedy_bars.py`. Renamed `generate.py` to
`run_tasks.py`, `activation_steer.py` to `run_residual.py`, and `concept_vectors.py`
to `run_attention_heads.py`. The old command names are no longer entry points.

## Validation

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s scripts/tests -v
```

Tests use tiny local models, fake embeddings and mocked API responses. They do
not call paid APIs or download full-size models. Scientific details are in
`writing/analysis_correctness_review.md`, `writing/concept_vectors.md` and
`writing/temperature_and_association.md`.
