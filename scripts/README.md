# Creativity experiments

Run commands from the project root with `.venv/bin/python`.
The main scripts follow three steps: run a task, analyze model activations,
and plot the results. Shared modules contain prompts, scoring and model hooks.

Start with these entry points:

| Workflow | Entry point | Main steps |
|---|---|---|
| Behavioral responses | `run_tasks.py --task aut` or `--tasks aut dat drat cdat` | Plan paired trials → load backend once → stream CSVs |
| Stories | `run_creative_writing.py generate` / `extract` | Generate paired stories → replay exact tokens → pool residuals |
| Residual interventions | `run_residual.py extract` / `generate` | Collect prompt states → average contrasts → inject saved vectors |
| Attention heads | `run_attention_heads.py` | Collect heads → select with RSA → audit held-out prompts → save vectors |
| Scoring | `evaluate_aut.py`, `evaluate_dat.py`, `evaluate_association.py` | Parse saved responses → score → export tables and quick plots |
| Seed summaries | `analyze_psychometric_results.py` | Plot responses → estimate paired effects → average seeds → test contrasts |
| Custom plots | `plot_tasks.py`, `plot_head_rsa.py` | Read scored tables or head rankings → save figures |

The `run_greedy_task.py` and `run_temperature_mechanistic.py` runners combine
these stages for the established intervention protocols. They share comparison
logic in `analyze_effects.py`. Task-specific scorers remain separate because
their parsing, models and validity rules differ.

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

Task runs default to the task-specific templates and condition criteria in
`task_config.py`. Conditions accept `standard`, `creative`, `effective`,
`conventional`, and `boring` (the existing capitalized labels also work).
Standard retains the original task instruction and is logged as `Standard`.
The old name `neutral` remains accepted as an alias. For example:

```bash
.venv/bin/python scripts/run_tasks.py --task aut --conditions creative --dry-run
```

These prompts provide one wording (`--paraphrases 0`). Use
`--prompt-protocol legacy` to reproduce the previous generic suffix protocol
and its three paraphrases. Mechanistic runners and the standalone greedy
comparison pipeline retain that protocol for their held-out wording experiments.
`task_config.build_prompt(task_name, item, condition)` also builds prompts directly;
`task_config.get_all_experiments()` enumerates all pilot items and five conditions.


### Five-seed behavioral runs on Snellius

The existing psychometric array jobs run seeds 42–46 across AUT, DAT, DRAT
and CDAT, using all five conditions and the new task-specific prompts.
They default to ten repetitions per item and condition (5,250 responses total),
then score each task and compare Creative with the control conditions.
Each array element loads the model once for its four tasks.

Submit from the project root, using new output and analysis directories:

```bash
mkdir -p logs
sbatch jobs/snellius_small_psychometrics.sh llama \
  outputs/llama8b_new_prompts analysis/llama8b_new_prompts
```

The same battery can be run directly (choose a new output directory):

```bash
.venv/bin/python scripts/run_tasks.py --tasks aut dat drat cdat \
  --model models/Llama-3.1-8B-Instruct --seed 42 --randomize \
  --repeats 10 --temperature 0.7 --output-root outputs/battery/seed_42
```

Each task keeps its own CSV under its task folder. The backend is loaded once.
`--task` keeps the original single-task output layout. Use `--dry-run` to preview
all selected tasks without loading weights or creating files.

Use `qwen` instead of `llama` for Qwen 7B. For Llama 70B:

```bash
sbatch jobs/snellius_llama70b_psychometrics.sh \
  outputs/llama70b_new_prompts analysis/llama70b_new_prompts
```

Override repetitions with `sbatch --export=ALL,REPEATS=20 ...`.
The arrays use up to two concurrent GPUs for small models, or one four-GPU
job at a time for 70B. The existing time limits are two and four hours per
seed respectively; ten repetitions may need a longer allocation. Set
`sbatch --time=HH:MM:SS ...` to the limit appropriate for your model and partition.

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

The battery is part of `run_tasks.py`; the former `run_psychometric_battery.py`
command is replaced by `run_tasks.py --tasks aut dat drat cdat --randomize`.
Seed plotting and testing are part of `analyze_psychometric_results.py`;
use `--mode plot` or `--mode test` instead of the former
`plot_psychometric_seed_average.py` and `test_psychometric_seed_averages.py`.
Maintained Slurm launchers use these commands.

## Validation

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s scripts/tests -v
```

Tests use tiny local models, fake embeddings and mocked API responses. They do
not call paid APIs or download full-size models. Scientific details are in
`writing/analysis_correctness_review.md`, `writing/concept_vectors.md` and
`writing/temperature_and_association.md`.

## Supplementary AGC-Judge scoring

`evaluate_agc.py` uses the published AGC-Human AUT rubric (10–50 per use),
then exports a response-level mean and individual-use audit table. DAT, CDAT
and DRAT retain explicitly local exploratory 1–5 rubrics; retain their existing
distance/relevance measures. See [setup, Snellius commands and validation guidance](../writing/agc_judge.md).

```bash
.venv/bin/python scripts/evaluate_agc.py --task aut \
  --inputs outputs/llama70b_psychometrics_20261017/seed_44/aut --dry-run
```

Use `jobs/snellius_evaluate_agc.sh` after installing PEFT and caching both pinned
model repositories. The default allocation is one H100; two A100s are an alternative.


## Replot and test behavioral seed runs

Use scored `analysis/` roots (containing `seed_42` through `seed_46`):

```bash
.venv/bin/python scripts/analyze_psychometric_results.py \
  analysis/llama8b_new_prompts_20261024T165914 \
  analysis/qwen7b_new_prompts_20261024T165914
```

This regenerates each task's `*_results.png` and `creative_vs_controls.csv`,
then writes `seed_average.png`, `seed_average.pdf`, `seed_means.csv`,
`seed_average.csv`, and `seed_paired_tests.csv`/`.md` in each model root.
Per-seed effects use paired item/generation bootstrap intervals.
Seed-level tests match Creative to each control across the five seed means,
with two-sided paired t-tests, Holm correction over 16 comparisons per model,
and exact sign-flip sensitivity tests. These tests describe fixed pilot items.

To create only the average plots and summary CSVs:

```bash
.venv/bin/python scripts/analyze_psychometric_results.py --mode plot \
  analysis/llama8b_new_prompts_20261024T165914 \
  analysis/qwen7b_new_prompts_20261024T165914
```

Use `--mode test` to rerun only the paired tests from an existing
`seed_means.csv`. The default `--mode all` regenerates all outputs.
`--mode plot --seeds ...` also supports other sets of at least two seeds;
the paired seed tests retain the five-seed protocol (42–46).

Averages give equal weight to items within seeds and to seeds within models.
Whiskers show one SD across seed means, not confidence intervals. Missing scores
are omitted and score coverage is recorded. CDAT novelty is shown before its
appropriateness gate; gate pass counts accompany each average plot.


## DSI scoring of paired creative stories

`evaluate_writing_dsi.py` scores the saved 1,400 Creative/Standard stories from
`outputs/creative_writing_20261014T115031Z`, separately by model and split.
Install the small extra dependency with
`uv pip install --python .venv/bin/python -r requirements-dsi.txt`.
The scorer expects a local `google-bert/bert-large-uncased` checkpoint at
`models/dsi/bert-large-uncased`; this model was downloaded for the current analysis.

```bash
.venv/bin/python scripts/evaluate_writing_dsi.py --dry-run
sbatch jobs/snellius_writing_dsi.sh --output-dir analysis/creative_writing_dsi_NEW_RUN
```

The output directory must be new. DSI uses sentence-context BERT hidden states
6 and 7, with the authors' 50-token sentence cap. `DSI_Full` is a 512-token
sensitivity analysis. Reports include pairing by cue/seed, cue-level t-tests
with Holm correction, complementary seed tests, crossed cue/seed bootstrap
intervals, word lengths, and sentence-truncation counts. Keep interpretation
limited to semantic integration; literary quality and coherence require
additional evaluation. See `report.md` in the analysis output for method details.

## AGC-only evaluation of all saved tasks

`evaluate_agc_sweep.py` discovers raw AUT, DAT, CDAT, DRAT, hypothesis,
metaphor, and cue-word story responses under one or more input roots. It loads
AGC-Judge once. AUT uses the published 10–50 per-use rubric; stories use the
four published 1–5 dimension rubrics. Other tasks retain local exploratory
rubrics. Stories are optional;
the sweep prefers their full `responses.jsonl` over duplicate readable CSVs.
A story CSV without JSONL needs its neighboring manifest for model/run metadata.
Existing scored exports are excluded.

Preview a selection without loading weights or writing output:

```bash
.venv/bin/python scripts/evaluate_agc_sweep.py \
  --inputs outputs/llama8b_new_prompts_20261024T165914 \
           outputs/creative_writing_20261014T115031Z \
  --output-dir analysis/agc_selected --dry-run
```

Run the same selection on one H100 using the downloaded scratch models:

```bash
mkdir -p logs
sbatch jobs/snellius_evaluate_agc_sweep.sh \
  --inputs outputs/llama8b_new_prompts_20261024T165914 \
           outputs/creative_writing_20261014T115031Z \
  --output-dir analysis/agc_selected
```

Use `--inputs outputs` to discover every supported raw run, including older
behavioral and mechanistic runs. Add further model-run roots to select the
psychometric cohorts you want. `--limit-per-run 2` is available for a GPU smoke
test; it takes the first rows and is not a balanced scientific sample. The
Slurm time limit defaults to eight hours; throughput has not been benchmarked.

A fresh output directory is required. It contains `run_index.csv`, separate
response score CSVs mirroring source paths, and `condition_summary.csv` with
response/scored/missing counts plus means and sample SDs per run/model/task/
split/condition/metric, including scale and rubric provenance. AUT additionally
exports `aut_agc_ideas.csv`; response `AGC_Score` is the mean of all parsed uses
only when every parsed use has a valid score. Missing/extra entries are flagged
with idea counts and `AGC_Format_Valid`; malformed unparsed lists stay missing.
Story columns are `AGC_Creativity`, `AGC_Originality`, `AGC_Effectiveness`, and
`AGC_Surprise`. Story `AGC_Score` aliases creativity, not a four-dimension mean.
Scores are descriptive outcomes, not calibrated
cross-task measures. The sweep does not run other scorers or inferential tests.

Source failures and invalid judge outputs retain missing scores. Input limits
are checked without truncation. The full saved story is judged, including a
length-capped generation if present; `Finish_Reason` remains available for
filtering. Outputs flush after each response. An interrupted directory remains
partial; there is no resume mode and reruns require a new directory.

Judging hides condition and model labels. Story instructions use only the
shared three cue words and five-sentence limit, omitting both the Creative
storyline exclusion and Standard's straightforward-style request. Published
rubric wording is saved verbatim in `agc_published_rubrics.json` with pinned
source URLs. The AUT rubric comes from the AGC-Human style-flip prompt, and the
story rubrics come from the cue-word-story scoring prompts. Neutral input
wrapping and the AUT response aggregation are local adaptations; this does not
reproduce the full upstream benchmark or its JRT calibration. See [AGC setup details](../writing/agc_judge.md).

The old custom 1–5 AUT and combined story rubrics have been replaced. Use a new
output directory when rerunning; old scores are not converted. AUT now makes one
judge call per parsed use, and stories make four calls per response, so these
runs take more inference work than the previous one-call-per-response protocol.

### Audit completed AGC psychometric scores

`validate_agc_results.py --input-dir ANALYSIS_ROOT` verifies response identities,
raw-text/condition/seed preservation, complete source coverage, scoring ranges,
rubric text, and AUT idea parsing/aggregation. It writes `validation/report.md`,
coverage tables, equal-item/equal-seed condition means, and descriptive paired
seed differences. Human-rating agreement and a new JRT calibration are not tested.
Use `--allow-subset --require-complete-scores` for a technical pilot. The
`jobs/snellius_agc_validation.sh OUTPUT_ROOT --inputs RAW_ROOT ...` launcher runs
the sweep followed by this audit automatically.

### Small paired AGC sample

Use `--paired-repeats-per-item 1` to choose one complete generation-seed block
per item, retaining Standard, Creative, Effective, Conventional and Boring.
Selection takes the lowest numeric generation seed with all five conditions;
it does not inspect response content or judge scores. The flag cannot be
combined with `--limit-per-run` and is for psychometric tasks only.

For seed 42 of both new-prompt models, this selects exactly 210 responses
(105 per model). Each model contributes 25 AUT, 5 DAT, 50 CDAT and 25 DRAT
responses. One selected AUT response contains nine uses and remains flagged;
the sample requires 659 individual judge calls. This single-seed sample is
an exploratory screen, not a five-seed confirmatory analysis.

```bash
sbatch --time=00:30:00 --export=ALL,AGC_ALLOW_SUBSET=1 \
  jobs/snellius_agc_validation.sh analysis/agc_paired_NEW \
  --inputs outputs/llama8b_new_prompts_20261024T165914/seed_42 \
           outputs/qwen7b_new_prompts_20261024T165914/seed_42 \
  --paired-repeats-per-item 1
```

### Plot AGC condition scores

```bash
.venv/bin/python scripts/plot_agc_results.py analysis/agc_new_prompts_210_20261026
```

Writes `plots/agc_condition_means.png` and `.pdf`. Each task keeps its original
score scale. Points show condition means and whiskers show one response-score
sample SD, not confidence intervals. Groups with one scored response (DAT in
the paired sample) have no SD whiskers. Pass `--output PATH_STEM` to save elsewhere.

## Plot scored psychometric runs

```bash
.venv/bin/python scripts/plot.py analysis/psychometric_prompt_comparison_20261026T143116
```

Pass a parent analysis folder or individual model/prompt folders. The script
finds all numeric `seed_*` folders, regenerates individual-seed task plots,
and **always** writes `seed_average.png`, `seed_average.pdf`, `seed_means.csv`,
and `seed_average.csv` in each model/prompt folder. Use `--averages-only` to
keep existing individual-seed plots. It requires at least two seeds and complete
scored AUT, DAT, DRAT and CDAT folders for every discovered seed; it does not
select a partial set of available seeds.

Means weight items equally within seeds and seeds equally within each model/prompt
version. Whiskers show one sample SD across seed means, not confidence intervals.
CDAT novelty is descriptive before appropriateness gating; gate counts accompany
the plot. Missing scores are excluded and coverage is exported.
