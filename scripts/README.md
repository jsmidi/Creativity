# Creativity experiments

The active pipeline is `generate.py` / `activation_steer.py` followed by
`evaluate_aut.py` or `evaluate_dat.py`. See `writing/mechanism_design.md` for the
scientific design and limits of the implemented experiments.

## What changed

- Protocol `creativity-v2` uses shared prompts for API and local runs; AUT asks
  for alternative uses without also saying "uncommon". The default order is
  Standard, Conventional, Effective, Boring, Creative. Use `--conditions` to
  select a subset and `--randomize` to shuffle.
- Repeated generations, logged request order, block/response IDs, explicit
  decoding parameters, and failure records replace one generation per cell.
- API seed submission is opt-in because provider support varies. The schedule
  seed is always saved. A shared seed does not make different prompts' outputs
  deterministic or identical. Reasoning settings now require explicit
  `--extra-body` JSON instead of guessed model-name rules. Provider defaults may
  differ: freeze and verify these settings before comparing providers.
- Local extraction and intervention use PyTorch hooks, not the former NNsight
  prototype. Chat formatting, tensor/tuple outputs, per-step hooks, and actual
  pre-output-projection head slices are explicit. Currently validated on the
  Llama decoder layout, not every architecture or remote NDIF.
- New scoring outputs live under `analysis/`; existing outputs remain intact.
  Legacy data require explicit opt-in and cannot be pooled with v2 prompts.

## Behavioral pilot

Run from the thesis directory with the project's Python environment (on Windows
replace `python` with `.venv/Scripts/python.exe` if it is not activated):

```powershell
python scripts/run_all.py --dry-run --repeats 10
python scripts/generate.py --task "Alternative Uses Task" --repeats 5
python scripts/generate.py --model meta-llama/Llama-3.3-70B-Instruct-Turbo --provider together --task "Alternative Uses Task" --repeats 10 --paraphrases 0 1 --split pilot
```

`generate.py` and `run_all.py` default to the local model at
`models/Llama-3.1-8B-Instruct`; no API key is needed. `run_all.py` runs both AUT
and DAT. `--dry-run` skips model loading and generation. Ten repeats is a pilot
default, not a justified final sample size. For API generation, explicitly pass
both `--provider` and `--model`. `run_all.py --api-models` selects the previous
API registry (availability unverified) and incurs provider usage unless dry-run.
API key loading still uses the existing local environment file.

## Local causal experiment

These commands require access to the model weights and suitable hardware.
Layer 16 and the strengths below are examples, not empirically selected values.

```powershell
python scripts/activation_steer.py extract --artifact outputs/vectors/aut_l16.pt --layer 16 --items book fork paperclip towel --paraphrases 0 --dry-run
python scripts/activation_steer.py extract --artifact outputs/vectors/aut_l16.pt --layer 16 --items book fork paperclip towel --paraphrases 0
python scripts/activation_steer.py generate --artifact outputs/vectors/aut_l16.pt --items brick rope --paraphrases 1 --split validation --repeats 10 --alphas 0.5 1 2
```

After selecting a layer and **one** strength on validation data, freeze them:

```powershell
python scripts/activation_steer.py generate --artifact outputs/vectors/aut_l16.pt --items can stick belt --paraphrases 2 --split test --alphas 1 --repeats 10
python scripts/activation_steer.py generate --artifact outputs/vectors/aut_l16.pt --task "Divergent Association Task" --paraphrases 2 --split test --alphas 1 --repeats 10
python scripts/activation_steer.py generate --artifact outputs/vectors/aut_l16.pt --task "Metaphor Generation" --paraphrases 2 --split test --alphas 1 --repeats 10
```

Use `--dry-run` to see the number of generations first. The full control set is
substantial: defaults include five random directions. Final-test runs allow one
alpha only. The code checks extraction item/wording overlap, but does **not**
track your full history of validation/test queries. You must keep validation
and test items, tasks, and paraphrases disjoint across runs yourself. Do not use
DAT to select hyperparameters and then call it held-out DAT transfer.

Extraction uses Creative minus Standard by default. `--baseline Conventional`
is a distinct contrast; it can conflate creative enhancement with conventional
suppression. Run it as a robustness check with a separate artifact filename.
`--tasks` can pool multiple tasks with equal task weights; omit `--items` for
multi-task extraction. Leave the intended transfer task out of extraction and
hyperparameter selection. `--head 3` extracts and intervenes on head 3 before
`o_proj`; without `--head`, the target is the post-block residual stream.

The vector is the **raw mean activation difference**, so alpha=1 adds one mean
contrast. It is not unit-normalized. Compare relative intervention norms across
layers as well as alpha. Random directions have the same norm as that vector.

The runner includes:

- Unsteered Standard, Creative, Conventional.
- Standard plus direction, Creative minus direction, and multiple random
  direction additions at matched alpha/norm.
- Centered projection suppression in both Standard and Creative prompts,
  plus random-direction suppression under Creative prompts.
- Standard at higher temperature, using otherwise matched sampling settings.
- Standard patched with Creative prompt-boundary state and the reverse.

Suppression computes `h - alpha * dot(h - baseline_center, unit_vector) *
unit_vector`. It clamps the relevant coordinate toward the extraction baseline;
it is not a claim that this coordinate is exclusively creative. At generated
tokens, a prompt-boundary center may be distributionally inappropriate, so also
run `--scope prefill`. Patching always acts only on the last prompt token during
prefill. Patching is item-specific donor transfer, not evidence of a reusable
general direction. Full-state residual patching is coarse; follow with localized
component tests. Head interventions must use vectors extracted at that head.

Local sampling sets `do_sample` explicitly, with top-p=1 and top-k=0. Generation
uses input token lengths to separate the completion. Vector artifacts record
model revision, chat template hash, extraction examples and package versions;
generation manifests record artifact hashes and settings. Exact reproducibility
across devices/kernels is not guaranteed.

## Scoring

Prefer exact raw run paths, particularly for confirmatory comparisons. Directory
inputs are convenient for a pilot but may collect multiple configurations.

```powershell
python scripts/evaluate_aut.py --inputs outputs --output-dir analysis/aut_v2
python scripts/evaluate_aut.py --inputs outputs --ratings-only --output-dir analysis/aut_ratings
python scripts/evaluate_dat.py --inputs outputs --scorer mpnet --output-dir analysis/dat_proxy
```

The DAT default is **official**, which fails clearly if required resources are
missing. Obtain the authors' `dat.py` and `words.txt` from
https://github.com/jayolson/divergent-association-task and
`glove.840B.300d.txt` from https://nlp.stanford.edu/projects/glove/.
They have not been downloaded automatically; the embedding is large.

```powershell
python scripts/evaluate_dat.py --inputs outputs --scorer official --official-code databases/dat.py --dictionary databases/words.txt --glove databases/glove.840B.300d.txt --output-dir analysis/dat_official
```

The strict numbered-list parser rejects multiword DAT entries instead of silently
using their first word. The official scorer additionally validates dictionary
membership. MPNet mode checks syntax only and reports `DAT_MPNet_Proxy`, never an
official DAT score. Scores require seven unique qualifying words. Syntax cannot
verify that an entry is a common noun; audit task-rule compliance separately.

Missing scores are NaN, not zero. Optional DAT frequency CSVs require probability
values in [0,1]; absent words remain missing, not maximally rare. An AUT rarity
of 1 means no match at the chosen threshold in this particular database, not
proof of originality. The denominator is response frequency, not an inferred
number of human participants. Inspect database provenance before treating it as
human-only norms. DBSCAN novelty is sample-dependent and exploratory.

AUT exports:

- `aut_ideas.csv`: idea-level proxies, not independent trial observations.
- `aut_responses.csv`: one row per response, including failed/invalid responses,
  with mean proxies, counts, and format status.
- `aut_blinded_ratings.csv`: shuffled ideas without model/condition labels.
- `aut_rating_key_private.csv`: separate key; do not give this to raters.

Have at least two blinded raters use a fixed rubric: originality relative to
ordinary uses (1-5), usefulness/feasibility (1-5), and whether it is a valid
alternative use (0/1). Train on development examples, assess agreement, and
freeze the rubric. Raters must judge the same meaning across paraphrases. Join
completed ratings via Rating_ID, then aggregate within Response_ID before
condition inference. Rating collection/merging and other task rubrics are not
automated in this patch. This is necessary remaining research work, not a
replacement for the proxy scorers.

## Inference and checks

Plots show response means with error bars of +/- one sample standard deviation
(SD), not confidence intervals. AUT uses one mean score per generated list;
its SD includes variation across both objects and repeated generations.
Groups with fewer than two scored responses have no SD bar; identical scores
have zero-width bars. For one scored run, estimate an explicit paired contrast:

```powershell
python scripts/analyze_effects.py analysis/aut_v2/aut_responses.csv --metric Semantic_Distance_Proxy --treatment Creative --control Standard
python -m unittest discover -s scripts/tests -v
```

The bootstrap resamples AUT items and paired generation blocks within items.
DAT intervals describe repeated generations within the tested task, not a
population of task items. Complete-pair analysis is conditional on successful
scoring; inspect condition-specific failure/compliance rates and perform
sensitivity analysis. These exploratory intervals do not correct for multiple
layer, strength, task, or metric comparisons. Preregister primary contrasts and
use a larger, frozen test set for confirmatory inference.

`evaluate.py`, `evaluate_musescorer.py`, `generate_responses.py`, and
`agc-scorer.py` are older prototypes, not the validated v2 pipeline. Do not use
their old zero fallbacks/aggregate metrics for the final study.
# Snellius local behavioral run

From the project root on Snellius (Bash), create a separate Linux environment:

```bash
uv venv --python 3.11 .venv-snellius
uv pip install --python .venv-snellius/bin/python torch transformers accelerate
sbatch scripts/snellius_behavioral.sh
```

The Slurm script requests one A100 and runs the downloaded
`models/Llama-3.1-8B-Instruct` on DAT and AUT with five repeats and the five
conditions in fixed order (25 DAT and 125 AUT responses). Submit from the project
root; add `--account=YOUR_ACCOUNT` to `sbatch` if your allocation requires it.
For a shorter pilot use `REPEATS=1 sbatch scripts/snellius_behavioral.sh`.
Monitor with `squeue -u "$USER"` and `tail -f creativity-JOBID.log`.
The one-hour wall time is an initial allocation, not a measured runtime.

Local generation uses the tokenizer chat template, paired generation seeds,
top_p=1 and top_k=0, and writes evaluator-compatible creativity-v2 CSVs under
`outputs/Llama-3.1-8B-Instruct/`. No API key is needed. The model loads once per
task. Existing API generation remains available. GPU inference must run inside
the Slurm allocation. A dry run needs no model or GPU:

```bash
.venv-snellius/bin/python scripts/generate.py --provider local --model models/Llama-3.1-8B-Instruct --task "Divergent Association Task" --repeats 5 --dry-run
```

## GPT-OSS-20B behavioral runs on Snellius

Keep Llama in its existing folder. Download the Transformers weights separately
from a node with internet access, then submit from the project root:

```bash
bash scripts/download_gpt_oss.sh
mkdir -p logs
REPEATS=1 sbatch --gpus=2 scripts/snellius_behavioral.sh gpt-oss
```

The runner uses `.venv/bin/python` and requires two 40GB A100 GPUs or one 80GB GPU. It reserves 8 GiB per
GPU and requires at least 60 GiB remaining across the visible GPUs, using
automatic layer placement without CPU/disk offloading. It explicitly dequantizes MXFP4 weights to BF16; it
does not require downloaded MXFP4 kernels. The existing Transformers, Torch,
and Accelerate dependencies provide the loader. This path still needs an actual
GPU smoke test before a full experiment.

The default is five repeats per condition, low reasoning effort and 4096 total
new tokens (reasoning plus answer). Override `REPEATS`, `REASONING_EFFORT`, or
`MAX_TOKENS` before submission. The one-hour allocation is not a runtime estimate.
The model loads once per task. GPT-OSS runs use the same behavioral prompts and
sampling settings, but their token budget differs from the Llama job; freeze and
record the intended comparison settings before a study.

Outputs go to `outputs/gpt-oss-20b/`. `Response` contains only the final channel;
`Raw_Completion` retains the complete generated text. Missing final answers are
logged as empty rather than scoring reasoning text. Inspect `Finish_Reason` for
truncation. GPT-OSS support is behavioral only; Llama intervention scripts have
not been adapted or validated for GPT-OSS.

Evaluate the latest GPT-OSS DAT and AUT files with:

```bash
INPUT_DIR=outputs/gpt-oss-20b OUTPUT_DIR=analysis/gpt_oss_pilot \
  sbatch scripts/snellius_evaluate.sh
```

Loading guidance: https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers

## Choose a model with one generation script

Submit from the project root after creating `logs/`:

```bash
mkdir -p logs
REPEATS=1 sbatch scripts/snellius_behavioral.sh llama
REPEATS=1 sbatch --gpus=2 scripts/snellius_behavioral.sh gpt-oss
# An explicit local directory also works:
sbatch --gpus=2 scripts/snellius_behavioral.sh models/gpt-oss-20b
```

Both choices run DAT and AUT. Omitting the argument defaults to Llama (or the
legacy `MODEL` environment setting). A positional argument takes precedence.
The script reads `config.json`: GPT-OSS defaults to 4096 new tokens and other
models to 800. `MAX_TOKENS` overrides either default; `REASONING_EFFORT` defaults
to low and applies only to GPT-OSS. Arbitrary paths still require a model
supported by the Python generation backend. Outputs remain separated by model
under `outputs/`; both jobs log to `logs/creativity-JOBID.log`. The shared job
requests one A100 for one hour; GPT-OSS retains its GPU memory check.

## First residual-stream pilot (Llama)

```bash
mkdir -p logs
sbatch scripts/snellius_residual.sh llama
```

This uses the downloaded Llama on one A100. It extracts the mean Creative minus
Standard post-block residual at the last formatted prompt token, at zero-based
layer 16, from AUT brick/rope/bottle/spoon. It then runs 240 validation
responses: book/fork/paperclip/towel/can, three repeats, and 16 intervention/control conditions.
The default alpha is 1 (one raw mean contrast), with two norm-matched random
vectors. Addition/subtraction and centered suppression act at the prompt
boundary (`SCOPE=prefill`); donor patching also acts there. Set `SCOPE=each_step`
for a separate experiment with intervention during generation.

Layer 16 and alpha 1 are pilot choices, not validated optima. Change `LAYER`,
`ALPHA`, `REPEATS`, or `RANDOM_VECTORS` before submission. Every job writes a
separate `outputs/residual_pilot_JOBID/` and `analysis/residual_pilot_JOBID/`.
Extraction refuses to overwrite an existing vector. The two-hour request is
an allocation limit, not a runtime estimate. GPT-OSS is not supported by this
intervention adapter; its behavioral loader is separate.

The job exports raw responses, extraction metadata, a generation manifest,
blinded rating sheets, format audits, AUT proxy scores and a paired
Standard-plus-vector minus Standard effect. MPNet must already be cached;
`SCORE=0` skips embedding scoring and effect estimation. Inspect all controls,
format failures and truncation, and collect originality/usefulness/validity
ratings before interpreting the proxy effect. Five objects provide limited evidence for
population-level claims. No DAT or final-test objects are used by this job;
freeze layer/strength using validation before testing transfer.

Protocol note: `creativity-v2` Standard is neutral, not the effective and
length-matched control described in the Word proposal. Paraphrase IDs change
condition suffixes, not the underlying task instruction or neutral Standard
prompt. This pilot holds out AUT objects and creative condition wording only.
Steering is a causal intervention on a candidate representation; it does not
alone establish a creativity-specific circuit or natural contribution.

## RSA-selected attention-head concept vectors

An Opielka-inspired creativity pilot is now available. See
[method, source attribution, controls and limitations](../writing/concept_vectors.md).

```bash
.venv/bin/python scripts/concept_vectors.py --output-dir /tmp/concept-preview --dry-run
sbatch scripts/snellius_concept.sh llama
```

This scans all Llama heads with RSA, audits the fixed top five on separate
objects/wording, and tests a projected Creative mean, a projected contrast, and
a direct intervention on the highest-ranked head. It is an adaptation for
creativity instruction conditions, not a replication of the original relational
concept tasks. All outputs remain compatible with the current AUT evaluators.

### Five-item AUT comparisons

Both Slurm MI scripts now evaluate book, fork, paperclip, towel, and can.
Each run includes unsteered Creative, Standard (no suffix), Effective,
Conventional, and Boring conditions. Evaluation uses wording index 0:
“Be creative.”, “Be effective.”, “Be conventional.”, and “Be boring.”
Extraction uses brick, rope, bottle, and spoon with different wording;
concept-head auditing uses umbrella and shoe. These are new exploratory runs,
not reuse of the old artifacts or a frozen confirmatory experiment.

With scoring enabled, `creative_vs_controls.csv` contains four separate paired
comparisons, each reporting Creative minus one control for
`Semantic_Distance_Proxy`. Positive differences mean greater semantic distance,
not necessarily more useful or valid ideas. Intervals are exploratory 95%
bootstrap intervals without multiple-comparison correction; blinded human
ratings remain necessary. The existing vector-addition comparison is retained.

Default generation counts are 240 for residual steering and 480 for concept
vectors (160 for each of three candidates). The concept discovery RSA still
ranks the original Creative/Conventional/Effective geometry; these behavioral
comparisons do not constitute four independently learned vector contrasts.

To analyze an existing scored run containing all five conditions:

```bash
.venv/bin/python scripts/analyze_effects.py PATH/aut_responses.csv \
  --metric Semantic_Distance_Proxy --treatment Creative \
  --controls Standard Effective Conventional Boring \
  --output PATH/creative_vs_controls.csv
```

## Temperature sweep and CDAT/DRAT follow-up

`snellius_behavioral.sh` now defaults to a four-temperature Slurm array
(0, 0.7, 1.0, 1.3), with 30 repeats per item/condition at nonzero temperature and
one greedy response per prompt. Set `REPEATS=60` for 60; use `--array=0` with a
single-entry `TEMPERATURES` value for a single-temperature run. This replaces
the older five-repeat, fixed-temperature shell defaults described above.

See [the complete protocol and commands](../writing/temperature_and_association.md)
for sampling provenance, evaluation by temperature, determinism checks, and
optional CDAT/DRAT tasks. `INCLUDE_ASSOCIATION=1` adds those tasks to the array;
they require new cue/anchor-conditioned answers and have explicitly documented
pilot adaptations.
