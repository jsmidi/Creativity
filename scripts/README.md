# Creativity experiment scripts

Run commands from the project root with `.venv/bin/python`. The maintained
pipeline uses the `creativity-v2` prompt protocol for AUT, DAT, CDAT and DRAT.
Raw generations and frozen source snapshots belong in `outputs/`; scored tables
and figures belong in `analysis/`. Running Slurm jobs execute their own frozen
copies, so changes to this directory do not alter submitted experiments.

## Where to start

| Script | Purpose |
|---|---|
| `generate.py` | Local/API behavioral generation, paired seeds, response CSVs and provenance |
| `evaluate_aut.py` | AUT distance/rarity proxies, response means and blinded rating sheets |
| `evaluate_dat.py` | Official GloVe DAT scoring or an explicitly labelled MPNet proxy |
| `evaluate_association.py` | CDAT novelty/appropriateness gates and DRAT relevance-filtered novelty |
| `analyze_effects.py` | Paired instruction/intervention contrasts and bootstrap intervals |
| `activation_steer.py` | Residual/head extraction, addition, suppression and donor patching |
| `concept_vectors.py` | Head RSA, held-out audit and residual/direct-head concept artifacts |
| `run_temperature_mechanistic.py` | Separate extraction, intervention and summary phases for Slurm arrays |
| `plot_head_rsa.py` | Pilot-style discovery heatmap and selected-head audit, PNG/PDF |
| `evaluate_sweep.py` | Score multiple raw runs separately, retaining four prompt contrasts |
| `audit_sampling.py` | Inspect generation uniqueness and greedy decoding settings |
| `prepare_association_resources.py` | Prepare the frozen WordNet noun vocabulary |
| `run_greedy_task.py` | Standalone temperature-zero behavioral and intervention pipeline |
| `run_all.py` | Configured local/API AUT and DAT pilots; forwards generator options |
| `plot_greedy_results.py`, `plot_greedy_bars.py` | Plot older greedy-run layouts |
| `check_providers.py` | List/filter provider model IDs without generating responses |

Shared implementation modules:

- `experiment.py`: condition suffixes, canonical item labels, prompts and paired trial schedules.
- `task_config.py`: task instructions and evaluation items.
- `sampling.py`: explicit greedy versus sampled decoding options.
- `scoring_common.py`: raw CSV loading, numbered-list parsing and descriptive plots.
- `interventions.py`: model hooks, residual/head coordinates and intervention scope.
- `local_generation.py`: behavioral model loading and GPT-OSS final-channel handling.

## Behavioral generation and scoring

Preview the exact prompt and response count without loading a model:

```bash
.venv/bin/python scripts/generate.py --provider local \
  --model models/Llama-3.1-8B-Instruct --task "Alternative Uses Task" \
  --temperature 0.7 --repeats 60 --randomize --dry-run
```

Remove `--dry-run` to generate. Qwen uses `models/Qwen2.5-7B-Instruct`.
Temperature zero normally needs one repetition per item/condition; identical
repeats are determinism checks. Nonzero temperatures use independently seeded
repetitions. Conditions within each block share a generation seed.

API providers are selected explicitly with `--provider together`, `groq`, or
`openrouter` and a matching `--model`. Credentials are loaded from the existing
local `scripts/ATT05522.env` file. `--send-seed` is opt-in for APIs.

Score one raw run at a time:

```bash
.venv/bin/python scripts/evaluate_aut.py --inputs PATH_TO_RAW_CSV \
  --norm-db aut_quality_scored_all.csv --output-dir analysis/aut_example

.venv/bin/python scripts/evaluate_dat.py --inputs PATH_TO_RAW_CSV \
  --scorer official --official-code databases/dat.py \
  --dictionary databases/words.txt --glove databases/glove.840B.300d.txt \
  --output-dir analysis/dat_example

.venv/bin/python scripts/evaluate_association.py --task cdat \
  --inputs PATH_TO_RAW_CSV --output-dir analysis/cdat_example --device cuda
```

Use `--task drat` for DRAT. Association scoring requires
`databases/association/nouns.txt`, prepared with
`prepare_association_resources.py`. It refuses to overwrite a calibration
folder. Missing scores remain missing, not zero; DRAT deliberately assigns
zero when fewer than three words pass its utility threshold.

AUT exports idea-level proxies and averages them per generated response.
`Relative_Novelty_ICF` depends on clusters fitted within the scoring sample;
its values are not fixed external norms. Human rating sheets remain separate
from automated scores. CDAT novelty must be interpreted with `cdat_gates.csv`.
The DAT MPNet option is a proxy, not the original GloVe score.

```bash
.venv/bin/python scripts/analyze_effects.py PATH/aut_responses.csv \
  --metric Semantic_Distance_Proxy --treatment Creative \
  --controls Standard Effective Conventional Boring \
  --output PATH/creative_vs_controls.csv
```

The bootstrap gives each item equal weight and resamples paired generations
within items. Temperature-zero differences are descriptive, without a sampling
interval. Other intervals are exploratory and unadjusted for multiple comparisons.
Bar-plot error bars show standard deviations, not confidence intervals.

## Mechanistic stages

`activation_steer.py extract` saves a Creative-minus-Standard state vector.
`concept_vectors.py` ranks pre-`o_proj` attention heads by Spearman RSA, freezes
the selection, audits new items/wording, and saves projected and direct-head
artifacts. Both Llama and Qwen2 head layouts are tested.

```bash
.venv/bin/python scripts/concept_vectors.py --model models/Qwen2.5-7B-Instruct \
  --output-dir outputs/concept_preview --dry-run

.venv/bin/python scripts/run_temperature_mechanistic.py --phase intervene \
  --index 4 --model models/Llama-3.1-8B-Instruct \
  --run-root outputs/EXPERIMENT --analysis-root analysis/EXPERIMENT --dry-run
```

The mechanistic array has 48 intervention entries: four tasks, three temperatures
and four candidates (`residual`, `concept_mean`, `concept_contrast`, `concept_head`).
There are four extraction entries, one per task. The summary phase combines
candidate contrasts. The sampled protocol uses 60 repeats at 0.7 and 1, and
one at 0. DAT concept artifacts transfer from AUT discovery prompts.

Composite suppression centers on the full Standard post-block residual at the
injection layer. Individual-head suppression uses the Standard head mean.
Addition uses raw vector norms; random directions match those norms.
Interventions here use layer 16, alpha 1 and prefill scope. Donor patching always
acts at the prompt boundary. These are exploratory choices, not validated optima.
RSA characterizes instruction-condition geometry rather than measured creativity.

Head-RSA figures are shared across decoding temperatures because extraction
measures prompt processing before sampling:

```bash
.venv/bin/python scripts/plot_head_rsa.py \
  --run-root outputs/MECHANISTIC_RUN --analysis-root analysis/COMPARISON_RUN
```

## Slurm entry points

Create `logs/` and submit from the project root. Check each script's resource
request and model default before submission; the older greedy/pilot launchers
currently default to a 70B model with four H100 GPUs.

| Launcher | Purpose |
|---|---|
| `snellius_behavioral.sh` | Configurable behavioral temperature sweep; optional association tasks |
| `snellius_temperature_comparison.sh` | Prepared Llama-8B behavioral comparison |
| `snellius_qwen_temperature_comparison.sh` | Prepared Qwen-7B comparison at 0, 0.7 and 1 |
| `snellius_temperature_mechanistic.sh` | Frozen model/task extraction and intervention phases |
| `snellius_evaluate.sh` | Score selected/latest AUT/DAT runs or use `--sweep` |
| `snellius_residual.sh`, `snellius_concept.sh` | Standalone exploratory residual/concept pilots |
| `snellius_greedy_{aut,dat,cdat,drat}.sh` | Standalone greedy task pipelines |

Prepared comparison jobs require a `RUN_ROOT` containing frozen `code/`.
The mechanistic launcher additionally needs `MODEL`, `ANALYSIS_ROOT` and `PHASE`.
Keep frozen source directories and submitted manifests intact when refactoring.
GPT-OSS uses the behavioral backend only and has separate GPU-memory and
final-answer handling; it is not supported by the intervention adapter.

## Validation and provider diagnostics

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s scripts/tests -v

.venv/bin/python scripts/check_providers.py together --contains Qwen
.venv/bin/python scripts/check_providers.py openrouter --free-only
```

The unit tests use tiny local models, fake embeddings and mocked API responses;
they do not generate paid API traffic or require downloaded full-size models.
See `writing/analysis_correctness_review.md`, `writing/concept_vectors.md`, and
`writing/temperature_and_association.md` for scoring and scientific limitations.

Removed prototypes: `evaluate.py`, `evaluate_musescorer.py`,
`generate_responses.py`, `agc-scorer.py`, `vector_extraction.py`,
`vector_injection.py`, and `test_api.py`. The four old `check_*` provider scripts
are replaced by `check_providers.py`. Existing frozen run snapshots retain their
original sources for reproducibility.
