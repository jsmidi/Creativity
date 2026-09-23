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
python scripts/generate.py --model meta-llama/Llama-3.3-70B-Instruct-Turbo --provider together --task "Alternative Uses Task" --repeats 10 --paraphrases 0 1 --split pilot
```

`--dry-run` makes no requests. Removing it incurs provider usage. Ten repeats is
a pilot default, not a justified final sample size. `run_all.py` retains your
model/provider registry, which has not been checked for current availability.
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
