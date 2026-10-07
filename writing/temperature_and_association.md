# Temperature sweeps and contextual association tasks

## Meeting follow-up, 4 October 2026

The behavioral Slurm runner now submits four array tasks, one per temperature:
0, 0.7, 1.0, 1.3. These are configurable exploratory settings, not tuned optima.
Each runs DAT and all five AUT objects, under Standard, Creative, Effective,
Conventional and Boring. Standard has no suffix; wording index 0 uses the literal
“Be creative.” etc. Conditions are randomized, using the same seed blocks across
conditions and temperatures. No activation intervention is applied.

`REPEATS` means responses **per item, condition and temperature**, not total per
condition. Defaults are 30 at nonzero temperatures and 1 at zero. With five AUT
objects this means 150 AUT responses per condition at each nonzero temperature.
DAT has one item and therefore 30 responses per condition. The full default
DAT/AUT sweep generates 2,730 responses. REPEATS=60 gives 5,430 responses.
Each AUT response requests ten uses; each DAT response requests ten nouns.

Greedy decoding removes sampling variability, not hardware/kernel variability.
Repeating the identical greedy prompt does not create independent observations.
`GREEDY_REPEATS=30` is available for a determinism audit; the effect analyzer
reports descriptive differences without sampling bootstrap intervals at T=0.
Do not interpret repeated identical answers as a precisely estimated population
creativity effect. The five AUT objects also limit generalization.

```bash
mkdir -p logs
sbatch scripts/snellius_behavioral.sh llama
# Or 60 responses per item/condition at nonzero temperatures:
REPEATS=60 sbatch scripts/snellius_behavioral.sh llama
# Custom grid: array length must match the grid length.
TEMPERATURES='0 0.5 1.0' sbatch --array=0-2 scripts/snellius_behavioral.sh llama
# Single-temperature run:
TEMPERATURES='0.7' sbatch --array=0 scripts/snellius_behavioral.sh llama
```

Each array task requests one A100 for up to four hours, an allocation limit
rather than a runtime estimate. GPT-OSS BF16 still needs `--gpus=2` on 40GB A100s.
Outputs: `outputs/temperature_sweep_ARRAYID/tINDEX/MODEL/`. No new jobs were
submitted as part of this implementation.

Once generation completes, score each raw run independently:

```bash
.venv/bin/python scripts/evaluate_sweep.py \
  --inputs outputs/temperature_sweep_ARRAYID \
  --output-dir analysis/temperature_sweep_ARRAYID --dry-run
sbatch scripts/snellius_evaluate.sh --sweep outputs/temperature_sweep_ARRAYID
```

The evaluation job's default root is `analysis/llama_evaluation_JOBID`; set
`OUTPUT_DIR` to choose another. Each raw run gets its own subdirectory, scores,
and `creative_vs_controls.csv`. Temperatures and runs are never pooled. The
four comparisons are Creative minus each control; intervals are exploratory,
not adjusted for multiple comparisons. DAT uses the existing official GloVe
scorer, AUT the existing MPNet semantic-distance proxy. Human validity,
originality and usefulness ratings remain relevant.

## What sampling settings can be confirmed?

`analysis/sampling_audit.csv` is an evidence-only audit of stored raw records,
with file hashes and counts. Refresh it on the cluster using:

```bash
.venv/bin/python scripts/audit_sampling.py
```

The September 24 Llama DAT/AUT raw CSVs record T=0.7, top_p=1.0 and top_k=0.
The older MI CSVs record T=0.7 and the high-temperature control T=1.0 but omit
p/k. The runner inspected on October 4 explicitly used p=1 and k=0. This supports
an inference about those MI jobs, not an independently recorded historical fact.
Any changed settings from unrecorded runs remain unknown. Model-directory
`generation_config.json` defaults are not evidence of applied parameters when
call-time overrides are present.

New behavioral runs expose `--temperature`, `--top-p`, and local `--top-k`.
CSV rows record requested values, decoding mode and whether sampling filters
apply. T=0 uses greedy decoding, with neutral effective p/k values. JSON
sidecars record actual local overrides, base generation config, model config,
dtype, library versions, GPU/CUDA details, Slurm IDs, source hashes and a source
snapshot directory. Seeds are recorded but do not ensure cross-hardware equality.
New MI rows also record their fixed p=1/k=0. API top-k defaults remain explicitly
unknown; nonzero `--top-k` is rejected for APIs rather than silently ignored.

## CDAT and DRAT exploratory support

These require new generations with cue/anchor prompts. Existing DAT/AUT answers
cannot be retroactively treated as CDAT/DRAT responses. Keep the old measures
for comparison rather than assuming either new test universally replaces them.

Sources:

- [Nakajima et al., CDAT paper](https://arxiv.org/html/2601.20546v1), especially §4.2.
- [Authors' CDAT repository](https://github.com/knakajima1225/beyond_divergent_creativity),
  inspected commit `904f7a1cf559ceaef66df814250eee7ef601165f`.
- [Schapiro et al., DRAT paper](https://arxiv.org/html/2605.13450v1), §4.2 and Appendix F.3.

`Conditional Divergent Association Task` requests ten diverse nouns relevant
to one cue. The pilot has ten locally chosen cues, not the published cue bank.
For the first seven unique valid nouns, CDAT_Novelty is mean pairwise cosine
distance ×100; CDAT_Appropriateness is 100×(1+mean cosine to the cue).
The original CDAT paper uses this 100-shifted appropriateness scale.
Its novelty is interpreted conditional on a group-level appropriateness gate,
not as a per-word filter or a product of novelty and relevance.

Our gate is an explicit repeated-condition adaptation: average repetitions
within each cue first, compare cue means with seeded random seven-noun lists
using a two-sided Welch test, and BH-adjust across the model/condition/run tests
within each temperature supplied to the scorer. Passing requires adjusted p<.001
and higher mean appropriateness. `cdat_gates.csv` reports eligibility and a
CDAT_Score only for passing groups. The family differs if you score different
sets of runs together; predefine the family for inferential use. Response-level
novelty contrasts are descriptive and must be read alongside the gate.

`Divergent Remote Association Test` requests ten diverse nouns related literally
or metaphorically to all anchors. Defaults use the first five scientific anchor
quadruples from the DRAT paper, not the full thirty-set benchmark. Scoring follows
its max-anchor utility, fixed random-noun 90th-percentile threshold, strict `>`
gate, and mean pairwise distance ×100 among survivors. Fewer than three survivors
scores zero. Despite the prompt asking about all anchors, the published default
scorer uses the closest anchor, not the minimum across anchors.

Both implementations are independent and labeled `association-exploratory-v1`.
They use a frozen WordNet single-word noun-lemma whitelist, allow hyphens,
deduplicate in order and examine only the first ten numbered entries. This is
stricter than the CDAT authors' POS-tagger-or-WordNet validity rule, and misses
some inflected nouns. Fewer than seven valid CDAT nouns produces missing scores;
failed generation requests produce missing scores, not zero creativity. Audits
retain every response. The default embedding is the already-used MPNet model,
not an exact reproduction of a published multi-embedding experiment.

The scorer samples 5,000 unique calibration nouns with seed 42, independent of
responses, and 500 random lists per cue for the CDAT baseline. It saves the noun
pool, baseline scores, every embedding used, input/resource hashes and scoring
settings. Calibration must remain fixed across conditions. WordNet provenance
and its license are in `databases/association/`; recreate in a *new* directory:

```bash
.venv/bin/python scripts/prepare_association_resources.py --output-dir /tmp/new-association-resources
```

Preview or generate a small pilot within a GPU allocation:

```bash
.venv/bin/python scripts/generate.py \
  --task 'Conditional Divergent Association Task' --repeats 1 --randomize --dry-run
.venv/bin/python scripts/generate.py \
  --task 'Divergent Remote Association Test' --repeats 1 --randomize --dry-run
```

Remove `--dry-run` to generate. To include both tasks in the Slurm sweep:

```bash
INCLUDE_ASSOCIATION=1 REPEATS=30 sbatch scripts/snellius_behavioral.sh llama
```

This expands the default sweep to 9,555 responses; use a small pilot before that
larger measurement run. For example, `INCLUDE_ASSOCIATION=1 REPEATS=1` produces
420 responses across all four tasks/temperatures. `evaluate_sweep.py` recognizes
all four tasks. You can also score a specific new-task run:

```bash
.venv/bin/python scripts/evaluate_association.py --task cdat \
  --inputs PATH/RAW_CDAT.csv --output-dir analysis/cdat_new --device cuda
.venv/bin/python scripts/evaluate_association.py --task drat \
  --inputs PATH/RAW_DRAT.csv --output-dir analysis/drat_new --device cuda
```

These additions enable a comparison of measures. They do not establish that a
higher association score corresponds to more useful creative behavior.

## Requested 60-repeat greedy task jobs

Four separate entry points now run on local Llama:

```bash
sbatch scripts/snellius_greedy_aut.sh
sbatch scripts/snellius_greedy_dat.sh
sbatch scripts/snellius_greedy_cdat.sh
sbatch scripts/snellius_greedy_drat.sh
```

Each requests one A100 for up to eight hours. All five prompt conditions receive
60 actual generations per item, at temperature zero. Behavioral totals are
1,500 AUT, 300 DAT, 3,000 CDAT and 1,500 DRAT. Executable Python sources are copied
into each run directory when the job starts. `plan.json` records the full design.

Each job then extracts its own Creative-minus-Standard residual vector and
runs four candidate intervention suites: residual, projected concept mean,
projected concept contrast, and the highest-ranked individual attention head.
Each suite contains 15 all-greedy arms (the high-temperature control is omitted).
Default MI_REPEATS=1 yields respectively 300, 60, 600 and 300 additional answers.
REPEATS and MI_REPEATS can be overridden independently at submission.

AUT/CDAT/DRAT concept discovery uses task-specific instructions, separate
items and wording indices 1/2; evaluation uses wording 0. DRAT anchor words are
disjoint across discovery, audit and evaluation, not just the sets themselves.
DAT has no item axis, so its residual extraction holds out condition wording;
its concept vectors deliberately transfer from AUT discovery. It is not a
DAT-specific RSA head search. Concept discovery retains the existing
Creative/Conventional/Effective condition geometry; behavioral evaluation
includes all five prompt conditions. Vectors are still Creative minus Standard.

At completion, `analysis/greedy_TASK_JOBID/` contains per-candidate task scores,
`effects.csv` with all four prompt contrasts and intervention-versus-baseline
and intervention-versus-random comparisons, `determinism_audit.csv` with exact
response uniqueness/failure/truncation counts, and a README. Head rankings and
audits live in `outputs/greedy_TASK_JOBID/concept_discovery/`. Effects are
strictly descriptive at T=0, and unavailable scored comparisons are reported
explicitly. CDAT novelty must be read with candidate-specific gate results.
No replication claim is made for the exploratory CDAT/DRAT task implementations.
