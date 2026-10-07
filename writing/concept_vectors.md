# Attention-head concept-vector pilot

Source repository: https://github.com/gucioopielka/concept_vectors
Inspected commit: `dcbcd0ef16d8a6c1af8b0a2378b7717a9ba66c42`.
The repository describes the 2025 preprint (https://arxiv.org/abs/2503.03666).
The related expanded study is https://arxiv.org/abs/2602.22424.

Inspected implementation entry points: `src/calculate_RSA.py` and
`src/utils/query_utils.py` (`get_att_simmats`, `get_avg_att_output`,
`get_avg_summed_vec`), plus `src/utils/eval_utils.py` (`rsa`). This repository
implements the method independently with PyTorch hooks; it does not vendor or
execute the authors' code and does not require their NNsight/remote setup.

## What is implemented

1. Capture last-prompt-token activations at the input of each Llama attention
   output projection. Reshape using the number of query heads, not KV heads.
   Llama-3.1-8B has 32 layers × 32 heads, so all 1,024 heads are searched.
2. Compute cosine similarities between prompt activations for each head. Compare
   off-diagonal similarities against same-condition/different-condition labels
   using Spearman correlation with average ranks for ties.
3. Export `Full_RSA` (all prompt pairs), and select the top K by `Cross_RSA`,
   restricted to pairs differing in object, output format, and condition wording.
   Also report format RSA as a nuisance diagnostic. Constant RSMs are undefined
   and cannot be selected. A positive ranking alone is not proof of a concept.
4. Permute condition labels independently within matched object/format/wording
   blocks. For each permutation retain the maximum score across all heads;
   `MaxT_P` is the exploratory one-sided maximum-statistic p-value. With the
   default 199 permutations its smallest value is 0.005. This null relies on
   label exchangeability; shared instruction wording is an important confound.
   Do not use ordinary pairwise-correlation p-values: RSM entries are dependent.
5. Freeze the selected head identities, then audit them on separate objects and
   condition wording. The audit cannot change head ranking. No significance
   claim is made from audit RSA alone.
6. Use the selected heads' respective W_O columns to map their activation means
   into the shared residual coordinates and sum those contributions. Never sum
   raw 128-dimensional head slices from unrelated layers. Projection biases
   are not added once per head.
7. Export `creative_mean.pt` (sum of Creative means), `contrast.pt` (Creative
   minus neutral Standard sum), and individual `head_lL_hH.pt` contrasts. The
   first two add to a chosen residual layer; the last directly intervenes in
   its head before W_O. All work with `activation_steer.py generate` and the
   existing random, suppression, subtraction, temperature, and patch controls.

The paper-style mean construction differs from the contrastive adaptation:
`creative_mean.pt` is not a Creative-minus-Standard vector. Suppression along a
raw mean is not automatically selective removal of creativity. Composite raw
mean and contrast norms differ; alpha=1 is not matched perturbation size between
these two candidates. Each candidate's random controls match its own norm.

## Creativity adaptation and splits

Discovery uses AUT book/fork/paperclip/towel, Creative/Conventional/Effective,
condition wordings 0 and 1, and numbered versus bulleted output instructions:
48 prompts. These conditions all have nonempty instruction suffixes, avoiding
ranking a duplicated neutral prompt as a separate wording. Neutral Standard
activations are collected separately in 16 matched prompts for contrast and
centering; duplicate Standard wordings have equal weight and are not RSA data.

Audit uses umbrella/shoe, both formats, and new condition wording 2: 12 prompts.
The causal validation objects are brick/rope with wording 2 and the existing
numbered-list protocol. This is an object-held-out validation, not a wholly
unseen wording test (wording 2 was used in the representation audit). Audit
objects are refused by the causal validation guard. DAT remains untouched.

This is an instruction-condition representation experiment, not a replication
of the authors' few-shot relational-concept benchmark. It uses explicit style
instructions and two closely related formats; lexical cues, prompt lengths and
instruction following remain alternative explanations. Creativity is not a
known discrete relation with a single correct next token, so causal evaluation
uses AUT generation, proxy scoring and blinded ratings rather than the paper's
target-token probability task. No AIE/function-vector head search is implemented.

Usefulness, validity, noncreative competence, additional task formats and
instruction-free creativity are not established by this pilot. Top-K, injection
layer, alpha and scope must be fixed or selected only on validation; any
subsequent held-out task must remain unused in that selection. Selection does
not require a significance threshold: the pilot runs even if no head has a low
MaxT_P. In that case describe the selected heads as exploratory candidates.

## Run and inspect

```bash
# No model loading:
.venv/bin/python scripts/concept_vectors.py --output-dir /tmp/concept-preview --dry-run
# Full pilot (one A100, at most two hours):
mkdir -p logs
sbatch scripts/snellius_concept.sh llama
```

Settings: `TOP_K=5`, `LAYER=16`, `ALPHA=1`, `REPEATS=2`, `SCOPE=prefill`,
`PERMUTATIONS=199`. These are unoptimized pilot settings. `SCORE=0` exports
format audits and blinded ratings but skips MPNet scoring. The normal scoring
path expects cached MPNet. Outputs are job-specific and discovery refuses
existing output directories to prevent accidental overwrite.

By default the job generates 56 responses for each of three candidates (168
responses total), from 14 arms × 2 objects × 2 repeats. Read:

- `outputs/concept_pilot_JOBID/discovery/head_ranking.csv`: all heads.
- `selected_heads_audit.csv`: frozen selected heads on separate prompts.
- `*.json`: vector construction, norms, heads, source commit and provenance.
- `head_activations.pt` and prompt CSVs: audit/reanalysis inputs.
- `analysis/concept_pilot_JOBID/CANDIDATE/aut_responses.csv`: unit of inference.
- `addition_effect.csv`: paired addition minus Standard, exploratory only.
- `aut_blinded_ratings.csv`: human originality, usefulness and validity ratings.

Compare addition against Standard and random controls, and suppression against
Creative and random suppression. Do not pool raw-mean, contrast and direct-head
runs as independent replications. Report null effects, failures and truncations.
Inspect both discovery and audit RSA before calling a head representation
stable. Even positive RSA plus steering does not establish a creativity circuit.
