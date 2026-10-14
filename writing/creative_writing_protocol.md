# Creative writing response vectors

This pipeline implements the response-pooling approach described in Glazer et al.,
*Creativity and Hallucinations Share Mechanisms in LLMs*, sections 3.4 and p. 6
([paper](../1840048420.pdf)). It is separate from the existing final-prompt-token
extractor. No jobs are submitted by the Python code.

## Prompts and cues

`scripts/creative_writing.py` defines the paired task. The Creative instruction is
the original benchmark's single-user-message `TEST_USER_INSTRUCTION_TEMPLATE`,
which requests at most five sentences, includes three cue words, and excludes
their obvious storyline. The Standard instruction is our implementation of the
description in Glazer et al.: a straightforward, coherent story using the same
words and sentence limit, without the creativity or excluded-storyline requirement.
Neither arm receives an additional system message. This is an explicit prompt
choice; the new paper does not provide its complete rendered prompt or chat layout.

Sources: [original benchmark](https://github.com/mismayil/creative-story-gen),
[prompt templates](https://github.com/mismayil/creative-story-gen/blob/e0a187cd9a54c7a7772ace4c655526efd23bacea/src/prompts.py),
and its `data/pilot_data.json` / `data/test_data.json` at that pinned commit.
The upstream MIT license is retained in `data/creative_writing/LICENSE.benchmark`.

The bundled `data/creative_writing/pilot_cues.json` contains ten original cues:
four discovery, three validation, three test. These split assignments are ours.
Only the discovery set has a mix of high/low cue relatedness; the tiny validation
and test sets are low-relatedness examples and are suitable for pipeline checks,
not a balanced thesis evaluation.

**This is not the paper's 50-triplet selection.** The attached paper does not list
that selection. Before the thesis experiment, supply a larger frozen cue file,
with at least 50 discovery triplets spanning relatedness, plus distinct validation
and test triplets. Every record must contain `id`, three `items`, `boring_theme`
and `split` (`discovery`, `validation`, or `test`). IDs and unordered triplets must
be unique across the entire file. Metadata should describe selection and provenance.

The paired task has its own configuration rather than a generic entry in
`task_config.py`: the general task registry appends condition suffixes and cannot
express the item-specific excluded storyline. Use the writing CLI below for these
paper-style extraction prompts. Existing task generation remains unchanged.

## Generate and extract

Inside a GPU allocation, from the project root:

```bash
.venv/bin/python scripts/run_creative_writing.py generate \
  --model models/Llama-3.1-8B-Instruct \
  --cues data/creative_writing/pilot_cues.json \
  --split discovery --temperature 0.7 --repeats 1 \
  --output-dir outputs/writing_llama_pilot/discovery

.venv/bin/python scripts/run_creative_writing.py extract \
  --run-dir outputs/writing_llama_pilot/discovery \
  --output-dir outputs/writing_llama_pilot/vectors
```

Replace the model with `models/Qwen2.5-7B-Instruct` and choose a new output folder
for Qwen. `--dry-run` validates generation inputs and prints paired prompts without
loading weights; extraction dry runs validate the saved discovery run. Output
directories must be new, preventing accidental overwrites.

The generation defaults are temperature 0.7, one response per cue and condition,
top-p 1, top-k 0, maximum 256 new tokens, seed 42. Temperature 0.7 follows the
paper's short-story setup; remaining defaults are explicit implementation choices,
not recovered unpublished settings. Increase repeats if wanted. Complete pairs
share a seed and their execution order is shuffled reproducibly.

Generation saves:

- `responses.csv`: readable stories, prompts, conditions, seeds, finish reasons.
- `responses.jsonl`: the same records plus exact prompt and completion token IDs.
- `manifest.json`, `cues.json`, `code/`: settings, model/template identity, file
  hashes, cue selection and source snapshot. The completion marker is only set
  after every planned generation returns.

Empty responses are retained and extraction rejects them. Length-capped responses
are retained and extraction rejects them by default: raise `--max-tokens` and
generate a new run, or explicitly use extraction `--allow-truncated` and report it.
Missing cue words or other content defects are not silently filtered; review the
CSV and measure compliance separately before interpreting creativity.

For Snellius, create `logs/` if needed and submit from the project root:

```bash
sbatch jobs/snellius_creative_writing.sh llama outputs/writing_llama_pilot
sbatch jobs/snellius_creative_writing.sh qwen outputs/writing_qwen_pilot
```

Optional third argument: a custom cue JSON. Each launcher freezes code/cues and
runs discovery generation followed by extraction on one A100. These commands
submit new jobs; setup and tests do not execute them.

## Exact extraction semantics

Extraction teacher-forces the **saved prompt and completion token IDs**, without
re-tokenizing decoded stories or sampling new responses. Causal attention means
each response token retains its original prefix. A hook captures the residual
after each decoder block, before the model's final normalization. We pool states
at response-token positions, not the preceding states used to predict those tokens.
All prompt positions and tokenizer special tokens, including terminal EOS, are
excluded. Replay is mathematically the same causal computation but may have small
floating-point differences from cached generation kernels.

Each story contributes one token mean, regardless of its length. Repetitions are
averaged within a cue and each cue is weighted equally. At each layer:

`vector = mean(creative response means) - mean(standard response means)`.

The Standard response mean is also saved as `center`. Vectors use raw differences;
there is no unit normalization. This direction contrasts instruction-conditioned
outputs, not independently rated successful versus unsuccessful creativity.

Extraction checks complete matched discovery pairs, response/cue hashes, model
revision and chat-template identity, reconstructed prompt IDs, and decoded
completion text. Local model metadata does not hash every checkpoint weight;
preserve the local checkpoint unchanged between generation and extraction.

Extraction saves `layer_00.pt` / `.json` etc. (all blocks by default; restrict with
`--layers 14 16`), `response_means.pt` for stability analyses, a manifest and source
snapshot. Layer indices are **zero-based decoder blocks**. No layer is selected
automatically. Validate layers/strengths on held-out writing and freeze the choice
before testing or transfer. The reported paper layers require checking its indexing
convention; do not assume layer 14 or 16 is optimal for this pilot.

## Transfer to the existing tasks

The `.pt` artifacts are compatible with `run_residual.py generate`. For example,
after selecting a layer, a pilot AUT transfer can use:

```bash
.venv/bin/python scripts/run_residual.py generate \
  --model models/Llama-3.1-8B-Instruct \
  --artifact outputs/writing_llama_pilot/vectors/layer_14.pt \
  --task 'Alternative Uses Task' --split validation --paraphrases 1 \
  --alphas 0.5 1 2 --repeats 5 --temperature 0.7 \
  --scope each_step --omit-temperature-control \
  --output-dir outputs/writing_llama_pilot/aut_transfer
```

`each_step` edits the final prompt position for first-token prediction, then every
cached response step; earlier prompt positions remain unchanged. The existing
runner also includes directional suppression and prompt-donor patching extensions.
Those are not the paper's component mean ablations. Prioritize the addition and
subtraction contrasts when comparing with its activation-steering results.

Story scoring and automatic layer/alpha selection are not part of this setup.
The existing AUT/DAT scorers are not story-creativity measures. Assess writing
originality/surprise together with coherence and cue/sentence compliance on held-out
items; semantic surprise alone can reward incoherent stories. Do not choose a
layer using the final AUT/DAT/CDAT/DRAT test outcomes.
