# Creative writing audit — 14 October 2026

Scope: paired story generation and response-pooled residual vector extraction for
Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, following Glazer et al., *Creativity
and Hallucinations Share Mechanisms in LLMs*, §3.4, p. 6, Appendix A.1.
This is a documented adaptation, not an exact reproduction of unpublished inputs.

## Verified implementation

| Paper requirement | Current implementation | Assessment |
|---|---|---|
| Same cue words in Creative and Standard stories | `story_prompt` uses the same cue record in both conditions | Matches |
| Creative prompt requests novelty and excludes the obvious storyline | Original benchmark TEST template, with item-specific excluded theme | Matches described task; exact new-paper rendered prompt unavailable |
| Standard asks for a straightforward, coherent story without creativity constraint | Separate Standard template with the same five-sentence limit | Matches description; exact wording is ours |
| Temperature 0.7 for Creative Short Story | Explicit CLI setting in both arms | Matches Appendix A.1 |
| 50 triplets spanning cue relatedness | Frozen adaptation contains 25 familiar and 25 unusual sets | Matches count/intention; groups are qualitative, not measured distance bins |
| One mean-pooled activation per generated response, Eq. 3 | Replay exact prompt/completion IDs; mean response positions separately | Matches |
| Creative mean minus Standard mean, Eq. 4 | `aggregate_layer`; equal cue weights, raw difference | Matches |
| Vectors at every layer | Hook every decoder block in one causal replay per response | Matches; indices are zero-based blocks, before final norm |
| Report experiments across five random seeds | Five independent discovery generations/vector sets and held-out baselines per model | Enables five-seed evaluation; paper does not specify whether extraction itself was repeated across seeds |
| Held-out validation selects layer by steering-induced surprise change | Separate validation cues reserved; no automatic layer selection in these jobs | Not yet implemented/evaluated |

The numerical tests use real tiny random Llama and Qwen2 decoders. They compare
the pooled tensors with explicit response-position means, exclude special tokens,
vary prompt length, verify equal cue weighting and hook cleanup, and exercise the
full generation/replay/artifact path. All 55 regression tests passed before submission.

## Choices and limits

- The authors' 50 triplets are not listed in the PDF, and a matching code release
  was not found. The user approved a documented 50-triplet adaptation. Four
  discovery records come from the original benchmark, 46 are new. The 10
  validation and 10 test triplets are disjoint from discovery. Cue selection was
  frozen before generation and does not use model outputs.
- Creative wording uses the original benchmark's single-user TEST template.
  No creativity-biased system message is added to Standard. The new paper does
  not specify its full rendered chat layout or exact Standard wording.
- Top-p=1, top-k=0 and maximum 512 new tokens are explicit local choices because
  the paper does not specify those values for vector extraction. Its CS4
  top-p=0.95 / temperature=0.8 settings belong to a different task.
- Audit found an implicit Qwen repetition penalty of 1.05 in its local generation
  configuration. Writing generation now explicitly uses repetition_penalty=1.0
  for both models and records it; other existing tasks retain their previous
  defaults. This parameter is unspecified in the paper and is an explicit choice.
- One generation per condition/triplet is collected within each independent
  base seed (42–46). Actual per-pair seeds are derived reproducibly from the base
  seed and cue identifier; the two conditions share a seed.
- Prompt and tokenizer special-token positions, including EOS, are excluded from
  response pooling. The paper does not state its special-token convention.
  Replay captures states at response-token positions, rather than the preceding
  states that predicted those tokens. Causal replay can differ slightly from
  cached generation in floating-point arithmetic.
- The saved vector contrasts instruction-conditioned outputs. It does not
  isolate independently judged successful creativity. No content-quality
  filtering is used. Truncated/empty discovery stories block extraction rather
  than silently changing the selected cue sample.
- These jobs generate 100 discovery stories plus 20 validation and 20 test
  baseline stories per seed, 700 stories/model, 1,400 total. They extract five
  independent sets of all-layer vectors per model. Held-out stories are not
  included in extraction.
- These jobs do not reproduce the paper's component-ablation scan, CS4 judge,
  QA factuality experiment, surprise scorer, layer-selection sweep, or downstream
  AUT/DAT steering evaluation. No claimed best layer or creativity effect follows
  from vector extraction alone.

For subsequent steering, `each_step` edits the final prompt position to influence
the first generated token, then every cached generation step. This is the existing
engine's operational convention; earlier prompt positions are unchanged. The
paper's all-position component mean ablations are a separate intervention from
our directional suppression or prompt-state patching.

Relevant code: `scripts/creative_writing.py`, `scripts/run_creative_writing.py`,
`ActivationEngine.capture_response_means` in `scripts/interventions.py`.
Source: attached [paper](../1840048420.pdf) and
[original benchmark](https://github.com/mismayil/creative-story-gen).

## Output simplification and seed-46 recovery

At the user's request, automatic source copies and file fingerprints were removed. Replay retains complete-pair, cue/split, model revision, and exact-token checks. The last Llama seed generated all 100 discovery stories but could not start extraction because the shared source copy had been deleted. Recovery reuses these stories and runs only extraction plus the missing held-out generations from the main scripts. All 55 regression tests pass after these changes.
