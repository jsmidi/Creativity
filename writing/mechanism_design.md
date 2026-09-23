# Searching for a general creativity mechanism

The objective is to explain a shared computation supporting original **and
appropriate** outputs across tasks. Generality is an empirical result, not an
assumption that creativity occupies one vector. Distinguish generality across
items, instructions, task formats, domains, and model families.

## Evidence ladder

1. **Behavior:** establish creative-minus-standard effects with repeated samples,
   an independent conventional condition, and task-specific validity/usefulness.
2. **Candidate representation:** extract task-specific directions, compare their
   stability across items and paraphrases, and construct a shared direction or
   low-rank subspace on development tasks only. Alignment alone is correlational.
3. **Causal transfer:** freeze extraction and hyperparameters, then test addition
   on a genuinely unseen task. Start with AUT, DAT, and metaphor generation to
   vary both task content and output format. Report every source-to-target effect
   in a transfer matrix, including failures. Do not average raw task scales.
4. **Natural contribution:** suppress or patch the candidate during unsteered
   creative prompting. Look for attenuation of the prompt effect with preserved
   basic task competence. Compare damage under standard prompting and random
   suppression, not just creative-versus-suppressed output scores.
5. **Instruction independence:** under the same neutral instruction, sample many
   responses and use blinded quality ratings to identify high/low creativity.
   At the prompt boundary all deterministic activations are identical, so the
   relevant observations must be during generation or controlled teacher forcing.
   Control content, position, length, and teacher-forcing leakage. Test whether
   the same causal process contributes without the word "creative" present.
6. **Computational explanation:** trace which heads/MLPs write and read the
   candidate representation. Ablate selected components, then restore their
   relevant contribution and test recovery across tasks. A representation is a
   starting point; the mechanism explains what computation changes and why.
7. **Replication:** repeat within another model family. Compare functional effects
   and computations, not raw vector coordinates across unrelated hidden spaces.

## Competing mechanisms to distinguish

| Candidate explanation | Discriminating evidence |
|---|---|
| Broad instruction/style following | Transfer to unrelated style instructions or effects only when creative wording is present |
| More random sampling | Compare temperature/entropy-matched output distributions and usefulness; entropy matching is a follow-up beyond the current temperature control |
| Suppression of dominant associations | Reduced probability of common but valid uses, increased probability of uncommon valid uses, preserved unrelated control answers |
| Wider candidate generation | More semantic categories among valid ideas at matched response count and length |
| Different evaluation/selection of ideas | Changed preference between fixed candidate ideas, separable from candidate generation |
| A distributed, multi-stage process | No stable one-dimensional effect, but reproducible temporally or component-specific interventions |

These are hypotheses to test, not conclusions from successful steering. One
tractable mechanistic follow-up is to build held-out pairs of common-valid,
uncommon-valid, and irrelevant-invalid uses for the same object. Measure how
interventions change their conditional log probabilities, controlling token
length and lexical differences. Combine this constrained assay with open-ended
generation and human judgments so the explanation is not specific to an answer
format. Multi-token answer likelihood is not the same quantity as a one-token
logit difference.

## Splits and controls

For a first pilot, extract on AUT book/fork/paperclip/towel with paraphrase 0;
select layer and strength on brick/rope with paraphrase 1; freeze settings and
test on can/stick/belt with paraphrase 2 and unseen DAT/metaphor tasks. This small
example is a software sanity check, not enough objects for a broad generality
claim. Expand the final item/task sample based on pilot uncertainty.

Use a full leave-one-task-out design if several tasks are used in discovery.
Neither vector construction nor layer/strength selection may access the held-out
task. Separately test whether combining task-specific directions improves transfer
or merely cancels incompatible directions. If rank-one steering fails, select a
small subspace rank on validation data rather than interpreting failure as proof
that creativity is noise.

Add control tasks for factual retrieval, simple reasoning, output-format
compliance, verbosity and unrelated style. Multiple norm-matched random vectors
control intervention magnitude; they do not span every possible confound.
Suppression can damage general processing, and full-state patching transfers
more than a creativity-specific feature. Necessity/sufficiency language should
be bounded to the intervention, task, and competence controls actually tested.

The current code implements repeated behavioral generation; residual or individual
head extraction; addition/subtraction; centered projection suppression; random
controls; temperature comparison; prompt-boundary donor patching; and held-out
item/wording checks. It does **not yet** implement automatic circuit discovery,
MLP localization, generation/selection decomposition, spontaneous-creativity
analysis, low-rank subspaces, cross-model alignment, or judge validation. Those
are staged follow-ups once the pilot establishes reliable effects.

## Interpretation

Support for a shared causal contributor requires consistent effects on held-out
tasks plus preserved validity/usefulness, specificity controls, and evidence of
natural contribution. A general mechanistic account additionally identifies the
computations that produce and use it. A low-dimensional control signal can
modulate a complex process; it does not imply creativity is a simple algorithm.

Failure to transfer is scientifically informative. CreativityNeuro reports
DAT activation-steering gains without transfer to AUT and Task Task. Replicating
or explaining that boundary provides a direct target for this thesis.

Sources:

- Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*:
  https://aclanthology.org/2024.acl-long.828/
- Todd et al., *Function Vectors in Large Language Models*:
  https://arxiv.org/abs/2310.15213
- Schapiro et al., *CreativityNeuro*:
  https://arxiv.org/abs/2607.01433
- Olson et al., DAT implementation:
  https://github.com/jayolson/divergent-association-task
