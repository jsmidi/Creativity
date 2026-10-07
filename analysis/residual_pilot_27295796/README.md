# Llama residual-stream pilot: 28 September 2026

Slurm job 27295796 completed extraction, generation and automated scoring.
Model: local Llama-3.1-8B-Instruct. All 84 responses finished with `stop`,
passed the numbered-list format check, and supplied 840 ideas. Format validity
does not establish semantic validity or usefulness.

## Intervention

At zero-based decoder layer 16 (the seventeenth block), extract the post-block
residual at the final formatted prompt token:

`v = mean(h(Creative, item) - h(Standard, item))`.

Extraction used book, fork, paperclip, towel and condition wording 0. The vector
norm was 1.3203865. Validation used brick and rope with condition wording 1,
three repeated generation blocks per object and 14 arms. Addition used
`h' = h + v`, only at the prompt boundary; the vector was not unit-normalized.
Controls included two norm-matched random directions, creative/conventional
prompting, higher temperature, subtraction, centered projection suppression,
and donor-state patching. See the generation manifest for exact settings.

## Initial finding

Mean AUT semantic-distance proxy: Standard 0.7247, Creative 0.8260,
Standard plus vector 0.7223. The paired addition-minus-Standard estimate was
-0.002384, with exploratory 95% item/block bootstrap interval
[-0.010097, 0.003925], based on six pairs across only two objects.

This pilot provides no evidence that this layer/strength/prompt-boundary
intervention improves that proxy. It is not evidence that creativity has no
residual-stream representation. Creative prompting itself differs substantially
in the proxy. Suppression also did not visibly remove that advantage in these
means; compare the complete control results before drawing conclusions.

`condition_summary.csv` and `pilot_contrasts.csv` summarize all arms and selected
exploratory paired comparisons. These comparisons are not multiplicity-corrected.
Use `aut_responses.csv` for response-level inference, not the 840 idea rows as
independent samples. Blinded rating sheets still need human originality,
usefulness and validity judgments.

## Next experiment

Compare intervention scope (`each_step` versus this `prefill` run), then a small
prespecified layer/strength grid on validation objects. Freeze settings before
using new objects and DAT for transfer. Do not select settings on DAT and later
label the same DAT evaluation held out. Add unrelated task-competence controls
before claims of a creativity-specific mechanism.

The Word proposal describes an effective, length-matched baseline. This run
instead used the repository's neutral creativity-v2 Standard baseline. Wording
IDs vary condition suffixes only; they do not create entirely new task prompts.
The pilot therefore holds out objects and creative suffix wording, not all
prompt components. These differences must be explicit in the methods.
