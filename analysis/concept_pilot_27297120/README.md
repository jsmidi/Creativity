# Opielka-inspired concept-head pilot — completed

Snellius job 27297120 completed on 28 September 2026, using local
Llama-3.1-8B-Instruct. See `writing/concept_vectors.md` for the method, upstream
source commit, and differences from the authors' relational-concept experiment.

The upstream repository is https://github.com/gucioopielka/concept_vectors.
The implementation here uses independent PyTorch hooks rather than their
NNsight setup. This is a creativity instruction-condition adaptation.

## Candidate representations

The fixed top five of 1,024 heads were (zero-based layer, head):
(10,25), (8,15), (6,5), (12,4), (14,24). Discovery cross-pair RSA ranged from
0.663 to 0.817. Each had audit RSA approximately 0.818 on separate objects and
wording. In the 199-permutation blockwise maximum-statistic null, each reached
the minimum reportable p-value of 0.005. This is exploratory evidence for
instruction-condition organization, subject to that null's assumptions; it is
not evidence of a creativity-specific mechanism. Head (6,5) also had notable
full-pair format RSA (0.556), so format invariance is not absolute.

`head_rsa.png` shows discovery scores and the audit. The identical audit scores
are compatible with tied binary design labels and complete separation of same-
versus different-condition similarity ranks in a small audit; they do not imply
identical activation vectors. Selection was frozen before audit and behavior.

## Causal results

Each candidate used 14 conditions × 2 validation objects × 2 repeats = 56
responses, for 168 total. All completed without token-limit truncation and all
passed numbered-list formatting. Semantic validity/usefulness remain unrated.

Primary addition-minus-Standard semantic-distance proxy effects:

| Candidate | Mean effect | Exploratory 95% bootstrap interval |
|---|---:|---:|
| Sum of selected Creative head means, injected into residual layer 16 | -0.00176 | [-0.01579, 0.01116] |
| Sum of selected Creative-minus-Standard head contributions, residual layer 16 | -0.01064 | [-0.02129, approximately 0] |
| Direct pre-o_proj contrast in layer 10, head 25 | +0.00551 | [-0.00813, 0.01891] |

None provides clear evidence for improved semantic distance under this pilot
setting. The individual-head point estimate is positive but uncertain. These
are four paired blocks across only two objects, with no correction across the
behavioral comparisons. Random-direction and suppression comparisons are in
`pilot_contrasts.csv`; do not treat the three runs as independent replications.

All interventions here used alpha=1 and prompt-boundary-only scope. The
summed mean and contrast have different norms. Random directions match each
candidate's own norm. These settings were not optimized.

## Interpretation and next steps

There are candidate heads that reliably distinguish the requested instruction
conditions across the limited object/format/wording variations. We have not
established a vector that causally improves creative quality. Explicit lexical
instruction cues, generic style following, and format effects remain plausible.

A prespecified follow-up can vary prefill versus each_step intervention or
validate injection strength/layer while holding the selected heads fixed.
Collect blinded originality, usefulness, and validity ratings. Keep DAT and
additional task formats outside hyperparameter selection for later transfer.
Avoid announcing a creativity circuit from RSA or this tiny behavior pilot.
