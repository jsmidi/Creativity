# Behavioral and mechanistic correctness review

Reviewed 7 October 2026. Scope: active Python analysis/generation code, current
Slurm settings, saved greedy results, and Opielka's paper and upstream code.
No implementation changes or new GPU experiments were made.

The behavioral contrast and residual extraction have the correct signs and
units. The concept-vector pipeline implements the core Opielka-style capture,
RSA and projection operations. Two implementation issues need correction
before the default sampled intervention analyses and composite suppression can
be relied on. The current evidence supports task-score and instruction-condition
claims; a creativity-specific mechanism remains unestablished.

## Confirmed implementation issues

### 1. Unrelated temperature arms prevent matched-condition inference

`scripts/analyze_effects.py:36–39` validates decoding settings on the entire
frame before selecting treatment/control. Default intervention files include
Standard_high_temperature at T=1 alongside ordinary arms at T=0.7. Consequently,
Creative versus Standard and Standard_add_1 versus Standard both raise
`ValueError: Analyze one Temperature setting at a time.` This affects the
default effect stages in `snellius_residual.sh` and `snellius_concept.sh`.
All-greedy jobs omit this arm and are unaffected.

Reproduced using `analysis/residual_pilot_27409580/aut_responses.csv`: the full
frame raises, whereas selecting Creative and Standard first succeeds with mean
difference -0.0082047477, 15 complete pairs and five items. This is an exploratory
AUT semantic-distance difference, not a human creativity effect.

Correction: retain the single-run guard, select the requested arms, then check
matched decoding settings and detect greedy decoding. Continue rejecting mixed
settings within an ordinary prompt contrast. An intentional temperature contrast
would need a separate explicit policy.

### 2. Composite suppression centers a full residual on selected-head contributions

`scripts/concept_vectors.py:385–400` saves the projected Standard contribution
of the selected heads as `center` for both composite artifacts, which are
injected into the full post-block residual at `args.layer`.
`scripts/interventions.py:58–66` computes:

```text
h' = h - dot(h - center, unit_vector) * unit_vector
```

For baseline-centered suppression of that residual, the center must be the
mean full Standard residual at the target layer/token. The selected-head sum
omits embeddings, other heads and MLP contributions, and can combine heads
from different layers. Equal dimensionality does not imply equal baseline
coordinates. At the true Standard residual mean mu, the current implementation
moves the state by `-dot(mu - component_center, unit_vector) * unit_vector`.
It can therefore remove baseline processing as well as the condition contrast.

A CPU reproduction with a two-layer random Llama moved its neutral residual
mean by L2=0.0050619664 with the component center, versus zero with the full
residual center. This demonstrates the mismatch, not its trained-model magnitude.

Correction: capture the full Standard residual at the composite injection
layer on the extraction prompts and save that as the suppression center. Keep
the projected component mean separately. Composite addition remains well-defined.
Ordinary residual and individual-head artifacts already center in their own
target coordinates. For cross-task or each-step suppression, also assess whether
the extraction-task prompt center suits the target activation distribution.

### 3. Compliance flags do not fully constrain scoring

`scripts/evaluate_aut.py:68–75` marks non-ten-idea lists Format_Valid=False but
scores every parsed idea in Status=ok responses. A one-idea response can enter
the main mean contrast. Duplicate ideas and invalid alternative uses are not
excluded by the ten-entry format flag. This is a scoring-policy gap when claiming
compliant task performance, rather than an arithmetic error.

`scripts/evaluate_dat.py:24–31` parses all numbered entries before selecting the
first seven valid unique words. An eleventh entry can rescue a list whose first
ten contain only six valid words. A focused reproduction confirmed that an
11-entry answer is then Scorable=True. CDAT/DRAT already restrict lexical
selection to the first ten entries.

Define eligibility before confirmatory analysis. Restrict DAT to the requested
ten positions while retaining official first-seven-valid scoring. Report score
contrasts alongside failure, format, noun-validity and truncation rates. For AUT,
collect validity, originality and usefulness ratings and show a compliance
sensitivity analysis. Replacing all missing scores with zero changes the estimand
and is not automatically a valid correction.

## Behavioral comparison

`scripts/experiment.py:32–54` pairs conditions by item, wording and repeat,
assigns the same generation seed within each block and supports randomized
request order. Local generation resets seeds and sets sampling filters explicitly.
Shared seeds define matched random-number blocks; they do not ensure identical
outputs across prompts or a strong correlation between paired scores.

AUT averages idea scores within each response before inference.
`scripts/analyze_effects.py:43–64` computes treatment minus control on complete
pairs, averages within items and gives items equal weight. At nonzero temperature,
it resamples items and paired differences within items. This is a reasonable
exploratory estimator for its stated scope. DAT has one item, so its interval
concerns generations of that fixed task. Complete-pair exclusion conditions the
effect on successful scoring and can bias broader performance claims if failures
differ by condition. Small, chosen item sets and multiple exploratory comparisons
also limit confirmatory interpretation.

Creative minus Standard is the primary incremental instruction effect. The other
controls ask different questions and should remain separate. The base prompts
already demand alternative or divergent outputs; this measures adding a suffix
to those instructions, rather than creative behavior under instruction-free prompts.

At T=0, the analyzer correctly omits sampling bootstrap intervals. All 60
successful greedy responses were identical within every saved behavioral
item/condition group. The distinct design comprises five AUT items, one DAT
prompt, ten CDAT cues and five DRAT anchor sets per condition. These repeats
are determinism checks and do not expand the tested item population. Current
greedy plot SD bars describe spread across item means, not sampling uncertainty.

DAT's dictionary/embedding validation and mean cosine distance among the first
seven qualifying unique words, times 100, match the [authors' scorer](https://github.com/jayolson/divergent-association-task/blob/master/dat.py).
Dictionary membership does not independently establish nounhood.

AUT's primary measure (`scripts/evaluate_aut.py:41–46`) is MPNet cosine distance
between the object label and each generated use, averaged within the response.
A larger score can reflect originality, irrelevance, different wording or loss
of reference to the object. It does not measure usefulness or establish validity
as an alternative use. Database rarity is another proxy; within-sample DBSCAN
novelty additionally depends on which answers/conditions enter scoring.

CDAT's conditional novelty logic follows the [paper's section 4.2](https://arxiv.org/html/2601.20546v1#S4.SS2).
The cue bank, noun whitelist and condition-specific testing are adaptations.
Interpret novelty with both the gate and the appropriateness change. DRAT's
closest-anchor utility, random-noun 90th percentile, strict threshold and
three-survivor minimum match its [published scoring rule](https://arxiv.org/html/2605.13450v1).
The score measures sufficient relevance to at least one anchor, although the
prompt requests relation to all anchors; describe that distinction accurately.

## Concept vectors and Opielka's method

Compared directly with upstream commit
`dcbcd0ef16d8a6c1af8b0a2378b7717a9ba66c42`, particularly
[calculate_RSA.py](https://github.com/gucioopielka/concept_vectors/blob/dcbcd0ef16d8a6c1af8b0a2378b7717a9ba66c42/src/calculate_RSA.py),
[query_utils.py](https://github.com/gucioopielka/concept_vectors/blob/dcbcd0ef16d8a6c1af8b0a2378b7717a9ba66c42/src/utils/query_utils.py)
and [eval_utils.py](https://github.com/gucioopielka/concept_vectors/blob/dcbcd0ef16d8a6c1af8b0a2378b7717a9ba66c42/src/utils/eval_utils.py).
Paper references: [2025, sections 2.8–2.9](https://arxiv.org/html/2503.03666v1)
and [2026, sections 2.1.4–2.1.5 and 3.1](https://arxiv.org/html/2602.22424v1).

| Operation | Assessment |
|---|---|
| Last-token head capture | Correct: input of o_proj, split by query heads; supports Llama GQA. |
| Cosine similarities and Spearman RSA | Correct: off-diagonal pairs, average ranks for ties, constant heads excluded. |
| Head ranking | Adapted: Cross_RSA across different items, formats and wordings; Full_RSA is exported but not used for selection. |
| Vector projection | Correct: selected head means mapped through their W_O columns, then summed in residual coordinates. |
| creative_mean.pt | Paper-style mean construction, adapted to explicit creative instructions and pooled formats; not a condition difference. |
| contrast.pt | Contrastive extension: projected Creative mean minus projected Standard mean. |
| Individual head artifact | Contrastive extension in pre-o_proj coordinates. |
| Evaluation | Adapted: generated task scores rather than target-token probability in relational ICL. |

The RSA design labels Creative, Conventional and Effective. It finds heads
organizing those instruction conditions, rather than specifically isolating
creativity or learning four independent contrasts. Numbered/bulleted lists are
a limited format manipulation. Audit objects are held out, but audit wording
is also used in validation. These checks do not establish instruction independence
or broad format invariance.

The MaxT permutations account for searching heads under their label-exchangeability
null. They test condition-related geometry and cannot distinguish creative
computation from recognizing condition words. High RSA or a small permutation
p-value therefore does not by itself identify a creativity mechanism. More
formats, matched-length instructions, unrelated style controls and creativity
measurements without explicit creative wording would address this interpretation.

Mean and contrast vectors have different norms; alpha=1 does not match perturbation
size between candidates. Each candidate's random controls match its own norm.
Direct candidate comparisons need matched magnitudes and frozen validation-selected
settings. The code does not implement the paper's AIE-selected function-vector
comparison or its separate extraction-format portability assay.

## Residual contrast

`scripts/activation_steer.py:111–123` computes:

```text
v_layer = mean_i[h_layer(last_prompt_token, Creative_i)
               - h_layer(last_prompt_token, Standard_i)]
```

Item and base instruction are matched. Whole-residual capture is post-decoder-block;
layer 16 is zero based, hence the seventeenth block. Capture/intervention coordinates
and token positions agree; addition/subtraction signs are correct. Direct-head
capture and intervention likewise target the same pre-o_proj slice.

This contrasts prompt-induced states at one layer/position, rather than measured
high- and low-quality answers. Extraction never selects on generated-answer
quality. Appending a suffix also changes prompt length, token positions and
generic instruction following. A prefill intervention tests that boundary,
not the whole subsequent creative process.

Prompt-boundary patching transfers the donor state while earlier recipient
states and KV caches remain from the recipient prompt. It is a localized
intervention, not reconstruction of the complete donor condition.

To assess natural contribution, compare addition, subtraction and suppression
with matched random controls, preserved competence and Standard suppression.
A useful interaction is `(Creative - Creative_suppress) -
(Standard - Standard_suppress)`, with corresponding random-suppression checks.
Coarse residual steering does not itself localize a circuit.

## Existing greedy results

Behavioral effects from jobs 27552571 (AUT), 27552572 (DAT), 27552573 (CDAT)
and 27552574 (DRAT):

| Task metric | Creative minus Standard |
|---|---:|
| AUT Semantic_Distance_Proxy | -0.013576 |
| DAT official GloVe score | +1.620377 |
| CDAT novelty | +2.461392 |
| DRAT score | -14.774582 |

All five behavioral CDAT conditions passed the local gate. Creative appropriateness
was lower than Standard: 138.028573 versus 139.821951. Thus the novelty gain
coexists with reduced contextual similarity. Every AUT behavioral list passed
the ten-entry format flag; 1,080/3,000 CDAT and 300/1,500 DRAT behavioral requests
failed their format flag despite receiving scores. Those flags include lexical
counts and do not themselves establish semantic invalidity.

Interventions do not show a clear shared causal pattern. DAT residual addition
raises the score by +0.264526, but both random additions produce exactly that
same gain; Creative suppression has zero effect. DRAT residual addition lowers
its score by -0.329452. These observations do not establish specific, general
creativity enhancement. Small/null greedy effects also do not exclude effects
at other validated layers, strengths or sampling settings.

The reviewed core Python files match the frozen source snapshots for all four
greedy jobs. Stored results use Llama-3.1-8B-Instruct. Current uncommitted Slurm
scripts instead select Llama-3.1-70B-Instruct, still at layer 16 and alpha 1.
New 70B runs require their own extraction/validation and clear model identification.

## Verification and next work

All 44 existing CPU tests passed, including small real-Llama tests of head
capture, projection, hook cleanup, intervention scope and logits. Focused
reproductions additionally confirmed the two primary issues and malformed-list
behaviors above. No new trained-model GPU experiment or reproduction of the
authors' benchmark was run.

1. Correct decoding-setting validation after selecting comparison arms, with
   a regression for matched arms in a file containing a separate temperature arm.
2. Correct composite residual suppression centering; regenerate affected arms
   and compare with random and Standard suppression controls.
3. Freeze compliance rules and primary endpoints; collect blinded AUT validity,
   originality and usefulness ratings.
4. Use independent nonzero-temperature seed blocks for sampling effects, expand
   held-out items and freeze validation-selected layers/strengths before final testing.
5. Assess transfer, specificity and natural contribution before claiming a
   general creativity mechanism.
