# Small-model behavioral results

Copied on 2026-10-07 from the scored temperature-comparison analysis folders.
Models: Llama-3.1-8B-Instruct (llama31_8b) and Qwen2.5-7B-Instruct (qwen25_7b).
All four tasks (aut, dat, cdat, drat) are included at temperatures 0, 0.7 and 1.
The original files are unchanged. This copy contains existing results; no new
scoring or statistical tests were performed.

## Layout and files

Each model has t0/, t0.7/ and t1/, each containing aut/, dat/, cdat/ and drat/.
These folder labels match the actual Temperature column in the response CSVs.

- TASK_results.png: behavioral score plots. Error bars show one standard
  deviation across responses, not confidence intervals.
- TASK_responses.csv: individual responses, exact prompts, instruction conditions,
  generation settings, pairing identifiers, validity fields and response scores.
  These tables already contain raw response text, so duplicate raw-generation
  CSV/JSON files are unnecessary for interpreting the results.
- creative_vs_controls.csv: existing paired Creative-minus-control comparisons
  against Standard, Effective, Conventional and Boring. Read Metric to identify
  which score was compared; these files do not test every exported metric.
- aut_ideas.csv: parsed individual uses and their automated scores, linked by
  Response_ID and Idea_Index to the response-level table.
- cdat_gates.csv: condition-level appropriateness tests and gated CDAT scores.
- random_baseline.csv (CDAT): the random-list appropriateness scores used by
  the CDAT gates.
- manifest.json and calibration_nouns.txt (CDAT/DRAT): original scoring settings,
  resource hashes and frozen calibration noun pool. Paths inside copied
  manifests and Source_File columns refer to the original experiment locations.
- coverage.csv: all 24 combinations, response counts and original directories.
- file_inventory.csv: original source path, size and SHA-256 for every copied
  artifact. All copies were checked against their source hash.

Mechanistic interpretation directories and files are excluded, including head
RSA/rankings/audits, residual/concept interventions, archived intervention results
and mechanistic comparison plots. Embedding arrays, model weights and code
snapshots are also excluded because this folder is for interpretation, rather
than rerunning scoring. Empty AUT human-rating templates and their private
lookup keys are excluded; automated proxies are not human ratings.

## Reading the paired comparisons

Mean_Difference is Treatment minus Control, with equal weight for each item.
Responses are paired within Run_ID using Block_ID (the item/paraphrase/repeat
block); instruction conditions within a block share a generation seed.
Complete_Pairs counts usable scored pairs; Incomplete_Pairs counts excluded
blocks with a missing condition or score. Missing scores are not zero.

At temperatures 0.7 and 1, CI_Lower and CI_Upper are exploratory 95% percentile
bootstrap intervals. Items and paired generations within items are resampled;
DAT has only one item and therefore uses within-item sampling only. Scope,
Bootstrap_Draws and Bootstrap_Seed describe each comparison. These paired
intervals are unadjusted for multiple comparisons. The paired tables contain
effect estimates and intervals, not paired-test p-values.

At temperature 0, comparisons are descriptive and have no sampling confidence
intervals. Llama's t0 tables contain 60 repetitions per item/condition used as
determinism checks; they are not independent samples. Qwen's t0 tables contain
one response per item/condition. Both sampled temperatures use 60 repetitions
per item/condition for each model.

## Reading the scores

AUT: Semantic_Distance_Proxy is cosine distance between an object and a parsed
use. DB_Rarity_Score compares uses with the normative database.
Relative_Novelty_ICF is log(total ideas / cluster size), fitted within the original
scoring sample; it is not a fixed external novelty norm. The response tables
average the idea scores. Check Idea_Count, Unique_Idea_Count and Format_Valid.
These automated measures do not establish usefulness or human-rated creativity.

DAT: DAT_GloVe_Score uses the official GloVe scoring path. Check Parsed_Words,
Valid_Word_Count and Scorable. DAT_MPNet_Proxy, if populated, is a separate proxy.

CDAT: CDAT_Novelty is 100 times mean pairwise cosine distance among the first
seven valid nouns. CDAT_Appropriateness is 100 times mean (1 + cosine similarity)
to the cue. A successful response requires at least seven valid nouns to score;
Format_Valid separately checks the requested ten-word format. Interpret novelty
with cdat_gates.csv: Welch tests compare cue-level mean appropriateness with
random lists, with Benjamini-Hochberg adjustment within each temperature in the
original scoring invocation. Gate_Passed requires FDR_P < .001 and greater
appropriateness than the random baseline. CDAT_Score is missing if the gate
fails. Paired CDAT_Novelty contrasts are before this condition-level gate.

DRAT: each noun's utility is its maximum cosine similarity to any anchor.
Words survive if utility exceeds the 90th percentile of maximum-anchor utility
in the frozen calibration pool. DRAT_Score is 100 times mean pairwise cosine
distance among surviving nouns, or zero if fewer than three survive. Failed
requests retain missing scores. Inspect Utilities, Utility_Threshold, Survivors
and Format_Valid. This implemented relevance proxy uses the best-matching
anchor; the exact prompt still asks each noun to relate to all anchors.

Association noun validity uses a WordNet noun-lemma whitelist and the first ten
list entries. Original manifests record the calibration and scoring protocol.
Scores and gates remain exactly as originally exported.

## Original analysis sources

- analysis/temperature_comparison_llama31_8b_20261007T094311Z
- analysis/temperature_comparison_qwen25_7b_20261007T122952Z

Use these scored analysis layouts to resolve actual temperatures. The Llama
raw-output sweep originally labels temperature 0.7 as t0; this copy labels it
t0.7 and uses the consolidated greedy results for actual temperature zero.
