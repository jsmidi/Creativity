# Creative writing response vectors

The paired task follows Glazer et al., *Creativity and Hallucinations Share Mechanisms in LLMs*, section 3.4 and p. 6. See [correctness review](creative_writing_correctness_review.md).

The Creative prompt uses the original benchmark's single-user TEST template. Standard requests a straightforward, coherent story. Both use the same three cue words and at most five sentences.

## Inputs and runs

The approved adaptation in `data/creative_writing/adaptation_50.json` has 50 discovery, 10 validation and 10 test triplets. It is not the authors' unpublished 50-triplet selection. Discovery contains 25 familiar and 25 unusual sets; these are qualitative groups, not measured embedding-distance bins. The upstream license is retained with the data.

Submit from the project root:

```bash
sbatch --array=0-4%1 jobs/snellius_creative_writing.sh llama \
  outputs/writing_llama data/creative_writing/adaptation_50.json
sbatch --array=0-4%1 jobs/snellius_creative_writing.sh qwen \
  outputs/writing_qwen data/creative_writing/adaptation_50.json
```

Indices 0–4 map to independent base seeds 42–46. Each seed generates 100 discovery stories, extracts vectors at every layer, and generates 20 validation and 20 test stories. The launcher uses temperature 0.7, top-p 1, top-k 0, repetition penalty 1 and maximum 512 new tokens. Only temperature is explicitly specified by the paper for this task.

All launchers execute `scripts/` directly. No source copies or file fingerprints are saved. Scripts must remain available while jobs run. Slurm keeps its normal internal batch-script record.

To recover an existing seed, submit only its array index with `--export=ALL,RESUME=1`. Resume reuses completed story stages and completed vector stages. It does not overwrite incomplete directories.

## Generate or extract separately

```bash
.venv/bin/python scripts/run_creative_writing.py generate \
  --model models/Llama-3.1-8B-Instruct \
  --cues data/creative_writing/adaptation_50.json --split discovery \
  --temperature 0.7 --seed 42 --max-tokens 512 \
  --output-dir outputs/writing_llama/seed_42/discovery

.venv/bin/python scripts/run_creative_writing.py extract \
  --run-dir outputs/writing_llama/seed_42/discovery \
  --output-dir outputs/writing_llama/seed_42/vectors
```

The smaller pilot cue file remains the Python CLI default, so pass the adaptation explicitly. Use a new output directory for each new stage; `--dry-run` validates inputs without loading a model.

## Outputs and interpretation

- `responses.csv`: ten-column readable stories.
- `responses.jsonl`: exact prompts, token IDs, responses, seeds and settings.
- `manifest.json` and `cues.json`: ordinary metadata, completion marker and cue definitions.
- `vectors/layer_00.pt` / `.json`, etc.: raw vectors and Standard centers.
- `vectors/response_means.pt`: pooled activations for stability analysis.

Extraction checks complete discovery pairs, expected cue splits, model revision, reconstructed prompt IDs and decoded completion text. Empty or truncated stories are rejected unless truncation is explicitly allowed. No content-quality filtering is applied.

Exact saved tokens are replayed with causal attention. Hooks capture post-block residuals before final normalization. Response means exclude prompt positions and tokenizer special tokens, including EOS. States are at response-token positions, not preceding prediction positions. Each story contributes one mean, repeats are averaged within cues, and cues receive equal weight. Vectors are Creative minus Standard means, without unit normalization.

The paired task uses `creative_writing.py` rather than the generic task registry because excluded storylines are cue-specific. Artifacts work with `run_residual.py generate`; use `--scope each_step` for sustained steering.

No best layer is selected automatically. Choose layer and strength on held-out writing before transfer or final testing. Our indices are zero-based decoder blocks. Story scoring, layer-selection sweeps, component ablations and factuality evaluation are separate work. Assess originality together with coherence and task compliance.
