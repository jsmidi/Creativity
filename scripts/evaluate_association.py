"""Exploratory CDAT/DRAT scoring with a frozen noun pool and explicit provenance.

See writing/temperature_and_association.md for adaptations and source protocols.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scoring_common import load_responses, parse_dat_words, parse_numbered_list

CDAT = 'Conditional Divergent Association Task'
DRAT = 'Divergent Remote Association Test'


def normalized(values):
    """Normalize finite, nonzero embedding rows; reject unusable embeddings."""
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or (norms == 0).any():
        raise ValueError('Nonfinite or zero embedding')
    return values / norms


def novelty(vectors):
    """Return 100 times mean pairwise cosine distance, requiring two words."""
    if len(vectors) < 2:
        return float('nan')
    similarities = np.clip(vectors @ vectors.T, -1, 1)
    return float(100 * (1 - similarities[np.triu_indices(len(vectors), 1)]).mean())


def drat_score(words, anchors, pool, quantile=.9, minimum=3):
    """Score normalized word vectors after strict max-anchor utility gating.

    pool is a fixed, response-independent random noun embedding matrix. Return
    score, threshold, survivor count and utilities; too few survivors scores zero.
    """
    threshold = float(np.quantile(np.clip(pool @ anchors.T, -1, 1).max(axis=1), quantile))
    utility = np.clip(words @ anchors.T, -1, 1).max(axis=1)
    survivors = words[utility > threshold]
    score = novelty(survivors) if len(survivors) >= minimum else 0.0
    return score, threshold, len(survivors), utility


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjusted p-values, retaining missing tests as NaN."""
    values = np.asarray(pvalues, dtype=float)
    result = np.full(len(values), np.nan)
    indices = np.flatnonzero(np.isfinite(values))
    order = indices[np.argsort(values[indices])]
    if len(order):
        adjusted = values[order] * len(order) / np.arange(1, len(order)+1)
        result[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


def cdat_gates(frame, baseline):
    """Gate each run/condition using cue means to avoid treating repeats as cues.

    Compare appropriateness to frozen random-list baselines with Welch tests;
    adjust across conditions/models within each temperature in this invocation.
    This repeated-condition design is an adaptation of the model-level CDAT gate.
    """
    rows = []
    for keys, group in frame.groupby(['Model', 'Run_ID', 'Temperature', 'Condition']):
        cues = group.groupby('Item')[['CDAT_Novelty', 'CDAT_Appropriateness']].mean().dropna()
        random = baseline[baseline.Item.isin(cues.index)].Appropriateness.to_numpy()
        observed = cues.CDAT_Appropriateness.to_numpy()
        p = float(ttest_ind(observed, random, equal_var=False).pvalue) if len(observed) >= 2 else np.nan
        rows.append(dict(zip(['Model', 'Run_ID', 'Temperature', 'Condition'], keys),
                         Cues=len(cues), Mean_Novelty=cues.CDAT_Novelty.mean(),
                         Mean_Appropriateness=np.mean(observed) if len(observed) else np.nan,
                         Random_Appropriateness=np.mean(random) if len(random) else np.nan, P=p))
    result = pd.DataFrame(rows)
    result['FDR_P'] = result.groupby('Temperature').P.transform(lambda x: bh_adjust(x.to_numpy()))
    result['Gate_Passed'] = (result.FDR_P < .001) & (result.Mean_Appropriateness > result.Random_Appropriateness)
    result['CDAT_Score'] = result.Mean_Novelty.where(result.Gate_Passed)
    return result


def main():
    """Score raw task CSVs and save response audits, calibration and a manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--task', choices=['cdat', 'drat'], required=True)
    parser.add_argument('--noun-vocabulary', type=Path, default=Path('databases/association/nouns.txt'))
    parser.add_argument('--embedding-model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pool-size', type=int, default=5000)
    parser.add_argument('--baseline-draws', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.noun_vocabulary.is_file():
        parser.error('Prepare frozen nouns with scripts/prepare_association_resources.py first')
    if args.output_dir.exists():
        parser.error('Choose a new scoring directory to preserve calibration and results')
    vocabulary = sorted(set(args.noun_vocabulary.read_text().splitlines()))
    if not 10 <= args.pool_size <= len(vocabulary) or args.baseline_draws < 100:
        parser.error('Require 10 <= pool-size <= vocabulary size and baseline-draws >= 100')
    frame = load_responses(args.inputs, CDAT if args.task == 'cdat' else DRAT)
    rng = np.random.default_rng(args.seed)
    pool_words = rng.choice(vocabulary, args.pool_size, replace=False).tolist()
    allowed = set(vocabulary)
    parsed = []
    for text in frame.Response:
        entries = parse_numbered_list(text)[:10]
        # Reconstruct only first ten entries before lexical validation; no cherry-picking later words.
        words = parse_dat_words('\n'.join(f'{i+1}. {word}' for i, word in enumerate(entries)))
        parsed.append(list(dict.fromkeys(w for w in words if w in allowed)))
    anchors = {item: [w.strip() for w in item.split('|')] for item in frame.Item.unique()}
    if any(not all(words) or (len(words) != 1 if args.task == 'cdat' else len(words) < 2)
           for words in anchors.values()):
        parser.error('CDAT needs one cue; DRAT needs pipe-separated anchors')
    texts = sorted(set(pool_words + [w for row in parsed for w in row] + [w for row in anchors.values() for w in row]))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embedding_model, device=args.device)
    vectors = normalized(model.encode(texts, convert_to_numpy=True, show_progress_bar=True))
    lookup = dict(zip(texts, vectors))
    pool = np.stack([lookup[w] for w in pool_words])
    baseline_rows = []
    if args.task == 'cdat':
        for item, cue in anchors.items():
            similarities = np.clip(pool @ lookup[cue[0]], -1, 1)
            for draw in range(args.baseline_draws):
                selected = rng.choice(len(pool), 7, replace=False)
                baseline_rows.append(dict(Item=item, Draw=draw, Appropriateness=float(100*(1+similarities[selected]).mean())))
    scores = []
    for row, words in zip(frame.to_dict('records'), parsed):
        record = dict(Parsed_Words=json.dumps(words), Valid_Nouns=len(words),
                      Format_Valid=len(parse_numbered_list(row['Response'])) == 10 and len(words) == 10)
        valid = row['Status'] == 'ok'
        if args.task == 'cdat':
            record.update(CDAT_Novelty=np.nan, CDAT_Appropriateness=np.nan)
            if valid and len(words) >= 7:
                values = np.stack([lookup[w] for w in words[:7]])
                record.update(CDAT_Novelty=novelty(values), CDAT_Appropriateness=float(
                    100*(1+np.clip(values @ lookup[anchors[row['Item']][0]], -1, 1)).mean()))
        else:
            values = np.stack([lookup[w] for w in words]) if words else np.empty((0, pool.shape[1]))
            score, threshold, survivors, utilities = drat_score(values, np.stack([lookup[w] for w in anchors[row['Item']]]), pool)
            record.update(DRAT_Score=score if valid else np.nan, Utility_Threshold=threshold,
                          Survivors=survivors, Utilities=json.dumps(utilities.tolist()))
        scores.append(record)
    result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(scores)], axis=1)
    args.output_dir.mkdir(parents=True)
    result.to_csv(args.output_dir/f'{args.task}_responses.csv', index=False)
    if baseline_rows:
        baseline = pd.DataFrame(baseline_rows)
        baseline.to_csv(args.output_dir/'random_baseline.csv', index=False)
        cdat_gates(result, baseline).to_csv(args.output_dir/'cdat_gates.csv', index=False)
    np.savez_compressed(args.output_dir/'scoring_embeddings.npz', texts=np.array(texts), vectors=vectors)
    (args.output_dir/'calibration_nouns.txt').write_text('\n'.join(pool_words)+'\n')
    manifest = dict(vars(args), protocol='association-exploratory-v1',
                    noun_sha256=hashlib.sha256(args.noun_vocabulary.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    embedding_sha256=hashlib.sha256((args.output_dir/'scoring_embeddings.npz').read_bytes()).hexdigest(),
                    raw_files={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in frame.Source_File.unique()},
                    drat_quantile=.9, drat_minimum=3, cdat_minimum=7,
                    validity='WordNet noun-lemma whitelist; no POS tagger; first ten entries only',
                    cdat_gate='Welch on cue means versus random lists, BH within temperature, alpha .001')
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    print(f'Saved {len(result)} scored responses to {args.output_dir}')


if __name__ == '__main__':
    main()
