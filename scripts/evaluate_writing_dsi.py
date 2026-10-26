"""Score paired stories with BERT DSI and compare Creative with Standard.

Method: Johnson et al. (2023), doi:10.3758/s13428-022-01986-2.
Independent implementation of the published algorithm. The author's Python
reference is https://osf.io/download/6ye4j/ (John D. Patterson, CC BY-NC-SA 4.0).
We report the reference's 50-token/sentence setting and a 512-token sensitivity
analysis. Punkt is trained separately on each story to avoid input-order effects.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import string

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp


def mean_cosine_distance(vectors):
    """Mean over all distinct vector pairs, without constructing a square matrix."""
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two finite embedding vectors")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero embedding")
    x = x / norms
    n = len(x)
    return float(1 - (np.square(x.sum(axis=0)).sum() - n) / (n * (n - 1)))


def load_stories(root):
    frames = []
    for model in ('llama', 'qwen'):
        for seed in range(42, 47):
            for split in ('discovery', 'validation', 'test'):
                path = root / model / f'seed_{seed}' / split / 'responses.csv'
                frame = pd.read_csv(path, keep_default_na=False)
                frame['Model'] = model
                frame['Seed'] = seed
                frame['Source_File'] = str(path)
                frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    keys = ['Model', 'Seed', 'Split', 'Cue_ID', 'Repeat']
    if frame.duplicated(keys + ['Condition']).any() or frame.Response_ID.duplicated().any():
        raise ValueError('Duplicate story identities')
    for _, group in frame.groupby(keys):
        if set(group.Condition) != {'Creative', 'Standard'} or len(group) != 2:
            raise ValueError('Incomplete condition pair')
        if group.Generation_Seed.nunique() != 1 or group.Item.nunique() != 1:
            raise ValueError('Pair must share cue and generation seed')
    if not frame.Status.eq('ok').all() or frame.Response.str.strip().eq('').any():
        raise ValueError('Failed or empty stories: inspect input before scoring')
    if not frame.Finish_Reason.eq('stop').all():
        raise ValueError('Unexpected/truncated finish reason')
    return frame


class DSIScorer:
    def __init__(self, model_path, device):
        import torch
        from transformers import BertModel, BertTokenizer
        self.torch = torch
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = BertModel.from_pretrained(model_path, local_files_only=True,
                                               attn_implementation='eager').to(device).eval()
        if self.model.config.hidden_size != 1024 or self.model.config.num_hidden_layers != 24:
            raise ValueError('Published DSI requires BERT-large (24 layers, width 1024)')
        # Later blocks cannot affect states 6 and 7 in this feed-forward encoder.
        self.model.encoder.layer = torch.nn.ModuleList(list(self.model.encoder.layer[:7]))

    def sentence_features(self, sentences, max_length):
        torch = self.torch
        encoded = self.tokenizer(sentences, truncation=True, max_length=max_length,
                                 padding=True, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self.model(**inputs, output_hidden_states=True).hidden_states
        result = []
        excluded = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
        for i in range(len(sentences)):
            keep = [j for j, token in enumerate(encoded['input_ids'][i].tolist()) if token not in excluded]
            features = torch.stack([hidden[6][i, keep], hidden[7][i, keep]], dim=1).reshape(-1, 1024)
            result.append(features.cpu().numpy())
        return result

    def score(self, text):
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        segmenter = PunktSentenceTokenizer()
        segmenter.train(text)
        sentences = [s.translate(str.maketrans('', '', string.punctuation)) for s in segmenter.tokenize(text)]
        lengths = [len(self.tokenizer.encode(s, add_special_tokens=True)) for s in sentences]
        if not sentences or max(lengths) > 512:
            raise ValueError('Empty story or a sentence exceeds 512 BERT tokens; no silent full-text truncation')
        full = self.sentence_features(sentences, 512)
        reference = list(full)
        long_indices = [i for i, length in enumerate(lengths) if length > 50]
        if long_indices:
            shortened = self.sentence_features([sentences[i] for i in long_indices], 50)
            for i, features in zip(long_indices, shortened):
                reference[i] = features
        return dict(DSI=mean_cosine_distance(np.concatenate(reference)),
                    DSI_Full=mean_cosine_distance(np.concatenate(full)),
                    Words=len(text.split()), Sentences=len(sentences),
                    BERT_Tokens=sum(length - 2 for length in lengths),
                    Reference_Truncated_Sentences=len(long_indices),
                    Reference_Dropped_Tokens=sum(max(0, length - 50) for length in lengths))


def one_sample(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean, sd = x.mean(), x.std(ddof=1)
    margin = t.ppf(.975, n - 1) * sd / np.sqrt(n)
    if sd == 0:
        stat, pvalue = (0., 1.) if mean == 0 else (np.copysign(np.inf, mean), 0.)
    else:
        stat, pvalue = ttest_1samp(x, 0)
    return dict(Pairs=n, Mean_Difference=mean, CI_Lower=mean-margin, CI_Upper=mean+margin,
                T_Statistic=stat, P_Two_Sided=pvalue)


def summarize(frame, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analyze_effects import holm
    keys = ['Model', 'Seed', 'Split', 'Cue_ID', 'Repeat']
    tables = []
    for metric in ('DSI', 'DSI_Full', 'Words'):
        table = frame.pivot(index=keys, columns='Condition', values=metric)
        table['Difference'] = table.Creative - table.Standard
        table['Metric'] = metric
        tables.append(table.reset_index())
    pairs = pd.concat(tables, ignore_index=True)
    pairs.to_csv(output / 'paired_differences.csv', index=False)
    from writing_metrics import paired_contrast
    crossed = []
    for (model, split), group in frame.groupby(['Model', 'Split']):
        group = group.assign(Base_Seed=group.Seed)
        for metric in ('DSI', 'DSI_Full', 'Words'):
            crossed.append(dict(Model=model, Split=split, **paired_contrast(group, metric)))
    pd.DataFrame(crossed).to_csv(output / 'crossed_bootstrap_effects.csv', index=False)
    rows, seed_rows = [], []
    for (model, split, metric), group in pairs.groupby(['Model', 'Split', 'Metric']):
        cue_means = group.groupby('Cue_ID').Difference.mean()
        seed_means = group.groupby('Seed').Difference.mean()
        row = dict(Model=model, Split=split, Metric=metric, Unit='cue averaged across seeds',
                   Creative_Mean=group.Creative.mean(), Standard_Mean=group.Standard.mean(),
                   **one_sample(cue_means))
        rows.append(row)
        seed_row = dict(Model=model, Split=split, Metric=metric, Unit='seed averaged across cues',
                        **one_sample(seed_means))
        signs = np.array(list(product([-1, 1], repeat=len(seed_means))))
        seed_row['P_Exact_Sign_Flip'] = np.mean(abs((signs * seed_means.to_numpy()).mean(axis=1))
                                              >= abs(seed_means.mean()) - 1e-12)
        seed_rows.append(seed_row)
    result = pd.DataFrame(rows)
    result['P_Holm'] = result.groupby('Metric').P_Two_Sided.transform(lambda x: holm(x.to_numpy()))
    result.to_csv(output / 'paired_tests.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(output / 'seed_paired_tests.csv', index=False)
    summary = frame.groupby(['Model', 'Split', 'Seed', 'Condition'])[['DSI', 'DSI_Full', 'Words']].mean().reset_index()
    summary.to_csv(output / 'seed_means.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, ('discovery', 'validation', 'test')):
        for index, model in enumerate(('llama', 'qwen')):
            g = summary[(summary.Model == model) & (summary.Split == split)]
            for j, condition in enumerate(('Standard', 'Creative')):
                x = index + (j - .5) * .3
                values = g[g.Condition == condition].DSI
                ax.bar(x, values.mean(), yerr=values.std(), width=.28, capsize=4,
                       color=('C0', 'C1')[j], label=condition if index == 0 else None)
        ax.set_xticks([0, 1], ['Llama 8B', 'Qwen 7B'])
        ax.set_title(split.capitalize())
        ax.legend()
    axes[0].set_ylabel('BERT DSI (reference 50-token sentence setting)')
    fig.suptitle('Mean DSI across five seeds; error bars: ±1 seed SD')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'dsi_conditions.{suffix}', dpi=180)
    plt.close(fig)
    lines = ['# DSI: Creative versus Standard', '',
             'Positive differences favor Creative. Conditions are paired by model, seed, split, cue and repeat.', '',
             'Primary tests average the five generation seeds within each cue, then test cue-level differences. '
             'Holm correction covers six model/split comparisons separately for each metric. '
             'Unadjusted 95% t intervals describe cue variation conditional on these five seeds. '
             'Seed-level tests are complementary fixed-cue analyses; neither analysis treats all 700 pairs as independent.', '',
             'crossed_bootstrap_effects.csv also resamples both cue and seed axes jointly, preserving condition pairs; '
             'its percentile intervals are exploratory and unadjusted.', '',
             '| Model | Split | Metric | Cue pairs | Creative − Standard | 95% CI | Holm p |',
             '|---|---|---|---:|---:|---|---:|']
    for r in result.itertuples():
        lines.append(f'| {r.Model} | {r.Split} | {r.Metric} | {r.Pairs} | {r.Mean_Difference:.5f} | '
                     f'[{r.CI_Lower:.5f}, {r.CI_Upper:.5f}] | {r.P_Holm:.4g} |')
    lines += ['', 'DSI follows the authors’ Python reference: uncased BERT-large, sentence-level context, '
              'ASCII punctuation removed before encoding, retained stopwords, token representations from hidden states 6 and 7, '
              'both layers treated as separate vectors, mean cosine distance across all distinct pairs. '
              'The reference caps each sentence at 50 tokens including special tokens. DSI_Full uses up to 512 tokens '
              'and is a sensitivity analysis, not the reference setting. Truncation counts and word lengths accompany each story.', '',
              'Implementation differences: Punkt is trained afresh per story rather than cumulatively across the input file; '
              'dynamic padding replaces fixed padding; algebraically equivalent vector-sum distance computation replaces '
              'explicit pair enumeration; only the first seven encoder blocks run because later blocks do not affect the selected states.', '',
              'Higher DSI indicates greater semantic divergence/integration, not demonstrated overall literary quality. '
              'The Creative instruction also excludes an obvious storyline, so the comparison estimates the full instruction package. '
              'Word-length differences are reported separately; no causal length-adjusted effect is claimed. '
              'The split labels were created for activation experiments, not a preregistered DSI analysis.', '',
              'Sources: https://doi.org/10.3758/s13428-022-01986-2 and https://osf.io/ath2s/.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    print(result.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('outputs/creative_writing_20261014T115031Z'))
    parser.add_argument('--output-dir', type=Path, default=Path('analysis/creative_writing_dsi_20261025'))
    parser.add_argument('--model-path', type=Path, default=Path('models/dsi/bert-large-uncased'))
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    frame = load_stories(args.input_root)
    print(frame.groupby(['Model', 'Split', 'Condition']).size().to_string(), flush=True)
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scorer = DSIScorer(args.model_path, args.device)
    scores = []
    for i, row in frame.iterrows():
        scores.append(scorer.score(row.Response))
        if (i + 1) % 50 == 0:
            print(f'Scored {i + 1}/{len(frame)} stories', flush=True)
    frame = pd.concat([frame, pd.DataFrame(scores)], axis=1)
    frame.to_csv(args.output_dir / 'dsi_responses.csv', index=False)
    metadata = args.model_path / '.cache/huggingface/download/config.json.metadata'
    (args.output_dir / 'scoring_config.json').write_text(json.dumps(dict(
        model='google-bert/bert-large-uncased', model_path=str(args.model_path.resolve()),
        revision=metadata.read_text().splitlines()[0] if metadata.exists() else 'unknown',
        layers=[6, 7], reference_max_sentence_tokens=50, full_max_sentence_tokens=512,
        input_root=str(args.input_root.resolve()), stories=len(frame), device=args.device), indent=2))
    summarize(frame, args.output_dir)


if __name__ == '__main__':
    main()
