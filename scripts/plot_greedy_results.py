"""Plot completed greedy runs; repeats are determinism checks, not independent samples."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = {'aut': 'Semantic_Distance_Proxy', 'dat': 'DAT_GloVe_Score',
           'cdat': 'CDAT_Novelty', 'drat': 'DRAT_Score'}
CANDIDATES = ['behavioral', 'residual', 'concept_mean', 'concept_contrast', 'concept_head']
BASE = ['Standard', 'Creative', 'Effective', 'Conventional', 'Boring']


def heatmap(frame, title, subtitle, destination, diverging=False, percent=False):
    frame = frame.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if frame.empty:
        return
    values = frame.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(9, len(frame.columns)*1.8), max(4, len(frame)*.43+2)))
    kwargs = {}
    if diverging:
        limit = max(float(np.nanmax(np.abs(values))), 1e-9)
        kwargs = dict(vmin=-limit, vmax=limit)
    elif percent:
        kwargs = dict(vmin=0, vmax=100)
    im = ax.imshow(np.ma.masked_invalid(values), aspect='auto',
                   cmap='RdBu_r' if diverging else 'viridis', **kwargs)
    ax.set_xticks(range(len(frame.columns)), [str(x).replace('_', ' ') for x in frame.columns], rotation=25, ha='right')
    ax.set_yticks(range(len(frame)), [str(x).replace('_', ' ') for x in frame.index])
    for i in range(len(frame)):
        for j in range(len(frame.columns)):
            value = values[i, j]
            label = '—' if np.isnan(value) else (f'{value:.1f}%' if percent else f'{value:.3g}')
            rgba = im.cmap(im.norm(value)) if np.isfinite(value) else (1,1,1,1)
            luminance = .299*rgba[0]+.587*rgba[1]+.114*rgba[2]
            ax.text(j, i, label, ha='center', va='center', color='black' if luminance>.55 else 'white', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.8)
    ax.set_title(title, fontsize=14, pad=20)
    fig.text(.5, .015, subtitle, ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    for suffix in ['png', 'pdf']:
        fig.savefig(destination.with_suffix('.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_run(root):
    task = root.name.split('_')[1]
    metric = METRICS[task]
    out = root/'plots'
    out.mkdir(exist_ok=True)
    frames = []
    for candidate in CANDIDATES:
        path = root/candidate/f'{task}_responses.csv'
        if path.exists():
            frame = pd.read_csv(path, keep_default_na=False)
            frame['Candidate'] = candidate
            frames.append(frame)
    responses = pd.concat(frames, ignore_index=True)
    metrics = [metric] + (['CDAT_Appropriateness'] if task == 'cdat' else [])
    for selected in metrics:
        responses[selected] = pd.to_numeric(responses[selected], errors='coerce')
        # Equal weight per item, after averaging repeated requests within each item.
        good = responses.loc[responses.Status == 'ok']
        item_means = good.groupby(['Candidate','Condition','Item'])[selected].mean()
        matrix = item_means.groupby(['Candidate','Condition']).mean().unstack('Candidate').reindex(columns=CANDIDATES)
        order = [c for c in BASE if c in matrix.index]+[c for c in matrix.index if c not in BASE]
        matrix = matrix.reindex(order)
        heatmap(matrix, f'{task.upper()} greedy: {selected}',
                'Equal-weight item means; missing scores excluded; descriptive only.' +
                (' Novelty alone is not the gated CDAT score.' if selected == 'CDAT_Novelty' else ''),
                out/f'scores_{selected}')
        matrix.to_csv(out/f'scores_{selected}.csv')
    effects = pd.read_csv(root/'effects.csv')
    effects['Contrast'] = effects.Treatment+' − '+effects.Control
    matrix = effects.pivot(index='Contrast', columns='Candidate', values='Mean_Difference').reindex(columns=CANDIDATES)
    heatmap(matrix, f'{task.upper()} greedy: paired effects ({metric})',
            'Treatment minus control; positive values indicate higher scores. Descriptive effects; no sampling confidence intervals.',
            out/'paired_effects', diverging=True)
    audit = pd.read_csv(root/'determinism_audit.csv')
    summaries = []
    for candidate, group in audit.groupby('Candidate'):
        requests = group.Requests.sum()
        summaries.append({'Candidate':candidate,
            'Successful requests':100*group.Successful.sum()/requests,
            'Scored requests':100*group.Scored.sum()/requests,
            'Truncated requests':100*group.Truncated.sum()/requests,
            'Item/condition groups with >1 unique response':100*(group.Unique_Responses>1).mean()})
    quality = pd.DataFrame(summaries).set_index('Candidate').reindex(CANDIDATES).T
    heatmap(quality, f'{task.upper()} greedy: validity and determinism',
            'Request rates use all requests; variability uses item/condition groups. Single-repeat intervention groups cannot test determinism.',
            out/'validity_determinism', percent=True)
    (out/'README.md').write_text(
        f'# {task.upper()} greedy plots\n\nPNG and PDF versions are provided.\n\n'
        'Score heatmaps average successful, scored requests within each item, then give each item equal weight. '
        'Missing scores are excluded; consult the scored-request rates alongside score means. '
        'Paired effects come directly from effects.csv (treatment minus control). '
        'Repeated greedy requests are determinism checks, not independent samples; no sampling confidence intervals are shown. '
        'Intervention groups with one request cannot establish determinism.\n\n'
        + ('CDAT novelty and appropriateness are shown separately. Inspect candidate cdat_gates.csv tables before interpreting novelty as task performance.\n' if task=='cdat' else '')
    )
    print(f'{root}: {len(list(out.glob("*.png")))} plots (PNG + PDF)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='*', type=Path)
    args = parser.parse_args()
    for root in args.runs or sorted(Path('analysis').glob('greedy_*')):
        if (root/'effects.csv').exists():
            plot_run(root)
