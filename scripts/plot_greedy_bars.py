"""Bar plots for greedy runs, with descriptive spread across item means."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_greedy_results import CANDIDATES, METRICS

ORDER = ['Standard', 'Conventional', 'Effective', 'Boring', 'Creative']
COLORS = dict(zip(ORDER, ['#3274a1', '#e1812c', '#3a923a', '#c03d3e', '#9372b2']))
PANELS = {
    'aut': ['Semantic_Distance_Proxy', 'DB_Rarity_Score', 'Relative_Novelty_ICF'],
    'dat': ['DAT_GloVe_Score', 'DAT_MPNet_Proxy', 'Rarity_Score'],
    'cdat': ['CDAT_Novelty', 'CDAT_Appropriateness'],
    'drat': ['DRAT_Score'],
}


def save(fig, path):
    """Save one figure as PNG and PDF, then release its plotting resources."""
    fig.tight_layout(rect=(0, .09, 1, .92))
    for ext in ['png', 'pdf']:
        fig.savefig(path.with_suffix('.'+ext), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_bars(root):
    """Plot one greedy run with repeats averaged per item before descriptive SD bars."""
    task = root.name.split('_')[1]
    out = root/'plots'
    out.mkdir(exist_ok=True)
    comparison = {}
    for candidate in CANDIDATES:
        source = root/candidate/f'{task}_responses.csv'
        if not source.exists():
            continue
        frame = pd.read_csv(source, keep_default_na=False)
        frame = frame.loc[frame.Status == 'ok'].copy()
        metrics = [m for m in PANELS[task] if m in frame]
        for metric in metrics:
            frame[metric] = pd.to_numeric(frame[metric], errors='coerce')
        # Average repeats before computing descriptive spread across items.
        item = frame.groupby(['Condition','Item'])[metrics].mean()
        means = item.groupby('Condition').mean()
        spread = item.groupby('Condition').std()
        comparison[candidate] = (means, spread)
        conditions = [c for c in ORDER if c in means.index]
        conditions += [c for c in means.index if c not in conditions]
        fig, axes = plt.subplots(1, len(metrics), figsize=(max(8, len(conditions)*.7)*len(metrics), 5.8), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            vals = means.reindex(conditions)[metric]
            sd = spread.reindex(conditions)[metric]
            ax.bar(range(len(conditions)), vals, yerr=sd.fillna(0), capsize=4,
                   color=[COLORS.get(c, '#78909c') for c in conditions], ecolor='#444444')
            ax.set_xticks(range(len(conditions)), [c.replace('_',' ') for c in conditions], rotation=55, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle(f'{task.upper()} greedy — {candidate.replace("_", " ")}', fontsize=15)
        fig.text(.5,.025,'Mean ± 1 SD across item means; repeats averaged within item. Descriptive spread, not a confidence interval.'+
                 ('\nCDAT novelty and appropriateness are separate; novelty alone is not the gated task score.' if task=='cdat' else ''),ha='center',fontsize=10)
        save(fig, out/f'{candidate}_bars')
    metric = METRICS[task]
    fig, ax = plt.subplots(figsize=(12,6))
    names = list(comparison)
    x = np.arange(len(names))
    width = .15
    for i, condition in enumerate(ORDER):
        values = [comparison[c][0][metric].get(condition, np.nan) for c in names]
        errors = [comparison[c][1][metric].get(condition, np.nan) for c in names]
        ax.bar(x+(i-2)*width, values, width, yerr=np.nan_to_num(errors), capsize=3,
               label=condition, color=COLORS[condition], ecolor='#444444')
    ax.set_xticks(x, [c.replace('_',' ') for c in names])
    ax.set_ylabel(metric)
    ax.set_ylim(bottom=0)
    ax.legend(title='Condition', bbox_to_anchor=(1.02,1), loc='upper left')
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    fig.suptitle(f'{task.upper()} greedy — baseline conditions across intervention candidates',fontsize=14)
    fig.text(.5,.025,'Mean ± 1 SD across item means; repeats averaged within item. Descriptive spread, not a confidence interval.',ha='center',fontsize=10)
    save(fig,out/'condition_comparison_bars')
    with (out/'README.md').open('a') as handle:
        handle.write('\nBar plots: candidate-specific bars include all conditions and available legacy metrics. '
                     'condition_comparison_bars compares the five baseline conditions across candidates. '
                     'Colors match the older AUT/DAT figures. Error bars show ±1 sample SD across item means '
                     '(repeats averaged first), not sampling uncertainty; with one item, no SD is available.\n')
    print(f'{task.upper()}: {len(comparison)+1} bar plots, PNG + PDF')


def main():
    """Find legacy greedy runs and render their candidate and condition summaries."""
    for root in sorted(Path('analysis').glob('greedy_*')):
        if (root/'effects.csv').exists():
            plot_bars(root)


if __name__ == '__main__':
    main()
