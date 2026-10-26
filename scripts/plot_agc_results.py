"""Plot AGC psychometric condition means and response SDs on their original scales."""
import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITIONS = ['Standard', 'Creative', 'Effective', 'Conventional', 'Boring']
TASKS = {
    'Alternative Uses Task': ('AUT', 'Published per-use rubric; response averages'),
    'Divergent Association Task': ('DAT', 'Exploratory local rubric'),
    'Conditional Divergent Association Task': ('CDAT', 'Exploratory local rubric'),
    'Divergent Remote Association Test': ('DRAT', 'Exploratory local rubric'),
}
MODEL_LABELS = {'Llama-3.1-8B-Instruct': 'Llama 3.1 8B',
                'Qwen2.5-7B-Instruct': 'Qwen 2.5 7B'}


def plot_results(source, output=None):
    source = source / 'condition_summary.csv' if source.is_dir() else source
    frame = pd.read_csv(source)
    required = {'Model', 'Task', 'Condition', 'AGC_Mean', 'AGC_SD', 'Scored',
                'Responses', 'AGC_Scale_Min', 'AGC_Scale_Max'}
    if not required.issubset(frame):
        raise ValueError(f'Require an AGC condition summary with {sorted(required)}')
    if 'Metric' in frame:
        frame = frame[frame.Metric.eq('AGC_Score')].copy()
    frame = frame[frame.Task.isin(TASKS)].copy()
    if frame.empty:
        raise ValueError('No psychometric AGC scores found')
    if frame.duplicated(['Model', 'Task', 'Condition']).any():
        raise ValueError('Choose one run per model/task; multiple runs require explicit aggregation')
    for column in ['AGC_Mean', 'AGC_SD', 'Scored', 'AGC_Scale_Min', 'AGC_Scale_Max']:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    models = sorted(frame.Model.unique())
    colors = ['#2563eb', '#d97706', '#15803d', '#9333ea']
    markers = ['o', 's', '^', 'D']
    offsets = np.linspace(-.12, .12, len(models)) if len(models) > 1 else [0]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    handles, labels = [], []
    for ax, (task, (short, method)) in zip(axes.flat, TASKS.items()):
        data = frame[frame.Task.eq(task)]
        if data.empty:
            ax.set_visible(False)
            continue
        scales = data[['AGC_Scale_Min', 'AGC_Scale_Max']].drop_duplicates()
        if len(scales) != 1:
            raise ValueError(f'Mixed score scales in {task}')
        low, high = scales.iloc[0]
        counts = sorted(data.Scored.unique())
        n_label = str(int(counts[0])) if len(counts) == 1 else '/'.join(str(int(n)) for n in counts)
        for index, model in enumerate(models):
            group = data[data.Model.eq(model)].set_index('Condition').reindex(CONDITIONS)
            x = np.arange(len(CONDITIONS)) + offsets[index]
            valid = group.AGC_Mean.notna() & group.Scored.gt(0)
            color, marker = colors[index % len(colors)], markers[index % len(markers)]
            points, = ax.plot(x[valid], group.loc[valid, 'AGC_Mean'], linestyle='none',
                marker=marker, markersize=7, color=color, zorder=3)
            has_sd = valid & group.Scored.gt(1) & group.AGC_SD.notna()
            ax.errorbar(x[has_sd], group.loc[has_sd, 'AGC_Mean'],
                yerr=group.loc[has_sd, 'AGC_SD'], fmt='none', ecolor=color,
                elinewidth=1.4, capsize=4, capthick=1.4, zorder=2)
            for position, (_, value) in zip(x[valid], group.loc[valid].iterrows()):
                mean = value.AGC_Mean
                spread = value.AGC_SD if value.Scored > 1 and pd.notna(value.AGC_SD) else 0
                ax.annotate(f'{mean:.1f}', (position, mean + spread), xytext=(0, 6),
                    textcoords='offset points', ha='center', color=color, fontsize=8)
            name = MODEL_LABELS.get(Path(model).name, Path(model).name)
            if name not in labels:
                handles.append(points)
                labels.append(name)
        ax.set_title(f'{short} · {method}\nn = {n_label} scored responses per model/condition',
                     loc='left', fontsize=11, pad=12)
        ax.set_xticks(np.arange(len(CONDITIONS)), CONDITIONS, rotation=15)
        ax.set_xlim(-.45, len(CONDITIONS)-.55)
        ax.set_ylim(low-.04*(high-low), high+.04*(high-low))
        ax.set_yticks(np.linspace(low, high, 5))
        ax.set_ylabel(f'AGC score ({int(low)}–{int(high)})')
        ax.grid(axis='y', color='#e5e7eb', linewidth=.8)
        ax.set_axisbelow(True)
    total = int(frame.Responses.sum())
    fig.suptitle(f'AGC-Judge · condition comparison · {total} responses',
                 fontsize=17, fontweight='bold', y=.985)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .947),
               ncol=len(models), frameon=False)
    fig.text(.5, .052,
        'Points: condition means. Whiskers: ±1 sample SD across response scores, not confidence intervals.\n'
        'DAT has one response per condition, so no SD is shown. AUT averages individual-use ratings.',
        ha='center', va='center', fontsize=9, color='#374151')
    seeds = sorted({match.group(1) for name in frame.get('Source_File', [])
                    for match in re.finditer(r'(?:^|/)seed_(\d+)(?:/|$)', str(name))})
    seed_note = ('Seed ' + ', '.join(seeds) + ' · ') if seeds else ''
    fig.text(.5, .015,
        seed_note + 'Descriptive condition means. AUT idea counts and format flags are saved in the response tables.',
        ha='center', fontsize=9, color='#374151')
    fig.tight_layout(rect=(.015, .09, .995, .89), h_pad=3, w_pad=3)
    stem = output if output is not None else source.parent / 'plots' / 'agc_condition_means'
    if stem.suffix.lower() in ('.png', '.pdf'):
        stem = stem.with_suffix('')
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf']:
        path = stem.with_suffix('.'+extension)
        fig.savefig(path, dpi=200, facecolor='white')
        print(path.resolve())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='AGC output directory or condition_summary.csv')
    parser.add_argument('--output', type=Path, help='Output filename stem; writes PNG and PDF')
    args = parser.parse_args()
    try:
        plot_results(args.input, args.output)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
