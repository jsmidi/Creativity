"""Reproduce the pilot discovery head-RSA heatmap and selected-head audit panel."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_head_rsa(discovery, output, title):
    """Plot all discovery head scores and compare frozen selected heads against their held-out audit."""
    ranking_path = discovery/'head_ranking.csv'
    audit_path = discovery/'selected_heads_audit.csv'
    ranking = pd.read_csv(ranking_path)
    audit = pd.read_csv(audit_path)
    if ranking.duplicated(['Layer', 'Head']).any() or audit.empty:
        raise ValueError('Require unique head scores and a nonempty selected-head audit')
    selected = audit[['Layer', 'Head', 'Audit_RSA']].merge(
        ranking[['Layer', 'Head', 'Cross_RSA']], on=['Layer', 'Head'], validate='one_to_one')
    if len(selected) != len(audit):
        raise ValueError('Audit head absent from ranking')
    matrix = ranking.pivot(index='Layer', columns='Head', values='Cross_RSA').reindex(
        index=range(int(ranking.Layer.max())+1), columns=range(int(ranking.Head.max())+1))
    output.mkdir(parents=True, exist_ok=True)
    fig, (heat, bars) = plt.subplots(1, 2, figsize=(18, 7.5), constrained_layout=True)
    im = heat.imshow(matrix.to_numpy(), origin='lower', aspect='auto',
                     cmap='coolwarm', vmin=-1, vmax=1)
    heat.scatter(selected.Head, selected.Layer, s=100, facecolors='none', edgecolors='black', linewidths=1.5)
    heat.set_xlabel('Head (zero based)')
    heat.set_ylabel('Layer (zero based)')
    heat.set_title('Discovery: cross-format/item/wording RSA')
    fig.colorbar(im, ax=heat, label='Spearman correlation')
    x = np.arange(len(selected))
    bars.bar(x-.18, selected.Cross_RSA, width=.36, label='Discovery')
    bars.bar(x+.18, selected.Audit_RSA, width=.36, label='Separate items + wording')
    bars.set_xticks(x, [f'L{r.Layer} H{r.Head}' for r in selected.itertuples()])
    bars.set_ylim(-1 if selected[['Cross_RSA', 'Audit_RSA']].min().min() < 0 else -.05, 1)
    bars.set_ylabel('Spearman correlation')
    bars.set_title('Fixed selected heads: descriptive audit')
    bars.legend()
    fig.suptitle(title)
    for extension in ['png', 'pdf']:
        fig.savefig(output/f'head_rsa.{extension}', dpi=180)
    plt.close(fig)
    for path in [ranking_path, audit_path]:
        shutil.copy2(path, output/path.name)
    (output/'manifest.json').write_text(json.dumps(dict(
        discovery_directory=str(discovery), figure='pilot discovery Cross_RSA heatmap and fixed-head Audit_RSA comparison',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ranking_path, audit_path]}), indent=2)+'\n')


def main():
    """Publish task-specific head-RSA PNG/PDF figures and shared copies across decoding temperatures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--analysis-root', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run_root/'plan.json').read_text())
    for task in ['aut', 'dat', 'cdat', 'drat']:
        discovery = args.run_root/'artifacts'/task/'concept_discovery'
        output = args.analysis_root/'head_rsa'/task
        source = 'AUT-to-DAT transfer' if task == 'dat' else task.upper()
        plot_head_rsa(discovery, output, f"{Path(plan['model']).name}: {source}")
        # Prompt-processing RSA is shared across decoding temperatures.
        for temperature in ['t0', 't0.7', 't1']:
            target = args.analysis_root/temperature/task
            target.mkdir(parents=True, exist_ok=True)
            for name in ['head_rsa.png', 'head_rsa.pdf', 'head_ranking.csv', 'selected_heads_audit.csv']:
                shutil.copy2(output/name, target/name)
        print(output/'head_rsa.png', flush=True)
    (args.analysis_root/'head_rsa'/'README.md').write_text(
        'Head-RSA figures reproduce the pilot discovery heatmap and frozen selected-head audit panel. '
        'The same figure appears under each task at t0, t0.7 and t1 because RSA measures prompt processing '
        'before temperature-dependent generation. Each model/task has its own extraction; DAT uses AUT concept prompts. '
        'Black circles mark the selected heads. Undefined correlations are blank. '
        'These correlations concern instruction-condition geometry, not measured creativity.\n')


if __name__ == '__main__':
    main()
