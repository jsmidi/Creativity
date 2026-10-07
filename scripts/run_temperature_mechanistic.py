"""Extract prompt-state vectors once; test frozen vectors at sampled temperatures."""
import argparse
import csv
import json
from pathlib import Path
import pandas as pd
from run_greedy_task import NAMES, DISCOVERY, AUDIT, METRICS, run, score
from task_config import get_task_config
from analyze_effects import paired_effect

CANDIDATES = ['residual', 'concept_mean', 'concept_contrast', 'concept_head']
TEMPERATURES = [0, 0.7, 1]
CONTROLS = ['Standard', 'Effective', 'Conventional', 'Boring']


def contrasts(frame, metric):
    """Estimate the prespecified instruction and intervention comparisons from complete paired blocks."""
    pairs = [('Creative', c) for c in CONTROLS]
    pairs += [('Standard_add_1', 'Standard'), ('Creative_subtract_1', 'Creative'),
              ('Creative_suppress', 'Creative'), ('Standard_suppress', 'Standard'),
              ('Standard_patch_creative', 'Standard'), ('Creative_patch_standard', 'Creative')]
    for index in range(2):
        pairs += [(f'Standard_random{index}_1', 'Standard'),
                  (f'Creative_random_suppress{index}', 'Creative'),
                  ('Standard_add_1', f'Standard_random{index}_1'),
                  ('Creative_suppress', f'Creative_random_suppress{index}')]
    rows = []
    for treatment, control in pairs:
        try:
            rows.append(dict(**paired_effect(frame, metric, treatment, control), Analysis_Status='ok'))
        except ValueError as exc:
            if str(exc) != 'No complete scored pairs for requested contrast.':
                raise
            rows.append(dict(Metric=metric, Treatment=treatment, Control=control,
                             Analysis_Status=str(exc)))
    return pd.DataFrame(rows)


def effect_plot(effects, output, title):
    """Plot intervention differences and unadjusted bootstrap intervals; greedy runs show points only."""
    import matplotlib.pyplot as plt
    import numpy as np
    valid = effects[effects.Analysis_Status.eq('ok')].copy()
    valid = valid[valid.Treatment.ne('Creative')]
    labels = [f'{r.Treatment} minus {r.Control}' for r in valid.itertuples()]
    y = np.arange(len(valid))
    fig, ax = plt.subplots(figsize=(11, max(5, len(valid)*.35)))
    for position, row in zip(y, valid.itertuples()):
        ax.plot([row.CI_Lower, row.CI_Upper], [position, position], color='C0')
        ax.plot(row.Mean_Difference, position, 'o', color='C0')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel('Paired score difference; exploratory 95% bootstrap interval'
                  if valid.CI_Lower.notna().any() else 'Paired score difference; greedy decoding, descriptive only')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args():
    """Parse the phase and validate its Slurm array index and repetition count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['extract', 'intervene', 'summarize'], required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=60)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Require positive repeats')
    limit = 4 if args.phase == 'extract' else 48
    if args.phase != 'summarize' and not 0 <= args.index < limit:
        parser.error(f'Index must be 0..{limit-1}')
    return args


def summarize_runs(args):
    """Combine candidate effect tables per task and temperature; document interpretation."""
    tasks = list(NAMES)
    if args.dry_run:
        print('Summarize all four candidates for each task and temperature')
        return
    for temperature in TEMPERATURES:
        for task in tasks:
            root = args.analysis_root/f't{temperature}'/task
            tables = []
            for candidate in CANDIDATES:
                frame = pd.read_csv(root/candidate/'effects.csv')
                frame.insert(0, 'Candidate', candidate)
                tables.append(frame)
            pd.concat(tables, ignore_index=True).to_csv(root/'mechanistic_effects.csv', index=False)
            (root/'mechanistic_README.md').write_text(
                'Frozen model-specific vectors extracted from held-out prompts; injection at layer 16, '
                'final prompt token, prefill only. One repetition at temperature zero, 60 at sampled temperatures; '
                '15 conditions per candidate. '
                'Alpha 1; two norm-matched random vectors. effects.csv within each candidate and '
                'mechanistic_effects.csv report paired bootstrap intervals, without multiple-comparison correction. '
                'Composite suppression uses the full Standard residual center, corrected from the older greedy run. '
                'This change matters for comparing suppression with the old temperature-zero results. '
                'DAT concept vectors are AUT-to-DAT transfer. CDAT requires candidate-specific gate checks.\n')
    return


def extract_vectors(args):
    """Extract one task residual and RSA-selected concepts from held-out prompts."""
    tasks = list(NAMES)
    task = tasks[args.index]
    concept_task = 'aut' if task == 'dat' else task
    root = args.run_root/'artifacts'/task
    print(json.dumps(dict(phase=args.phase, task=task, model=args.model,
                         residual_discovery=DISCOVERY[task], concept_discovery=DISCOVERY[concept_task],
                         concept_audit=AUDIT[concept_task], layer=16, wordings=[1, 2]), indent=2), flush=True)
    if args.dry_run:
        return
    root.mkdir(parents=True, exist_ok=False)
    run('activation_steer.py', 'extract', '--model', args.model, '--artifact', root/'residual.pt',
        '--tasks', NAMES[task], '--items', *DISCOVERY[task], '--paraphrases', 1, 2,
        '--baseline', 'Standard', '--layer', 16)
    run('concept_vectors.py', '--model', args.model, '--task', NAMES[concept_task],
        '--output-dir', root/'concept_discovery', '--items', *DISCOVERY[concept_task],
        '--audit-items', *AUDIT[concept_task], '--paraphrases', 1, 2,
        '--audit-paraphrase', 0, '--top-k', 5, '--layer', 16, '--permutations', 199)
    return


def run_intervention(args):
    """Generate, score and compare one frozen candidate at one temperature."""
    tasks = list(NAMES)
    task = tasks[args.index // 12]
    temperature = TEMPERATURES[(args.index % 12) // 4]
    candidate = CANDIDATES[args.index % 4]
    items = get_task_config(NAMES[task])['items']
    repeats = 1 if temperature == 0 else args.repeats
    artifacts = args.run_root/'artifacts'/task
    discovery = artifacts/'concept_discovery'
    if candidate == 'concept_head':
        if args.dry_run:
            artifact = discovery/'selected_head.pt'
        else:
            with (discovery/'head_ranking.csv').open() as handle:
                head = next(csv.DictReader(handle))
            artifact = discovery/f"head_l{head['Layer']}_h{head['Head']}.pt"
    else:
        artifact = {'residual': artifacts/'residual.pt', 'concept_mean': discovery/'creative_mean.pt',
                    'concept_contrast': discovery/'contrast.pt'}[candidate]
    raw = args.run_root/f't{temperature}'/task/candidate
    output = args.analysis_root/f't{temperature}'/task/candidate
    print(json.dumps(dict(task=task, temperature=temperature, candidate=candidate,
                         responses=len(items)*15*repeats, artifact=str(artifact),
                         analysis=str(output)), indent=2), flush=True)
    if args.dry_run:
        return
    if output.exists() or raw.exists():
        raise ValueError('Choose unused candidate directories; do not overwrite results')
    run('activation_steer.py', 'generate', '--model', args.model, '--artifact', artifact,
        '--task', NAMES[task], '--items', *items, '--paraphrases', 0, '--split', 'validation',
        '--temperature', temperature, '--omit-temperature-control', '--repeats', repeats,
        '--alphas', 1, '--random-vectors', 2, '--scope', 'prefill', '--max-tokens', 800,
        '--output-dir', raw)
    score(task, raw, output)
    frame = pd.read_csv(output/f'{task}_responses.csv', keep_default_na=False)
    effects = contrasts(frame, METRICS[task])
    effects.insert(0, 'Model', args.model)
    effects.insert(1, 'Task', NAMES[task])
    effects.insert(2, 'Temperature', temperature)
    effects.to_csv(output/'effects.csv', index=False)
    effects[effects.Treatment.eq('Creative') & effects.Control.isin(CONTROLS)].to_csv(
        output/'creative_vs_controls.csv', index=False)
    effect_plot(effects, output/'paired_effects.png', f'{task.upper()} {candidate}, temperature {temperature}')
    print(f'Complete: {output}', flush=True)


def main():
    """Dispatch extraction, one intervention job, or final result aggregation."""
    args = parse_args()
    phases = {'extract': extract_vectors, 'intervene': run_intervention, 'summarize': summarize_runs}
    phases[args.phase](args)


if __name__ == '__main__':
    main()
