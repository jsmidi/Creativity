"""Extract prompt-state vectors once; test frozen vectors at sampled temperatures."""
import argparse
import csv
import json
from pathlib import Path
import pandas as pd
from run_greedy_task import NAMES, DISCOVERY, AUDIT, METRICS, run, score
from task_config import get_task_config
from analyze_effects import intervention_effects
from plot_tasks import effect_plot

CANDIDATES = ['residual', 'concept_mean', 'concept_contrast', 'concept_head']
TEMPERATURES = [0, 0.7, 1]
CONTROLS = ['Standard', 'Effective', 'Conventional', 'Boring']


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
    run('run_residual.py', 'extract', '--model', args.model, '--artifact', root/'residual.pt',
        '--tasks', NAMES[task], '--items', *DISCOVERY[task], '--paraphrases', 1, 2,
        '--baseline', 'Standard', '--layer', 16)
    run('run_attention_heads.py', '--model', args.model, '--task', NAMES[concept_task],
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
    run('run_residual.py', 'generate', '--model', args.model, '--artifact', artifact,
        '--task', NAMES[task], '--items', *items, '--paraphrases', 0, '--split', 'validation',
        '--temperature', temperature, '--omit-temperature-control', '--repeats', repeats,
        '--alphas', 1, '--random-vectors', 2, '--scope', 'prefill', '--max-tokens', 800,
        '--output-dir', raw)
    score(task, raw, output)
    frame = pd.read_csv(output/f'{task}_responses.csv', keep_default_na=False)
    effects = intervention_effects(frame, METRICS[task])
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
