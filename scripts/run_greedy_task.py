"""One task: 60 greedy behavioral repeats, residual and concept interventions, scoring."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
import numpy as np
from task_config import get_task_config
from analyze_effects import paired_effect

SCRIPTS = Path(__file__).resolve().parent
NAMES = dict(aut='Alternative Uses Task', dat='Divergent Association Task',
             cdat='Conditional Divergent Association Task', drat='Divergent Remote Association Test')
DISCOVERY = dict(aut=['brick', 'rope', 'bottle', 'spoon'], dat=[''],
                 cdat=['ocean', 'memory', 'travel', 'weather'],
                 drat=['neuron | particle | graph | ritual', 'lung | turbulence | turbine | matrix',
                       'cell | crystal | factory | treaty', 'ecosystem | neighborhood | graph | pressure'])
AUDIT = dict(aut=['umbrella', 'shoe'], cdat=['garden', 'language'],
             drat=['entropy | information | circuit | virus', 'rumor | broadcast | transmission | parasite'])
METRICS = dict(aut='Semantic_Distance_Proxy', dat='DAT_GloVe_Score', cdat='CDAT_Novelty', drat='DRAT_Score')


def run(script, *arguments):
    """Run a frozen sibling script, propagating failures to Slurm."""
    command = [sys.executable, str(SCRIPTS/script), *map(str, arguments)]
    print(command, flush=True)
    subprocess.run(command, check=True)


def score(task, inputs, output):
    """Score one run, keeping output directories and calibration separate."""
    if task in ('cdat', 'drat'):
        run('evaluate_association.py', '--task', task, '--inputs', inputs, '--output-dir', output, '--device', 'cuda')
    else:
        extra = ['--scorer', 'official', '--official-code', 'databases/dat.py', '--dictionary',
                 'databases/words.txt', '--glove', 'databases/glove.840B.300d.txt'] if task == 'dat' else ['--norm-db', 'aut_quality_scored_all.csv']
        run(f'evaluate_{task}.py', '--inputs', inputs, '--output-dir', output, *extra)


def summarize(task, raw_root, scored_root, output):
    """Export response uniqueness, paired effects and validity; do not infer independent greedy samples."""
    metric = METRICS[task]
    effects, audit = [], []
    for path in sorted(scored_root.rglob(f'{task}_responses.csv')):
        frame = pd.read_csv(path, keep_default_na=False)
        candidate = str(path.parent.relative_to(scored_root))
        pairs = [('Creative', control) for control in ['Standard', 'Effective', 'Conventional', 'Boring']]
        if 'Standard_add_1' in set(frame.Condition):
            pairs += [('Standard_add_1', 'Standard'), ('Creative_subtract_1', 'Creative'),
                      ('Creative_suppress', 'Creative'), ('Standard_suppress', 'Standard'),
                      ('Standard_patch_creative', 'Standard'), ('Creative_patch_standard', 'Creative')]
            for index in range(2):
                pairs += [(f'Standard_random{index}_1', 'Standard'),
                          (f'Creative_random_suppress{index}', 'Creative'),
                          ('Standard_add_1', f'Standard_random{index}_1'),
                          ('Creative_suppress', f'Creative_random_suppress{index}')]
        for treatment, control in pairs:
            base = dict(Candidate=candidate, Treatment=treatment, Control=control)
            try:
                effects.append(dict(base, **{k:v for k,v in paired_effect(frame, metric, treatment, control).items()
                                              if k not in ('Treatment', 'Control')}, Analysis_Status='ok'))
            except ValueError as exc:
                if str(exc) != 'No complete scored pairs for requested contrast.':
                    raise
                effects.append(dict(base, Metric=metric, Mean_Difference=np.nan, Analysis_Status=str(exc)))
        for (item, condition), group in frame.groupby(['Item', 'Condition'], dropna=False):
            valid = group[group.Status == 'ok']
            audit.append(dict(Candidate=candidate, Item=item, Condition=condition, Requests=len(group),
                              Successful=len(valid), Unique_Responses=valid.Response.nunique(),
                              Scored=pd.to_numeric(group[metric], errors='coerce').notna().sum(),
                              Truncated=(group.Finish_Reason == 'length').sum() if 'Finish_Reason' in group else 0))
    pd.DataFrame(effects).to_csv(output/'effects.csv', index=False)
    pd.DataFrame(audit).to_csv(output/'determinism_audit.csv', index=False)
    (output/'README.md').write_text(
        f'# {NAMES[task]} greedy analysis\n\n'
        'effects.csv reports descriptive paired differences, with no sampling confidence intervals.\n'
        'determinism_audit.csv reports unique successful answers for each item/condition.\n'
        'Compare learned additions/suppression with their random controls, and inspect validity and truncation.\n'
        'Concept RSA concerns instruction-condition geometry, not measured creativity.\n'
        + ('DAT concept artifacts are AUT-to-DAT transfer; the residual vector is extracted on DAT with held-out wording.\n' if task == 'dat' else '')
        + ('CDAT novelty effects require inspection of each candidate cdat_gates.csv; ungated novelty alone is not the CDAT score.\n' if task == 'cdat' else '')
        + 'These exploratory tests do not establish a creativity-specific mechanism.\n')


def main():
    """Execute all stages in one GPU allocation; --dry-run prints design without loading models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=NAMES, required=True)
    parser.add_argument('--model', default='models/Llama-3.1-8B-Instruct')
    parser.add_argument('--repeats', type=int, default=60)
    parser.add_argument('--mi-repeats', type=int, default=1)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1 or args.mi_repeats < 1:
        parser.error('Repeat counts must be positive')
    task = NAMES[args.task]
    concept_task = 'aut' if args.task == 'dat' else args.task
    items = get_task_config(task)['items']
    plan = dict(task=task, model=args.model, temperature=0, top_p=1, top_k=0,
                behavioral_repeats=args.repeats, intervention_repeats=args.mi_repeats,
                behavioral_responses=len(items)*5*args.repeats,
                intervention_responses=len(items)*15*4*args.mi_repeats,
                evaluation_items=items, residual_discovery=DISCOVERY[args.task],
                concept_source_task=NAMES[concept_task], concept_discovery=DISCOVERY[concept_task],
                concept_audit=AUDIT[concept_task], extraction_wordings=[1,2], evaluation_wording=0,
                layer=16, alpha=1, scope='prefill', random_vectors=2,
                interpretation='exploratory greedy interventions; repeats are determinism checks')
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    run('generate.py', '--task', task, '--model', args.model, '--provider', 'local',
        '--repeats', args.repeats, '--temperature', 0, '--top-p', 1, '--top-k', 0,
        '--paraphrases', 0, '--randomize', '--output-root', args.run_dir/'behavioral')
    score(args.task, args.run_dir/'behavioral', args.analysis_dir/'behavioral')
    residual = args.run_dir/'residual.pt'
    run('activation_steer.py', 'extract', '--model', args.model, '--artifact', residual,
        '--tasks', task, '--items', *DISCOVERY[args.task], '--paraphrases', 1, 2, '--baseline', 'Standard', '--layer', 16)
    discovery = args.run_dir/'concept_discovery'
    run('concept_vectors.py', '--model', args.model, '--task', NAMES[concept_task], '--output-dir', discovery,
        '--items', *DISCOVERY[concept_task], '--audit-items', *AUDIT[concept_task],
        '--paraphrases', 1, 2, '--audit-paraphrase', 0, '--top-k', 5, '--layer', 16, '--permutations', 199)
    with (discovery/'head_ranking.csv').open() as handle:
        top = next(csv.DictReader(handle))
    head = f"head_l{top['Layer']}_h{top['Head']}"
    for candidate, artifact in [('residual', residual), ('concept_mean', discovery/'creative_mean.pt'),
                                ('concept_contrast', discovery/'contrast.pt'), ('concept_head', discovery/f'{head}.pt')]:
        output = args.run_dir/candidate
        run('activation_steer.py', 'generate', '--model', args.model, '--artifact', artifact,
            '--task', task, '--items', *items, '--paraphrases', 0, '--split', 'validation',
            '--temperature', 0, '--omit-temperature-control', '--repeats', args.mi_repeats,
            '--alphas', 1, '--random-vectors', 2, '--scope', 'prefill', '--max-tokens', 800,
            '--output-dir', output)
        score(args.task, output, args.analysis_dir/candidate)
    summarize(args.task, args.run_dir, args.analysis_dir, args.analysis_dir)
    print(f'All stages complete: {args.analysis_dir}', flush=True)


if __name__ == '__main__':
    main()
