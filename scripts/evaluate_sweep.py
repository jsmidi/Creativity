"""Evaluate each temperature/task run separately, retaining all four prompt contrasts."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

TASKS = {
    'Divergent Association Task': ('dat', 'DAT_GloVe_Score'),
    'Alternative Uses Task': ('aut', 'Semantic_Distance_Proxy'),
    'Conditional Divergent Association Task': ('cdat', 'CDAT_Novelty'),
    'Divergent Remote Association Test': ('drat', 'DRAT_Score'),
}


def main():
    """Build or execute scoring commands; --dry-run requires no embedding model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    planned = 0
    for path in sorted(args.inputs.rglob('results_*.csv')):
        with path.open() as handle:
            rows = csv.DictReader(handle)
            first = next(rows, None)
            if not first or first.get('Task') not in TASKS:
                continue
        task, metric = TASKS[first['Task']]
        output = args.output_dir/path.stem
        if output.exists():
            parser.error(f'Refusing to overwrite {output}; choose a new output directory')
        if task in ('dat', 'aut'):
            command = [sys.executable, f'scripts/evaluate_{task}.py', '--inputs', str(path), '--output-dir', str(output)]
            if task == 'dat':
                command += ['--scorer', 'official', '--official-code', 'databases/dat.py',
                            '--dictionary', 'databases/words.txt', '--glove', 'databases/glove.840B.300d.txt']
        else:
            command = [sys.executable, 'scripts/evaluate_association.py', '--task', task,
                       '--inputs', str(path), '--output-dir', str(output), '--device', 'cuda']
        effects = [sys.executable, 'scripts/analyze_effects.py', str(output/f'{task}_responses.csv'),
                   '--metric', metric, '--treatment', 'Creative', '--controls',
                   'Standard', 'Effective', 'Conventional', 'Boring', '--output', str(output/'creative_vs_controls.csv')]
        for call in (command, effects):
            print(call, flush=True)
            if not args.dry_run:
                subprocess.run(call, check=True)
        planned += 1
    if not planned:
        parser.error('No supported raw runs found')
    print(f'{planned} separate runs; CDAT novelty contrasts require checking cdat_gates.csv before interpretation.')


if __name__ == '__main__':
    main()
