"""Discover raw psychometric/story responses and score them using only AGC-Judge.

Loads the judge once. Keeps source runs/models/tasks separate, with a run index
and descriptive per-condition summaries. No embeddings or other judges are used.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd

from evaluate_agc import (AGC_TASKS, ADAPTER_REVISION, BASE_REVISION, evaluate,
                          judge_prompt, load_judge, prepare_responses, WRITING_DIMENSIONS)
from experiment import PROTOCOL_VERSION

TASK_KEYS = {name: key for key, name in AGC_TASKS.items()}
STORY_TASK = AGC_TASKS['writing']
RAW_COLUMNS = {'Task', 'Model', 'Response', 'Condition', 'Item'}
STORY_COLUMNS = {'Response_ID', 'Cue_ID', 'Item', 'Condition', 'Response'}
SCORED_COLUMNS = {'AGC_Score', 'Parsed_Ideas', 'Parsed_Words', 'DSI', 'DSI_Full',
                  'Pairwise_Semantic_Distance_Proxy', 'DAT_GloVe_Score', 'DRAT_Score',
                  'CDAT_Novelty', 'Semantic_Distance_Proxy'}


def discover_sources(inputs, output_dir):
    """Deduplicate overlapping roots; prefer full story JSONL over its CSV copy."""
    paths = set()
    output_dir = output_dir.resolve()
    for root in inputs:
        root = root.resolve()
        if not root.exists():
            raise ValueError(f'Input does not exist: {root}')
        paths.update(root.rglob('*.csv') if root.is_dir() else [root])
        if root.is_dir():
            paths.update(root.rglob('responses.jsonl'))
    paths = {p.resolve() for p in paths if not p.resolve().is_relative_to(output_dir)}
    result = []
    for path in sorted(paths):
        if path.suffix == '.jsonl':
            with path.open(encoding='utf-8') as handle:
                first = next((json.loads(line) for line in handle if line.strip()), {})
            if first.get('Task') == STORY_TASK and not SCORED_COLUMNS.intersection(first):
                result.append((path, ['writing']))
            continue
        if path.suffix != '.csv':
            continue
        with path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            header = set(reader.fieldnames or [])
            if SCORED_COLUMNS.intersection(header):
                continue
            if STORY_COLUMNS.issubset(header) and not RAW_COLUMNS.issubset(header):
                sibling = path.with_suffix('.jsonl')
                if sibling in paths:
                    continue
                if not (path.parent / 'manifest.json').is_file():
                    raise ValueError(f'Story CSV needs its manifest or full JSONL: {path}')
                result.append((path, ['writing']))
            elif RAW_COLUMNS.issubset(header):
                tasks = {row['Task'] for row in reader}
                keys = sorted(TASK_KEYS[name] for name in tasks if name in TASK_KEYS)
                if keys:
                    if keys == ['writing'] and path.with_suffix('.jsonl') in paths:
                        continue
                    result.append((path, keys))
    return result


def prepare_stories(path, include_legacy=False):
    """Load full stories or reconstruct metadata from a readable CSV's manifest."""
    manifest_path = path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path, keep_default_na=False)
    frame = frame.drop(columns=[c for c in frame if c.endswith('Token_IDs')])
    frame = frame[frame.get('Task', pd.Series(STORY_TASK, index=frame.index)).eq(STORY_TASK)].copy()
    if frame.empty:
        raise ValueError(f'No stories in {path}')
    if 'Item' not in frame and 'Cue_Words' in frame:
        frame['Item'] = frame.Cue_Words.map(lambda words: ', '.join(words))
    defaults = {'Task': STORY_TASK, 'Model': manifest.get('model'),
                'Run_ID': manifest.get('run_id'), 'Split': manifest.get('split', ''),
                'Protocol': manifest.get('protocol', 'legacy'),
                'Base_Seed': manifest.get('seed', '')}
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    for column in ['Temperature', 'Top_P', 'Top_K', 'Max_Tokens']:
        if column not in frame and column.lower() in manifest:
            frame[column] = manifest[column.lower()]
    required = ['Response_ID', 'Run_ID', 'Model', 'Item', 'Condition', 'Response']
    if not set(required).issubset(frame) or frame[required].isna().any().any():
        raise ValueError(f'Missing story identity/text metadata: {path}')
    if any(frame[column].astype(str).str.strip().eq('').any()
           for column in ['Response_ID', 'Run_ID', 'Model', 'Item', 'Condition']):
        raise ValueError(f'Empty story identity fields: {path}')
    if not include_legacy and not frame.Protocol.eq(PROTOCOL_VERSION).all():
        raise ValueError(f'Legacy story protocol in {path}; use --include-legacy')
    if frame.Response_ID.duplicated().any():
        raise ValueError(f'Duplicate story response IDs: {path}')
    # Source errors and empty responses stay visible even if an old Status says ok.
    if 'Status' not in frame:
        frame['Status'] = 'ok'
    frame.loc[frame.Response.astype(str).str.strip().eq(''), 'Status'] = 'empty'
    error = frame.Response.astype(str).str.lstrip().str.startswith('Error:')
    if 'Error' in frame:
        error |= frame.Error.astype(str).str.strip().ne('')
    frame.loc[error, 'Status'] = 'error'
    if 'Block_ID' not in frame:
        if {'Cue_ID', 'Repeat'}.issubset(frame):
            frame['Block_ID'] = frame.Cue_ID.astype(str) + '|r' + frame.Repeat.astype(str)
        elif 'Generation_Seed' in frame:
            frame['Block_ID'] = frame.Item.astype(str) + '|seed=' + frame.Generation_Seed.astype(str)
        else:
            raise ValueError(f'Missing story pairing fields: {path}')
    frame['Source_File'] = str(path.resolve())
    frame['AGC_Prompt'] = [judge_prompt('writing', str(r.Item), str(r.Response))
                           for r in frame.itertuples()]
    return frame


def select_paired_responses(frame, repeats_per_item):
    """Select the lowest complete generation-seed blocks without inspecting answers."""
    required = {'Item', 'Generation_Seed', 'Condition', 'Response_ID'}
    if not required.issubset(frame):
        raise ValueError('Paired sampling needs item, generation seed, condition and response ID')
    conditions = {'Standard', 'Creative', 'Effective', 'Conventional', 'Boring'}
    if set(frame.Condition) != conditions:
        raise ValueError('Paired sampling requires all five psychometric conditions')
    if frame.duplicated(['Item', 'Generation_Seed', 'Condition']).any():
        raise ValueError('Duplicate item/generation-seed/condition rows')
    selected = []
    for item, group in frame.groupby('Item', sort=True, dropna=False):
        seeds = pd.to_numeric(group.Generation_Seed, errors='raise')
        if seeds.isna().any() or not seeds.mod(1).eq(0).all():
            raise ValueError(f'Invalid generation seeds for item {item}')
        complete = []
        for seed in sorted(seeds.unique()):
            block = group.loc[seeds.eq(seed)]
            if len(block) == 5 and set(block.Condition) == conditions:
                complete.append(block)
        if len(complete) < repeats_per_item:
            raise ValueError(f'Not enough complete condition blocks for item {item}')
        for block in complete[:repeats_per_item]:
            selected.extend(block.index)
    result = frame.loc[sorted(selected)].copy()
    result['AGC_Paired_Repeats_Per_Item'] = repeats_per_item
    result['AGC_Sampling'] = 'lowest complete generation-seed blocks per item; all five conditions'
    return result


def make_plan(args):
    """Validate every source before allocating a GPU or writing any output."""
    sources = discover_sources(args.inputs, args.output_dir)
    anchors = [p.resolve() if p.is_dir() else p.resolve().parent for p in args.inputs]
    common = Path(os.path.commonpath(anchors))
    plans, seen = [], set()
    for path, tasks in sources:
        for task in tasks:
            frame = (prepare_stories(path, args.include_legacy) if task == 'writing' else
                     prepare_responses([path], task, args.include_legacy))
            for number, ((run_id, model), group) in enumerate(
                    frame.groupby(['Run_ID', 'Model'], sort=True, dropna=False)):
                identities = set(zip(group.Run_ID.astype(str), group.Response_ID.astype(str)))
                if identities.intersection(seen):
                    raise ValueError(f'Duplicate responses across selected sources: {path}')
                seen.update(identities)
                paired_repeats = getattr(args, 'paired_repeats_per_item', None)
                if paired_repeats is not None:
                    if task == 'writing':
                        raise ValueError('Paired item sampling is for psychometric tasks only')
                    group = select_paired_responses(group, paired_repeats)
                if args.limit_per_run:
                    group = group.head(args.limit_per_run)
                relative = path.relative_to(common)
                output = (args.output_dir / relative.parent / relative.stem /
                          f'{task}_{number:03d}' / f'{task}_agc_responses.csv')
                plans.append(dict(task=task, source=path, run_id=run_id, model=model,
                                  frame=group.reset_index(drop=True), output=output))
    if not plans:
        raise ValueError('No supported raw psychometric or story responses found')
    return plans


def run_sweep(args):
    if args.output_dir.exists():
        raise ValueError(f'Output directory already exists: {args.output_dir}; choose a new one')
    plans = make_plan(args)
    for plan in plans:
        frame = plan['frame']
        print(f"{plan['task']:10s} {len(frame):6d} responses | {plan['model']} | "
              f"{plan['source']} -> {plan['output']}", flush=True)
    print(f'{len(plans)} task/run/model groups; {sum(len(p["frame"]) for p in plans)} responses', flush=True)
    if args.dry_run:
        return plans
    tokenizer, model = load_judge(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    index = pd.DataFrame([dict(Task=AGC_TASKS[p['task']], Run_ID=p['run_id'],
                              Model=p['model'], Source_File=str(p['source']),
                              Responses=len(p['frame']), Output_File=str(p['output'])) for p in plans])
    index.to_csv(args.output_dir / 'run_index.csv', index=False)
    summaries = []
    for plan in plans:
        task_args = argparse.Namespace(**{**vars(args), 'task': plan['task']})
        evaluate(plan['frame'], plan['output'], task_args, tokenizer, model)
        scored = pd.read_csv(plan['output'], keep_default_na=False)
        keys = ['Run_ID', 'Model', 'Task'] + [c for c in ('Split', 'Condition') if c in scored]
        metrics = (['AGC_' + dimension.title() for dimension in WRITING_DIMENSIONS]
                   if plan['task'] == 'writing' else ['AGC_Score'])
        for metric in metrics:
            scored[metric] = pd.to_numeric(scored[metric], errors='coerce')
            for values, group in scored.groupby(keys, dropna=False):
                record = dict(zip(keys, values))
                record.update(Source_File=str(plan['source']), Metric=metric,
                              AGC_Scale_Min=group.AGC_Scale_Min.iloc[0],
                              AGC_Scale_Max=group.AGC_Scale_Max.iloc[0],
                              AGC_Rubric_Kind=group.AGC_Rubric_Kind.iloc[0],
                              AGC_Rubric_Version=group.AGC_Rubric_Version.iloc[0],
                              Responses=len(group), Scored=int(group[metric].notna().sum()),
                              Missing=int(group[metric].isna().sum()),
                              AGC_Mean=group[metric].mean(), AGC_SD=group[metric].std())
                summaries.append(record)
        pd.DataFrame(summaries).to_csv(args.output_dir / 'condition_summary.csv', index=False)
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', default=[Path('outputs')])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--include-legacy', action='store_true')
    parser.add_argument('--limit-per-run', type=int, help='First N responses per run/task/model; smoke tests only')
    parser.add_argument('--paired-repeats-per-item', type=int,
                        help='Complete five-condition blocks per item, selected by lowest generation seed')
    parser.add_argument('--adapter', default='/scratch-shared/jsmidi/models/AGC-Judge')
    parser.add_argument('--adapter-revision', default=ADAPTER_REVISION)
    parser.add_argument('--base-model', default='/scratch-shared/jsmidi/models/Qwen3-30B-A3B-Instruct-2507')
    parser.add_argument('--base-revision', default=BASE_REVISION)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    args = parser.parse_args()
    if (args.limit_per_run is not None and args.limit_per_run < 1) or args.max_input_tokens < 1:
        parser.error('limit-per-run and max-input-tokens must be positive')
    if args.paired_repeats_per_item is not None:
        if args.paired_repeats_per_item < 1 or args.limit_per_run is not None:
            parser.error('paired-repeats-per-item must be positive and cannot be combined with limit-per-run')
    try:
        run_sweep(args)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
