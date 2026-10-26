"""Audit completed psychometric AGC exports and summarize paired seed differences.

Checks identities, completeness, ranges and AUT per-use aggregation. These are
technical and experimental checks, not validation against human creativity ratings.
"""
import argparse
import json
import math
from pathlib import Path
import re

import pandas as pd
from evaluate_agc import AGC_TASKS, idea_output_path, rubric, score_scale
from scoring_common import parse_numbered_list


def audit(root, allow_subset=False, require_complete_scores=False):
    index = pd.read_csv(root / 'run_index.csv', keep_default_na=False)
    coverage, means = [], []
    seen = set()
    for entry in index.itertuples():
        path = Path(entry.Output_File)
        frame = pd.read_csv(path, keep_default_na=False)
        task = next(key for key, name in AGC_TASKS.items() if name == entry.Task)
        if task == 'writing':
            raise ValueError('This validation report is for psychometric cohorts')
        if len(frame) != int(entry.Responses) or frame.Response_ID.duplicated().any():
            raise ValueError(f'Incomplete/duplicate response export: {path}')
        identities = set(frame.Response_ID)
        if seen.intersection(identities):
            raise ValueError(f'Duplicate response identities across runs: {path}')
        seen.update(identities)
        source = pd.read_csv(entry.Source_File, keep_default_na=False)
        source = source[(source.Run_ID == entry.Run_ID) & (source.Model == entry.Model)
                        & (source.Task == entry.Task)]
        if not identities.issubset(set(source.Response_ID)):
            raise ValueError(f'Unknown response IDs: {path}')
        if not allow_subset and identities != set(source.Response_ID):
            raise ValueError(f'Missing raw responses: {path}')
        if 'AGC_Paired_Repeats_Per_Item' in frame:
            expected_conditions = {'Standard', 'Creative', 'Effective', 'Conventional', 'Boring'}
            for _, block in frame.groupby(['Item', 'Generation_Seed'], dropna=False):
                if len(block) != 5 or set(block.Condition) != expected_conditions:
                    raise ValueError(f'Incomplete sampled condition pair block: {path}')
            for _, item_rows in frame.groupby('Item', dropna=False):
                repeats = item_rows.AGC_Paired_Repeats_Per_Item.unique()
                if len(repeats) != 1 or len(item_rows) != 5*int(repeats[0]):
                    raise ValueError(f'Incorrect paired repetitions per item: {path}')
        original = source.set_index('Response_ID').loc[frame.Response_ID]
        for column in ['Response', 'Item', 'Condition', 'Generation_Seed']:
            if original[column].astype(str).tolist() != frame[column].astype(str).tolist():
                raise ValueError(f'Changed {column}: {path}')
        low, high = score_scale(task)
        scores = pd.to_numeric(frame.AGC_Score, errors='coerce')
        if not scores.dropna().between(low, high).all():
            raise ValueError(f'Out-of-range scores: {path}')
        if not frame.AGC_Rubric.eq(rubric(task)).all():
            raise ValueError(f'Unexpected rubric: {path}')
        if not frame.AGC_Scale_Min.eq(low).all() or not frame.AGC_Scale_Max.eq(high).all():
            raise ValueError(f'Incorrect scale metadata: {path}')
        if frame.AGC_Status.eq('ok').tolist() != scores.notna().tolist():
            raise ValueError(f'Score/status mismatch: {path}')
        malformed, idea_count, idea_missing = 0, 0, 0
        if task == 'aut':
            ideas = pd.read_csv(idea_output_path(path), keep_default_na=False)
            if ideas.duplicated(['Response_ID', 'Idea_Index']).any():
                raise ValueError(f'Duplicate idea identities: {path}')
            idea_scores = pd.to_numeric(ideas.AGC_Score, errors='coerce')
            if not idea_scores.dropna().between(low, high).all():
                raise ValueError(f'Out-of-range idea scores: {path}')
            if not set(ideas.Response_ID).issubset(identities):
                raise ValueError(f'Unknown idea response identity: {path}')
            for row in frame.itertuples():
                parsed = parse_numbered_list(row.Response) if row.Status == 'ok' else []
                group = ideas[ideas.Response_ID == row.Response_ID].sort_values('Idea_Index')
                if group.AGC_Idea.tolist() != parsed:
                    raise ValueError(f'Missing/reordered parsed ideas: {row.Response_ID}')
                numbers = pd.to_numeric(group.AGC_Score, errors='coerce')
                expected = numbers.mean() if len(numbers) and numbers.notna().all() else float('nan')
                actual = pd.to_numeric(pd.Series([row.AGC_Score]), errors='coerce').iloc[0]
                if not ((pd.isna(expected) and pd.isna(actual)) or math.isclose(expected, actual, abs_tol=1e-9)):
                    raise ValueError(f'Incorrect idea aggregation: {row.Response_ID}')
                prompts = json.loads(row.AGC_Prompt)
                if prompts != group.AGC_Prompt.tolist():
                    raise ValueError(f'Incorrect prompt audit: {row.Response_ID}')
            malformed = int((~frame.AGC_Format_Valid.astype(str).str.lower().eq('true')).sum())
            idea_count, idea_missing = len(ideas), int(idea_scores.isna().sum())
        if require_complete_scores and (scores.isna() & frame.Status.eq('ok')).any():
            raise ValueError(f'Missing judge scores in pilot: {path}')
        seed_match = re.search(r'(?:^|/)seed_(\d+)(?:/|$)', entry.Source_File)
        seed = int(seed_match.group(1)) if seed_match else None
        model = Path(entry.Model).name
        coverage.append(dict(Model=model, Task=task, Seed=seed, Responses=len(frame),
            Scored=int(scores.notna().sum()), Missing=int(scores.isna().sum()),
            Score_Min=low, Score_Max=high, Minimum_Observed=scores.min(), Maximum_Observed=scores.max(),
            Distinct_Scores=scores.nunique(), Non_Ten_Idea_Lists=malformed,
            Ideas=idea_count, Missing_Idea_Scores=idea_missing, Source_File=entry.Source_File))
        frame = frame.assign(Score=scores)
        for condition, group in frame.groupby('Condition'):
            means.append(dict(Model=model, Task=task, Seed=seed, Condition=condition,
                Mean=group.groupby('Item').Score.mean().mean(), Responses=len(group),
                Scored=int(group.Score.notna().sum()), Items=group.Item.nunique()))
    return pd.DataFrame(coverage), pd.DataFrame(means)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--allow-subset', action='store_true', help='For pilot outputs')
    parser.add_argument('--require-complete-scores', action='store_true')
    args = parser.parse_args()
    coverage, means = audit(args.input_dir, args.allow_subset, args.require_complete_scores)
    output = args.input_dir / 'validation'
    output.mkdir(exist_ok=False)
    coverage.to_csv(output / 'coverage.csv', index=False)
    means.to_csv(output / 'seed_condition_means.csv', index=False)
    summary = means.groupby(['Model', 'Task', 'Condition']).agg(
        Mean=('Mean', 'mean'), Seed_SD=('Mean', 'std'), Seeds=('Seed', 'nunique'),
        Responses=('Responses', 'sum'), Scored=('Scored', 'sum')).reset_index()
    summary.to_csv(output / 'condition_means.csv', index=False)
    differences = []
    for (model, task), group in means.groupby(['Model', 'Task']):
        table = group.pivot(index='Seed', columns='Condition', values='Mean')
        if 'Creative' in table:
            for control in ['Standard', 'Effective', 'Conventional', 'Boring']:
                if control in table:
                    values = (table.Creative-table[control]).dropna()
                    differences.append(dict(Model=model, Task=task, Treatment='Creative',
                        Control=control, Paired_Seeds=len(values), Mean_Difference=values.mean(),
                        Seed_SD=values.std()))
    pd.DataFrame(differences).to_csv(output / 'paired_seed_differences.csv', index=False)
    totals = coverage.groupby(['Model','Task'])[['Responses','Scored','Missing','Non_Ten_Idea_Lists']].sum()
    lines = ['# AGC-Judge technical validation', '',
        'Passed identity, source coverage, scale, rubric, and AUT aggregation checks.', '',
        'Scope: selected subset of source responses (see AGC_Sampling in score files).' if args.allow_subset else 'Scope: all responses in the selected source runs.', '',
        f'Responses: {coverage.Responses.sum()}; scored: {coverage.Scored.sum()}; missing: {coverage.Missing.sum()}.', '',
        'AUT uses the published 10–50 per-use rubric and an equal-use response mean. '
        'Non-ten-use lists are flagged and retained. Other psychometric tasks use exploratory local 1–5 rubrics.', '',
        'Condition means average items within each seed, then seeds equally. Differences are paired seed means; '
        'no independent-response significance tests are claimed.', '',
        'This checks scoring execution and experimental differences. No human-rating agreement or new JRT calibration was evaluated.', '',
        '```text', totals.to_string(), '```', '']
    (output / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
