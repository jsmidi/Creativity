"""Supplementary AGC-Judge ratings using exploratory, task-specific rubrics."""
import argparse
import csv
import json
from contextlib import ExitStack
import re
from pathlib import Path

from scoring_common import load_responses, parse_numbered_list
from task_config import TASK_NAMES, build_prompt

ADAPTER = 'agcbench-2026/AGC-Judge'
ADAPTER_REVISION = 'e97b7895d8c16e8438ab869f752a37f19908da4c'
BASE = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
BASE_REVISION = '0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'
RUBRIC_VERSION = 'local-list-creativity-v1'
PUBLISHED = json.loads(Path(__file__).with_name('agc_published_rubrics.json').read_text())
WRITING_DIMENSIONS = ('creativity', 'originality', 'effectiveness', 'surprise')
AGC_TASKS = {**TASK_NAMES, 'hypothesis': 'Scientific Hypothesis Generation',
             'metaphor': 'Metaphor Generation', 'writing': 'Creative Short Story'}
SHORT_CRITERIA = {
    'hypothesis': 'Assess originality and explanatory insight, scientific plausibility, testability, and relevance to the observation. Unsupported fanciful claims are not creative merely because they are surprising.',
    'metaphor': 'Assess original and surprising conceptual connections that clearly communicate the intended concept. Incoherent or irrelevant comparisons do not count as creative.',
}
CRITERIA = {
    'dat': 'Assess semantic diversity across single English nouns and how far apart their meanings are. Repetition, phrases and invalid words do not add diversity. This is a subjective impression, not a calculation of embedding distance.',
    'cdat': 'Assess unusual, original associations and diversity across single English nouns, while requiring meaningful relevance to the cue. Unrelated nouns do not count as creative associations merely because they are distant from each other.',
    'drat': 'Assess original, remote connections and diversity across single English nouns. Each noun must meaningfully connect, literally or metaphorically, to EVERY anchor. Connections to only some anchors do not satisfy the task.',
}


def rubric_version(task):
    if task == 'aut':
        return 'agc-human-aut-per-use-10-50-v1'
    if task == 'writing':
        return 'agc-cue-word-story-four-dimensions-v1'
    return 'local-short-creativity-v1' if task in SHORT_CRITERIA else RUBRIC_VERSION


def published_spec(task, dimension='creativity'):
    key = 'aut' if task == 'aut' else 'writing_' + dimension
    return PUBLISHED['rubrics'][key] if task in ('aut', 'writing') else None


def score_scale(task):
    return (10, 50) if task == 'aut' else (1, 5)


def rubric(task, dimension='creativity'):
    spec = published_spec(task, dimension)
    if spec:
        return spec['text']
    if task in SHORT_CRITERIA:
        return ('Rate the creativity of the entire response. ' + SHORT_CRITERIA[task]
                + ' Use the same standard for every response. Scale: 1-5 integers, where '
                '1 = very low creativity or largely invalid; 2 = low creativity, mostly obvious; '
                '3 = moderate creativity, some original and appropriate ideas; '
                '4 = high creativity, original and appropriate; '
                '5 = exceptional creativity, consistently original and appropriate.')
    return ('Rate the creativity of the entire response as one list. ' + CRITERIA[task]
            + ' Judge ideas rather than writing polish or verbosity. Consider the whole list, '
            'not only its best entry. Duplicates and missing or invalid entries reduce overall '
            'quality; do not invent omitted entries. Use the same standard for every response. '
            'Scale: 1-5 integers, where 1 = very low creativity or largely invalid; '
            '2 = low creativity, mostly obvious; 3 = moderate creativity with some original '
            'and appropriate ideas; 4 = high creativity with many original and appropriate '
            'ideas; 5 = exceptional creativity, consistently original, diverse and appropriate.')


def judge_prompt(task, item, response, dimension='creativity'):
    # The wrapper and neutral task instructions are local; published rubric text is verbatim.
    if task == 'writing':
        instruction = ('Write a short story using a maximum of five sentences. The story must '
                       f'include the following three words: {item}. Incorporate the words naturally.')
    elif task == 'aut':
        instruction = f'Give an alternative use for the object {item} other than its primary use.'
    else:
        instruction = build_prompt(AGC_TASKS[task], item, 'standard')
    return (f'Benchmark rubric:\n{rubric(task, dimension)}\n\n'
            f'Prompt:\n{instruction}\n\nResponse:\n{response}\n\n'
            'Output a single integer score on the scale specified in the rubric. '
            'No explanation, no formatting, just the number.\n\nScore:')


def response_prompts(task, item, response):
    if task == 'aut':
        return json.dumps([judge_prompt(task, item, idea)
                           for idea in parse_numbered_list(response)], ensure_ascii=False)
    return judge_prompt(task, item, response)


def parse_score(text, score_min=1, score_max=5):
    """Require a single integer within the particular rubric's scale."""
    value = text.strip()
    if not re.fullmatch(r'(?:0|[1-9][0-9]*)', value):
        return None
    number = int(value)
    return number if score_min <= number <= score_max else None


def prepare_responses(inputs, task, include_legacy=False, limit=None):
    files = sorted({file.resolve() for path in inputs
                    for file in (path.rglob('*.csv') if path.is_dir() else [path])})
    raw_files = []
    for path in files:
        with path.open(newline='', encoding='utf-8-sig') as handle:
            header = next(csv.reader(handle), [])
        if 'AGC_Score' not in header:
            raw_files.append(path)
    frame = load_responses(raw_files, AGC_TASKS[task], include_legacy)
    if limit is not None:
        frame = frame.head(limit).copy()
    frame['AGC_Prompt'] = [response_prompts(task, str(row.Item), str(row.Response))
                           for row in frame.itertuples()]
    return frame


def load_judge(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError('Run inference inside a GPU allocation; --dry-run needs no GPU')
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter, revision=args.adapter_revision, local_files_only=args.local_files_only)
    # Reserve runtime headroom and forbid silent CPU/disk offload.
    memory = {i: int(torch.cuda.get_device_properties(i).total_memory * .90)
              for i in range(torch.cuda.device_count())}
    memory['cpu'] = 0
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.base_revision, torch_dtype=torch.bfloat16,
        device_map='auto', max_memory=memory, local_files_only=args.local_files_only)
    # Transformers may omit hf_device_map when the model fits on one GPU.
    # Inspect actual parameter placement rather than an optional dispatch attribute.
    if any(parameter.device.type != 'cuda' for parameter in base.parameters()):
        raise RuntimeError('Judge does not fit on allocated GPUs; request more GPU memory')
    model = PeftModel.from_pretrained(
        base, args.adapter, revision=args.adapter_revision, is_trainable=False,
        local_files_only=args.local_files_only)
    model.eval()
    return tokenizer, model


def score_prompt(prompt, tokenizer, model, max_input_tokens, score_min=1, score_max=5):
    import torch
    inputs = tokenizer(prompt, return_tensors='pt', truncation=False)
    length = inputs.input_ids.shape[1]
    if length > max_input_tokens:
        return '', None, 'input_too_long', length
    inputs = inputs.to(model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=4,
                                stop_strings=['\n'], tokenizer=tokenizer,
                                pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0][length:], skip_special_tokens=True)
    score = parse_score(raw, score_min, score_max)
    return raw, score, 'ok' if score is not None else 'invalid_score', length


def idea_output_path(output):
    stem = output.stem.removesuffix('_responses')
    return output.with_name(stem + '_ideas.csv')


def evaluate(frame, output, args, tokenizer, model):
    """Score AUT ideas separately and story dimensions independently; retain all audits."""
    task = args.task
    score_min, score_max = score_scale(task)
    spec = published_spec(task)
    metadata = dict(AGC_Adapter=args.adapter, AGC_Adapter_Revision=args.adapter_revision,
                    AGC_Base_Model=args.base_model, AGC_Base_Revision=args.base_revision,
                    AGC_Rubric_Version=rubric_version(task), AGC_Rubric=rubric(task),
                    AGC_Rubric_Kind='published' if spec else 'local_exploratory',
                    AGC_Rubric_Source=spec['source'] if spec else '',
                    AGC_Scale_Min=score_min, AGC_Scale_Max=score_max,
                    AGC_Rating_Unit='response mean of individual uses' if task == 'aut' else 'whole response',
                    AGC_Protocol='raw completion; neutral instruction; greedy; newline stop; bf16')
    extra = ['AGC_Score', 'AGC_Status', 'AGC_Raw_Output', 'AGC_Input_Tokens', *metadata]
    if task == 'aut':
        extra += ['AGC_Idea_Count', 'AGC_Scored_Ideas', 'AGC_Expected_Ideas',
                  'AGC_Format_Valid', 'AGC_Aggregation']
    if task == 'writing':
        for dimension in WRITING_DIMENSIONS:
            name = 'AGC_' + dimension.title()
            extra += [name, name+'_Status', name+'_Raw_Output', name+'_Input_Tokens',
                      name+'_Prompt', name+'_Rubric', name+'_Rubric_Source']
    targets = [output] + ([idea_output_path(output)] if task == 'aut' else [])
    for path in targets:
        if path.exists():
            raise FileExistsError(f'Output already exists: {path}')
    output.parent.mkdir(parents=True, exist_ok=True)

    def rate(row, response, dimension='creativity'):
        prompt = judge_prompt(task, str(row['Item']), response, dimension)
        if row['Status'] != 'ok':
            return prompt, '', None, 'source_' + row['Status'], ''
        raw, score, status, tokens = score_prompt(
            prompt, tokenizer, model, args.max_input_tokens,
            score_min=score_min, score_max=score_max)
        return prompt, raw, score, status, tokens

    with ExitStack() as stack:
        handle = stack.enter_context(output.open('x', newline='', encoding='utf-8'))
        writer = csv.DictWriter(handle, fieldnames=[*frame.columns, *extra])
        writer.writeheader()
        if task == 'aut':
            idea_handle = stack.enter_context(targets[1].open('x', newline='', encoding='utf-8'))
            idea_writer = csv.DictWriter(idea_handle, fieldnames=[*frame.columns,
                'Idea_Index', 'AGC_Idea', 'AGC_Score', 'AGC_Status', 'AGC_Raw_Output',
                'AGC_Input_Tokens', *metadata])
            idea_writer.writeheader()
        for index, source in enumerate(frame.to_dict('records'), 1):
            row = {**source, **metadata}
            if task == 'aut':
                ideas = parse_numbered_list(source['Response'])
                results = []
                if source['Status'] == 'ok':
                    for idea_index, idea in enumerate(ideas, 1):
                        prompt, raw, score, status, tokens = rate(source, idea)
                        results.append((prompt, raw, score, status, tokens))
                        idea_writer.writerow({**source, **metadata, 'Idea_Index': idea_index,
                            'AGC_Idea': idea, 'AGC_Prompt': prompt, 'AGC_Score': score,
                            'AGC_Status': status, 'AGC_Raw_Output': raw,
                            'AGC_Input_Tokens': tokens, 'AGC_Rating_Unit': 'individual use'})
                        idea_handle.flush()
                scores = [result[2] for result in results if result[2] is not None]
                complete = bool(results) and len(scores) == len(ideas)
                status = ('source_' + source['Status'] if source['Status'] != 'ok' else
                          'no_parsed_ideas' if not ideas else
                          'ok' if complete else 'incomplete_idea_scores')
                row.update(AGC_Score=sum(scores)/len(scores) if complete else None,
                    AGC_Status=status, AGC_Prompt=json.dumps([r[0] for r in results]),
                    AGC_Raw_Output=json.dumps([r[1] for r in results]),
                    AGC_Input_Tokens=sum(r[4] for r in results),
                    AGC_Idea_Count=len(ideas), AGC_Scored_Ideas=len(scores),
                    AGC_Expected_Ideas=10, AGC_Format_Valid=len(ideas) == 10 and source['Status'] == 'ok',
                    AGC_Aggregation='mean of all parsed idea ratings; missing if any rating is missing')
            elif task == 'writing':
                for dimension in WRITING_DIMENSIONS:
                    prompt, raw, score, status, tokens = rate(source, str(source['Response']), dimension)
                    name = 'AGC_' + dimension.title()
                    row.update({name: score, name+'_Status': status, name+'_Raw_Output': raw,
                        name+'_Input_Tokens': tokens, name+'_Prompt': prompt,
                        name+'_Rubric': rubric(task, dimension),
                        name+'_Rubric_Source': published_spec(task, dimension)['source']})
                    if dimension == 'creativity':
                        # Compatibility alias: never average the four dimensions.
                        row.update(AGC_Score=score, AGC_Status=status, AGC_Raw_Output=raw,
                                   AGC_Input_Tokens=tokens, AGC_Prompt=prompt)
            else:
                prompt, raw, score, status, tokens = rate(source, str(source['Response']))
                row.update(AGC_Prompt=prompt, AGC_Score=score, AGC_Status=status,
                           AGC_Raw_Output=raw, AGC_Input_Tokens=tokens)
            writer.writerow(row)
            handle.flush()
            if index % 25 == 0 or index == len(frame):
                print(f'Scored {index}/{len(frame)} responses: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--task', choices=[key for key in AGC_TASKS if key != 'writing'], default='aut')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--include-legacy', action='store_true')
    parser.add_argument('--limit', type=int, help='First N rows, for smoke tests only')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--adapter', default=ADAPTER)
    parser.add_argument('--adapter-revision', default=ADAPTER_REVISION)
    parser.add_argument('--base-model', default=BASE)
    parser.add_argument('--base-revision', default=BASE_REVISION)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.max_input_tokens < 1:
        parser.error('limit and max-input-tokens must be positive')
    if not args.dry_run and args.output_dir is None:
        parser.error('--output-dir is required for scoring')
    frame = prepare_responses(args.inputs, args.task, args.include_legacy, args.limit)
    print(f'{len(frame)} responses; {args.task}; {rubric_version(args.task)}; scale={score_scale(args.task)}')
    print(frame.groupby(['Model', 'Condition']).size().to_string())
    if args.dry_run:
        print('\nJudge input(s) for the first response:\n' + frame.iloc[0]['AGC_Prompt'])
        return
    output = args.output_dir / f'{args.task}_agc_responses.csv'
    if output.exists():
        parser.error(f'Output already exists: {output}; choose a new output directory')
    tokenizer, model = load_judge(args)
    evaluate(frame, output, args, tokenizer, model)


if __name__ == '__main__':
    main()
