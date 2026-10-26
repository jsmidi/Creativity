"""RSA-selected attention-head vectors: a creativity adaptation of Opielka et al.

Independent implementation; upstream method/code references are in
writing/concept_vectors.md. RSA uses pre-o_proj heads; composite vectors use
W_O-projected head contributions, never raw head coordinates summed across layers.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch

from run_residual import provenance
from experiment import PROTOCOL_VERSION, CONDITION_TEXT, canonical_item
from interventions import load_engine
from task_config import get_task_config

UPSTREAM_COMMIT = 'dcbcd0ef16d8a6c1af8b0a2378b7717a9ba66c42'
CONCEPTS = ('Creative', 'Conventional', 'Effective')


def stimuli(items, paraphrases, conditions=CONCEPTS, task="Alternative Uses Task"):
    """Build an item/wording/condition grid in numbered and bulleted formats.

    Standard prompts are collected separately for the suppression center."""
    template = get_task_config(task)["instruction"]
    formats = dict(numbered=template, bullets=template.replace("a numbered list (1-10)", "a bulleted list"))
    if formats["numbered"] == formats["bullets"]:
        raise ValueError("Task has no supported output-format contrast")
    return [dict(Item=item, Paraphrase=p, Format=fmt, Condition=condition,
                 Prompt=(template.format(item=item) + '\n' + CONDITION_TEXT[condition][p]).strip())
            for item in items for p in paraphrases for fmt, template in formats.items()
            for condition in conditions]


@torch.inference_mode()
def capture_attention_heads(engine, inputs):
    """Read all heads before o_proj at the last prompt token.

    Returns CPU float32 [layers, query_heads, head_dim]. Each o_proj input
    contains concatenated head outputs; reshape it to separate the heads.
    """
    if engine.model.config.model_type not in ('llama', 'qwen2'):
        raise ValueError('Only the tested Llama and Qwen2 adapters are supported.')
    captured, handles = {}, []
    n_heads = engine.model.config.num_attention_heads

    def hook_for(layer):
        def read_head_outputs(_module, projection_inputs):
            head_outputs = projection_inputs[0]  # [batch, tokens, heads * head_dim]
            if head_outputs.shape[0] != 1:
                raise ValueError('Capture expects one unpadded prompt.')
            final_token = head_outputs[0, -1].detach().float().cpu()
            captured[layer] = final_token.reshape(n_heads, -1)
        return read_head_outputs

    try:
        for layer, block in enumerate(engine.model.model.layers):
            projection = block.self_attn.o_proj
            handles.append(projection.register_forward_pre_hook(hook_for(layer)))
        engine.model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack([captured[layer] for layer in range(len(engine.model.model.layers))])


def collect_head_activations(engine, examples):
    """Collect [prompts, layers, heads, head_dim], keeping the prompt order."""
    activations = []
    for index, example in enumerate(examples):
        inputs = engine.inputs(example['Prompt'])
        example['Prompt_Tokens'] = inputs['input_ids'].shape[1]
        activations.append(capture_attention_heads(engine, inputs))
        print(f'Collected head activations {index + 1}/{len(examples)}', flush=True)
    return torch.stack(activations)


def collect_residual_center(engine, examples, layer):
    """Collect Standard post-block residuals and average for suppression."""
    return torch.stack([engine.capture_prompt_activation(engine.inputs(row['Prompt']), layer)
                        for row in examples]).mean(0)


def centered_ranks(values):
    """Return centered unit ranks along the last axis, averaging ties.

    Constant vectors become zeros. Other vectors' dot products equal Spearman rho."""
    ranks = rankdata(values, axis=-1, method='average')
    ranks -= ranks.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=-1, keepdims=True)
    return np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > 0)


def rsa_features(activations, examples, cross_only=True, cross_wording=True):
    """Prepare each head's ranked off-diagonal prompt cosine similarities.

    Args:
        activations: Array [examples, layers, heads, head_dim] of finite activations.
        examples: Aligned records with Item, Format, and Paraphrase.
        cross_only: If True, retain pairs with different objects and formats.
        cross_wording: With cross_only, also require different wording indices.
            Disable for audits where all prompts use the same new wording.
    Returns:
        (features, valid, i, j): unit centered similarity ranks [layers*heads, pairs],
        a per-head nonconstant mask, and the selected prompt-pair indices.
    Raises:
        ValueError: For nonfinite/misaligned data or fewer than three retained pairs.
    Notes:
        Excludes diagonals and duplicate symmetric pairs. Cosines are rounded to
        12 decimals so numerical noise does not break mathematically tied ranks.
    """
    x = np.asarray(activations, dtype=np.float64)
    if not np.isfinite(x).all() or x.shape[0] != len(examples):
        raise ValueError('Nonfinite activations or mismatched examples.')
    i, j = np.triu_indices(len(examples), 1)
    if cross_only:
        mask = np.array([examples[a]['Format'] != examples[b]['Format']
                         and canonical_item(examples[a]['Item']) != canonical_item(examples[b]['Item'])
                         and (not cross_wording or examples[a]['Paraphrase'] != examples[b]['Paraphrase'])
                         for a, b in zip(i, j)])
        i, j = i[mask], j[mask]
    if len(i) < 3:
        raise ValueError('RSA needs cross-item/format/wording pairs; supply at least two of each.')
    # Arrange each head as an independent [prompts, head_dim] matrix.
    x = x.transpose(1, 2, 0, 3).reshape(-1, x.shape[0], x.shape[-1])
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
    # Collapse floating-point noise in mathematically tied cosine values before
    # ranking. Precision is far finer than the captured float32 activations.
    similarities = np.round(np.clip(np.einsum('hnd,hmd->hnm', x, x)[:, i, j], -1, 1), 12)
    ranks = centered_ranks(similarities)
    valid = np.linalg.norm(ranks, axis=-1) > 0
    return ranks, valid, i, j


def rsa_scores(features, labels, i, j):
    """Correlate each head's similarity ranks with a same-label design.

    features is [heads, pairs]; i/j index prompt pairs and labels names each
    prompt's condition. Constant heads return zero; callers apply the validity
    mask. Both same-label and different-label pairs must be present."""
    labels = np.asarray(labels)
    target = (labels[i] == labels[j]).astype(float)
    if np.unique(target).size != 2:
        raise ValueError('RSA requires both same-concept and different-concept pairs.')
    return features @ centered_ranks(target)


def rank_heads(activations, examples, permutations=199, seed=42):
    """Rank heads by cross-object/format/wording instruction-condition RSA.

    Args:
        activations: Array [examples, layers, heads, head_dim].
        examples: Aligned discovery records containing Condition and design factors.
        permutations: Number of blockwise label shuffles; zero omits null testing.
        seed: NumPy seed controlling label permutations.
    Returns:
        Records sorted by descending Cross_RSA, with Layer, Head, Full_RSA,
        Format_RSA, and MaxT_P. Undefined Cross_RSA heads sort last.
    Notes:
        Labels shuffle within each item/wording/format block. Each null draw uses
        the maximum over valid heads, accounting for the head search under this
        exchangeability assumption. Selection is not gated on a p-value threshold.
        RSA measures condition-related geometry, not a causal creativity effect.
    """
    cross, valid, i, j = rsa_features(activations, examples)
    full, full_valid, a, b = rsa_features(activations, examples, cross_only=False)
    labels = np.array([e['Condition'] for e in examples])
    observed = rsa_scores(cross, labels, i, j)
    full_scores = rsa_scores(full, labels, a, b)
    format_scores = rsa_scores(full, [e['Format'] for e in examples], a, b)
    # Permute labels within matched item/wording/format blocks. Recompute the
    # maximum over ALL heads, accounting for selection in this exploratory null.
    blocks = {}
    for index, e in enumerate(examples):
        blocks.setdefault((e['Item'], e['Paraphrase'], e['Format']), []).append(index)
    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(permutations):
        shuffled = labels.copy()
        for indices in blocks.values():
            shuffled[indices] = rng.permutation(labels[indices])
        scores = rsa_scores(cross, shuffled, i, j)
        maxima.append(np.max(scores[valid]) if valid.any() else np.nan)
    n_heads = activations.shape[2]
    rows = []
    for index, score in enumerate(observed):
        rows.append(dict(Layer=index // n_heads, Head=index % n_heads,
                         Cross_RSA=float(score) if valid[index] else float('nan'),
                         Full_RSA=float(full_scores[index]) if full_valid[index] else float('nan'),
                         Format_RSA=float(format_scores[index]) if full_valid[index] else float('nan'),
                         MaxT_P=(1 + sum(m >= score for m in maxima)) / (permutations + 1)
                         if permutations and valid[index] else float('nan')))
    return sorted(rows, key=lambda r: (-r['Cross_RSA'] if np.isfinite(r['Cross_RSA']) else np.inf,
                                       r['Layer'], r['Head']))


@torch.inference_mode()
def project_heads(engine, head_values, selected):
    """Map selected heads to residual coordinates and sum their contributions.

    head_values is [layers, heads, head_dim]; selected contains Layer/Head
    records. Returns CPU float32 [hidden_size]. Each contribution is
    W_O[:, head_slice] @ head_value, without bias or normalization.
    Raw head coordinates from different layers cannot be added directly."""
    contributions = []
    for row in selected:
        layer, head = row['Layer'], row['Head']
        projection, section = engine.target(layer, head)
        weight = projection.weight[:, section].detach().float()
        z = head_values[layer, head].to(weight.device, dtype=torch.float32)
        contributions.append((weight @ z).cpu())
    if not contributions:
        raise ValueError('No selected heads.')
    return torch.stack(contributions).sum(0)


def write_csv(path, rows):
    """Write nonempty records to UTF-8 CSV, using the first row for column order."""
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_artifact(path, metadata, vector, center):
    """Save a concept/head vector in the shared causal runner's artifact format.

    Args:
        path: Destination .pt Path; parent must exist and the file is overwritten.
        metadata: JSON-serializable provenance and target fields, including layer,
            head, examples, vector_kind, and selection details.
        vector: Nonzero finite residual/head steering tensor.
        center: Baseline tensor in matching coordinates, used for suppression.
    Returns:
        None. Writes a torch artifact including its model and coordinate metadata.
    Raises:
        ValueError: If vector is zero or nonfinite. Center shape is checked later
            by the intervention engine when suppression is requested.
    """
    if not torch.isfinite(vector).all() or vector.norm() < 1e-10:
        raise ValueError('Zero or nonfinite concept vector.')
    torch.save(dict(**metadata, vector=vector, center=center, task_vectors={}), path)


def parse_args():
    """Validate disjoint discovery/audit items, wording splits and vector parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Llama-3.1-8B-Instruct')
    parser.add_argument('--revision')
    parser.add_argument('--task', choices=['Alternative Uses Task', 'Conditional Divergent Association Task', 'Divergent Remote Association Test'], default='Alternative Uses Task')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--items', nargs='+', default=['book', 'fork', 'paperclip', 'towel'])
    parser.add_argument('--audit-items', nargs='+', default=['umbrella', 'shoe'])
    parser.add_argument('--paraphrases', type=int, nargs='+', choices=range(3), default=[0, 1],
                        help='Two distinct discovery wording indices')
    parser.add_argument('--audit-paraphrase', type=int, choices=range(3), default=2)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--layer', type=int, default=16, help='Composite residual injection layer; zero based')
    parser.add_argument('--permutations', type=int, default=199)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.top_k < 1 or args.permutations < 1 or args.layer < 0:
        parser.error('Require positive top-k/permutations and nonnegative layer.')
    for values in (args.items, args.audit_items):
        if len(values) < 2 or len({canonical_item(i) for i in values}) != len(values):
            parser.error('Use at least two distinct items in each split.')
    if {canonical_item(i) for i in args.items} & {canonical_item(i) for i in args.audit_items}:
        parser.error('Discovery and audit objects must be disjoint.')
    if len(set(args.paraphrases)) != 2 or len(args.paraphrases) != 2 or args.audit_paraphrase in args.paraphrases:
        parser.error('Use two distinct discovery paraphrases and a separate audit paraphrase.')
    return args


def build_stimuli(args):
    """Build discovery, held-out audit and Standard calibration prompts."""
    discovery = stimuli(args.items, args.paraphrases, task=args.task)
    # Audit uses new objects and new Creative/Conventional/Effective suffixes.
    # Both formats retained; one wording means cross-wording masking is omitted
    # in the audit via an explicitly separate pair construction below.
    audit = stimuli(args.audit_items, [args.audit_paraphrase], task=args.task)
    baseline = stimuli(args.items, args.paraphrases, ['Standard'], task=args.task)
    return discovery, audit, baseline


def select_heads(args, discovery, discovery_activations):
    """Rank discovery heads and freeze the top defined cross-pair RSA scores."""
    ranking = rank_heads(discovery_activations.numpy(), discovery, args.permutations, args.seed)
    selected = [r for r in ranking if np.isfinite(r['Cross_RSA'])][:args.top_k]
    if len(selected) < args.top_k:
        raise ValueError('Not enough heads with defined RSA.')
    write_csv(args.output_dir / 'head_ranking.csv', ranking)
    print('Selected heads:', selected, flush=True)
    return selected


def audit_heads(args, audit, audit_activations, selected):
    """Evaluate the frozen selection on new items and wording without re-ranking."""
    # All audit prompts use the held-out wording; exclude same-item/same-format pairs.
    audit_features, audit_valid, i, j = rsa_features(audit_activations.numpy(), audit, cross_wording=False)
    audit_scores = rsa_scores(audit_features, [e['Condition'] for e in audit], i, j)
    n_heads = audit_activations.shape[2]
    audit_rows = [dict(**r, Audit_RSA=float(audit_scores[r['Layer']*n_heads+r['Head']])
                       if audit_valid[r['Layer']*n_heads+r['Head']] else float('nan')) for r in selected]
    write_csv(args.output_dir / 'selected_heads_audit.csv', audit_rows)


def export_vectors(engine, args, discovery, audit, baseline, discovery_activations,
                   audit_activations, baseline_activations, center, selected):
    """Save composite residual vectors, correctly centered suppression and direct-head contrasts."""
    standard = baseline_activations.mean(0)
    creative = discovery_activations[[i for i,e in enumerate(discovery) if e['Condition']=='Creative']].mean(0)
    positive = project_heads(engine, creative, selected)
    negative = project_heads(engine, standard, selected)
    meta = dict(protocol=PROTOCOL_VERSION, **provenance(engine, args), layer=args.layer,
                head=None, baseline='Standard', examples=[dict(Task=args.task, Item=i, Paraphrase=p)
                for i in args.items for p in args.paraphrases],
                audit_examples=[dict(Task=args.task, Item=i, Paraphrase=args.audit_paraphrase) for i in args.audit_items],
                selected_heads=[{'Layer':r['Layer'], 'Head':r['Head']} for r in selected],
                upstream_repository='https://github.com/gucioopielka/concept_vectors', upstream_commit=UPSTREAM_COMMIT,
                extraction_position='last token of formatted chat prompt',
                selection='cross-item/cross-format/cross-wording Spearman RSA on instruction condition',
                permutation_seed=args.seed, permutations=args.permutations,
                scaling='sum of W_O-projected selected-head means; no unit normalization')
    save_artifact(args.output_dir/'contrast.pt', dict(meta, vector_kind='selected_head_creative_minus_standard',
                  center_kind='full_standard_post_block_residual_mean'), positive-negative, center)
    save_artifact(args.output_dir/'creative_mean.pt', dict(meta, vector_kind='selected_head_creative_mean',
                  center_kind='full_standard_post_block_residual_mean'), positive, center)
    # Individual artifacts permit direct pre-o_proj head interventions using the
    # existing causal runner; head coordinates from different layers are never added.
    for r in selected:
        layer, head = r['Layer'], r['Head']
        single = dict(meta, layer=layer, head=head, vector_kind='individual_head_contrast',
                      scaling='raw mean pre-o_proj Creative minus Standard head activation')
        save_artifact(args.output_dir/f'head_l{layer}_h{head}.pt', single,
                      creative[layer,head]-standard[layer,head], standard[layer,head])
    write_csv(args.output_dir/'discovery_prompts.csv', discovery)
    write_csv(args.output_dir/'audit_prompts.csv', audit)
    write_csv(args.output_dir/'baseline_prompts.csv', baseline)
    torch.save(dict(discovery=discovery_activations, audit=audit_activations, baseline=baseline_activations), args.output_dir/'head_activations.pt')
    print(f'Saved rankings, held-out audit, composite vectors and head contrasts in {args.output_dir}')


def main():
    """Discover condition geometry, audit fixed heads, and export intervention vectors."""
    args = parse_args()
    discovery, audit, baseline = build_stimuli(args)
    print(f'{len(discovery)} discovery + {len(audit)} audit + {len(baseline)} baseline forward passes; top {args.top_k} heads.')
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    engine = load_engine(args.model, args.revision)
    engine.target(args.layer)
    if args.top_k > len(engine.model.model.layers) * engine.model.config.num_attention_heads:
        raise ValueError('top-k exceeds the number of heads.')
    # 1. COLLECT: prompt forward passes read every layer/head before o_proj.
    discovery_activations = collect_head_activations(engine, discovery)

    # 2. SELECT: RSA uses discovery activations only; freeze the chosen heads.
    selected = select_heads(args, discovery, discovery_activations)

    # 3. CHECK/CENTER: collect held-out and Standard prompts separately.
    audit_activations = collect_head_activations(engine, audit)
    baseline_activations = collect_head_activations(engine, baseline)
    center = collect_residual_center(engine, baseline, args.layer)
    audit_heads(args, audit, audit_activations, selected)

    # 4. SAVE: average selected heads and export Creative-minus-Standard vectors.
    export_vectors(engine, args, discovery, audit, baseline, discovery_activations,
                   audit_activations, baseline_activations, center, selected)


if __name__ == '__main__':
    main()
