"""Plot condition-related RSA from generated-response attention-head activations."""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

try:
    from .activation_grouping import response_unit_key
except ImportError:
    from activation_grouping import response_unit_key


def load_artifact(path):
    """Load an activation artifact produced by generation/generate_responses.py."""
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        artifact = torch.load(path, map_location="cpu")
    if artifact.get("format") != "prompt_response_head_means":
        raise ValueError(f"{path} is not a prompt_response_head_means artifact")
    return artifact


def average_repeats(rows, conditions, position):
    """Average one activation position across repeats per task/item/condition."""
    grouped = {}
    for row in rows:
        meta = row["metadata"]
        condition = str(meta["Condition"]).lower()
        if condition not in conditions:
            continue
        unit = response_unit_key(meta)
        if unit is None:
            continue
        key = (str(meta["Task"]), unit, condition)
        grouped.setdefault(key, []).append(row["activations"][position].float())
    return {key: torch.stack(values).mean(0).numpy() for key, values in grouped.items()}


def condition_rsa(item_activations, positive, negative, within_item_center=True):
    """Return [heads, layers] condition RSA, optionally removing item means first."""
    keys = sorted(item_activations)
    if len(keys) < 4:
        raise ValueError("Need at least two paired items (four item-condition means).")
    items = sorted({item for item, _ in keys})
    complete = [item for item in items
                if (item, positive) in item_activations and (item, negative) in item_activations]
    if len(complete) < 2:
        raise ValueError("Need at least two items with both requested conditions.")
    examples = [(item, condition) for item in complete for condition in (positive, negative)]
    values = np.stack([item_activations[key] for key in examples]).astype(np.float64)
    if values.ndim != 4:
        raise ValueError("Expected activation arrays shaped [layers, heads, head_dim].")
    n, n_layers, n_heads, _ = values.shape
    if within_item_center:
        values = values.reshape(len(complete), 2, n_layers, n_heads, -1)
        values = values - values.mean(axis=1, keepdims=True)
        values = values.reshape(n, n_layers, n_heads, -1)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    # Compare condition geometry only across different task items. This prevents
    # the paired copies of one item from dominating the same/different condition contrast.
    left, right = np.triu_indices(n, 1)
    keep = np.array([examples[i][0] != examples[j][0] for i, j in zip(left, right)])
    left, right = left[keep], right[keep]
    target = np.array([examples[i][1] == examples[j][1] for i, j in zip(left, right)], dtype=float)
    target_ranks = rankdata(target, method="average")
    target_ranks -= target_ranks.mean()
    target_norm = np.linalg.norm(target_ranks)
    if target_norm == 0:
        raise ValueError("RSA target has no same-condition/different-condition variation.")
    target_ranks /= target_norm
    scores = np.full((n_heads, n_layers), np.nan, dtype=float)
    for layer in range(n_layers):
        for head in range(n_heads):
            head_values = values[:, layer, head, :]
            similarities = np.clip(np.einsum("id,jd->ij", head_values, head_values)[left, right], -1, 1)
            ranks = rankdata(similarities, method="average")
            ranks -= ranks.mean()
            norm = np.linalg.norm(ranks)
            if norm > 0:
                scores[head, layer] = float(np.dot(ranks / norm, target_ranks))
    return scores, len(complete)


def plot_task(task, activations, output_dir, positive, negative, position, within_item_center):
    scores, n_items = condition_rsa(activations, positive, negative, within_item_center)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, scores.shape[1] * .28),
                                    max(5, scores.shape[0] * .18)),
                           constrained_layout=True)
    image = ax.imshow(scores, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("Layer (zero based)")
    ax.set_ylabel("Attention head (zero based)")
    ax.set_title(f"{task}: {position} {positive} vs {negative} RSA ({n_items} matched units)")
    fig.colorbar(image, ax=ax, label="Spearman RSA (condition similarity)")
    center_suffix = "within_item_centered" if within_item_center else "raw_geometry"
    plot_name = f"response_head_rsa_{position}_{center_suffix}"
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{plot_name}.{extension}", dpi=180)
    plt.close(fig)

    records = [dict(Task=task, Layer=layer, Head=head,
                    RSA=float(scores[head, layer]) if np.isfinite(scores[head, layer]) else "")
               for layer in range(scores.shape[1]) for head in range(scores.shape[0])]
    pd.DataFrame(records).to_csv(output_dir / f"{plot_name}.csv", index=False)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activations", type=Path,
                        help="*.pt artifact from extract_response_activations.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive", default="creative")
    parser.add_argument("--negative", default="non_creative")
    parser.add_argument("--within-item-center", action=argparse.BooleanOptionalAction, default=True,
                        help="Remove each item's two-condition mean before RSA (default: enabled)")
    parser.add_argument("--position", choices=("response", "response-predictor", "last-prompt"),
                        default="response",
                        help="Activation position to analyze (default: response)")
    args = parser.parse_args()
    positive, negative = args.positive.lower(), args.negative.lower()
    if positive == negative:
        parser.error("--positive and --negative must differ")
    artifact = load_artifact(args.activations)
    means = average_repeats(artifact["rows"], {positive, negative}, args.position)
    tasks = sorted({task for task, _item, _condition in means})
    if not tasks:
        raise SystemExit("No records match the requested conditions.")
    for task in tasks:
        task_values = {(item, condition): value for (row_task, item, condition), value in means.items()
                       if row_task == task}
        try:
            output = plot_task(task, task_values, args.output_dir / re.sub(r"[^A-Za-z0-9_-]+", "_", task),
                               positive, negative, args.position, args.within_item_center)
        except ValueError as exc:
            print(f"Skipping {task}: {exc}")
            continue
        suffix = "within_item_centered" if args.within_item_center else "raw_geometry"
        print(output / f"response_head_rsa_{args.position}_{suffix}.png", flush=True)


if __name__ == "__main__":
    main()
