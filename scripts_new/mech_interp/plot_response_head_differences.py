"""Plot layer/head attention activation differences between two conditions."""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from .activation_grouping import response_unit_key
except ImportError:
    from activation_grouping import response_unit_key


def load_artifact(path):
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        artifact = torch.load(path, map_location="cpu")
    if artifact.get("format") != "prompt_response_head_means":
        raise ValueError(f"{path} is not a prompt_response_head_means artifact")
    return artifact


def average_repeats(rows, position):
    """Average repeat activations per task/item/condition."""
    grouped = {}
    for row in rows:
        meta = row["metadata"]
        condition = str(meta.get("Condition", "")).strip().lower()
        if not condition:
            continue
        unit = response_unit_key(meta)
        if unit is None:
            continue
        key = (str(meta.get("Task", "")), unit, condition)
        grouped.setdefault(key, []).append(row["activations"][position].float())
    return {key: torch.stack(values).mean(0) for key, values in grouped.items()}


def paired_differences(item_means, creative, non_creative):
    differences = {}
    tasks = sorted({task for task, _item, _condition in item_means})
    for task in tasks:
        items = sorted({item for row_task, item, _condition in item_means if row_task == task})
        complete = [item for item in items
                    if (task, item, creative) in item_means and (task, item, non_creative) in item_means]
        if not complete:
            print(f"Skipping {task}: no items have both {creative!r} and {non_creative!r} responses.")
            continue
        item_diffs = []
        relative_shifts = []
        for item in complete:
            creative_values = item_means[(task, item, creative)]
            non_creative_values = item_means[(task, item, non_creative)]
            delta = creative_values - non_creative_values
            denominator = (creative_values.norm(dim=-1) + non_creative_values.norm(dim=-1)) / 2
            relative_shifts.append(torch.where(denominator > 0, delta.norm(dim=-1) / denominator,
                                               torch.full_like(denominator, float("nan"))))
            item_diffs.append(delta)
        differences[task] = (torch.stack(item_diffs), torch.stack(relative_shifts), complete)
    return differences


def save_task_plots(task, item_differences, relative_shifts, items, position,
                    creative, non_creative, output_dir):
    """Save signed mean differences and two summaries of per-head shift size."""
    output_dir.mkdir(parents=True, exist_ok=True)
    item_values = item_differences.numpy()
    if item_values.ndim != 4:
        raise ValueError("Expected paired difference tensor shaped [items, layers, heads, head_dim].")
    if item_values.shape[0] != len(items):
        raise ValueError("Item labels do not match the number of paired differences.")
    values = item_values.mean(axis=0)
    if values.ndim != 3:
        raise ValueError("Expected difference tensor shaped [layers, heads, head_dim].")
    n_layers, n_heads, head_dim = values.shape
    stem = f"head_difference_{position}"

    # Signed component values: one row per layer/head, columns are head dimensions.
    flattened = values.reshape(n_layers * n_heads, head_dim)
    limit = float(np.abs(flattened).max())
    if limit == 0:
        limit = 1.0
    fig, ax = plt.subplots(figsize=(12, max(7, min(18, n_layers * 0.45))), constrained_layout=True)
    image = ax.imshow(flattened, origin="lower", aspect="auto", cmap="coolwarm",
                      vmin=-limit, vmax=limit)
    for layer in range(1, n_layers):
        ax.axhline(layer * n_heads - 0.5, color="black", linewidth=0.35, alpha=0.6)
    ax.set_yticks([layer * n_heads + (n_heads - 1) / 2 for layer in range(n_layers)])
    ax.set_yticklabels([str(layer) for layer in range(n_layers)])
    ax.set_xlabel("Attention head dimension")
    ax.set_ylabel("Layer (each row group contains its heads)")
    ax.set_title(f"{task}: {creative} − {non_creative} {position} head activations ({len(items)} matched units)")
    fig.colorbar(image, ax=ax, label="Mean activation difference")
    fig.savefig(output_dir / f"{stem}_components.png", dpi=180)
    fig.savefig(output_dir / f"{stem}_components.pdf", dpi=180)
    plt.close(fig)

    # Magnitude summary: one cell per layer/head.
    norms = np.linalg.norm(values, axis=-1)
    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.28), max(5, n_heads * 0.18)),
                           constrained_layout=True)
    image = ax.imshow(norms.T, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("Layer (zero based)")
    ax.set_ylabel("Attention head (zero based)")
    ax.set_title(f"{task}: ||{creative} − {non_creative}|| by layer/head ({len(items)} matched units)")
    fig.colorbar(image, ax=ax, label="L2 norm of mean difference vector")
    fig.savefig(output_dir / f"{stem}_norm.png", dpi=180)
    fig.savefig(output_dir / f"{stem}_norm.pdf", dpi=180)
    plt.close(fig)

    pd.DataFrame([
        {"Layer": layer, "Head": head,
         "L2_Difference": float(norms[layer, head]),
         "Mean_Absolute_Difference": float(np.abs(values[layer, head]).mean())}
        for layer in range(n_layers) for head in range(n_heads)
    ]).to_csv(output_dir / f"{stem}_norm.csv", index=False)

    # Unlike ||mean(item delta)||, this averages each item's shift magnitude
    # before aggregation, so shifts in different directions do not cancel.
    per_item_norms = np.linalg.norm(item_values, axis=-1)  # [items, layers, heads]
    mean_per_item_norm = per_item_norms.mean(axis=0)
    sd_per_item_norm = per_item_norms.std(axis=0, ddof=1) if len(items) > 1 else np.zeros_like(mean_per_item_norm)
    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.28), max(5, n_heads * 0.18)),
                           constrained_layout=True)
    image = ax.imshow(mean_per_item_norm.T, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("Layer (zero based)")
    ax.set_ylabel("Attention head (zero based)")
    ax.set_title(f"{task}: mean per-unit ||{creative} − {non_creative}|| ({len(items)} matched units)")
    fig.colorbar(image, ax=ax, label="Mean per-item L2 shift")
    fig.savefig(output_dir / f"{stem}_per_item_norm.png", dpi=180)
    fig.savefig(output_dir / f"{stem}_per_item_norm.pdf", dpi=180)
    plt.close(fig)

    pd.DataFrame([
        {"Layer": layer, "Head": head,
         "Mean_Per_Item_L2_Shift": float(mean_per_item_norm[layer, head]),
         "SD_Per_Item_L2_Shift": float(sd_per_item_norm[layer, head]),
         "N_Items": len(items)}
        for layer in range(n_layers) for head in range(n_heads)
    ]).to_csv(output_dir / f"{stem}_per_item_norm.csv", index=False)

    relative_values = relative_shifts.numpy()  # [items, layers, heads]
    mean_relative = np.nanmean(relative_values, axis=0)
    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.28), max(5, n_heads * 0.18)),
                           constrained_layout=True)
    image = ax.imshow(mean_relative.T, origin="lower", aspect="auto", cmap="magma", vmin=0)
    ax.set_xlabel("Layer (zero based)")
    ax.set_ylabel("Attention head (zero based)")
    ax.set_title(f"{task}: mean per-item relative shift ({len(items)} matched units)")
    fig.colorbar(image, ax=ax, label="Relative L2 shift")
    fig.savefig(output_dir / f"{stem}_relative_norm.png", dpi=180)
    fig.savefig(output_dir / f"{stem}_relative_norm.pdf", dpi=180)
    plt.close(fig)
    pd.DataFrame([
        {"Layer": layer, "Head": head,
         "Mean_Per_Item_Relative_Shift": float(mean_relative[layer, head]),
         "N_Units": len(items)}
        for layer in range(n_layers) for head in range(n_heads)
    ]).to_csv(output_dir / f"{stem}_relative_norm.csv", index=False)
    torch.save({
        "task": task,
        "creative_condition": creative,
        "non_creative_condition": non_creative,
        "position": position,
        "items": items,
        "difference": item_differences.mean(0),
        "item_differences": item_differences,
    }, output_dir / f"{stem}.pt")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activations", type=Path, help="Artifact from extract_response_activations.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--creative", default="creative", help="Creative condition label")
    parser.add_argument("--non-creative", default="non_creative", help="Comparison condition label")
    parser.add_argument("--position", choices=("response", "response-predictor", "last-prompt"),
                        default="response",
                        help="Select answer-token, answer-predictor, or final-prompt head activations")
    args = parser.parse_args()
    creative = args.creative.strip().lower()
    non_creative = args.non_creative.strip().lower()
    if not creative or not non_creative or creative == non_creative:
        parser.error("--creative and --non-creative must be distinct non-empty condition labels")

    artifact = load_artifact(args.activations)
    means = average_repeats(artifact["rows"], args.position)
    differences = paired_differences(means, creative, non_creative)
    if not differences:
        raise SystemExit("No task has matched items in both requested conditions.")
    for task, (difference, relative_shifts, items) in differences.items():
        output = save_task_plots(task, difference, relative_shifts, items, args.position, creative, non_creative,
                                 args.output_dir / re.sub(r"[^A-Za-z0-9_-]+", "_", task))
        print(output / f"head_difference_{args.position}_components.png", flush=True)
        print(output / f"head_difference_{args.position}_norm.png", flush=True)
        print(output / f"head_difference_{args.position}_per_item_norm.png", flush=True)
        print(output / f"head_difference_{args.position}_relative_norm.png", flush=True)


if __name__ == "__main__":
    main()
