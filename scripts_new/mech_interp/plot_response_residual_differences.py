"""Plot layerwise residual-stream differences between two response conditions."""
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
    if artifact.get("format") != "prompt_response_residual_means":
        raise ValueError(f"{path} is not a prompt_response_residual_means artifact")
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
    tasks = sorted({task for task, _item, _condition in item_means})
    result = {}
    for task in tasks:
        items = sorted({item for row_task, item, _condition in item_means if row_task == task})
        complete = [item for item in items
                    if (task, item, creative) in item_means and (task, item, non_creative) in item_means]
        if not complete:
            print(f"Skipping {task}: no items have both {creative!r} and {non_creative!r} responses.")
            continue
        differences = []
        relative_shifts = []
        for item in complete:
            creative_values = item_means[(task, item, creative)]
            non_creative_values = item_means[(task, item, non_creative)]
            delta = creative_values - non_creative_values
            denominator = (creative_values.norm(dim=-1) + non_creative_values.norm(dim=-1)) / 2
            relative_shifts.append(torch.where(denominator > 0, delta.norm(dim=-1) / denominator,
                                               torch.full_like(denominator, float("nan"))))
            differences.append(delta)
        result[task] = (torch.stack(differences), torch.stack(relative_shifts), complete)
    return result


def save_task_plot(task, item_differences, relative_shifts, items, position,
                   creative, non_creative, output_dir):
    """Save signed per-dimension differences and their per-layer L2 norms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    item_values = item_differences.numpy()
    values = item_values.mean(axis=0)
    if values.ndim != 2:
        raise ValueError("Expected residual difference tensor shaped [layers, hidden_size].")

    limit = float(np.abs(values).max())
    if limit == 0:
        limit = 1.0
    fig, ax = plt.subplots(figsize=(12, max(5, values.shape[0] * 0.22)), constrained_layout=True)
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xlabel("Residual-stream dimension")
    ax.set_ylabel("Layer (zero based)")
    ax.set_title(f"{task}: {creative} − {non_creative} {position} residual ({len(items)} matched units)")
    fig.colorbar(image, ax=ax, label="Mean activation difference")
    stem = f"residual_difference_{position}"
    fig.savefig(output_dir / f"{stem}.png", dpi=180)
    fig.savefig(output_dir / f"{stem}.pdf", dpi=180)
    plt.close(fig)

    layer_norms = np.linalg.norm(values, axis=1)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(np.arange(len(layer_norms)), layer_norms, marker="o")
    ax.set_xlabel("Layer (zero based)")
    ax.set_ylabel("L2 norm of mean difference vector")
    ax.set_title(f"{task}: {creative} − {non_creative} {position} residual ({len(items)} matched units)")
    ax.grid(True, alpha=0.3)
    norm_stem = f"residual_difference_norm_{position}"
    fig.savefig(output_dir / f"{norm_stem}.png", dpi=180)
    fig.savefig(output_dir / f"{norm_stem}.pdf", dpi=180)
    plt.close(fig)

    pd.DataFrame({
        "Layer": np.arange(len(layer_norms)),
        "L2_Difference": layer_norms,
        "Mean_Absolute_Difference": np.abs(values).mean(axis=1),
    }).to_csv(output_dir / f"{norm_stem}.csv", index=False)

    relative_values = relative_shifts.numpy()  # [items, layers]
    mean_relative = np.nanmean(relative_values, axis=0)
    sd_relative = np.nanstd(relative_values, axis=0, ddof=1) if len(items) > 1 else np.zeros_like(mean_relative)
    relative_stem = f"residual_difference_relative_norm_{position}"
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    layers = np.arange(len(mean_relative))
    ax.plot(layers, mean_relative, marker="o")
    ax.fill_between(layers, np.maximum(0, mean_relative - sd_relative),
                    mean_relative + sd_relative, alpha=0.2)
    ax.set_xlabel("Layer (zero based)")
    ax.set_ylabel("Mean per-item relative L2 shift")
    ax.set_title(f"{task}: relative {creative} − {non_creative} shift ({len(items)} matched units)")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / f"{relative_stem}.png", dpi=180)
    fig.savefig(output_dir / f"{relative_stem}.pdf", dpi=180)
    plt.close(fig)
    pd.DataFrame({"Layer": layers, "Mean_Per_Item_Relative_Shift": mean_relative,
                  "SD_Per_Item_Relative_Shift": sd_relative, "N_Units": len(items)}).to_csv(
                      output_dir / f"{relative_stem}.csv", index=False)
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
    parser.add_argument("activations", type=Path, help="Residual artifact from extract_response_residuals.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--creative", default="creative", help="Creative condition label")
    parser.add_argument("--non-creative", default="non_creative", help="Comparison condition label")
    parser.add_argument("--position", choices=("response", "response-predictor", "last-prompt"),
                        default="response",
                        help="Select answer-token, answer-predictor, or final-prompt residuals")
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
    for task, (item_differences, relative_shifts, items) in differences.items():
        output = save_task_plot(task, item_differences, relative_shifts, items, args.position, creative, non_creative,
                                args.output_dir / re.sub(r"[^A-Za-z0-9_-]+", "_", task))
        print(output / f"residual_difference_{args.position}.png", flush=True)
        print(output / f"residual_difference_norm_{args.position}.png", flush=True)
        print(output / f"residual_difference_relative_norm_{args.position}.png", flush=True)


if __name__ == "__main__":
    main()
