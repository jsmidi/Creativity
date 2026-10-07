"""Paired response effects with item/block bootstrap; one frozen run per analysis."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import canonical_item


def paired_effect(frame, metric, treatment, control, draws=2000, seed=42):
    """Estimate a response-level treatment effect with an item/block bootstrap.

    Args:
        frame: DataFrame for one run with Run_ID, Block_ID, Response_ID, Item,
            Condition, and the requested numeric metric.
        metric: Score column name.
        treatment: Condition label whose score is added in the contrast.
        control: Condition label whose score is subtracted.
        draws: Number of bootstrap draws; CLI requires at least 100.
        seed: NumPy bootstrap seed.
    Returns:
        Dictionary with equal-item-weighted mean paired difference, percentile
        95% interval, complete/incomplete pair counts, and analysis metadata.
    Raises:
        ValueError: For missing columns, multiple runs, duplicate block/condition
            rows, or no complete scored pairs.
    Notes:
        Resamples objects and paired differences within objects. Missing pairs are
        excluded; with fewer than two complete pairs interval bounds are NaN.
        These exploratory intervals do not correct for multiple comparisons.
    """
    needed = ["Run_ID", "Block_ID", "Response_ID", "Item", "Condition", metric]
    if not set(needed).issubset(frame):
        raise ValueError(f"Require response-level v2 data with columns {needed}")
    if frame["Run_ID"].nunique() != 1:
        raise ValueError("Analyze one frozen run at a time; do not pool runs with different settings.")
    for column in ["Temperature", "Top_P", "Top_K"]:
        if column in frame and frame[column].nunique() > 1:
            raise ValueError(f"Analyze one {column} setting at a time.")
    subset = frame[frame["Condition"].isin([control, treatment])].copy()
    if subset.duplicated(["Block_ID", "Condition"]).any():
        raise ValueError("Multiple rows per block/condition: input must be response-level.")
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    table = subset.pivot(index="Block_ID", columns="Condition", values=metric).reindex(columns=[control, treatment])
    complete = table.dropna()
    if complete.empty:
        raise ValueError("No complete scored pairs for requested contrast.")
    differences = complete[treatment] - complete[control]
    item_map = subset.drop_duplicates("Block_ID").set_index("Block_ID")["Item"].map(canonical_item)
    grouped = [group.to_numpy() for _, group in differences.groupby(item_map.reindex(differences.index))]
    rng = np.random.default_rng(seed)
    effects = []
    greedy = "Temperature" in frame and pd.to_numeric(frame["Temperature"]).eq(0).all()
    for _ in range(0 if greedy else draws):
        # AUT objects get equal weight, independently of successful response count.
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        effects.append(np.mean([rng.choice(grouped[i], size=len(grouped[i]), replace=True).mean() for i in sampled]))
    enough = len(differences) > 1 and not greedy
    interval = np.quantile(effects, [.025, .975]) if enough else [np.nan, np.nan]
    return dict(Metric=metric, Treatment=treatment, Control=control,
                Mean_Difference=float(np.mean([x.mean() for x in grouped])),
                CI_Lower=interval[0], CI_Upper=interval[1], Complete_Pairs=len(complete),
                Incomplete_Pairs=len(table) - len(complete), Items=len(grouped),
                Scope="greedy decoding; descriptive only, no sampling CI" if greedy else ("within-item sampling only" if len(grouped) == 1 else "item and paired-generation bootstrap"),
                Bootstrap_Draws=0 if greedy else draws, Bootstrap_Seed=seed)


def main():
    """Parse the response-effect CLI and save requested contrasts per model/task group.

    Args:
        None. Reads sys.argv: input CSV, metric, treatment/control labels,
        draws/seed for the bootstrap, and output CSV Path.
    Returns:
        None. Writes and prints the result table. Each group must represent one
        run; this tool does not merge repeated experiments or select hypotheses.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--treatment", default="Creative")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--control", default="Standard")
    controls.add_argument("--controls", nargs="+", help="Separate controls; each is compared with --treatment")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("analysis/paired_effects.csv"))
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("Use at least 100 bootstrap draws.")
    frame = pd.read_csv(args.input, keep_default_na=False)
    results = []
    for (model, task), group in frame.groupby(["Model", "Task"]):
        for control in dict.fromkeys(args.controls or [args.control]):
            results.append(dict(Model=model, Task=task, **paired_effect(group, args.metric, args.treatment, control, args.draws, args.seed)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
