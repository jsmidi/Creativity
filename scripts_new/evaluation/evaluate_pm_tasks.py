"""Score generated AUT, DAT, CDAT, and DRAT response CSVs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .metrics import (cdat_appropriateness_gate, score_aut, score_cdat,
                          score_dat, score_drat)
except ImportError:
    from metrics import cdat_appropriateness_gate, score_aut, score_cdat, score_dat, score_drat


TASKS = {"aut": "Alternative Uses Task", "dat": "Divergent Association Task",
         "cdat": "Conditional Divergent Association Task",
         "drat": "Divergent Remote Association Test"}
METRICS = {"aut": ["Pairwise_Semantic_Distance_Proxy", "Semantic_Distance_Proxy"],
           "dat": ["DAT_GloVe_Score", "DAT_MPNet_Proxy"],
           "cdat": ["CDAT_Novelty", "CDAT_Appropriateness"], "drat": ["DRAT_Score"]}


def evaluate_file(path, task, model, noun_pool=None, dat_model=None):
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"Task", "Item", "Condition", "Response"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain {', '.join(sorted(required))}")
    frame = frame[frame.Task == TASKS[task]].copy()
    if frame.empty:
        return None
    if task == "aut":
        scores = [score_aut(row.Response, row.Item, model.encode) for row in frame.itertuples()]
    elif task == "dat":
        scores = [score_dat(row.Response, model.encode, dat_model)
                  for row in frame.itertuples()]
    elif task == "cdat":
        scores = [score_cdat(row.Response, row.Item, model.encode)
                  for row in frame.itertuples()]
    else:
        scores = [score_drat(row.Response, row.Item, model.encode, noun_pool)
                  for row in frame.itertuples()]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(scores)], axis=1)


def plot_results(frame, task, output):
    metrics = [metric for metric in METRICS[task] if metric in frame]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        summary = frame.groupby(["Model", "Condition"])[metric].agg(["mean", "std"]).reset_index()
        models = list(summary.Model.unique())
        conditions = list(summary.Condition.unique())
        x = np.arange(len(models))
        width = .8 / max(1, len(conditions))
        for i, condition in enumerate(conditions):
            values = summary[summary.Condition == condition].set_index("Model").reindex(models)
            ax.bar(x + (i - (len(conditions) - 1) / 2) * width, values["mean"], width,
                   yerr=values["std"], capsize=3, label=condition)
        ax.set_xticks(x, models, rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(title="Condition")
    fig.suptitle(f"{TASKS[task]}: mean ± 1 SD across responses")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def aggregate_run_metrics(frame, task):
    """Average items within each run, then weight run means equally."""
    replicate_column = next((column for column in ("RUN_ID", "Run_ID", "Model_Seed")
                             if column in frame), None)
    run_column = "Aggregate_Run_ID"
    work = frame.copy()
    work[run_column] = work[replicate_column].astype(str) if replicate_column else "single"
    dimensions = ["Model", run_column]
    if "Temperature" in work:
        dimensions.insert(1, "Temperature")
    if "Model_Seed" in work:
        dimensions.append("Model_Seed")
    dimensions.append("Condition")

    seed_rows = []
    metric_columns = [metric for metric in METRICS[task] + (["CDAT_Score"] if task == "cdat" else [])
                      if metric in work]
    for metric in metric_columns:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        metric_work = work.dropna(subset=[metric])
        if metric == "CDAT_Score":  # One score is shared by the seed/condition gate.
            grouped = metric_work.drop_duplicates(dimensions).groupby(dimensions, dropna=False)[metric].mean()
            values = grouped.rename("Mean").reset_index()
            values["Metric"] = metric
        else:
            by_item = metric_work.groupby(dimensions + ["Item"], dropna=False)[metric].agg(
                Item_Mean="mean", Responses="size", Scored_Responses="count").reset_index()
            values = by_item.groupby(dimensions, dropna=False).agg(
                Mean=("Item_Mean", "mean"), Items=("Item", "nunique"),
                Responses=("Responses", "sum"),
                Scored_Responses=("Scored_Responses", "sum")).reset_index()
            values["Metric"] = metric
        seed_rows.append(values)
    seed_means = pd.concat(seed_rows, ignore_index=True) if seed_rows else pd.DataFrame()
    summary_dimensions = [column for column in dimensions
                          if column not in {run_column, "Model_Seed"}]
    aggregations = {"Mean": ("Mean", "mean"), "Run_SD": ("Mean", "std"),
                    "Runs": (run_column, "nunique")}
    if "Model_Seed" in seed_means:
        aggregations["Distinct_Model_Seeds"] = ("Model_Seed", "nunique")
    if "Responses" in seed_means:
        aggregations.update(Responses=("Responses", "sum"),
                            Scored_Responses=("Scored_Responses", "sum"))
    summary = seed_means.groupby(summary_dimensions + ["Metric"], dropna=False).agg(
        **aggregations).reset_index()
    return seed_means, summary


def plot_run_average(summary, task, output):
    """Plot equal-weighted run means with one sample SD across runs."""
    metrics = [metric for metric in METRICS[task] + (["CDAT_Score"] if task == "cdat" else [])
               if metric in set(summary.Metric)]
    temperatures = list(summary.Temperature.unique()) if "Temperature" in summary else [None]
    fig, axes = plt.subplots(len(temperatures), len(metrics),
                             figsize=(7 * len(metrics), 4.8 * len(temperatures)), squeeze=False)
    for row_index, temperature in enumerate(temperatures):
        temperature_data = summary if temperature is None else summary[summary.Temperature == temperature]
        for column_index, metric in enumerate(metrics):
            ax = axes[row_index, column_index]
            data = temperature_data[temperature_data.Metric == metric]
            models = list(data.Model.unique())
            conditions = list(data.Condition.unique())
            x = np.arange(len(models))
            width = .8 / max(1, len(conditions))
            for i, condition in enumerate(conditions):
                values = data[data.Condition == condition].set_index("Model").reindex(models)
                ax.bar(x + (i - (len(conditions) - 1) / 2) * width, values.Mean, width,
                       yerr=values.Run_SD.fillna(0), capsize=3, label=condition)
            ax.set_xticks(x, models, rotation=20, ha="right")
            ax.set_ylabel(metric)
            ax.set_title(metric if temperature is None else f"{metric} · temperature {temperature}")
            ax.legend(title="Condition")
    fig.suptitle(f"{TASKS[task]}: mean ± 1 SD across generation runs")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Generated response CSV files")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--noun-vocabulary", type=Path,
                        help="Frozen noun pool required for CDAT baseline and DRAT threshold")
    parser.add_argument("--glove", type=Path, help="GloVe 840B 300d file for the original DAT score")
    parser.add_argument("--dictionary", type=Path, help="Olson DAT words.txt for official GloVe validation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random baseline calibration")
    args = parser.parse_args()
    if {"cdat", "drat"} & set(args.tasks) and args.noun_vocabulary is None:
        parser.error("CDAT/DRAT scoring requires --noun-vocabulary for author-style calibration")
    if "dat" in args.tasks and (args.glove is None or args.dictionary is None):
        parser.error("DAT scoring requires --glove and --dictionary; GloVe is the default DAT score")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embedding_model, device=args.device)
    dat_model = None
    if args.glove:
        if not args.glove.is_file() or not args.dictionary.is_file():
            parser.error("GloVe file and DAT dictionary must both exist")
        import importlib.util
        repo_root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("olson_dat", repo_root / "databases/dat.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dat_model = module.Model(str(args.glove), str(args.dictionary))
    noun_pool = None
    if args.noun_vocabulary:
        if not args.noun_vocabulary.is_file():
            parser.error(f"Noun vocabulary not found: {args.noun_vocabulary}")
        nouns = sorted({noun.strip().lower() for noun in args.noun_vocabulary.read_text().splitlines()
                        if noun.strip()})
        if len(nouns) < 10:
            parser.error("Noun vocabulary must contain at least 10 nouns")
        noun_pool = model.encode(nouns, convert_to_numpy=True, show_progress_bar=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        # The frozen noun pool defines random baselines/calibration thresholds;
        # it must not filter the model's generated response words.
        frames = [evaluate_file(path, task, model, noun_pool, dat_model)
                  for path in args.inputs]
        frames = [frame for frame in frames if frame is not None]
        if not frames:
            print(f"Skipping {task.upper()}: no matching responses")
            continue
        results = pd.concat(frames, ignore_index=True)
        task_dir = args.output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(task_dir / f"{task}_responses.csv", index=False)
        if task == "cdat":
            gates = cdat_appropriateness_gate(results, model, nouns, args.seed)
            gates.to_csv(task_dir / "cdat_gates.csv", index=False)
            gate_keys = [column for column in gates.columns
                         if column in results and column in
                         {"Model", "Temperature", "Model_Seed", "RUN_ID", "Run_ID", "Condition"}]
            results = results.merge(gates[gate_keys + ["Gate_Passed", "CDAT_Score"]],
                                    on=gate_keys, how="left", validate="many_to_one")
            results.to_csv(task_dir / f"{task}_responses.csv", index=False)
        plot_results(results, task, task_dir / f"{task}_results.png")
        run_means, run_average = aggregate_run_metrics(results, task)
        run_means.to_csv(task_dir / "run_means.csv", index=False)
        run_average.to_csv(task_dir / "run_average.csv", index=False)
        plot_run_average(run_average, task, task_dir / "run_average.png")
        print(f"Saved {len(results)} {task.upper()} scores to {task_dir}")


if __name__ == "__main__":
    main()
