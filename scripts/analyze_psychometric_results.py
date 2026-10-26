"""Regenerate task plots, paired condition effects, seed averages and paired seed tests.

Consumes scored analysis folders; no model loading or generation is performed.
"""
import argparse
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel
import seaborn as sns

from analyze_effects import holm, paired_effect
from scoring_common import descriptive_plot
from experiment import CONDITION_TEXT


PLOT_METRICS = {
    "aut": {name: name for name in ("Pairwise_Semantic_Distance_Proxy", "Semantic_Distance_Proxy",
                                   "DB_Rarity_Score", "Relative_Novelty_ICF")},
    "dat": {"DAT_GloVe_Score": "DAT_GloVe_Score", "Rarity_Score": "Lexical rarity (missing if unavailable)"},
    "drat": {"DRAT_Score": "DRAT score"},
    "cdat": {"CDAT_Novelty": "CDAT novelty (before condition appropriateness gate)",
             "CDAT_Appropriateness": "CDAT appropriateness"},
}


TASKS = {
    "aut": ("Pairwise_Semantic_Distance_Proxy", "AUT", "Mean pairwise cosine distance"),
    "dat": ("DAT_GloVe_Score", "DAT", "Official GloVe score"),
    "drat": ("DRAT_Score", "DRAT", "DRAT score"),
    "cdat": ("CDAT_Novelty", "CDAT", "Descriptive novelty (before gating)"),
}


def plot_root(root, seeds):
    """Average items within seeds, then plot equally weighted seeds with one SD."""
    records, gates = [], []
    models = set()
    for seed in seeds:
        for task, (metric, _, _) in TASKS.items():
            frame = pd.read_csv(root / f"seed_{seed}" / task / f"{task}_responses.csv", keep_default_na=False)
            if frame.Response_ID.duplicated().any():
                raise ValueError("Expected one row per response")
            if set(frame.Condition) != set(CONDITION_TEXT):
                raise ValueError("Expected all five conditions in each seed/task")
            if not pd.to_numeric(frame.Temperature).eq(0.7).all():
                raise ValueError("Expected temperature 0.7 throughout")
            models.update(frame.Model.unique())
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
            for condition, group in frame.groupby("Condition"):
                # Preserve equal item weights if a future run has repeated responses.
                mean = group.groupby("Item")[metric].mean().mean()
                records.append(dict(Task=task, Metric=metric, Condition=condition, Seed=seed,
                                    Mean=mean, Responses=len(group), Scored_Responses=group[metric].count()))
            if task == "cdat":
                gate = pd.read_csv(root / f"seed_{seed}/cdat/cdat_gates.csv")
                gate["Seed"] = seed
                gates.append(gate)
    if len(models) != 1:
        raise ValueError("Each plot root must contain one model")
    means = pd.DataFrame(records)
    means["Model"] = next(iter(models))
    if means.Mean.isna().any():
        raise ValueError("A seed/condition has no scores; cannot plot all seeds")
    summary = means.groupby(["Task", "Metric", "Condition"], sort=False).agg(
        Mean=("Mean", "mean"), Seed_SD=("Mean", "std"), Seeds=("Seed", "nunique"),
        Responses=("Responses", "sum"), Scored_Responses=("Scored_Responses", "sum")).reset_index()
    assert summary.Seeds.eq(len(seeds)).all()
    gate_summary = pd.concat(gates).groupby("Condition").Gate_Passed.agg(["sum", "count"])
    order = list(CONDITION_TEXT)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (task, (_, title, ylabel)) in zip(axes.flat, TASKS.items()):
        # Use the same Seaborn defaults, hue order and caps as descriptive_plot.
        sns.barplot(data=means[means.Task.eq(task)], x="Model", y="Mean",
                    hue="Condition", hue_order=order, errorbar="sd", capsize=0.12, ax=ax)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel(ylabel)
        ax.set_title(title + "\nMean +/- 1 SD across seeds (not a confidence interval)")
    model = next(iter(models)).rstrip("/").split("/")[-1]
    fig.suptitle(f"{model} · temperature 0.7\nAverage across {len(seeds)} seeds ({', '.join(map(str, seeds))})", fontsize=16)
    gate_text = "; ".join(f"{condition} {int(gate_summary.loc[condition, 'sum'])}/{int(gate_summary.loc[condition, 'count'])}"
                          for condition in order)
    fig.text(.5, .025, "Bars: equal-weight seed means · whiskers: ±1 seed SD (not confidence intervals)\n"
             + "CDAT appropriateness gates passed: " + gate_text, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .91))
    for extension in ("png", "pdf"):
        fig.savefig(root / f"seed_average.{extension}", dpi=200)
    plt.close(fig)
    means.to_csv(root / "seed_means.csv", index=False)
    summary.to_csv(root / "seed_average.csv", index=False)
    (root / "seed_average_notes.md").write_text(
        "# Seed-average plots\n\n"
        "Each condition is averaged over items within each seed, then equally over seeds. "
        "Whiskers show one sample standard deviation across seed means. "
        "Colors, legends and capped error bars match the individual-seed plots. "
        "These are descriptive plots, not inferential confidence intervals.\n\n"
        "AUT uses Pairwise_Semantic_Distance_Proxy; DAT uses the official GloVe score; "
        "DRAT uses DRAT_Score. CDAT shows descriptive novelty for all five seeds before "
        "appropriateness gating, rather than a validated CDAT score. Gate pass counts: "
        + gate_text + ".\n\n"
        "Missing response scores are excluded from within-seed means, never replaced by zero. "
        "Scored_Responses and Responses in the CSVs record coverage. "
        "seed_means.csv contains each seed mean; seed_average.csv contains plotted means and SDs.\n")
    print(f"Saved {root}/seed_average.png and .pdf; {len(means)} seed/task/condition means")


def analyze_seed_means(root):
    """Compare five paired seed means and write corrected tests and their report."""
    frame = pd.read_csv(root / "seed_means.csv")
    if frame.Model.nunique() != 1:
        raise ValueError("Analyze one model per folder")
    rows, differences = [], []
    for (task, metric), group in frame.groupby(["Task", "Metric"], sort=False):
        table = group.pivot(index="Seed", columns="Condition", values="Mean").sort_index()
        if set(table.index) != {42, 43, 44, 45, 46} or not np.isfinite(table.to_numpy()).all():
            raise ValueError("Require five complete, finite seed means per condition")
        for control in ["Standard", "Effective", "Conventional", "Boring"]:
            diff = (table.Creative - table[control]).to_numpy()
            mean = float(diff.mean())
            sd = float(diff.std(ddof=1))
            se = sd / np.sqrt(len(diff))
            if sd == 0:
                statistic, pvalue = (0.0, 1.0) if mean == 0 else (np.copysign(np.inf, mean), 0.0)
            else:
                statistic, pvalue = ttest_rel(table.Creative, table[control], alternative="two-sided")
            margin = float(t.ppf(.975, len(diff)-1) * se)
            # Exhaust all 32 paired sign assignments, retaining ties in both tails.
            signs = np.array(list(product([-1, 1], repeat=len(diff))))
            null = (signs * diff).mean(axis=1)
            exact_p = float(np.mean(np.abs(null) >= abs(mean) - 1e-12))
            rows.append(dict(Model=frame.Model.iloc[0], Task=task, Metric=metric,
                             Treatment="Creative", Control=control, Pairs=len(diff), DF=len(diff)-1,
                             Treatment_Mean=table.Creative.mean(), Control_Mean=table[control].mean(),
                             Mean_Difference=mean, Difference_SD=sd, CI_Lower=mean-margin, CI_Upper=mean+margin,
                             T_Statistic=statistic, P_Two_Sided=pvalue, P_Exact_Sign_Flip=exact_p))
            differences.extend(dict(Task=task, Metric=metric, Control=control, Seed=seed,
                                    Difference=value) for seed, value in zip(table.index, diff))
    result = pd.DataFrame(rows)
    if len(result) != 16:
        raise ValueError("Expected four tasks and four controls")
    result["P_Holm"] = holm(result.P_Two_Sided)
    result["P_Exact_Holm"] = holm(result.P_Exact_Sign_Flip)
    result["Significant_Holm_005"] = result.P_Holm.lt(.05)
    result.to_csv(root / "seed_paired_tests.csv", index=False)
    pd.DataFrame(differences).to_csv(root / "seed_paired_differences.csv", index=False)
    report = ["# Paired tests of seed averages", "",
              "Two-sided paired t-tests compare Creative with each control, matching seeds 42–46 "
              "(five pairs; four degrees of freedom). Positive differences favor Creative. "
              "Holm correction covers all 16 task/control comparisons within this model.", "",
              "| Test | Control | Mean difference | 95% CI | Raw p | Holm p |", "|---|---|---:|---|---:|---:|"]
    for row in result.itertuples():
        report.append(f"| {row.Task.upper()} | {row.Control} | {row.Mean_Difference:.5g} | "
                      f"[{row.CI_Lower:.5g}, {row.CI_Upper:.5g}] | {row.P_Two_Sided:.4g} | {row.P_Holm:.4g} |")
    report.extend(["", "Intervals are ordinary, unadjusted 95% t intervals; significance uses Holm-adjusted p-values. "
                   "Tests describe variation across generation seeds on these fixed items, not generalization "
                   "to new items. With only five seeds, t-test assumptions are difficult to check; treat findings as exploratory.", "",
                   "The CSV also includes exact two-sided paired sign-flip p-values (all 32 assignments), "
                   "and their separate Holm corrections as a sensitivity check. The smallest possible "
                   "two-sided exact p-value with five pairs is 0.0625. Sign-flip inference assumes "
                   "exchangeability/symmetry of paired differences under the null.", "",
                   "CDAT uses descriptive novelty before appropriateness gating, matching the average plots. "
                   "Missing response scores were excluded when creating the seed means; see seed_means.csv "
                   "for score coverage. These tests do not change the plotted SD error bars.", ""])
    (root / "seed_paired_tests.md").write_text("\n".join(report))
    print(root)
    print(result[["Task", "Control", "Mean_Difference", "P_Two_Sided", "P_Holm", "Significant_Holm_005"]].to_string(index=False))


def analyze_root(root, seeds, draws=2000):
    # Validate the complete set before replacing any analysis artifacts.
    inputs = []
    for seed in seeds:
        for task, (metric, _, _) in TASKS.items():
            directory = root / f"seed_{seed}" / task
            frame = pd.read_csv(directory / f"{task}_responses.csv", keep_default_na=False)
            if frame.Response_ID.duplicated().any() or frame.Model.nunique() != 1:
                raise ValueError(f"Expected unique responses from one model: {directory}")
            for column in PLOT_METRICS[task].keys() & set(frame.columns):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            effects = [dict(Model=frame.Model.iloc[0], Task=frame.Task.iloc[0],
                            **paired_effect(frame, metric, "Creative", control, draws=draws))
                       for control in ("Standard", "Effective", "Conventional", "Boring")]
            inputs.append((directory, frame, effects))
    for directory, frame, effects in inputs:
        descriptive_plot(frame, {column: label for column, label in PLOT_METRICS[directory.name].items()
                                 if column in frame},
                         directory / f"{directory.name}_results.png")
        pd.DataFrame(effects).to_csv(directory / "creative_vs_controls.csv", index=False)
        print(f"Saved task plot and paired effects: {directory}", flush=True)
    plot_root(root, seeds)
    analyze_seed_means(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Scored analysis roots containing seed_42 through seed_46")
    parser.add_argument("--mode", choices=["all", "plot", "test"], default="all",
                        help="All outputs, seed-average plots only, or tests of an existing seed_means.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
                        help="Seed folders to average; all/test modes require seeds 42–46")
    parser.add_argument("--draws", type=int, default=2000, help="Bootstrap draws for within-seed paired effects")
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("Use at least 100 bootstrap draws")
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error("Choose at least two distinct seeds")
    if args.mode != "plot" and set(args.seeds) != {42, 43, 44, 45, 46}:
        parser.error("Seed tests require seeds 42–46; use --mode plot for other seed sets")
    for root in args.roots:
        if args.mode == "plot":
            plot_root(root, args.seeds)
        elif args.mode == "test":
            analyze_seed_means(root)
        else:
            analyze_root(root, args.seeds, args.draws)


if __name__ == "__main__":
    main()
