"""Plot scored psychometric runs, always including averages across all seeds.

Pass either model/protocol folders containing seed_* directories or a parent
containing several such folders. Each model/protocol is plotted separately.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from analyze_psychometric_results import PLOT_METRICS, TASKS, plot_root
from scoring_common import descriptive_plot


def seed_numbers(root):
    """Discover every numeric seed directory without assuming specific seeds."""
    return sorted(int(path.name[5:]) for path in root.glob("seed_*")
                  if path.is_dir() and path.name[5:].isdigit())


def discover_roots(inputs):
    """Find model/protocol roots and reject incomplete seed task collections."""
    roots = set()
    for supplied in inputs:
        supplied = supplied.resolve()
        if not supplied.is_dir():
            raise ValueError(f"Analysis directory does not exist: {supplied}")
        candidates = [supplied, *(path.parent for path in supplied.rglob("seed_*")
                                  if path.is_dir() and path.name[5:].isdigit())]
        found = {root for root in candidates if seed_numbers(root)}
        if not found:
            raise ValueError(f"No seed directories found under {supplied}")
        roots.update(found)
    for root in sorted(roots):
        seeds = seed_numbers(root)
        if len(seeds) < 2:
            raise ValueError(f"Need at least two seeds to plot their variation: {root}")
        for seed in seeds:
            for task in TASKS:
                required = [root / f"seed_{seed}" / task / f"{task}_responses.csv"]
                if task == "cdat":
                    required.append(root / f"seed_{seed}" / task / "cdat_gates.csv")
                for path in required:
                    if not path.is_file():
                        raise ValueError(f"Incomplete scored seed; missing {path}")
    return sorted(roots)


def plot_results(root, averages_only=False):
    """Always write an equal-item, equal-seed summary for this model/protocol."""
    seeds = seed_numbers(root)
    # This validates all seeds and produces PNG/PDF plots and summary CSVs.
    plot_root(root, seeds)
    if not averages_only:
        for seed in seeds:
            for task in TASKS:
                directory = root / f"seed_{seed}" / task
                frame = pd.read_csv(directory / f"{task}_responses.csv")
                metrics = {column: label for column, label in PLOT_METRICS[task].items()
                           if column in frame}
                for column in metrics:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
                descriptive_plot(frame, metrics, directory / f"{task}_results.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path,
                        help="Scored model/protocol folders, or their shared parent directory")
    parser.add_argument("--averages-only", action="store_true",
                        help="Keep existing individual-seed plots; regenerate seed averages only")
    args = parser.parse_args()
    try:
        roots = discover_roots(args.roots)
    except ValueError as exc:
        parser.error(str(exc))
    for root in roots:
        plot_results(root, averages_only=args.averages_only)


if __name__ == "__main__":
    main()
