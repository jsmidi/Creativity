"""Plot scored task responses or paired intervention effects."""
import argparse
from pathlib import Path
import pandas as pd
from scoring_common import descriptive_plot


def effect_plot(effects, output, title):
    """Plot intervention differences and unadjusted bootstrap intervals; greedy runs show points only."""
    import matplotlib.pyplot as plt
    import numpy as np
    valid = effects[effects.Analysis_Status.eq('ok')].copy() if 'Analysis_Status' in effects else effects.copy()
    labels = [f'{r.Treatment} minus {r.Control}' for r in valid.itertuples()]
    y = np.arange(len(valid))
    fig, ax = plt.subplots(figsize=(11, max(5, len(valid)*.35)))
    for position, row in zip(y, valid.itertuples()):
        ax.plot([row.CI_Lower, row.CI_Upper], [position, position], color='C0')
        ax.plot(row.Mean_Difference, position, 'o', color='C0')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel('Paired score difference; exploratory 95% bootstrap interval'
                  if valid.CI_Lower.notna().any() else 'Paired score difference; greedy decoding, descriptive only')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Scored responses or effects CSV")
    parser.add_argument("--metrics", nargs="+", help="Score columns for response plots")
    parser.add_argument("--effects", action="store_true", help="Plot paired effect estimates")
    parser.add_argument("--title", default="Paired intervention effects")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if not args.effects and not args.metrics:
        parser.error("Choose --metrics COLUMN ... or --effects")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.effects:
        effect_plot(frame, args.output, args.title)
    else:
        descriptive_plot(frame, {column: column for column in args.metrics}, args.output)


if __name__ == "__main__":
    main()
