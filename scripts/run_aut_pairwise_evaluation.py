"""Evaluate one saved AUT run from an explicit plan, retaining separate run outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from analyze_effects import paired_effect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not 0 <= args.index < len(plan["runs"]):
        parser.error("Index outside the prepared run list")
    run = plan["runs"][args.index]
    output = Path(run["output"])
    # An atomic directory reservation prevents accidental reruns from overwriting scores.
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_aut.py")),
                    "--inputs", run["input"], "--output-dir", str(output)], check=True)
    frame = pd.read_csv(output / "aut_responses.csv")
    metric = "Pairwise_Semantic_Distance_Proxy"
    contrasts = []
    for control in ("Standard", "Effective", "Conventional", "Boring"):
        if control in set(frame.Condition):
            subset = frame[frame.Condition.isin(["Creative", control])]
            contrasts.append(paired_effect(subset, metric, "Creative", control))
    pd.DataFrame(contrasts).to_csv(output / "creative_vs_controls.csv", index=False)
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
