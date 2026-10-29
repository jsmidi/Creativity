"""Generate responses for selected local-model tasks and conditions."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import uuid

try:
    from .generation_engine import GenerationEngine
    from .pm_task_config import TASK_NAMES, CONDITIONS, get_task_config, build_prompt
except ImportError:  # Also support running this file directly.
    from scripts_new.generation.generation_engine import GenerationEngine
    from pm_task_config import TASK_NAMES, CONDITIONS, get_task_config, build_prompt


def stable_seed(*parts):
    """Return a repeatable 31-bit seed for the supplied trial identifiers."""
    value = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") % (2**31)


def parse_args():
    """Parse and validate runner options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a locally downloaded model")
    parser.add_argument("--tasks", nargs="+", required=True,
                        type=lambda value: TASK_NAMES.get(value.lower(), value),
                        choices=list(get_task_config()), help="Task names or short names such as aut and dat")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["standard"])
    parser.add_argument("--items", nargs="*", help="Use these items for each selected task")
    parser.add_argument("--repeats", type=int, default=1, help="Generations per task/item/condition")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for repeatable trial seeds")
    parser.add_argument("--output", type=Path, default=Path("responses.csv"))
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error("--repeats and --max-tokens must be at least 1.")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("Require --temperature >= 0 and --top-p in (0, 1].")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicates.")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("--conditions must not contain duplicates.")
    return args


def create_runs(args):
    """Build prompts and paired, stable seeds for each trial."""
    runs = []
    for task in args.tasks:
        config = get_task_config(task)
        items = args.items if args.items is not None else config["items"]
        for item in items:
            for repeat in range(args.repeats):
                seed = stable_seed(args.seed, task, item, repeat)
                for condition in args.conditions:
                    prompt = build_prompt(task, item, condition)
                    runs.append({"Task": task, "Condition": condition, "Item": item,
                                 "Repeat": repeat, "Seed": seed, "Prompt": prompt})
    if not runs:
        raise ValueError("No prompts were created; check the selected tasks and items.")
    return runs


def prepare_outputs(output_path):
    """Create the output directory and return a unique run ID and CSV path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    return run_id, output_path


def write_responses(args, runs, engine, run_id, output_path):
    """Generate each response, record failures and stream rows to CSV."""
    fields = ["RUN_ID", "RESPONSE_ID", "Model", "Seed", "Task", "Condition",
              "Item", "Repeat", "Prompt", "Response", "Generated_Tokens",
              "Prompt_Tokens", "Error"]
    failures = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, run in enumerate(runs, start=1):
            error = ""
            try:
                result = engine.generate(
                    run["Prompt"], seed=run["Seed"], max_new_tokens=args.max_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                )
            except Exception as exc:
                result = {"Response": "", "Generated_Tokens": "", "Prompt_Tokens": ""}
                error = str(exc)
            writer.writerow({
                "RUN_ID": run_id,
                "RESPONSE_ID": f"{run_id}:{index}",
                "Model": args.model,
                **run,
                "Response": result["Response"],
                "Generated_Tokens": result["Generated_Tokens"],
                "Prompt_Tokens": result["Prompt_Tokens"],
                "Error": error,
            })
            handle.flush()
            status = "error" if error else "ok" if result["Response"] else "empty"
            failures += status != "ok"
            print(f"{index}/{len(runs)} {run['Task']} · {run['Condition']} · {run['Item']!r}: {status}")
    return failures


def main():
    """Plan runs, load the model once, and write responses."""
    args = parse_args()
    try:
        runs = create_runs(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_id, output_path = prepare_outputs(args.output)
    print(f"{len(runs)} responses planned for {args.model}.", flush=True)
    engine = GenerationEngine(args.model)
    failures = write_responses(args, runs, engine, run_id, output_path)
    print(f"Saved {output_path}; {failures} generation errors logged.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
