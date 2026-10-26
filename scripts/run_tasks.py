"""Generate paired task responses, loading one backend for a single task or battery."""
import argparse
from copy import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid
from experiment import CONDITION_TEXT, build_prompt as build_legacy_prompt, trial_schedule
from task_config import get_task_config, TASK_NAMES, build_prompt
from sampling import sampling_kwargs


def normalize_condition(value):
    """Accept attachment names while preserving existing CSV condition labels."""
    return "Standard" if value.lower() in ("neutral", "standard") else value.capitalize()


def trial_prompt(args, config, trial):
    """Use the selected protocol consistently for previews and model requests."""
    if getattr(args, "prompt_protocol", "task-specific") == "legacy":
        return build_legacy_prompt(config["instruction"], trial["Item"], trial["Condition"], trial["Paraphrase"])
    if trial["Paraphrase"] != 0:
        raise ValueError("Task-specific prompts only support paraphrase 0.")
    return build_prompt(args.task, trial["Item"], trial["Condition"])


def query_model(client, prompt, model_id, provider="together", max_retries=5,
                temperature=0.7, max_tokens=800, seed=None, extra_body=None, top_p=1.0):
    """Request one API completion, retry transient server errors, and return logged status fields."""
    kwargs = dict(model=model_id, messages=[{"role": "user", "content": prompt}],
                  temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    if seed is not None:
        kwargs["seed"] = seed
    if extra_body:
        kwargs["extra_body"] = extra_body
    for attempt in range(max_retries):
        try:
            result = client.chat.completions.create(**kwargs)
            choice = result.choices[0]
            content = choice.message.content
            return dict(Response=(content or "").strip(),
                        Status="ok" if content and content.strip() else "empty",
                        Error="", Finish_Reason=choice.finish_reason,
                        Generated_Tokens=getattr(result.usage, "completion_tokens", "") if result.usage else "",
                        Prompt_Tokens=getattr(result.usage, "prompt_tokens", "") if result.usage else "")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (429, 500, 502, 503, 504) and attempt + 1 < max_retries:
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            return dict(Response="", Status="error", Error=str(exc), Finish_Reason="",
                        Generated_Tokens="", Prompt_Tokens="")


def parse_args():
    """Parse task selection and validate decoding options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Llama-3.1-8B-Instruct")
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument("--task", type=lambda name: TASK_NAMES.get(name.lower(), name), choices=list(get_task_config()), default="Alternative Uses Task")
    tasks.add_argument("--tasks", nargs="+", type=lambda name: TASK_NAMES.get(name.lower(), name),
                       choices=list(get_task_config()),
                       help="Run multiple tasks with one backend; save each in its own task folder")
    parser.add_argument("--provider", choices=["together", "groq", "openrouter", "local"], default="local")
    parser.add_argument("--repeats", type=int, default=10, help="Pilot setting, not a power calculation")
    parser.add_argument("--conditions", nargs="+", type=normalize_condition, choices=list(CONDITION_TEXT), default=list(CONDITION_TEXT))
    parser.add_argument("--randomize", action="store_true", help="Shuffle requests instead of using the fixed condition order")
    parser.add_argument("--prompt-protocol", choices=["task-specific", "legacy"], default="task-specific",
                        help="Task-specific instructions (default), or the previous generic condition suffixes")
    parser.add_argument("--paraphrases", nargs="+", type=int, choices=range(3), default=[0])
    parser.add_argument("--items", nargs="+", help="Override pilot objects/items")
    parser.add_argument("--seed", type=int, default=42, help="Schedule randomization seed")
    parser.add_argument("--send-seed", action="store_true", help="Only if supported by provider; does not guarantee determinism")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="Local only; 0 disables top-k truncation")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low",
                        help="Local GPT-OSS reasoning effort (recorded in its output)")
    parser.add_argument("--extra-body", type=json.loads, default={}, help="Explicit provider/reasoning settings as JSON")
    parser.add_argument("--split", choices=["pilot", "extraction", "validation", "test"], default="pilot")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.temperature < 0 or args.max_tokens < 1 or not isinstance(args.extra_body, dict):
        parser.error("Require temperature >= 0, max-tokens >= 1 and an extra-body JSON object.")
    try:
        sampling_kwargs(args.temperature, args.top_p, args.top_k)
    except ValueError as exc:
        parser.error(str(exc))
    if args.provider != "local" and args.top_k != 0:
        parser.error("--top-k is local-only; API providers have no portable top-k parameter.")
    if set(args.extra_body) & {"temperature", "top_p", "top_k", "seed", "max_tokens"}:
        parser.error("Sampling overrides in --extra-body are not allowed; use explicit CLI settings.")
    if args.prompt_protocol == "task-specific" and args.paraphrases != [0]:
        parser.error("Task-specific prompts have one wording; use --paraphrases 0, or --prompt-protocol legacy for older paraphrases.")
    if args.tasks and len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicate tasks.")
    return args


def plan_tasks(args):
    """Validate every task schedule before loading a model or creating files."""
    plans = []
    for task in args.tasks or [args.task]:
        task_args = copy(args)
        task_args.task = task
        if args.tasks:
            key = next((key for key, name in TASK_NAMES.items() if name == task),
                       task.lower().replace(" ", "_"))
            task_args.output_root = args.output_root / key
        config = get_task_config(task)
        trials = trial_schedule(args.items or config["items"], args.conditions,
                                args.repeats, args.paraphrases, args.seed, randomize=args.randomize)
        plans.append((task_args, config, trials))
    return plans


def load_backend(args):
    """Load one local engine or configure the requested API client."""
    engine, client = None, None
    if args.provider == "local":
        if args.extra_body:
            raise ValueError("--extra-body applies only to API providers.")
        from local_generation import load_behavioral_engine
        print(f"Loading {args.model} locally...", flush=True)
        engine = load_behavioral_engine(args.model, args.reasoning_effort)
    else:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(Path(__file__).with_name("ATT05522.env"))
        urls = {"together": "https://api.together.xyz/v1", "groq": "https://api.groq.com/openai/v1", "openrouter": "https://openrouter.ai/api/v1"}
        key = os.getenv(f"{args.provider.upper()}_API_KEY")
        if not key:
            raise ValueError(f"Missing {args.provider.upper()}_API_KEY")
        client = OpenAI(base_url=urls[args.provider], api_key=key)
    return engine, client


def prepare_output(args):
    """Choose a unique CSV path; settings are saved in each row, without a manifest."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = args.output_root / args.model.split("/")[-1].replace(":", "_")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"results_{args.task.lower().replace(' ', '_')}_{run_id}.csv"
    return run_id, path


def response_row(args, config, trial, run_id, order, engine, client):
    """Generate one response and save its prompt, settings and token counts."""
    prompt = trial_prompt(args, config, trial)
    timestamp = datetime.now(timezone.utc).isoformat()
    if args.provider == "local":
        result = engine.generate(prompt, seed=trial["Generation_Seed"],
                                 temperature=args.temperature, max_tokens=args.max_tokens,
                                 top_p=args.top_p, top_k=args.top_k)
        error = ""
        model_dtype = str(engine.model.dtype)
    else:
        result = query_model(client, prompt, args.model, args.provider,
                             temperature=args.temperature, max_tokens=args.max_tokens,
                             seed=trial["Generation_Seed"] if args.send_seed else None,
                             extra_body=args.extra_body, top_p=args.top_p)
        error = result["Error"]
        model_dtype = "unknown"  # API providers do not report the weight dtype.
    return dict(Run_ID=run_id, Response_ID=f"{run_id}:{order}", Model=args.model,
                Task=args.task, Instruction=config["instruction"], Item=trial["Item"],
                Condition=trial["Condition"], Generation_Seed=trial["Generation_Seed"],
                Temperature=args.temperature, Max_Tokens=args.max_tokens,
                Top_P=args.top_p, Top_K=args.top_k if args.provider == "local" else "unknown",
                Timestamp_UTC=timestamp, Prompt=prompt, Response=result["Response"],
                Generated_Tokens=result.get("Generated_Tokens", ""),
                Prompt_Tokens=result.get("Prompt_Tokens", ""), Error=error, Model_Dtype=model_dtype)


def write_responses(args, config, trials, run_id, path, engine, client):
    """Stream scheduled responses to CSV; retain failures and flush after every request."""
    failures = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for order, trial in enumerate(trials):
            row = response_row(args, config, trial, run_id, order, engine, client)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            status = "error" if row["Error"] else "ok" if row["Response"].strip() else "empty"
            print(f"{order + 1}/{len(trials)} {trial['Condition']}: {status}")
            failures += status != "ok"
            if order == 0 and status == "error":
                print(row["Error"])
                break
    print(f"Saved {path}; {failures} failed/empty requests retained in log.")
    if failures:
        raise SystemExit(1)


def main():
    """Plan all tasks, load one backend, then stream a separate CSV for each task."""
    args = parse_args()
    try:
        plans = plan_tasks(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for task_args, config, trials in plans:
        print(f"{len(trials)} requests planned for {args.model}: {task_args.task}.")
        if args.dry_run:
            print(trial_prompt(task_args, config, trials[0]))
    if args.dry_run:
        return

    # Reserve the battery root so resubmissions cannot mix separate experiments.
    if args.tasks:
        args.output_root.mkdir(parents=True, exist_ok=False)
    try:
        engine, client = load_backend(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for task_args, config, trials in plans:
        run_id, path = prepare_output(task_args)
        write_responses(task_args, config, trials, run_id, path, engine, client)


if __name__ == "__main__":
    main()
