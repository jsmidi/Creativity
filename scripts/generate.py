"""Repeated, logged behavioral API or local generations with optional randomization."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid
from experiment import PROTOCOL_VERSION, CONDITION_TEXT, build_prompt, trial_schedule
from task_config import get_task_config


def query_model(client, prompt, model_id, provider="together", max_retries=5,
                temperature=0.7, max_tokens=800, seed=None, extra_body=None):
    kwargs = dict(model=model_id, messages=[{"role": "user", "content": prompt}],
                  temperature=temperature, max_tokens=max_tokens)
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
                        Returned_Model=result.model,
                        System_Fingerprint=getattr(result, "system_fingerprint", None),
                        Usage_JSON=json.dumps(result.usage.model_dump() if result.usage else {}))
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (429, 500, 502, 503, 504) and attempt + 1 < max_retries:
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            return dict(Response="", Status="error", Error=str(exc), Finish_Reason="",
                        Returned_Model="", System_Fingerprint="", Usage_JSON="{}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Llama-3.1-8B-Instruct")
    parser.add_argument("--task", choices=list(get_task_config()), default="Alternative Uses Task")
    parser.add_argument("--provider", choices=["together", "groq", "openrouter", "local"], default="local")
    parser.add_argument("--repeats", type=int, default=10, help="Pilot setting, not a power calculation")
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITION_TEXT), default=list(CONDITION_TEXT))
    parser.add_argument("--randomize", action="store_true", help="Shuffle requests instead of using the fixed condition order")
    parser.add_argument("--paraphrases", nargs="+", type=int, choices=range(3), default=[0])
    parser.add_argument("--items", nargs="+", help="Override pilot objects/items")
    parser.add_argument("--seed", type=int, default=42, help="Schedule randomization seed")
    parser.add_argument("--send-seed", action="store_true", help="Only if supported by provider; does not guarantee determinism")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--extra-body", type=json.loads, default={}, help="Explicit provider/reasoning settings as JSON")
    parser.add_argument("--split", choices=["pilot", "extraction", "validation", "test"], default="pilot")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.temperature < 0 or args.max_tokens < 1 or not isinstance(args.extra_body, dict):
        parser.error("Require temperature >= 0, max-tokens >= 1 and an extra-body JSON object.")
    config = get_task_config(args.task)
    try:
        trials = trial_schedule(args.items or config["items"], args.conditions,
                                args.repeats, args.paraphrases, args.seed, randomize=args.randomize)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{len(trials)} requests planned for {args.model}; protocol {PROTOCOL_VERSION}.")
    if args.dry_run:
        print(build_prompt(config["instruction"], trials[0]["Item"], trials[0]["Condition"], trials[0]["Paraphrase"]))
        return
    if args.provider == "local":
        if args.extra_body:
            parser.error("--extra-body applies only to API providers.")
        from interventions import load_engine
        print(f"Loading {args.model} locally...", flush=True)
        engine = load_engine(args.model)
    else:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(Path(__file__).with_name("ATT05522.env"))
        urls = {"together": "https://api.together.xyz/v1", "groq": "https://api.groq.com/openai/v1", "openrouter": "https://openrouter.ai/api/v1"}
        key = os.getenv(f"{args.provider.upper()}_API_KEY")
        if not key:
            parser.error(f"Missing {args.provider.upper()}_API_KEY")
        client = OpenAI(base_url=urls[args.provider], api_key=key)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = args.output_root / args.model.split("/")[-1].replace(":", "_")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"results_{args.task.lower().replace(' ', '_')}_{run_id}.csv"
    failures = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for order, trial in enumerate(trials):
            prompt = build_prompt(config["instruction"], trial["Item"], trial["Condition"], trial["Paraphrase"])
            row = dict(Protocol=PROTOCOL_VERSION, Run_ID=run_id, Response_ID=f"{run_id}:{order}",
                       Model=args.model, Provider=args.provider, Task=args.task, Split=args.split,
                       Instruction=config["instruction"], **trial, Request_Order=order,
                       Schedule_Seed=args.seed, Seed_Sent=args.send_seed or args.provider == "local",
                       Request_Ordering="randomized" if args.randomize else "fixed",
                       Temperature=args.temperature, Max_Tokens=args.max_tokens,
                       Extra_Body_JSON=json.dumps(args.extra_body, sort_keys=True),
                       Timestamp_UTC=datetime.now(timezone.utc).isoformat(), Prompt=prompt)
            if args.provider == "local":
                result = engine.generate(prompt, seed=trial["Generation_Seed"],
                                         temperature=args.temperature, max_tokens=args.max_tokens)
                row.update(result, Status="ok" if result["Response"] else "empty", Error="",
                           Top_P=1.0, Top_K=0, Model_Dtype=str(engine.model.dtype))
            else:
                row.update(query_model(client, prompt, args.model, args.provider,
                                   temperature=args.temperature, max_tokens=args.max_tokens,
                                   seed=trial["Generation_Seed"] if args.send_seed else None,
                                   extra_body=args.extra_body))
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(f"{order + 1}/{len(trials)} {trial['Condition']}: {row['Status']}")
            failures += row["Status"] != "ok"
            if order == 0 and row["Status"] == "error":
                print(row["Error"])
                break
    print(f"Saved {path}; {failures} failed/empty requests retained in log.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
