"""Run the configured local or API AUT/DAT pilot using generate.py.

Additional CLI arguments are forwarded to each generation run.
"""
import subprocess
import time
import sys
import argparse
from pathlib import Path

MODELS_CONFIG = {"models/Llama-3.1-8B-Instruct": "local"}

API_MODELS_CONFIG = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "google/gemma-4-31b-it": "together",

    "qwen/qwen3.6-27b": "groq",
    "openai/gpt-oss-20b": "groq",

}

TASKS_TO_RUN = [
    "Alternative Uses Task",
    "Divergent Association Task"
]

def run_model(model, provider, args, extra):
    """Run each pilot task for one model; stop that model on the first failed command."""
    failed = []
    print(f"\n{'='*60}\nStarting test suite for Model: {model}\n{'='*60}")

    for task in TASKS_TO_RUN:
        print("*" * 60)
        print(f"Executing >> Provider: {provider.upper()} | Model: {model}")
        print(f"             Task: {task}")
        print("*" * 60)

        result = subprocess.run([
            sys.executable, str(Path(__file__).with_name("generate.py")),
            "--model", model,
            "--task", task,
            "--provider", provider,
            *(["--dry-run"] if args.dry_run else []), *extra
        ], cwd=Path(__file__).resolve().parents[1])

        if result.returncode != 0:
            failed.append((model, task))
            print(f"\n[Model Skipped] '{model}' failed. Skipping remaining tasks for this model.")
            break  # This breaks out of the task loop and moves to the NEXT model

        sleep_time = 4 if provider == "openrouter" else 2
        if not args.dry_run and provider != "local":
            time.sleep(sleep_time)

    return failed


def main():
    """Run the configured model/task pilots, forwarding generation arguments and failing per model."""
    parser = argparse.ArgumentParser(description="Run configured model/task pilots; extra arguments go to generate.py.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-models", action="store_true", help="Use the API model registry instead of the default local Llama")
    args, extra = parser.parse_known_args()
    failed = []
    print("Starting creativity pilot pipeline...\n")

    models = API_MODELS_CONFIG if args.api_models else MODELS_CONFIG
    for model, provider in models.items():
        failed.extend(run_model(model, provider, args, extra))

    print(f"\nPipeline finished; {len(failed)} failed model/task runs.")
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
