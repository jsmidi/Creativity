import subprocess
import time
import sys
import argparse
from pathlib import Path

# 1. Map each model ID to its provider ('together', 'groq', 'openrouter')
MODELS_CONFIG = {
    # Together AI
    # "meta-llama/Llama-3.2-3B-Instruct-Turbo": "together",
    # "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "together",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    # "google/gemma-4-12B-it": "together",
    # "google/gemma-2-27b-it": "together",
    "google/gemma-4-31b-it": "together",
    # "Qwen/Qwen2.5-7B-Instruct": "together",
    # "Qwen/Qwen2.5-14B-Instruct": "together",
    # "Qwen/Qwen2.5-72B-Instruct": "together",
    
    # Groq
    # "mixtral-8x7b-32768": "groq",
    "qwen/qwen3.6-27b": "groq",
    "openai/gpt-oss-20b": "groq",
    
    # # OpenRouter
    # "z-ai/glm-5.2:free": "openrouter",
}

# 2. Define the tasks
TASKS_TO_RUN = [
    "Alternative Uses Task",
    "Divergent Association Task"
]

def main():
    parser = argparse.ArgumentParser(description="Run configured model/task pilots; extra arguments go to generate.py.")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    failed = []
    print("Starting creativity pilot pipeline...\n")
    
    for model, provider in MODELS_CONFIG.items():
        print(f"\n{'='*60}\nStarting test suite for Model: {model}\n{'='*60}")
        
        for task in TASKS_TO_RUN:
            print("*" * 60)
            print(f"Executing >> Provider: {provider.upper()} | Model: {model}")
            print(f"             Task: {task}")
            print("*" * 60)
            
            # Capture the result of the script execution
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name("generate.py")),
                "--model", model, 
                "--task", task,
                "--provider", provider,
                *(["--dry-run"] if args.dry_run else []), *extra
            ], cwd=Path(__file__).resolve().parents[1])
            
            # FAIL-FAST CATCHER: If generate.py exited with an error (like a 400 or 404), skip the rest of the tasks
            if result.returncode != 0:
                failed.append((model, task))
                print(f"\n[Model Skipped] '{model}' failed. Skipping remaining tasks for this model.")
                break  # This breaks out of the task loop and moves to the NEXT model
            
            # Provider-conscious sleep to respect rate limits
            sleep_time = 4 if provider == "openrouter" else 2
            if not args.dry_run:
                time.sleep(sleep_time)

    print(f"\nPipeline finished; {len(failed)} failed model/task runs.")
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
