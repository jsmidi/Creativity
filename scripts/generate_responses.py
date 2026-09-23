import os
import csv
import time
import random
import argparse
import functools
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# ---------- SDKs ----------
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai

# ---------- LOAD .ENV FILE ----------
# Force Python to load from your specific file in the same folder
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, "ATT05522.env")
load_dotenv(env_path)

# ---------- API KEYS ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY") # We now pull your Together key!
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY_LS")
GEMINI_API_KEY = os.getenv("GEMINI_CREA_API_KEY")

# ---------- CLIENTS ----------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# We use the standard OpenAI SDK, but point the URL strictly to Together AI's servers
together_client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
google_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------- WORKERS & LIMITS ----------
MAX_WORKERS = {
    "openai": 5,
    "together": 5, 
    "anthropic": 3,
    "google": 5,
}

TEMPERATURE = 0.7
MAX_TOKENS = 300

# ---------- MODEL REGISTRY ----------
MODEL_REGISTRY = {
    # OpenAI
    "gpt-4.1": {
        "provider": "openai",
        "call_fn": "call_openai",
        "supports_reasoning": False,
    },
    "gpt-5.2": {
        "provider": "openai",
        "call_fn": "call_openai",
        "supports_reasoning": True,
        "reasoning_effort": "low",
    },

    # Together AI (Direct Serverless Endpoints)
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {
        "provider": "together",
        "call_fn": "call_together",
    },
    "Qwen/Qwen2.5-7B-Instruct-Turbo": {
        "provider": "together",
        "call_fn": "call_together",
    },
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "provider": "together",
        "call_fn": "call_together",
    },

    # Anthropic
    "claude-3-5-sonnet-20250219": {
        "provider": "anthropic",
        "call_fn": "call_anthropic",
    },

    # Google
    "gemini-2.5-flash": {
        "provider": "google",
        "call_fn": "call_gemini",
    },
}

# ---------- RETRY WRAPPER ----------
def with_retries(fn, max_retries=5, base_delay=1.0):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[Error] Failed after {max_retries} attempts: {e}")
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
    return wrapper

# ---------- PROMPT BUILDER ----------
def build_prompt_payload(problem_text, condition):
    if condition == "Creative":
        sys_inst = "You are a creative participant. Be creative, think divergently, and provide highly original ideas."
        user_prompt = f"Provide a highly creative and original response to the following task:\n\nTask: {problem_text}\n\nResponse:"
    elif condition == "Standard":
        sys_inst = "You are an effective assistant. Be effective and provide the most correct, reasonable completion."
        user_prompt = f"Provide a reasonable, effective response to the following task:\n\nTask: {problem_text}\n\nResponse:"
    elif condition == "Baseline":
        sys_inst = "You are a helpful assistant."
        user_prompt = f"Task: {problem_text}\n\nResponse:"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return sys_inst, user_prompt

# ---------- MODEL API HANDLERS ----------
@with_retries
def call_openai(sys_inst, prompt, model, supports_reasoning=False, reasoning_effort=None):
    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "user", "content": prompt},
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
    }
    if supports_reasoning:
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = TEMPERATURE

    response = openai_client.chat.completions.create(**kwargs)
    usage = response.usage

    return {
        "text": response.choices[0].message.content.strip(),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }

@with_retries
def call_together(sys_inst, prompt, model):
    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "user", "content": prompt},
    ]
    # Together AI natively supports the OpenAI Python library structure
    response = together_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    usage = response.usage

    return {
        "text": response.choices[0].message.content.strip(),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }

@with_retries
def call_anthropic(sys_inst, prompt, model):
    response = anthropic_client.messages.create(
        model=model,
        system=sys_inst,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = getattr(response, "usage", None)
    return {
        "text": response.content[0].text.strip(),
        "prompt_tokens": getattr(usage, "input_tokens", None),
        "completion_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": (getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)) if usage else None,
    }

@with_retries
def call_gemini(sys_inst, prompt, model):
    response = google_client.models.generate_content(
        model=model,
        contents={'text': prompt},
        config={
            'temperature': TEMPERATURE,
            'max_output_tokens': MAX_TOKENS,
            'system_instruction': sys_inst,
        },
    )
    usage = getattr(response, "usage_metadata", None)
    return {
        "text": response.text.strip(),
        "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "completion_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
        "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
    }

CALL_FN_MAP = {
    "call_openai": call_openai,
    "call_together": call_together,
    "call_anthropic": call_anthropic,
    "call_gemini": call_gemini,
}

# ---------- WORKER FUNCTION ----------
def process_task(task_item, condition, model_name):
    cfg = MODEL_REGISTRY[model_name]
    call_fn = CALL_FN_MAP[cfg["call_fn"]]
    
    sys_inst, prompt = build_prompt_payload(task_item["problem"], condition)

    t0 = time.time()
    if cfg["provider"] == "openai":
        result = call_fn(
            sys_inst,
            prompt,
            model_name,
            supports_reasoning=cfg.get("supports_reasoning", False),
            reasoning_effort=cfg.get("reasoning_effort"),
        )
    else:
        result = call_fn(sys_inst, prompt, model_name)
    rt = time.time() - t0

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "provider": cfg["provider"],
        "model": model_name,
        "task_domain": task_item.get("task_domain", "general"),
        "item_id": task_item.get("item_id", ""),
        "condition": condition,
        "prompt": prompt,
        "response": result["text"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "rt": rt,
    }

# ---------- MAIN EXECUTION ----------
def main():
    parser = argparse.ArgumentParser(description="Run AGC-Bench Divergent Thinking Evaluation")
    parser.add_argument("--items_file", default="agc_bench_items.csv", help="CSV with columns: item_id, task_domain, problem")
    parser.add_argument("--models", default="all", help="Provider name or 'all'")
    parser.add_argument("--conditions", nargs="+", default=["Baseline", "Standard", "Creative"])
    parser.add_argument("--output_dir", default="results_divergent_thinking")
    parser.add_argument("--max_items", type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(args.items_file)
    if args.max_items is not None:
        df = df.head(args.max_items)

    selected_models = (
        list(MODEL_REGISTRY.keys())
        if args.models == "all"
        else [m for m, cfg in MODEL_REGISTRY.items() if cfg["provider"] == args.models or m == args.models]
    )

    fieldnames = [
        "timestamp", "provider", "model", "task_domain", "item_id",
        "condition", "prompt", "response",
        "prompt_tokens", "completion_tokens", "total_tokens", "rt"
    ]

    for model_name in selected_models:
        provider = MODEL_REGISTRY[model_name]["provider"]
        output_file = os.path.join(args.output_dir, f"results_{model_name.replace('/', '_')}_{timestamp}.csv")
        
        print("==========================================")
        print(f"Running Model: {model_name} ({provider})")
        print(f"Output: {output_file}")
        print("==========================================")

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            with ThreadPoolExecutor(max_workers=MAX_WORKERS[provider]) as executor:
                futures = []
                for _, row in df.iterrows():
                    task_item = row.to_dict()
                    for cond in args.conditions:
                        futures.append(
                            executor.submit(process_task, task_item, cond, model_name)
                        )

                for future in as_completed(futures):
                    res = future.result()
                    writer.writerow(res)
                    f.flush()

    print("\nAll generation tasks completed successfully.")

if __name__ == "__main__":
    main()