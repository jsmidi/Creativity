"""Extract response-pooled and final-prompt-token attention-head activations."""
import argparse
import csv
import json
from pathlib import Path

import torch

try:
    from .activation_engine import ActivationEngine
except ImportError:  # Also support running this file directly.
    from activation_engine import ActivationEngine


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path, help="CSV written by generation/generate_responses.py")
    parser.add_argument("--model", help="Local model path; inferred from the CSV if it contains one model")
    parser.add_argument("--output", type=Path, required=True, help="Destination .pt activation artifact")
    parser.add_argument("--positions", nargs="+",
                        choices=("response", "response-predictor", "last-prompt"),
                        default=("response", "last-prompt"),
                        help=("Capture answer-token states, states that predict answer tokens, "
                              "and/or the final prompt token (default: answer states and final prompt)"))
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def token_ids_from_row(row):
    raw = row.get("Response_Token_IDs", "")
    if raw:
        try:
            return json.loads(raw), "saved_generated_token_ids"
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Response_Token_IDs JSON: {exc}") from exc
    return None, "reconstructed_from_response_text"


def prompt_token_ids_from_row(row):
    raw = row.get("Prompt_Token_IDs", "")
    if raw:
        try:
            return json.loads(raw), "saved_formatted_prompt_token_ids"
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Prompt_Token_IDs JSON: {exc}") from exc
    return None, "reconstructed_from_prompt_text"


def main():
    args = parse_args()
    rows = read_rows(args.responses)
    if not rows:
        raise SystemExit(f"No response rows found in {args.responses}")
    models = sorted({row.get("Model", "").strip() for row in rows if row.get("Model", "").strip()})
    model = args.model or (models[0] if len(models) == 1 else None)
    if not model:
        raise SystemExit("Specify --model when the CSV does not identify exactly one model.")
    output_rows, errors = [], []
    engine = ActivationEngine(model)
    for index, row in enumerate(rows, start=1):
        if row.get("Error"):
            continue
        response = row.get("Response", "")
        if not response.strip():
            continue
        try:
            token_ids, token_source = token_ids_from_row(row)
            prompt_token_ids, prompt_token_source = prompt_token_ids_from_row(row)
            activations = engine.prompt_response_head_means(
                row.get("Prompt", ""), response, response_token_ids=token_ids,
                positions=args.positions, prompt_token_ids=prompt_token_ids,
            )
            output_rows.append({
                "metadata": {
                    key: row.get(key, "") for key in
                    ("RUN_ID", "RESPONSE_ID", "Model", "Task", "Condition", "Item", "Repeat", "Seed")
                } | {"Response": response, "Token_Source": token_source,
                     "Prompt_Token_Source": prompt_token_source},
                "activations": activations,
            })
            print(f"{index}/{len(rows)} captured {row.get('Task')} · {row.get('Condition')} · {row.get('Item')!r}",
                  flush=True)
        except Exception as exc:
            errors.append({"RESPONSE_ID": row.get("RESPONSE_ID", ""), "Error": str(exc)})
            print(f"{index}/{len(rows)} activation error: {exc}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "prompt_response_head_means",
        "model": model,
        "positions": list(args.positions),
        "response_pooling": "mean of generated answer-token attention o_proj inputs; prompt and configured terminal EOS tokens excluded",
        "response_predictor_pooling": "mean of causal states immediately before each generated response token",
        "last_prompt": "attention o_proj input at the final formatted prompt token",
        "token_ids": "saved formatted prompt and generated response ids when present; otherwise reconstructed from saved text",
        "source_csv": str(args.responses),
        "rows": output_rows,
        "errors": errors,
    }, args.output)
    print(f"Saved {len(output_rows)} activation records and {len(errors)} errors to {args.output}")
    if not output_rows:
        raise SystemExit("No activations were captured.")


if __name__ == "__main__":
    main()
