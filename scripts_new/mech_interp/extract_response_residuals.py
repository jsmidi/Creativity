"""Extract response-pooled and final-prompt-token residual stream activations."""
import argparse
import csv
import json
from pathlib import Path

import torch

try:
    from .activation_engine import ActivationEngine
    from .activation_grouping import activation_metadata_from_row
except ImportError:  # Also support running this file directly.
    from activation_engine import ActivationEngine
    from activation_grouping import activation_metadata_from_row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path, help="CSV written by generation/generate_responses.py")
    parser.add_argument("--model", help="Local model path; inferred when the CSV identifies one model")
    parser.add_argument("--output", type=Path, required=True, help="Destination .pt residual artifact")
    parser.add_argument("--positions", nargs="+",
                        choices=("response", "response-predictor", "last-prompt"),
                        default=("response", "last-prompt"),
                        help=("Capture answer-token states, states that predict answer tokens, "
                              "and/or the final prompt token"))
    return parser.parse_args()


def main():
    args = parse_args()
    with args.responses.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No response rows found in {args.responses}")
    models = sorted({row.get("Model", "").strip() for row in rows if row.get("Model", "").strip()})
    model = args.model or (models[0] if len(models) == 1 else None)
    if not model:
        raise SystemExit("Specify --model when the CSV does not identify exactly one model.")

    engine = ActivationEngine(model)
    output_rows, errors = [], []
    for index, row in enumerate(rows, start=1):
        if row.get("Error"):
            continue
        response = row.get("Response", "")
        try:
            raw_ids = row.get("Response_Token_IDs", "")
            token_ids = json.loads(raw_ids) if raw_ids else None
            raw_prompt_ids = row.get("Prompt_Token_IDs", "")
            prompt_token_ids = json.loads(raw_prompt_ids) if raw_prompt_ids else None
            token_source = "saved_generated_token_ids" if token_ids is not None else "reconstructed_from_response_text"
            prompt_token_source = ("saved_formatted_prompt_token_ids" if prompt_token_ids is not None
                                   else "reconstructed_from_prompt_text")
            activations = engine.prompt_response_residual_means(
                row.get("Prompt", ""), response, response_token_ids=token_ids,
                positions=args.positions, prompt_token_ids=prompt_token_ids,
            )
            output_rows.append({
                "metadata": activation_metadata_from_row(row) | {
                    "Response": response, "Token_Source": token_source,
                    "Prompt_Token_Source": prompt_token_source},
                "activations": activations,
            })
            print(f"{index}/{len(rows)} captured {row.get('Task')} · {row.get('Condition')} · {row.get('Item')!r}",
                  flush=True)
        except Exception as exc:
            errors.append({"RESPONSE_ID": row.get("RESPONSE_ID", ""), "Error": str(exc)})
            print(f"{index}/{len(rows)} residual extraction error: {exc}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "prompt_response_residual_means",
        "model": model,
        "positions": list(args.positions),
        "residual_capture": "post-block residual stream, before final model norm",
        "response_pooling": "mean of generated answer-token states; prompt and configured terminal EOS tokens excluded",
        "response_predictor_pooling": "mean of causal states immediately before each generated response token",
        "last_prompt": "residual stream at the final formatted prompt token",
        "source_csv": str(args.responses),
        "rows": output_rows,
        "errors": errors,
    }, args.output)
    print(f"Saved {len(output_rows)} residual records and {len(errors)} errors to {args.output}")
    if not output_rows:
        raise SystemExit("No residual activations were captured.")


if __name__ == "__main__":
    main()
