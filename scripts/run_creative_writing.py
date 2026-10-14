"""Generate paired short stories, then extract response-pooled residual vectors.

Generation and extraction are separate so stories can be reviewed without
regenerating them. Exact token IDs, prompts and seeds are saved for replay.
No GPU inference or downloads occur on import or in a generation dry run.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

from creative_writing import DEFAULT_CUES, PROMPT_VERSION, TASK, load_cues, story_prompt
from experiment import PROTOCOL_VERSION, stable_seed


def writing_trials(cues, repeats, seed):
    """Shuffle complete paired blocks, sharing the generation seed within each pair."""
    trials = []
    for cue in cues:
        for repeat in range(repeats):
            block = f"{cue['id']}|r{repeat}"
            for condition in ("Standard", "Creative"):
                trials.append((cue, repeat, condition, block, stable_seed(seed, TASK, block)))
    random.Random(seed).shuffle(trials)
    return trials


def write_stories(engine, args, trials, run_id):
    """Stream readable CSV and exact-token JSONL; retain incomplete/empty stories."""
    json_path = args.output_dir / "responses.jsonl"
    csv_path = args.output_dir / "responses.csv"
    with json_path.open("x", encoding="utf-8") as raw, csv_path.open("x", encoding="utf-8", newline="") as readable:
        writer = None
        for order, (cue, repeat, condition, block, seed) in enumerate(trials):
            prompt = story_prompt(cue, condition)
            result = engine.generate(prompt, seed=seed, temperature=args.temperature,
                                     max_tokens=args.max_tokens, top_p=args.top_p,
                                     top_k=args.top_k, return_token_ids=True,
                                     repetition_penalty=getattr(args, "repetition_penalty", 1.0))
            row = dict(Protocol=PROTOCOL_VERSION, Prompt_Version=PROMPT_VERSION,
                       Run_ID=run_id, Response_ID=f"{run_id}:{order}", Model=args.model,
                       Task=TASK, Cue_ID=cue["id"], Item=", ".join(cue["items"]),
                       Cue_Words=cue["items"], Boring_Theme=cue["boring_theme"],
                       Split=cue["split"], Condition=condition, Prompt=prompt,
                       Paraphrase=0, Repeat=repeat, Block_ID=block, Generation_Seed=seed,
                       Request_Order=order, Temperature=args.temperature,
                       Top_P=args.top_p, Top_K=args.top_k, Max_Tokens=args.max_tokens,
                       Repetition_Penalty=getattr(args, "repetition_penalty", 1.0),
                       Status="ok" if result["Response"] else "empty", **result)
            raw.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw.flush()
            summary = {key: row[key] for key in (
                "Response_ID", "Cue_ID", "Item", "Split", "Condition", "Repeat",
                "Generation_Seed", "Response", "Status", "Finish_Reason") }
            if writer is None:
                writer = csv.DictWriter(readable, fieldnames=list(summary))
                writer.writeheader()
            writer.writerow(summary)
            readable.flush()
            print(f"{order + 1}/{len(trials)} {cue['id']} {condition}: {row['Finish_Reason']}", flush=True)


def generate(args):
    """Generate only the requested split and save a complete replay manifest."""
    document = load_cues(args.cues)
    cues = [cue for cue in document["data"] if cue["split"] == args.split]
    if not cues:
        raise ValueError(f"No cues in split {args.split}.")
    trials = writing_trials(cues, args.repeats, args.seed)
    print(f"{len(cues)} cue triplets; {len(trials)} paired-condition responses; split={args.split}.")
    if args.dry_run:
        print(json.dumps({condition: story_prompt(cues[0], condition)
                          for condition in ("Standard", "Creative")}, indent=2))
        return
    from run_residual import provenance
    from interventions import load_engine
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.cues, args.output_dir / "cues.json")
    engine = load_engine(args.model, args.revision)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = dict(**provenance(engine, args), protocol=PROTOCOL_VERSION,
                    prompt_version=PROMPT_VERSION, task=TASK, run_id=run_id,
                    cue_metadata=document.get("metadata", {}), split=args.split,
                    cue_ids=[cue["id"] for cue in cues], repeats=args.repeats,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    repetition_penalty=getattr(args, "repetition_penalty", 1.0),
                    max_tokens=args.max_tokens, seed=args.seed,
                    planned_responses=len(trials), completed=False)
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_stories(engine, args, trials, run_id)
    manifest.update(completed=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_discovery(run_dir, allow_truncated=False):
    """Require a complete discovery run with every planned matched pair.

    Empty responses and incomplete blocks fail closed. Length-capped responses
    require explicit inclusion; content quality is not silently used for selection.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if (not manifest.get("completed") or manifest["split"] != "discovery" or
            manifest["prompt_version"] != PROMPT_VERSION or manifest["protocol"] != PROTOCOL_VERSION):
        raise ValueError("Extraction requires a completed compatible discovery run.")
    cues = {cue["id"]: cue for cue in load_cues(run_dir / "cues.json")["data"]
            if cue["split"] == "discovery"}
    if set(cues) != set(manifest["cue_ids"]):
        raise ValueError("Discovery cues differ from the planned run.")
    rows = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest["planned_responses"]:
        raise ValueError("Incomplete response file.")
    pairs = {}
    for row in rows:
        condition = row["Condition"]
        cue = cues.get(row["Cue_ID"])
        if (cue is None or condition not in ("Creative", "Standard") or
                row["Split"] != "discovery" or row["Model"] != manifest["model"] or
                row["Prompt"] != story_prompt(cue, condition) or row["Status"] != "ok"):
            raise ValueError("Unexpected or empty response in discovery data.")
        if row["Finish_Reason"] == "length" and not allow_truncated:
            raise ValueError("Truncated story: increase max-tokens or explicitly use --allow-truncated.")
        for name in ("Prompt_Token_IDs", "Completion_Token_IDs"):
            ids = row[name]
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("Exact nonempty integer token IDs are required.")
        key = (row["Cue_ID"], row["Repeat"])
        pair = pairs.setdefault(key, {})
        if condition in pair:
            raise ValueError("Duplicate response in a paired block.")
        pair[condition] = row
    expected = {(cue_id, repeat) for cue_id in cues for repeat in range(manifest["repeats"])}
    if set(pairs) != expected or any(set(pair) != {"Creative", "Standard"} for pair in pairs.values()):
        raise ValueError("Missing or unexpected paired blocks.")
    for pair in pairs.values():
        if pair["Creative"]["Generation_Seed"] != pair["Standard"]["Generation_Seed"]:
            raise ValueError("Paired conditions have different seeds.")
    return manifest, pairs


def pool_stories(engine, pairs, layers):
    """Replay exact token sequences once per story; retain one mean per layer."""
    pooled = {}
    for index, (key, pair) in enumerate(sorted(pairs.items())):
        pooled[key] = {}
        for condition, row in pair.items():
            actual_prompt = engine.inputs(row["Prompt"])["input_ids"][0].tolist()
            if actual_prompt != row["Prompt_Token_IDs"]:
                raise ValueError("Saved prompt IDs do not match this tokenizer/template.")
            completion = row["Completion_Token_IDs"]
            if engine.tokenizer.decode(completion, skip_special_tokens=True).strip() != row["Response"]:
                raise ValueError("Saved completion IDs do not match the response text.")
            pooled[key][condition] = engine.capture_response_means(actual_prompt, completion, layers)
        print(f"Pooled pair {index + 1}/{len(pairs)}", flush=True)
    return pooled


def aggregate_layer(pooled, layer):
    """Average responses within cues, then cues equally; return contrast and center."""
    import torch
    by_cue = {}
    for (cue_id, _repeat), pair in pooled.items():
        by_cue.setdefault(cue_id, []).append(pair)
    positives, negatives = [], []
    for pairs in by_cue.values():
        positives.append(torch.stack([pair["Creative"][layer] for pair in pairs]).mean(0))
        negatives.append(torch.stack([pair["Standard"][layer] for pair in pairs]).mean(0))
    positive, center = torch.stack(positives).mean(0), torch.stack(negatives).mean(0)
    vector = positive - center
    if not torch.isfinite(vector).all() or not torch.isfinite(center).all() or vector.norm() < 1e-10:
        raise ValueError("Response-pooled direction is zero or nonfinite.")
    return vector, center


def export_vectors(output, pooled, pairs, manifest, run_dir, allow_truncated):
    """Save replay means and one compatible steering artifact per decoder block."""
    import torch
    layers = sorted(next(iter(pooled.values()))["Creative"])
    examples = [dict(Task=TASK, Item=pair["Standard"]["Item"], Paraphrase=0,
                     Cue_ID=key[0], Repeat=key[1]) for key, pair in sorted(pairs.items())]
    base = dict(protocol=PROTOCOL_VERSION, model=manifest["model"],
                requested_revision=manifest["requested_revision"],
                resolved_revision=manifest["resolved_revision"],
                head=None,
                baseline="Standard", examples=examples, prompt_version=PROMPT_VERSION,
                vector_kind="story_response_mean_contrast", extraction_split="discovery",
                extraction_position="post-block residual at response tokens; prompt/special tokens excluded",
                pooling="mean tokens per response; mean repeats per cue; equal mean across cues",
                scaling="raw Creative minus Standard mean; no unit normalization",
                center_kind="standard_response_pooled_post_block_residual_mean",
                recommended_scope="each_step", generation_manifest=manifest,
                generation_run=str(run_dir.resolve()), allow_truncated=allow_truncated,
                layer_indexing="zero-based decoder block; before final model norm")
    norms = []
    for layer in layers:
        vector, center = aggregate_layer(pooled, layer)
        artifact = dict(base, layer=layer, vector=vector, center=center)
        torch.save(artifact, output / f"layer_{layer:02d}.pt")
        metadata = {key: value for key, value in artifact.items() if key not in ("vector", "center")}
        metadata["vector_norm"] = vector.norm().item()
        (output / f"layer_{layer:02d}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        norms.append(dict(layer=layer, vector_norm=metadata["vector_norm"]))
    torch.save(dict(pooled=pooled, metadata=base), output / "response_means.pt")
    (output / "manifest.json").write_text(json.dumps(dict(**base, layers=norms), indent=2), encoding="utf-8")


def extract(args):
    """Extract all requested layers without regenerating or selecting stories by score."""
    manifest, pairs = read_discovery(args.run_dir, args.allow_truncated)
    print(f"{len(pairs)} discovery pairs; layers={args.layers or 'all'}.")
    if args.dry_run:
        return
    from argparse import Namespace
    from run_residual import provenance
    from interventions import load_engine
    args.output_dir.mkdir(parents=True, exist_ok=False)
    revision = manifest.get("resolved_revision") or manifest.get("requested_revision")
    engine = load_engine(manifest["model"], revision)
    current = provenance(engine, Namespace(model=manifest["model"], revision=revision))
    for field in ("resolved_revision",):
        if manifest.get(field) and current[field] != manifest[field]:
            raise ValueError(f"Extraction/generation mismatch in {field}.")
    pooled = pool_stories(engine, pairs, args.layers)
    export_vectors(args.output_dir, pooled, pairs, manifest, args.run_dir, args.allow_truncated)
    print(f"Saved vectors in {args.output_dir}; select a layer using held-out validation.")


def parse_args():
    """Parse the two explicit stages and validate generation settings before loading."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate")
    generation.add_argument("--model", required=True)
    generation.add_argument("--revision")
    generation.add_argument("--cues", type=Path, default=DEFAULT_CUES)
    generation.add_argument("--split", choices=["discovery", "validation", "test"], default="discovery")
    generation.add_argument("--repeats", type=int, default=1)
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--temperature", type=float, default=0.7)
    generation.add_argument("--top-p", type=float, default=1.0)
    generation.add_argument("--top-k", type=int, default=0)
    generation.add_argument("--repetition-penalty", type=float, default=1.0)
    generation.add_argument("--max-tokens", type=int, default=256)
    extraction = commands.add_parser("extract")
    extraction.add_argument("--run-dir", type=Path, required=True)
    extraction.add_argument("--layers", type=int, nargs="+")
    extraction.add_argument("--allow-truncated", action="store_true")
    for command in (generation, extraction):
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        import math
        if (args.repeats < 1 or args.max_tokens < 1 or not math.isfinite(args.temperature) or
                args.temperature < 0 or not 0 < args.top_p <= 1 or args.top_k < 0 or
                not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0):
            parser.error("Require repeats/max-tokens >=1, finite temperature >=0, 0<top-p<=1, top-k>=0 and finite repetition-penalty>0.")
    elif args.layers is not None and (min(args.layers) < 0 or len(set(args.layers)) != len(args.layers)):
        parser.error("Layers must be unique nonnegative decoder block indices.")
    return args


def main():
    """Dispatch generation or extraction; no cluster jobs are submitted automatically."""
    args = parse_args()
    (generate if args.command == "generate" else extract)(args)


if __name__ == "__main__":
    main()
