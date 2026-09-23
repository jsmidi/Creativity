"""Extract candidate directions; test addition, suppression, patching and transfer.

Examples in scripts/README.md. No inference or downloads occur on import.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import uuid
from experiment import PROTOCOL_VERSION, build_prompt, canonical_item, stable_seed
from task_config import get_task_config


def extraction_examples(tasks, items, paraphrases):
    if items and len(tasks) != 1:
        raise ValueError("--items requires a single extraction task.")
    return [dict(Task=task, Item=item, Paraphrase=p)
            for task in tasks for item in (items or get_task_config(task)["items"])
            for p in paraphrases]


def validate_holdout(artifact, task, items, paraphrases, split):
    if split == "pilot":
        return
    examples = artifact["examples"]
    used_p = {e["Paraphrase"] for e in examples}
    if set(paraphrases) & used_p:
        raise ValueError("Validation/test instruction paraphrases must be held out from extraction.")
    used_items = {canonical_item(e["Item"]) for e in examples if e["Task"] == task}
    # DAT has no object axis; reserve instruction wordings and fresh generations.
    if task != "Divergent Association Task" and used_items & {canonical_item(i) for i in items}:
        raise ValueError("Validation/test items overlap extraction items.")


def provenance(engine, args):
    return dict(model=args.model, requested_revision=args.revision or "main",
                resolved_revision=getattr(engine.model.config, "_commit_hash", None),
                torch=importlib.metadata.version("torch"),
                transformers=importlib.metadata.version("transformers"),
                chat_template_sha256=hashlib.sha256(engine.tokenizer.chat_template.encode()).hexdigest())


def extract(args):
    examples = extraction_examples(args.tasks, args.items, args.paraphrases)
    print(f"{len(examples) * 2} forward passes; layer {args.layer}, head {args.head}.")
    if args.dry_run:
        return
    import torch
    from interventions import load_engine
    engine = load_engine(args.model, args.revision)
    task_deltas, task_centers = {}, {}
    for example in examples:
        task, item, p = example["Task"], example["Item"], example["Paraphrase"]
        instruction = get_task_config(task)["instruction"]
        creative = build_prompt(instruction, item, "Creative", p)
        baseline = build_prompt(instruction, item, args.baseline, p)
        positive = engine.capture(engine.inputs(creative), args.layer, args.head)
        negative = engine.capture(engine.inputs(baseline), args.layer, args.head)
        task_deltas.setdefault(task, []).append(positive - negative)
        task_centers.setdefault(task, []).append(negative)
    # Equal task weights prevent a task with more items dominating the direction.
    by_task = {task: torch.stack(values).mean(0) for task, values in task_deltas.items()}
    vector = torch.stack(list(by_task.values())).mean(0)
    center = torch.stack([torch.stack(values).mean(0) for values in task_centers.values()]).mean(0)
    if not torch.isfinite(vector).all() or vector.norm() < 1e-10:
        raise ValueError("Extracted direction is zero or nonfinite.")
    artifact = dict(protocol=PROTOCOL_VERSION, **provenance(engine, args), layer=args.layer,
                    head=args.head, baseline=args.baseline, examples=examples,
                    vector=vector, center=center, task_vectors=by_task,
                    scaling="raw mean delta; alpha=1 adds one extraction contrast",
                    extraction_position="last token of formatted chat prompt")
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    if args.artifact.exists():
        raise FileExistsError(f"Choose a new artifact path: {args.artifact}")
    torch.save(artifact, args.artifact)
    metadata = {key: value for key, value in artifact.items() if key not in ("vector", "center", "task_vectors")}
    metadata["vector_norm"] = vector.norm().item()
    args.artifact.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.artifact}; norm {vector.norm().item():.4f}.")


def generate(args):
    import torch
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    if artifact["protocol"] != PROTOCOL_VERSION:
        raise ValueError("Incompatible vector protocol.")
    if args.model != artifact["model"]:
        raise ValueError("Use the same model as extraction. Cross-model tests require re-extraction.")
    config = get_task_config(args.task)
    items = args.items or config["items"]
    validate_holdout(artifact, args.task, items, args.paraphrases, args.split)
    arms = [(c, c, "none", 0.0, -1) for c in ["Standard", "Creative", "Conventional"]]
    for alpha in args.alphas:
        arms.extend([(f"Standard_add_{alpha:g}", "Standard", "add", alpha, -1),
                     (f"Creative_subtract_{alpha:g}", "Creative", "add", -alpha, -1)])
        for index in range(args.random_vectors):
            arms.append((f"Standard_random{index}_{alpha:g}", "Standard", "add", alpha, index))
    arms.extend([("Creative_suppress", "Creative", "suppress", 1.0, -1),
                 ("Standard_suppress", "Standard", "suppress", 1.0, -1),
                 ("Standard_high_temperature", "Standard", "temperature", 0.0, -1),
                 ("Standard_patch_creative", "Standard", "patch", 1.0, -1),
                 ("Creative_patch_standard", "Creative", "patch", 1.0, -1)])
    for index in range(args.random_vectors):
        arms.append((f"Creative_random_suppress{index}", "Creative", "suppress", 1.0, index))
    trials = [(item, p, repeat, arm) for item in items for p in args.paraphrases
              for repeat in range(args.repeats) for arm in arms]
    random.Random(args.seed).shuffle(trials)
    print(f"{len(trials)} generations; split {args.split}; extraction tasks {sorted({e['Task'] for e in artifact['examples']})}.")
    if args.dry_run:
        return
    from interventions import load_engine
    # Resolve to the extraction commit where available; refuse silent revision drift.
    args.revision = args.revision or artifact.get("resolved_revision")
    engine = load_engine(args.model, args.revision)
    metadata = provenance(engine, args)
    for field in ["resolved_revision", "chat_template_sha256"]:
        if artifact.get(field) and artifact[field] != metadata[field]:
            raise ValueError(f"Extraction/generation mismatch in {field}")
    vector, center = artifact["vector"], artifact["center"]
    rng = torch.Generator().manual_seed(args.seed)
    controls = []
    for _ in range(args.random_vectors):
        noise = torch.randn(vector.shape, generator=rng)
        controls.append(noise / noise.norm() * vector.norm())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"results_{args.task.lower().replace(' ', '_')}_{run_id}"
    path = args.output_dir / f"{stem}.csv"
    manifest = dict(**metadata, protocol=PROTOCOL_VERSION, run_id=run_id,
                    artifact=str(args.artifact.resolve()), artifact_sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
                    split=args.split, task=args.task, items=items, paraphrases=args.paraphrases,
                    repeats=args.repeats, alphas=args.alphas, random_vectors=args.random_vectors,
                    random_seed=args.seed, scope=args.scope, temperature=args.temperature,
                    high_temperature=args.high_temperature, max_tokens=args.max_tokens)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    patch_cache = {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = None
        for order, (item, p, repeat, arm) in enumerate(trials):
            label, condition, mode, alpha, random_index = arm
            prompt = build_prompt(config["instruction"], item, condition, p)
            block = f"{args.task}|{canonical_item(item)}|p{p}|r{repeat}"
            seed = stable_seed(args.seed, block)
            direction = controls[random_index] if random_index >= 0 else vector
            scope = args.scope
            if mode == "patch":
                donor_condition = "Creative" if condition == "Standard" else "Standard"
                key = (item, p, donor_condition)
                if key not in patch_cache:
                    donor = build_prompt(config["instruction"], item, donor_condition, p)
                    patch_cache[key] = engine.capture(engine.inputs(donor), artifact["layer"], artifact["head"])
                direction = patch_cache[key]
                scope = "prefill"  # Matched donor state exists only at the prompt boundary.
            temperature = args.high_temperature if mode == "temperature" else args.temperature
            intervention = direction if mode in ("add", "suppress", "patch") else None
            result = engine.generate(prompt, seed, temperature, args.max_tokens,
                                     artifact["layer"], intervention, mode if intervention is not None else "add",
                                     alpha, center, artifact["head"], scope)
            row = dict(Protocol=PROTOCOL_VERSION, Run_ID=run_id, Response_ID=f"{run_id}:{order}",
                       Model=args.model, Task=args.task, Item=item, Split=args.split,
                       Condition=label, Prompt_Condition=condition, Instruction=config["instruction"],
                       Prompt=prompt, Paraphrase=p, Repeat=repeat, Block_ID=block, Generation_Seed=seed,
                       Request_Order=order, Temperature=temperature, Max_Tokens=args.max_tokens,
                       Layer=artifact["layer"], Head=artifact["head"], Intervention=mode, Scope=scope,
                       Alpha=alpha, Random_Vector=random_index, Vector_Norm=direction.norm().item(),
                       Artifact_SHA256=manifest["artifact_sha256"], **result,
                       Status="ok" if result["Response"] else "empty")
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(f"{order + 1}/{len(trials)} {label}: {result['Intervention_Calls']} interventions")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["extract", "generate"]:
        sub = commands.add_parser(command)
        sub.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
        sub.add_argument("--revision")
        sub.add_argument("--artifact", type=Path, required=True)
        sub.add_argument("--items", nargs="+")
        sub.add_argument("--paraphrases", type=int, nargs="+", choices=range(3), default=[0] if command == "extract" else [1])
        sub.add_argument("--dry-run", action="store_true")
        if command == "extract":
            sub.add_argument("--tasks", nargs="+", choices=list(get_task_config()), default=["Alternative Uses Task"])
            sub.add_argument("--layer", type=int, required=True)
            sub.add_argument("--head", type=int, help="Query head index; omitted means full residual stream")
            sub.add_argument("--baseline", choices=["Standard", "Conventional"], default="Standard")
        else:
            sub.add_argument("--task", choices=list(get_task_config()), default="Alternative Uses Task")
            sub.add_argument("--split", choices=["pilot", "validation", "test"], default="validation")
            sub.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
            sub.add_argument("--random-vectors", type=int, default=5)
            sub.add_argument("--repeats", type=int, default=10)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--temperature", type=float, default=0.7)
            sub.add_argument("--high-temperature", type=float, default=1.0)
            sub.add_argument("--max-tokens", type=int, default=800)
            sub.add_argument("--scope", choices=["each_step", "prefill"], default="each_step")
            sub.add_argument("--output-dir", type=Path, default=Path("outputs/interventions"))
    args = parser.parse_args()
    if args.command == "generate":
        if args.repeats < 1 or args.random_vectors < 1 or args.temperature < 0 or args.high_temperature <= args.temperature or args.max_tokens < 1:
            parser.error("Require repeats/random-vectors/max-tokens >=1 and high-temperature > temperature >=0.")
        if args.split == "test" and len(args.alphas) != 1:
            parser.error("Freeze one validation-selected alpha for test runs.")
    (extract if args.command == "extract" else generate)(args)


if __name__ == "__main__":
    main()
