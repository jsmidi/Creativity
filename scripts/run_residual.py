"""Extract candidate directions; test addition, suppression, patching and transfer.

Examples in scripts/README.md. No inference or downloads occur on import.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import random
import uuid
from experiment import PROTOCOL_VERSION, build_prompt, canonical_item, stable_seed
from task_config import get_task_config


def extraction_examples(tasks, items, paraphrases):
    """Expand tasks, items and wordings into extraction examples."""
    if items and len(tasks) != 1:
        raise ValueError("--items requires a single extraction task.")
    return [dict(Task=task, Item=item, Paraphrase=p)
            for task in tasks for item in (items or get_task_config(task)["items"])
            for p in paraphrases]


def validate_holdout(artifact, task, items, paraphrases, split):
    """Reject extraction item/wording overlap in validation and test runs."""
    if split == "pilot":
        return
    examples = artifact["examples"]
    used_p = {e["Paraphrase"] for e in examples}
    if set(paraphrases) & used_p:
        raise ValueError("Validation/test instruction paraphrases must be held out from extraction.")
    used_items = {canonical_item(e["Item"]) for e in examples + artifact.get("audit_examples", []) if e["Task"] == task}
    # DAT has no object axis; reserve instruction wordings and fresh generations.
    if task != "Divergent Association Task" and used_items & {canonical_item(i) for i in items}:
        raise ValueError("Validation/test items overlap extraction items.")


def provenance(engine, args):
    """Record the model revision and execution library versions."""
    return dict(model=args.model, requested_revision=args.revision or "main",
                resolved_revision=getattr(engine.model.config, "_commit_hash", None),
                torch=importlib.metadata.version("torch"),
                transformers=importlib.metadata.version("transformers"))


def collect_residual_activations(engine, examples, layer, baseline="Standard", head=None):
    """Collect Creative/baseline states at the last chat-prompt token.

    Each returned tensor is [examples, hidden_size]. With an explicit head,
    it is [examples, head_dim] instead. The post-block hook that reads the
    residual is in ActivationEngine.capture_prompt_activation (interventions.py).
    This stage runs prompt forward passes; it does not generate responses.
    """
    import torch
    creative_states, baseline_states = [], []
    for index, example in enumerate(examples):
        instruction = get_task_config(example["Task"])["instruction"]
        creative_prompt = build_prompt(instruction, example["Item"], "Creative", example["Paraphrase"])
        baseline_prompt = build_prompt(instruction, example["Item"], baseline, example["Paraphrase"])
        creative_states.append(engine.capture_prompt_activation(engine.inputs(creative_prompt), layer, head))
        baseline_states.append(engine.capture_prompt_activation(engine.inputs(baseline_prompt), layer, head))
        print(f"Collected residual pair {index + 1}/{len(examples)}", flush=True)
    return torch.stack(creative_states), torch.stack(baseline_states)


def extract(args):
    """Save the Creative-minus-baseline prompt activation and baseline center."""
    examples = extraction_examples(args.tasks, args.items, args.paraphrases)
    print(f"{len(examples) * 2} forward passes; layer {args.layer}, head {args.head}.")
    if args.dry_run:
        return
    import torch
    from interventions import load_engine
    engine = load_engine(args.model, args.revision)
    # 1. COLLECT: one Creative and one baseline activation per prompt example.
    creative_states, baseline_states = collect_residual_activations(
        engine, examples, args.layer, args.baseline, args.head)

    # 2. AVERAGE: Creative minus baseline, with equal weights for each task.
    by_task, centers = {}, []
    for task in dict.fromkeys(example["Task"] for example in examples):
        indices = [i for i, example in enumerate(examples) if example["Task"] == task]
        by_task[task] = (creative_states[indices] - baseline_states[indices]).mean(0)
        centers.append(baseline_states[indices].mean(0))
    vector = torch.stack(list(by_task.values())).mean(0)
    center = torch.stack(centers).mean(0)

    # 3. SAVE: vector, baseline center and model/layer metadata in one .pt file.
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
    print(f"Saved {args.artifact}; norm {vector.norm().item():.4f}.")


def intervention_arms(args):
    """Build baseline, learned, norm-matched random, suppression and donor-patching conditions."""
    arms = [(c, c, "none", 0.0, -1) for c in ["Standard", "Creative", "Effective", "Conventional", "Boring"]]
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
    if getattr(args, "omit_temperature_control", False):
        arms = [arm for arm in arms if arm[2] != "temperature"]
    return arms


def random_directions(vector, count, seed):
    """Draw reproducible Gaussian directions with the learned vector norm."""
    import torch
    rng = torch.Generator().manual_seed(seed)
    controls = []
    for _ in range(count):
        noise = torch.randn(vector.shape, generator=rng)
        controls.append(noise / noise.norm() * vector.norm())
    return controls


def write_interventions(args, engine, artifact, config, controls, trials, run_id, path):
    """Stream intervention responses to CSV, caching prompt-specific donor activations."""
    vector, center = artifact["vector"], artifact["center"]
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
                    patch_cache[key] = engine.capture_prompt_activation(engine.inputs(donor), artifact["layer"], artifact["head"])
                direction = patch_cache[key]
                scope = "prefill"  # Matched donor state exists only at the prompt boundary.
            temperature = args.high_temperature if mode == "temperature" else args.temperature
            intervention = direction if mode in ("add", "suppress", "patch") else None
            timestamp = datetime.now(timezone.utc).isoformat()
            result = engine.generate(prompt, seed, temperature, args.max_tokens,
                                     artifact["layer"], intervention, mode if intervention is not None else "add",
                                     alpha, center, artifact["head"], scope)
            row = dict(Run_ID=run_id, Response_ID=f"{run_id}:{order}", Model=args.model,
                       Task=args.task, Instruction=config["instruction"], Item=item,
                       Condition=label, Generation_Seed=seed, Temperature=temperature,
                       Max_Tokens=args.max_tokens, Top_P=1.0, Top_K=0, Timestamp_UTC=timestamp,
                       Prompt=prompt, Response=result["Response"],
                       Generated_Tokens=result["Generated_Tokens"], Prompt_Tokens=result["Prompt_Tokens"],
                       Error="", Model_Dtype=str(engine.model.dtype),
                       Artifact=str(args.artifact.resolve()), Scope=scope)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(f"{order + 1}/{len(trials)} {label}: {result['Intervention_Calls']} interventions")
    print(f"Saved {path}")


def generate(args):
    """Load a compatible vector, validate held-out prompts and run paired interventions."""
    import torch
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    if artifact["protocol"] != PROTOCOL_VERSION:
        raise ValueError("Incompatible vector protocol.")
    if args.model != artifact["model"]:
        raise ValueError("Use the same model as extraction. Cross-model tests require re-extraction.")
    config = get_task_config(args.task)
    items = args.items or config["items"]
    validate_holdout(artifact, args.task, items, args.paraphrases, args.split)
    arms = intervention_arms(args)
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
    for field in ["resolved_revision"]:
        if artifact.get(field) and artifact[field] != metadata[field]:
            raise ValueError(f"Extraction/generation mismatch in {field}")
    vector = artifact["vector"]
    controls = random_directions(vector, args.random_vectors, args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"results_{args.task.lower().replace(' ', '_')}_{run_id}"
    path = args.output_dir / f"{stem}.csv"
    write_interventions(args, engine, artifact, config, controls, trials, run_id, path)


def main():
    """Validate settings and dispatch extraction or generation.

    Final-test generation accepts one alpha; prior tuning is not tracked."""
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
            sub.add_argument("--baseline", choices=["Standard", "Conventional", "Boring"], default="Conventional")
        else:
            sub.add_argument("--task", choices=list(get_task_config()), default="Alternative Uses Task")
            sub.add_argument("--split", choices=["pilot", "validation", "test"], default="validation")
            sub.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
            sub.add_argument("--random-vectors", type=int, default=5)
            sub.add_argument("--repeats", type=int, default=10)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--temperature", type=float, default=0.7)
            sub.add_argument("--omit-temperature-control", action="store_true", help="Exclude the sampling-temperature arm for all-greedy experiments")
            sub.add_argument("--high-temperature", type=float, default=1.0)
            sub.add_argument("--max-tokens", type=int, default=800)
            sub.add_argument("--scope", choices=["each_step", "prefill"], default="each_step")
            sub.add_argument("--output-dir", type=Path, default=Path("outputs/interventions"))
    args = parser.parse_args()
    if args.command == "generate":
        if args.repeats < 1 or args.random_vectors < 1 or args.temperature < 0 or args.max_tokens < 1:
            parser.error("Require repeats/random-vectors/max-tokens >=1 and temperature >=0.")
        if not args.omit_temperature_control and args.high_temperature <= args.temperature:
            parser.error("Require high-temperature > temperature when the temperature control is enabled.")
        if args.split == "test" and len(args.alphas) != 1:
            parser.error("Freeze one validation-selected alpha for test runs.")
    (extract if args.command == "extract" else generate)(args)


if __name__ == "__main__":
    main()
