"""Shared prompt protocol and reproducible trial scheduling (no model imports)."""
import hashlib
import random
import re

PROTOCOL_VERSION = "creativity-v2"
CONDITION_TEXT = {
    "Standard": ("", "", ""),
    "Conventional": ("Be conventional.", "Provide familiar ideas.", "Think conventionally."),
    "Effective": ("Be effective.", "Provide effective ideas.", "Focus on effectiveness."),
    "Boring": ("Be boring.", "Provide boring ideas.", "Keep your ideas dull."),
    "Creative": ("Be creative.", "Provide original ideas.", "Think inventively."),
}


def canonical_item(item):
    key = re.sub(r"\s+", " ", str(item).lower().strip())
    key = re.sub(r"^(?:a|an|the)\s+", "", key)
    return {"tin can": "can"}.get(key, key)


def build_prompt(instruction, item, condition, paraphrase=0):
    base = instruction.replace("{item}", str(item))
    suffix = CONDITION_TEXT[condition][paraphrase]
    return f"{base}\n{suffix}" if suffix else base


def stable_seed(*parts):
    value = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") % (2**31)


def trial_schedule(items, conditions, repeats, paraphrases, seed, randomize=False):
    if repeats < 1 or not items or not conditions or not paraphrases:
        raise ValueError("Items, conditions, paraphrases and repeats must be nonempty.")
    if len(set(conditions)) != len(conditions) or len(set(paraphrases)) != len(paraphrases):
        raise ValueError("Conditions and paraphrases must be unique.")
    if len({canonical_item(i) for i in items}) != len(items):
        raise ValueError("Items contain duplicate canonical labels.")
    for condition in conditions:
        for paraphrase in paraphrases:
            CONDITION_TEXT[condition][paraphrase]  # validate before a paid request
    trials = []
    for item in items:
        for repeat in range(repeats):
            for paraphrase in paraphrases:
                block = f"{canonical_item(item)}|p{paraphrase}|r{repeat}"
                for condition in CONDITION_TEXT:
                    if condition not in conditions:
                        continue
                    trials.append(dict(Item=item, Condition=condition, Repeat=repeat,
                                       Paraphrase=paraphrase, Block_ID=block,
                                       Generation_Seed=stable_seed(seed, block)))
    if randomize:
        random.Random(seed).shuffle(trials)
    return trials
