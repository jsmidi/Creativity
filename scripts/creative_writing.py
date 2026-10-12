"""Paired cue-word story prompts and explicit research splits.

Creative wording follows the MIT-licensed Creative Short Story benchmark's TEST
template. Standard wording and discovery/validation/test assignments are ours.
The bundled ten cues are a pilot, not Glazer et al.'s undisclosed 50-triplet set.
"""
import json
from pathlib import Path

TASK = "Creative Short Story"
DEFAULT_CUES = Path(__file__).resolve().parents[1] / "data/creative_writing/pilot_cues.json"
PROMPT_VERSION = "creative-short-story-paired-v1"
CREATIVE_TEMPLATE = (
    "You are an expert story writer. Write a creative short story using a maximum "
    "of five sentences. The story must include the following three words: {items}. "
    "However, the story should not be about {boring_theme}."
)
STANDARD_TEMPLATE = (
    "You are an expert story writer. Write a straightforward, coherent short story "
    "using a maximum of five sentences. The story must include the following "
    "three words: {items}. Incorporate the words naturally."
)


def load_cues(path=DEFAULT_CUES):
    """Read a cue JSON; reject duplicate triplets/IDs and unspecified splits."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    cues = document["data"]
    if not cues:
        raise ValueError("Cue file is empty.")
    seen_ids, seen_words = set(), set()
    for cue in cues:
        words = cue["items"]
        if (len(words) != 3 or any(not isinstance(w, str) or not w.strip() or
                                  len(w.split()) != 1 for w in words)):
            raise ValueError("Each cue needs exactly three single words.")
        key = tuple(sorted(w.lower().strip() for w in words))
        if len(set(key)) != 3 or key in seen_words or cue["id"] in seen_ids:
            raise ValueError("Cue IDs and unordered word triplets must be unique.")
        if not isinstance(cue["id"], str) or not cue["id"].strip():
            raise ValueError("Cue IDs must be nonempty strings.")
        if cue["split"] not in ("discovery", "validation", "test"):
            raise ValueError("Every cue needs an explicit discovery/validation/test split.")
        if not isinstance(cue["boring_theme"], str) or not cue["boring_theme"].strip():
            raise ValueError("Every cue needs an obvious storyline to exclude.")
        seen_ids.add(cue["id"])
        seen_words.add(key)
    return document


def story_prompt(cue, condition):
    """Construct the paired instruction; both arms share cues and sentence limit."""
    templates = {"Creative": CREATIVE_TEMPLATE, "Standard": STANDARD_TEMPLATE}
    return templates[condition].format(items=", ".join(cue["items"]),
                                       boring_theme=cue["boring_theme"])
