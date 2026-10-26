"""Pilot tasks; these are not a verified reproduction of AGC-Bench items."""
TASK_NAMES = dict(aut="Alternative Uses Task", dat="Divergent Association Task",
                  cdat="Conditional Divergent Association Task", drat="Divergent Remote Association Test")


def get_task_config(task_name=None):
    """Return the complete task registry or one task definition containing instructions and pilot items."""
    tasks = {
        "Alternative Uses Task": {
            "instruction": "List exactly 10 uses for the object {item} other than its primary use. Provide only a numbered list (1-10), with one use per line and no explanations.",
            "items": [
                "book",
                "fork",
                "paperclip",
                "towel",
                "tin can",
                "boot",
                "envelope",
                "bench",
                "screwdriver",
                "ladder",
                "drawer",
                "stand"
            ]
        },
        "Scientific Hypothesis Generation": {
            "instruction": "Generate a concise scientific hypothesis (1-2 sentences) to explain the following observation: {item}",
            "items": [
                "On a field trip, you drive past a massive field with hundreds of large holes visible as far as you can see.",
                "A specific type of local plant only blooms when there is a full moon.",
                "A newly discovered species of insect completely ignores natural sugar but is highly attracted to artificial sweeteners.",
                "In a controlled lab environment, mice living in blue-lit cages sleep for 12 hours, while mice in red-lit cages sleep for only 4 hours.",
                "A stream running through a dense forest never freezes in the winter, even when the surrounding temperature drops well below freezing."
            ]
        },
        "Metaphor Generation": {
            "instruction": "Generate a metaphor (1-2 sentences) to describe the following concept or experience: {item}",
            "items": [
                "a boring experience",
                "the feeling of being completely overwhelmed with work",
                "an idea that comes to you suddenly and unexpectedly",
                "a conversation that goes in circles without resolving anything",
                "the feeling of learning a difficult, highly complex new skill"
            ]
        },
        "Conditional Divergent Association Task": {
            "instruction": "List 10 distinct single English nouns that are related to the cue {item} but as different in meaning from one another as possible. Return only a numbered list (1-10), without explanations.",
            # Exploratory cues, not the published 500-cue benchmark.
            "items": ["music", "river", "school", "light", "food", "time", "plant", "city", "machine", "art"]
        },
        "Divergent Remote Association Test": {
            "instruction": "List 10 distinct single English nouns that can each relate, literally or metaphorically, to all of these anchors: {item}. Make the nouns as different in meaning from one another as possible. Return only a numbered list (1-10), without explanations.",
            # First five scientific anchor sets, Schapiro et al. (2026), Appendix F.3.
            "items": ["heart | engine | marketplace | equation",
                      "immune system | fire | organization | theorem",
                      "evolution | friction | election | polynomial",
                      "genome | lattice | algorithm | court",
                      "heartbeat | wave | oscillator | function"]
        },
        "Divergent Association Task": {
            "instruction": "Name 10 different single English nouns that are as semantically unrelated and distant from each other as possible. Provide only a numbered list (1-10) of single words with no explanations, sentences, or categories.",
            "items": [
                "" # The DAT doesn't need a target item, so we leave it blank
            ]
        }
    }

    if task_name is None:
        return tasks
    return tasks.get(task_name)


TASK_TEMPLATES = {
    "Alternative Uses Task": (
        "List exactly 10 highly {style} uses for the object {item} "
        "other than its primary use. {criterion} "
        "Every use must be physically possible. "
        "Provide only a numbered list (1-10), "
        "with one use per line and no explanations."
    ),

    "Scientific Hypothesis Generation": (
        "Generate a highly {style} scientific hypothesis "
        "(1-2 sentences) to explain the following observation: "
        "{item}. {criterion} "
        "The hypothesis must be scientifically testable."
    ),

    "Metaphor Generation": (
        "Generate a highly {style} metaphor (1-2 sentences) "
        "to describe the following concept or experience: "
        "{item}. {criterion} "
        "The metaphor must communicate the intended concept."
    ),

    "Conditional Divergent Association Task": (
        "List exactly 10 highly {style} associations with the "
        "cue {item}, using distinct single English nouns. "
        "{criterion} "
        "Every noun must meaningfully relate to the cue. "
        "Provide only a numbered list (1-10), "
        "without explanations."
    ),

    "Divergent Remote Association Test": (
        "List exactly 10 highly {style} single English nouns "
        "that each relate, literally or metaphorically, "
        "to all these anchors: {item}. "
        "{criterion} "
        "Every noun must connect to every anchor. "
        "Provide only a numbered list (1-10), "
        "without explanations."
    ),

    "Divergent Association Task": (
        "Generate exactly 10 distinct single English nouns "
        "using a highly {style} approach. "
        "{criterion} "
        "Provide only a numbered list (1-10), "
        "without explanations."
    )
}


TASK_CONDITIONS = {
    "Alternative Uses Task": {
        "creative": (
            "Look for unexpected applications that exploit "
            "the object's properties in unusual ways."
        ),
        "effective": (
            "Look for practical applications that exploit "
            "the object's properties in useful ways."
        ),
        "conventional": (
            "Look for familiar applications that exploit "
            "the object's properties in established ways."
        ),
        "boring": (
            "Look for obvious applications that exploit "
            "the object's properties in predictable ways."
        )
    },

    "Scientific Hypothesis Generation": {
        "creative": (
            "Favor unexpected explanatory mechanisms "
            "that reveal surprising connections."
        ),
        "effective": (
            "Favor plausible explanatory mechanisms "
            "that provide useful predictions."
        ),
        "conventional": (
            "Favor established explanatory mechanisms "
            "that follow familiar principles."
        ),
        "boring": (
            "Favor obvious explanatory mechanisms "
            "that provide unsurprising explanations."
        )
    },

    "Metaphor Generation": {
        "creative": (
            "Connect conceptually distant ideas through "
            "surprising and imaginative comparisons."
        ),
        "effective": (
            "Connect clearly relevant ideas through "
            "useful and informative comparisons."
        ),
        "conventional": (
            "Connect commonly associated ideas through "
            "familiar and established comparisons."
        ),
        "boring": (
            "Connect obviously related ideas through "
            "simple and predictable comparisons."
        )
    },

    "Conditional Divergent Association Task": {
        "creative": (
            "Explore distant and unexpected connections "
            "while maximizing differences between nouns."
        ),
        "effective": (
            "Explore relevant and useful connections "
            "while maintaining practical relationships between nouns."
        ),
        "conventional": (
            "Explore familiar and typical connections "
            "while maintaining common relationships between nouns."
        ),
        "boring": (
            "Explore obvious and predictable connections "
            "while minimizing differences between nouns."
        )
    },

    "Divergent Remote Association Test": {
        "creative": (
            "Favor surprising and remote connections "
            "that differ substantially between nouns."
        ),
        "effective": (
            "Favor useful and meaningful connections "
            "that work consistently across anchors."
        ),
        "conventional": (
            "Favor familiar and established connections "
            "that follow common associative patterns."
        ),
        "boring": (
            "Favor obvious and predictable connections "
            "that require little associative imagination."
        )
    },

    "Divergent Association Task": {
        "creative": (
            "Choose words from distant semantic domains, "
            "maximizing unusual and unexpected differences."
        ),
        "effective": (
            "Choose words from useful everyday contexts, "
            "prioritizing practical and relevant associations."
        ),
        "conventional": (
            "Choose words from familiar semantic categories, "
            "prioritizing typical and established associations."
        ),
        "boring": (
            "Choose words from closely related categories, "
            "minimizing unusual and unexpected differences."
        )
    }
}


def build_prompt(task_name, item, condition="standard"):
    config = get_task_config(task_name)
    if config is None:
        raise ValueError(f"Unknown task: {task_name}")
    condition = condition.lower()
    if condition == "neutral":
        condition = "standard"
    if condition not in ("standard", "creative", "effective", "conventional", "boring"):
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "standard":
        # Retain existing standard instructions
        return config["instruction"].format(
            item=item
        )

    template = TASK_TEMPLATES[task_name]
    criterion = TASK_CONDITIONS[task_name][condition]

    return template.format(
        item=item,
        style=condition,
        criterion=criterion
    )


def get_all_experiments():
    experiments = []

    for task_name, config in get_task_config().items():
        for item in config["items"]:
            for condition in [
                "standard", "creative", "effective",
                "conventional", "boring"
            ]:
                experiments.append({
                    "task": task_name,
                    "item": item,
                    "condition": condition,
                    "prompt": build_prompt(
                        task_name, item, condition
                    )
                })

    return experiments
