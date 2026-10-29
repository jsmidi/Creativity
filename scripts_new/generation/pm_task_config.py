"""Pilot tasks; these are not a verified reproduction of AGC-Bench items."""
TASK_NAMES = dict(aut="Alternative Uses Task", dat="Divergent Association Task",
                  cdat="Conditional Divergent Association Task", drat="Divergent Remote Association Test")
CONDITIONS = ["standard", "creative", "effective", "conventional", "boring", "non_creative"]

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


TASK_PROMPTS = {
    "Alternative Uses Task": {
        "standard": "List exactly 10 uses for the object {item} other than its primary use. Provide only a numbered list (1-10), with one use per line and no explanations.",
        "creative": "List exactly 10 unusual and original uses for the object {item}. Look for unexpected applications that exploit the object's properties in surprising ways. Every use must be physically possible. Provide only a numbered list (1-10), with one use per line and no explanations.",
        "effective": "List exactly 10 practical and useful alternative uses for the object {item}. Focus on applications that solve everyday problems or make effective use of the object's properties. Every use must be physically possible. Provide only a numbered list (1-10), with one use per line and no explanations.",
        "conventional": "List exactly 10 familiar and established uses for the object {item}. Choose conventional applications that most people would commonly think of. Every use must be physically possible. Provide only a numbered list (1-10), with one use per line and no explanations.",
        "boring": "List exactly 10 obvious and predictable uses for the object {item}. Choose boring applications that most people would readily think of. Every use must be physically possible. Provide only a numbered list (1-10), with one use per line and no explanations.",
        "non_creative": "List exactly 10 common uses for the object {item}. Choose familiar uses that people would typically mention first, and avoid unusual or novel ideas. Every use must be physically possible. Provide only a numbered list (1-10), with one use per line and no explanations."
    },
    "Conditional Divergent Association Task": {
        "standard": "List 10 distinct single English nouns that are as semantically unrelated from each other as possible, yet semantically associated with the following cue word: {item}. Return only a numbered list (1-10), without explanations.",
        "creative": "List 10 unusual and creative single English nouns that are as semantically unrelated from each other as possible, yet semantically associated with the following cue word: {item}. Return only a numbered list (1-10), without explanations.",
        "effective": "List 10 useful and effective single English nouns that are as semantically unrelated from each other as possible, yet semantically associated with the following cue word: {item}. Return only a numbered list (1-10), without explanations.",
        "conventional": "List 10 familiar and conventional single English nouns that are as semantically unrelated from each other as possible, yet semantically associated with the following cue word: {item}. Return only a numbered list (1-10), without explanations.",
        "boring": "List 10 obvious and boring single English nouns that are as semantically unrelated from each other as possible, yet semantically associated with the following cue word: {item}. Return only a numbered list (1-10), without explanations.",
        "non_creative": "List 10 common single English nouns that are as semantically related to each other as possible, yet semantically associated with the following cue word: {item}. Return only a numbered list (1-10), without explanations.",
    },
    "Divergent Remote Association Test": {
        "standard": "List 10 distinct single English nouns that can each relate, literally or metaphorically, to all of these anchors: {item}. Make the nouns as different in meaning from one another as possible. Return only a numbered list (1-10), without explanations.",
        "creative": "List 10 unusual and creative English nouns that are as semantically unrelated as possible from each other and each of which could be metaphorically applied to all of these anchors: {item}. Return only a numbered list (1-10), without explanations.",
        "effective": "List 10 useful and effective English nouns each of which could be metaphorically applied to all of these anchors: {item}. Return only a numbered list (1-10), without explanations.",
        "conventional": "List 10 familiar and conventional English nouns each of which could be metaphorically applied to all of these anchors: {item}. Return only a numbered list (1-10), without explanations.",
        "boring": "List 10 obvious and boring English nouns each of which could be metaphorically applied to all of these anchors: {item}. Return only a numbered list (1-10), without explanations.",
        "non_creative": "List 10 common English nouns that are as semantically related as possible to each other and each of which could be metaphorically applied to all of these anchors: {item}. Return only a numbered list (1-10), without explanations."
    },
    "Divergent Association Task": {
        "standard": "List 10 distinct single English nouns that are as semantically unrelated and distant from each other as possible. Provide only a numbered list (1-10) of single words with no explanations, sentences, or categories.",
        "creative": "List 10 unusual and creative English nouns that are as semantically unrelated and distant from each other as possible. Return only a numbered list (1-10), without explanations.",
        "effective": "List 10 useful everyday English nouns from different domains. Choose words that are easy to understand and have clear meanings. Return only a numbered list (1-10), without explanations.",
        "conventional": "List 10 familiar English nouns from common semantic categories. Choose conventional words that most people would recognize. Return only a numbered list (1-10), without explanations.",
        "boring": "List 10 ordinary English nouns from common semantic categories. Choose boring words without trying to be unusual. Return only a numbered list (1-10), without explanations.",
        "non_creative": "List 10 common English nouns that are as semantically related and close to each other as possible. Return only a numbered list (1-10), without explanations."
    },
    "Scientific Hypothesis Generation": {
            "standard": "Generate a concise scientific hypothesis (1-2 sentences) to explain the following observation: {item}",
            "creative": "Generate an unusual and imaginative scientific hypothesis (1-2 sentences) to explain the following observation: {item}. Favor unexpected explanatory mechanisms or surprising connections. The hypothesis must be scientifically testable.",
            "effective": "Generate a useful and plausible scientific hypothesis (1-2 sentences) to explain the following observation: {item}. Favor an explanation that leads to clear, informative predictions. The hypothesis must be scientifically testable.",
            "conventional": "Generate a conventional scientific hypothesis (1-2 sentences) to explain the following observation: {item}. Favor established explanatory mechanisms that follow familiar scientific principles. The hypothesis must be scientifically testable.",
            "boring": "Generate an obvious and unsurprising scientific hypothesis (1-2 sentences) to explain the following observation: {item}. Favor a simple explanation that most people would readily suggest. The hypothesis must be scientifically testable.",
            "non_creative": "Generate the most standard and familiar scientific hypothesis (1-2 sentences) that people would typically suggest to explain the following observation: {item}. Avoid unusual or novel mechanisms. The hypothesis must be scientifically testable."
        },
    "Metaphor Generation": {
            "standard": "Generate a metaphor (1-2 sentences) to describe the following concept or experience: {item}",
            "creative": "Generate an unusual and imaginative metaphor (1-2 sentences) to describe the following concept or experience: {item}. Combine conceptually distant ideas in a surprising way while communicating the intended meaning.",
            "effective": "Generate a clear and informative metaphor (1-2 sentences) to describe the following concept or experience: {item}. Use a comparison that helps explain the intended meaning.",
            "conventional": "Generate a familiar and conventional metaphor (1-2 sentences) to describe the following concept or experience: {item}. Use a comparison that is commonly associated with this concept.",
            "boring": "Generate a simple and predictable metaphor (1-2 sentences) to describe the following concept or experience: {item}. Use an obvious comparison that clearly communicates the intended meaning.",
            "non_creative": "Generate the most familiar and commonly used metaphor (1-2 sentences) to describe the following concept or experience: {item}. Choose a standard comparison that people would typically make, and avoid unusual or novel connections."
        }
}

def build_prompt(task_name, item, condition="standard"):
    config = get_task_config(task_name)
    if config is None:
        raise ValueError(f"Unknown task: {task_name}")
    condition = condition.lower()
    if condition not in ("standard", "creative", "effective", "conventional", "boring", "non_creative"):
        raise ValueError(f"Unknown condition: {condition}")
    return TASK_PROMPTS[task_name][condition].format(item=item)


def get_all_experiments():
    experiments = []

    for task_name, config in get_task_config().items():
        for item in config["items"]:
            for condition in [
                "standard", "creative", "effective",
                "conventional", "boring", "non_creative"
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
