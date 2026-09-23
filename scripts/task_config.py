"""Pilot tasks; these are not a verified reproduction of AGC-Bench items."""
def get_task_config(task_name=None):
    tasks = {
        "Alternative Uses Task": {
            "instruction": "List exactly 10 uses for the object {item} other than its primary use. Provide only a numbered list (1-10), with one use per line and no explanations.",
            "items": [
                "book", 
                "fork", 
                "paperclip", 
                "towel", 
                "can"
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

