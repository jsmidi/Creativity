"""Audit paired writing outputs for length and literal cues, separately by model/split.

These are instruction-compliance and length checks, not creativity ratings or the
benchmark's semantic surprise score. Sentence counts use spaCy's rule sentencizer.
"""
import argparse
import json
from pathlib import Path
import re

import pandas as pd
import spacy

from writing_metrics import cue_coverage, load_stories, paired_contrast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = load_stories([args.inputs])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    frame["Word_Count"] = frame.Response.map(lambda text: len(re.findall(r"\b\w+(?:['’]\w+)*\b", text)))
    frame["Sentence_Count_Proxy"] = [len(list(doc.sents)) for doc in nlp.pipe(frame.Response, batch_size=64)]
    coverage = [cue_coverage(row.Response, row.Cue_Words) for row in frame.itertuples()]
    frame["All_Cues_Present"] = [int(all(values)) for values in coverage]
    frame["Missing_Cues"] = [json.dumps([word for word, present in zip(words, values) if not present])
                             for words, values in zip(frame.Cue_Words, coverage)]
    frame["Sentence_Limit_Passed_Proxy"] = frame.Sentence_Count_Proxy.between(1, 5).astype(int)
    frame["Literal_Compliance_Proxy"] = (frame.All_Cues_Present.eq(1) & frame.Sentence_Limit_Passed_Proxy.eq(1)
                                           & frame.Status.eq("ok")).astype(int)
    metrics = ["Word_Count", "Sentence_Count_Proxy", "All_Cues_Present",
               "Sentence_Limit_Passed_Proxy", "Literal_Compliance_Proxy"]
    summary = frame.groupby(["Model", "Split", "Condition"])[metrics].mean().reset_index()
    summary["Responses"] = frame.groupby(["Model", "Split", "Condition"]).size().to_numpy()
    effects = pd.DataFrame([dict(Model=model, Split=split, **paired_contrast(group, metric))
                           for (model, split), group in frame.groupby(["Model", "Split"])
                           for metric in metrics])
    summary.to_csv(args.output_dir / "condition_summary.csv", index=False)
    effects.to_csv(args.output_dir / "paired_effects.csv", index=False)
    frame.drop(columns=["Prompt_Token_IDs", "Response_Token_IDs"], errors="ignore").to_csv(
        args.output_dir / "response_audit.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(dict(
        inputs=str(args.inputs.resolve()), responses=len(frame), sentence_counter="spacy.blank('en') + sentencizer (proxy)",
        cue_match="case-insensitive whole word, no stemming", bootstrap="crossed cue and base-seed, 2000 draws, seed 42",
        limitations="No creativity/originality/coherence ratings or semantic surprise score; intervals exploratory and uncorrected"
    ), indent=2) + "\n")
    print(summary.to_string(index=False))
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
