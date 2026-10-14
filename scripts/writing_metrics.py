"""Creative Short Story surprise, compliance, and paired cue/seed statistics.

Surprise follows the MIT-licensed mismayil/creative-story-gen implementation at
e0a187cd9a54c7a7772ace4c655526efd23bacea, src/metrics.py. For each sentence,
D is the average cosine distance among its unique content-word lemmas, not a
sentence embedding. Surprise = 2 * mean(abs(diff(D))). Originality/coherence
remain separate ratings. Upstream license: data/creative_writing/LICENSE.benchmark.
"""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

DEFAULT_EMBEDDING = "thenlper/gte-large"
DEFAULT_LANGUAGE = "en_core_web_sm"
RATING_COLUMNS = ["Creativity_Rating", "Originality_Rating", "Surprise_Rating",
                  "Effectiveness_Rating", "Coherence_Rating", "Theme_Avoided"]


def load_stories(inputs):
    """Load raw JSONL plus base-seed metadata; compact CSV inputs use their JSONL sibling."""
    files = set()
    for path in inputs:
        path = Path(path)
        found = path.rglob("responses.jsonl") if path.is_dir() else [path.with_suffix(".jsonl")]
        files.update(p.resolve() for p in found)
    frames = []
    for path in sorted(files):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not manifest.get("completed") or len(rows) != manifest["planned_responses"]:
            raise ValueError(f"Incomplete generation: {path}")
        frame = pd.DataFrame(rows)
        required = {"Response_ID", "Cue_ID", "Model", "Split", "Repeat", "Condition",
                    "Generation_Seed", "Cue_Words", "Boring_Theme", "Response", "Status"}
        if not required.issubset(frame):
            raise ValueError(f"Missing writing metadata in {path}")
        if not frame.Model.eq(manifest["model"]).all() or not frame.Split.eq(manifest["split"]).all():
            raise ValueError("Response model/split differs from its generation manifest.")
        frame["Base_Seed"] = manifest["seed"]
        frame["Source_File"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No writing responses.jsonl found.")
    frame = pd.concat(frames, ignore_index=True)
    if frame.Response_ID.duplicated().any():
        raise ValueError("Duplicate response IDs.")
    keys = ["Model", "Split", "Cue_ID", "Base_Seed", "Repeat", "Condition"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate cue/seed/repetition/condition blocks.")
    return frame


def content_words(doc):
    """Match upstream defaults: remove punctuation/stopwords, lowercase lemmas, unique words."""
    return sorted({token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop})


def sentence_dispersion(words, embeddings):
    """Mean ordered off-diagonal cosine distance; a one-word sentence has dispersion zero."""
    if not words:
        return float("nan")
    if len(words) == 1:
        return 0.0
    vectors = np.stack([embeddings[word] for word in words]).astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms == 0).any():
        raise ValueError("Embedding vectors must be finite and nonzero.")
    normalized = vectors / norms
    distances = 1 - normalized @ normalized.T
    # Exclude self-distances rather than assuming their floating-point value is zero.
    return float((distances.sum() - np.trace(distances)) / (len(words) * (len(words) - 1)))


def story_surprise(dispersions):
    """Match upstream factor two and <=1 sentence rule; undefined dispersion stays NaN."""
    if len(dispersions) <= 1:
        return 0.0
    return float(2 * np.abs(np.diff(dispersions)).mean())


def cue_coverage(text, words):
    """Case-insensitive literal whole-word compliance; no stemming or substring matches."""
    return [bool(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.I)) for word in words]


def prepare_sentences(frame, nlp):
    """Split with the trained parser, then reparse each sentence as the upstream code does."""
    sentences = [[span.text for span in doc.sents] for doc in nlp.pipe(frame.Response.tolist(), batch_size=64)]
    unique = sorted({sentence for story in sentences for sentence in story})
    words = {sentence: content_words(doc) for sentence, doc in
             zip(unique, nlp.pipe(unique, batch_size=64))}
    return sentences, words


def score_stories(frame, sentences, words, embeddings):
    """Retain every response; report surprise separately from literal instruction compliance."""
    rows, sentence_rows = [], []
    for row, story in zip(frame.to_dict("records"), sentences):
        usable = row["Status"] == "ok" and bool(row["Response"].strip())
        dispersions = [sentence_dispersion(words[sentence], embeddings) for sentence in story]
        coverage = cue_coverage(row["Response"], row["Cue_Words"])
        surprise = story_surprise(dispersions) if usable else float("nan")
        compliant = usable and all(coverage) and 1 <= len(story) <= 5
        rows.append(dict(Response_ID=row["Response_ID"], Sentence_Count=len(story),
                         Cue_Words_Present=sum(coverage), All_Cues_Present=int(usable and all(coverage)),
                         Missing_Cues=json.dumps([word for word, present in zip(row["Cue_Words"], coverage) if not present]),
                         Sentence_Limit_Passed=int(usable and 1 <= len(story) <= 5),
                         Literal_Compliance=int(compliant), Story_Surprise=surprise,
                         Compliant_Story_Surprise=surprise if compliant else float("nan"),
                         Surprise_Status="failed_response" if not usable else
                             "no_transition_upstream_zero" if len(story) < 2 else
                             "empty_content_sentence" if not np.isfinite(surprise) else "ok"))
        sentence_rows.extend(dict(Response_ID=row["Response_ID"], Sentence_Index=index,
            Sentence=sentence, Content_Words=json.dumps(words[sentence]),
            Word_Dispersion=dispersion) for index, (sentence, dispersion) in enumerate(zip(story, dispersions)))
    return frame.merge(pd.DataFrame(rows), on="Response_ID", validate="one_to_one"), pd.DataFrame(sentence_rows)


def export_ratings(frame, output, seed=42):
    """Create deterministic shuffled rating forms without model, condition, seed or split labels."""
    ordered = frame.sort_values("Response_ID").sample(frac=1, random_state=seed).reset_index(drop=True)
    ids = [f"story_{index + 1:06d}" for index in range(len(ordered))]
    key = ordered[["Response_ID", "Model", "Split", "Cue_ID", "Base_Seed", "Condition"]].copy()
    key.insert(0, "Rating_ID", ids)
    key.to_csv(output / "rating_key_private.csv", index=False)
    form = ordered[["Cue_Words", "Boring_Theme", "Response"]].copy()
    form["Cue_Words"] = form.Cue_Words.map(lambda values: ", ".join(values))
    form.insert(0, "Rating_ID", ids)
    form["Rater_ID"] = "rater_1"
    for column in RATING_COLUMNS:
        form[column] = ""
    form.to_csv(output / "blinded_ratings.csv", index=False)
    return key


def import_ratings(frame, path, key):
    """Validate independent raters, rating ranges and IDs, then average available judgments."""
    ratings = pd.read_csv(path, keep_default_na=False)
    if not {"Rating_ID", "Rater_ID", *RATING_COLUMNS}.issubset(ratings):
        raise ValueError("Use the exported blinded rating form schema.")
    ratings = ratings.loc[ratings[RATING_COLUMNS].astype(str).apply(lambda column: column.str.strip()).ne("").any(axis=1)].copy()
    if ratings.empty:
        raise ValueError("Rating file has no completed ratings.")
    if not set(ratings.Rating_ID).issubset(set(key.Rating_ID)):
        raise ValueError("Unknown Rating_ID in ratings.")
    if ratings.Rater_ID.str.strip().eq("").any() or ratings.duplicated(["Rating_ID", "Rater_ID"]).any():
        raise ValueError("Each story/rater combination must be unique and have a Rater_ID.")
    for column in RATING_COLUMNS:
        numeric = pd.to_numeric(ratings[column].replace("", np.nan), errors="raise")
        lower, upper = (0, 1) if column == "Theme_Avoided" else (1, 5)
        if (numeric.notna() & (~numeric.between(lower, upper) | numeric.mod(1).ne(0))).any():
            raise ValueError(f"Invalid discrete rating in {column}.")
        ratings[column] = numeric
    mapped = ratings.merge(key[["Rating_ID", "Response_ID"]], on="Rating_ID", validate="many_to_one")
    means = mapped.groupby("Response_ID")[RATING_COLUMNS].mean()
    counts = mapped.groupby("Response_ID")[RATING_COLUMNS].count().add_suffix("_N")
    return frame.merge(means.join(counts), on="Response_ID", how="left", validate="one_to_one")


def paired_contrast(frame, metric, draws=2000, seed=42):
    """Equal-cue Creative minus Standard; jointly bootstrap crossed cue and base-seed axes.

    Conditions stay paired; repeated generations are averaged within cue/seed.
    Different base seeds are runs of the same frozen protocol, not different items.
    Missing scored pairs are counted and never replaced by zero. Intervals are
    exploratory percentile intervals without multiple-comparison correction.
    """
    subset = frame[frame.Condition.isin(["Creative", "Standard"])].copy()
    settings = ["Temperature", "Top_P", "Top_K", "Repetition_Penalty", "Prompt_Version"]
    if any(column in subset and subset[column].nunique() > 1 for column in settings):
        raise ValueError("Do not pool different writing protocols or decoding settings.")
    keys = ["Cue_ID", "Base_Seed", "Repeat"]
    table = subset.pivot(index=keys, columns="Condition", values=metric).reindex(columns=["Standard", "Creative"])
    complete = table.dropna()
    result = dict(Metric=metric, Treatment="Creative", Control="Standard", Complete_Pairs=len(complete),
                  Incomplete_Pairs=len(table) - len(complete), Mean_Difference=np.nan,
                  CI_Lower=np.nan, CI_Upper=np.nan, Cues=0, Seeds=0,
                  Bootstrap_Draws=draws, Bootstrap_Seed=seed,
                  Scope="crossed cue and base-seed bootstrap; exploratory, uncorrected")
    if complete.empty:
        return result
    differences = (complete.Creative - complete.Standard).groupby(level=[0, 1]).mean().unstack("Base_Seed")
    matrix = differences.to_numpy()
    result.update(Mean_Difference=float(differences.mean(axis=1).mean()), Cues=len(differences), Seeds=matrix.shape[1])
    if len(differences) < 2 or matrix.shape[1] < 2:
        return result
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        cue_indices = rng.integers(0, matrix.shape[0], matrix.shape[0])
        seed_indices = rng.integers(0, matrix.shape[1], matrix.shape[1])
        sample = matrix[np.ix_(cue_indices, seed_indices)]
        valid = np.isfinite(sample).any(axis=1)
        if valid.any():
            values.append(float(np.nanmean(np.nanmean(sample[valid], axis=1))))
    if values:
        result["CI_Lower"], result["CI_Upper"] = np.quantile(values, [.025, .975])
    return result
