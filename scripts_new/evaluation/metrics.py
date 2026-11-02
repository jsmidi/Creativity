"""Small, embedding-based metrics for the four pilot psychometric tasks."""

import re

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def parse_numbered_list(text):
    """Extract numbered lines, preserving their original order."""
    if not isinstance(text, str):
        return []
    return [value.strip().strip("*").strip() for value in
            re.findall(r"(?:^|\n)\s*(?:\*\*)?\d+[.)](?:\*\*)?\s+([^\n]+)", text)]


def parse_nouns(text, limit=10):
    """Return unique single-word entries among the first ``limit`` list items."""
    words = []
    for entry in parse_numbered_list(text)[:limit]:
        word = entry.strip().lower()
        if re.fullmatch(r"[a-z]+(?:-[a-z]+)*", word) and word not in words:
            words.append(word)
    return words


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def pairwise_distance(vectors, scale=1.0):
    """Mean cosine distance over unordered pairs; missing with fewer than two items."""
    vectors = normalize(vectors)
    if len(vectors) < 2:
        return float("nan")
    similarities = np.clip(vectors @ vectors.T, -1, 1)
    return float(scale * (1 - similarities[np.triu_indices(len(vectors), 1)]).mean())


def cue_appropriateness(word_vectors, cue_vector):
    """Mean 100 * (1 + cosine) to the cue over the response's valid words."""
    words = normalize(word_vectors)
    cue = normalize(np.asarray(cue_vector).reshape(1, -1))[0]
    if len(words) == 0:
        return float("nan")
    return float(100 * (1 + np.clip(words @ cue, -1, 1)).mean())


def _bh_adjust(pvalues):
    """Benjamini–Hochberg adjusted p-values, preserving missing values."""
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    if len(order):
        raw = values[order] * len(order) / np.arange(1, len(order) + 1)
        adjusted[order] = np.minimum(1, np.minimum.accumulate(raw[::-1])[::-1])
    return adjusted


def cdat_appropriateness_gate(frame, model, nouns, seed, draws=500):
    """Gate each model/condition using random seven-noun lists per cue.

    Tests cue-level response appropriateness against a calibrated random noun
    baseline, applies BH correction, and only reports a CDAT score for passing
    groups.
    """
    replicate_column = next((column for column in ("RUN_ID", "Run_ID", "Model_Seed")
                             if column in frame), None)
    group_columns = ["Model"] + (["Temperature"] if "Temperature" in frame else [])
    if replicate_column:
        group_columns.append(replicate_column)
    group_columns.append("Condition")
    cues = sorted(frame.Item.astype(str).unique())
    noun_vectors = normalize(model.encode(nouns, convert_to_numpy=True, show_progress_bar=True))
    cue_vectors = dict(zip(cues, model.encode(cues, convert_to_numpy=True)))
    rng = np.random.default_rng(seed)
    baseline = {}
    for cue in cues:
        values = [cue_appropriateness(noun_vectors[rng.choice(len(nouns), 7, replace=False)],
                                      cue_vectors[cue]) for _ in range(draws)]
        baseline[cue] = np.asarray(values)

    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        cue_means = group.groupby("Item")["CDAT_Appropriateness"].mean().dropna()
        random_values = (np.concatenate([baseline[str(cue)] for cue in cue_means.index])
                         if len(cue_means) else np.array([]))
        pvalue = (float(ttest_ind(cue_means.to_numpy(), random_values, equal_var=False).pvalue)
                  if len(cue_means) >= 2 and len(random_values) else np.nan)
        row = dict(zip(group_columns, keys))
        row.update(Cues=int(len(cue_means)),
                   Mean_Appropriateness=float(cue_means.mean()) if len(cue_means) else np.nan,
                   Random_Appropriateness=float(random_values.mean()) if len(random_values) else np.nan,
                   Mean_Novelty=float(group.groupby("Item")["CDAT_Novelty"].mean().mean()), P=pvalue)
        rows.append(row)
    gates = pd.DataFrame(rows)
    if "Temperature" in gates:
        gates["FDR_P"] = gates.groupby("Temperature", dropna=False).P.transform(
            lambda values: _bh_adjust(values.to_numpy()))
    else:
        gates["FDR_P"] = _bh_adjust(gates.P.to_numpy())
    gates["Gate_Passed"] = gates.FDR_P.lt(.001) & gates.Mean_Appropriateness.gt(gates.Random_Appropriateness)
    gates["CDAT_Score"] = gates.Mean_Novelty.where(gates.Gate_Passed)
    return gates


def _response_audit(response):
    entries = parse_numbered_list(response)
    words = parse_nouns(response)
    return entries, words


def clean_aut_idea(entry, item):
    """Remove a repeated AUT target and common list-answer framing before embedding."""
    idea = str(entry).lower().strip()
    target = str(item).lower().strip() if item is not None else ""
    if target:
        idea = re.sub(rf"(?<!\w){re.escape(target)}(?!\w)", " ", idea)

    # Responses often repeat the prompt object and use a shared frame such as
    # "Use a book as ..." or "Book as ...". Remove that boilerplate so the
    # embedding represents the proposed use rather than the prompt template.
    idea = re.sub(
        r"^\s*(?:use|using|reuse|reusing|repurpose|repurposing|turn|turning|"
        r"transform|transforming|convert|converting|make|making)\b", " ", idea)
    idea = re.sub(r"^\s*(?:(?:a|an|the)\s+)+", " ", idea)
    idea = re.sub(r"^\s*(?:(?:as|into|for|to)\s+)+", " ", idea)
    idea = re.sub(r"^\s*(?:(?:a|an|the)\s+)+", " ", idea)
    idea = re.sub(r"\s+", " ", idea).strip(" \t\n.,;:!?—–-\"'`()[]{}")
    return idea


def score_aut(response, item, embed):
    """Score one AUT response for within-list diversity and distance from its item."""
    entries, words = _response_audit(response)
    ideas = [clean_aut_idea(entry, item) for entry in entries if entry.strip()]
    ideas = [idea for idea in ideas if idea]
    result = {"List_Count": len(entries), "Valid_Word_Count": len(words),
              "Format_Valid": len(entries) == 10}
    if not ideas:
        result.update(Semantic_Distance_Proxy=float("nan"),
                      Pairwise_Semantic_Distance_Proxy=float("nan"))
        return result
    vectors = normalize(embed(ideas))
    result["Semantic_Distance_Proxy"] = (float(1 - np.mean(vectors @ normalize(embed([item]))[0]))
                                         if item else float("nan"))
    result["Pairwise_Semantic_Distance_Proxy"] = pairwise_distance(vectors)
    return result


def score_dat(response, embed, dat_model):
    """Score one DAT response with Olson GloVe and the optional embedding proxy."""
    entries, words = _response_audit(response)
    official = list(dict.fromkeys(w for word in words if (w := dat_model.validate(word))))[:7]
    selected = words[:7]
    return {"List_Count": len(entries), "Valid_Word_Count": len(words),
            "Format_Valid": len(entries) == 10 and len(words) == 10,
            "DAT_GloVe_Score": dat_model.dat(official) if len(official) == 7 else float("nan"),
            "DAT_MPNet_Proxy": pairwise_distance(embed(selected), scale=100)
            if len(selected) == 7 else float("nan")}


def score_cdat(response, cue, embed):
    """Score one CDAT response for first-seven novelty and cue appropriateness."""
    entries, words = _response_audit(response)
    vectors = embed(words[:7]) if len(words) >= 7 else []
    return {"List_Count": len(entries), "Valid_Word_Count": len(words),
            "Format_Valid": len(entries) == 10 and len(words) == 10,
            "CDAT_Novelty": pairwise_distance(vectors, scale=100),
            "CDAT_Appropriateness": (cue_appropriateness(vectors, embed([cue])[0])
                                     if len(words) >= 7 and cue else float("nan"))}


def score_drat(response, anchors_text, embed, noun_pool):
    """Score one DRAT response using its max-anchor calibrated relevance gate."""
    entries, words = _response_audit(response)
    anchors = [part.strip() for part in anchors_text.split("|") if part.strip()]
    result = {"List_Count": len(entries), "Valid_Word_Count": len(words),
              "Format_Valid": len(entries) == 10 and len(words) == 10,
              "DRAT_Score": float("nan")}
    if words and len(anchors) >= 2 and noun_pool is not None:
        anchor_vectors = normalize(embed(anchors))
        pool = normalize(noun_pool)
        threshold = np.quantile(np.clip(pool @ anchor_vectors.T, -1, 1).max(axis=1), .9)
        word_vectors = normalize(embed(words))
        utility = np.clip(word_vectors @ anchor_vectors.T, -1, 1).max(axis=1)
        survivors = word_vectors[utility > threshold]
        result["DRAT_Score"] = pairwise_distance(survivors, scale=100) if len(survivors) >= 3 else 0.0
        result["DRAT_Utility_Threshold"] = float(threshold)
        result["DRAT_Survivors"] = int(len(survivors))
    return result
