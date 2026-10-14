"""AUT proxies, response-level exports, and blinded human-rating sheets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import canonical_item
from scoring_common import load_responses, parse_numbered_list, descriptive_plot

parse_aut_list = parse_numbered_list


def normalize(embeddings):
    """Convert embedding rows to unit vectors, keeping zero rows finite."""
    array = np.asarray(embeddings, dtype=float)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def load_aut_normative_database(filepath, model):
    """Embed per-item reference uses and retain their response-frequency weights; missing norms return an empty mapping."""
    if not Path(filepath).exists():
        print("Normative database unavailable; rarity will be missing.")
        return {}
    frame = pd.read_csv(filepath)
    frame["Item_Key"] = frame["prompt"].map(canonical_item)
    database = {}
    for item, group in frame.groupby("Item_Key"):
        counts = pd.to_numeric(group["n_duplicates"], errors="raise").to_numpy(float)
        if not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
            raise ValueError(f"Invalid normative counts for {item}")
        database[item] = dict(embeddings=normalize(model.encode(group["final_response"].astype(str).tolist())),
                              counts=counts, total_responses=counts.sum())
    return database


def database_rarities(frame, formatted_emb, aut_db, threshold):
    """Return frequency-weighted rarity for each reference-formatted idea; unmatched items stay missing."""
    rarities = []
    for i, item in enumerate(frame["Item_Key"]):
        db = aut_db.get(item) if aut_db else None
        if db is None:
            rarities.append(np.nan)
        else:
            matched = (db["embeddings"] @ formatted_emb[i]) >= threshold
            rarities.append(1 - db["counts"][matched].sum() / db["total_responses"])
    return rarities


def add_cluster_novelty(frame, ideas, cluster_eps):
    """Assign within-item ICF novelty from cosine DBSCAN clusters fitted to this sample."""
    from sklearn.cluster import DBSCAN
    frame["Relative_Novelty_ICF"] = np.nan
    # Descriptive, sample-dependent: do not treat as a fixed external novelty measure.
    for _, group in frame.groupby("Item_Key"):
        indices = group.index.to_numpy()
        labels = DBSCAN(eps=cluster_eps, min_samples=1, metric="cosine").fit_predict(ideas[indices])
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        frame.loc[indices, "Relative_Novelty_ICF"] = np.log(len(labels) / counts[inverse])
    return frame


def evaluate_aut_metrics(eval_df, emb_model, aut_db, threshold=0.75, cluster_eps=0.35):
    """Score ideas by item distance, within-response pairwise distance and rarity.

    Pairwise distance is mean cosine distance to the other ideas in the same response.
    Averaging these idea scores gives the mean over all unordered pairs, excluding self-pairs.
    Responses with fewer than two ideas have missing pairwise scores; duplicates are retained.
    ICF is log(total ideas / cluster size), fitted to this scoring sample; it is not an external novelty norm."""
    frame = eval_df.reset_index(drop=True).copy()
    if frame.empty:
        for column in ["Pairwise_Semantic_Distance_Proxy", "Semantic_Distance_Proxy", "DB_Rarity_Score", "Relative_Novelty_ICF"]:
            frame[column] = pd.Series(dtype=float)
        return frame
    frame["Item_Key"] = frame["Item"].map(canonical_item)
    items = normalize(emb_model.encode(frame["Item_Key"].tolist()))
    ideas = normalize(emb_model.encode(frame["Parsed_Ideas"].tolist()))
    formatted = [f"use {item} as {idea.lower()}" for item, idea in zip(frame["Item_Key"], frame["Parsed_Ideas"])]
    formatted_emb = normalize(emb_model.encode(formatted))
    frame["Semantic_Distance_Proxy"] = 1 - np.sum(items * ideas, axis=1)
    frame["Pairwise_Semantic_Distance_Proxy"] = np.nan
    for _, group in frame.groupby("Response_ID"):
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue
        distances = np.clip(1 - ideas[indices] @ ideas[indices].T, 0, 2)
        np.fill_diagonal(distances, 0)
        frame.loc[indices, "Pairwise_Semantic_Distance_Proxy"] = distances.sum(axis=1) / (len(indices) - 1)
    rarities = database_rarities(frame, formatted_emb, aut_db, threshold)
    frame["DB_Rarity_Score"] = rarities
    frame = add_cluster_novelty(frame, ideas, cluster_eps)
    return frame


def prepare_ideas(responses):
    """Return response audits and one row per parsed idea; retain malformed lists for explicit format review."""
    frame = responses.copy()
    frame["Parsed_Ideas"] = frame["Response"].apply(parse_aut_list)
    frame["Idea_Count"] = frame["Parsed_Ideas"].map(len)
    frame["Unique_Idea_Count"] = frame["Parsed_Ideas"].map(lambda x: len({s.lower() for s in x}))
    frame["Format_Valid"] = frame["Idea_Count"].eq(10) & frame["Status"].eq("ok")
    ideas = frame[frame["Status"] == "ok"].explode("Parsed_Ideas").dropna(subset=["Parsed_Ideas"]).copy()
    ideas["Idea_Index"] = ideas.groupby("Response_ID").cumcount()
    return frame.drop(columns="Parsed_Ideas"), ideas.reset_index(drop=True)


def export_ratings(ideas, output_dir, seed=42):
    """Write shuffled blinded rating rows and a separate private key linking them to conditions."""
    frame = ideas.copy()
    frame["Rating_ID"] = [hashlib.sha256(f"{rid}:{index}".encode()).hexdigest()[:20]
                          for rid, index in zip(frame["Response_ID"], frame["Idea_Index"])]
    blinded = frame[["Rating_ID", "Item", "Parsed_Ideas"]].sample(frac=1, random_state=seed)
    for column in ["Rater_ID", "Originality_1_to_5", "Usefulness_1_to_5", "Valid_Use_0_or_1", "Notes"]:
        blinded[column] = ""
    blinded.to_csv(output_dir / "aut_blinded_ratings.csv", index=False)
    frame[["Rating_ID", "Response_ID", "Idea_Index", "Model", "Condition"]].to_csv(output_dir / "aut_rating_key_private.csv", index=False)


def main():
    """Score AUT ideas, average metrics within each response, and save audits, ratings and plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=[Path("outputs")])
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/aut_v2"))
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--norm-db", type=Path, default=Path(__file__).resolve().parents[1] / "aut_quality_scored_all.csv")
    parser.add_argument("--match-threshold", type=float, default=0.75)
    parser.add_argument("--cluster-eps", type=float, default=0.35)
    parser.add_argument("--ratings-only", action="store_true", help="Export blinded sheets without downloading/loading an embedding model")
    args = parser.parse_args()
    if not 0 <= args.match_threshold <= 1 or not 0 < args.cluster_eps <= 2:
        parser.error("Require match-threshold in [0,1] and cluster-eps in (0,2].")
    responses = load_responses(args.inputs, "Alternative Uses Task", args.include_legacy)
    responses, ideas = prepare_ideas(responses)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_ratings(ideas, args.output_dir)
    if args.ratings_only:
        responses.drop(columns=["Protocol", "Source_File"], errors="ignore").to_csv(
            args.output_dir / "aut_response_audit.csv", index=False)
        return
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    database = load_aut_normative_database(args.norm_db, model)
    scored = evaluate_aut_metrics(ideas, model, database, args.match_threshold, args.cluster_eps)
    metrics = ["Pairwise_Semantic_Distance_Proxy", "Semantic_Distance_Proxy", "DB_Rarity_Score", "Relative_Novelty_ICF"]
    means = scored.groupby("Response_ID")[metrics].mean()
    responses = responses.merge(means, on="Response_ID", how="left", validate="one_to_one")
    manifest = dict(embedding_model="all-mpnet-base-v2", match_threshold=args.match_threshold,
                    primary_metric="Pairwise_Semantic_Distance_Proxy",
                    pairwise_distance="Mean cosine distance over unordered idea pairs within each Response_ID; duplicates retained; fewer than two ideas is missing",
                    cluster_eps=args.cluster_eps, norm_db=str(args.norm_db.resolve()),
                    source_files=responses.Source_File.unique().tolist())
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    scored = scored.drop(columns=["Response", "Protocol", "Source_File"], errors="ignore")
    responses = responses.drop(columns=["Protocol", "Source_File"], errors="ignore")
    scored.to_csv(args.output_dir / "aut_ideas.csv", index=False)
    responses.to_csv(args.output_dir / "aut_responses.csv", index=False)
    descriptive_plot(responses, {metric: metric for metric in metrics}, args.output_dir / "aut_results.png")
    print(f"Saved {len(responses)} responses and {len(scored)} ideas to {args.output_dir}.")


if __name__ == "__main__":
    main()
