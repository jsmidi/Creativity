"""AUT proxies, response-level exports, and blinded human-rating sheets."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import canonical_item
from scoring_common import load_responses, parse_numbered_list, descriptive_plot

parse_aut_list = parse_numbered_list


def normalize(embeddings):
    array = np.asarray(embeddings, dtype=float)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def load_aut_normative_database(filepath, model):
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


def evaluate_aut_metrics(eval_df, emb_model, aut_db, threshold=0.75, cluster_eps=0.35):
    from sklearn.cluster import DBSCAN
    frame = eval_df.reset_index(drop=True).copy()
    if frame.empty:
        for column in ["Semantic_Distance_Proxy", "DB_Rarity_Score", "Relative_Novelty_ICF"]:
            frame[column] = pd.Series(dtype=float)
        return frame
    frame["Item_Key"] = frame["Item"].map(canonical_item)
    items = normalize(emb_model.encode(frame["Item_Key"].tolist()))
    ideas = normalize(emb_model.encode(frame["Parsed_Ideas"].tolist()))
    formatted = [f"use {item} as {idea.lower()}" for item, idea in zip(frame["Item_Key"], frame["Parsed_Ideas"])]
    formatted_emb = normalize(emb_model.encode(formatted))
    frame["Semantic_Distance_Proxy"] = 1 - np.sum(items * ideas, axis=1)
    rarities = []
    for i, item in enumerate(frame["Item_Key"]):
        db = aut_db.get(item) if aut_db else None
        if db is None:
            rarities.append(np.nan)
        else:
            matched = (db["embeddings"] @ formatted_emb[i]) >= threshold
            rarities.append(1 - db["counts"][matched].sum() / db["total_responses"])
    frame["DB_Rarity_Score"] = rarities
    frame["Relative_Novelty_ICF"] = np.nan
    # Descriptive, sample-dependent: do not treat as a fixed external novelty measure.
    for _, group in frame.groupby("Item_Key"):
        indices = group.index.to_numpy()
        labels = DBSCAN(eps=cluster_eps, min_samples=1, metric="cosine").fit_predict(ideas[indices])
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        frame.loc[indices, "Relative_Novelty_ICF"] = np.log(len(labels) / counts[inverse])
    frame["DB_Match_Threshold"] = threshold
    frame["DBSCAN_Eps"] = cluster_eps
    return frame


def prepare_ideas(responses):
    frame = responses.copy()
    frame["Parsed_Ideas"] = frame["Response"].apply(parse_aut_list)
    frame["Idea_Count"] = frame["Parsed_Ideas"].map(len)
    frame["Unique_Idea_Count"] = frame["Parsed_Ideas"].map(lambda x: len({s.lower() for s in x}))
    frame["Format_Valid"] = frame["Idea_Count"].eq(10) & frame["Status"].eq("ok")
    ideas = frame[frame["Status"] == "ok"].explode("Parsed_Ideas").dropna(subset=["Parsed_Ideas"]).copy()
    ideas["Idea_Index"] = ideas.groupby("Response_ID").cumcount()
    return frame.drop(columns="Parsed_Ideas"), ideas.reset_index(drop=True)


def export_ratings(ideas, output_dir, seed=42):
    frame = ideas.copy()
    frame["Rating_ID"] = [hashlib.sha256(f"{rid}:{index}".encode()).hexdigest()[:20]
                          for rid, index in zip(frame["Response_ID"], frame["Idea_Index"])]
    blinded = frame[["Rating_ID", "Item", "Parsed_Ideas"]].sample(frac=1, random_state=seed)
    for column in ["Rater_ID", "Originality_1_to_5", "Usefulness_1_to_5", "Valid_Use_0_or_1", "Notes"]:
        blinded[column] = ""
    blinded.to_csv(output_dir / "aut_blinded_ratings.csv", index=False)
    frame[["Rating_ID", "Response_ID", "Idea_Index", "Model", "Condition"]].to_csv(output_dir / "aut_rating_key_private.csv", index=False)


def main():
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
        responses.to_csv(args.output_dir / "aut_response_audit.csv", index=False)
        return
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    database = load_aut_normative_database(args.norm_db, model)
    scored = evaluate_aut_metrics(ideas, model, database, args.match_threshold, args.cluster_eps)
    metrics = ["Semantic_Distance_Proxy", "DB_Rarity_Score", "Relative_Novelty_ICF"]
    means = scored.groupby("Response_ID")[metrics].mean()
    responses = responses.merge(means, on="Response_ID", how="left", validate="one_to_one")
    responses["Embedding_Model"] = "all-mpnet-base-v2"
    responses["DB_Match_Threshold"] = args.match_threshold
    responses["DBSCAN_Eps"] = args.cluster_eps
    responses["Norm_DB"] = str(args.norm_db.resolve())
    responses["Norm_DB_SHA256"] = hashlib.sha256(args.norm_db.read_bytes()).hexdigest() if args.norm_db.exists() else ""
    scored.to_csv(args.output_dir / "aut_ideas.csv", index=False)
    responses.to_csv(args.output_dir / "aut_responses.csv", index=False)
    descriptive_plot(responses, {metric: metric for metric in metrics}, args.output_dir / "aut_results.png")
    print(f"Saved {len(responses)} responses and {len(scored)} ideas to {args.output_dir}.")


if __name__ == "__main__":
    main()
