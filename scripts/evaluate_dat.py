"""DAT scoring: official GloVe scorer or explicitly named MPNet proxy."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scoring_common import load_responses, parse_dat_words, parse_numbered_list, descriptive_plot


def load_dat_frequency_database(filepath):
    if filepath is None or not Path(filepath).exists():
        return None
    frame = pd.read_csv(filepath)
    values = pd.to_numeric(frame["Frequency"], errors="raise")
    if not values.between(0, 1).all():
        raise ValueError("Frequency must contain probabilities between 0 and 1, not raw counts.")
    return dict(zip(frame["Word"].str.lower(), values))


def evaluate_dat_metrics(df, emb_model=None, dat_db=None, official_model=None):
    records = []
    for _, row in df.iterrows():
        words = list(dict.fromkeys(parse_dat_words(row["Response"])))
        if official_model is not None:
            # Use the authors' dictionary/embedding validation and deduplication.
            words = list(dict.fromkeys(w for word in words if (w := official_model.validate(word))))
        valid = row.get("Status", "ok") == "ok" and len(words) >= 7
        selected = words[:7]
        official, proxy, rarity = np.nan, np.nan, np.nan
        if valid:
            if official_model is not None:
                score = official_model.dat(selected)
                official = np.nan if score is None else float(score)
            if emb_model is not None:
                emb = np.asarray(emb_model.encode(selected), dtype=float)
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                similarities = emb @ emb.T
                proxy = 100 * (1 - similarities[np.triu_indices(7, 1)].mean())
            if dat_db is not None and all(w in dat_db for w in selected):
                rarity = float(np.mean([1 - dat_db[w] for w in selected]))
        records.append(dict(Parsed_Words=words, Valid_Word_Count=len(words),
                            List_Count=len(parse_numbered_list(row["Response"])),
                            Scorable=valid, Validation="dictionary" if official_model else "syntax_only",
                            DAT_GloVe_Score=official, DAT_MPNet_Proxy=proxy, Rarity_Score=rarity,
                            Frequency_Coverage=sum(w in (dat_db or {}) for w in selected) / 7))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=[Path("outputs")])
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/dat_v2"))
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--scorer", choices=["official", "mpnet"], default="official")
    parser.add_argument("--official-code", type=Path, help="Authors' dat.py (github.com/jayolson/divergent-association-task)")
    parser.add_argument("--glove", type=Path, help="glove.840B.300d.txt")
    parser.add_argument("--dictionary", type=Path, help="Authors' words.txt")
    parser.add_argument("--frequency-db", type=Path)
    args = parser.parse_args()
    model, official = None, None
    if args.scorer == "official":
        if not all(p and p.is_file() for p in [args.official_code, args.glove, args.dictionary]):
            parser.error("Official scoring requires --official-code, --glove, --dictionary. For exploratory scoring choose --scorer mpnet.")
    frame = load_responses(args.inputs, "Divergent Association Task", args.include_legacy)
    if args.scorer == "official":
        spec = importlib.util.spec_from_file_location("olson_dat", args.official_code)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        official = module.Model(str(args.glove), str(args.dictionary))
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")
    scored = evaluate_dat_metrics(frame, model, load_dat_frequency_database(args.frequency_db), official)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "dat_responses.csv", index=False)
    column = "DAT_GloVe_Score" if official is not None else "DAT_MPNet_Proxy"
    descriptive_plot(scored, {column: column, "Rarity_Score": "Lexical rarity (missing if unavailable)"}, args.output_dir / "dat_results.png")
    print(f"Saved {len(scored)} responses, {scored['Scorable'].sum()} scorable; {args.output_dir}")


if __name__ == "__main__":
    main()
