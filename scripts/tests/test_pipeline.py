"""Offline regressions: python -m unittest discover -s scripts/tests -v"""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import canonical_item, trial_schedule
from scoring_common import load_responses, parse_dat_words
from evaluate_dat import evaluate_dat_metrics, load_dat_frequency_database
from evaluate_aut import prepare_ideas, evaluate_aut_metrics
from activation_steer import validate_holdout
from generate import query_model
from analyze_effects import paired_effect


class FakeEmbedding:
    def encode(self, texts):
        return np.array([[len(text) + 1, sum(map(ord, text)) % 17 + 1, 1] for text in texts], dtype=float)


class PipelineTests(unittest.TestCase):
    def test_effect_pairs_and_item_weighting(self):
        rows = []
        for item, count, effect in [("book", 3, 1), ("can", 1, 3)]:
            for repeat in range(count):
                for condition, score in [("Standard", 0), ("Creative", effect)]:
                    rows.append(dict(Run_ID="run", Block_ID=f"{item}{repeat}",
                                     Response_ID=f"{item}{repeat}{condition}", Item=item,
                                     Condition=condition, Score=score))
        frame = pd.DataFrame(rows)
        result = paired_effect(frame, "Score", "Creative", "Standard", draws=100)
        self.assertEqual(result["Mean_Difference"], 2)
        self.assertEqual(result["Complete_Pairs"], 4)
        frame.loc[0, "Score"] = np.nan
        result = paired_effect(frame, "Score", "Creative", "Standard", draws=100)
        self.assertEqual(result["Incomplete_Pairs"], 1)

    def test_schedule_replicates_and_seed_blocks(self):
        args = (["book", "can"], ["Standard", "Creative", "Conventional"], 4, [0, 1], 42)
        schedule = trial_schedule(*args)
        self.assertEqual(len(schedule), 48)
        self.assertEqual(schedule, trial_schedule(*args))
        for block in {row["Block_ID"] for row in schedule}:
            self.assertEqual(len({r["Generation_Seed"] for r in schedule if r["Block_ID"] == block}), 1)
        self.assertNotEqual(schedule, trial_schedule(*args[:-1], 43))

    def test_canonical_items(self):
        self.assertEqual(canonical_item("A tin can"), "can")
        self.assertEqual(canonical_item("  a BOOK "), "book")

    def test_dat_does_not_truncate_phrases(self):
        self.assertEqual(parse_dat_words("1. blue sky\n2. spoon\n3. **violin**"), ["spoon", "violin"])

    def test_dat_requires_seven_unique_words_and_missing_is_nan(self):
        text = "\n".join(f"{i+1}. {word}" for i, word in enumerate(["cat", "dog", "cat", "spoon", "violin", "cloud", "book"]))
        rows = pd.DataFrame([dict(Response=text, Status="ok"), dict(Response=text + "\n8. democracy", Status="ok")])
        result = evaluate_dat_metrics(rows, FakeEmbedding())
        self.assertTrue(np.isnan(result.loc[0, "DAT_MPNet_Proxy"]))
        self.assertTrue(np.isfinite(result.loc[1, "DAT_MPNet_Proxy"]))
        self.assertTrue(result["Rarity_Score"].isna().all())
        self.assertTrue(result["DAT_GloVe_Score"].isna().all())

    def test_frequency_counts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freq.csv"
            path.write_text("Word,Frequency\ncat,500\n")
            with self.assertRaises(ValueError):
                load_dat_frequency_database(path)

    def test_aut_response_and_failure_retention(self):
        rows = pd.DataFrame([dict(Response_ID="a", Response="1. Doorstop\n2. Weight", Status="ok", Item="a book"),
                             dict(Response_ID="b", Response="", Status="error", Item="can")])
        responses, ideas = prepare_ideas(rows)
        self.assertEqual(len(responses), 2)
        self.assertEqual(len(ideas), 2)
        scored = evaluate_aut_metrics(ideas, FakeEmbedding(), {})
        self.assertTrue(scored["DB_Rarity_Score"].isna().all())
        self.assertEqual(set(scored["Item_Key"]), {"book"})

    def test_holdout_rejects_item_and_wording_leakage(self):
        artifact = dict(examples=[dict(Task="Alternative Uses Task", Item="book", Paraphrase=0)])
        with self.assertRaises(ValueError):
            validate_holdout(artifact, "Alternative Uses Task", ["a book"], [1], "test")
        with self.assertRaises(ValueError):
            validate_holdout(artifact, "Divergent Association Task", [""], [0], "test")
        validate_holdout(artifact, "Divergent Association Task", [""], [2], "test")

    def test_legacy_opt_in_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            pd.DataFrame([dict(Task="Alternative Uses Task", Response="1. weight", Model="test",
                               Condition="Standard", Item="book", Response_ID="same")]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_responses([path], "Alternative Uses Task")
            self.assertEqual(len(load_responses([path], "Alternative Uses Task", True)), 1)
            (Path(directory) / "copy.csv").write_bytes(path.read_bytes())
            with self.assertRaises(ValueError):
                load_responses([Path(directory)], "Alternative Uses Task", True)

    def test_api_failure_is_structured(self):
        class Client:
            def __init__(self):
                self.chat = self.completions = self
            def create(self, **kwargs):
                raise ValueError("request failed")
        result = query_model(Client(), "prompt", "model")
        self.assertEqual(result["Status"], "error")
        self.assertEqual(result["Response"], "")


if __name__ == "__main__":
    unittest.main()
