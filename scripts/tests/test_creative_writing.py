"""Offline numerical checks for story-token pooling and complete paired extraction."""
import copy
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from creative_writing import load_cues, story_prompt, TASK, PROMPT_VERSION
from experiment import PROTOCOL_VERSION
from interventions import ActivationEngine
from run_creative_writing import aggregate_layer, read_discovery, file_fingerprint, writing_trials, export_vectors
import run_creative_writing as runner


class ResponsePoolingTests(unittest.TestCase):
    def test_pooling_matches_response_positions_for_llama_and_qwen(self):
        """Check real decoder blocks, special-token exclusion and unequal prompt lengths."""
        for config_type, model_type in [(LlamaConfig, LlamaForCausalLM), (Qwen2Config, Qwen2ForCausalLM)]:
            torch.manual_seed(8)
            model = model_type(config_type(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
            engine = ActivationEngine(model, SimpleNamespace(all_special_ids=[0]))
            for prompt in ([1, 2], [1, 2, 3, 4]):
                expected, handles = {}, []
                for layer, block in enumerate(model.model.layers):
                    def observe(_module, _args, output, layer=layer):
                        expected[layer] = output.detach().float()[0, len(prompt):len(prompt)+2].mean(0)
                    handles.append(block.register_forward_hook(observe))
                with torch.inference_mode():
                    model.model(input_ids=torch.tensor([list(prompt) + [5, 6, 0]]), use_cache=False)
                for handle in handles:
                    handle.remove()
                actual = engine.capture_response_means(list(prompt), [5, 6, 0])
                for layer in expected:
                    torch.testing.assert_close(actual[layer], expected[layer])
                self.assertTrue(all(not block._forward_hooks for block in model.model.layers))
            with self.assertRaisesRegex(ValueError, "non-special"):
                engine.capture_response_means([1], [0])
            with patch.object(model.model, "forward", side_effect=RuntimeError("failure")):
                with self.assertRaises(RuntimeError):
                    engine.capture_response_means([1], [2], [0])
            self.assertTrue(all(not block._forward_hooks for block in model.model.layers))

    def test_equal_cue_weighting_and_artifact_compatibility(self):
        """A longer or more repeated cue must not dominate the vector."""
        def pair(creative, standard):
            return {"Creative": {0: torch.tensor(creative)}, "Standard": {0: torch.tensor(standard)}}
        pooled = {("a", 0): pair([6., 2.], [2., 2.]),
                  ("a", 1): pair([10., 2.], [2., 2.]),
                  ("b", 0): pair([2., 4.], [2., 2.])}
        vector, center = aggregate_layer(pooled, 0)
        torch.testing.assert_close(vector, torch.tensor([3., 1.]))
        torch.testing.assert_close(center, torch.tensor([2., 2.]))
        pairs = {key: {"Standard": {"Item": key[0]}} for key in pooled}
        manifest = dict(model="fake", requested_revision="main", resolved_revision=None,
                        chat_template_sha256="fake")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            export_vectors(output, pooled, pairs, manifest, output, False)
            artifact = torch.load(output / "layer_00.pt", weights_only=True)
            self.assertEqual(artifact["protocol"], PROTOCOL_VERSION)
            self.assertEqual(artifact["recommended_scope"], "each_step")
            self.assertIsNone(artifact["head"])
            torch.testing.assert_close(artifact["vector"], vector)


class StoryProtocolTests(unittest.TestCase):
    def test_generation_replay_and_existing_transfer_cli(self):
        """Run both stages with real tiny residuals and controlled completion IDs."""
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        raw = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "user": 2,
            "assistant": 3, "a": 4, "story": 5, "ended": 6}, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]",
            chat_template="{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}assistant")
        torch.manual_seed(12)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, eos_token_id=1)).eval()
        engine = ActivationEngine(model, tokenizer)

        def completion(**kwargs):
            """Return the same story; prompt-conditioned states still differ."""
            return torch.cat([kwargs["input_ids"], torch.tensor([[4, 5, 6, 1]])], dim=1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = load_cues()
            document["data"] = document["data"][:1]
            cues = root / "input_cues.json"
            cues.write_text(json.dumps(document))
            generation = SimpleNamespace(cues=cues, split="discovery", repeats=1, seed=42,
                model="fake", revision=None, output_dir=root / "discovery", dry_run=False,
                temperature=.7, max_tokens=20, top_p=1., top_k=0)
            extraction = SimpleNamespace(run_dir=generation.output_dir, output_dir=root / "vectors",
                layers=[0, 1], allow_truncated=False, dry_run=False)
            with patch("interventions.load_engine", return_value=engine), \
                    patch.object(model, "generate", side_effect=completion), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.generate(generation)
                runner.extract(extraction)
            with (generation.output_dir / "responses.csv").open() as handle:
                stories = list(csv.DictReader(handle))
            self.assertEqual(len(stories), 2)
            self.assertEqual(len(stories[0]), 10)
            self.assertNotIn("Prompt_Token_IDs", stories[0])
            manifest, pairs = read_discovery(generation.output_dir)
            self.assertTrue(manifest["completed"])
            self.assertEqual(len(pairs), 1)
            self.assertEqual(next(iter(pairs.values()))["Creative"]["Response"], "a story ended")
            artifact = root / "vectors/layer_00.pt"
            loaded = torch.load(artifact, weights_only=True)
            self.assertGreater(loaded["vector"].norm().item(), 0)
            import run_residual
            argv = ["activation_steer", "generate", "--model", "fake", "--artifact", str(artifact),
                    "--task", "Alternative Uses Task", "--paraphrases", "1", "--dry-run"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                run_residual.main()

    def test_cues_prompts_and_paired_schedule(self):
        cues = load_cues()["data"]
        self.assertEqual(len(cues), 10)
        self.assertEqual({cue["split"] for cue in cues}, {"discovery", "validation", "test"})
        for cue in cues:
            creative, standard = story_prompt(cue, "Creative"), story_prompt(cue, "Standard")
            for word in cue["items"]:
                self.assertIn(word, creative)
                self.assertIn(word, standard)
            self.assertIn(cue["boring_theme"], creative)
            self.assertNotIn("should not be about", standard)
        trials = writing_trials(cues[:2], 2, 42)
        self.assertEqual(trials, writing_trials(cues[:2], 2, 42))
        blocks = {}
        for _cue, _repeat, condition, block, seed in trials:
            blocks.setdefault(block, []).append((condition, seed))
        for block in blocks.values():
            self.assertEqual({condition for condition, _seed in block}, {"Standard", "Creative"})
            self.assertEqual(block[0][1], block[1][1])

    def test_duplicate_triplet_across_splits_is_rejected(self):
        document = load_cues()
        duplicate = copy.deepcopy(document["data"][0])
        duplicate.update(id="another-id", split="test", items=list(reversed(duplicate["items"])))
        document["data"].append(duplicate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cues.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_cues(path)

    def test_extraction_rejects_missing_pairs_leakage_and_truncation(self):
        document = load_cues()
        document["data"] = document["data"][:1]
        cue = document["data"][0]
        rows = [dict(Condition=condition, Cue_ID=cue["id"], Split="discovery", Model="fake",
                     Prompt=story_prompt(cue, condition), Status="ok", Finish_Reason="stop",
                     Prompt_Token_IDs=[1], Completion_Token_IDs=[2], Repeat=0,
                     Generation_Seed=42) for condition in ("Standard", "Creative")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cues.json").write_text(json.dumps(document))
            manifest = dict(completed=True, split="discovery", prompt_version=PROMPT_VERSION,
                            protocol=PROTOCOL_VERSION, cue_ids=[cue["id"]], repeats=1,
                            planned_responses=2, model="fake", cues_sha256=file_fingerprint(root / "cues.json"))
            def save(data):
                (root / "responses.jsonl").write_text("\n".join(json.dumps(row) for row in data))
                manifest["responses_sha256"] = file_fingerprint(root / "responses.jsonl")
                (root / "manifest.json").write_text(json.dumps(manifest))
            save(rows)
            self.assertEqual(len(read_discovery(root)[1]), 1)
            for mutation, message in [("Split", "Unexpected"), ("Finish_Reason", "Truncated")]:
                modified = copy.deepcopy(rows)
                modified[0][mutation] = "test" if mutation == "Split" else "length"
                save(modified)
                with self.assertRaisesRegex(ValueError, message):
                    read_discovery(root)
                if mutation == "Finish_Reason":
                    self.assertEqual(len(read_discovery(root, True)[1]), 1)
            save(rows[:1])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                read_discovery(root)
            save([rows[0], rows[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_discovery(root)


if __name__ == "__main__":
    unittest.main()
