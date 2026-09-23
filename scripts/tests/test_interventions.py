"""Real tiny random Llama on CPU; no model weights or network required."""
from pathlib import Path
import sys
import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interventions import ActivationEngine, changed_activation, hidden_tensor, replace_hidden


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, eos_token_id=None, pad_token_id=0)).eval()
        cls.engine = ActivationEngine(cls.model)
        cls.inputs = dict(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long))

    def test_capture_shapes_and_actual_head(self):
        residual = self.engine.capture(self.inputs, 0)
        head = self.engine.capture(self.inputs, 0, 2)
        self.assertEqual(residual.shape, (16,))
        self.assertEqual(head.shape, (4,))

    def test_chat_generation_returns_only_completion(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        raw = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "user": 2, "assistant": 3, "hello": 4}, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]",
            chat_template="{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}assistant")
        engine = ActivationEngine(self.model, tokenizer)
        inputs = engine.inputs("hello")
        self.assertEqual(inputs["input_ids"].tolist(), [[2, 4, 3]])
        baseline = engine.generate("hello", temperature=0, max_tokens=3)
        zero = engine.generate("hello", temperature=0, max_tokens=3, layer=0, vector=torch.ones(16), alpha=0)
        self.assertEqual(baseline["Response"], zero["Response"])
        self.assertEqual(zero["Generated_Tokens"], 3)
        self.assertEqual(zero["Intervention_Calls"], 3)
        self.assertEqual(zero["Prompt_Tokens"], 3)

    def test_tuple_and_tensor_outputs(self):
        h = torch.ones(1, 2, 3)
        self.assertIs(hidden_tensor(h), h)
        result = replace_hidden((h, "cache"), h * 2)
        self.assertEqual(result[1], "cache")

    def test_suppression_center_and_other_tokens(self):
        h = torch.tensor([[[2., 3.], [4., 5.]]])
        changed = changed_activation(h, torch.tensor([2., 0.]), "suppress", center=torch.tensor([1., 0.]))
        torch.testing.assert_close(changed, torch.tensor([[[2., 3.], [1., 5.]]]))
        torch.testing.assert_close(h, torch.tensor([[[2., 3.], [4., 5.]]]))

    def test_zero_alpha_reproduces_logits_and_nonzero_changes_them(self):
        with torch.inference_mode():
            original = self.model(**self.inputs).logits
            with self.engine.intervene(0, torch.arange(16).float(), alpha=0):
                zero = self.model(**self.inputs).logits
            with self.engine.intervene(0, torch.arange(16).float(), alpha=1):
                changed = self.model(**self.inputs).logits
        torch.testing.assert_close(original, zero, rtol=0, atol=0)
        self.assertFalse(torch.allclose(original[:, -1], changed[:, -1]))

    def test_generation_scope_and_cleanup(self):
        for scope, expected in [("prefill", 1), ("each_step", 4)]:
            with self.engine.intervene(0, torch.ones(16), scope=scope) as stats:
                self.model.generate(**self.inputs, max_new_tokens=4, do_sample=False)
            self.assertEqual(stats["forward_calls"], 4)
            self.assertEqual(stats["interventions"], expected)
        self.assertEqual(len(self.model.model.layers[0]._forward_hooks), 0)

    def test_only_selected_head_is_changed(self):
        projection = self.model.model.layers[0].self_attn.o_proj
        seen = []
        def observe(_module, args):
            seen.append(args[0].detach().clone())
        before = projection.register_forward_pre_hook(observe)
        with self.engine.intervene(0, torch.ones(4), head=2):
            after = projection.register_forward_pre_hook(observe)
            try:
                self.model(**self.inputs)
            finally:
                after.remove()
        before.remove()
        delta = seen[1] - seen[0]
        expected = torch.zeros_like(delta)
        expected[:, -1, 8:12] = 1
        torch.testing.assert_close(delta, expected)

    def test_hook_removed_after_exception(self):
        with self.assertRaises(RuntimeError):
            with self.engine.intervene(0, torch.ones(16)):
                raise RuntimeError("test")
        self.assertEqual(len(self.model.model.layers[0]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
