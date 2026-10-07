"""Ensure GPT-OSS reasoning cannot enter behavioral scoring."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_generation import final_answer, gpt_oss_memory_budget


class FinalAnswerTests(unittest.TestCase):
    def test_two_40gb_gpus(self):
        with patch('torch.cuda.device_count', return_value=2), patch(
                'torch.cuda.mem_get_info', return_value=(40 * 1024**3, 40 * 1024**3)):
            self.assertEqual(gpt_oss_memory_budget(), {0: '32GiB', 1: '32GiB'})

    def test_one_40gb_gpu_rejected(self):
        with patch('torch.cuda.device_count', return_value=1), patch(
                'torch.cuda.mem_get_info', return_value=(40 * 1024**3, 40 * 1024**3)):
            with self.assertRaisesRegex(RuntimeError, '--gpus=2'):
                gpt_oss_memory_budget()

    def test_one_80gb_gpu(self):
        with patch('torch.cuda.device_count', return_value=1), patch(
                'torch.cuda.mem_get_info', return_value=(80 * 1024**3, 80 * 1024**3)):
            self.assertEqual(gpt_oss_memory_budget(), {0: '72GiB'})

    def test_reasoning_is_excluded(self):
        raw = ('<|channel|>analysis<|message|>1. reasoning example'
               '<|end|><|start|>assistant<|channel|>final<|message|>'
               '1. violin\n2. glacier<|fim_suffix|>')
        self.assertEqual(final_answer(raw), '1. violin\n2. glacier')

    def test_truncated_reasoning_is_not_an_answer(self):
        self.assertEqual(final_answer('<|channel|>analysis<|message|>1. violin'), '')

    def test_partial_final_is_retained_for_format_audit(self):
        self.assertEqual(final_answer('<|channel|>final<|message|>1. violin'), '1. violin')

    def test_return_token_is_removed(self):
        self.assertEqual(final_answer('<|channel|>final<|message|>1. violin<|return|>'), '1. violin')


if __name__ == '__main__':
    unittest.main()
