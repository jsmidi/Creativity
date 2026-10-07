"""Task splits and all-greedy intervention schedules, without model downloads."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from activation_steer import main as steer_main, validate_holdout
from concept_vectors import stimuli
from run_greedy_task import NAMES, DISCOVERY, AUDIT
from task_config import get_task_config


class GreedyTaskTests(unittest.TestCase):
    def test_task_specific_prompts_and_disjoint_splits(self):
        for key in ('aut', 'cdat', 'drat'):
            task = NAMES[key]
            examples = stimuli(DISCOVERY[key], [1, 2], task=task)
            self.assertEqual(len(examples), 48)
            self.assertIn(DISCOVERY[key][0], examples[0]['Prompt'])
            self.assertEqual({e['Format'] for e in examples}, {'numbered', 'bullets'})
            if key != 'aut':
                self.assertNotIn('uses for the object', examples[0]['Prompt'])
            artifact = dict(examples=[dict(Task=task, Item=i, Paraphrase=p) for i in DISCOVERY[key] for p in [1, 2]],
                            audit_examples=[dict(Task=task, Item=i, Paraphrase=0) for i in AUDIT[key]])
            validate_holdout(artifact, task, get_task_config(task)['items'], [0], 'validation')

    def test_drat_anchor_words_are_held_out(self):
        groups = [DISCOVERY['drat'], AUDIT['drat'], get_task_config(NAMES['drat'])['items']]
        words = [{word.strip() for item in group for word in item.split('|')} for group in groups]
        for i in range(3):
            for j in range(i):
                self.assertFalse(words[i] & words[j])

    def test_fifteen_greedy_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)/'vector.pt'
            from experiment import PROTOCOL_VERSION
            torch.save(dict(protocol=PROTOCOL_VERSION, model='local-test',
                            examples=[dict(Task=NAMES['aut'], Item='brick', Paraphrase=1)]), artifact)
            argv = ['steer', 'generate', '--model', 'local-test', '--artifact', str(artifact),
                    '--task', NAMES['aut'], '--items', 'book', '--paraphrases', '0',
                    '--repeats', '1', '--alphas', '1', '--random-vectors', '2',
                    '--temperature', '0', '--omit-temperature-control', '--dry-run']
            output = io.StringIO()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output):
                steer_main()
            self.assertIn('15 generations', output.getvalue())


if __name__ == '__main__':
    unittest.main()
