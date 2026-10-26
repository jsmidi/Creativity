"""Task splits and all-greedy intervention schedules, without model downloads."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_residual import main as steer_main, validate_holdout, intervention_arms
from analyze_effects import intervention_effects
from run_attention_heads import stimuli
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

    def test_shared_comparisons_keep_random_controls_and_missing_scores(self):
        args = Namespace(alphas=[1.0], random_vectors=2, omit_temperature_control=True)
        labels = [arm[0] for arm in intervention_arms(args)]
        frame = pd.DataFrame([
            dict(Run_ID="one", Response_ID=label, Item="book", Block_ID="book|0",
                 Condition=label, Temperature=0, Score=float(index))
            for index, label in enumerate(labels)
        ])
        effects = intervention_effects(frame, "Score")
        self.assertEqual(len(effects), 18)
        self.assertTrue(effects.Analysis_Status.eq("ok").all())
        self.assertTrue(effects.CI_Lower.isna().all())
        self.assertIn("Standard_random0_1", set(effects.Control))
        frame.loc[frame.Condition.eq("Creative_suppress"), "Score"] = np.nan
        effects = intervention_effects(frame, "Score")
        missing = effects[effects.Treatment.eq("Creative_suppress")]
        self.assertTrue(missing.Mean_Difference.isna().all())
        self.assertTrue(missing.Analysis_Status.ne("ok").all())
        with self.assertRaisesRegex(ValueError, "Multiple rows"):
            intervention_effects(pd.concat([frame, frame]), "Score")

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
