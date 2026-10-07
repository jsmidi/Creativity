"""Offline regressions for sampling, calibrated metrics and temperature isolation."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampling import sampling_kwargs
from evaluate_association import drat_score, novelty, normalized, bh_adjust, cdat_gates, main as score_main, CDAT, DRAT
from experiment import PROTOCOL_VERSION
from analyze_effects import paired_effect
from generate import query_model


class SamplingTests(unittest.TestCase):
    def test_greedy_disables_sampling_filters(self):
        settings = sampling_kwargs(0, .8, 20)
        self.assertFalse(settings['do_sample'])
        self.assertEqual((settings['top_p'], settings['top_k']), (1, 0))
        self.assertEqual(settings['num_beams'], 1)

    def test_sampling_and_invalid_inputs(self):
        settings = sampling_kwargs(1.3, .9, 40)
        self.assertEqual((settings['temperature'], settings['top_p'], settings['top_k']), (1.3, .9, 40))
        for args in [(float('nan'), 1, 0), (float('inf'), 1, 0), (-1, 1, 0), (.7, 0, 0), (.7, 1, -1)]:
            with self.assertRaises(ValueError):
                sampling_kwargs(*args)

    def test_api_top_p_is_sent(self):
        client = Mock()
        result = Mock(model='model', usage=None)
        choice = Mock(finish_reason='stop')
        choice.message.content = 'answer'
        result.choices = [choice]
        client.chat.completions.create.return_value = result
        self.assertEqual(query_model(client, 'prompt', 'model', top_p=.85)['Response'], 'answer')
        self.assertEqual(client.chat.completions.create.call_args.kwargs['top_p'], .85)


class MetricTests(unittest.TestCase):
    def test_scoring_cli_retains_failures_and_freezes_calibration(self):
        words = ['apple', 'chair', 'river', 'school', 'music', 'plant', 'engine',
                 'heart', 'book', 'fork', 'paperclip', 'towel', 'can', 'food', 'time']
        encoder = Mock()
        encoder.encode.side_effect = lambda texts, **kwargs: np.random.default_rng(42).normal(size=(len(texts), 8))
        module = SimpleNamespace(SentenceTransformer=Mock(return_value=encoder))
        for task, name, item, metric in [('cdat', CDAT, 'music', 'CDAT_Novelty'),
                                         ('drat', DRAT, 'heart | engine', 'DRAT_Score')]:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'nouns.txt').write_text('\n'.join(words))
                rows = [dict(Protocol=PROTOCOL_VERSION, Model='m', Task=name, Condition='Creative',
                             Run_ID='r', Response_ID=str(i), Block_ID=str(i), Item=item,
                             Temperature=.7, Status=status, Response=text)
                        for i, (status, text) in enumerate([
                            ('ok', '\n'.join(f'{i+1}. {w}' for i, w in enumerate(words[:10]))),
                            ('error', '')])]
                pd.DataFrame(rows).to_csv(root/'input.csv', index=False)
                argv = ['score', '--task', task, '--inputs', str(root/'input.csv'),
                        '--noun-vocabulary', str(root/'nouns.txt'), '--pool-size', '10',
                        '--baseline-draws', '100', '--output-dir', str(root/'result')]
                with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'sentence_transformers': module}), patch('builtins.print'):
                    score_main()
                result = pd.read_csv(root/'result'/f'{task}_responses.csv')
                self.assertEqual(len(result), 2)
                self.assertTrue(np.isfinite(result.loc[0, metric]))
                self.assertTrue(np.isnan(result.loc[1, metric]))
                self.assertEqual(len((root/'result'/'calibration_nouns.txt').read_text().splitlines()), 10)
                self.assertIn('embedding_sha256', json.loads((root/'result'/'manifest.json').read_text()))

    def test_geometry(self):
        self.assertAlmostEqual(novelty(np.eye(3)), 100)
        self.assertAlmostEqual(novelty(normalized([[1, 0], [-1, 0]])), 200)
        with self.assertRaises(ValueError):
            normalized([[0, 0]])

    def test_drat_gate_is_strict_and_requires_three_survivors(self):
        anchors = np.array([[1., 0, 0]])
        pool = np.tile([0., 1, 0], (10, 1))
        words = normalized([[1, 1, 0], [1, -1, 0], [1, 0, 1], [0, 1, 0]])
        score, threshold, count, _ = drat_score(words, anchors, pool)
        self.assertEqual(threshold, 0)
        self.assertEqual(count, 3)  # exact-threshold last word is excluded
        self.assertAlmostEqual(score, novelty(words[:3]))
        self.assertEqual(drat_score(words[:2], anchors, pool)[0], 0)
        self.assertEqual(drat_score(np.empty((0, 3)), anchors, pool)[0], 0)

    def test_drat_uses_closest_anchor_not_mean(self):
        anchors = np.eye(3)
        pool = normalized([[1, 1, 1]]*10)
        self.assertEqual(drat_score(np.eye(3), anchors, pool)[2], 3)

    def test_bh_correction(self):
        np.testing.assert_allclose(bh_adjust([.01, .04, .03, np.nan]), [.03, .04, .04, np.nan])

    def test_cdat_gate_uses_cues_not_repeat_count(self):
        frame = pd.DataFrame([dict(Model='m', Run_ID='r', Temperature=.7, Condition='Creative',
                                  Item=item, CDAT_Novelty=80, CDAT_Appropriateness=score)
                              for item, score in [('a', 170), ('b', 171), ('c', 169)] for _ in range(30)])
        baseline = pd.DataFrame([dict(Item=item, Appropriateness=score) for item in ['a', 'b', 'c'] for score in [100, 101, 99]*100])
        gates = cdat_gates(frame, baseline)
        self.assertEqual(gates.iloc[0].Cues, 3)
        self.assertTrue(gates.iloc[0].Gate_Passed)
        self.assertEqual(gates.iloc[0].CDAT_Score, 80)

    def test_greedy_no_sampling_ci_and_mixed_temperature_rejected(self):
        frame = pd.DataFrame([dict(Run_ID='r', Block_ID=str(i), Response_ID=f'{i}{c}', Item='book',
                                  Condition=c, Score=s, Temperature=0)
                              for i in range(3) for c, s in [('Creative', 2), ('Standard', 1)]])
        result = paired_effect(frame, 'Score', 'Creative', 'Standard', draws=100)
        self.assertTrue(np.isnan(result['CI_Lower']))
        self.assertEqual(result['Bootstrap_Draws'], 0)
        frame.loc[0, 'Temperature'] = 1
        with self.assertRaisesRegex(ValueError, 'Temperature'):
            paired_effect(frame, 'Score', 'Creative', 'Standard')


if __name__ == '__main__':
    unittest.main()
