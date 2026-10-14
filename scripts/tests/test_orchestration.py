"""Self-contained result CSVs remain usable for scoring and paired comparisons."""
import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_tasks
from scoring_common import load_responses
from analyze_effects import paired_effect


class BehavioralOrchestrationTests(unittest.TestCase):
    def test_csv_without_manifest_preserves_responses_settings_and_pairs(self):
        """Exercise the full CLI without API calls; failures must survive CSV export."""
        result = dict(Response='1. A use', Status='ok', Error='', Finish_Reason='stop',
                      Generated_Tokens=10, Prompt_Tokens=30)
        empty = dict(result, Response='', Status='empty')
        with tempfile.TemporaryDirectory() as directory:
            argv = ['generate', '--provider', 'together', '--model', 'fake-model',
                    '--task', 'Alternative Uses Task', '--items', 'book',
                    '--conditions', 'Standard', 'Creative', '--repeats', '2',
                    '--output-root', directory]
            with patch.object(sys, 'argv', argv), patch.object(run_tasks, 'load_backend', return_value=(None, object())), \
                 patch.object(run_tasks, 'query_model', side_effect=[result, empty, result, result]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run_tasks.main()
            self.assertEqual(raised.exception.code, 1)
            path = next(Path(directory).rglob('results_*.csv'))
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            expected = ["Run_ID", "Response_ID", "Model", "Task", "Instruction", "Item",
                        "Condition", "Generation_Seed", "Temperature", "Max_Tokens", "Top_P", "Top_K",
                        "Timestamp_UTC", "Prompt", "Response", "Generated_Tokens", "Prompt_Tokens",
                        "Error", "Model_Dtype"]
            self.assertEqual(list(rows[0]), expected)
            self.assertEqual(rows[0]['Top_P'], '1.0')
            self.assertEqual(rows[0]['Top_K'], 'unknown')
            self.assertEqual(rows[0]['Generated_Tokens'], '10')
            self.assertEqual(rows[0]['Prompt_Tokens'], '30')
            self.assertFalse(path.with_suffix('.json').exists())
            self.assertEqual(list(path.parent.iterdir()), [path])
            for offset in [0, 2]:
                standard, creative = rows[offset:offset+2]
                self.assertEqual(standard['Condition'], 'Standard')
                self.assertEqual(creative['Condition'], 'Creative')
                self.assertEqual(standard['Generation_Seed'], creative['Generation_Seed'])
            scored = load_responses([path], 'Alternative Uses Task')
            self.assertEqual(scored.Status.tolist(), ['ok', 'empty', 'ok', 'ok'])
            self.assertEqual(scored.Block_ID.nunique(), 2)
            scored['Score'] = [1., float('nan'), 1., 3.]
            effect = paired_effect(scored, 'Score', 'Creative', 'Standard', draws=100)
            self.assertEqual(effect['Mean_Difference'], 2.)
            self.assertEqual(effect['Complete_Pairs'], 1)
            self.assertEqual(effect['Incomplete_Pairs'], 1)

    def test_local_backend_rejects_api_overrides_before_loading_weights(self):
        """Invalid local settings must fail before initializing a model."""
        from argparse import Namespace
        args = Namespace(provider='local', extra_body={'reasoning': True})
        with self.assertRaisesRegex(ValueError, 'only to API providers'):
            run_tasks.load_backend(args)


if __name__ == '__main__':
    unittest.main()
