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

    def test_battery_loads_backend_once_and_keeps_task_pairing(self):
        """One shared model must still produce independently scorable task CSVs."""
        result = dict(Response="1. Example", Error="", Generated_Tokens=2, Prompt_Tokens=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "battery"
            argv = ["run_tasks", "--tasks", "aut", "dat", "drat", "cdat",
                    "--provider", "together", "--model", "fake-model", "--repeats", "1",
                    "--conditions", "standard", "creative", "--randomize",
                    "--output-root", str(root)]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(run_tasks, "load_backend", return_value=(None, object())) as load, \
                    patch.object(run_tasks, "query_model", return_value=result):
                run_tasks.main()
            load.assert_called_once()
            run_ids = set()
            for key in ("aut", "dat", "drat", "cdat"):
                path = next((root / key).rglob("results_*.csv"))
                task = run_tasks.TASK_NAMES[key]
                frame = load_responses([path], task)
                self.assertEqual(len(frame), 2 * len(run_tasks.get_task_config(task)["items"]))
                self.assertEqual(set(frame.Task), {task})
                self.assertTrue(frame.groupby("Block_ID").Condition.nunique().eq(2).all())
                self.assertTrue(frame.groupby("Block_ID").Generation_Seed.nunique().eq(1).all())
                run_ids.update(frame.Run_ID)
            self.assertEqual(len(run_ids), 4)
            # A repeated submission must fail before loading a backend or writing CSVs.
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(run_tasks, "load_backend") as load, self.assertRaises(FileExistsError):
                run_tasks.main()
            load.assert_not_called()

    def test_battery_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "preview"
            argv = ["run_tasks", "--tasks", "aut", "dat", "--dry-run", "--output-root", str(root)]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(run_tasks, "load_backend") as load:
                run_tasks.main()
            load.assert_not_called()
            self.assertFalse(root.exists())

    def test_local_backend_rejects_api_overrides_before_loading_weights(self):
        """Invalid local settings must fail before initializing a model."""
        from argparse import Namespace
        args = Namespace(provider='local', extra_body={'reasoning': True})
        with self.assertRaisesRegex(ValueError, 'only to API providers'):
            run_tasks.load_backend(args)


if __name__ == '__main__':
    unittest.main()
