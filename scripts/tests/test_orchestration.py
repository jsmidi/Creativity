"""Offline regressions for orchestration and provenance after pipeline refactoring."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate


class BehavioralOrchestrationTests(unittest.TestCase):
    def test_api_run_preserves_blocks_failures_and_source_manifest(self):
        """Exercise the full CLI without API calls; failures must survive CSV export."""
        result = dict(Response='1. A use', Status='ok', Error='', Finish_Reason='stop',
                      Returned_Model='fake-model', System_Fingerprint='', Usage_JSON='{}')
        empty = dict(result, Response='', Status='empty')
        with tempfile.TemporaryDirectory() as directory:
            argv = ['generate', '--provider', 'together', '--model', 'fake-model',
                    '--task', 'Alternative Uses Task', '--items', 'book',
                    '--conditions', 'Standard', 'Creative', '--repeats', '2',
                    '--output-root', directory]
            with patch.object(sys, 'argv', argv), patch.object(generate, 'load_backend', return_value=(None, object())), \
                 patch.object(generate, 'query_model', side_effect=[result, empty, result, result]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    generate.main()
            self.assertEqual(raised.exception.code, 1)
            path = next(Path(directory).rglob('results_*.csv'))
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual([row['Status'] for row in rows], ['ok', 'empty', 'ok', 'ok'])
            self.assertEqual([row['Request_Order'] for row in rows], ['0', '1', '2', '3'])
            for offset in [0, 2]:
                standard, creative = rows[offset:offset+2]
                self.assertEqual(standard['Condition'], 'Standard')
                self.assertEqual(creative['Condition'], 'Creative')
                self.assertEqual(standard['Block_ID'], creative['Block_ID'])
                self.assertEqual(standard['Generation_Seed'], creative['Generation_Seed'])
            manifest = json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(manifest['planned_responses'], 4)
            for name in manifest['source_sha256']:
                saved = Path(manifest['source_directory'])/name
                self.assertEqual(saved.read_bytes(), (Path(generate.__file__).parent/name).read_bytes())

    def test_local_backend_rejects_api_overrides_before_loading_weights(self):
        """Invalid local settings must fail before initializing a model."""
        from argparse import Namespace
        args = Namespace(provider='local', extra_body={'reasoning': True})
        with self.assertRaisesRegex(ValueError, 'only to API providers'):
            generate.load_backend(args)


if __name__ == '__main__':
    unittest.main()
