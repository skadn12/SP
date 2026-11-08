import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import summarize


class SummaryTests(unittest.TestCase):
    def test_duplicate_index_is_reported(self):
        audit = summarize.Audit()
        summarize.index([{"candidate_id": "x"}, {"candidate_id": "x"}], "candidate_id", audit, "test")
        self.assertEqual(len(audit.errors), 1)

    def test_pending_prevents_any_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report"
            report = {"integrity_errors": [], "pending": ["unfinished solver"]}
            with mock.patch.object(summarize, "build", return_value=([], report)), \
                 mock.patch("sys.argv", ["summarize", "--export", str(output)]), \
                 contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(summarize.main(), 1)
            self.assertTrue(json.loads(stdout.getvalue())["export_blocked"])
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("unchanged", encoding="utf-8")
            report = {"integrity_errors": [], "pending": []}
            with mock.patch.object(summarize, "build", return_value=([], report)), \
                 mock.patch("sys.argv", ["summarize", "--export", str(output)]):
                with self.assertRaises(FileExistsError):
                    summarize.main()
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")


if __name__ == "__main__":
    unittest.main()
