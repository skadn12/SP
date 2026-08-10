from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.simple_append_v1.qa_live import (
    WRITER_MAX_TOKENS as SHARED_WRITER_MAX_TOKENS,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_fullr1_devdiv_qa_structural_v19_probe_v1.structural_qa_live import (
    WRITER_COMPLETION_TOKENS,
    preflight,
)


class StructuralQAV192WriterBudgetTests(unittest.TestCase):
    def test_budget_is_writer_specific(self):
        self.assertEqual(WRITER_COMPLETION_TOKENS, 16_384)
        self.assertEqual(SHARED_WRITER_MAX_TOKENS, 8_192)

    def test_preflight_reports_new_budget_without_api(self):
        with tempfile.TemporaryDirectory() as directory:
            result = preflight(
                run_root=Path(directory) / "fresh-run-that-must-not-exist")
        self.assertEqual(result["decision"], "pass")
        self.assertEqual(
            result["details"]["writer_max_output_tokens"], 16_384)
        self.assertTrue(result["checks"]["writer_prompt_fits"])
        self.assertEqual(result["api_calls"], 0)


if __name__ == "__main__":
    unittest.main()
