"""Offline contract tests for the A/B input preparation renderer."""

from __future__ import annotations

import unittest

from .prepare import _mode_boundary, _shared_realizer_suffix


class ModeBoundaryTest(unittest.TestCase):
    def test_full_parent_boundary_is_explicit(self):
        text = _mode_boundary("A_FULL_PARENT")
        self.assertIn("A_FULL_PARENT", text)
        self.assertIn("only the newly generated append-only Child", text)

    def test_text_free_boundary_forbids_reconstruction(self):
        text = _mode_boundary("B_TEXT_FREE")
        self.assertIn("B_TEXT_FREE", text)
        self.assertIn("No historical dialogue wording", text)
        self.assertIn("do not reconstruct", text)

    def test_shared_suffix_preserves_identity_and_expansion_task(self):
        text = _shared_realizer_suffix(
            root_id="root-1", source_cluster_id="cluster-1",
            source_prefix="Parent+R01+R02-feedback")
        self.assertIn("【扩充专属任务——固定规则】", text)
        self.assertIn('"root_id": "root-1"', text)
        self.assertIn('"source_cluster_id": "cluster-1"', text)


if __name__ == "__main__":
    unittest.main()
