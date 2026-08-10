from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.run_single_item_semantic_parity_retry import (
    retry_eligible,
)


class RunSingleItemSemanticParityRetryTests(unittest.TestCase):
    def test_only_bad_ground_is_eligible_for_same_qa_option_retry(self) -> None:
        self.assertTrue(retry_eligible("bad_ground"))
        for decision in ("reject_easy", "reject_all_wrong", "retain_frontier",
                         "minimal_admission_reject", ""):
            self.assertFalse(retry_eligible(decision))


if __name__ == "__main__":
    unittest.main()
