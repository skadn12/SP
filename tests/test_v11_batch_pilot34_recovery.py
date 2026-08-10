from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_qa_canary_1 import (
    build_questioner_retry_suffix,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)


PILOT34 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_34"
)
PILOT34_SHA = "7f081ebc236f4282442f837d9d6f007fef59492d448c63a75ba6d05c9e77f8d3"
QA = PILOT34 / (
    "roots/cc-train-episode-216742/qa_candidates/candidate_01/02_questioner"
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BatchPilot34RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.freeze = load(QA / "run_freeze_manifest.json")
        cls.view = load(QA / "frozen/qa_source_view.json")
        cls.summary = load(QA / "reports/QA_CANARY_SUMMARY.json")

    def test_historical_unsupported_interlocutor_is_strictly_rejected(self):
        attempt = self.summary["questioner_result"]["attempts"][1]
        self.assertIn(
            "cross_boundary_qa_question_speaker_not_grounded:Classmates B",
            attempt["validation"]["reasons"])

    def test_retry_names_only_the_frozen_evidence_speaker(self):
        attempt = self.summary["questioner_result"]["attempts"][1]
        suffix = build_questioner_retry_suffix(
            {}, attempt["validation"], self.view["lineage_pair_registry"],
            self.freeze["frozen_support_pair"])
        self.assertIn("only dialogue speaker label allowed", suffix)
        self.assertIn("Do not name the person they spoke with", suffix)
        self.assertIn(self.freeze["frozen_support_pair"]["speaker"], suffix)

    def test_detail_retry_keeps_only_an_abstract_shared_theme(self):
        attempt = self.summary["questioner_result"]["attempts"][0]
        suffix = build_questioner_retry_suffix(
            {}, attempt["validation"], self.view["lineage_pair_registry"],
            self.freeze["frozen_support_pair"])
        self.assertIn("remove the concrete parent object/event", suffix)

    def test_pilot34_remains_read_only(self):
        self.assertEqual(PILOT34_SHA, hash_tree(PILOT34)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
