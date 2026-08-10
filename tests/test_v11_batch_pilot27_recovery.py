from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
    live_qa_canary_1,
    production_batch_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


PILOT27 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_27"
)
PILOT27_SHA = "be5d911624d2850b07ce52574bd0f45272344d63a6fb8d931190fa2f13e0c67f"


class BatchPilot27RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qa_root = PILOT27 / (
            "roots/cc-train-episode-216742/qa_candidates/candidate_01/"
            "02_questioner")
        cls.summary = json.loads((cls.qa_root / (
            "reports/QA_CANARY_SUMMARY.json")).read_text())
        cls.view = json.loads((cls.qa_root / (
            "frozen/qa_source_view.json")).read_text())

    def test_valid_first_support_pair_is_frozen_for_question_only_retry(self) -> None:
        attempt = self.summary["questioner_result"]["attempts"][0]
        record = json.loads((self.qa_root / attempt["raw_record_path"]).read_text())
        raw = json.loads(record["raw_response"])
        pair = raw["answer_support_span_ids"]
        suffix = live_qa_canary_1.build_questioner_retry_suffix(
            raw, attempt["validation"], self.view["lineage_pair_registry"])
        self.assertIn("Reuse this exact frozen support pair", suffix)
        self.assertIn(json.dumps(pair), suffix)
        self.assertIn("Rewrite only the question", suffix)

    def test_failed_attempts_enter_cross_candidate_exclusion_pool(self) -> None:
        questions: list[str] = []
        pairs: list[tuple[str, str]] = []
        production_batch_v11._remember_questioner_attempts(
            self.summary, excluded_questions=questions,
            excluded_support_pairs=pairs)
        attempts = self.summary["questioner_result"]["attempts"]
        self.assertEqual(len(attempts), len(questions))
        self.assertEqual(len(attempts), len(pairs))
        for attempt in attempts:
            validation = attempt["validation"]
            self.assertIn(validation["question"], questions)
            self.assertIn(tuple(validation["answer_support_span_ids"]), pairs)

    def test_pilot27_repeated_candidates_remain_historical_failures(self) -> None:
        questions = []
        for candidate in range(1, 5):
            path = PILOT27 / (
                "roots/cc-train-episode-216742/qa_candidates/"
                f"candidate_{candidate:02d}/02_questioner/reports/"
                "QA_CANARY_SUMMARY.json")
            summary = json.loads(path.read_text())
            questions.append(summary["questioner_result"]["attempts"][0][
                "validation"]["question"])
        self.assertEqual(1, len(set(questions[:3])))
        self.assertLess(len(set(questions)), len(questions))

    def test_pilot27_history_is_unchanged(self) -> None:
        self.assertEqual(PILOT27_SHA, hash_tree(PILOT27)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
