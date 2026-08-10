from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    build_cross_boundary_questioner_prompt,
    validate_cross_boundary_core_object,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_qa_canary_1 import (
    build_questioner_retry_suffix,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)


PILOT30 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_30"
)
PILOT30_SHA = "a038f4f39bf1e474de247706231c803b539d7c62ea1bfe3401ac7317fb387081"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BatchPilot30RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qa = PILOT30 / (
            "roots/cc-train-episode-41625/qa_candidates/candidate_02/"
            "02_questioner")
        cls.view = load(cls.qa / "frozen/qa_source_view.json")
        cls.catalog = load(cls.qa / "frozen/span_catalog.json")
        cls.freeze = load(cls.qa / "run_freeze_manifest.json")

    def test_question_only_output_is_bound_to_the_program_pair(self) -> None:
        raw = {"question": (
            "How did the speaker's approach to community challenges differ "
            "between the earlier and later conversations?")}
        result = validate_cross_boundary_core_object(
            raw, span_catalog=self.catalog,
            partition_by_turn=self.view["partition_by_turn"],
            lineage_pair_registry=self.view["lineage_pair_registry"],
            frozen_support_pair=self.freeze["frozen_support_pair"],
            evidence_mismatch_policy="mark_only")
        self.assertEqual("cross_boundary_qa_accept", result["decision"],
                         result.get("reasons"))
        self.assertEqual({
            self.freeze["frozen_support_pair"]["parent_span_id"],
            self.freeze["frozen_support_pair"]["child_span_id"],
        }, set(result["core_qa"]["answer_support_span_ids"]))

    def test_historical_same_pair_in_narrative_order_is_no_longer_rejected(self) -> None:
        summary = load(self.qa / "reports/QA_CANARY_SUMMARY.json")
        historical = summary["questioner_result"]["attempts"][0]
        raw = historical["validation"]["rejected_core_qa"]
        result = validate_cross_boundary_core_object(
            raw,
            span_catalog=self.catalog,
            partition_by_turn=self.view["partition_by_turn"],
            lineage_pair_registry=self.view["lineage_pair_registry"],
            frozen_support_pair=self.freeze["frozen_support_pair"],
            evidence_mismatch_policy="mark_only")
        self.assertEqual("cross_boundary_qa_accept", result["decision"],
                         result.get("reasons"))

    def test_genuinely_wrong_explicit_ids_remain_strictly_rejected(self) -> None:
        frozen = self.freeze["frozen_support_pair"]
        wrong_child = next(
            span["span_id"] for span in self.catalog["spans"]
            if span["span_id"].startswith("R01:") and
            span["span_id"] != frozen["child_span_id"])
        raw = {
            "question": "How did the speaker's approach differ across the two conversations?",
            "answer_support_span_ids": [frozen["parent_span_id"], wrong_child],
        }
        result = validate_cross_boundary_core_object(
            raw, span_catalog=self.catalog,
            partition_by_turn=self.view["partition_by_turn"],
            lineage_pair_registry=self.view["lineage_pair_registry"],
            frozen_support_pair=frozen, evidence_mismatch_policy="mark_only")
        self.assertEqual("cross_boundary_qa_contract_reject", result["decision"])
        self.assertIn("cross_boundary_qa_frozen_support_pair_mismatch",
                      result["reasons"])

    def test_live_prompt_schema_returns_only_question(self) -> None:
        prompt = build_cross_boundary_questioner_prompt(
            root_id="cc-train-episode-41625", span_catalog=self.catalog,
            partition_by_turn=self.view["partition_by_turn"],
            lineage_pair_registry=self.view["lineage_pair_registry"],
            frozen_support_pair=self.freeze["frozen_support_pair"])
        schema = prompt.rsplit("OUTPUT JSON SCHEMA\n", 1)[1]
        parsed = json.loads(schema)
        self.assertEqual(["question"], parsed["required"])
        self.assertEqual({"question"}, set(parsed["properties"]))

    def test_retry_does_not_echo_long_frozen_ids(self) -> None:
        frozen = self.freeze["frozen_support_pair"]
        suffix = build_questioner_retry_suffix(
            {}, {"reasons": ["cross_boundary_qa_schema_invalid"]},
            self.view["lineage_pair_registry"], frozen)
        self.assertNotIn(frozen["parent_span_id"], suffix)
        self.assertNotIn(frozen["child_span_id"], suffix)
        self.assertIn("Return only the question field", suffix)

    def test_pilot30_is_unchanged(self) -> None:
        self.assertEqual(PILOT30_SHA, hash_tree(PILOT30)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
