from __future__ import annotations

import json
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from scripts.current_method.restart_n5_spell.live_backends import (
    ChallengerAPIConfig,
    OpenAICompatibleChallengerClient,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
    live_qa_canary_1,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    select_frozen_lineage_pair,
    validate_cross_boundary_core_object,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.qa_semantic_audit import (
    validate_qa_semantic_audit_output,
)


PILOT28 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_28"
)
PILOT28_SHA = "93fb403c7a04c29236062b0a9b7f2054456c7dcf7d47c463450b236ab5d530b6"


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps({
            "choices": [{"message": {"content": "{}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1,
                      "total_tokens": 4},
        }).encode("utf-8")


class BatchPilot28RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qa_root = PILOT28 / (
            "roots/cc-train-episode-216742/qa_candidates/candidate_01/"
            "02_questioner")
        cls.view = json.loads((cls.qa_root / (
            "frozen/qa_source_view.json")).read_text(encoding="utf-8"))
        cls.catalog = json.loads((cls.qa_root / (
            "frozen/span_catalog.json")).read_text(encoding="utf-8"))

    def test_candidate_pair_is_program_frozen_to_an_anchor_row(self) -> None:
        frozen = select_frozen_lineage_pair(
            self.view["lineage_pair_registry"], candidate_index=1)
        first_row = self.view["lineage_pair_registry"]["rows"][0]
        expected = first_row["eligible_same_speaker_pairs"][0]
        self.assertEqual(expected["parent_span_id"], frozen["parent_span_id"])
        self.assertEqual(expected["child_span_id"], frozen["child_span_id"])
        self.assertEqual(first_row["scene_ref"], frozen["scene_ref"])

    def test_pilot28_unanchored_pair_cannot_override_frozen_pair(self) -> None:
        summary = json.loads((self.qa_root / (
            "reports/QA_CANARY_SUMMARY.json")).read_text(encoding="utf-8"))
        raw = summary["questioner_result"]["accepted_raw"]
        frozen = select_frozen_lineage_pair(
            self.view["lineage_pair_registry"], candidate_index=1)
        result = validate_cross_boundary_core_object(
            raw, span_catalog=self.catalog,
            partition_by_turn=self.view["partition_by_turn"],
            lineage_pair_registry=self.view["lineage_pair_registry"],
            frozen_support_pair=frozen, evidence_mismatch_policy="mark_only")
        self.assertEqual("cross_boundary_qa_contract_reject", result["decision"])
        self.assertIn("cross_boundary_qa_frozen_support_pair_mismatch",
                      result["reasons"])
        suffix = live_qa_canary_1.build_questioner_retry_suffix(
            raw, result, self.view["lineage_pair_registry"], frozen)
        self.assertIn("program has already frozen", suffix)
        self.assertNotIn(frozen["parent_span_id"], suffix)
        self.assertIn("do not return any support ID", suffix)

    def test_pilot28_inferred_causality_cannot_be_accepted(self) -> None:
        audit_path = self.qa_root.parent / (
            "03_semantic_audit/reports/semantic_audit.json")
        historical = json.loads(audit_path.read_text(encoding="utf-8"))["audit"]
        migrated = {**historical,
                    "cross_boundary_relation_entailment": "requires_inference"}
        result = validate_qa_semantic_audit_output(
            migrated,
            expected_span_ids=historical["evidence_span_ids_checked"])
        self.assertEqual("qa_semantic_audit_schema_invalid", result["decision"])
        self.assertIn("audit_accept_conditions_not_met", result["errors"])

    def test_http_429_retries_the_byte_identical_request(self) -> None:
        client = OpenAICompatibleChallengerClient(ChallengerAPIConfig(
            base_url="https://example.invalid/v1", api_key="x", model="m"))
        rate_limit = urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions", 429,
            "Too Many Requests", {"Retry-After": "0"}, None)
        outcomes = [rate_limit, _Response()]
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("urllib.request.urlopen", fake_urlopen), \
                patch("scripts.current_method.restart_n5_spell."
                      "live_backends.time.sleep") as sleep:
            response = client.complete(
                prompt="frozen", max_tokens=8, json_object=True)
        self.assertEqual(2, len(requests))
        self.assertIs(requests[0], requests[1])
        self.assertEqual(requests[0].data, requests[1].data)
        sleep.assert_called_once_with(0.0)
        self.assertEqual(1, response.metadata["rate_limit_retry_count"])
        self.assertEqual(2, response.metadata["transport_attempt_count"])

    def test_pilot28_spell_failure_remains_historical(self) -> None:
        spell = json.loads((self.qa_root.parent / (
            "05_spell_filter/reports/SPELL_FILTER_SUMMARY.json"
        )).read_text(encoding="utf-8"))
        self.assertEqual("reject_all_wrong", spell["final_decision"])
        self.assertEqual(2, spell["no_context_permutation_diagnostic"][
            "semantic_correct_count"])
        self.assertEqual([0, 0, 0, 0, 0],
                         spell["result"]["full_context"]["verified_rewards"])

    def test_pilot28_history_is_unchanged(self) -> None:
        self.assertEqual(PILOT28_SHA, hash_tree(PILOT28)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
