from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from scripts.current_method.stage3_truth_constructors import (
    BINDING_SCHEMA_VERSION,
    construct_truth_skeleton,
    example_content_bindings,
    materialize_truth_skeleton,
    validate_content_bindings,
    validate_materialized_truth,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "stage3-data-fix" / "run229_truth_first_negative_fixture.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bridge_target() -> dict[str, object]:
    return {
        "qa_operation": "bridge",
        "preferred_answer_type": "Person",
        "reasoning_subtype": "conjunctive_two_support_bridge",
        "reasoning_template": "join_two_support_facts",
        "hop_count": 2,
        "hard_negative_strategy": "one_hop_preserving_alternative",
        "forbid_max_time_shortcut": False,
        "target_basis": "run229_negative_regression",
    }


class Run229TruthFirstNegativeRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        clean_path = ROOT / cls.fixture["artifacts"]["clean_samples"]["path"]
        cls.legacy_sample = json.loads(clean_path.read_text(encoding="utf-8").splitlines()[0])

    def test_frozen_negative_artifact_hashes_have_not_changed(self) -> None:
        self.assertEqual(self.fixture["disposition"], "frozen_negative_never_training")
        for record in self.fixture["artifacts"].values():
            self.assertEqual(_sha256(ROOT / record["path"]), record["sha256"])

    def test_fixture_still_contains_the_three_generic_failure_shapes(self) -> None:
        events = self.legacy_sample["graph_update"]["events"]
        by_position = {index + 1: event for index, event in enumerate(events)}
        self.assertEqual(
            (by_position[2]["slot"], by_position[2]["value"]),
            ("discovery_date", "Lila"),
        )
        self.assertEqual(
            [(event["slot"], event["value"]) for event in events[3:6]],
            [("confirmation", "Mira"), ("confirmation", "Eli"), ("confirmation", "Rhea")],
        )
        question = self.legacy_sample["qa"]["question"]
        self.assertEqual(by_position[1]["value"], by_position[2]["entity"])
        self.assertIn(by_position[1]["value"], question)

    def test_legacy_graph_has_no_constructor_admission_manifest(self) -> None:
        skeleton = construct_truth_skeleton(
            _bridge_target(), graph_id="truth:first:bridge", parent_graph_id="parent:1"
        )
        report = validate_materialized_truth(skeleton, self.legacy_sample)
        self.assertFalse(report["pass"])
        self.assertIn("materialized_graph_missing", {item["code"] for item in report["issues"]})
        self.assertIn("truth_manifest_missing", {item["code"] for item in report["issues"]})

    def test_api_cannot_replay_run229_events_or_choose_slots(self) -> None:
        skeleton = construct_truth_skeleton(
            _bridge_target(), graph_id="truth:first:bridge", parent_graph_id="parent:1"
        )
        payload = example_content_bindings(skeleton)
        payload["events"] = copy.deepcopy(self.legacy_sample["graph_update"]["events"])
        payload["answer_slot"] = "discovery_date"
        report = validate_content_bindings(skeleton, payload)
        self.assertFalse(report["pass"])
        forbidden = next(
            item for item in report["issues"] if item["code"] == "forbidden_binding_payload_keys"
        )
        self.assertEqual(forbidden["keys"], ["answer_slot", "events"])

    def test_constructor_slots_and_roles_cannot_reproduce_run229(self) -> None:
        skeleton = construct_truth_skeleton(
            _bridge_target(), graph_id="truth:first:bridge", parent_graph_id="parent:1"
        )
        payload = example_content_bindings(skeleton)
        self.assertEqual(payload["schema_version"], BINDING_SCHEMA_VERSION)
        materialized = materialize_truth_skeleton(skeleton, payload)
        self.assertTrue(validate_materialized_truth(skeleton, materialized)["pass"])
        events = materialized["graph"]["events"]
        self.assertNotIn("discovery_date", {event["slot"] for event in events})
        self.assertNotIn("confirmation", {event["slot"] for event in events})
        self.assertEqual(events[0]["value"], events[1]["entity"])
        self.assertEqual(events[1]["slot"], "assigned_partner")


if __name__ == "__main__":
    unittest.main()
