"""Read-only regression coverage against the delivered, immutable export."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import tempfile
import unittest

from . import planner_feedback_source as adapter
from ..round0_ordering_active_feedback_exclusion_v1 import NO_FEEDBACK
from ..round0_ordering_active_feedback_exclusion_v1.loader import bind_snapshot, load_export
from ..round0_ordering_active_feedback_exclusion_v1.sources import file_sha, value_sha


EXPORT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_round0_ordering_feedback_exclusion_v1_offline_live_1/export/artifact_inventory.json")
ROOT = "cc-train-episode-122711"


class PlannerFeedbackSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EXPORT.is_file():
            raise unittest.SkipTest("immutable delivered export unavailable")
        cls.loaded = load_export(EXPORT)
        cls.views, cls.artifact_ids = adapter._export_views(cls.loaded)
        cls.identity = cls.loaded["reference_identity"][ROOT]
        cls.cid = cls.loaded["envelopes"][f"expansion:{ROOT}"]["records"][0]["source_candidate_id"]

    def test_explicit_manifest_drives_complete_selected_feedback(self):
        result = adapter.load_planner_feedback(export_manifest=EXPORT, root_id=ROOT, planning_history_identity=self.identity)
        self.assertEqual(result["export_lineage"]["export_manifest"], str(EXPORT))
        self.assertEqual(result["export_lineage"]["candidate_scope_count"], 2493)
        self.assertEqual(result["export_lineage"]["active_shared_record_count"], 1018)
        self.assertEqual(result["export_lineage"]["selected_feedback_count"], 1)
        self.assertEqual(result["feedback_envelope"], self.loaded["envelopes"][f"expansion:{ROOT}"])
        feedback = result["shared_feedback_view"]["feedback"]
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["source_candidate_id"], self.cid)
        record = self.loaded["records"][self.cid]
        self.assertEqual(feedback[0]["historical_qa"], record["historical_qa"])
        self.assertEqual(feedback[0]["solver_observations"], record["performance"])
        self.assertFalse(result["gold_semantics_validated"])
        self.assertFalse(result["qa_snapshot_binding_performed"])
        self.assertEqual(result["external_calls"], 0)

    def test_current_selected_card_identity_may_remain_same(self):
        envelope, selection = adapter._select_expansion(self.loaded, self.views, ROOT)
        self.assertEqual(envelope["records"][0]["source_arm"], "E1Q1_FEEDBACK_GUIDED")
        self.assertEqual(envelope["records"][0]["historical_qa"]["question_type"], "ordering")
        self.assertEqual(selection[0]["expansion_card"]["source_candidate_id"], self.cid)
        self.assertEqual(selection[0]["routing_and_selection"]["canonical_representative_candidate_id"], self.cid)

    def test_later_191_ordering_scope_is_not_excluded(self):
        ordering = [row for row in self.loaded["scope"].values() if row["question_type"] == "ordering"]
        excluded = {row["candidate_id"] for row in self.views["excluded_candidates.jsonl"]}
        self.assertEqual(len(ordering), 191)
        self.assertTrue(all(row["normalized_experiment_arm"] != "ROUND0_PARENT_PLUS_R01" for row in ordering))
        self.assertFalse({row["candidate_id"] for row in ordering} & excluded)
        self.assertNotIn(self.cid, excluded)

    def test_selected_identity_cannot_be_replaced_by_another_eligible_card(self):
        views = dict(self.views)
        views["active_expansion_selection.jsonl"] = deepcopy(self.views["active_expansion_selection.jsonl"])
        selection = next(row for row in views["active_expansion_selection.jsonl"] if row["root_id"] == ROOT)
        other = next(row for row in views["active_expansion_seed_pool.jsonl"] if row["root_id"] == ROOT and row["canonical_representative_candidate_id"] != self.cid)
        selection["canonical_representative_candidate_id"] = other["canonical_representative_candidate_id"]
        with self.assertRaisesRegex(ValueError, "planner_card_selection_identity_mismatch"):
            adapter._select_expansion(self.loaded, views, ROOT)

    def test_excluded_round0_cannot_be_backfilled_as_selected_record(self):
        loaded = dict(self.loaded)
        loaded["envelopes"] = dict(self.loaded["envelopes"])
        envelope = deepcopy(self.loaded["envelopes"][f"expansion:{ROOT}"])
        envelope["records"][0]["source_candidate_id"] = next(row["candidate_id"] for row in self.views["excluded_candidates.jsonl"] if row["root_id"] == ROOT)
        loaded["envelopes"][f"expansion:{ROOT}"] = envelope
        with self.assertRaisesRegex(ValueError, "feedback_source_outside_scope_or_duplicate"):
            adapter._select_expansion(loaded, self.views, ROOT)

    def test_selected_source_must_remain_eligible_with_positive_weight(self):
        for name, key, change in (
            ("candidate_active_eligibility.jsonl", "candidate_id", {"targeted_eligible": False}),
            ("active_target_weights.jsonl", "canonical_representative_candidate_id", {"raw_weight": 0}),
        ):
            with self.subTest(name=name):
                views = dict(self.views)
                views[name] = deepcopy(self.views[name])
                next(row for row in views[name] if row[key] == self.cid).update(change)
                with self.assertRaisesRegex(ValueError, "not_expansion_eligible|nonpositive_weight"):
                    adapter._select_expansion(self.loaded, views, ROOT)

    def test_authoritative_planning_r02_does_not_relax_qa_r03_requirement(self):
        history, refs = adapter._validate_history(self.loaded, ROOT, self.identity)
        self.assertEqual(history["turn_count"], 143)
        self.assertIn("R02:S03:T011", refs)
        with self.assertRaisesRegex(ValueError, "new_frozen_r03_required_r02_rejected"):
            bind_snapshot(self.loaded, self.loaded["source_manifest"]["inputs"]["dialogue_identity"]["path"])

    def test_planning_history_and_old_qa_snapshot_are_separately_identified(self):
        envelope, selection = adapter._select_expansion(self.loaded, self.views, ROOT)
        old = selection[0]["historical_qa_evaluation_identity"]
        self.assertNotEqual(old["candidate_record_sha256"], self.identity["complete_dialogue_sha256"])
        self.assertFalse(old["scores_reinterpreted_against_planning_history"])
        self.assertEqual(old["observation_snapshot_ids"]["M_clean_policy"], envelope["records"][0]["performance"]["M_clean_policy"]["observation_snapshot_id"])

    def test_current_r03_or_wrong_branch_cannot_be_mixed_into_planning_history(self):
        for change in ({"source_prefix": "Parent+R01+R02-feedback+R03"}, {"input_dialogue_file_sha256": "0" * 64}):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "planner_history_"):
                    adapter._validate_history(self.loaded, ROOT, {**self.identity, **change})

    def test_history_content_sha_is_recomputed(self):
        original = json.loads(Path(self.identity["input_dialogue_path"]).read_text())
        original["turns"][0]["text"] = "Unretained altered history."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "altered_history.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            identity = {**self.identity, "input_dialogue_path": str(path), "input_dialogue_file_sha256": file_sha(path)}
            loaded = {**self.loaded, "reference_identity": {**self.loaded["reference_identity"], ROOT: identity}}
            with self.assertRaisesRegex(ValueError, "planner_history_content_sha_mismatch"):
                adapter._validate_history(loaded, ROOT, identity)

    def test_extra_role_views_are_inventory_and_sha_bound(self):
        changed = dict(self.loaded)
        changed["export_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "planner_export_inventory_changed_after_load"):
            adapter._export_views(changed)
        target = self.artifact_ids["active_target_weights.jsonl"]["path"]
        original = adapter.file_sha
        with patch.object(adapter, "file_sha", side_effect=lambda path: "0" * 64 if str(path) == target else original(path)):
            with self.assertRaisesRegex(ValueError, "planner_export_view_sha_or_path_mismatch"):
                adapter._export_views(self.loaded)

    def test_forged_empty_feedback_not_allowed_when_selection_exists(self):
        loaded = dict(self.loaded)
        loaded["envelopes"] = dict(self.loaded["envelopes"])
        loaded["envelopes"][f"expansion:{ROOT}"] = {**self.loaded["envelopes"][f"expansion:{ROOT}"], "status": NO_FEEDBACK, "reason": "No source", "records": []}
        with self.assertRaisesRegex(ValueError, "planner_expansion_selection_cardinality_mismatch"):
            adapter._select_expansion(loaded, self.views, ROOT)

    def test_declared_empty_export_state_preserves_reason(self):
        loaded = {**self.loaded, "envelopes": dict(self.loaded["envelopes"])}
        expected = {**self.loaded["envelopes"][f"expansion:{ROOT}"], "status": NO_FEEDBACK, "reason": "NO_RETAINED_SOURCE_PASSES_EXISTING_TARGETED_ELIGIBILITY", "records": []}
        loaded["envelopes"][f"expansion:{ROOT}"] = expected
        views = dict(self.views)
        for name in ("active_expansion_selection.jsonl", "active_expansion_feedback_cards.jsonl", "active_expansion_seed_pool.jsonl"):
            views[name] = [row for row in views[name] if row["root_id"] != ROOT]
        envelope, selection = adapter._select_expansion(loaded, views, ROOT)
        self.assertEqual(envelope, expected)
        self.assertEqual(selection, [])

    def test_unsupported_root_rejected_without_forging_no_feedback(self):
        with patch.object(adapter, "load_export", return_value=self.loaded), patch.object(adapter, "_export_views", return_value=(self.views, self.artifact_ids)):
            with self.assertRaisesRegex(ValueError, "planner_expansion_envelope_unavailable_in_export"):
                adapter.load_planner_feedback(export_manifest=EXPORT, root_id="unsupported", planning_history_identity={})

    def test_selected_views_are_not_mutated(self):
        before = value_sha(self.loaded["envelopes"][f"expansion:{ROOT}"])
        envelope, selection = adapter._select_expansion(self.loaded, self.views, ROOT)
        envelope["records"][0]["historical_qa"]["canonical_gold"] = ["replacement"]
        selection[0]["routing_and_selection"]["raw_weight"] = 0
        self.assertEqual(value_sha(self.loaded["envelopes"][f"expansion:{ROOT}"]), before)
        self.assertGreater(next(row for row in self.views["active_expansion_selection.jsonl"] if row["root_id"] == ROOT)["raw_weight"], 0)


if __name__ == "__main__":
    unittest.main()
