"""Contract-only root increment fixtures, never real model observations."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from .fixtures import failed_bundle, fixture_patch
from .planning import prepare_original, merge_patch, sha
from .root_scope import ROOT_POLICY_ID, ROOT_SCOPE, ROOT_PROMPT_VERSION
from .root_planning import prepare_root_original, merge_root_patch
from .run import offline_network_guard
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import build_speaker_registry


def root_fixture_bundle():
    bundle = failed_bundle()
    bundle.update(policy_id=ROOT_POLICY_ID, incremental_scope=ROOT_SCOPE,
                  supplement_prompt_version=ROOT_PROMPT_VERSION)
    return bundle


def existing_phase_patch(bundle, sources=("ME03:C01", "ME07:C01", "ME08:C01")):
    prepared = prepare_root_original(bundle)
    patch = fixture_patch(bundle)
    example = patch["phases"][0]
    patch.update(reason="OFFLINE MOCK root source contract test", phases=[], new_developments=[])
    for index, source in enumerate(sources, 1):
        row = deepcopy(example)
        row.update(phase_id=f"phase:root_{index}", source_candidate_id=source,
                   original_goal=prepared["candidate_registry"][source]["scene"]["goal"],
                   dependencies=[{"phase_id": "initial:" + source,
                                  "condition": "OFFLINE MOCK actual prerequisite established"}])
        patch["phases"].append(row)
    return patch


def independent_new_patch(bundle, *, count=1, new_people=False):
    speakers = build_speaker_registry(bundle["parent"])["speakers"]
    patch = {"status": "PATCH", "original_plan_sha256": bundle["original_plan_sha256"],
             "phases": [], "new_developments": [], "reason": "OFFLINE MOCK new event contract fixture"}
    for index in range(1, count + 1):
        patch["new_developments"].append({"phase_id": f"phase:independent_{index}", "dependencies": [],
            "candidate": {
                "source_basis": ["OFFLINE MOCK a new local activity, not an asserted prior event"],
                "new_people": ([{"name": "MockVisitor" + str(index),
                                 "relationship_basis": "Introduced at the new community activity"}]
                               if new_people else []),
                "goal": "Prepare a labelled community workshop card " + str(index),
                "speaker_refs": [speaker["speaker_ref"] for speaker in speakers],
                "transition_from_previous": "The participants begin a new local workshop discussion.",
                "before_state": "The workshop card has not been prepared.",
                "required_transition": "The participants choose and record a workshop card label " + str(index),
                "after_state": "The workshop card has an agreed label.",
                "actor_bindings": [{"actor_ref": speakers[0]["speaker_ref"],
                                    "binding": "Records the chosen label without changing previous events."}],
                "interaction_beats": ["One participant proposes a label.", "Another participant checks the label.",
                                      "They record the agreed label on the workshop card."]}})
    return patch


class RootPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = offline_network_guard()
        cls.network.__enter__()
        cls.bundle = root_fixture_bundle()
        cls.prepared = prepare_root_original(cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls.network.__exit__(None, None, None)

    def test_01_complete_original_registry_including_three_events(self):
        self.assertEqual(len(self.prepared["eligible_source_candidate_ids"]), 11)
        self.assertEqual(self.prepared["affected_candidate_ids"], ["ME07:C01"])
        patch = existing_phase_patch(self.bundle)
        before = sha(self.prepared)
        result = merge_root_patch(self.prepared, patch)
        self.assertTrue(result["valid"], result)
        self.assertEqual(sha(self.prepared), before)
        self.assertEqual(len(result["merged"]["added_phases"]), 3)
        self.assertEqual(sha(result["merged"]["original_plan"]), self.bundle["original_plan_sha256"])
        self.assertEqual(result["diff"]["original_candidate_changes"], [])

    def test_02_old_scope_still_rejects_actual_failed_patch(self):
        run = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_contract_alignment_extra_validation_authorized_live_1")
        saved = json.loads((run / "execution/increment_patch_and_diff.json").read_text())
        prepared = prepare_original(self.bundle["original_plan"], parent=self.bundle["parent"],
                                    candidates=self.bundle["candidates"], verified_tracks=self.bundle["verified_tracks"])
        self.assertEqual(merge_patch(prepared, saved["raw_patch"])["errors"], saved["errors"])
        self.assertEqual(saved["errors"], ["patch_phase_2_unrelated_candidate"])

    def test_03_unknown_source_is_not_rebound(self):
        patch = existing_phase_patch(self.bundle, ("ME07:C01",))
        patch["phases"][0]["source_candidate_id"] = "ME999:C01"
        self.assertIn("patch_phase_1_unknown_source", merge_root_patch(self.prepared, patch)["errors"])

    def test_04_other_source_goal_is_not_automatically_filled(self):
        patch = existing_phase_patch(self.bundle, ("ME07:C01",))
        patch["phases"][0]["original_goal"] = self.prepared["candidate_registry"]["ME03:C01"]["scene"]["goal"]
        self.assertIn("patch_phase_1_original_goal_changed", merge_root_patch(self.prepared, patch)["errors"])

    def test_05_invalid_duplicate_or_future_phase_rejected(self):
        for kind in ("invalid", "duplicate", "future"):
            patch = existing_phase_patch(self.bundle)
            if kind == "invalid":
                patch["phases"][0]["phase_id"] = "phase:ME07:C01"
            elif kind == "duplicate":
                patch["phases"][1]["phase_id"] = patch["phases"][0]["phase_id"]
            else:
                patch["phases"][0]["dependencies"][0]["phase_id"] = "phase:root_3"
            self.assertFalse(merge_root_patch(self.prepared, patch)["valid"], kind)

    def test_06_root_round_snapshot_and_registry_tampering_fail_closed(self):
        for key in ("root", "round", "history", "candidate", "plan", "sha"):
            bundle = deepcopy(self.bundle)
            if key == "root":
                bundle["identity"]["root_id"] = "foreign-root"
            elif key == "round":
                bundle["identity"]["round_index"] = 4
            elif key == "history":
                bundle["parent"]["dialogue"]["turns"][0]["text"] = "foreign snapshot"
                bundle["history_sha256"] = sha(bundle["parent"])
            elif key == "candidate":
                bundle["candidates"][0]["root_id"] = "foreign-root"
            elif key == "plan":
                bundle["original_plan"]["round_goal"] = "unfrozen replacement"
                bundle["original_plan_sha256"] = sha(bundle["original_plan"])
            else:
                bundle["frozen_sources"]["history"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError, msg=key):
                prepare_root_original(bundle)

    def test_07_prepared_registry_cannot_be_rebound(self):
        prepared = deepcopy(self.prepared)
        prepared["candidate_registry"]["ME07:C01"]["scene"]["goal"] = "foreign target"
        self.assertEqual(merge_root_patch(prepared, independent_new_patch(self.bundle))["errors"],
                         ["root_patch_source_scope_mismatch"])

    def test_08_independent_new_event_needs_no_fake_old_goal_or_source(self):
        patch = independent_new_patch(self.bundle)
        raw = patch["new_developments"][0]
        self.assertNotIn("original_goal", raw)
        self.assertNotIn("source_candidate_id", raw)
        result = merge_root_patch(self.prepared, patch)
        self.assertTrue(result["valid"], result)
        scene = result["merged"]["added_phases"][0]["scene"]
        self.assertEqual(scene["source_candidate_id"], "ND09:C01")
        self.assertEqual(scene["goal_origin"], "new_development_goal_not_historical")
        self.assertEqual(scene["dependencies"], [])

    def test_09_new_person_compiles_after_existing_people_without_mutation(self):
        result = merge_root_patch(self.prepared, independent_new_patch(self.bundle, new_people=True, count=2))
        self.assertTrue(result["valid"], result)
        old_people = [person for row in self.prepared["original"]["new_development_candidates"]
                      for person in row["new_people"]]
        scenes = result["merged"]["added_phases"]
        self.assertEqual(scenes[0]["scene"]["new_people"][0]["person_ref"], f"NP{len(old_people) + 1:02d}")
        self.assertEqual(scenes[1]["scene"]["new_people"][0]["person_ref"], f"NP{len(old_people) + 2:02d}")
        self.assertIn("MockVisitor1", scenes[0]["scene"]["required_participants"])
        self.assertEqual(result["merged"]["original_plan"], self.bundle["original_plan"])

    def test_10_new_person_identity_and_name_collisions_reject(self):
        speaker = build_speaker_registry(self.bundle["parent"])["speakers"][0]
        old_names = [person["name"] for row in self.prepared["original"]["new_development_candidates"]
                     for person in row["new_people"]]
        for name in ["SP999", "NP01", "user", speaker["canonical_name"], *old_names]:
            patch = independent_new_patch(self.bundle, new_people=True)
            patch["new_developments"][0]["candidate"]["new_people"][0]["name"] = name
            self.assertFalse(merge_root_patch(self.prepared, patch)["valid"], name)

    def test_11_unknown_speaker_and_actor_or_no_parent_cannot_pass(self):
        for field in ("speaker", "actor", "no_parent"):
            patch = independent_new_patch(self.bundle, new_people=True)
            candidate = patch["new_developments"][0]["candidate"]
            if field == "speaker":
                candidate["speaker_refs"] = ["SP999"]
            elif field == "actor":
                candidate["actor_bindings"][0]["actor_ref"] = "NP01"
            else:
                candidate["speaker_refs"] = ["MockVisitor1"]
            self.assertFalse(merge_root_patch(self.prepared, patch)["valid"], field)

    def test_12_new_dependencies_prior_only_and_cross_branch_ids_unique(self):
        patch = independent_new_patch(self.bundle, count=2)
        patch["new_developments"][1]["dependencies"] = [{"phase_id": "phase:independent_1", "condition": "label recorded"}]
        self.assertTrue(merge_root_patch(self.prepared, patch)["valid"])
        for ref in ("phase:independent_2", "phase:unknown", "initial:ND09:C01"):
            bad = deepcopy(patch)
            bad["new_developments"][0]["dependencies"] = [{"phase_id": ref, "condition": "label recorded"}]
            self.assertFalse(merge_root_patch(self.prepared, bad)["valid"], ref)
        patch["phases"] = existing_phase_patch(self.bundle, ("ME03:C01",))["phases"]
        patch["new_developments"][0]["phase_id"] = patch["phases"][0]["phase_id"]
        self.assertFalse(merge_root_patch(self.prepared, patch)["valid"])

    def test_13_new_pool_limit_is_eight_not_quota(self):
        for count in (1, 8):
            self.assertTrue(merge_root_patch(self.prepared, independent_new_patch(self.bundle, count=count))["valid"])
        self.assertIn("root_patch_new_developments_max_8",
                      merge_root_patch(self.prepared, independent_new_patch(self.bundle, count=9))["errors"])

    def test_14_no_supplement_and_new_only_status_are_explicit(self):
        patch = independent_new_patch(self.bundle, count=0)
        patch["status"] = "NO_SUPPLEMENT"
        self.assertTrue(merge_root_patch(self.prepared, patch)["valid"])
        patch["status"] = "PATCH"
        self.assertIn("root_patch_empty_status_mismatch", merge_root_patch(self.prepared, patch)["errors"])

    def test_15_strict_candidate_shape_does_not_accept_fake_source_or_legacy_names(self):
        for field in ("source_candidate_id", "original_goal", "participants"):
            patch = independent_new_patch(self.bundle)
            patch["new_developments"][0]["candidate"][field] = "fake"
            self.assertIn("root_new_development_1_candidate_shape", merge_root_patch(self.prepared, patch)["errors"])

    def test_16_max_two_people_and_mechanical_binding_contract_preserved(self):
        patch = independent_new_patch(self.bundle, new_people=True)
        candidate = patch["new_developments"][0]["candidate"]
        candidate["new_people"] *= 3
        self.assertFalse(merge_root_patch(self.prepared, patch)["valid"])
        patch = independent_new_patch(self.bundle)
        patch["new_developments"][0]["candidate"]["actor_bindings"] = []
        self.assertFalse(merge_root_patch(self.prepared, patch)["valid"])

    def test_17_scope_explicit_and_never_legacy_implicit(self):
        bundle = deepcopy(self.bundle)
        bundle.pop("incremental_scope")
        with self.assertRaises(ValueError):
            prepare_root_original(bundle)
        with self.assertRaises(ValueError):
            prepare_root_original(failed_bundle())

    def test_18_materialization_cannot_swap_parent_after_scope_check(self):
        parent = deepcopy(self.bundle["parent"])
        parent["root_id"] = "foreign-root"
        self.assertEqual(merge_root_patch(self.prepared, independent_new_patch(self.bundle), parent=parent)["errors"],
                         ["root_patch_source_scope_mismatch"])

    def test_19_bad_declaration_shapes_reject_without_exception(self):
        for value in (None, "not an array", [None], [{"name": "MockVisitor"}]):
            patch = independent_new_patch(self.bundle)
            patch["new_developments"][0]["candidate"]["new_people"] = value
            self.assertFalse(merge_root_patch(self.prepared, patch)["valid"], value)

    def test_20_registry_file_must_belong_to_original_planner_run(self):
        bundle = deepcopy(self.bundle)
        bundle["shared_feedback_lineage"]["sources"]["original_request_manifest"]["path"] = (
            "/tmp/foreign-root/planner/actual_request/request_manifest.json")
        with self.assertRaisesRegex(ValueError, "original_run_source_association"):
            prepare_root_original(bundle)


if __name__ == "__main__":
    unittest.main()
