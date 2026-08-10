from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.core import canonical_json, sha256_file
from scripts.current_method.restart_n5_spell.input_adapter import dialogue_content_hash
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.affordance_compiler import (
    AFFORDANCE_SCHEMA_VERSION,
    action_menu_parent_turn_ids,
    compact_action_menu,
    compile_delta_affordances,
    compile_selected_actions,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.binder import (
    StructuralBindingError,
    bind_scene,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.inputs import (
    load_locomo_parent_read_only,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.memory import (
    commit_confirmed_delta,
    initialize_world_memory,
    stage_model_claims,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.preflight import (
    run_preflight_batch,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.prompts import (
    build_slim_planner_prompt,
    build_slim_planner_retry_prompt,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    audit_legacy_import_isolation,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.schemas import (
    EVIDENCE_COMMITTER_SCHEMA_VERSION,
    PLANNER_SCHEMA_VERSION,
    REALIZER_SCHEMA_VERSION,
    validate_natural_realizer_output,
    validate_slim_planner_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural import (
    validate_structural_deltas,
)


PROJECT = Path(__file__).resolve().parents[1]
CC_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots")
CC_IDS = (
    "cc-train-episode-12596", "cc-train-episode-13836",
    "cc-train-episode-146700", "cc-train-episode-178392",
    "cc-train-episode-216742", "cc-train-episode-34406",
    "cc-train-episode-41409", "cc-train-episode-41625",
    "cc-train-episode-61555", "cc-train-episode-91716",
)
LOCOMO_SOURCE = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "05_stage2_repair_topup_from_stage1/train_mix/frontier_train.jsonl")
LOCOMO_ATTESTATION = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_cross_dialogue_calibration_5x3_v1/dialogues/conv-43/questioner/"
    "frozen/parent_source_attestation.json")
HISTORICAL_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v1")


def _root_parent(root_id: str = "root-a") -> tuple[dict, dict]:
    turns = [{
        "turn_id": f"{root_id}:P01:T001", "session_id": f"{root_id}:S01",
        "speaker": "A", "text": "One source utterance.", "partition": "parent",
        "source_id": root_id, "source_session_index": 1, "source_turn_index": 1,
    }, {
        "turn_id": f"{root_id}:P01:T002", "session_id": f"{root_id}:S01",
        "speaker": "B", "text": "Another source utterance.", "partition": "parent",
        "source_id": root_id, "source_session_index": 1, "source_turn_index": 2,
    }]
    root = {
        "root_id": root_id,
        "dialogue": {"dialogue_hash": dialogue_content_hash(turns), "turns": turns},
        "source": {"source_path": "/readonly", "source_record_sha256": "0" * 64},
    }
    parent = {
        "sample_id": root_id, "lineage_group": root_id,
        "dialogue": {
            "dialogue_hash": root["dialogue"]["dialogue_hash"],
            "turns": [{key: turn[key] for key in
                       ("turn_id", "session_id", "speaker", "text", "partition")}
                      for turn in turns],
        },
    }
    return root, parent


def _plan(menu: dict, actions: list[dict] | None = None) -> dict:
    selected = actions or menu["actions"][:2]
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "selected_memory_threads": list(dict.fromkeys(
            row["thread_id"] for row in selected)),
        "scene_budget": dict(menu["scene_budget"]),
        "scene_goals": [{
            "goal": f"Natural goal {index}",
            "participants": list(action["eligible_participants"]),
            "selected_action_ids": [action["action_id"]],
        } for index, action in enumerate(selected, 1)],
    }


def _realizer(scene: dict, delta: dict, text: str = "Natural test turn.") -> dict:
    return {
        "realizer_schema_version": REALIZER_SCHEMA_VERSION,
        "scene_key": scene["scene_key"],
        "dialogue_turns": [{
            "turn_key": "t01", "speaker": scene["participants"][0], "text": text,
        }],
        "claimed_graph_updates": [{
            "proposed_delta_id": delta["proposed_delta_id"],
            "claimed_status": "realized", "candidate_evidence_turn_keys": ["t01"],
        }],
        "new_open_thread_claims": [],
    }


def _confirmed(delta: dict, evidence_key: str = "t01") -> dict:
    return {
        "evidence_committer_schema_version": EVIDENCE_COMMITTER_SCHEMA_VERSION,
        "proposed_delta_id": delta["proposed_delta_id"], "decision": "confirmed",
        "confirmed_post_state": delta["intended_post_state"],
        "evidence_turn_keys": [evidence_key], "reason": "External semantic decision",
        "semantic_status": "semantically_verified",
    }


def _rejected(delta: dict) -> dict:
    return {
        "evidence_committer_schema_version": EVIDENCE_COMMITTER_SCHEMA_VERSION,
        "proposed_delta_id": delta["proposed_delta_id"], "decision": "rejected",
        "confirmed_post_state": "", "evidence_turn_keys": ["t01"],
        "reason": "Independent semantic rejection", "semantic_status": "rejected",
    }


def _tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(row for row in path.rglob("*") if row.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


class WorldMemorySlimAffordanceV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        root, parent = _root_parent()
        self.root = root
        self.parent = parent
        self.memory = initialize_world_memory(root, parent)
        self.menu = compile_delta_affordances(
            self.memory, root_identity=self.memory["root_id"])
        self.plan = _plan(self.menu)
        self.compiled = compile_selected_actions(self.plan, self.menu)
        self.scene = self.compiled["scene_goals"][0]
        self.delta = self.scene["proposed_graph_deltas"][0]

    def test_scene_and_delta_ids_are_stable_and_program_owned(self) -> None:
        again = compile_selected_actions(self.plan, self.menu)
        self.assertEqual(
            [row["scene_key"] for row in self.compiled["scene_goals"]],
            ["scene_01", "scene_02"])
        self.assertEqual(canonical_json(self.compiled), canonical_json(again))
        self.assertEqual(self.delta["delta_id"], "D0001")
        self.assertEqual(self.delta["proposed_delta_id"], "root-a:D0001")
        self.assertEqual(self.delta["operation_class"], "create")

    def test_planner_schema_rejects_program_fields(self) -> None:
        for key, value in (
                ("scene_key", "scene_99"),
                ("proposed_graph_deltas", []),
                ("pre_state", "invented")):
            bad = copy.deepcopy(self.plan)
            bad["scene_goals"][0][key] = value
            result = validate_slim_planner_output(bad, action_menu=self.menu)
            self.assertFalse(result["valid"], key)
            self.assertTrue(any("planner_program_field" in error
                                for error in result["errors"]), result)

    def test_planner_can_only_select_existing_action_id(self) -> None:
        bad = copy.deepcopy(self.plan)
        bad["scene_goals"][0]["selected_action_ids"] = ["A9999"]
        result = validate_slim_planner_output(bad, action_menu=self.menu)
        self.assertIn("selected_action_unknown:A9999", result["errors"])

    def test_trust_and_authority_actions_require_existing_state(self) -> None:
        operations = {row["operation"] for row in self.menu["actions"]}
        self.assertNotIn("change_trust", operations)
        self.assertNotIn("change_authority", operations)

    def test_add_shared_task_is_create_and_does_not_require_pre_state(self) -> None:
        action = next(row for row in self.menu["actions"]
                      if row["operation"] == "add_shared_task")
        partner = next(row for row in self.menu["actions"]
                       if row["action_id"] != action["action_id"])
        compiled = compile_selected_actions(_plan(self.menu, [action, partner]), self.menu)
        delta = compiled["scene_goals"][0]["proposed_graph_deltas"][0]
        self.assertIsNone(delta["pre_state"])
        result = validate_structural_deltas(
            self.memory, [delta], expected_root_id="root-a", action_menu=self.menu)
        self.assertTrue(result["structurally_valid"], result["errors"])

    def test_update_state_binds_authoritative_current_state(self) -> None:
        memory = copy.deepcopy(self.memory)
        state = copy.deepcopy(memory["event_history"][0])
        state.update({
            "memory_id": "root-a:STATE001", "event_identity_key": "root-a:state:one",
            "record_type": "state", "value": "actual", "semantic_status": "source_grounded",
        })
        memory["current_states"].append(state)
        menu = compile_delta_affordances(memory, root_identity="root-a", maximum_actions=20)
        action = next(row for row in menu["actions"] if row["operation"] == "update_state")
        partner = next(row for row in menu["actions"]
                       if row["action_id"] != action["action_id"])
        compiled = compile_selected_actions(_plan(menu, [action, partner]), menu)
        delta = compiled["scene_goals"][0]["proposed_graph_deltas"][0]
        self.assertEqual(delta["target_memory_id"], "root-a:STATE001")
        self.assertEqual(delta["pre_state"], "actual")
        self.assertTrue(validate_structural_deltas(
            memory, [delta], expected_root_id="root-a", action_menu=menu
        )["structurally_valid"])

    def test_pending_and_model_claimed_records_do_not_create_update_affordance(self) -> None:
        memory = copy.deepcopy(self.memory)
        pending = copy.deepcopy(memory["event_history"][0])
        pending.update({
            "memory_id": "root-a:PENDING", "event_identity_key": "root-a:pending",
            "record_type": "trust", "value": "invented", "semantic_status": "model_claimed",
        })
        memory["current_states"].append(pending)
        memory["pending_memory_updates"].append({
            "event_identity_key": "root-a:pending", "semantic_status": "model_claimed"})
        menu = compile_delta_affordances(memory, root_identity="root-a")
        self.assertFalse(any(row["target_memory_id"] == "root-a:PENDING"
                             for row in menu["actions"]))
        self.assertNotIn("change_trust", {row["operation"] for row in menu["actions"]})

    def test_completed_fact_is_not_recompleted_but_can_seed_derived_action(self) -> None:
        memory = copy.deepcopy(self.memory)
        memory["event_history"][0]["status"] = "completed"
        completed_key = memory["event_history"][0]["event_identity_key"]
        menu = compile_delta_affordances(memory, root_identity="root-a", maximum_actions=20)
        self.assertFalse(any(row["event_identity_key"] == completed_key
                             for row in menu["actions"]))
        derived = [row for row in menu["actions"] if row["derived_from_completed"]]
        self.assertTrue(derived)
        self.assertTrue(all(row["event_identity_key"] != completed_key for row in derived))

    def test_same_action_cannot_be_selected_across_scenes(self) -> None:
        bad = copy.deepcopy(self.plan)
        repeated = bad["scene_goals"][0]["selected_action_ids"]
        bad["scene_goals"][1]["selected_action_ids"] = list(repeated)
        bad["scene_goals"][1]["participants"] = list(
            bad["scene_goals"][0]["participants"])
        result = validate_slim_planner_output(bad, action_menu=self.menu)
        self.assertTrue(any("action_selected_more_than_once" in error
                            for error in result["errors"]))

    def test_different_root_compiled_delta_is_rejected(self) -> None:
        bad = copy.deepcopy(self.delta)
        bad["root_id"] = "other-root"
        bad["event_identity_key"] = "other-root:event"
        result = validate_structural_deltas(
            self.memory, [bad], expected_root_id="root-a", action_menu=self.menu)
        self.assertTrue(any("mixed_root" in error for error in result["errors"]))

    def test_prompt_is_compact_and_excludes_full_world_memory(self) -> None:
        turn_ids = set(action_menu_parent_turn_ids(self.menu))
        turns = [row for row in self.parent["dialogue"]["turns"]
                 if row["turn_id"] in turn_ids]
        prompt = build_slim_planner_prompt(
            parent_turns=turns, action_menu=self.menu, role_table=self.memory["entities"])
        self.assertNotIn("world_memory_view", prompt)
        self.assertNotIn("pending_memory_updates", prompt)
        self.assertNotIn("event_history", prompt)
        self.assertIn(AFFORDANCE_SCHEMA_VERSION, prompt)
        self.assertLess(len(prompt), len(canonical_json(self.memory)) + 6000)

    def test_retry_prompt_freezes_menu_without_parent_dialogue(self) -> None:
        prompt = build_slim_planner_retry_prompt(
            action_menu=self.menu, role_table=self.memory["entities"],
            errors=["selected_action_unknown:A9999"])
        self.assertIn("frozen_action_menu_sha256", prompt)
        self.assertNotIn("relevant_parent_turns", prompt)
        self.assertNotIn("turn_id", prompt)
        for action in compact_action_menu(self.menu)["actions"]:
            self.assertIn(action["action_id"], prompt)

    def test_local_pipeline_does_not_scan_modal_or_agreement_words(self) -> None:
        output = _realizer(self.scene, self.delta, "might will if when agreed")
        self.assertTrue(validate_natural_realizer_output(output)["valid"])
        binding = bind_scene(
            root_id="root-a", round_index=1, scene_goal=self.scene,
            realizer_output=output, deltas=[self.delta],
            committer_outputs=[_confirmed(self.delta)])
        updated = commit_confirmed_delta(
            self.memory, self.delta, _confirmed(self.delta), binding,
            scene_key="scene_01", round_index=1)
        self.assertEqual(updated["current_states"][0]["semantic_status"],
                         "semantically_verified")

    def test_model_claimed_does_not_update_authoritative_state(self) -> None:
        staged = stage_model_claims(
            self.memory, [self.delta], _realizer(self.scene, self.delta))
        self.assertEqual(staged["current_states"], self.memory["current_states"])
        self.assertEqual(staged["pending_memory_updates"][0]["semantic_status"],
                         "model_claimed")

    def test_rejected_decision_is_logged_pending_not_authoritative(self) -> None:
        output = _realizer(self.scene, self.delta)
        decision = _rejected(self.delta)
        binding = bind_scene(
            root_id="root-a", round_index=1, scene_goal=self.scene,
            realizer_output=output, deltas=[self.delta],
            committer_outputs=[decision])
        updated = commit_confirmed_delta(
            self.memory, self.delta, decision, binding,
            scene_key="scene_01", round_index=1)
        self.assertEqual(updated["memory_version"], self.memory["memory_version"])
        self.assertEqual(updated["current_states"], self.memory["current_states"])
        self.assertEqual(updated["pending_memory_updates"][0]["semantic_status"],
                         "rejected")
        self.assertFalse(updated["memory_update_log"][0]["authoritative_applied"])

    def test_binder_rejects_nonexistent_evidence_key(self) -> None:
        with self.assertRaisesRegex(StructuralBindingError, "evidence_turn_key_missing"):
            bind_scene(
                root_id="root-a", round_index=1, scene_goal=self.scene,
                realizer_output=_realizer(self.scene, self.delta), deltas=[self.delta],
                committer_outputs=[_confirmed(self.delta, "t99")])

    def test_thread_id_for_open_thread_is_program_owned(self) -> None:
        action = next(row for row in self.menu["actions"]
                      if row["operation"] == "open_thread")
        partner = next(row for row in self.menu["actions"]
                       if row["action_id"] != action["action_id"])
        compiled = compile_selected_actions(_plan(self.menu, [action, partner]), self.menu)
        scene = compiled["scene_goals"][0]
        delta = scene["proposed_graph_deltas"][0]
        output = _realizer(scene, delta)
        binding = bind_scene(
            root_id="root-a", round_index=1, scene_goal=scene,
            realizer_output=output, deltas=[delta],
            committer_outputs=[_confirmed(delta)])
        self.assertEqual(binding["allocated_threads"][0]["thread_id"], "thread_0001")
        self.assertEqual(binding["allocated_handoffs"], [])
        self.assertNotIn("thread_id", output)

    def test_confirmed_role_handoff_id_is_program_owned(self) -> None:
        memory = copy.deepcopy(self.memory)
        responsibility = copy.deepcopy(memory["event_history"][0])
        responsibility.update({
            "memory_id": "root-a:TASK001",
            "event_identity_key": "root-a:task:one",
            "record_type": "responsibility", "value": "assigned",
            "semantic_status": "source_grounded",
        })
        memory["current_states"].append(responsibility)
        menu = compile_delta_affordances(
            memory, root_identity="root-a", maximum_actions=20)
        action = next(row for row in menu["actions"]
                      if row["operation"] == "add_role_handoff")
        partner = next(row for row in menu["actions"]
                       if row["action_id"] != action["action_id"])
        compiled = compile_selected_actions(_plan(menu, [action, partner]), menu)
        scene = compiled["scene_goals"][0]
        delta = scene["proposed_graph_deltas"][0]
        self.assertTrue(validate_structural_deltas(
            memory, [delta], expected_root_id="root-a", action_menu=menu
        )["structurally_valid"])
        output = _realizer(scene, delta)
        binding = bind_scene(
            root_id="root-a", round_index=1, scene_goal=scene,
            realizer_output=output, deltas=[delta],
            committer_outputs=[_confirmed(delta)])
        self.assertEqual(binding["allocated_handoffs"][0]["handoff_id"], "H0001")
        self.assertNotIn("handoff_id", output)

    def test_legacy_marker_validator_is_not_imported(self) -> None:
        package = PROJECT / "scripts/current_method/restart_n5_spell/world_memory_slim_v1"
        audit = audit_legacy_import_isolation(package)
        self.assertTrue(audit["isolated"], audit["findings"])

    def test_parent_and_historical_run_sha_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "parent.json"
            path.write_text(json.dumps({"immutable": True}), encoding="utf-8")
            before = sha256_file(path)
            _ = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(before, sha256_file(path))
        history_before = _tree_sha(HISTORICAL_RUN)
        _ = [json.loads(path.read_text(encoding="utf-8"))
             for path in sorted((HISTORICAL_RUN / "attempts").glob("planner-a*.json"))]
        self.assertEqual(history_before, _tree_sha(HISTORICAL_RUN))

    def test_conv43_record_77_and_attestation_are_both_verified(self) -> None:
        loaded = load_locomo_parent_read_only(LOCOMO_SOURCE, LOCOMO_ATTESTATION)
        audit = loaded["compatibility_audit"]
        self.assertTrue(audit["all_identity_checks_pass"])
        self.assertEqual(audit["source_row_index"], 76)
        self.assertEqual(audit["attested_parent_turn_count"], 237)
        self.assertFalse(loaded["temporary_unified_root_persisted"])

    def test_eleven_root_zero_api_preflight(self) -> None:
        paths = [CC_ROOT / root_id / "unified_root.json" for root_id in CC_IDS]
        result = run_preflight_batch(
            cc_root_paths=paths, locomo_source_path=LOCOMO_SOURCE,
            locomo_attestation_path=LOCOMO_ATTESTATION)
        self.assertTrue(result["slim_pipeline_v2_preflight_ready"])
        self.assertEqual(result["root_count"], 11)
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["request_ledger_summary"]["request_count"], 0)
        self.assertTrue(result["source_inputs_unchanged"])
        self.assertFalse(result["generation_ready"])


if __name__ == "__main__":
    unittest.main()
