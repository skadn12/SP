from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.core import sha256_file
from scripts.current_method.restart_n5_spell.input_adapter import dialogue_content_hash
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.affordance_compiler import (
    compile_delta_affordances,
    compile_selected_actions,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.binder import bind_scene
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.memory import (
    commit_confirmed_delta,
    commit_emergent_thread,
    initialize_world_memory,
    stage_emergent_claims,
    stage_model_claims,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    audit_legacy_import_isolation,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.realizer_claims import (
    build_emergent_committer_routes,
    build_planned_committer_routes,
    compile_emergent_pending_proposals,
    validate_realizer_scene_contract,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.schemas import (
    EVIDENCE_COMMITTER_SCHEMA_VERSION,
    PLANNER_SCHEMA_VERSION,
    REALIZER_SCHEMA_VERSION,
    validate_natural_realizer_output,
)


PROJECT = Path(__file__).resolve().parents[1]
HISTORICAL_V2_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v2")


def _tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(row for row in path.rglob("*") if row.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def _fixture() -> tuple[dict, dict, dict, dict]:
    root_id = "claim-contract-root"
    turns = [{
        "turn_id": f"{root_id}:P01:T001", "session_id": f"{root_id}:S01",
        "speaker": "A", "text": "Source A.", "partition": "parent",
        "source_id": root_id, "source_session_index": 1, "source_turn_index": 1,
    }, {
        "turn_id": f"{root_id}:P01:T002", "session_id": f"{root_id}:S01",
        "speaker": "B", "text": "Source B.", "partition": "parent",
        "source_id": root_id, "source_session_index": 1, "source_turn_index": 2,
    }]
    dialogue_hash = dialogue_content_hash(turns)
    root = {
        "root_id": root_id,
        "dialogue": {"dialogue_hash": dialogue_hash, "turns": turns},
        "source": {"source_path": "/readonly", "source_record_sha256": "0" * 64},
    }
    parent = {
        "sample_id": root_id, "lineage_group": root_id,
        "dialogue": {
            "dialogue_hash": dialogue_hash,
            "turns": [{key: row[key] for key in
                       ("turn_id", "session_id", "speaker", "text", "partition")}
                      for row in turns],
        },
    }
    memory = initialize_world_memory(root, parent)
    menu = compile_delta_affordances(memory, root_identity=root_id)
    open_action = next(row for row in menu["actions"]
                       if row["operation"] == "open_thread")
    partner = next(row for row in menu["actions"]
                   if row["action_id"] != open_action["action_id"])
    selected = [open_action, partner]
    plan = {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "selected_memory_threads": list(dict.fromkeys(
            row["thread_id"] for row in selected)),
        "scene_budget": dict(menu["scene_budget"]),
        "scene_goals": [{
            "goal": f"Goal {index}",
            "participants": list(action["eligible_participants"]),
            "selected_action_ids": [action["action_id"]],
        } for index, action in enumerate(selected, 1)],
    }
    compiled = compile_selected_actions(plan, menu)
    scene = compiled["scene_goals"][0]
    delta = scene["proposed_graph_deltas"][0]
    assert delta["operation"] == "open_thread"
    return memory, menu, scene, delta


def _output(scene: dict, delta: dict, *, emergent: bool = False) -> dict:
    return {
        "realizer_schema_version": REALIZER_SCHEMA_VERSION,
        "scene_key": scene["scene_key"],
        "dialogue_turns": [{
            "turn_key": "t01", "speaker": scene["participants"][0],
            "text": "A natural follow-up is discussed.",
        }],
        "claimed_graph_updates": [{
            "proposed_delta_id": delta["proposed_delta_id"],
            "claimed_status": "realized",
            "candidate_evidence_turn_keys": ["t01"],
        }],
        "new_open_thread_claims": ([{
            "description": "Discuss moving arrangements next week.",
            "candidate_evidence_turn_keys": ["t01"],
            "related_proposed_delta_id": None,
        }] if emergent else []),
    }


def _decision(identity: str, decision: str) -> dict:
    status = {"confirmed": "semantically_verified", "rejected": "rejected",
              "uncertain": "model_claimed"}[decision]
    return {
        "evidence_committer_schema_version": EVIDENCE_COMMITTER_SCHEMA_VERSION,
        "proposed_delta_id": identity,
        "decision": decision,
        "confirmed_post_state": ("A follow-up remains open."
                                 if decision == "confirmed" else ""),
        "evidence_turn_keys": ["t01"],
        "reason": "Independent fixture decision.",
        "semantic_status": status,
    }


class RealizerClaimContractV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory, self.menu, self.scene, self.delta = _fixture()

    def _gate(self, output: dict, deltas: list[dict] | None = None) -> dict:
        return validate_realizer_scene_contract(
            output, scene=self.scene, accepted_deltas=deltas or [self.delta],
            expected_root_id=self.memory["root_id"])

    def test_planned_open_thread_only_claimed_graph_update_passes(self) -> None:
        output = _output(self.scene, self.delta)
        result = self._gate(output)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["emergent_claim_count"], 0)

    def test_new_open_thread_claims_empty_or_omitted_is_valid(self) -> None:
        output = _output(self.scene, self.delta)
        self.assertTrue(validate_natural_realizer_output(output)["valid"])
        del output["new_open_thread_claims"]
        result = self._gate(output)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["normalized_output"]["new_open_thread_claims"], [])

    def test_realized_is_valid_and_completed_is_forbidden_in_v2(self) -> None:
        output = _output(self.scene, self.delta)
        self.assertTrue(self._gate(output)["valid"])
        output["claimed_graph_updates"][0]["claimed_status"] = "completed"
        self.assertFalse(self._gate(output)["valid"])

    def test_nonempty_emergent_open_thread_is_structurally_valid(self) -> None:
        output = _output(self.scene, self.delta, emergent=True)
        result = self._gate(output)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["emergent_claim_count"], 1)

    def test_emergent_related_planned_delta_is_rejected(self) -> None:
        output = _output(self.scene, self.delta, emergent=True)
        output["new_open_thread_claims"][0]["related_proposed_delta_id"] = \
            self.delta["proposed_delta_id"]
        self.assertFalse(self._gate(output)["valid"])

    def test_emergent_claim_cannot_carry_formal_memory_or_thread_ids(self) -> None:
        for field in ("thread_id", "memory_id", "root_id"):
            output = _output(self.scene, self.delta, emergent=True)
            output["new_open_thread_claims"][0][field] = "invented"
            self.assertFalse(self._gate(output)["valid"], field)

    def test_emergent_missing_or_other_scene_evidence_is_rejected(self) -> None:
        for key in ("missing", "other_scene:t01"):
            output = _output(self.scene, self.delta, emergent=True)
            output["new_open_thread_claims"][0]["candidate_evidence_turn_keys"] = [key]
            self.assertFalse(self._gate(output)["valid"], key)

    def test_duplicate_planned_claim_is_rejected(self) -> None:
        output = _output(self.scene, self.delta)
        output["claimed_graph_updates"].append(
            copy.deepcopy(output["claimed_graph_updates"][0]))
        result = self._gate(output)
        self.assertIn(
            f"realizer_claim_duplicate_delta:{self.delta['proposed_delta_id']}",
            result["errors"])

    def test_planned_claim_requires_existing_current_scene_evidence(self) -> None:
        for keys in ([], ["missing"], ["other_scene:t01"]):
            output = _output(self.scene, self.delta)
            output["claimed_graph_updates"][0]["candidate_evidence_turn_keys"] = keys
            self.assertFalse(self._gate(output)["valid"], keys)

    def test_unknown_or_other_scene_delta_is_rejected(self) -> None:
        for identity in ("claim-contract-root:D9999", "other-root:D0001"):
            output = _output(self.scene, self.delta)
            output["claimed_graph_updates"][0]["proposed_delta_id"] = identity
            result = self._gate(output)
            self.assertTrue(any("not_current_scene_delta" in row
                                for row in result["errors"]), result)

    def test_planned_route_contains_claim_status_and_exact_evidence(self) -> None:
        output = _output(self.scene, self.delta)
        routes = build_planned_committer_routes(
            realizer_output=output, deltas=[self.delta], memory_view=self.memory)
        self.assertEqual(routes[0]["realizer_claimed_status"], "realized")
        self.assertEqual(routes[0]["candidate_evidence_turns"][0]["turn_key"], "t01")
        self.assertIn(self.delta["proposed_delta_id"], routes[0]["prompt"])

    def test_planned_thread_id_and_commit_require_confirmed_decision(self) -> None:
        output = _output(self.scene, self.delta)
        confirmed = _decision(self.delta["proposed_delta_id"], "confirmed")
        binding = bind_scene(
            root_id=self.memory["root_id"], round_index=1,
            scene_goal=self.scene, realizer_output=output, deltas=[self.delta],
            committer_outputs=[confirmed])
        self.assertEqual(binding["allocated_threads"][0]["thread_id"], "thread_0001")
        updated = commit_confirmed_delta(
            self.memory, self.delta, confirmed, binding,
            scene_key=self.scene["scene_key"], round_index=1)
        self.assertEqual(len(updated["open_threads"]), 1)
        self.assertEqual(updated["open_threads"][0]["provenance"]["formal_thread_id"],
                         "thread_0001")

    def test_rejected_and_uncertain_planned_claims_never_commit(self) -> None:
        output = _output(self.scene, self.delta)
        for decision_name in ("rejected", "uncertain"):
            decision = _decision(self.delta["proposed_delta_id"], decision_name)
            binding = bind_scene(
                root_id=self.memory["root_id"], round_index=1,
                scene_goal=self.scene, realizer_output=output, deltas=[self.delta],
                committer_outputs=[decision])
            self.assertEqual(binding["allocated_threads"], [])
            updated = commit_confirmed_delta(
                self.memory, self.delta, decision, binding,
                scene_key=self.scene["scene_key"], round_index=1)
            self.assertEqual(updated["open_threads"], [])
            self.assertFalse(updated["memory_update_log"][0]["authoritative_applied"])

    def test_realizer_claimed_status_alone_only_stages_pending(self) -> None:
        staged = stage_model_claims(
            self.memory, [self.delta], _output(self.scene, self.delta))
        self.assertEqual(staged["open_threads"], [])
        self.assertEqual(staged["memory_version"], self.memory["memory_version"])
        self.assertEqual(staged["pending_memory_updates"][0]["semantic_status"],
                         "model_claimed")

    def test_emergent_route_stays_pending_until_confirmed(self) -> None:
        output = _output(self.scene, self.delta, emergent=True)
        proposals = compile_emergent_pending_proposals(
            root_id=self.memory["root_id"], scene=self.scene,
            realizer_output=output, memory=self.memory)
        staged = stage_emergent_claims(self.memory, proposals)
        self.assertEqual(staged["open_threads"], [])
        self.assertIsNone(staged["pending_memory_updates"][0]["formal_thread_id"])
        routes = build_emergent_committer_routes(
            proposals=proposals, realizer_output=output, memory_view=self.memory)
        self.assertEqual(routes[0]["route_type"], "emergent_open_thread")
        self.assertIsNone(routes[0]["formal_thread_id"])
        self.assertIn(proposals[0]["proposal_id"], routes[0]["prompt"])

    def test_confirmed_emergent_thread_gets_program_id_and_commits(self) -> None:
        output = _output(self.scene, self.delta, emergent=True)
        proposals = compile_emergent_pending_proposals(
            root_id=self.memory["root_id"], scene=self.scene,
            realizer_output=output, memory=self.memory)
        confirmed = _decision(proposals[0]["proposal_id"], "confirmed")
        planned_uncertain = _decision(self.delta["proposed_delta_id"], "uncertain")
        binding = bind_scene(
            root_id=self.memory["root_id"], round_index=1,
            scene_goal=self.scene, realizer_output=output, deltas=[self.delta],
            committer_outputs=[planned_uncertain], emergent_proposals=proposals,
            emergent_committer_outputs=[confirmed])
        thread = next(row for row in binding["allocated_threads"]
                      if row["proposal_id"] == proposals[0]["proposal_id"])
        self.assertEqual(thread["thread_id"], "thread_0001")
        updated = commit_emergent_thread(
            self.memory, proposals[0], confirmed, binding, round_index=1)
        self.assertEqual(updated["open_threads"][0]["provenance"]["formal_thread_id"],
                         "thread_0001")

    def test_unconfirmed_emergent_thread_has_no_formal_id_or_authority(self) -> None:
        output = _output(self.scene, self.delta, emergent=True)
        proposal = compile_emergent_pending_proposals(
            root_id=self.memory["root_id"], scene=self.scene,
            realizer_output=output, memory=self.memory)[0]
        for name in ("rejected", "uncertain"):
            decision = _decision(proposal["proposal_id"], name)
            binding = bind_scene(
                root_id=self.memory["root_id"], round_index=1,
                scene_goal=self.scene, realizer_output=output, deltas=[self.delta],
                committer_outputs=[], emergent_proposals=[proposal],
                emergent_committer_outputs=[decision])
            self.assertEqual(binding["allocated_threads"], [])
            updated = commit_emergent_thread(
                self.memory, proposal, decision, binding, round_index=1)
            self.assertEqual(updated["open_threads"], [])
            self.assertIsNone(updated["pending_memory_updates"][0]["formal_thread_id"])

    def test_contract_has_no_text_similarity_or_keyword_gate(self) -> None:
        source = (PROJECT / "scripts/current_method/restart_n5_spell/"
                  "world_memory_slim_v1/realizer_claims.py").read_text(encoding="utf-8")
        for forbidden in ("SequenceMatcher", "edit_distance", "embedding_similarity",
                          "fuzzy_match", "terminal_keyword"):
            self.assertNotIn(forbidden, source)

    def test_legacy_marker_validator_is_not_imported(self) -> None:
        package = PROJECT / "scripts/current_method/restart_n5_spell/world_memory_slim_v1"
        result = audit_legacy_import_isolation(package)
        self.assertTrue(result["isolated"], result["findings"])

    def test_historical_v2_shadow_inputs_are_read_only(self) -> None:
        before = _tree_sha(HISTORICAL_V2_RUN)
        for path in sorted((HISTORICAL_V2_RUN / "attempts").glob(
                "realizer-scene_01-a*.json")):
            json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(before, _tree_sha(HISTORICAL_V2_RUN))

    def test_three_historical_outputs_pass_read_only_v1_to_v2_shadow_gate(self) -> None:
        compiled = json.loads(
            (HISTORICAL_V2_RUN / "compiled_deltas.json").read_text(encoding="utf-8"))
        plan = json.loads(
            (HISTORICAL_V2_RUN / "planner_output.json").read_text(encoding="utf-8"))
        delta = compiled["scenes"][0]["proposed_graph_deltas"][0]
        scene = {
            "scene_key": "scene_01", "goal": plan["scene_goals"][0]["goal"],
            "participants": plan["scene_goals"][0]["participants"],
            "related_memory_ids": delta["related_memory_ids"],
        }
        paths = sorted((HISTORICAL_V2_RUN / "attempts").glob(
            "realizer-scene_01-a*.json"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            parsed = json.loads(path.read_text(encoding="utf-8"))["parsed_response"]
            result = validate_realizer_scene_contract(
                parsed, scene=scene, accepted_deltas=[delta],
                expected_root_id=delta["root_id"],
                allow_legacy_shadow_migration=True)
            self.assertTrue(result["valid"], result)
            self.assertEqual(
                result["normalized_output"]["claimed_graph_updates"][0][
                    "claimed_status"], "realized")
            self.assertEqual(result["normalized_output"]["new_open_thread_claims"], [])
            self.assertFalse(result["semantic_decision_performed"])


if __name__ == "__main__":
    unittest.main()
