from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.binder import bind_scene
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_fresh_e2e import (
    _run_version_and_summary_stem,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.memory import (
    commit_confirmed_delta,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.prompts import (
    NATURAL_REALIZER_CLAIM_ONLY_RETRY_PROMPT_VERSION,
    build_evidence_committer_prompt,
    build_realizer_claim_only_retry_prompt,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    audit_legacy_import_isolation,
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.realizer_claims import (
    reconstruct_scene_from_claim_only,
    validate_claim_only_retry_contract,
    validate_committer_candidate_subset,
    validate_realizer_scene_contract,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.realizer_claim_only_retry_v1_audit import (
    build_shadow_audit,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.schemas import (
    EVIDENCE_COMMITTER_SCHEMA_VERSION,
    REALIZER_SCHEMA_VERSION,
    validate_realizer_claim_only_output,
)


PROJECT = Path(__file__).resolve().parents[1]
HISTORICAL_V5_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v5")
V5_TREE_SHA256 = "61cfcd39623b0dd0c440df117f0f47639831b24b93b431923b3a3f06fcf94db8"


def _claim(delta_id: str, status: str, keys: list[str]) -> dict:
    return {
        "claimed_graph_updates": [{
            "proposed_delta_id": delta_id,
            "claimed_status": status,
            "candidate_evidence_turn_keys": keys,
        }],
        "new_open_thread_claims": [],
    }


def _decision(delta_id: str, decision: str, keys: list[str]) -> dict:
    semantic = {
        "confirmed": "semantically_verified",
        "rejected": "rejected",
        "uncertain": "model_claimed",
    }[decision]
    return {
        "evidence_committer_schema_version": EVIDENCE_COMMITTER_SCHEMA_VERSION,
        "proposed_delta_id": delta_id,
        "decision": decision,
        "confirmed_post_state": "reflection_open" if decision == "confirmed" else "",
        "evidence_turn_keys": keys,
        "reason": "Offline contract fixture.",
        "semantic_status": semantic,
    }


class RealizerClaimOnlyRetryV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads((HISTORICAL_V5_RUN / "planner_output.json").read_text())
        cls.compiled = json.loads((HISTORICAL_V5_RUN / "compiled_deltas.json").read_text())
        cls.memory = json.loads((HISTORICAL_V5_RUN / "world_memory_initial.json").read_text())
        cls.delta = cls.compiled["scenes"][2]["proposed_graph_deltas"][0]
        plan_scene = cls.plan["scene_goals"][2]
        cls.scene = {
            "scene_key": "scene_03",
            "goal": plan_scene["goal"],
            "participants": plan_scene["participants"],
            "related_memory_ids": cls.delta["related_memory_ids"],
        }
        cls.historical = [json.loads(
            (HISTORICAL_V5_RUN / f"attempts/realizer-scene_03-a{i:02d}.json").read_text()
        )["parsed_response"] for i in (1, 2, 3)]
        cls.frozen_turns = copy.deepcopy(cls.historical[0]["dialogue_turns"])
        cls.valid_keys = {turn["turn_key"] for turn in cls.frozen_turns}
        cls.delta_id = cls.delta["proposed_delta_id"]

    def _full_gate(self, output: dict) -> dict:
        return validate_realizer_scene_contract(
            output, scene=self.scene, accepted_deltas=[self.delta],
            expected_root_id=self.memory["root_id"])

    def _claim_gate(self, output: dict) -> dict:
        return validate_claim_only_retry_contract(
            output, frozen_scene_key=self.scene["scene_key"],
            frozen_dialogue_turns=self.frozen_turns,
            accepted_deltas=[self.delta])

    def test_three_v5_dialogues_are_valid_but_claims_are_not(self) -> None:
        for output in self.historical:
            result = self._full_gate(output)
            self.assertTrue(result["dialogue_valid"], result)
            self.assertFalse(result["claims_valid"], result)
            self.assertEqual(result["retry_mode"], "claim_only_retry")
            self.assertTrue(any("evidence_not_in_current_scene" in error
                                for error in result["claim_errors"]))

    def test_invalid_dialogue_routes_to_full_scene_retry(self) -> None:
        output = copy.deepcopy(self.historical[0])
        output["dialogue_turns"][0]["text"] = ""
        result = self._full_gate(output)
        self.assertFalse(result["dialogue_valid"])
        self.assertEqual(result["retry_mode"], "full_scene_retry")

    def test_claim_only_prompt_freezes_dialogue_and_omits_rejected_response(self) -> None:
        errors = self._full_gate(self.historical[0])["errors"]
        prompt = build_realizer_claim_only_retry_prompt(
            scene_key="scene_03", frozen_dialogue_turns=self.frozen_turns,
            proposed_delta_ids=[self.delta_id], errors=errors)
        self.assertIn(NATURAL_REALIZER_CLAIM_ONLY_RETRY_PROMPT_VERSION, prompt)
        self.assertIn("valid and frozen", prompt)
        self.assertNotIn("cc-train-episode-61555:P05:T016", prompt)
        self.assertNotIn("Previous rejected response", prompt)
        self.assertEqual(set(json.loads(prompt.splitlines()[-1])), {
            "claimed_graph_updates", "new_open_thread_claims"})
        for key in self.valid_keys:
            self.assertIn(key, prompt)

    def test_claim_only_output_cannot_return_dialogue_or_scene(self) -> None:
        for field in ("dialogue_turns", "scene_key", "operation", "schema"):
            output = _claim(self.delta_id, "realized", [next(iter(self.valid_keys))])
            output[field] = [] if field == "dialogue_turns" else "forbidden"
            self.assertFalse(self._claim_gate(output)["valid"], field)

    def test_allowed_evidence_keys_come_only_from_current_scene(self) -> None:
        result = self._claim_gate(
            _claim(self.delta_id, "realized", [next(iter(self.valid_keys))]))
        self.assertEqual(set(result["allowed_evidence_turn_keys"]), self.valid_keys)
        self.assertNotIn("cc-train-episode-61555:P05:T016",
                         result["allowed_evidence_turn_keys"])

    def test_realized_requires_nonempty_exact_current_evidence(self) -> None:
        self.assertTrue(self._claim_gate(
            _claim(self.delta_id, "realized", [next(iter(self.valid_keys))]))["valid"])
        self.assertFalse(self._claim_gate(_claim(self.delta_id, "realized", []))["valid"])
        self.assertFalse(self._claim_gate(
            _claim(self.delta_id, "realized", ["missing"]))["valid"])

    def test_not_realized_requires_empty_evidence(self) -> None:
        self.assertTrue(self._claim_gate(
            _claim(self.delta_id, "not_realized", []))["valid"])
        self.assertFalse(self._claim_gate(_claim(
            self.delta_id, "not_realized", [next(iter(self.valid_keys))]))["valid"])

    def test_uncertain_allows_empty_or_exact_evidence_only(self) -> None:
        self.assertTrue(self._claim_gate(_claim(self.delta_id, "uncertain", []))["valid"])
        self.assertTrue(self._claim_gate(_claim(
            self.delta_id, "uncertain", [next(iter(self.valid_keys))]))["valid"])
        self.assertFalse(self._claim_gate(
            _claim(self.delta_id, "uncertain", ["missing"]))["valid"])

    def test_parent_other_scene_and_final_binder_ids_are_rejected(self) -> None:
        forbidden = [
            "cc-train-episode-61555:P05:T016",
            "other_scene:t01",
            "cc-train-episode-61555:child:R01:scene_03:session:T001",
        ]
        for key in forbidden:
            self.assertFalse(self._claim_gate(
                _claim(self.delta_id, "realized", [key]))["valid"], key)

    def test_duplicate_delta_claim_routes_to_claim_only_retry(self) -> None:
        output = copy.deepcopy(self.historical[0])
        key = output["dialogue_turns"][0]["turn_key"]
        output["claimed_graph_updates"] = [
            _claim(self.delta_id, "realized", [key])["claimed_graph_updates"][0],
            _claim(self.delta_id, "realized", [key])["claimed_graph_updates"][0],
        ]
        result = self._full_gate(output)
        self.assertTrue(result["dialogue_valid"])
        self.assertFalse(result["claims_valid"])
        self.assertEqual(result["retry_mode"], "claim_only_retry")

    def test_illegal_key_is_neither_replaced_nor_deleted(self) -> None:
        output = _claim(self.delta_id, "realized", ["illegal"])
        original = copy.deepcopy(output)
        self.assertFalse(self._claim_gate(output)["valid"])
        self.assertEqual(output, original)

    def test_reconstruction_preserves_dialogue_field_for_field(self) -> None:
        repaired = _claim(
            self.delta_id, "realized", sorted(self.valid_keys))
        self.assertTrue(self._claim_gate(repaired)["valid"])
        combined = reconstruct_scene_from_claim_only(
            frozen_scene_key="scene_03", frozen_dialogue_turns=self.frozen_turns,
            claim_only_output=repaired)
        self.assertEqual(combined["dialogue_turns"], self.frozen_turns)
        self.assertTrue(self._full_gate(combined)["valid"])

    def test_committer_can_only_return_candidate_subset(self) -> None:
        candidates = [self.frozen_turns[0]["turn_key"]]
        valid = validate_committer_candidate_subset(
            _decision(self.delta_id, "confirmed", candidates),
            candidate_evidence_turn_keys=candidates,
            realizer_claimed_status="realized")
        self.assertTrue(valid["valid"], valid)
        invalid = validate_committer_candidate_subset(
            _decision(self.delta_id, "confirmed", [self.frozen_turns[1]["turn_key"]]),
            candidate_evidence_turn_keys=candidates,
            realizer_claimed_status="realized")
        self.assertFalse(invalid["valid"])

    def test_not_realized_cannot_be_confirmed_by_committer(self) -> None:
        result = validate_committer_candidate_subset(
            _decision(self.delta_id, "confirmed", []),
            candidate_evidence_turn_keys=[], realizer_claimed_status="not_realized")
        self.assertFalse(result["valid"])
        self.assertIn("committer_cannot_confirm_not_realized_claim", result["errors"])

    def test_committer_prompt_exposes_candidate_boundary(self) -> None:
        key = self.frozen_turns[0]["turn_key"]
        prompt = build_evidence_committer_prompt(
            proposed_delta=self.delta, dialogue_turns=self.frozen_turns,
            realizer_claims=[_claim(
                self.delta_id, "realized", [key])["claimed_graph_updates"][0]],
            memory_view=self.memory)
        self.assertIn("allowed_candidate_evidence_turn_keys", prompt)
        self.assertIn("only a subset", prompt)

    def test_only_confirmed_verified_decision_commits(self) -> None:
        key = self.frozen_turns[0]["turn_key"]
        claims = _claim(self.delta_id, "realized", [key])
        scene_output = reconstruct_scene_from_claim_only(
            frozen_scene_key="scene_03", frozen_dialogue_turns=self.frozen_turns,
            claim_only_output=claims)
        confirmed = _decision(self.delta_id, "confirmed", [key])
        binding = bind_scene(
            root_id=self.memory["root_id"], round_index=1,
            scene_goal=self.scene, realizer_output=scene_output,
            deltas=[self.delta], committer_outputs=[confirmed])
        updated = commit_confirmed_delta(
            self.memory, self.delta, confirmed, binding,
            scene_key="scene_03", round_index=1, dry_run=True)
        self.assertEqual(len(updated["open_threads"]), 1)
        for name in ("rejected", "uncertain"):
            decision = _decision(self.delta_id, name, [] if name == "rejected" else [key])
            pending_binding = bind_scene(
                root_id=self.memory["root_id"], round_index=1,
                scene_goal=self.scene, realizer_output=scene_output,
                deltas=[self.delta], committer_outputs=[decision])
            pending = commit_confirmed_delta(
                self.memory, self.delta, decision, pending_binding,
                scene_key="scene_03", round_index=1, dry_run=True)
            self.assertEqual(pending["open_threads"], [])
            self.assertFalse(pending["memory_update_log"][-1]["authoritative_applied"])

    def test_claim_only_retry_exhaustion_remains_claim_reference_failure(self) -> None:
        results = [self._full_gate(output) for output in self.historical]
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result["retry_mode"] == "claim_only_retry"
                            for result in results))
        failure_stage = (
            "Natural Realizer claim/reference"
            if results[-1]["dialogue_valid"] and not results[-1]["claims_valid"]
            else "Natural Realizer structure")
        self.assertEqual(failure_stage, "Natural Realizer claim/reference")

    def test_strict_claim_only_schema_has_exact_two_fields(self) -> None:
        output = _claim(self.delta_id, "uncertain", [])
        result = validate_realizer_claim_only_output(
            output, valid_turn_keys=self.valid_keys,
            expected_delta_ids={self.delta_id})
        self.assertTrue(result["valid"])
        output["previous_errors"] = []
        self.assertFalse(validate_realizer_claim_only_output(
            output, valid_turn_keys=self.valid_keys,
            expected_delta_ids={self.delta_id})["valid"])

    def test_future_summary_name_uses_actual_run_version(self) -> None:
        version, stem = _run_version_and_summary_stem(Path("example_fresh_e2e_v6"))
        self.assertEqual(version, "v6")
        self.assertEqual(stem, "slim_fresh_e2e_v6_summary")

    def test_zero_api_shadow_preflight_passes(self) -> None:
        audit = build_shadow_audit()
        self.assertTrue(audit["ready"], audit["checks"])
        self.assertEqual(audit["api_calls"], 0)

    def test_legacy_validator_import_is_zero(self) -> None:
        package = PROJECT / "scripts/current_method/restart_n5_spell/world_memory_slim_v1"
        self.assertTrue(audit_legacy_import_isolation(package)["isolated"])

    def test_historical_v5_run_is_unchanged(self) -> None:
        self.assertEqual(hash_tree(HISTORICAL_V5_RUN)["tree_sha256"], V5_TREE_SHA256)


if __name__ == "__main__":
    unittest.main()
