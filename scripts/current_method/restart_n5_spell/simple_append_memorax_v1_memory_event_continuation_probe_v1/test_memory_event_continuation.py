from __future__ import annotations

import json
import unittest
from pathlib import Path

from .contracts import (
    build_event_candidates,
    build_person_event_candidates,
    build_speaker_registry,
    build_semantic_gate_context,
    build_planner_outline_pairs,
    compile_planner_output,
    materialize_verified_tracks,
    materialize_memory_only_tracks,
    materialize_planner_output,
    materialize_planner_speaker_refs,
    select_planner_variant,
    normalize_realizer_output,
    validate_planner_outline_distinctness,
    validate_planner_outline_semantic_output,
    select_ranked_legal_candidate_set,
    validate_planner_repair_output,
    validate_planner_output,
    validate_event_verifier_output,
    validate_realizer_plan_isolation,
    validate_semantic_gate_output,
)
from .live_runtime import (
    ALLOWED_LIVE_ROLES,
    PER_ROLE_REQUEST_BUDGET,
    TOTAL_REQUEST_BUDGET,
    validate_outbound_role_coverage,
)
from .payloads import build_add_intent, build_bucket_search_payload, memorax_user_id
from .memory_context import (
    ALL_BUCKET_TYPES,
    BUCKET_SNAPSHOT_VERSION,
    audit_bucket_delta_visibility,
    normalize_bucket_snapshot,
)
from .prompts import (build_realizer_prompt,
                      build_planner_candidate_repair_prompt,
                      build_planner_outline_semantic_prompt,
                      build_semantic_continuation_gate_prompt)
from .prompts import build_planner_prompt
from .workflow import (
    WorkflowError,
    _build_realizer_style_context,
    _candidate_repair_targets,
    execute_round,
    build_realizer_plan_batches,
)
from .profiles import LEGACY_PROFILE, EXPANDED_PROFILE, EXPANDED_SINGLE_PLAN_PROFILE
from .live import _resolve_profile_route

PARENT_PATH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/"
    "roots/cc-train-episode-13836/unified_root.json")


def _parent():
    turns=[]
    for i in range(1,7):
        turns.append({"turn_id":f"root:P01:T{i:03d}","session_id":"root:parent_session_01",
                      "speaker":"A" if i%2 else "B","text":f"parent {i}"})
    return {"root_id":"root","dialogue":{"dialogue_hash":"a"*64,"turns":turns}}


def _next_fields(index: int) -> dict[str, object]:
    return {"next_step_candidates":[
        {"development_mode":"continuation",
         "before_state":f"shared current state {index}",
         "required_transition":f"A completes rejected transition {index}",
         "after_state":f"rejected end state {index}",
         "actor_bindings":[{"actor":"A",
                            "binding":f"rejected-binding-{index}"}],
         "interaction_beats":[f"rejected-only-{index}","next beat 2","next beat 3"]},
        {"development_mode":"continuation",
         "before_state":f"shared current state {index}",
         "required_transition":f"B completes selected transition {index}",
         "after_state":f"selected end state {index}",
         "actor_bindings":[{"actor":"B",
                            "binding":f"selected-binding-{index}"}],
         "interaction_beats":[f"selected-only-{index}","later beat 2","later beat 3"]},
    ]}


def _transition(index: int, *, actor: str = "A") -> dict[str, object]:
    return {
        "before_state": f"supported current state {index}",
        "required_transition": f"{actor} completes concrete transition {index}",
        "after_state": f"concrete end state {index} is now true",
        "actor_bindings": [{"actor": actor,
                            "binding": f"{actor} owns transition {index}."}],
        "interaction_beats": [f"transition-{index}-beat-1",
                              f"transition-{index}-beat-2",
                              f"transition-{index}-beat-3"],
    }


def _continuation(index: int, *, ref: str | None = None,
                  evidence_ref: str = "root:P01:T001") -> dict[str, object]:
    value = {
        "verified_track_ref": ref or f"VT{index:02d}",
        "new_people": [],
        "goal": f"continue event {index}",
        "participants": ["A", "B"],
        "transition_from_previous": f"continue to event {index}",
        **_transition(index),
    }
    value.pop("before_state")
    return value


def _new_development(index: int, *, new_people=None,
                     participants=None) -> dict[str, object]:
    return {
        "source_basis": [f"A has grounded basis {index}.",
                         f"A and B share context {index}."],
        "new_people": list(new_people or []),
        "goal": f"start controlled development {index}",
        "participants": list(participants or ["A", "B"]),
        "transition_from_previous": f"move to new development {index}",
        **_transition(100 + index),
    }


def _split_plan(continuation_count: int, new_count: int,
                *, evidence_ref: str = "root:P01:T001") -> dict[str, object]:
    return {
        "round_goal": "continue useful events and grow the world",
        "selected_continuations": [
            _continuation(index, evidence_ref=evidence_ref)
            for index in range(1, continuation_count + 1)],
        "new_development_candidates": [
            _new_development(index) for index in range(1, new_count + 1)],
    }


def _verified_tracks(count: int, *, turn_ref: str = "root:P01:T001"):
    return {"verified_tracks": [{
        "memory_event_ref": f"ME{index:02d}",
        "verified_current_state": f"verified current state {index}",
        "evidence_positions": [1],
    } for index in range(1, count + 1)]}


def _gate_reviews(*, event_count: int = 3, reject_event: int | None = None,
                  parent_ref: str = "root:P01:T001",
                  candidates_per_event: int = 2) -> dict[str, object]:
    reviews=[]
    for event_index in range(1,event_count+1):
        for candidate_index in range(1, candidates_per_event + 1):
            reject=event_index==reject_event
            reviews.append({
                "candidate_ref":f"ME{event_index:02d}:C{candidate_index:02d}",
                "parent_relation": "REALIZED" if reject else "NONE",
                "decision":"reject" if reject else "pass",
                "duplicate_with_parent":reject,
                "evidence_turn_refs":[parent_ref] if reject else [],
                "reason":"duplicate" if reject else "new",
            })
    return {"candidate_reviews":reviews}


def _reviews_for_refs(refs, *, reject_ref=None,
                      parent_ref="root:P01:T001") -> dict[str, object]:
    return {"candidate_reviews": [{
        "candidate_ref": ref,
        "parent_relation": "REALIZED" if ref == reject_ref else "NONE",
        "decision": "reject" if ref == reject_ref else "pass",
        "duplicate_with_parent": ref == reject_ref,
        "evidence_turn_refs": [parent_ref] if ref == reject_ref else [],
        **({"reason": "The candidate repeats the Parent."}
           if ref == reject_ref else {}),
    } for ref in refs]}


def _adherence_reviews(scene_count=3) -> dict[str, object]:
    """Fixture for the post-Realizer whole-round audit gate."""
    return {"scene_audits": [{
        "participant_adherence": "pass",
        "participant_evidence_positions": [1],
        "development_status": "realized",
        "development_evidence_positions": [1],
        "durable_outcome_status": "realized",
        "durable_outcome_evidence_positions": [1],
        "reason": "Fixture realizes the selected transition and outcome.",
    } for _ in range(scene_count)]}


class MemoryEventContinuationContractTest(unittest.TestCase):
    def test_bucket_snapshot_preserves_provider_source_turn_refs(self):
        response = {
            "success": True,
            "data": {"buckets": {
                **{bucket_type: {} for bucket_type in ALL_BUCKET_TYPES},
                "event": {"event-a": {
                    "display_key": "Event A",
                    "content": ["event content"],
                    "metadata": {"source_turn_refs": [
                        "R02:S01:T002", "R02:S01:T002"]},
                }},
            }},
        }
        snapshot = normalize_bucket_snapshot(
            response, root_id="root", user_id="user")
        event = snapshot["buckets"]["event"]["event-a"]
        self.assertEqual(event["source_turn_refs"], ["R02:S01:T002"])
        self.assertTrue(snapshot["source_turn_refs_present"])

    def test_bucket_snapshot_does_not_infer_missing_provenance(self):
        response = {
            "success": True,
            "data": {"buckets": {
                **{bucket_type: {} for bucket_type in ALL_BUCKET_TYPES},
                "event": {"event-a": {
                    "display_key": "Event A",
                    "content": ["R02:S01:T002 appears as text"],
                }},
            }},
        }
        snapshot = normalize_bucket_snapshot(
            response, root_id="root", user_id="user")
        event = snapshot["buckets"]["event"]["event-a"]
        self.assertEqual(event["source_turn_refs"], [])
        self.assertFalse(snapshot["source_turn_refs_present"])

    @staticmethod
    def _bucket_snapshot(*, event_content=None, person_content=None,
                         root_id="root", user_id="user"):
        buckets = {bucket_type: {} for bucket_type in ALL_BUCKET_TYPES}
        if event_content is not None:
            buckets["event"]["event-a"] = {
                "canonical_key": "event-a",
                "display_key": "Event A",
                "content": list(event_content),
            }
        if person_content is not None:
            buckets["person"]["person-a"] = {
                "canonical_key": "person-a",
                "display_key": "Person A",
                "content": list(person_content),
            }
        return {
            "bucket_snapshot_version": BUCKET_SNAPSHOT_VERSION,
            "root_id": root_id,
            "user_id": user_id,
            "buckets": buckets,
            "snapshot_sha256": f"sha-{len(str(buckets))}",
        }

    def test_bucket_delta_without_change_is_not_visible(self):
        before = self._bucket_snapshot(event_content=["old"])
        after = self._bucket_snapshot(event_content=["old"])
        audit = audit_bucket_delta_visibility(before, after)
        self.assertTrue(audit["valid"])
        self.assertFalse(audit["bucket_visibility_verified"])
        self.assertEqual(audit["changed_bucket_refs"], [])

    def test_gate_derives_missing_reason_for_grounded_reject(self):
        parent = _parent()
        plan = {"scenes": [
            {"memory_event_id": f"ME{index:02d}", **_next_fields(index)}
            for index in range(1, 4)]}
        reviews = _gate_reviews(event_count=3, candidates_per_event=2)
        reviews["candidate_reviews"][0].update({
            "parent_relation": "REALIZED",
            "decision": "reject",
            "duplicate_with_parent": True,
            "evidence_turn_refs": ["root:P01:T001"],
            "reason": "",
        })

        result = validate_semantic_gate_output(
            reviews, plan=plan, parent=parent)

        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["overall_decision"], "pass")
        self.assertEqual(
            result["normalized_candidate_reviews"][0]["reason"],
            "The candidate duplicates a transition already realized in "
            "the authoritative Parent.")

    def test_gate_missing_reject_reason_does_not_relax_evidence_gate(self):
        parent = _parent()
        plan = {"scenes": [
            {"memory_event_id": f"ME{index:02d}", **_next_fields(index)}
            for index in range(1, 4)]}
        reviews = _gate_reviews(event_count=3, candidates_per_event=1)
        reviews["candidate_reviews"][0].update({
            "parent_relation": "REALIZED",
            "decision": "reject",
            "duplicate_with_parent": True,
            "evidence_turn_refs": [],
            "reason": "",
        })

        result = validate_semantic_gate_output(
            reviews, plan=plan, parent=parent)

        self.assertFalse(result["valid"])
        self.assertIn("gate_review_1_reject_evidence_required",
                      result["errors"])

    def test_bucket_delta_added_bucket_is_visible(self):
        before = self._bucket_snapshot(event_content=["old"])
        after = self._bucket_snapshot(event_content=["old", "child update"])
        audit = audit_bucket_delta_visibility(before, after)
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["bucket_visibility_verified"])
        self.assertEqual(audit["updated_bucket_refs"], ["event:event-a"])

    def test_bucket_delta_new_bucket_is_visible(self):
        before = self._bucket_snapshot(event_content=["old"])
        after = self._bucket_snapshot(event_content=["old"])
        after["buckets"]["person"]["person-b"] = {
            "canonical_key": "person-b", "display_key": "Person B",
            "content": ["introduced in Child"],
        }
        audit = audit_bucket_delta_visibility(before, after)
        self.assertTrue(audit["bucket_visibility_verified"])
        self.assertEqual(audit["added_bucket_refs"], ["person:person-b"])

    def test_bucket_delta_removed_only_is_not_positive_visibility(self):
        before = self._bucket_snapshot(event_content=["old"])
        after = self._bucket_snapshot(event_content=None)
        audit = audit_bucket_delta_visibility(before, after)
        self.assertTrue(audit["valid"])
        self.assertFalse(audit["bucket_visibility_verified"])
        self.assertEqual(audit["removed_bucket_refs"], ["event:event-a"])

    def test_bucket_delta_rejects_namespace_mismatch(self):
        before = self._bucket_snapshot(event_content=["old"], user_id="u1")
        after = self._bucket_snapshot(event_content=["new"], user_id="u2")
        audit = audit_bucket_delta_visibility(before, after)
        self.assertFalse(audit["valid"])
        self.assertIn("user_id_mismatch", audit["errors"])
        self.assertFalse(audit["bucket_visibility_verified"])

    def test_speaker_registry_preserves_parent_canonical_names(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"},
            {"speaker": "Co-workers A"},
        ]}}
        registry = build_speaker_registry(parent)
        self.assertEqual(registry["speakers"], [
            {"speaker_ref": "SP01", "canonical_name": "Co-workers A"},
            {"speaker_ref": "SP02", "canonical_name": "Co-workers B"},
        ])

    def test_speaker_refs_materialize_to_canonical_parent_names(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"},
        ]}}
        raw = _split_plan(1, 2)
        for item in [*raw["selected_continuations"],
                     *raw["new_development_candidates"]]:
            item.pop("participants")
            bindings = item.pop("actor_bindings")
            item["speaker_refs"] = ["SP01", "SP02"]
            item["actor_bindings"] = [{
                "actor_ref": "SP01", "binding": bindings[0]["binding"]}]
        result = materialize_planner_speaker_refs(raw, parent=parent)
        self.assertTrue(result["valid"], result["errors"])
        scene = result["value"]["selected_continuations"][0]
        self.assertEqual(scene["participants"],
                         ["Co-workers A", "Co-workers B"])
        self.assertEqual(scene["actor_bindings"][0]["actor"],
                         "Co-workers A")
        self.assertNotIn("speaker_refs", scene)

    def test_transport_role_cannot_be_used_as_speaker_ref(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"},
        ]}}
        raw = _split_plan(1, 0)
        item = raw["selected_continuations"][0]
        item.pop("participants")
        bindings = item.pop("actor_bindings")
        item["speaker_refs"] = ["user"]
        item["actor_bindings"] = [{
            "actor_ref": "user", "binding": bindings[0]["binding"]}]
        item["required_transition"] = (
            "The user completes the selected transition.")
        result = materialize_planner_speaker_refs(raw, parent=parent)
        self.assertFalse(result["valid"])
        self.assertIn(
            "selected_continuations_1_transport_role_ref_forbidden",
            result["errors"])

    def test_transport_role_in_action_text_is_never_mapped_to_actor(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"},
        ]}}
        raw = _split_plan(1, 0)
        item = raw["selected_continuations"][0]
        item.pop("participants")
        item.pop("actor_bindings")
        item["speaker_refs"] = ["SP01", "SP02"]
        item["actor_bindings"] = [{
            "actor_ref": "SP02", "binding": "This speaker owns the action."}]
        item["required_transition"] = (
            "The user completes the selected transition.")
        result = materialize_planner_speaker_refs(raw, parent=parent)
        self.assertFalse(result["valid"])
        self.assertIn(
            "selected_continuations_1_transport_role_alias_forbidden",
            result["errors"])

    def test_transport_role_in_interaction_beat_is_rejected(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"},
        ]}}
        raw = _split_plan(1, 0)
        item = raw["selected_continuations"][0]
        item.pop("participants")
        item["speaker_refs"] = ["SP01", "SP02"]
        item["actor_bindings"] = [{
            "actor_ref": "SP02", "binding": "This speaker owns the action."}]
        item["required_transition"] = (
            "Co-workers B offers the user a concrete workplace favor.")
        item["after_state"] = "Co-workers B has offered the user the favor."
        item["interaction_beats"] = [
            "The user thanks Co-workers B.",
            "Co-workers B offers the favor.",
            "The user accepts the offer.",
        ]
        result = materialize_planner_speaker_refs(raw, parent=parent)
        self.assertFalse(result["valid"])
        self.assertIn(
            "selected_continuations_1_transport_role_alias_forbidden",
            result["errors"])

    def test_canonical_subject_text_passes_single_speaker_contract(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"},
        ]}}
        raw = _split_plan(1, 0)
        item = raw["selected_continuations"][0]
        item.pop("participants")
        bindings = item.pop("actor_bindings")
        item["speaker_refs"] = ["SP02"]
        item["actor_bindings"] = [{
            "actor_ref": "SP02", "binding": bindings[0]["binding"]}]
        item["required_transition"] = (
            "Co-workers B completes the selected transition.")
        result = materialize_planner_speaker_refs(raw, parent=parent)
        self.assertTrue(result["valid"], result["errors"])

    def test_alias_or_mentioned_person_cannot_bypass_speaker_registry(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"},
        ]}}
        raw = _split_plan(1, 2)
        raw["selected_continuations"][0]["participants"] = ["同事A", "上司"]
        result = materialize_planner_speaker_refs(raw, parent=parent)
        self.assertFalse(result["valid"])
        self.assertIn("selected_continuations_1_legacy_speaker_fields",
                      result["errors"])

    def test_unknown_speaker_ref_is_rejected_without_alias_guessing(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"},
        ]}}
        raw = _split_plan(1, 2)
        item = raw["selected_continuations"][0]
        item.pop("participants")
        item.pop("actor_bindings")
        item["speaker_refs"] = ["SP01", "BOSS"]
        item["actor_bindings"] = [{
            "actor_ref": "SP01", "binding": "SP01 owns the transition."}]
        result = materialize_planner_speaker_refs(raw, parent=parent)
        self.assertFalse(result["valid"])
        self.assertIn("selected_continuations_1_unknown_speaker_ref:BOSS",
                      result["errors"])

    def test_declared_new_person_is_added_without_fabricated_speaker_ref(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"},
        ]}}
        raw = _split_plan(0, 1)
        item = raw["new_development_candidates"][0]
        item["new_people"] = [{
            "name": "Elena", "relationship_basis": "new neighbor of A"}]
        item.pop("participants")
        bindings = item.pop("actor_bindings")
        item["speaker_refs"] = ["SP01", "Elena"]
        item["actor_bindings"] = [{
            "actor_ref": "SP01", "binding": bindings[0]["binding"]}]
        result = materialize_planner_speaker_refs(raw, parent=parent)
        self.assertTrue(result["valid"], result["errors"])
        materialized = result["value"]["new_development_candidates"][0]
        self.assertEqual(materialized["participants"],
                         ["Co-workers A", "Elena"])
        self.assertNotIn("speaker_refs", materialized)

    def test_single_canonical_speaker_is_allowed(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"},
        ]}}
        raw = _split_plan(1, 2)
        for item in [*raw["selected_continuations"],
                     *raw["new_development_candidates"]]:
            item.pop("participants")
            bindings = item.pop("actor_bindings")
            item["speaker_refs"] = ["SP01"]
            item["actor_bindings"] = [{
                "actor_ref": "SP01", "binding": bindings[0]["binding"]}]
        materialized = materialize_planner_speaker_refs(raw, parent=parent)
        self.assertTrue(materialized["valid"], materialized["errors"])
        self.assertEqual(
            materialized["value"]["selected_continuations"][0]["participants"],
            ["Co-workers A"])

    def test_strict_planner_prompt_exposes_refs_not_free_speaker_names(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "Co-workers A", "text": "one"},
            {"speaker": "Co-workers B", "text": "two"},
        ]}}
        registry = build_speaker_registry(parent)
        prompt = build_planner_prompt(
            parent=parent, memory_overview={}, candidates=[],
            verified_tracks=[], memory_only_fast=True,
            speaker_registry=registry, strict_speaker_refs=True)
        self.assertEqual(prompt["prompt_version"],
                         "memory_context_fusion_planner_prompt_v2_bounded_scene_outcome_person_bucket_only")
        self.assertIn('"speaker_refs": ["SP01", "SP02"]', prompt["prompt"])
        self.assertIn('"actor_ref": "SP01"', prompt["prompt"])
        self.assertIn("must never be encoded as a", prompt["prompt"])
        self.assertIn("fabricated SP03", prompt["prompt"])
        self.assertIn("transport roles with no character identity",
                      prompt["prompt"])
        self.assertNotIn('"participants": ["Exact Parent speaker"',
                         prompt["prompt"])

    def test_memory_only_tracks_use_bucket_contents_without_fabricating_evidence(self):
        candidates = [{
            "memory_event_id": "ME01", "memory_event_key": "soccer",
            "memory_event_display": "Soccer",
            "memory_event_content": ["A plans to resume practice.",
                                     "A has not yet played again."],
        }]
        tracks = materialize_memory_only_tracks(candidates)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["verified_track_ref"], "VT01")
        self.assertEqual(tracks[0]["evidence_turn_refs"], [])
        self.assertEqual(tracks[0]["state_source"],
                         "memorax_memory_only_unverified")
        self.assertIn("A plans to resume practice.",
                      tracks[0]["verified_current_state"])
        self.assertEqual(
            tracks[0]["memory_planning_packet"]["state_authority"],
            "memorax_unverified")

    def test_memory_only_planner_prompt_excludes_full_parent_transcript(self):
        parent = {"dialogue": {"turns": [
            {"speaker": "A", "text": "PRIVATE_PARENT_SENTENCE"},
            {"speaker": "B", "text": "another sentence"},
        ]}}
        candidates = [{"memory_event_id": "ME01",
                       "memory_event_key": "event-a",
                       "memory_event_display": "Event A",
                       "memory_event_content": ["A plans a next step"]}]
        tracks = materialize_memory_only_tracks(candidates)
        prompt = build_planner_prompt(
            parent=parent, memory_overview={}, candidates=candidates,
            verified_tracks=tracks, memory_only_fast=True)
        self.assertEqual(prompt["prompt_version"],
                         "memory_context_fusion_planner_prompt_v2_bounded_scene_outcome")
        self.assertNotIn("PRIVATE_PARENT_SENTENCE", prompt["prompt"])
        self.assertNotIn("Memory-only Fast mode", prompt["prompt"])
        self.assertIn("CANDIDATE MEMORY PLANNING PACKETS", prompt["prompt"])

    def test_live_request_budget_allows_one_verifier_planner_gate_and_realizer(self):
        self.assertEqual(PER_ROLE_REQUEST_BUDGET, 1)
        self.assertEqual(TOTAL_REQUEST_BUDGET, 12)
        self.assertIn("transcript_event_verifier_v1", ALLOWED_LIVE_ROLES)
        self.assertIn("planner_outline_semantic_auditor_v1",
                      ALLOWED_LIVE_ROLES)
        self.assertIn("simple_append_scene_continuation_realizer_v1",
                      ALLOWED_LIVE_ROLES)
        self.assertIn("simple_append_scene_replacement_realizer_v1",
                      ALLOWED_LIVE_ROLES)
        self.assertIn("memory_event_realizer_adherence_reaudit_v1",
                      ALLOWED_LIVE_ROLES)

    def test_candidate_repair_prompt_requests_only_missing_passes(self):
        parent = _parent()
        candidates = [{
            "memory_event_id": "ME01", "memory_event_key": "event-a",
            "memory_event_display": "Event A", "memory_event_content": ["fact"],
        }]
        tracks = materialize_memory_only_tracks(candidates)
        plan = {"candidate_pool": [{
            "memory_event_id": "ND01",
            "next_step_candidates": [{
                "required_transition": "old", "after_state": "old",
            }],
        }]}
        gate = {
            "passed_candidate_count": 2,
            "minimum_pass_candidates": 3,
            "rejected_candidate_refs": ["ND01:C01"],
            "normalized_candidate_reviews": [{
                "candidate_ref": "ND01:C01", "reason": "already realized",
            }],
        }
        result = build_planner_candidate_repair_prompt(
            candidates=candidates, verified_tracks=tracks, plan=plan,
            gate_validation=gate)
        self.assertEqual(result["needed_candidate_count"], 1)
        self.assertIn("exactly 1 replacement", result["prompt"])
        self.assertIn("ND01:C01", result["prompt"])

    def test_targeted_repair_skips_non_retryable_event_and_fills_new_slot(self):
        plan = {"candidate_pool": [
            {"memory_event_id": "ME01", "development_mode": "continuation",
             "next_step_candidates": [{"required_transition": "old"}]},
            {"memory_event_id": "ND01", "development_mode": "new_development",
             "next_step_candidates": [{"required_transition": "old-new"}]},
        ]}
        gate = {
            "passed_candidate_refs": ["ME02:C01"],
            "rejected_candidate_refs": ["ME01:C01", "ND01:C01"],
            "normalized_candidate_reviews": [
                {"candidate_ref": "ME01:C01", "reason": "event identity is ambiguous"},
                {"candidate_ref": "ND01:C01", "reason": "duplicate realized transition"},
            ],
        }
        targets = _candidate_repair_targets(
            plan=plan, gate_validation=gate, required_count=2)
        self.assertEqual([row["target_kind"] for row in targets], [
            "replace_new_development", "new_development_slot"])
        self.assertEqual(targets[0]["target_ref"], "ND01:C01")

    def test_targeted_repair_is_bounded_to_one_attempt(self):
        prompt = build_planner_candidate_repair_prompt(
            candidates=[], verified_tracks=[], plan={"candidate_pool": []},
            gate_validation={"passed_candidate_count": 1,
                              "minimum_pass_candidates": 3,
                              "rejected_candidate_refs": ["ME01:C01"]},
            repair_targets=[{"target_ref": "ME01:C01",
                             "target_kind": "retry_same_memory_event",
                             "reason": "duplicate realized transition"}],
        )
        self.assertIn("only repair attempt", prompt["prompt"])
        self.assertEqual(prompt["repair_targets"][0]["target_kind"],
                         "retry_same_memory_event")

    def test_same_event_target_gets_same_call_fallback(self):
        plan = {"candidate_pool": [{
            "memory_event_id": "ME01", "development_mode": "continuation",
            "next_step_candidates": [{"required_transition": "old"}],
        }]}
        gate = {
            "rejected_candidate_refs": ["ME01:C01"],
            "normalized_candidate_reviews": [{
                "candidate_ref": "ME01:C01",
                "parent_relation": "REALIZED",
                "reason": "duplicate realized transition",
            }],
        }
        targets = _candidate_repair_targets(
            plan=plan, gate_validation=gate, required_count=1)
        self.assertEqual([row["target_kind"] for row in targets], [
            "retry_same_memory_event", "new_development_slot"])
        prompt = build_planner_candidate_repair_prompt(
            candidates=[], verified_tracks=[], plan=plan,
            gate_validation={"passed_candidate_count": 2,
                              "minimum_pass_candidates": 3,
                              "rejected_candidate_refs": ["ME01:C01"]},
            repair_targets=targets,
        )
        self.assertEqual(prompt["needed_candidate_count"], 2)

    def test_candidate_repair_contract_requires_exact_replacement_count(self):
        parent = _parent()
        candidate = {
            "source_basis": ["A concrete Parent fact"],
            "new_people": [], "goal": "make a concrete change",
            "participants": ["A", "B"],
            "transition_from_previous": "A and B act now",
            "before_state": "the current state",
            "required_transition": "A and B complete a new action",
            "after_state": "the action is complete",
            "actor_bindings": [{"actor": "A", "binding": "A owns the action"}],
            "interaction_beats": ["A proposes", "B agrees", "they act"],
        }
        valid = validate_planner_repair_output(
            {"new_development_candidates": [candidate]},
            parent=parent, required_count=1)
        self.assertTrue(valid["valid"], valid["errors"])
        invalid = validate_planner_repair_output(
            {"new_development_candidates": []},
            parent=parent, required_count=1)
        self.assertFalse(invalid["valid"])
        self.assertIn("planner_repair_candidate_count", invalid["errors"])

    def test_realization_aware_gate_role_is_explicitly_allowed(self):
        role = "memory_event_realization_aware_semantic_gate_v1"
        self.assertIn(role, ALLOWED_LIVE_ROLES)
        coverage = validate_outbound_role_coverage([role])
        self.assertTrue(coverage["valid"])

    def test_unknown_role_remains_rejected_by_explicit_allowlist(self):
        coverage = validate_outbound_role_coverage(
            ["memory_event_unknown_gate_v999"])
        self.assertFalse(coverage["valid"])
        self.assertEqual(coverage["missing_outbound_roles"],
                         ["memory_event_unknown_gate_v999"])

    def test_event_verifier_positions_are_materialized_deterministically(self):
        parent = _parent()
        candidates = [{"memory_event_id": "ME01", "memory_event_key": "event-a",
                       "memory_event_display": "A",
                       "memory_event_content": ["fact"]}]
        raw = {"verified_tracks": [{
            "memory_event_ref": "ME01",
            "verified_current_state": "A has completed the known step.",
            "evidence_positions": [1, 2],
        }]}
        self.assertTrue(validate_event_verifier_output(
            raw, candidates=candidates, parent=parent)["valid"])
        tracks = materialize_verified_tracks(
            raw, candidates=candidates, parent=parent)
        self.assertEqual(tracks[0]["verified_track_ref"], "VT01")
        self.assertEqual(tracks[0]["evidence_turn_refs"],
                         ["root:P01:T001", "root:P01:T002"])

    def test_event_verifier_rejects_more_than_four_positions(self):
        parent = _parent()
        candidates = [{"memory_event_id": "ME01"}]
        raw = {"verified_tracks": [{
            "memory_event_ref": "ME01", "verified_current_state": "state",
            "evidence_positions": [1, 2, 3, 4, 5]}]}
        result = validate_event_verifier_output(
            raw, candidates=candidates, parent=parent)
        self.assertFalse(result["valid"])
        self.assertIn("verified_track_1_evidence_positions", result["errors"])

    def test_event_verifier_rejects_placeholder_current_state(self):
        raw = {"verified_tracks": [{
            "memory_event_ref": "ME01",
            "verified_current_state": "latest state already true in Parent",
            "evidence_positions": [1],
        }]}
        result = validate_event_verifier_output(
            raw, candidates=[{"memory_event_id": "ME01"}], parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("verified_track_1_placeholder_current_state",
                      result["errors"])

    def test_event_verifier_prompt_has_no_copyable_state_example(self):
        from .prompts import build_event_verifier_prompt
        prompt = build_event_verifier_prompt(
            parent=_parent(), candidates=[{"memory_event_id": "ME01",
            "memory_event_key": "event-a", "memory_event_display": "A",
            "memory_event_content": ["fact"]}])
        self.assertEqual(prompt["prompt_version"],
                         "transcript_event_verifier_prompt_v3")
        self.assertNotIn("latest state already true in Parent",
                         prompt["prompt"])
        self.assertIn("Four positions is a hard maximum", prompt["prompt"])
        for field in ("memory_event_ref", "verified_current_state",
                      "evidence_positions"):
            self.assertIn(field, prompt["prompt"])
        for field in ("event_ref", "intention", "action", "result"):
            self.assertIn(f"`{field}`", prompt["prompt"])

    def test_event_verifier_rejects_legacy_state_split_fields(self):
        raw = {"verified_tracks": [{
            "event_ref": "ME01", "intention": "do something",
            "action": "did something", "result": "something happened",
        }]}
        result = validate_event_verifier_output(
            raw, candidates=[{"memory_event_id": "ME01"}], parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("verified_track_1_legacy_state_split_fields",
                      result["errors"])

    def test_planner_continuation_contract_has_no_evidence_field(self):
        plan = _split_plan(1, 3)
        self.assertNotIn("evidence_turn_refs", plan["selected_continuations"][0])
        tracks = [{"verified_track_ref": "VT01", "memory_event_ref": "ME01",
                   "verified_current_state": "state",
                   "evidence_turn_refs": ["root:P01:T001"]}]
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent(),
            verified_tracks=tracks)
        self.assertTrue(result["valid"], result["errors"])

    def test_planner_rejects_legacy_evidence_and_before_state_fields(self):
        plan = _split_plan(1, 3)
        plan["selected_continuations"][0]["evidence_turn_refs"] = [
            "root:P01:T001"]
        plan["selected_continuations"][0]["before_state"] = "legacy state"
        tracks = [{"verified_track_ref": "VT01", "memory_event_ref": "ME01",
                   "verified_current_state": "state",
                   "evidence_turn_refs": ["root:P01:T001"]}]
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent(),
            verified_tracks=tracks)
        self.assertFalse(result["valid"])
        self.assertIn("continuation_1_shape", result["errors"])

    def test_outbound_role_preflight_requires_verifier_planner_gate_and_realizer(self):
        required = [
            "transcript_event_verifier_v1",
            "simple_append_planner_v1",
            "memory_event_realization_aware_semantic_gate_v1",
            "simple_append_round_realizer_v1",
        ]
        self.assertTrue(validate_outbound_role_coverage(required)["valid"])
        for absent in required:
            allowed = set(ALLOWED_LIVE_ROLES)
            allowed.remove(absent)
            coverage = validate_outbound_role_coverage(
                required, allowed_roles=allowed)
            self.assertFalse(coverage["valid"])
            self.assertEqual(coverage["missing_outbound_roles"], [absent])

    def test_namespace_isolated_and_payload_secret_free(self):
        self.assertEqual(
            memorax_user_id("root"),
            "gxy-selfplay:memory-event-continuation-semantic-gate-probe-v1:root")
        intent=build_add_intent(root_id="root",round_index=0,source_dialogue_sha256="a"*64,turns=_parent()["dialogue"]["turns"],async_mode=True)
        serialized=json.dumps(intent)
        self.assertFalse(intent["api_key_in_artifact"])
        self.assertNotIn("Authorization",serialized)
        self.assertNotIn("MEMORAX_API_KEY",serialized)

    def test_memorax_transport_roles_are_all_user(self):
        parent = _parent()
        intent = build_add_intent(
            root_id="root", round_index=0,
            source_dialogue_sha256="a" * 64,
            turns=parent["dialogue"]["turns"],
            recent_context_turns=parent["dialogue"]["turns"],
            async_mode=True)
        payload = intent["payload"]
        self.assertEqual(intent["intent_version"],
                         "memory_event_continuation_add_intent_v2_all_user")
        self.assertTrue(all(row["role"] == "user"
                            for row in payload["messages"]))
        self.assertTrue(all(row["role"] == "user"
                            for row in payload["window_context"]))
        self.assertNotIn("assistant", {
            row["role"] for row in [*payload["messages"],
                                     *payload["window_context"]]})
        self.assertEqual(
            payload["metadata"]["memorax_role_policy"],
            "all_user_speaker_identity_in_content_v1")
        self.assertTrue(payload["messages"][0]["content"].startswith(
            "[Speaker: A]"))

    def test_candidates_are_stable(self):
        overview={"events":[{"canonical_key":"event-a","display_key":"A","content":["fact-a"]},{"canonical_key":"event-b","display_key":"B","content":["fact-b"]},{"canonical_key":"event-c","display_key":"C","content":["fact-c"]}]}
        self.assertEqual([x["memory_event_id"] for x in build_event_candidates(overview)], ["ME01","ME02","ME03"])

    def test_one_event_key_with_three_contents_is_one_candidate(self):
        overview={"events":[{"canonical_key":"party","display_key":"Party",
                              "content":["invitation","movie night","later follow-up"]}]}
        candidates=build_event_candidates(overview)
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0]["memory_event_content"],
                         ["invitation","movie night","later follow-up"])

    def test_distinct_event_keys_are_distinct_candidates(self):
        overview={"events":[
            {"canonical_key":"party","display_key":"Party","content":["party fact"]},
            {"canonical_key":"budget","display_key":"Budget","content":["budget fact"]}]}
        candidates=build_event_candidates(overview)
        self.assertEqual([row["memory_event_key"] for row in candidates],
                         ["party","budget"])

    def test_person_recent_events_create_bound_person_candidates_when_unmatched(self):
        overview={
            "events":[
                {"canonical_key":"event-a","display_key":"A","content":["fact-a"]},
                {"canonical_key":"event-b","display_key":"B","content":["fact-b"]},
            ],
            "people":[{
                "canonical_key":"person-a","display_key":"Person A",
                "content":["[关系] lives nearby",
                           "[事件 @近期] Person A scheduled the next meeting."]}],
        }
        candidates=build_event_candidates(overview)
        self.assertEqual(len(candidates),3)
        self.assertEqual(
            [row["source_type"] for row in candidates],
            ["event_bucket", "event_bucket", "person_bucket_event"])
        self.assertEqual(candidates[-1]["person_event_facts"][0]["person_display"],
                         "Person A")

    def test_duplicate_event_bucket_and_person_event_is_not_doubled(self):
        text="[事件 @近期] Person A scheduled the next meeting."
        overview={"events":[{"canonical_key":"event-a","display_key":"A",
                             "content":[text]}],
                  "people":[{"canonical_key":"person-a","display_key":"Person A",
                              "content":[text]}]}
        candidates=build_event_candidates(overview)
        self.assertEqual(len(candidates),1)
        self.assertEqual(
            {row["source_type"] for row in candidates[0]["supporting_sources"]},
            {"event_bucket","person_bucket","person_recent_event"})

    def test_person_recent_event_matching_event_display_is_support_only(self):
        overview={"events":[{"canonical_key":"聚会","display_key":"party",
                             "content":["A party was proposed."]}],
                  "people":[{"canonical_key":"a","display_key":"A",
                             "content":["[事件 @recent] A discussed party arrangements."]}]}
        candidates=build_event_candidates(overview)
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0]["memory_event_key"],"聚会")
        self.assertEqual(
            {row["source_type"] for row in candidates[0]["supporting_sources"]},
            {"event_bucket","person_bucket","person_recent_event"})
        self.assertEqual(candidates[0]["memory_event_content"],
                         ["A party was proposed."])

    def test_person_event_facts_bind_event_summary_to_named_people(self):
        overview = {
            "events": [{
                "canonical_key": "unsafe vehicle ticket",
                "display_key": "unsafe vehicle ticket",
                "content": [
                    "The user received a ticket after delaying a broken window repair.",
                ],
            }],
            "people": [
                {"canonical_key": "a", "display_key": "Classmates A",
                 "content": [
                     "[事件 @car ticket regret] Classmates A received a ticket for the broken window.",
                 ]},
                {"canonical_key": "b", "display_key": "Classmates B",
                 "content": [
                     "[事件 @car advice] Classmates B advised Classmates A to fix the broken window before another ticket.",
                 ]},
            ],
        }
        candidates = build_event_candidates(overview)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["source_type"], "event_bucket")
        self.assertEqual(
            {row["person_display"] for row in candidates[0]["person_event_facts"]},
            {"Classmates A", "Classmates B"})
        self.assertEqual(
            {row["source_type"] for row in candidates[0]["supporting_sources"]},
            {"event_bucket", "person_recent_event", "person_bucket"})

    def test_person_events_with_only_one_shared_token_stay_separate(self):
        overview = {"events": [], "people": [
            {"canonical_key": "a", "display_key": "Classmates A",
             "content": [
                 "[事件 @watching TV] Classmates A watched Friends with Classmates B.",
             ]},
            {"canonical_key": "b", "display_key": "Classmates B",
             "content": [
                 "[事件 @TV blocking] Classmates B blocked Classmates A's TV view and apologized.",
             ]},
        ]}
        candidates = build_event_candidates(overview)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(
            row["source_type"] == "person_bucket_event" for row in candidates))

    def test_person_events_merge_with_two_facts_and_shared_event_tag_token(self):
        overview = {"events": [], "people": [
            {"canonical_key": "a", "display_key": "Classmates A",
             "content": [
                 "[事件 @car ticket regret] Classmates A received a ticket for the broken window.",
             ]},
            {"canonical_key": "b", "display_key": "Classmates B",
             "content": [
                 "[事件 @car advice] Classmates B advised A to fix the broken window before another ticket.",
             ]},
        ]}
        candidates = build_person_event_candidates(overview)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0]["person_event_facts"]), 2)

    def test_person_bucket_builder_ignores_event_bucket_identity(self):
        overview = {
            "events": [{"canonical_key": "user-only-event",
                        "display_key": "user-only-event",
                        "content": ["The user did a thing."]}],
            "people": [{"canonical_key": "a", "display_key": "Classmates A",
                        "content": [
                            "[事件 @car repair] Classmates A repaired the broken window."]}],
        }
        candidates = build_person_event_candidates(overview)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["source_type"], "person_bucket_event")
        self.assertTrue(candidates[0]["memory_event_key"].startswith(
            "person_event:"))
        self.assertNotIn("user-only-event", candidates[0]["memory_event_key"])

    def test_person_bucket_builder_skips_broad_recent_rollup(self):
        overview = {"events": [{
            "canonical_key": "user-only-event", "display_key": "user-only-event",
            "content": ["The user did a thing."],
        }], "people": [{
            "canonical_key": "a", "display_key": "Classmates A",
            "content": ["[事件 @recent] Classmates A has several recent updates."],
        }]}
        self.assertEqual(build_person_event_candidates(overview), [])

    def test_event_fallback_keeps_event_key_and_skips_broad_person_rollup(self):
        overview = {"events": [{
            "canonical_key": "new-trail", "display_key": "New trail",
            "content": ["Neighbors A tried a new walking trail."],
        }], "people": [{
            "canonical_key": "neighbors-a", "display_key": "Neighbors A",
            "content": [
                "[事件 @recent] Neighbors A discussed several unrelated updates."
            ],
        }]}
        candidates = build_event_candidates(
            overview, person_events_only=True)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["memory_event_key"], "new-trail")
        self.assertEqual(candidates[0]["source_type"], "event_bucket")

    def test_person_recent_event_without_event_match_is_retained(self):
        overview={"events":[{"canonical_key":"party","display_key":"Party",
                             "content":["The party ended."]}],
                  "people":[{"canonical_key":"a","display_key":"A",
                             "content":["[事件 @recent] A met a new cashier."]}]}
        candidates=build_event_candidates(overview)
        self.assertEqual(len(candidates),2)
        self.assertEqual(candidates[0]["memory_event_key"], "party")
        self.assertRegex(
            candidates[1]["memory_event_key"],
            r"^person_event:a:fact:[0-9a-f]{12}$")

    def test_different_person_recent_wording_is_retained_as_own_candidate(self):
        overview={"events":[{
            "canonical_key":"周末徒步", "display_key":"周末徒步",
            "content":["Classmates A计划周末徒步。"]}],
            "people":[{
                "canonical_key":"classmates a", "display_key":"Classmates A",
                "content":["[事件 @recent] Classmates A准备新的户外冒险。"]}]}
        candidates=build_event_candidates(overview)
        self.assertEqual(len(candidates),2)
        self.assertEqual(candidates[0]["memory_event_key"], "周末徒步")
        self.assertRegex(
            candidates[1]["memory_event_key"],
            r"^person_event:classmates a:fact:[0-9a-f]{12}$")

    def test_same_person_same_time_tag_different_events_stay_separate(self):
        overview = {"events": [], "people": [{
            "canonical_key": "co-workers a",
            "display_key": "Co-workers A",
            "content": [
                "[事件 @past] Co-workers A missed a kick during a soccer game.",
                "[事件 @past] Co-workers A apologized to a friend.",
                "[事件 @past] Co-workers A took sick leave because of the flu.",
            ],
        }]}
        candidates = build_person_event_candidates(overview)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            [len(row["person_event_facts"]) for row in candidates],
            [1, 1, 1])
        self.assertEqual(len({
            row["memory_event_key"] for row in candidates}), 3)
        self.assertNotIn(
            "past",
            {row["memory_event_display"] for row in candidates})

    def test_different_people_same_time_tag_unrelated_events_stay_separate(self):
        overview = {"events": [], "people": [
            {"canonical_key": "a", "display_key": "Co-workers A",
             "content": [
                 "[事件 @current] Co-workers A scheduled a medical appointment.",
             ]},
            {"canonical_key": "b", "display_key": "Co-workers B",
             "content": [
                 "[事件 @current] Co-workers B corrected a report mistake.",
             ]},
        ]}
        candidates = build_person_event_candidates(overview)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [len(row["person_event_facts"]) for row in candidates],
            [1, 1])

    def test_same_event_across_people_merges_without_using_time_tag(self):
        overview = {"events": [], "people": [
            {"canonical_key": "a", "display_key": "Co-workers A",
             "content": [
                 "[事件 @past] Co-workers A missed a soccer kick and planned more practice after work.",
             ]},
            {"canonical_key": "b", "display_key": "Co-workers B",
             "content": [
                 "[事件 @current] Co-workers B comforted A about the missed soccer kick and offered after-work practice.",
             ]},
        ]}
        candidates = build_person_event_candidates(overview)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0]["person_event_facts"]), 2)
        self.assertEqual(len(candidates[0]["memory_event_content"]), 2)
        self.assertEqual(
            {row["event_tag"] for row in candidates[0]["person_event_facts"]},
            {"past", "current"})

    def test_person_facts_materialize_only_for_related_event(self):
        overview={"events":[
            {"canonical_key":"music festival","display_key":"Music Festival",
             "content":["Classmates B plans a music festival."]},
            {"canonical_key":"weather","display_key":"Weather",
             "content":["The weather will be sunny."]}],
            "people":[{
                "canonical_key":"classmates b","display_key":"Classmates B",
                "content":["[身份] Classmates B is a musician.",
                           "[关系] Classmates B and A are friends.",
                           "[偏好] Classmates B likes live music.",
                           "[性格] Classmates B is patient."]}]}
        candidates=build_event_candidates(overview)
        self.assertEqual(len(candidates[0]["related_people"]),1)
        person=candidates[0]["related_people"][0]
        self.assertEqual(person["name"],"Classmates B")
        self.assertEqual(person["identity_facts"],
                         ["Classmates B is a musician."])
        self.assertEqual(person["relationship_facts"],
                         ["Classmates B and A are friends."])
        self.assertEqual(person["preference_facts"],
                         ["Classmates B likes live music."])
        self.assertEqual(person["trait_facts"],
                         ["Classmates B is patient."])
        self.assertEqual(candidates[1]["related_people"],[])

    def test_supporting_sources_survive_into_planner_packet(self):
        overview={"events":[{
            "canonical_key":"party","display_key":"Party",
            "content":["Classmates A proposed a party."]}],
            "people":[{
                "canonical_key":"classmates a","display_key":"Classmates A",
                "content":["[关系] Classmates A and B are friends.",
                           "[事件 @recent] Classmates A updated the party plan."]}]}
        candidates=build_event_candidates(overview)
        tracks=materialize_memory_only_tracks(candidates)
        packet=tracks[0]["memory_planning_packet"]
        source_types={row["source_type"] for row in packet["supporting_sources"]}
        self.assertEqual(source_types,
                         {"event_bucket","person_bucket","person_recent_event"})
        prompt=build_planner_prompt(
            parent=_parent(),memory_overview=overview,candidates=candidates,
            verified_tracks=tracks,memory_only_fast=True)
        self.assertIn('"related_people"',prompt["prompt"])
        self.assertIn('"supporting_sources"',prompt["prompt"])
        self.assertIn('"state_authority": "memorax_unverified"',prompt["prompt"])

    def test_event_bucket_only_packet_excludes_person_context(self):
        overview={"events":[{
            "canonical_key":"party","display_key":"Party",
            "content":["Classmates A proposed a party."]}],
            "people":[{
                "canonical_key":"classmates a","display_key":"Classmates A",
                "content":["[关系] Classmates A and B are friends.",
                           "[事件 @recent] Classmates A updated the party plan."]}]}
        candidates=build_event_candidates(overview)
        tracks=materialize_memory_only_tracks(candidates,event_bucket_only=True)
        packet=tracks[0]["memory_planning_packet"]
        self.assertEqual(packet["related_people"],[])
        self.assertEqual(
            {row["source_type"] for row in packet["supporting_sources"]},
            {"event_bucket"})
        prompt=build_planner_prompt(
            parent=_parent(),memory_overview=overview,candidates=candidates,
            verified_tracks=tracks,memory_only_fast=True,
            event_bucket_only=True)
        self.assertIn("event-bucket keys and event-bucket contents",prompt["prompt"])
        self.assertNotIn("Classmates A updated the party plan",prompt["prompt"])
        self.assertNotIn("Classmates A and B are friends",prompt["prompt"])

    def test_related_non_parent_person_does_not_gain_speaker_permission(self):
        candidates=[{
            "memory_event_id":"ME01","memory_event_key":"event-a",
            "memory_event_display":"Event A","memory_event_content":["Elena helps A."],
            "source_type":"event_bucket","supporting_sources":[],
            "related_people":[{"name":"Elena","identity_facts":[],
                "relationship_facts":[],"preference_facts":[],"trait_facts":[]}]}]
        tracks=materialize_memory_only_tracks(candidates)
        plan=_split_plan(1,2)
        plan["selected_continuations"][0]["participants"]=["A","Elena"]
        result=validate_planner_output(
            plan,candidates=candidates,parent=_parent(),verified_tracks=tracks)
        self.assertFalse(result["valid"])
        self.assertIn("continuation_1_participant_not_declared",result["errors"])

    def test_memory_omission_does_not_materialize_negative_state(self):
        candidates=[{
            "memory_event_id":"ME01","memory_event_key":"hiking",
            "memory_event_display":"Hiking","memory_event_content":[
                "A discussed joining a hiking activity."],
            "source_type":"event_bucket","supporting_sources":[],
            "related_people":[]}]
        track=materialize_memory_only_tracks(candidates)[0]
        self.assertNotIn("has not happened",track["memory_state_hint"])
        self.assertNotIn("never joined",track["memory_state_hint"])

    def test_candidate_provenance_preserves_event_and_person_sources(self):
        overview={"events":[{"canonical_key":"party","display_key":"Party",
                             "content":["Party plan one.","Party plan two."]}],
                  "people":[{"canonical_key":"a","display_key":"A",
                             "content":["[事件 @recent] A updated the party plan."]}]}
        candidate=build_event_candidates(overview)[0]
        self.assertEqual(candidate["source_type"],"event_bucket")
        self.assertEqual(candidate["supporting_sources"][0],{
            "source_type":"event_bucket","memory_key":"party",
            "display_key":"Party","contents":["Party plan one.","Party plan two."]})
        self.assertEqual(candidate["supporting_sources"][1]["source_type"],
                         "person_recent_event")

    def test_repeated_rows_for_same_event_key_do_not_restore_flattening(self):
        overview={"events":[
            {"canonical_key":"party","display_key":"Party","content":["first"]},
            {"canonical_key":"party","display_key":"Party","content":["second"]}]}
        candidates=build_event_candidates(overview)
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0]["memory_event_content"],["first","second"])

    def _candidates(self, count: int):
        return build_event_candidates({"events":[
            {"canonical_key":f"e{i}","display_key":f"E{i}",
             "content":[f"f{i}"]} for i in range(1, count + 1)]})

    def test_three_existing_events_can_be_three_continuations(self):
        plan = _split_plan(3, 0)
        result = validate_planner_output(
            plan, candidates=self._candidates(3), parent=_parent())
        self.assertTrue(result["valid"])
        compiled = compile_planner_output(plan, candidates=self._candidates(3))
        self.assertEqual([row["development_mode"] for row in compiled["scenes"]],
                         ["continuation"] * 3)

    def test_outline_audit_keeps_fuzzy_overlap_non_blocking(self):
        plan = {"scenes": [
            {"scene_ref": "S01", "memory_event_id": "ME05",
             "participants": ["A", "B"],
             "required_transition": "A and B complete the walk at the new trail with their dogs.",
             "after_state": "A and B successfully completed the walk at the new trail."},
            {"scene_ref": "S02", "memory_event_id": "ND02",
             "participants": ["A", "B"],
             "required_transition": "A and B take their dogs on the walk at the new trail and enjoy it.",
             "after_state": "A and B have completed the walk and feel refreshed."},
        ]}
        result = validate_planner_outline_distinctness(plan)
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(
            result["non_blocking_similarities"][0]["scene_refs"],
            ["S01", "S02"])

    def test_outline_audit_rejects_exact_transition_duplicate(self):
        plan = {"scenes": [
            {"scene_ref": "S01", "memory_event_id": "ME01",
             "participants": ["A", "B"],
             "required_transition": "A and B complete the walk.",
             "after_state": "The walk is complete."},
            {"scene_ref": "S02", "memory_event_id": "ND01",
             "participants": ["A", "B"],
             "required_transition": "A and B complete the walk.",
             "after_state": "The walk is complete again."},
        ]}
        result = validate_planner_outline_distinctness(plan)
        self.assertFalse(result["valid"])
        self.assertEqual(result["errors"], ["planner_outline_duplicate:S01:S02"])

    def test_outline_semantic_prompt_contains_only_compiled_pairs(self):
        plan = {"scenes": [
            {"scene_ref": "S01", "memory_event_id": "ME01",
             "participants": ["A", "B"],
             "verified_current_state": "A plans a walk.",
             "required_transition": "A starts the walk.",
             "after_state": "The walk is underway."},
            {"scene_ref": "S02", "memory_event_id": "ND01",
             "participants": ["A", "C"],
             "required_transition": "A meets C at the station.",
             "after_state": "They exchange contact details."},
            {"scene_ref": "S03", "memory_event_id": "ND02",
             "participants": ["B", "C"],
             "required_transition": "B sends C the schedule.",
             "after_state": "C receives the schedule."},
        ]}
        built = build_planner_outline_semantic_prompt(plan=plan)
        self.assertEqual(built["expected_pair_refs"], ["P01", "P02", "P03"])
        self.assertNotIn("Parent transcript", built["prompt"])
        self.assertNotIn("evidence_turn_refs", built["prompt"])

    def test_outline_semantic_output_passes_all_distinct_pairs(self):
        plan = {"scenes": [
            {"scene_ref": "S01", "participants": ["A"],
             "required_transition": "A starts the walk.",
             "after_state": "The walk is underway."},
            {"scene_ref": "S02", "participants": ["B"],
             "required_transition": "B sends the report.",
             "after_state": "The report is received."},
            {"scene_ref": "S03", "participants": ["C"],
             "required_transition": "C schedules a visit.",
             "after_state": "The visit is scheduled."},
        ]}
        output = {"pair_reviews": [
            {"pair_ref": "P01", "decision": "distinct", "reason": "different actors"},
            {"pair_ref": "P02", "decision": "distinct", "reason": "different actors"},
            {"pair_ref": "P03", "decision": "distinct", "reason": "different actors"},
        ]}
        result = validate_planner_outline_semantic_output(output, plan=plan)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["overall_decision"], "pass")

    def test_outline_semantic_output_duplicate_is_semantic_stop(self):
        plan = {"scenes": [
            {"scene_ref": "S01", "participants": ["A"],
             "required_transition": "A starts the walk.",
             "after_state": "The walk is underway."},
            {"scene_ref": "S02", "participants": ["A"],
             "required_transition": "A begins the same walk.",
             "after_state": "The walk is underway again."},
        ]}
        output = {"pair_reviews": [{
            "pair_ref": "P01", "decision": "duplicate",
            "reason": "Both scenes start the same walk.",
        }]}
        result = validate_planner_outline_semantic_output(output, plan=plan)
        self.assertTrue(result["valid"])
        self.assertEqual(result["overall_decision"], "duplicate")
        self.assertEqual(result["errors"], [
            "planner_outline_semantic_duplicate:P01"])

    def test_outline_pairs_cover_full_candidate_pool_with_memory_source(self):
        plan = {
            "scenes": [{
                "memory_event_id": "ME01",
                "next_step_candidates": [{
                    "required_transition": "A starts the walk.",
                    "after_state": "The walk is underway.",
                }],
            }],
            "candidate_pool": [
                {
                    "memory_event_id": "ME01",
                    "memory_event_ref": "ME01",
                    "memory_event_key": "Saturday walk",
                    "memory_event_content": ["A agreed to take a walk."],
                    "supporting_sources": [{
                        "source_type": "person_event_fact",
                        "memory_key": "A",
                    }],
                    "next_step_candidates": [{
                        "required_transition": "A starts the walk.",
                        "after_state": "The walk is underway.",
                    }],
                },
                {
                    "memory_event_id": "ND01",
                    "source_basis": ["B recently joined a class."],
                    "next_step_candidates": [{
                        "required_transition": "B meets the instructor.",
                        "after_state": "B knows the class procedure.",
                    }],
                },
                {
                    "memory_event_id": "ND02",
                    "source_basis": ["A has an unused camera."],
                    "next_step_candidates": [{
                        "required_transition": "A lends B the camera.",
                        "after_state": "B has the camera.",
                    }],
                },
                {
                    "memory_event_id": "ND03",
                    "source_basis": ["B needs a route map."],
                    "next_step_candidates": [{
                        "required_transition": "A gives B a route map.",
                        "after_state": "B has a route map.",
                    }],
                },
            ],
        }
        pairs = build_planner_outline_pairs(plan)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(pairs[0]["left"]["candidate_ref"], "ME01:C01")
        self.assertEqual(
            pairs[0]["left"]["memory_source"]["memory_event_key"],
            "Saturday walk")
        self.assertEqual(
            pairs[-1]["right"]["candidate_ref"], "ND03:C01")

    def test_outline_semantic_filters_later_duplicate_and_keeps_fallback(self):
        plan = {"candidate_pool": [
            {"memory_event_id": "ME01", "next_step_candidates": [{
                "required_transition": "A starts the walk.",
                "after_state": "The walk begins."}]},
            {"memory_event_id": "ND01", "next_step_candidates": [{
                "required_transition": "B submits the report.",
                "after_state": "The report is submitted."}]},
            {"memory_event_id": "ND02", "next_step_candidates": [{
                "required_transition": "A completes the same walk.",
                "after_state": "The walk is complete."}]},
            {"memory_event_id": "ND03", "next_step_candidates": [{
                "required_transition": "B meets the new cashier.",
                "after_state": "B knows the cashier."}]},
        ]}
        pairs = build_planner_outline_pairs(plan)
        output = {"pair_reviews": [{
            "pair_ref": row["pair_ref"],
            "decision": ("duplicate" if
                row["left"]["candidate_ref"] == "ME01:C01" and
                row["right"]["candidate_ref"] == "ND02:C01"
                else "distinct"),
            "reason": "same walk action" if
                row["right"]["candidate_ref"] == "ND02:C01"
                else "different development",
        } for row in pairs]}
        result = validate_planner_outline_semantic_output(output, plan=plan)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["blocked_candidate_refs"], ["ND02:C01"])
        self.assertEqual(result["kept_candidate_refs"], [
            "ME01:C01", "ND01:C01", "ND03:C01"])

    def test_outline_semantic_missing_last_pair_is_conservative_not_distinct(self):
        plan = {"candidate_pool": [
            {"memory_event_id": "ME01", "next_step_candidates": [{
                "required_transition": "A starts the walk.",
                "after_state": "The walk begins."}]},
            {"memory_event_id": "ND01", "next_step_candidates": [{
                "required_transition": "B submits the report.",
                "after_state": "The report is submitted."}]},
            {"memory_event_id": "ND02", "next_step_candidates": [{
                "required_transition": "C opens the gate.",
                "after_state": "The gate is open."}]},
            {"memory_event_id": "ND03", "next_step_candidates": [{
                "required_transition": "D joins the meeting.",
                "after_state": "D is present."}]},
        ]}
        pairs = build_planner_outline_pairs(plan)
        output = {"pair_reviews": [
            {"pair_ref": row["pair_ref"], "decision": "distinct",
             "reason": "different development"}
            for row in pairs[:-1]
        ]}
        result = validate_planner_outline_semantic_output(output, plan=plan)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["missing_pair_refs"], ["P06"])
        self.assertIn("P06", result["uncertain_pair_refs"])
        self.assertIn(["ND02:C01", "ND03:C01"], result["conflict_edges"])
        selection = select_ranked_legal_candidate_set(
            result["kept_candidate_refs"], outline_audit=result,
            required_count=3)
        self.assertTrue(selection["valid"])
        self.assertEqual(selection["selected_candidate_refs"], [
            "ME01:C01", "ND01:C01", "ND02:C01"])

    def test_uncertain_graph_uses_legal_combination_not_greedy_deletion(self):
        audit = {"conflict_edges": [
            ["A", "B"], ["A", "C"],
        ]}
        result = select_ranked_legal_candidate_set(
            ["A", "B", "C", "D"], outline_audit=audit,
            required_count=3)
        self.assertTrue(result["valid"])
        self.assertEqual(result["selected_candidate_refs"], ["B", "C", "D"])

    def test_uncertain_graph_stops_when_no_legal_three_scene_set_exists(self):
        audit = {"conflict_edges": [
            ["A", "B"], ["A", "C"], ["A", "D"],
            ["B", "C"], ["B", "D"], ["C", "D"],
        ]}
        result = select_ranked_legal_candidate_set(
            ["A", "B", "C", "D"], outline_audit=audit,
            required_count=3)
        self.assertFalse(result["valid"])
        self.assertEqual(result["selected_candidate_refs"], [])

    def test_compiler_binds_continuation_to_programmatic_memory_source(self):
        candidates = [{
            "memory_event_id": "ME01",
            "memory_event_key": "Saturday walk",
            "memory_event_display": "Saturday walk",
            "memory_event_content": ["A agreed to walk on Saturday."],
            "supporting_sources": [{
                "source_type": "person_event_fact",
                "memory_key": "A",
            }],
        }]
        plan = _split_plan(1, 3)
        compiled = compile_planner_output(plan, candidates=candidates)
        scene = compiled["candidate_pool"][0]
        self.assertEqual(scene["verified_track_ref"], "VT01")
        self.assertEqual(scene["memory_event_ref"], "ME01")
        self.assertEqual(scene["memory_event_key"], "Saturday walk")
        self.assertEqual(scene["memory_event_content"], [
            "A agreed to walk on Saturday."])
        self.assertEqual(scene["supporting_sources"][0]["memory_key"], "A")

    def test_outline_audit_allows_different_stage_or_action(self):
        plan = {"scenes": [
            {"scene_ref": "S01", "memory_event_id": "ME05",
             "participants": ["A", "B"],
             "required_transition": "A and B plan a walk at the new trail.",
             "after_state": "The walk remains scheduled for next weekend."},
            {"scene_ref": "S02", "memory_event_id": "ND01",
             "participants": ["A", "B", "Elena"],
             "required_transition": "A introduces Elena to Marco during the walk.",
             "after_state": "Elena connects with another outdoor enthusiast."},
        ]}
        result = validate_planner_outline_distinctness(plan)
        self.assertTrue(result["valid"], result["errors"])

    def test_three_existing_events_may_select_one_and_add_two(self):
        plan = _split_plan(1, 3)
        result = validate_planner_output(
            plan, candidates=self._candidates(3), parent=_parent())
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["required_new_candidate_count"], 3)

    def test_three_existing_events_may_select_two_and_add_one(self):
        plan = _split_plan(2, 2)
        result = validate_planner_output(
            plan, candidates=self._candidates(3), parent=_parent())
        self.assertTrue(result["valid"])
        self.assertEqual(result["selected_continuation_count"], 2)
        self.assertEqual(result["raw_new_candidate_count"], 2)

    def test_gate_uses_ranked_new_fallback_after_rejection(self):
        plan = _split_plan(2, 2)
        candidates = self._candidates(2)
        validation = validate_planner_output(
            plan, candidates=candidates, parent=_parent())
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_output(
            plan, candidates=candidates,
            selected_new_candidate_indices=validation[
                "selected_new_candidate_indices"],
            valid_new_candidate_indices=validation[
                "valid_new_candidate_indices"])
        gate = _gate_reviews(event_count=2, candidates_per_event=1)
        gate["candidate_reviews"].extend([
            {"candidate_ref": "ND01:C01", "parent_relation": "REALIZED",
             "decision": "reject", "duplicate_with_parent": True,
             "evidence_turn_refs": ["root:P01:T001"],
             "reason": "repeats Parent"},
            {"candidate_ref": "ND02:C01", "parent_relation": "NONE",
             "decision": "pass", "duplicate_with_parent": False,
             "evidence_turn_refs": [], "reason": "new"},
        ])
        result = validate_semantic_gate_output(
            gate, plan=compiled, parent=_parent())
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["passed_candidate_count"], 3)
        materialized = materialize_planner_output(
            compiled, passed_candidate_refs=result["passed_candidate_refs"])
        self.assertEqual(
            [row["selected_candidate_ref"] for row in materialized["scenes"]],
            ["ME01:C01", "ME02:C01", "ND02:C01"])

    def test_one_existing_event_compiles_one_continuation_and_two_new(self):
        plan = _split_plan(1, 3)
        candidates = self._candidates(1)
        self.assertTrue(validate_planner_output(
            plan, candidates=candidates, parent=_parent())["valid"])
        compiled = compile_planner_output(plan, candidates=candidates)
        self.assertEqual([row["memory_event_id"] for row in compiled["scenes"]],
                         ["ME01", "ND01", "ND02"])
        self.assertEqual([row["development_mode"] for row in compiled["scenes"]],
                         ["continuation", "new_development", "new_development"])
        materialized = materialize_planner_output(
            compiled, passed_candidate_refs=["ME01:C01", "ND01:C01", "ND02:C01"])
        self.assertEqual(materialized["scenes"][1]["new_development"],
                         plan["new_development_candidates"][0]["required_transition"])
        self.assertEqual(materialized["scenes"][1]["durable_outcome"],
                         plan["new_development_candidates"][0]["after_state"])
        self.assertNotIn("completion_conditions", materialized["scenes"][1])
        self.assertNotIn("next_step_candidates", materialized["scenes"][1])

    def test_zero_existing_events_can_compile_three_new_development_candidates(self):
        plan = _split_plan(0, 3)
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertTrue(result["valid"])
        compiled = compile_planner_output(plan, candidates=[])
        self.assertEqual([row["memory_event_id"] for row in compiled["scenes"]],
                         ["ND01", "ND02", "ND03"])

    def test_duplicate_memory_event_ref_is_rejected(self):
        plan = _split_plan(2, 1)
        plan["selected_continuations"][1]["verified_track_ref"] = "VT01"
        result = validate_planner_output(
            plan, candidates=self._candidates(2), parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("continuation_2_duplicate_verified_track_ref", result["errors"])

    def test_insufficient_new_candidate_pool_is_rejected(self):
        plan = _split_plan(2, 1)
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("insufficient_new_development_candidates", result["errors"])

    def test_programmatic_quota_materializes_ranked_prefix(self):
        for continuation_count, expected_new in ((0, 3), (1, 2),
                                                 (2, 1), (3, 0)):
            plan = _split_plan(continuation_count, 3)
            candidates = self._candidates(max(continuation_count, 1))
            result = validate_planner_output(
                plan, candidates=candidates, parent=_parent())
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["remaining_new_quota"], expected_new)
            self.assertEqual(result["materialized_new_development_count"],
                             expected_new)
            compiled = compile_planner_output(plan, candidates=candidates)
            self.assertEqual(len(compiled["scenes"]), 3)
            self.assertEqual(
                compiled["scene_quota_audit"]["unused_new_candidate_count"],
                3 - expected_new)

    def test_four_new_candidates_are_a_bounded_ranked_pool(self):
        plan = _split_plan(1, 4)
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["raw_new_candidate_count"], 4)
        self.assertEqual(result["selected_new_candidate_indices"], [1, 2])
        self.assertEqual(result["final_materialized_scene_count"], 3)
        self.assertEqual(result["unused_new_candidate_count"], 2)

    def test_fifth_new_candidate_still_exceeds_bounded_pool(self):
        plan = _split_plan(1, 5)
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("new_development_candidates_max_4", result["errors"])

    def test_invalid_ranked_candidate_is_skipped_for_valid_fallback(self):
        plan = _split_plan(1, 3)
        plan["new_development_candidates"][1]["required_transition"] = (
            "A discusses the already completed repayment")
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["valid_new_candidate_indices"], [1, 3])
        self.assertEqual(result["selected_new_candidate_indices"], [1, 3])
        self.assertEqual(result["skipped_new_candidate_indices"], [2])
        compiled = compile_planner_output(
            plan, candidates=self._candidates(1),
            selected_new_candidate_indices=result[
                "selected_new_candidate_indices"])
        self.assertEqual(
            [row["memory_event_id"] for row in compiled["scenes"]],
            ["ME01", "ND01", "ND03"])
        self.assertEqual(
            compiled["scene_quota_audit"]["skipped_new_candidate_indices"],
            [2])

    def test_invalid_candidates_still_block_when_quota_cannot_be_filled(self):
        plan = _split_plan(1, 2)
        plan["new_development_candidates"][0]["required_transition"] = (
            "A discusses the already completed repayment")
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("insufficient_valid_new_development_candidates",
                      result["errors"])
        self.assertIn("new_development_candidate_1_weak_transition_candidate",
                      result["errors"])

    def test_new_development_requires_source_basis(self):
        plan = _split_plan(1, 2)
        plan["new_development_candidates"][0]["source_basis"] = []
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("new_development_candidate_1_source_basis", result["errors"])

    def test_one_declared_new_person_can_participate(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0] = _new_development(
            1,
            new_people=[{"name": "Elena",
                         "relationship_basis": "new neighbor of A"}],
            participants=["A", "Elena"])
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_two_declared_new_people_can_participate(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0] = _new_development(
            1,
            new_people=[
                {"name": "Elena", "relationship_basis": "new neighbor of A"},
                {"name": "Marco",
                 "relationship_basis": "Elena's partner and new neighbor"},
            ],
            participants=["A", "Elena", "Marco"])
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_undeclared_non_parent_participant_is_rejected(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0] = _new_development(
            1,
            new_people=[{"name": "Elena",
                         "relationship_basis": "new neighbor of A"}],
            participants=["A", "Elena", "Marco"])
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("new_development_candidate_1_participant_not_declared",
                      result["errors"])

    def test_undeclared_actor_is_rejected(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0]["actor_bindings"] = [{
            "actor": "Elena", "binding": "Elena owns the key action."}]
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("new_development_candidate_1_actor_not_declared", result["errors"])

    def test_three_new_people_are_rejected(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0]["new_people"] = [
            {"name": name, "relationship_basis": f"new neighbor {name}"}
            for name in ("Elena", "Marco", "Noor")]
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("new_development_candidate_1_new_people", result["errors"])

    def test_compound_new_person_name_is_rejected(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0]["new_people"] = [{
            "name": "Elena and Marco",
            "relationship_basis": "new neighbors of A",
        }]
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("new_development_candidate_1_new_person_1_compound_name",
                      result["errors"])

    def test_new_person_relationship_basis_is_required(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0]["new_people"] = [{
            "name": "Elena", "relationship_basis": ""}]
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("new_development_candidate_1_new_person_1_relationship_basis",
                      result["errors"])

    def test_continuation_cannot_introduce_new_people(self):
        plan = _split_plan(1, 2)
        plan["selected_continuations"][0]["new_people"] = [{
            "name": "Elena", "relationship_basis": "new neighbor of A"}]
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("continuation_1_new_people_forbidden", result["errors"])

    def test_pure_talk_transition_is_rejected(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0]["required_transition"] = (
            "A talks with Elena about local interests.")
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("new_development_candidate_1_weak_transition_candidate",
                      result["errors"])

    def test_concrete_call_transition_is_scene_completable(self):
        plan = _split_plan(1, 3)
        plan["selected_continuations"][0]["required_transition"] = (
            "A calls Elena and agrees on a time to meet.")
        plan["selected_continuations"][0]["after_state"] = (
            "A has called Elena and the meeting time is scheduled.")
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_reaffirmed_agreement_is_scene_completable(self):
        plan = _split_plan(1, 3)
        plan["selected_continuations"][0]["required_transition"] = (
            "A and B reaffirm their mutual agreement and trust.")
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_suggested_action_is_scene_completable(self):
        plan = _split_plan(1, 3)
        plan["selected_continuations"][0]["required_transition"] = (
            "B acknowledges A's help and suggests A lead the next team task.")
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_pure_acknowledgement_remains_weak(self):
        plan = _split_plan(1, 3)
        plan["selected_continuations"][0]["required_transition"] = (
            "B acknowledges A's help and reflects on their teamwork.")
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("continuation_1_weak_transition_candidate", result["errors"])

    def test_encouragement_action_is_scene_completable(self):
        plan = _split_plan(1, 3)
        plan["selected_continuations"][0]["required_transition"] = (
            "B acknowledges A's progress and encourages A to continue the new behavior.")
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_future_only_after_state_is_rejected_before_gate(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0]["after_state"] = (
            "A will eventually complete a weekly walking routine.")
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn(
            "new_development_candidate_1_future_only_after_state",
            result["errors"])

    def test_future_only_unused_fallback_does_not_block_filled_quota(self):
        plan = _split_plan(1, 3)
        plan["new_development_candidates"][0]["after_state"] = (
            "A will eventually complete a weekly walking routine.")
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["valid_new_candidate_indices"], [2, 3])
        self.assertEqual(result["selected_new_candidate_indices"], [2, 3])
        self.assertEqual(result["final_materialized_scene_count"], 3)

    def test_current_agreement_with_future_date_is_scene_completable(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0]["after_state"] = (
            "A and B agree to walk together on Saturday.")
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_current_plural_schedule_with_future_action_is_completable(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0]["after_state"] = (
            "A and B have scheduled a weekly practice, and A will help B.")
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_current_report_with_future_follow_up_is_completable(self):
        plan = _split_plan(3, 0)
        plan["selected_continuations"][2]["after_state"] = (
            "A has sent the first update, and B has confirmed he will send "
            "the report by the end of the week.")
        candidates = [{"memory_event_id": f"ME{i:02d}",
                       "memory_event_key": f"event-{i}"}
                      for i in range(1, 4)]
        tracks = [{"verified_track_ref": f"VT{i:02d}",
                   "memory_event_ref": f"ME{i:02d}",
                   "verified_current_state": f"current state {i}"}
                  for i in range(1, 4)]
        result = validate_planner_output(
            plan, candidates=candidates, verified_tracks=tracks, parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_current_feeling_and_existing_plan_is_scene_completable(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0]["after_state"] = (
            "Elena feels more connected and has a plan to improve her yard.")
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_planner_prompt_defines_bounded_scene_outcome(self):
        prompt = build_planner_prompt(
            parent=_parent(), memory_overview={}, candidates=[],
            memory_only_fast=True, strict_speaker_refs=True)["prompt"]
        self.assertIn("bounded mini-arc", prompt)
        self.assertIn("smallest stable result already true", prompt)
        self.assertIn("over-scoped future outcome", prompt)
        self.assertIn("agree to walk together on Saturday", prompt)

    def test_talk_with_explicit_state_change_is_allowed(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0]["required_transition"] = (
            "A meets Elena, talks briefly, and exchanges contact information "
            "with her.")
        result = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertTrue(result["valid"], result["errors"])

    def test_program_materializes_new_person_refs_and_realizer_accepts_speaker(self):
        plan = _split_plan(0, 3)
        plan["new_development_candidates"][0] = _new_development(
            1,
            new_people=[{"name": "Elena",
                         "relationship_basis": "new neighbor of A"}],
            participants=["A", "Elena"])
        validation = validate_planner_output(plan, candidates=[], parent=_parent())
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_output(plan, candidates=[])
        self.assertEqual(compiled["scenes"][0]["new_people"][0]["person_ref"],
                         "NP01")
        materialized = materialize_planner_output(
            compiled, passed_candidate_refs=["ND01:C01", "ND02:C01", "ND03:C01"])
        raw = {"scenes": [
            {"dialogue_turns": [
                {"speaker": "A" if index % 2 else "Elena",
                 "text": f"first scene turn {index}"}
                for index in range(1, 10)]},
            {"dialogue_turns": [
                {"speaker": "A" if index % 2 else "B",
                 "text": f"second scene turn {index}"}
                for index in range(1, 10)]},
            {"dialogue_turns": [
                {"speaker": "A" if index % 2 else "B",
                 "text": f"third scene turn {index}"}
                for index in range(1, 10)]},
        ]}
        realized = normalize_realizer_output(
            raw, plan=materialized, parent=_parent())
        self.assertTrue(realized["valid"], realized["errors"])

    def test_fallback_candidate_reaches_gate_but_not_realizer_when_unselected(self):
        plan = _split_plan(1, 3)
        plan["new_development_candidates"][2]["interaction_beats"] = [
            "UNUSED CANDIDATE ONLY", "unused second beat", "unused third beat"]
        candidates = self._candidates(1)
        validation = validate_planner_output(
            plan, candidates=candidates, parent=_parent())
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_output(plan, candidates=candidates)
        self.assertEqual(
            [scene["memory_event_id"] for scene in compiled["scenes"]],
            ["ME01", "ND01", "ND02"])
        self.assertEqual(
            compiled["scene_quota_audit"]["unused_new_development_candidates"]
            [0]["interaction_beats"][0], "UNUSED CANDIDATE ONLY")
        gate_context = build_semantic_gate_context(
            parent=_parent(), candidates=candidates, plan=compiled)
        self.assertEqual(gate_context["expected_candidate_refs"],
                         ["ME01:C01", "ND01:C01", "ND02:C01", "ND03:C01"])
        serialized_gate = json.dumps(gate_context)
        self.assertIn("UNUSED CANDIDATE ONLY", serialized_gate)
        clean = materialize_planner_output(
            compiled, passed_candidate_refs=gate_context["expected_candidate_refs"])
        self.assertNotIn("scene_quota_audit", clean)
        serialized_realizer = build_realizer_prompt(
            parent=_parent(), plan=clean)["prompt"]
        self.assertNotIn("UNUSED CANDIDATE ONLY", serialized_realizer)

    def test_rejected_continuations_are_replaced_by_later_passing_new_candidates(self):
        plan = _split_plan(2, 3)
        candidates = self._candidates(2)
        validation = validate_planner_output(plan, candidates=candidates, parent=_parent())
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_output(plan, candidates=candidates)
        refs = ["ME01:C01", "ME02:C01", "ND01:C01", "ND02:C01", "ND03:C01"]
        gate = validate_semantic_gate_output(
            _reviews_for_refs(refs, reject_ref="ME01:C01"),
            plan=compiled, parent=_parent())
        self.assertTrue(gate["valid"], gate["errors"])
        self.assertTrue(gate["realizer_allowed"])
        self.assertEqual(gate["passed_candidate_count"], 4)
        clean = materialize_planner_output(
            compiled, passed_candidate_refs=gate["passed_candidate_refs"])
        self.assertEqual(
            [scene["memory_event_id"] for scene in clean["scenes"]],
            ["ME02", "ND01", "ND02"])
        self.assertNotIn("ME01", [scene["memory_event_id"] for scene in clean["scenes"]])

    def test_unused_weak_candidate_does_not_block_ranked_prefix(self):
        plan = _split_plan(1, 3)
        plan["new_development_candidates"][2]["required_transition"] = (
            "A talks about what might happen later.")
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertTrue(result["valid"], result["errors"])
        compiled = compile_planner_output(
            plan, candidates=self._candidates(1),
            valid_new_candidate_indices=result["valid_new_candidate_indices"])
        self.assertEqual(len(compiled["scenes"]), 3)
        self.assertEqual(
            compiled["scene_quota_audit"]["unused_new_development_candidates"]
            [0]["required_transition"],
            "A talks about what might happen later.")
        gate_context = build_semantic_gate_context(
            parent=_parent(), candidates=self._candidates(1), plan=compiled)
        self.assertEqual(gate_context["expected_candidate_refs"],
                         ["ME01:C01", "ND01:C01", "ND02:C01"])

    def test_only_materialized_candidates_receive_nd_and_person_refs(self):
        plan = _split_plan(2, 3)
        plan["new_development_candidates"][0]["new_people"] = [{
            "name": "Elena", "relationship_basis": "new neighbor of A"}]
        plan["new_development_candidates"][1]["new_people"] = [{
            "name": "Marco", "relationship_basis": "new neighbor of A"}]
        compiled = compile_planner_output(plan, candidates=self._candidates(2))
        self.assertEqual([row["memory_event_id"] for row in compiled["scenes"]],
                         ["ME01", "ME02", "ND01"])
        self.assertEqual(compiled["scenes"][2]["new_people"][0]["person_ref"],
                         "NP01")
        self.assertEqual(
            compiled["scene_quota_audit"]["unused_new_development_candidates"]
            [0]["new_people"][0].get("person_ref"), None)

    def test_state_transition_contract_stays_frozen_and_weak_transition_fails(self):
        plan = _split_plan(1, 3)
        self.assertTrue(validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())["valid"])
        plan["new_development_candidates"][0]["completion_conditions"] = ["legacy"]
        self.assertFalse(validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())["valid"])
        plan["new_development_candidates"][0].pop("completion_conditions")
        plan["new_development_candidates"][0]["required_transition"] = (
            "A discusses whether to act.")
        result = validate_planner_output(
            plan, candidates=self._candidates(1), parent=_parent())
        self.assertTrue(result["valid"], result["errors"])
        self.assertFalse(result["candidate_validation"][0]["valid"])
        self.assertIn("new_development_candidate_1_weak_transition_candidate",
                      result["candidate_validation"][0]["errors"])
        # The invalid ranked candidate is quarantined because two valid
        # fallbacks remain for the two required new-development slots.

    def test_rejected_candidate_is_absent_from_realizer_input_and_prompt(self):
        raw_plan={"round_goal":"g","scenes":[{
            "memory_event_id":f"ME{i:02d}","verified_current_state":"state",
            "evidence_turn_refs":["root:P01:T001"],"goal":"g",
            "participants":["A","B"],**_next_fields(i),
            "transition_from_previous":"t"}
            for i in range(1,4)]}
        passed=["ME01:C02","ME02:C01","ME03:C01"]
        clean=materialize_planner_output(raw_plan,passed_candidate_refs=passed)
        isolation=validate_realizer_plan_isolation(
            raw_plan=raw_plan,materialized_plan=clean,
            passed_candidate_refs=passed)
        self.assertTrue(isolation["valid"])
        self.assertEqual(isolation["rejected_content_leak_count"],0)
        self.assertEqual(clean["scenes"][0]["interaction_beats"],
                         ["selected-only-1","later beat 2","later beat 3"])
        self.assertEqual(clean["scenes"][0]["actor_bindings"],
                         [{"actor":"B","binding":"selected-binding-1"}])
        self.assertEqual(clean["scenes"][0]["required_transition"],
                         "B completes selected transition 1")
        self.assertEqual(clean["scenes"][0]["after_state"],
                         "selected end state 1")
        self.assertNotIn("completion_conditions", clean["scenes"][0])
        prompt=build_realizer_prompt(parent=_parent(),plan=clean)["prompt"]
        self.assertIn("selected-only-1",prompt)
        self.assertIn("selected-binding-1",prompt)
        self.assertIn("B completes selected transition 1",prompt)
        self.assertIn("selected end state 1",prompt)
        self.assertNotIn("rejected-only-1",prompt)
        self.assertNotIn("rejected-binding-1",prompt)
        self.assertNotIn("rejected end state 1",prompt)
        self.assertNotIn("next_step_candidates",prompt)

    def test_realizer_program_assigns_scene_local_turn_refs(self):
        parent=_parent(); plan={"scenes":[{"memory_event_id":"ME01","participants":["A","B"]}]}
        raw={"scenes":[{"dialogue_turns":[
            {"speaker":"A" if i % 2 else "B", "text":f"child {i}"}
            for i in range(9)]}]}
        result=normalize_realizer_output(raw,plan=plan,parent=parent)
        self.assertTrue(result["valid"]); self.assertEqual(result["child"]["scenes"][0]["dialogue_turns"][0]["turn_ref"],"R01:S01:T001")

    def test_semantic_gate_pass_and_reject_contracts(self):
        parent=_parent()
        plan={"scenes":[{"memory_event_id":"ME01",**_next_fields(1)},
                        {"memory_event_id":"ME02",**_next_fields(2)}]}
        passed=_gate_reviews(event_count=2)
        result=validate_semantic_gate_output(passed,plan=plan,parent=parent)
        self.assertTrue(result["valid"]); self.assertTrue(result["realizer_allowed"])

        first_rejected=_gate_reviews(event_count=2)
        first_rejected["candidate_reviews"][0]={
            "candidate_ref":"ME01:C01","parent_relation":"REALIZED","decision":"reject",
            "duplicate_with_parent":True,
            "evidence_turn_refs":["root:P01:T002"],
            "reason":"The first candidate repeats a Parent action."}
        result=validate_semantic_gate_output(first_rejected,plan=plan,parent=parent)
        self.assertTrue(result["valid"])
        self.assertTrue(result["realizer_allowed"])
        self.assertEqual(result["overall_decision"],"pass")
        self.assertIn("ME01:C02",result["passed_candidate_refs"])

        rejected=_gate_reviews(event_count=2,reject_event=2,
                               parent_ref="root:P01:T002")
        result=validate_semantic_gate_output(rejected,plan=plan,parent=parent)
        self.assertTrue(result["valid"]); self.assertFalse(result["realizer_allowed"])
        self.assertEqual(result["overall_decision"],"reject")

    def test_gate_coverage_one_memory_and_two_new(self):
        plan = compile_planner_output(
            _split_plan(1, 2), candidates=self._candidates(1))
        refs = ["ME01:C01", "ND01:C01", "ND02:C01"]
        result = validate_semantic_gate_output(
            _reviews_for_refs(refs), plan=plan, parent=_parent())
        self.assertTrue(result["valid"])
        self.assertEqual(result["passed_candidate_refs"], refs)

    def test_gate_coverage_three_memory_candidates(self):
        plan = compile_planner_output(
            _split_plan(3, 0), candidates=self._candidates(3))
        refs = ["ME01:C01", "ME02:C01", "ME03:C01"]
        self.assertTrue(validate_semantic_gate_output(
            _reviews_for_refs(refs), plan=plan, parent=_parent())["valid"])

    def test_gate_coverage_three_new_candidates(self):
        plan = compile_planner_output(_split_plan(0, 3), candidates=[])
        refs = ["ND01:C01", "ND02:C01", "ND03:C01"]
        self.assertTrue(validate_semantic_gate_output(
            _reviews_for_refs(refs), plan=plan, parent=_parent())["valid"])

    def test_gate_missing_continuation_is_rejected(self):
        plan = compile_planner_output(
            _split_plan(1, 2), candidates=self._candidates(1))
        result = validate_semantic_gate_output(
            _reviews_for_refs(["ND01:C01", "ND02:C01"]),
            plan=plan, parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("gate_review_count", result["errors"])
        self.assertIn("gate_candidate_coverage", result["errors"])

    def test_gate_missing_new_development_is_rejected(self):
        plan = compile_planner_output(_split_plan(0, 3), candidates=[])
        result = validate_semantic_gate_output(
            _reviews_for_refs(["ND01:C01", "ND02:C01"]),
            plan=plan, parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("gate_candidate_coverage", result["errors"])

    def test_gate_duplicate_output_ref_is_rejected(self):
        plan = compile_planner_output(
            _split_plan(1, 2), candidates=self._candidates(1))
        result = validate_semantic_gate_output(
            _reviews_for_refs(["ME01:C01", "ME01:C01", "ND01:C01"]),
            plan=plan, parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("gate_review_2_duplicate_candidate", result["errors"])
        self.assertIn("gate_candidate_coverage", result["errors"])

    def test_gate_unknown_and_altered_refs_are_rejected(self):
        plan = compile_planner_output(
            _split_plan(1, 2), candidates=self._candidates(1))
        for bad_ref in ("ME99:C01", "ME1:C1"):
            result = validate_semantic_gate_output(
                _reviews_for_refs([bad_ref, "ND01:C01", "ND02:C01"]),
                plan=plan, parent=_parent())
            self.assertFalse(result["valid"])
            self.assertIn("gate_review_1_unknown_candidate", result["errors"])

    def test_gate_mixed_pass_reject_preserves_existing_contract(self):
        plan = compile_planner_output(
            _split_plan(1, 2), candidates=self._candidates(1))
        refs = ["ME01:C01", "ND01:C01", "ND02:C01"]
        result = validate_semantic_gate_output(
            _reviews_for_refs(refs, reject_ref="ND01:C01"),
            plan=plan, parent=_parent())
        self.assertTrue(result["valid"])
        self.assertFalse(result["realizer_allowed"])
        self.assertIn("ND01:C01", result["rejected_candidate_refs"])

    def test_semantic_gate_derives_missing_reason_for_all_decisions(self):
        parent=_parent()
        plan={"scenes":[{"memory_event_id":"ME01",**_next_fields(1)}]}
        reviews=_gate_reviews(event_count=1)
        for review in reviews["candidate_reviews"]:
            review.pop("reason")
        result=validate_semantic_gate_output(reviews,plan=plan,parent=parent)
        self.assertTrue(result["valid"])
        self.assertFalse(result["realizer_allowed"])
        self.assertEqual(result["passed_candidate_count"], 2)
        self.assertEqual(
            result["normalized_candidate_reviews"][0]["reason"],
            "No realized Parent transition was identified.")

        reviews["candidate_reviews"][0]={
            "candidate_ref":"ME01:C01","parent_relation":"REALIZED","decision":"reject",
            "duplicate_with_parent":True,
            "evidence_turn_refs":["root:P01:T001"]}
        result=validate_semantic_gate_output(reviews,plan=plan,parent=parent)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(
            result["normalized_candidate_reviews"][0]["reason"],
            "The candidate duplicates a transition already realized in "
            "the authoritative Parent.")

    def test_semantic_gate_rejects_non_parent_evidence(self):
        reviews=_gate_reviews(event_count=1)
        reviews["candidate_reviews"][0]={
            "candidate_ref":"ME01:C01","parent_relation":"REALIZED","decision":"reject",
            "duplicate_with_parent":True,"evidence_turn_refs":["R01:S01:T001"],
            "reason":"duplicate"}
        result=validate_semantic_gate_output(reviews,
            plan={"scenes":[{"memory_event_id":"ME01",**_next_fields(1)}]},
            parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("gate_review_1_evidence_not_parent",result["errors"])

    def test_gate_context_includes_complete_parent_session(self):
        parent=_parent()
        context=build_semantic_gate_context(
            parent=parent,candidates=[{"memory_event_id":"ME01"}],
            plan={"scenes":[{"memory_event_id":"ME01",
                             "evidence_turn_refs":["root:P01:T001"],
                             "verified_current_state":"state",
                             **_next_fields(1)}]})
        refs=[row["turn_ref"] for row in context["scenes"][0]["authoritative_parent_evidence"]]
        self.assertEqual(refs,[f"root:P01:T{i:03d}" for i in range(1,7)])
        self.assertEqual(
            [row["candidate_ref"] for row in context["scenes"][0]["candidates"]],
            ["ME01:C01","ME01:C02"])
        self.assertEqual(context["scenes"][0]["candidates"][1]["interaction_beats"],
                         ["selected-only-1","later beat 2","later beat 3"])
        self.assertEqual(context["expected_candidate_count"], 2)
        self.assertEqual(context["expected_candidate_refs"],
                         ["ME01:C01", "ME01:C02"])

    def test_memory_only_fast_gate_context_omits_parent_transcript(self):
        parent = _parent()
        compiled = compile_planner_output(
            _split_plan(0, 3), candidates=self._candidates(0))
        context = build_semantic_gate_context(
            parent=parent, candidates=self._candidates(0), plan=compiled,
            memory_only_fast=True)
        self.assertTrue(context["memory_only_fast"])
        self.assertEqual(
            [len(scene["authoritative_parent_evidence"])
             for scene in context["scenes"]], [0, 0, 0])
        self.assertEqual(
            context["gate_context_version"],
            "semantic_gate_memory_candidate_coverage_v1")
        self.assertIn("planner_verified_current_state", context["scenes"][0])

    def test_memory_only_fast_gate_allows_legacy_empty_evidence(self):
        compiled = compile_planner_output(
            _split_plan(0, 3), candidates=self._candidates(0))
        refs = [str(value) for value in (
            "ND01:C01", "ND02:C01", "ND03:C01")]
        result = validate_semantic_gate_output(
            _reviews_for_refs(refs), plan=compiled, parent=_parent(),
            memory_only_fast=True)
        self.assertTrue(result["valid"], result["errors"])

    def test_gate_prompt_is_narrow_and_non_repairing(self):
        built=build_semantic_continuation_gate_prompt(gate_context={
            "expected_candidate_refs":["ME01:C01","ND01:C01","ND02:C01"],
            "scenes":[]})
        prompt=built["prompt"]
        for category in ("decision","intention","action","agreement","resolution","result"):
            self.assertIn(f"- {category}",prompt)
        self.assertIn("Do not repair, rewrite",prompt)
        self.assertIn("1. ME01:C01", prompt)
        self.assertIn("2. ND01:C01", prompt)
        self.assertIn("3. ND02:C01", prompt)
        for relation in ("REALIZED", "PLANNED_OR_INTENDED",
                         "RELATED_ONLY", "NONE"):
            self.assertIn(relation, prompt)
        self.assertIn("realization detector", prompt)
        self.assertEqual(built["expected_candidate_count"], 3)

    def test_realization_aware_gate_fixed_parent_relation_mapping(self):
        parent = _parent()
        plan = {"scenes": [{"memory_event_id": "ME01", **_next_fields(1)}]}
        refs = ["ME01:C01", "ME01:C02"]
        for relation in ("PLANNED_OR_INTENDED", "RELATED_ONLY", "NONE"):
            reviews = _reviews_for_refs(refs)
            reviews["candidate_reviews"][0]["parent_relation"] = relation
            reviews["candidate_reviews"][0]["evidence_turn_refs"] = (
                ["root:P01:T001"] if relation != "NONE" else [])
            result = validate_semantic_gate_output(
                reviews, plan=plan, parent=parent)
            self.assertTrue(result["valid"], (relation, result["errors"]))
            self.assertEqual(
                result["normalized_candidate_reviews"][0]["decision"], "pass")

        reviews = _reviews_for_refs(refs, reject_ref="ME01:C01")
        result = validate_semantic_gate_output(reviews, plan=plan, parent=parent)
        self.assertTrue(result["valid"])
        self.assertEqual(
            result["normalized_candidate_reviews"][0]["parent_relation"],
            "REALIZED")

    def test_realization_aware_gate_rejects_relation_decision_mismatch(self):
        plan = {"scenes": [{"memory_event_id": "ME01", **_next_fields(1)}]}
        reviews = _reviews_for_refs(["ME01:C01", "ME01:C02"])
        reviews["candidate_reviews"][0].update({
            "parent_relation": "PLANNED_OR_INTENDED",
            "decision": "reject",
            "duplicate_with_parent": True,
            "evidence_turn_refs": ["root:P01:T001"],
            "reason": "wrong mapping",
        })
        result = validate_semantic_gate_output(
            reviews, plan=plan, parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn(
            "gate_review_1_parent_relation_decision_conflict",
            result["errors"])

    def test_planner_prompt_delegates_scene_quota_to_program(self):
        from .prompts import build_planner_prompt
        prompt = build_planner_prompt(
            parent=_parent(), memory_overview={}, candidates=[],
            memory_only_fast=True, strict_speaker_refs=True)["prompt"]
        for phrase in (
                "ranked\nnew_development_candidates",
                "candidate pool, not final scenes",
                "local program owns the fixed scene quota",
                "Do not perform cross-field exact-fill arithmetic"):
            self.assertIn(phrase, prompt)
        self.assertIn("For a selected continuation, do not use", prompt)
        self.assertIn("concrete\nlater state", prompt)
        self.assertIn("transport roles with no character identity", prompt)
        self.assertIn("skip that continuation rather than guessing", prompt)

    def test_realizer_prompt_explicitly_requires_three_long_scenes(self):
        prompt=build_realizer_prompt(
            parent=_parent(),
            plan={"scenes":[{
                "memory_event_id":f"ME{i:02d}",
                "verified_current_state":"state",
                "goal":"goal","participants":["A","B"],
                "before_state":"current state",
                "required_transition":"A completes the next action.",
                "after_state":"The next action is complete.",
                "new_development":"A completes the next action.",
                "durable_outcome":"The next action is complete.",
                "interaction_beats":["beat"],
                "actor_bindings":[{"actor":"A","binding":"A owns next."}],
                "transition_from_previous":"transition"}
                for i in range(1,4)]})
        self.assertEqual(prompt["prompt_version"],
                         "memory_event_continuation_realizer_prompt_v6")
        self.assertIn("len(scenes) == 3",prompt["prompt"])
        self.assertIn("Use exactly 14 dialogue turns per scene by default",
                      prompt["prompt"])
        self.assertIn("complete mini-arc for every scene", prompt["prompt"])
        self.assertIn("8-turn scene is invalid",prompt["prompt"])
        for heading in ("CURRENT STATE", "REQUIRED TRANSITION",
                        "REQUIRED END STATE", "ACTOR BINDINGS",
                        "INTERACTION BEATS"):
            self.assertIn(heading,prompt["prompt"])
        self.assertIn("required_participants is a", prompt["prompt"])
        self.assertIn("not a speaker allowlist", prompt["prompt"])
        self.assertIn("Every required_participant must appear",
                      prompt["prompt"])
        self.assertNotIn("completion_conditions",prompt["prompt"])
        self.assertNotIn("MUST BE TRUE AT END",prompt["prompt"])

    def test_short_tail_style_context_limits_literal_history_and_keeps_text_free_profile(self):
        turns = [{
            "turn_id": f"root:P01:T{index:03d}",
            "speaker": "A" if index % 2 else "B",
            "text": f"unique old line {index}",
        } for index in range(1, 8)]
        context = _build_realizer_style_context(turns)
        self.assertEqual(context["selection_rule"],
                         "last_complete_turn_shapes_max_4_max_800_characters")
        self.assertEqual(context["continuity_tail"]["turn_count"], 4)
        self.assertEqual(
            [row["turn_ref"] for row in context["continuity_tail"]["turn_shapes"]],
            ["root:P01:T004", "root:P01:T005", "root:P01:T006",
             "root:P01:T007"])
        self.assertNotIn("turns", context["continuity_tail"])
        self.assertNotIn("text", context["style_profile"])
        serialized_profile = json.dumps(context["style_profile"])
        self.assertNotIn("unique old line", serialized_profile)

    def test_short_tail_prompt_preserves_continuity_but_excludes_older_verbatim_lines(self):
        turns = [{
            "turn_id": f"root:P01:T{index:03d}",
            "speaker": "A" if index % 2 else "B",
            "text": f"unique old line {index}",
        } for index in range(1, 8)]
        context = _build_realizer_style_context(turns)
        plan = {"scenes": [{
            "memory_event_id": f"ME{index:02d}",
            "verified_current_state": "state", "goal": "goal",
            "participants": ["A", "B"], "before_state": "current",
            "required_transition": "A completes an action.",
            "after_state": "The action is complete.",
            "interaction_beats": ["beat"],
            "actor_bindings": [{"actor": "A", "binding": "A acts."}],
            "transition_from_previous": "transition",
        } for index in range(1, 4)]}
        prompt = build_realizer_prompt(
            parent=_parent(), plan=plan,
            continuity_shape=context["continuity_tail"],
            style_profile=context["style_profile"],
            include_full_parent=False)
        self.assertEqual(prompt["prompt_version"],
                         "memory_event_continuation_realizer_continuity_shape_v1")
        self.assertEqual(prompt["style_excerpt_turn_count"], 0)
        self.assertTrue(prompt["continuity_shape_included"])
        self.assertTrue(prompt["style_profile_included"])
        self.assertNotIn("unique old line 1", prompt["prompt"])
        self.assertNotIn("unique old line 4", prompt["prompt"])
        self.assertIn("CONTINUITY SHAPE ONLY", prompt["prompt"])
        self.assertIn("Previous wording is intentionally omitted", prompt["prompt"])

    def test_complete_zero_api_workflow(self):
        parent=json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers=list(dict.fromkeys(row["speaker"] for row in parent["dialogue"]["turns"]))
        refs=[row["turn_id"] for row in parent["dialogue"]["turns"]]
        events={f"event-{i}":{"display_key":f"Event {i}","content":[f"event fact {i}"]} for i in range(1,4)}

        class Gateway:
            def __init__(self): self.calls=[]; self.search_count=0
            def add(self,payload,*,request_id):
                self.calls.append({"role":"add","request_id":request_id})
                return {"success":True,"data":{"status":"completed"}}
            def search_buckets(self,payload,*,request_id):
                self.calls.append({"role":"search","request_id":request_id}); self.search_count+=1
                if self.search_count == 1:
                    current = {}
                elif self.search_count == 2:
                    current = events
                else:
                    current = {**events, "event-child": {
                        "display_key": "Child event",
                        "content": ["new Child development"],
                    }}
                return {"success":True,"data":{"buckets":{"person":{},"location":{},"event":current,"item":{},"dreaming":{}}}}
            def add_status(self,*args,**kwargs): raise AssertionError("sync fixture must not poll")

        def planner_call(prompt):
            plan = _split_plan(3, 0, evidence_ref=refs[0])
            for i, item in enumerate(plan["selected_continuations"]):
                item["participants"] = speakers[:2]
                item["goal"] = f"goal {i + 1}"
                item["actor_bindings"] = [{
                    "actor": speakers[0],
                    "binding": f"{speakers[0]} owns transition {i + 1}."}]
            return plan

        def realizer_call(prompt):
            return {"scenes":[{"dialogue_turns":[{
                "speaker":speakers[j%2],"text":f"scene {i} generated turn {j}"}
                for j in range(1,10)]} for i in range(1,4)]}

        def semantic_gate_call(prompt):
            return _gate_reviews(event_count=3,parent_ref=refs[0],
                                 candidates_per_event=1)

        outline_calls = []
        def outline_semantic_call(prompt):
            outline_calls.append(prompt)
            return {"pair_reviews": [
                {"pair_ref": pair_ref, "decision": "distinct",
                 "reason": "The transitions have different indexed outcomes."}
                for pair_ref in prompt["expected_pair_refs"]
            ]}

        gateway=Gateway()
        result=execute_round(parent=parent,memorax=gateway,
            verifier_call=lambda prompt: _verified_tracks(3),
            planner_call=planner_call,
            semantic_gate_call=semantic_gate_call,realizer_call=realizer_call,
            adherence_call=lambda prompt: _adherence_reviews(),
            outline_semantic_call=outline_semantic_call,
            token_counter=lambda text:max(1,len(text)//4),
            poll_delays_seconds=(0.0,),sleep_fn=lambda _:None)
        self.assertTrue(result["generation_ready"])
        self.assertEqual(len(outline_calls), 1)
        self.assertEqual(
            result["planner_outline_semantic_audit"]["overall_decision"],
            "pass")
        self.assertEqual(len(result["child_dialogue"]["scenes"]),3)
        self.assertTrue(result["parent_visibility_audit"]["bucket_visibility_verified"])
        self.assertTrue(result["child_visibility_audit"]["bucket_visibility_verified"])
        self.assertTrue(result["memory_visibility_verified"])
        self.assertEqual([row["role"] for row in gateway.calls],
                         ["search","add","search","add","search"])

    def test_skip_planner_pre_gate_keeps_post_realizer_audit(self):
        """The experiment skips only the pre-Realizer Gate LLM request."""
        parent = json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers = list(dict.fromkeys(
            row["speaker"] for row in parent["dialogue"]["turns"]))
        refs = [row["turn_id"] for row in parent["dialogue"]["turns"]]
        events = {f"event-{i}": {
            "display_key": f"Event {i}",
            "content": [f"event fact {i}"],
        } for i in range(1, 4)}

        class Gateway:
            def __init__(self):
                self.calls = []
                self.search_count = 0

            def add(self, payload, *, request_id):
                self.calls.append("add")
                return {"success": True, "data": {"status": "completed"}}

            def search_buckets(self, payload, *, request_id):
                self.search_count += 1
                self.calls.append("search")
                current = {} if self.search_count == 1 else events
                if self.search_count >= 3:
                    current = {**events, "child-event": {
                        "display_key": "Child event",
                        "content": ["new Child development"],
                    }}
                return {"success": True, "data": {"buckets": {
                    "person": {}, "location": {}, "event": current,
                    "item": {}, "dreaming": {},
                }}}

            def add_status(self, *args, **kwargs):
                raise AssertionError("fixture must not poll")

        def planner_call(prompt):
            plan = _split_plan(3, 0, evidence_ref=refs[0])
            for index, item in enumerate(plan["selected_continuations"]):
                item["speaker_refs"] = ["SP01", "SP02"]
                item.pop("participants", None)
                item["goal"] = f"goal {index + 1}"
                item["actor_bindings"] = [{
                    "actor_ref": "SP01",
                    "binding": f"SP01 owns transition {index + 1}.",
                }]
            return plan

        def realizer_call(prompt):
            return {"scenes": [{"dialogue_turns": [{
                "speaker": speakers[index % 2],
                "text": f"scene {scene} generated turn {index}",
            } for index in range(1, 10)]} for scene in range(1, 4)]}

        gateway = Gateway()
        result = execute_round(
            parent=parent, memorax=gateway,
            verifier_call=lambda prompt: self.fail("verifier must be skipped"),
            planner_call=planner_call,
            semantic_gate_call=lambda prompt: self.fail("pre-Gate must be skipped"),
            realizer_call=realizer_call,
            adherence_call=lambda prompt: _adherence_reviews(),
            memory_only_fast=True,
            skip_planner_semantic_gate=True,
            token_counter=lambda text: max(1, len(text) // 4),
            poll_delays_seconds=(0.0,), sleep_fn=lambda _: None)
        self.assertTrue(result["generation_ready"])
        self.assertTrue(result["skip_planner_semantic_gate"])
        self.assertTrue(result["semantic_gate_validation"]["gate_skipped"])
        self.assertEqual(result["semantic_gate_output"]["gate_skipped"], True)
        self.assertEqual(len(result["child_dialogue"]["scenes"]), 3)
        self.assertEqual(gateway.calls,
                         ["search", "add", "search", "add", "search"])

    def test_empty_memory_menu_stops_before_planner(self):
        parent = json.loads(PARENT_PATH.read_text(encoding="utf-8"))

        class Gateway:
            def __init__(self):
                self.calls = []

            def add(self, payload, *, request_id):
                self.calls.append("add")
                return {"success": True, "data": {"status": "completed"}}

            def search_buckets(self, payload, *, request_id):
                self.calls.append("search")
                return {"success": True, "data": {"buckets": {
                    "person": {}, "location": {}, "event": {},
                    "item": {}, "dreaming": {},
                }}}

            def add_status(self, *args, **kwargs):
                raise AssertionError("sync fixture must not poll")

        artifacts = {}
        with self.assertRaises(WorkflowError) as raised:
            execute_round(
                parent=parent, memorax=Gateway(),
                verifier_call=lambda prompt: self.fail("verifier must not run"),
                planner_call=lambda prompt: self.fail("Planner must not run"),
                semantic_gate_call=lambda prompt: self.fail("Gate must not run"),
                realizer_call=lambda prompt: self.fail("Realizer must not run"),
                adherence_call=lambda prompt: self.fail("Auditor must not run"),
                artifact_sink=lambda name, value: artifacts.setdefault(name, value),
                memory_only_fast=True,
                token_counter=lambda text: max(1, len(text) // 4),
                poll_delays_seconds=(0.0,), sleep_fn=lambda _: None)
        self.assertEqual(raised.exception.stage, "memory_event_discovery")
        self.assertEqual(raised.exception.errors,
                         ["no_concrete_memory_candidates"])
        self.assertFalse(artifacts["memory/event_discovery_validation"]["valid"])

    def test_bounded_planner_repair_rechecks_merged_pool_before_realizer(self):
        parent = json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers = list(dict.fromkeys(
            row["speaker"] for row in parent["dialogue"]["turns"]))
        refs = [row["turn_id"] for row in parent["dialogue"]["turns"]]
        person_events = {"person-a": {
            "display_key": "A",
            "content": ["[事件 @event-a] A has a current fact."]}}

        class Gateway:
            def __init__(self):
                self.calls = []
                self.search_count = 0

            def add(self, payload, *, request_id):
                self.calls.append({"role": "add", "request_id": request_id})
                return {"success": True, "data": {"status": "completed"}}

            def search_buckets(self, payload, *, request_id):
                self.calls.append({"role": "search", "request_id": request_id})
                self.search_count += 1
                current = {} if self.search_count == 1 else person_events
                if self.search_count >= 3:
                    current = {**person_events, "person-child": {
                        "display_key": "Child event",
                        "content": ["new child development"],
                    }}
                return {"success": True, "data": {"buckets": {
                    "person": current, "location": {}, "event": {},
                    "item": {}, "dreaming": {},
                }}}

            def add_status(self, *args, **kwargs):
                raise AssertionError("sync fixture must not poll")

        def strictify(plan):
            result = json.loads(json.dumps(plan))
            for group in ("selected_continuations",
                          "new_development_candidates"):
                for item in result[group]:
                    item["speaker_refs"] = ["SP01", "SP02"]
                    item.pop("participants", None)
                    item["actor_bindings"] = [{
                        "actor_ref": "SP01",
                        "binding": "SP01 owns the planned action.",
                    }]
            return result

        def planner_call(prompt):
            return strictify(_split_plan(1, 3, evidence_ref=refs[0]))

        def repair_call(prompt):
            items = []
            for index in (4, 5):
                item = _new_development(index)
                item["speaker_refs"] = ["SP01", "SP02"]
                item.pop("participants", None)
                item["actor_bindings"] = [{
                    "actor_ref": "SP01",
                    "binding": f"SP01 owns replacement action {index}.",
                }]
                items.append(item)
            return {"new_development_candidates": items}

        gate_calls = []

        def gate_call(prompt):
            gate_calls.append(prompt)
            if len(gate_calls) == 1:
                refs = ["ME01:C01", "ND01:C01", "ND02:C01", "ND03:C01"]
                rejected = {"ND01:C01", "ND02:C01"}
            else:
                refs = ["ME01:C01", "ND01:C01", "ND02:C01",
                        "ND03:C01", "ND04:C01", "ND05:C01"]
                rejected = {"ND01:C01", "ND02:C01"}
            result = _reviews_for_refs(refs, parent_ref=parent["dialogue"]["turns"][0]["turn_id"])
            for row in result["candidate_reviews"]:
                row.pop("evidence_turn_refs", None)
                if row["candidate_ref"] in rejected:
                    row.update({"parent_relation": "REALIZED",
                                "decision": "reject",
                                "duplicate_with_parent": True,
                                "supporting_memory_refs": ["GM:P001:C001"],
                                "reason": "duplicate"})
                else:
                    row["supporting_memory_refs"] = []
                if len(gate_calls) == 2 and row["candidate_ref"] == "ME01:C01":
                    row.update({"parent_relation": "REALIZED",
                                "decision": "reject",
                                "duplicate_with_parent": True,
                                "supporting_memory_refs": ["GM:P001:C001"],
                                "reason": "late duplicate"})
            return result

        realizer_calls = []

        def realizer_call(prompt):
            realizer_calls.append(prompt)
            return {"scenes": [{"dialogue_turns": [{
                "speaker": speakers[index % 2],
                "text": f"scene {scene} turn {index}",
            } for index in range(1, 10)]} for scene in range(1, 4)]}

        gateway = Gateway()
        result = execute_round(
            parent=parent, memorax=gateway,
            verifier_call=lambda prompt: self.fail("verifier must be skipped"),
            planner_call=planner_call,
            planner_repair_call=repair_call,
            semantic_gate_call=gate_call,
            realizer_call=realizer_call,
            adherence_call=lambda prompt: _adherence_reviews(),
            memory_only_fast=True,
            token_counter=lambda text: max(1, len(text) // 4),
            poll_delays_seconds=(0.0,), sleep_fn=lambda _: None)
        self.assertTrue(result["generation_ready"])
        self.assertEqual(len(gate_calls), 2)
        self.assertEqual(len(realizer_calls), 1)
        self.assertEqual(
            [scene["selected_candidate_ref"] for scene in
             result["planner_output"]["scenes"]],
            ["ME01:C01", "ND03:C01", "ND04:C01"])
        self.assertEqual([row["role"] for row in gateway.calls],
                         ["search", "add", "search", "add", "search"])

    def test_memory_guard_receives_contract_roles_not_artifact_labels(self):
        parent=json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers=list(dict.fromkeys(row["speaker"] for row in parent["dialogue"]["turns"]))
        refs=[row["turn_id"] for row in parent["dialogue"]["turns"]]
        events={f"event-{i}":{"display_key":f"Event {i}","content":[f"event fact {i}"]} for i in range(1,4)}
        seen_roles=[]

        class Gateway:
            def __init__(self):
                self.calls=[]
                self.search_count=0
            def search_buckets(self,payload,*,request_id):
                self.search_count += 1
                current = {} if self.search_count == 1 else events
                return {"success":True,"data":{"buckets":{"person":{},"location":{},"event":current,"item":{},"dreaming":{}}}}
            def add(self,payload,*,request_id):
                return {"success":True,"data":{"status":"completed"}}
            def add_status(self,*args,**kwargs):
                raise AssertionError("sync fixture must not poll")

        def planner_call(prompt):
            plan = _split_plan(3, 0, evidence_ref=refs[0])
            for item in plan["selected_continuations"]:
                item["participants"] = speakers[:2]
                item["actor_bindings"] = [{
                    "actor": speakers[0],
                    "binding": f"{speakers[0]} owns the transition."}]
            return plan

        def realizer_call(prompt):
            return {"scenes":[{"dialogue_turns":[{
                "speaker":speakers[j%2],"text":f"scene {i} turn {j}"}
                for j in range(1,10)]} for i in range(1,4)]}

        def semantic_gate_call(prompt):
            return _gate_reviews(event_count=3,parent_ref=refs[0],
                                 candidates_per_event=1)

        execute_round(
            parent=parent, memorax=Gateway(),
            verifier_call=lambda prompt: _verified_tracks(3),
            planner_call=planner_call,
            semantic_gate_call=semantic_gate_call,realizer_call=realizer_call,
            adherence_call=lambda prompt: _adherence_reviews(),
            before_memory_request=lambda role, request_id: seen_roles.append(role),
            token_counter=lambda text:max(1,len(text)//4),
            poll_delays_seconds=(0.0,),sleep_fn=lambda _:None)
        self.assertEqual(seen_roles, [
            "isolation_search", "parent_add", "pre_generation_search",
            "child_add", "post_generation_search"])

    def test_gate_reject_stops_before_realizer_and_child_add(self):
        parent=json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers=list(dict.fromkeys(row["speaker"] for row in parent["dialogue"]["turns"]))
        refs=[row["turn_id"] for row in parent["dialogue"]["turns"]]
        events={f"event-{i}":{"display_key":f"Event {i}","content":[f"fact {i}"]}
                for i in range(1,4)}
        class Gateway:
            def __init__(self): self.calls=[]; self.search_count=0
            def add(self,payload,*,request_id):
                self.calls.append("add"); return {"success":True,"data":{"status":"completed"}}
            def search_buckets(self,payload,*,request_id):
                self.calls.append("search"); self.search_count+=1
                current={} if self.search_count==1 else events
                return {"success":True,"data":{"buckets":{"person":{},"location":{},"event":current,"item":{},"dreaming":{}}}}
            def add_status(self,*args,**kwargs): raise AssertionError("unexpected poll")
        def planner_call(prompt):
            plan = _split_plan(3, 0, evidence_ref=refs[0])
            for item in plan["selected_continuations"]:
                item["participants"] = speakers[:2]
                item["actor_bindings"] = [{
                    "actor": speakers[0],
                    "binding": f"{speakers[0]} owns the transition."}]
            return plan
        def gate_call(prompt):
            return _gate_reviews(event_count=3,reject_event=2,
                                 parent_ref=refs[1], candidates_per_event=1)
        gateway=Gateway()
        with self.assertRaisesRegex(WorkflowError,"semantic_continuation_gate"):
            execute_round(parent=parent,memorax=gateway,
                verifier_call=lambda prompt: _verified_tracks(3),
                planner_call=planner_call,
                semantic_gate_call=gate_call,
                realizer_call=lambda prompt: self.fail("Realizer must not run"),
                adherence_call=lambda prompt: self.fail("Auditor must not run"),
                token_counter=lambda text:max(1,len(text)//4),
                poll_delays_seconds=(0.0,),sleep_fn=lambda _:None)
        self.assertEqual(gateway.calls,["search","add","search"])

    def test_adherence_failure_partially_promotes_only_passing_scenes(self):
        """Failed scenes are quarantined while independent passing scenes write."""
        parent=json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers=list(dict.fromkeys(row["speaker"] for row in parent["dialogue"]["turns"]))
        refs=[row["turn_id"] for row in parent["dialogue"]["turns"]]
        events={f"event-{i}":{"display_key":f"Event {i}","content":[f"fact {i}"]}
                for i in range(1,4)}

        class Gateway:
            def __init__(self): self.calls=[]; self.search_count=0
            def add(self,payload,*,request_id):
                self.calls.append("add")
                return {"success":True,"data":{"status":"completed"}}
            def search_buckets(self,payload,*,request_id):
                self.calls.append("search"); self.search_count += 1
                current={} if self.search_count == 1 else events
                return {"success":True,"data":{"buckets":{
                    "person":{},"location":{},"event":current,
                    "item":{},"dreaming":{}}}}
            def add_status(self,*args,**kwargs):
                raise AssertionError("unexpected poll")

        def planner_call(prompt):
            plan = _split_plan(3, 0, evidence_ref=refs[0])
            for item in plan["selected_continuations"]:
                item["participants"] = speakers[:2]
                item["actor_bindings"] = [{
                    "actor": speakers[0],
                    "binding": f"{speakers[0]} owns the transition."}]
            return plan

        def realizer_call(prompt):
            return {"scenes":[{"dialogue_turns":[{
                "speaker":speakers[j%2],"text":f"scene {i} turn {j}"
                } for j in range(1,10)]} for i in range(1,4)]}

        def failing_adherence(prompt):
            result = _adherence_reviews()
            result["scene_audits"][1]["development_status"] = "partial"
            result["scene_audits"][1]["reason"] = "Selected development is only discussed."
            return result

        gateway=Gateway()
        result = execute_round(parent=parent,memorax=gateway,
            verifier_call=lambda prompt: _verified_tracks(3),
            planner_call=planner_call,
            semantic_gate_call=lambda prompt: _gate_reviews(
                event_count=3,parent_ref=refs[0],candidates_per_event=1),
            realizer_call=realizer_call,
            adherence_call=failing_adherence,
            token_counter=lambda text:max(1,len(text)//4),
            poll_delays_seconds=(0.0,),sleep_fn=lambda _:None)
        self.assertTrue(result["generation_ready"])
        self.assertTrue(result["partial_promotion"])
        self.assertFalse(result["full_round_ready"])
        self.assertEqual(result["promotion_audit"]["promoted_scene_refs"],
                         ["R01:S01", "R01:S03"])
        self.assertEqual([scene["scene_ref"] for scene in
                          result["promotable_child_dialogue"]["scenes"]],
                         ["R01:S01", "R01:S03"])
        # Parent Add and post-write Search still happen, but the Child Add
        # contains only scenes whose whole-round audit passed.
        self.assertEqual(gateway.calls,["search","add","search","add","search"])

    def test_failed_scene_continuation_is_reaudited_before_promotion(self):
        parent = json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers = list(dict.fromkeys(
            row["speaker"] for row in parent["dialogue"]["turns"]))
        refs = [row["turn_id"] for row in parent["dialogue"]["turns"]]
        events = {f"event-{i}": {
            "display_key": f"Event {i}",
            "content": [f"event fact {i}"],
        } for i in range(1, 4)}

        class Gateway:
            def __init__(self):
                self.calls = []
                self.search_count = 0

            def add(self, payload, *, request_id):
                self.calls.append("add")
                return {"success": True, "data": {"status": "completed"}}

            def search_buckets(self, payload, *, request_id):
                self.search_count += 1
                self.calls.append("search")
                current = {} if self.search_count == 1 else events
                if self.search_count >= 3:
                    current = {**events, "child-event": {
                        "display_key": "Child event",
                        "content": ["completed continuation"],
                    }}
                return {"success": True, "data": {"buckets": {
                    "person": {}, "location": {}, "event": current,
                    "item": {}, "dreaming": {},
                }}}

            def add_status(self, *args, **kwargs):
                raise AssertionError("fixture must not poll")

        def planner_call(prompt):
            plan = _split_plan(3, 0, evidence_ref=refs[0])
            for index, item in enumerate(plan["selected_continuations"]):
                item["participants"] = speakers[:2]
                item["goal"] = f"goal {index + 1}"
                item["actor_bindings"] = [{
                    "actor": speakers[0],
                    "binding": f"{speakers[0]} owns transition {index + 1}.",
                }]
            return plan

        def realizer_call(prompt):
            return {"scenes": [{"dialogue_turns": [{
                "speaker": speakers[index % 2],
                "text": f"initial scene {scene} turn {index}",
            } for index in range(1, 10)]} for scene in range(1, 4)]}

        adherence_calls = []

        def adherence_call(prompt):
            adherence_calls.append(prompt)
            result = _adherence_reviews()
            if len(adherence_calls) == 1:
                result["scene_audits"][1]["development_status"] = "partial"
                result["scene_audits"][1]["reason"] = (
                    "The first realization only discussed the transition.")
            return result

        continuation_calls = []

        def scene_continuation_call(prompt):
            continuation_calls.append(prompt)
            return {"continuations": [{"dialogue_turns": [
                {"speaker": speakers[0], "text": "The follow-up is now completed."},
                {"speaker": speakers[1], "text": "The result is confirmed."},
            ]}]}

        reaudit_calls = []

        def adherence_reaudit_call(prompt):
            reaudit_calls.append(prompt)
            return _adherence_reviews()

        gateway = Gateway()
        result = execute_round(
            parent=parent, memorax=gateway,
            verifier_call=lambda prompt: _verified_tracks(3),
            planner_call=planner_call,
            semantic_gate_call=lambda prompt: _gate_reviews(
                event_count=3, parent_ref=refs[0], candidates_per_event=1),
            realizer_call=realizer_call,
            adherence_call=adherence_call,
            scene_continuation_call=scene_continuation_call,
            adherence_reaudit_call=adherence_reaudit_call,
            token_counter=lambda text: max(1, len(text) // 4),
            poll_delays_seconds=(0.0,), sleep_fn=lambda _: None)

        self.assertEqual(len(adherence_calls), 1)
        self.assertEqual(len(continuation_calls), 1)
        self.assertEqual(len(reaudit_calls), 1)
        self.assertEqual(result["scene_continuation_request"][
            "eligible_scene_count"], 1)
        self.assertEqual(
            result["promotion_audit"]["promoted_scene_refs"],
            ["R01:S01", "R01:S02", "R01:S03"])
        self.assertFalse(result["partial_promotion"])
        self.assertTrue(result["full_round_ready"])
        scene_two = result["child_dialogue"]["scenes"][1]
        self.assertEqual(len(scene_two["dialogue_turns"]), 11)
        self.assertEqual(
            scene_two["dialogue_turns"][-1]["text"],
            "The result is confirmed.")
        # Passing scenes are frozen; only the eligible failed scene changes.
        self.assertEqual(
            result["child_dialogue"]["scenes"][0]["dialogue_turns"][-1]["text"],
            "initial scene 1 turn 9")
        self.assertEqual(
            result["child_dialogue"]["scenes"][2]["dialogue_turns"][-1]["text"],
            "initial scene 3 turn 9")
        self.assertEqual(gateway.calls,
                         ["search", "add", "search", "add", "search"])

    def test_failed_continuation_uses_one_unused_passed_replacement(self):
        parent = json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers = list(dict.fromkeys(
            row["speaker"] for row in parent["dialogue"]["turns"]))
        refs = [row["turn_id"] for row in parent["dialogue"]["turns"]]
        events = {f"event-{i}": {
            "display_key": f"Event {i}", "content": [f"fact {i}"],
        } for i in range(1, 4)}

        class Gateway:
            def __init__(self):
                self.calls = []
                self.search_count = 0
            def add(self, payload, *, request_id):
                self.calls.append("add")
                return {"success": True, "data": {"status": "completed"}}
            def search_buckets(self, payload, *, request_id):
                self.calls.append("search")
                self.search_count += 1
                current = {} if self.search_count == 1 else events
                return {"success": True, "data": {"buckets": {
                    "person": {}, "location": {}, "event": current,
                    "item": {}, "dreaming": {},
                }}}
            def add_status(self, *args, **kwargs):
                raise AssertionError("fixture must not poll")

        def planner_call(prompt):
            plan = _split_plan(2, 2, evidence_ref=refs[0])
            for item in [*plan["selected_continuations"],
                         *plan["new_development_candidates"]]:
                item["participants"] = speakers[:2]
                item["new_people"] = []
                item["actor_bindings"] = [{
                    "actor": speakers[0],
                    "binding": f"{speakers[0]} owns this action.",
                }]
            return plan

        def realizer_call(prompt):
            return {"scenes": [{"dialogue_turns": [{
                "speaker": speakers[index % 2],
                "text": f"initial scene {scene} turn {index}",
            } for index in range(1, 10)]} for scene in range(1, 4)]}

        def failing_audit(prompt):
            value = _adherence_reviews()
            value["scene_audits"][2]["durable_outcome_status"] = "partial"
            value["scene_audits"][2]["reason"] = "Outcome remains incomplete."
            return value

        replacement_calls = []
        def replacement_call(prompt):
            replacement_calls.append(prompt)
            return {"scenes": [{"dialogue_turns": [{
                "speaker": speakers[index % 2],
                "text": f"replacement turn {index}",
            } for index in range(1, 10)]}]}

        result = execute_round(
            parent=parent, memorax=Gateway(),
            verifier_call=lambda prompt: _verified_tracks(
                3, turn_ref=refs[0]),
            planner_call=planner_call,
            semantic_gate_call=lambda prompt: _reviews_for_refs([
                "ME01:C01", "ME02:C01", "ND01:C01", "ND02:C01"]),
            realizer_call=realizer_call,
            adherence_call=failing_audit,
            scene_continuation_call=lambda prompt: {
                "continuations": [{"dialogue_turns": [{
                    "speaker": speakers[0], "text": "one bounded continuation",
                }]}]},
            adherence_reaudit_call=failing_audit,
            scene_replacement_call=replacement_call,
            replacement_adherence_call=lambda prompt: _adherence_reviews(),
            token_counter=lambda text: max(1, len(text) // 4),
            poll_delays_seconds=(0.0,), sleep_fn=lambda _: None)

        self.assertEqual(len(replacement_calls), 1)
        self.assertTrue(result["full_round_ready"])
        self.assertFalse(result["partial_promotion"])
        self.assertEqual(
            result["scene_replacement_request"]["selected_replacement"][
                "candidate_ref"], "ND02:C01")
        self.assertEqual(
            result["child_dialogue"]["scenes"][2]["memory_event_id"], "ND02")
        self.assertEqual(
            result["child_dialogue"]["scenes"][2]["dialogue_turns"][0]["text"],
            "replacement turn 1")
        self.assertEqual(
            result["child_dialogue"]["scenes"][0]["dialogue_turns"][-1]["text"],
            "initial scene 1 turn 9")


class ExpansionProfileAndBatchTest(unittest.TestCase):
    @staticmethod
    def _candidate_rows(count):
        return build_event_candidates({"events": [
            {"canonical_key": f"e{i}", "display_key": f"E{i}",
             "content": [f"f{i}"]} for i in range(1, count + 1)]})

    def test_legacy_profile_keeps_three_scene_contract(self):
        self.assertEqual(LEGACY_PROFILE.total_scene_limit, 3)
        self.assertEqual(LEGACY_PROFILE.planner_max_tokens, 4096)
        self.assertTrue(LEGACY_PROFILE.require_full_scene_quota)

    def test_profile_route_defaults_do_not_change_legacy_behavior(self):
        self.assertEqual(
            _resolve_profile_route(
                profile=LEGACY_PROFILE, explicit_profile=False,
                memory_only_fast=None, event_bucket_only=None),
            (False, True))
        self.assertEqual(
            _resolve_profile_route(
                profile=EXPANDED_PROFILE, explicit_profile=True,
                memory_only_fast=None, event_bucket_only=None),
            (True, True))
        self.assertEqual(
            _resolve_profile_route(
                profile=EXPANDED_PROFILE, explicit_profile=True,
                memory_only_fast=False, event_bucket_only=False),
            (True, True))
        self.assertEqual(
            _resolve_profile_route(
                profile=LEGACY_PROFILE, explicit_profile=False,
                memory_only_fast=True, event_bucket_only=False),
            (True, False))

    def test_expanded_profile_keeps_independent_new_development_slots(self):
        self.assertEqual(EXPANDED_PROFILE.total_scene_limit, 6)
        self.assertEqual(EXPANDED_PROFILE.new_development_slots, 6)
        self.assertEqual(EXPANDED_PROFILE.planner_max_tokens, 8192)
        self.assertEqual(EXPANDED_PROFILE.realizer_batch_size, 2)
        self.assertEqual(EXPANDED_PROFILE.new_development_slots, 6)
        self.assertTrue(EXPANDED_PROFILE.sequential_batch_promotion)
        self.assertEqual(EXPANDED_PROFILE.max_plan_variants, 2)

    def test_current_expanded_profile_uses_one_plan_with_same_scene_capacity(self):
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.name, "expanded_single_plan_v1")
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.max_plan_variants, 1)
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.max_continuations, 3)
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.max_new_development_candidates, 8)
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.total_scene_limit, 6)
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.realizer_batch_size, 2)
        self.assertTrue(EXPANDED_SINGLE_PLAN_PROFILE.sequential_batch_promotion)

    def test_legacy_single_plan_and_expanded_variant_selection(self):
        legacy = {"round_goal": "x", "selected_continuations": [],
                  "new_development_candidates": []}
        selected = select_planner_variant(legacy)
        self.assertTrue(selected["valid"])
        self.assertEqual(selected["selection_mode"], "single_plan_legacy")
        variants = {"plan_variants": [
            {"round_goal": "first"}, {"round_goal": "second"}]}
        selected = select_planner_variant(variants, max_plan_variants=2)
        self.assertTrue(selected["valid"])
        self.assertEqual(selected["value"]["round_goal"], "first")
        self.assertEqual(selected["variant_count"], 2)
        rejected = select_planner_variant(variants, max_plan_variants=1)
        self.assertFalse(rejected["valid"])

    def test_variant_speaker_materialization_keeps_one_shared_contract(self):
        variant = {
            "round_goal": "new scenes",
            "selected_continuations": [],
            "new_development_candidates": [],
        }
        result = materialize_planner_speaker_refs(
            {"plan_variants": [variant, dict(variant)]}, parent=_parent())
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(len(result["value"]["plan_variants"]), 2)
        self.assertNotIn("participants", result["value"]["plan_variants"][0])

    def test_expanded_planner_prompt_is_explicit_without_changing_legacy_shape(self):
        parent = _parent()
        legacy = build_planner_prompt(
            parent=parent, memory_overview={}, candidates=[],
            event_bucket_only=True, memory_only_fast=True,
            include_authoritative_transcript=True)
        expanded = build_planner_prompt(
            parent=parent, memory_overview={}, candidates=[],
            event_bucket_only=True, memory_only_fast=True,
            include_authoritative_transcript=True, max_plan_variants=2)
        self.assertNotIn("plan_variants", legacy["prompt"])
        self.assertIn("plan_variants", expanded["prompt"])
        self.assertIn("FINAL OUTPUT SHAPE OVERRIDE", expanded["prompt"])

    def test_plan_batches_preserve_scene_identity_and_order(self):
        plan = {"round_goal": "x", "scenes": [
            {"memory_event_id": f"S{i}"} for i in range(1, 7)]}
        batches = build_realizer_plan_batches(plan, batch_size=2)
        self.assertEqual(len(batches), 3)
        self.assertEqual(
            [[row["memory_event_id"] for row in batch["scenes"]]
             for batch in batches],
            [["S1", "S2"], ["S3", "S4"], ["S5", "S6"]])
        self.assertEqual(batches[0]["prior_promoted_scene_refs"], [])
        self.assertEqual(batches[1]["prior_promoted_scene_refs"], [])
        self.assertEqual(batches[2]["prior_promoted_scene_refs"], [])
        self.assertEqual(batches[1]["scheduled_predecessor_scene_refs"],
                         ["R01:S01", "R01:S02"])
        self.assertEqual(batches[2]["scheduled_predecessor_scene_refs"],
                         ["R01:S01", "R01:S02", "R01:S03", "R01:S04"])
        self.assertEqual(plan["scenes"][0]["memory_event_id"], "S1")

    def test_expanded_three_continuations_can_add_new_scenes(self):
        plan = _split_plan(3, 3)
        result = validate_planner_output(
            plan, candidates=self._candidate_rows(3), parent=_parent(),
            max_new_candidates=8, max_continuations=3, scene_quota=6,
            new_development_slots=3, require_full_scene_quota=False)
        self.assertTrue(result["valid"], result["errors"])
        compiled = compile_planner_output(
            plan, candidates=self._candidate_rows(3),
            scene_quota=6, new_development_slots=3,
            selected_new_candidate_indices=result["selected_new_candidate_indices"],
            valid_new_candidate_indices=result["valid_new_candidate_indices"])
        self.assertEqual(len(compiled["scenes"]), 6)
        materialized = materialize_planner_output(
            compiled, passed_candidate_refs=[
                f"ME{i:02d}:C01" for i in range(1, 4)] +
                [f"ND{i:02d}:C01" for i in range(1, 4)], scene_count=6)
        self.assertEqual(len(materialized["scenes"]), 6)
        isolation = validate_realizer_plan_isolation(
            raw_plan=compiled, materialized_plan=materialized,
            passed_candidate_refs=[
                f"ME{i:02d}:C01" for i in range(1, 4)] +
                [f"ND{i:02d}:C01" for i in range(1, 4)], scene_count=6)
        self.assertTrue(isolation["valid"], isolation["errors"])

    def test_expanded_zero_continuations_can_materialize_more_than_three(self):
        plan = _split_plan(0, 6)
        result = validate_planner_output(
            plan, candidates=[], parent=_parent(), max_new_candidates=8,
            max_continuations=3, scene_quota=6, new_development_slots=6,
            require_full_scene_quota=False)
        self.assertTrue(result["valid"], result["errors"])
        compiled = compile_planner_output(
            plan, candidates=[], scene_quota=6, new_development_slots=6,
            selected_new_candidate_indices=result["selected_new_candidate_indices"],
            valid_new_candidate_indices=result["valid_new_candidate_indices"])
        self.assertEqual(len(compiled["scenes"]), 6)
        materialized = materialize_planner_output(
            compiled, passed_candidate_refs=[f"ND{i:02d}:C01" for i in range(1, 7)],
            scene_count=6)
        self.assertEqual([row["memory_event_id"] for row in materialized["scenes"]],
                         [f"ND{i:02d}" for i in range(1, 7)])
        isolation = validate_realizer_plan_isolation(
            raw_plan=compiled, materialized_plan=materialized,
            passed_candidate_refs=[f"ND{i:02d}:C01" for i in range(1, 7)],
            scene_count=6)
        self.assertTrue(isolation["valid"], isolation["errors"])

    def test_expanded_profile_allows_partial_scene_set_without_fabrication(self):
        plan = _split_plan(0, 2)
        result = validate_planner_output(
            plan, candidates=[], parent=_parent(), max_new_candidates=8,
            max_continuations=3, scene_quota=6, new_development_slots=6,
            require_full_scene_quota=False)
        self.assertTrue(result["valid"], result["errors"])
        compiled = compile_planner_output(
            plan, candidates=[], scene_quota=6, new_development_slots=6,
            selected_new_candidate_indices=result["selected_new_candidate_indices"],
            valid_new_candidate_indices=result["valid_new_candidate_indices"])
        materialized = materialize_planner_output(
            compiled, passed_candidate_refs=["ND01:C01", "ND02:C01"],
            scene_count=2)
        self.assertEqual(len(materialized["scenes"]), 2)
        self.assertEqual(
            [row["memory_event_id"] for row in materialized["scenes"]],
            ["ND01", "ND02"])

    def test_outline_selection_can_use_legal_partial_expanded_set(self):
        audit = {
            "conflict_edges": [["A", "B"], ["A", "C"]],
        }
        selected = select_ranked_legal_candidate_set(
            ["A", "B", "C"], outline_audit=audit, required_count=3,
            allow_partial=True)
        self.assertTrue(selected["valid"])
        self.assertEqual(selected["selected_candidate_refs"], ["B", "C"])


if __name__ == "__main__": unittest.main()
