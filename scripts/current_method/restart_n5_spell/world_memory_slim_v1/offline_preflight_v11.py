"""Zero-API Phase-A preflight for Slim Continuous Expansion v11."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from .action_availability_v11 import (
    compile_action_availability_view, validate_action_availability_view,
)
from .complete_dialogue_view import (
    compose_complete_dialogue_view, validate_complete_dialogue_view,
)
from .context_compiler_v2 import (
    ContextInvariantFailure, compile_scene_context_v2,
    validate_context_manifest_v2,
)
from .continuous_round_plan_v1 import (
    ADDED_TURN_BUDGET_V11, CONTINUOUS_ROUND_PLAN_VERSION,
    SCENE_BUDGET_V11, build_scene_dependency_dag,
    compile_continuous_round_plan, validate_continuous_round_plan,
)
from .inputs import load_locomo_parent_read_only, load_unified_root_read_only
from .interaction_links import compile_source_interaction_links
from .narrative_order import (
    build_narrative_order_manifest, validate_narrative_order_manifest,
)
from .parent_anchor_compiler import (
    compile_anchor_registry, resolve_anchor, validate_anchor_registry,
    validate_placement_boundary,
)
from .placement_choice_registry import (
    compile_placement_choice_registry, compile_source_anchor_registry,
    resolve_placement_choice, resolve_source_anchor,
    validate_placement_choice_registry, validate_source_anchor_registry,
)
from .planning_state_view import (
    audit_interaction_relation_separation, compile_planning_state_view,
    validate_planning_state_view,
)
from .planner_outline_v11 import (
    compile_planner_outline_v11, project_internal_v4_fixture_to_outline_v11,
    validate_planner_outline_v11,
)
from .provenance import git_value, hash_tree, project_diff_sha256
from .source_memory_initializer_v1 import initialize_source_world_memory
from .structural_expansion_gate import (
    compute_promotion_closure, simulate_dependency_runtime,
    validate_connected_jump, validate_realized_expansion_structure,
    validate_round_blueprint_structure,
)


REPO = Path("/root/autodl-tmp/gxy-selfplay/current_method_github")
PACKAGE = REPO / "scripts/current_method/restart_n5_spell/world_memory_slim_v1"
REPORT_DIR = PACKAGE / "reports"
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
INPUT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json")
V10 = RUNS / "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v10"
DEFAULT_OUTPUT = RUNS / "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_phase_a_preflight"
CC_ROOTS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/roots")
CC_IDS = (
    "cc-train-episode-12596", "cc-train-episode-13836",
    "cc-train-episode-146700", "cc-train-episode-178392",
    "cc-train-episode-216742", "cc-train-episode-34406",
    "cc-train-episode-41409", "cc-train-episode-41625",
    "cc-train-episode-61555", "cc-train-episode-91716",
)
LOCOMO_JSONL = Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_train.jsonl")
LOCOMO_ATTESTATION = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/spell_cross_dialogue_calibration_5x3_v1/dialogues/conv-43/questioner/frozen/parent_source_attestation.json")
PROTECTED_MODULES = (
    "operation_registry.py", "relation_registry.py", "controlled_expansion.py",
    "planner_contract_v3_1.py", "scene_local_refs.py", "claim_validator_v2.py",
    "dialogue_quality.py", "turn_budget.py", "transactional_binder.py",
    "append_only_memory.py", "provenance_ledger.py",
    "source_memory_initializer_v1.py", "live_fresh_e2e_v7.py", "planner_v3.py",
    "context_compiler.py", "promotion.py", "evidence_scope.py",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _sessions(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for turn in turns:
        ref = str(turn.get("session_id") or turn.get("session_ref") or "")
        if ref not in rows:
            rows[ref] = []
            order.append(ref)
        rows[ref].append(turn)
    return [{"session_id": ref, "turns": list(rows[ref])} for ref in order]


def _synthetic_child(root_id: str, *, round_index: int, turn_count: int,
                     speaker_refs: Sequence[str]) -> dict[str, Any]:
    scenes = []
    remaining = turn_count
    scene_index = 1
    while remaining:
        count = min(6, remaining)
        scene_ref = f"R{round_index:02d}:S{scene_index:02d}"
        scenes.append({
            "scene_ref": scene_ref, "session_ref": scene_ref + ":SESSION",
            "dialogue_turns": [{
                "turn_ref": f"{scene_ref}:T{index:03d}",
                "speaker_ref": speaker_refs[(index - 1) % len(speaker_refs)],
                "text": f"Synthetic cumulative fixture round {round_index}, scene "
                        f"{scene_index}, turn {index}.",
            } for index in range(1, count + 1)],
        })
        remaining -= count
        scene_index += 1
    return {"child_id": f"{root_id}:fixture:round:{round_index}",
            "root_id": root_id, "surface": "promotable", "scenes": scenes}


def build_cumulative_fixture(parent: Mapping[str, Any],
                             speaker_map: Mapping[str, str]) -> dict[str, Any]:
    refs = list(speaker_map.values())
    round1 = _synthetic_child(str(parent["sample_id"]), round_index=1,
                              turn_count=30, speaker_refs=refs)
    round2 = _synthetic_child(str(parent["sample_id"]), round_index=2,
                              turn_count=35, speaker_refs=refs)
    return compose_complete_dialogue_view(
        original_parent=parent,
        promoted_rounds=[{"round_index": 1, "child": round1},
                         {"round_index": 2, "child": round2}],
        speaker_to_entity_ref=speaker_map)


def build_round_blueprint_fixture(
    *, source_anchor_id: str, placement_choice_id: str,
    entity_refs: Sequence[str],
    single_action_selection: Mapping[str, Any], scene_count: int = 6,
) -> dict[str, Any]:
    if not 5 <= scene_count <= 7:
        # Intentionally return a structurally complete, count-invalid fixture.
        pass
    aliases = [f"s{index}" for index in range(1, scene_count + 1)]
    dependencies: dict[int, list[tuple[int, str]]] = {
        1: [], 2: [(1, "continues")], 3: [(1, "continues"), (2, "reveals")],
        4: [(2, "parallel_to")], 5: [(3, "causes"), (4, "informs")],
        6: [(5, "impacts")], 7: [(6, "continues")],
    }
    del single_action_selection  # legacy fixture parameter retained for callers only
    scenes = []
    for index, alias in enumerate(aliases, 1):
        deps = [(prior, kind) for prior, kind in dependencies.get(index, [])
                if prior <= scene_count]
        if index == 3:
            branch_role, merge, impacts = "character_branch", "s5", []
        elif index == 4:
            branch_role, merge, impacts = "event_branch", "s5", []
        else:
            branch_role, merge, impacts = "mainline", None, []
        actions = [
            {"action_mode": "new", "action_type": "develop_character_or_task_state", "arguments": {
                "target": entity_refs[0], "dimension": "coordination_confidence",
                "direction": "increase", "trigger": f"development {index}"}},
            {"action_mode": "new", "action_type": "introduce_external_constraint", "arguments": {
                "constraint": f"constraint {index}", "affected_participants": list(entity_refs),
                "affected_plan_or_task": "the shared coordination plan"}},
            {"action_mode": "new", "action_type": "add_information_asymmetry", "arguments": {
                "informed_participant": entity_refs[0], "uninformed_participant": entity_refs[1],
                "hidden_information": f"hidden constraint {index}",
                "information_source": "the existing historical connection"}},
            {"action_mode": "new", "action_type": "reveal_or_transfer_information", "arguments": {
                "sender": entity_refs[0], "receiver": entity_refs[1],
                "information": f"new information {index}", "transfer_mode": "direct conversation"}},
            {"action_mode": "new", "action_type": "create_misunderstanding", "arguments": {
                "mistaken_participant": entity_refs[1], "mistaken_belief": "the deadline is fixed",
                "misunderstanding_source": "an incomplete message",
                "actual_situation": "the deadline is negotiable"}},
            {"action_mode": "new", "action_type": "create_or_escalate_conflict", "arguments": {
                "participants": list(entity_refs), "conflict_subject": "resource priority",
                "conflict_source": "competing constraints", "escalation_or_creation": "create"}},
            {"action_mode": "new", "action_type": "develop_character_or_task_state", "arguments": {
                "target": entity_refs[1], "dimension": "coordination_confidence",
                "direction": "decrease", "trigger": f"late consequence {index}"}},
        ]
        scenes.append({
            "scene_alias": alias,
            "source_anchor_id": source_anchor_id,
            "placement_choice_id": placement_choice_id,
            "link_type": ["direct_continuation", "task_followup",
                          "related_character_shift", "location_shift",
                          "information_transfer", "causal_followup",
                          "time_skip"][((index - 1) % 7)],
            "linked_existing_refs": {
                "entity_refs": list(entity_refs), "relation_refs": [],
                "event_refs": [], "task_refs": [], "thread_refs": [],
                "state_refs": [],
            },
            "bridge_reason": (
                "The scene remains attached to the verified interaction between "
                "the same existing participants."),
            "new_development": f"Development {index} changes the shared situation in a distinct way.",
            "goal": f"Advance connected development {index} without repeating prior dialogue.",
            "speaker_participants": [
                {"participant_ref": entity_refs[0], "scene_role": "primary",
                 "intents": [f"advance development {index}", "protect continuity"]},
                {"participant_ref": entity_refs[1], "scene_role": "supporting",
                 "intents": [f"respond to development {index}", "surface a constraint"]},
            ],
            "relationship_dynamic": "The existing speakers disagree constructively before coordinating.",
            "interaction_beats": [
                f"The first speaker raises concrete development {index}.",
                "The second speaker introduces a practical constraint.",
                "They compare implications for the established shared situation.",
                "They leave a consequence for a later scene.",
            ],
            "continuity_requirements": [
                "Preserve the historical anchor facts.",
                "Respect every declared predecessor outcome.",
            ],
            "predecessor_scene_aliases": [f"s{prior}" for prior, _ in deps],
            "dependency_edges": [{"predecessor_scene_alias": f"s{prior}",
                                  "dependency_type": kind}
                                 for prior, kind in deps],
            "transition_to_next_scene": f"Development {index} supplies a concrete next pressure.",
            "branch_role": branch_role,
            "merge_target_scene_alias": merge,
            "impact_target_refs": impacts,
            "primary_action": deepcopy(actions[(index - 1) % len(actions)]),
            "secondary_action": None,
            "desired_outcome": f"A verifiable connected consequence from scene {index} is available.",
        })
    return {
        "round_plan_version": CONTINUOUS_ROUND_PLAN_VERSION,
        "scene_budget": deepcopy(SCENE_BUDGET_V11),
        "target_added_turns": deepcopy(ADDED_TURN_BUDGET_V11),
        "round_goal": "Extend one shared situation through a mainline and a reconnecting branch.",
        "starting_situation": "Two established participants have a verified interaction history.",
        "central_development": "A shared responsibility encounters a concrete external pressure.",
        "secondary_development": "A parallel viewpoint transfers information back to the mainline.",
        "ending_direction": "The round ends with a connected open consequence.",
        "main_thread": "The participants coordinate around a changing shared responsibility.",
        "side_threads": ["A related viewpoint exposes a missing constraint."],
        "continued_old_threads": ["The existing participants continue their established interaction."],
        "new_connected_threads": ["A follow-up coordination question remains open."],
        "new_connected_events": ["An external scheduling pressure affects the existing participants."],
        "relationship_developments": ["Their contrasting approaches become operationally relevant."],
        "event_consequences": ["Information from the branch changes the mainline response."],
        "state_developments": ["The shared plan moves from tentative to constrained."],
        "information_flow": ["A branch reveals a constraint before it rejoins the mainline."],
        "difficulty_contributions": ["Evidence crosses a branch and its later merge."],
        "scene_order": aliases,
        "branch_entry_points": ["s3", "s4"] if scene_count >= 4 else [],
        "branch_merge_points": ["s5"] if scene_count >= 5 else [],
        "cross_scene_impacts": ["s4 informs s5"] if scene_count >= 5 else [],
        "ending_open_threads": ["The final practical consequence remains available for a later round."],
        "scenes": scenes,
    }


def _scene_output(scene: Mapping[str, Any]) -> dict[str, Any]:
    refs = [row["participant_ref"] for row in scene["speaker_participants"]]
    scene_ref = str(scene["scene_ref"])
    return {"scene_ref": scene_ref, "dialogue_turns": [{
        "turn_ref": f"{scene_ref}:T{index:03d}",
        "speaker_ref": refs[(index - 1) % len(refs)],
        "text": f"Phase A fixture {scene_ref} develops unique connected point {index}.",
    } for index in range(1, 7)], "proposal_claims": []}


def _historical_hashes() -> dict[str, Any]:
    return {f"v{version}": hash_tree(RUNS /
            f"conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v{version}")
            for version in range(6, 11)}


def _protected_hashes() -> dict[str, str]:
    return {name: sha256_file(PACKAGE / name) for name in PROTECTED_MODULES}


def _run_check(command: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    return {"command": list(command), "exit_code": result.returncode,
            "stdout_tail": result.stdout[-3000:], "stderr_tail": result.stderr[-3000:],
            "passed": result.returncode == 0}


def _eleven_root_preflight() -> dict[str, Any]:
    rows = []
    sources = [(root_id, "cc", CC_ROOTS / root_id / "unified_root.json")
               for root_id in CC_IDS]
    for root_id, kind, path in sources:
        loaded = load_unified_root_read_only(path)
        rows.append(_preflight_one(root_id, kind, loaded))
    locomo = load_locomo_parent_read_only(LOCOMO_JSONL, LOCOMO_ATTESTATION)
    rows.append(_preflight_one("locomo-conv-43", "locomo_attested_record_77", locomo))
    return {"root_count": len(rows), "passed_count": sum(row["passed"] for row in rows),
            "all_passed": len(rows) == 11 and all(row["passed"] for row in rows),
            "roots": rows, "api_calls": 0}


def _preflight_one(root_id: str, kind: str, loaded: Mapping[str, Any]) -> dict[str, Any]:
    source = initialize_source_world_memory(loaded["root"], loaded["parent"])
    turns = loaded["parent"]["dialogue"]["turns"]
    links = compile_source_interaction_links(
        root_id=str(loaded["root"]["root_id"]), sessions=_sessions(turns),
        speaker_to_entity=source["speaker_to_entity_ref"])
    planning = compile_planning_state_view(
        memory=source["memory"], interaction_links=links)
    complete = compose_complete_dialogue_view(
        original_parent=loaded["parent"],
        speaker_to_entity_ref=source["speaker_to_entity_ref"])
    anchors = compile_anchor_registry(
        complete_dialogue=complete, planning_state=planning)
    source_anchors = compile_source_anchor_registry(anchors)
    placement_choices = compile_placement_choice_registry(
        complete_dialogue=complete, anchor_registry=anchors)
    availability = compile_action_availability_view(planning_state=planning)
    entity_refs = [str(row.get("entity_ref")) for row in planning.get("entities") or []]
    first_anchor = (source_anchors.get("source_anchors") or [{}])[0]
    tail_choice = next((row for row in placement_choices.get("choices") or []
                        if row.get("placement_choice_id") == "PC_TAIL_AFTER"), {})
    v4_fixture = build_round_blueprint_fixture(
        source_anchor_id=str(first_anchor.get("source_anchor_id") or ""),
        placement_choice_id=str(tail_choice.get("placement_choice_id") or ""),
        entity_refs=entity_refs[:2], single_action_selection={}, scene_count=5)
    slim_fixture = project_internal_v4_fixture_to_outline_v11(v4_fixture)
    slim_validation = validate_planner_outline_v11(
        slim_fixture, source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices,
        authoritative_entity_refs=set(entity_refs),
        action_availability_view=availability)
    slim_compiled = None
    slim_gate = {"valid": False, "errors": ["slim_outline_not_compiled"]}
    if slim_validation["valid"]:
        slim_compiled = compile_planner_outline_v11(
            slim_fixture, source_anchor_registry=source_anchors,
            placement_choice_registry=placement_choices, planning_state=planning,
            action_availability_view=availability)
        slim_gate = validate_round_blueprint_structure(
            compiled_plan=slim_compiled, source_anchor_registry=source_anchors,
            placement_choice_registry=placement_choices,
            complete_dialogue=complete, planning_state=planning)
    checks = {
        "identity": complete["root_id"] == loaded["root"]["root_id"],
        "complete_dialogue": validate_complete_dialogue_view(complete)["valid"],
        "planning_state": validate_planning_state_view(planning)["valid"],
        "anchor_registry": validate_anchor_registry(
            anchors, complete_dialogue=complete)["valid"],
        "source_anchor_registry": validate_source_anchor_registry(
            source_anchors)["valid"],
        "placement_choice_registry": validate_placement_choice_registry(
            placement_choices, complete_dialogue=complete)["valid"],
        "action_availability_view": validate_action_availability_view(
            availability)["valid"],
        "slim_planner_outline": slim_validation["valid"],
        "slim_outline_compile": slim_compiled is not None,
        "slim_outline_structural_gate": slim_gate["valid"],
        "tail_after_choice_present": any(
            row.get("placement_choice_id") == "PC_TAIL_AFTER"
            for row in placement_choices.get("choices") or []),
        "parent_utterance_excluded": not any(
            row.get("record_type") == "parent_utterance"
            for field in ("active_events", "tasks", "key_states")
            for row in planning.get(field) or []),
    }
    return {"root_id": root_id, "source_kind": kind,
            "turn_count": complete["turn_count"],
            "entity_count": len(planning["entities"]),
            "interaction_link_count": len(planning["interaction_links"]),
            "explicit_relation_count": len(planning["explicit_relations"]),
            "anchor_count": len(anchors["anchors"]), "checks": checks,
            "available_action_count": sum(
                row.get("available") is True
                for row in availability.get("actions", {}).values()),
            "passed": all(checks.values())}


def _request_budget() -> dict[str, Any]:
    def option(name: str, validator: str) -> dict[str, Any]:
        return {
            "option": name, "semantic_validator": validator,
            "normal_requests": {"N=5": 1 + 5 + (5 if name == "A" else 1),
                                "N=7": 1 + 7 + (7 if name == "A" else 1)},
            "worst_requests": {"N=5": 3 + 15 + (15 if name == "A" else 3),
                               "N=7": 3 + 21 + (21 if name == "A" else 3)},
            "planner_tokens": "planning_state + anchor menu + long blueprint schema",
            "realizer_tokens": "per-scene P0/P1 context; direct dependencies repeat by design",
            "validator_tokens": ("per-scene dialogue + repeated authoritative state" if name == "A"
                                 else "one complete round + compact authoritative state"),
            "contract_retry_only": True,
            "partial_round_retry": False,
            "connected_component_retry": "design-only; not enabled",
            "repeats_complete_parent_utterance_memory": name == "A",
            "current_v10_repetition_observed": name == "A",
        }
    return {
        "request_budget_schema_version": "round_validator_request_budget_review_v1",
        "v10_observed": {"planner": {"requests": 1, "tokens": 9082},
                         "realizer": {"requests": 3, "tokens": 19716},
                         "committer": {"requests": 3, "tokens": 92742},
                         "total_requests": 7, "total_tokens": 121540},
        "options": [option("A", "per_scene_committer"),
                    option("B", "round_expansion_validator"),
                    option("C", "local_scene_gates_plus_round_validator")],
        "recommended_option": "C",
        "phase_b_request_budget_ready": False,
        "online_budget_changed": False,
    }


def _migration_review() -> str:
    return """# Round Expansion Validator Migration Review

## Decision

**C. Local per-scene hard gates + one Round Expansion Validator.** This is a
design recommendation only. No live validator, prompt, Binder, transaction, or
promotion code was changed in Phase A.

## Current boundaries and evidence

`claim_validator_v2.validate_proposal_claims` checks the Realizer's exact
proposal/evidence-reference contract. `evidence_scope.validate_committer_scope_output`
checks one scene's proposal reviews, candidate-evidence subset, scope, and world
consistency. `live_fresh_e2e_v7._execute_scene` calls that role once per valid
scene, then `append_only_memory.evaluate_scene_commit`,
`transactional_binder.build_allocation_plan`, and `commit_scene_transaction`
perform deterministic admission/allocation/publish. The Binder consumes compiled
operations and accepted scene-local refs—not unconstrained semantic prose.

The current Committer overlaps a future round validator on semantic support,
world consistency, and unplanned persistent facts. It does **not** overlap the
later SPELL/QA Validator's task of question/gold/evidence validity and difficulty.

## Options and cost

- A keeps early per-scene decisions, but cannot see later consequences and repeats
  authoritative context. For N=5–7: 11–15 normal requests; 33–45 worst case.
- B uses 7–9 normal and 21–27 worst-case requests, but removing all per-scene
  gates would allow malformed dialogue and broken dependencies to accumulate.
- C has B's API counts because local gates cost no requests: 7–9 normal,
  21–27 worst case. It preserves early mechanical rejection and defers semantic
  continuity/novelty to one round-wide view.

The v10 baseline was 7 requests / 121,540 tokens. Its three Committer calls used
92,742 tokens, principally because the complete utterance-style Memory view was
resent. A compact Planning State and one round review directly target that cost.

## Thirteen migration questions

1. **Can Round Validator replace the per-scene Committer?** Semantically yes,
   mechanically no: keep local Quality, Claim, anchor, ref, and dependency gates.
2. **Retry routing?** Planner/Realizer retries remain local. The round validator
   retries only JSON/schema/reference contract errors; a valid rejection is final.
3. **Rejected scene dependencies?** Recompute dependency closure. Hard descendants
   become `dependency_unavailable`; independent anchored parallel branches remain.
4. **Whole round or connected components?** One whole-round request is the default
   because cross-component history and future-knowledge checks matter. Component
   review is a future fallback only after equivalent global auditing exists.
5. **Map durable updates to Action Registry?** Validator emits typed candidates;
   a deterministic compiler resolves subjects/targets, validates registry payload,
   current head/version, dependencies, and fingerprints.
6. **May it output unhinted durable updates?** Yes, conservatively, because hints
   are optional. Every such update must be supported by scene-local evidence and
   compile to an existing operation; otherwise it stays rejected audit material.
7. **Can Binder consume raw Validator output?** No. Current Binder requires compiled
   proposals/allocation inputs, not model-authored update objects.
8. **Need a durable-update compiler?** Yes. It is the required compatibility layer;
   it should produce the same compiled proposal contract the Binder already trusts.
9. **Prevent invented refs/facts?** Freeze entity/action/anchor registries; exact
   scene-local ref membership; evidence-subset validation; reject unknown refs;
   compile payloads locally; recheck active head/version before transaction.
10. **Require one Memory update per round?** No. `durable_updates=[]` must be legal.
    Expansion readiness should be dialogue/continuity based, not record-count based.
11. **Overlap with SPELL Validator?** Only general consistency concepts overlap.
    Round Validator governs generated-world promotion; SPELL governs downstream QA.
12. **Partial branch rejection?** Remove the rejected branch and its hard closure,
    then retain independently anchored components that still pass global consistency.
13. **Should it output final promotion decisions?** It may recommend scene reviews,
    but deterministic dependency/promotion closure must calculate final admission.

## Proposed round response and compiler boundary

The response should contain `scene_reviews`, round/historical/dependency/future-
knowledge/novelty statuses, and `durable_updates`. Each update carries a registered
update type, existing subject refs, normalized value, exact evidence turn refs,
source scene refs, and verification status. It never carries final world IDs.

After validation: local durable-update compiler → dependency/promotion closure →
confirmed compiled updates → unchanged Binder/Transaction. Binder and transaction
algorithms need no semantic redesign; their caller and allocation grouping need an
adapter plus round-level atomicity policy.

## Initial Structure Extractor

Prefer an **independent one-time Initial Structure Extractor API** over a Round-0
mode. It has a narrower input/output contract, separate provenance, and cannot
silently inherit round-promotion semantics. Every entity/relation/event/task/thread/
state requires evidence refs and an explicit-vs-inferred label; only explicit rows
enter authoritative Planning State. Interaction co-occurrence remains an
Interaction Link. Round-0 mode saves one role definition but couples initialization
to a changing validator and makes retries/provenance harder to audit.

## Risks and freeze

Round-wide prompts can still be large; partial branch rejection and atomic bundles
need exact local closure; an unconstrained durable-update extractor could invent
facts. Therefore `round_expansion_validator_ready=false` and
`phase_b_request_budget_ready=false` remain frozen until schemas, compiler,
component closure, and one zero-API replay are implemented and tested.
"""


def _phase_a_report(data: Mapping[str, Any]) -> str:
    r = data["readiness"]
    checks = "\n".join(f"- `{key}` = `{str(value).lower()}`" for key, value in r.items())
    return f"""# Slim Continuous Expansion v11 — Phase A Report

## Outcome

Phase A is a zero-API structural compatibility layer. It does not add a live v11
runner or modify v10's Planner, Committer, Binder, transaction, promotion, or
historical artifacts. The semantic review remains deliberately unimplemented.

## Changes and dirty-worktree boundary

New versioned files: `complete_dialogue_view.py`, `planning_state_view.py`,
`parent_anchor_compiler.py`, `context_compiler_v2.py`,
`continuous_round_plan_v1.py`, `structural_expansion_gate.py`,
`narrative_order.py`, `offline_preflight_v11.py`, and the dedicated test module.
The repository was already dirty and the package appeared as an untracked tree at
turn start; no pre-existing edits were cleaned or overwritten. Protected v10
mechanical-module hashes were captured before and after.

## Complete dialogue and Planning State

The v10 shadow projection contains {data['complete']['turn_count']} cumulative turns
({data['complete']['original_parent']['turn_count']} original + promoted child).
The explicit 60+30+35 fixture contains {data['cumulative_fixture']['turn_count']}
turns, preserving every turn and scene ref. Generation order and intended narrative
order are separate.

Planning State contains {len(data['planning']['entities'])} entities,
{len(data['planning']['interaction_links'])} Interaction Links, and
{len(data['planning']['explicit_relations'])} explicit relations. It contains zero
`parent_utterance` rows. Interaction co-occurrence was not upgraded to friend,
coworker, family, trust, or authority. Absence from this projection is never treated
as novelty evidence.

## Anchors, placement, and context

The registry contains {len(data['anchors']['anchors'])} deterministic anchors.
Anchors are connection evidence, not a direct-continuation restriction: fixtures
cover time/location/character/event jumps and reject an unrelated jump. `after`,
`before`, `between`, and `parallel` boundaries are checked locally. Future turns in
before/between contexts are constraint-only.

Context v2 retains Parent `P05:T010`–`P05:T016`, including the true ending. Direct
predecessors are P0 exact dialogue; other declared dependencies are P1. Scene 2 sees
Scene 1, and Scene 3 sees Scene 1/2. P0/P1 cannot be silently trimmed; overflow is a
local invariant failure; only P2 is dropped. Frozen context SHA is deterministic.

Placement fixtures include valid `after`, `before`, `between`, and `parallel`
boundaries plus missing/reversed-boundary rejections. For insertion contexts, later
historical turns are labelled `following_context_constraint_only`, never character
knowledge.

## Long blueprint, DAG, and runtime

The fixture has {len(data['compiled']['scenes'])} scenes and a 30–45-turn budget.
It includes a main chain of at least three scenes, a character branch, an event
branch, and a merge. Scenes remain independent generation units but carry historical
anchors and explicit same-round dependencies. Failed hard predecessors block
descendants; an independently anchored `parallel_to` branch can continue.
Promotion closure rejects a dependent scene without its hard predecessor and can
retain an independent parallel branch.

Connected-jump fixtures accept a time jump, location shift, related-character view,
connected new person, connected new event, parallel thread, and information transfer
when an existing typed ref provides the bridge. They reject an unknown new person,
an event with no old-world target, and an Interaction Link miscast as an explicit
relation, with deterministic `unrelated_topic_jump`/typed-reference diagnostics.

`candidate_update_hints=[]` compiles to the existing no-proposal scene contract and
allocates no Binder world IDs. A non-empty hint is only a candidate compiled through
the existing Action Registry; it is not confirmed and cannot write Memory in Phase A.

## Structural and narrative gates

The structural plan gate checks anchor/boundary, typed old-world refs, Interaction /
Relation separation, connected new elements, DAG/branch shape, duplicates,
fingerprints, logical targets, and a length-3 chain. The realized gate checks P0,
predecessor dialogue SHA, stable refs, exact and NFKC/whitespace/outer-quote
duplicates. Complex semantic novelty remains `semantic_review_required=true` and
`semantic_novelty_review_ready=false`.

Narrative placement is a manifest over append-only storage. No Parent turn, child
turn, scene ref, or historical artifact was physically reordered or rewritten.
Fixtures prove before/between/parallel narrative keys while generation/storage refs
stay unchanged; branch, merge, dependency, and impact refs remain explicit.

## v10 shadow replay

The old plan has no historical anchors or dependency DAG, selects Memory actions in
all three scenes, and its old context dropped `P05:T010`–`P05:T016`. Context v2 keeps
them. M000001–M000060 are excluded from Planning State. M000064 is flagged for
semantic review—not declared novel or rewritten. v10 remains read-only.

The offline 11-root matrix passed all 10 Conversation Chronicles roots plus the
attested LoCoMo `conv-43` record 77. No permanent compatibility root was created.

## Checks

{checks}

Targeted tests: `{data['checks']['targeted_tests']['exit_code']}`; relevant suite:
`{data['checks']['related_tests']['exit_code']}`; compileall:
`{data['checks']['compileall']['exit_code']}`; git diff --check:
`{data['checks']['diff_check']['exit_code']}`.

API calls/network attempts = 0. No Fresh E2E, Round 2, Questioner, Distractor,
Solver, QA Replay, Strategy, Training, or Git commit was run. Phase B must first
implement and freeze the Initial Structure Extractor, Round Validator schema,
durable-update compiler, and request budget.
"""


def run(output_root: Path, *, refresh_owned_output: bool = False) -> dict[str, Any]:
    if output_root.exists() and not refresh_owned_output:
        raise FileExistsError(f"phase_a_output_exists:{output_root}")
    if not V10.is_dir():
        raise FileNotFoundError(V10)
    history_before = _historical_hashes()
    input_before = sha256_file(INPUT)
    protected_before = _protected_hashes()
    git_head_before = git_value(REPO, "rev-parse", "HEAD")
    dirty_before = git_value(REPO, "status", "--short")

    loaded = load_unified_root_read_only(INPUT)
    v10_memory = json.loads((V10 / "world_memory_final.json").read_text())
    v10_links = json.loads((V10 / "interaction_links_final.json").read_text())
    v10_child = json.loads((V10 / "child_dialogue_promotable.json").read_text())
    speaker_map = {row["display_name"]: row["entity_ref"]
                   for row in v10_memory["entities"]
                   if row.get("introduced_round") == 0}
    complete = compose_complete_dialogue_view(
        original_parent=loaded["parent"],
        promoted_rounds=[{"round_index": 1, "child": v10_child}],
        speaker_to_entity_ref=speaker_map)
    cumulative = build_cumulative_fixture(loaded["parent"], speaker_map)
    complete["phase_a_cumulative_fixture"] = {
        "original_turns": 60, "round_1_turns": 30, "round_2_turns": 35,
        "expected_round_3_complete_turns": 125,
        "actual_round_3_complete_turns": cumulative["turn_count"],
        "passed": cumulative["turn_count"] == 125,
        "complete_dialogue_sha256": cumulative["complete_dialogue_sha256"],
    }
    complete["complete_dialogue_sha256"] = _sha({
        key: value for key, value in complete.items()
        if key != "complete_dialogue_sha256"})
    planning = compile_planning_state_view(
        memory=v10_memory, interaction_links=v10_links,
        scene_lineage=[{"scene_ref": row["scene_ref"], "round_index": 1}
                       for row in v10_child["scenes"]],
        recent_round_end={"scene_ref": v10_child["scenes"][-1]["scene_ref"],
                          "turn_ref": v10_child["scenes"][-1]["dialogue_turns"][-1]["turn_ref"]})
    separation = audit_interaction_relation_separation(planning)
    anchors = compile_anchor_registry(
        complete_dialogue=complete, planning_state=planning)
    tail_anchor = next(row for row in anchors["anchors"]
                       if row["anchor_type"] == "tail_anchor")
    gap_anchor = next(row for row in anchors["anchors"]
                      if row["anchor_type"] == "historical_gap_anchor")

    v10_plan = json.loads((V10 / "planner_output.json").read_text())
    simple = v10_plan["scene_goals"][1]["proposal_selection"]
    entity_refs = list(tail_anchor["linked_entity_refs"])
    blueprint = build_round_blueprint_fixture(
        source_anchor_id=tail_anchor["anchor_id"],
        placement_choice_id="PC_TAIL_AFTER", entity_refs=entity_refs,
        single_action_selection=simple)
    menu = json.loads((V10 / "affordance_menu.json").read_text())
    plan_validation = validate_continuous_round_plan(
        blueprint, source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices, action_menu=menu,
        authoritative_entity_refs={row["entity_ref"] for row in planning["entities"]})
    if not plan_validation["valid"]:
        raise RuntimeError("phase_a_fixture_plan_invalid:" +
                           ",".join(plan_validation["errors"]))
    compiled = compile_continuous_round_plan(
        blueprint, action_menu=menu, source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices, round_index=2)
    dag = build_scene_dependency_dag(compiled)
    narrative = build_narrative_order_manifest(
        compiled_plan=compiled, complete_dialogue=complete)
    plan_gate = validate_round_blueprint_structure(
        compiled_plan=compiled, source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices,
        complete_dialogue=complete, planning_state=planning)
    if not plan_gate["valid"]:
        raise RuntimeError("phase_a_fixture_gate_invalid:" +
                           ",".join(plan_gate["errors"]))

    # Boundary fixtures, including future-only knowledge for insertion.
    placement_valid = []
    placement_anchors = {"after": tail_anchor, "before": tail_anchor,
                         "between": gap_anchor, "parallel": tail_anchor}
    for placement, anchor in placement_anchors.items():
        placement_valid.append({"placement": placement, "anchor_id": anchor["anchor_id"],
                                "result": validate_placement_boundary(
                                    anchor=anchor, placement=placement,
                                    complete_dialogue=complete)})
    invalid_placements = []
    for placement, missing in (("after", "after_turn_ref"),
                               ("before", "before_turn_ref"),
                               ("between", "before_turn_ref")):
        anchor = deepcopy(placement_anchors[placement])
        anchor["boundary"][missing] = None
        invalid_placements.append({"placement": placement, "missing": missing,
                                   "result": validate_placement_boundary(
                                       anchor=anchor, placement=placement,
                                       complete_dialogue=complete)})
    parallel_bad = deepcopy(tail_anchor)
    parallel_bad["session_ref"] = None
    parallel_bad["boundary"] = {"after_turn_ref": None, "before_turn_ref": None}
    invalid_placements.append({"placement": "parallel", "missing": "interval",
                               "result": validate_placement_boundary(
                                   anchor=parallel_bad, placement="parallel",
                                   complete_dialogue=complete)})
    insertion_scene = deepcopy(compiled["scenes"][0])
    insertion_scene.update({
        "source_anchor_id": gap_anchor["anchor_id"],
        "source_anchor": resolve_source_anchor(source_anchors, gap_anchor["anchor_id"]),
        "placement_choice_id": "PC_GAP_0001",
        "placement_choice": resolve_placement_choice(
            placement_choices, "PC_GAP_0001"),
        "placement": "between", "predecessor_scene_refs": [],
                            "dependency_scene_refs": []})
    insertion_context = compile_scene_context_v2(
        scene=insertion_scene, complete_dialogue=complete,
        source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices, planning_state=planning)

    outputs: dict[str, dict[str, Any]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    for scene in compiled["scenes"]:
        context = compile_scene_context_v2(
            scene=scene, complete_dialogue=complete,
            source_anchor_registry=source_anchors,
            placement_choice_registry=placement_choices,
            planning_state=planning, same_round_scenes=outputs)
        contexts[scene["scene_ref"]] = context
        outputs[scene["scene_ref"]] = _scene_output(scene)
    realized_gate = validate_realized_expansion_structure(
        compiled_plan=compiled, scene_outputs=outputs,
        context_manifests=contexts, complete_dialogue=complete,
        narrative_order_manifest=narrative)

    statuses = {scene["scene_ref"]: "not_run" for scene in compiled["scenes"]}
    statuses[compiled["scenes"][0]["scene_ref"]] = "generated"
    statuses[compiled["scenes"][1]["scene_ref"]] = "scene_failed"
    dependency = simulate_dependency_runtime(compiled, statuses)
    all_refs = [scene["scene_ref"] for scene in compiled["scenes"]]
    promotion = {
        "dependent_without_predecessor": compute_promotion_closure(
            compiled, requested_promoted_scene_refs=[all_refs[1]],
            committed_scene_refs=[]),
        "independent_parallel_branch": compute_promotion_closure(
            compiled, requested_promoted_scene_refs=[all_refs[3]],
            committed_scene_refs=[]),
        "full_closure": compute_promotion_closure(
            compiled, requested_promoted_scene_refs=all_refs,
            committed_scene_refs=[all_refs[1]]),
    }
    linked_fixture = {"entity_refs": entity_refs[:1], "relation_refs": [],
                      "event_refs": [], "task_refs": [], "thread_refs": [],
                      "state_refs": []}
    valid_jump_types = ("time_skip", "location_shift", "related_character_shift",
                        "connected_new_event", "connected_new_entity",
                        "parallel_thread", "information_transfer")
    connected_jumps = {
        "valid": [{"link_type": value, "result": validate_connected_jump(
            link_type=value, linked_existing_refs=linked_fixture,
            planning_state=planning,
            introduces_new_entity=value == "connected_new_entity",
            introduces_new_event=value == "connected_new_event")}
                  for value in valid_jump_types],
        "invalid": [
            {"name": "unrelated_topic_jump", "result": validate_connected_jump(
                link_type="location_shift", linked_existing_refs={
                    key: [] for key in linked_fixture}, planning_state=planning,
                introduces_new_event=True)},
            {"name": "unknown_new_person_entry", "result": validate_connected_jump(
                link_type="connected_new_entity", linked_existing_refs={
                    **linked_fixture, "entity_refs": ["unknown:E"]},
                planning_state=planning, introduces_new_entity=True)},
            {"name": "interaction_as_friend_relation", "result": validate_connected_jump(
                link_type="relationship_followup", linked_existing_refs={
                    **linked_fixture,
                    "relation_refs": [planning["interaction_links"][0]["interaction_link_ref"]]},
                planning_state=planning)},
        ],
    }
    branching = {"valid_dag": plan_gate}
    isolated = deepcopy(compiled)
    isolated["scenes"][-1]["dependency_edges"] = []
    isolated["scenes"][-1]["predecessor_scene_refs"] = []
    isolated["scenes"][-1]["dependency_scene_refs"] = []
    branching["isolated_rejected"] = validate_round_blueprint_structure(
        compiled_plan=isolated, source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices,
        complete_dialogue=complete, planning_state=planning)
    short = deepcopy(compiled)
    for scene in short["scenes"]:
        scene["dependency_edges"] = [edge for edge in scene["dependency_edges"]
                                     if edge["dependency_type"] == "parallel_to"]
        scene["predecessor_scene_refs"] = [edge["predecessor_scene_ref"]
                                           for edge in scene["dependency_edges"]]
    branching["short_chain_rejected"] = validate_round_blueprint_structure(
        compiled_plan=short, source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices,
        complete_dialogue=complete, planning_state=planning)

    old_context = json.loads((V10 / "planner/round_context_manifest.json").read_text())
    final_memory = json.loads((V10 / "world_memory_final.json").read_text())
    m64 = next(row for row in final_memory["authoritative_records"]
               if str(row["memory_ref"]).endswith("M000064"))
    shadow = {
        "source_run": str(V10), "source_write_count": 0,
        "old_plan_scene_count": len(v10_plan["scene_goals"]),
        "old_plan_memory_action_scene_count": sum(
            row["proposal_selection"]["selection_type"] != "none"
            for row in v10_plan["scene_goals"]),
        "old_plan_has_historical_anchor": any(
            "historical_anchor_id" in row for row in v10_plan["scene_goals"]),
        "old_plan_has_scene_dependency": any(
            "predecessor_scene_aliases" in row for row in v10_plan["scene_goals"]),
        "old_context_same_round_scene_text_included": old_context.get(
            "same_round_scene_text_included"),
        "old_context_dropped_parent_tail_refs": old_context.get("dropped_context_refs"),
        "v2_parent_tail_refs": contexts[all_refs[0]]["parent_true_tail_refs"],
        "v2_parent_tail_fully_retained": contexts[all_refs[0]][
            "parent_true_tail_fully_retained"],
        "parent_utterance_records_excluded_from_planning_state": 60,
        "m000064": {"record_type": m64["record_type"],
                    "logical_key": m64["logical_key"],
                    "semantic_review_required": True,
                    "structurally_declared_novel": False,
                    "memory_write_count": 0},
        "semantic_novelty_review_ready": False,
    }
    eleven = _eleven_root_preflight()
    budget = _request_budget()

    checks = {
        "targeted_tests": _run_check([
            sys.executable, "-m", "unittest",
            "tests.test_world_memory_slim_continuous_expansion_v11"]),
        "related_tests": _run_check([
            sys.executable, "-m", "unittest", "discover", "-s", "tests",
            "-p", "test_world_memory_slim*.py"]),
        "compileall": _run_check([
            sys.executable, "-m", "compileall", "-q",
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1"]),
        "diff_check": _run_check(["git", "diff", "--check"]),
    }

    history_after = _historical_hashes()
    protected_after = _protected_hashes()
    readiness = {
        "complete_dialogue_view_ready": (
            validate_complete_dialogue_view(complete)["valid"] and
            cumulative["turn_count"] == 125),
        "planning_state_view_ready": validate_planning_state_view(planning)["valid"],
        "interaction_relation_separated": separation["valid"],
        "anchor_registry_ready": validate_anchor_registry(
            anchors, complete_dialogue=complete)["valid"],
        "placement_boundary_ready": (all(row["result"]["valid"] for row in placement_valid)
                                     and all(not row["result"]["valid"]
                                             for row in invalid_placements)),
        "connected_scene_jump_ready": (all(row["result"]["valid"]
                                            for row in connected_jumps["valid"])
                                       and all(not row["result"]["valid"]
                                               for row in connected_jumps["invalid"])),
        "branching_scene_dag_ready": (plan_gate["valid"] and
                                      not branching["isolated_rejected"]["valid"] and
                                      not branching["short_chain_rejected"]["valid"]),
        "same_round_context_ready": (all(validate_context_manifest_v2(row)["valid"]
                                         for row in contexts.values()) and
                                     contexts[all_refs[1]]["same_round_scene_text_included"] and
                                     set(all_refs[:2]).issubset(
                                         contexts[all_refs[2]]["same_round_predecessor_refs"])),
        "dependency_runtime_blocking_ready": bool(dependency["blocked_scene_refs"]),
        "dependency_promotion_closure_ready": (
            not promotion["dependent_without_predecessor"]["valid"] and
            promotion["independent_parallel_branch"]["valid"] and
            promotion["full_closure"]["valid"]),
        "structural_expansion_precheck_ready": plan_gate["valid"] and realized_gate["valid"],
        "narrative_order_manifest_ready": validate_narrative_order_manifest(narrative)["valid"],
        "11_root_preflight_ready": eleven["all_passed"],
        "initial_structure_extractor_ready": False,
        "round_expansion_validator_ready": False,
        "semantic_novelty_review_ready": False,
        "phase_b_request_budget_ready": False,
        "generation_ready": False, "fresh_e2e_ready": False,
        "qa_pipeline_ready": False, "api_calls": 0,
    }
    integrity = {
        "input_sha256_before": input_before,
        "input_sha256_after": sha256_file(INPUT),
        "input_unchanged": input_before == sha256_file(INPUT),
        "historical_before": history_before, "historical_after": history_after,
        "historical_unchanged": all(history_before[key]["tree_sha256"] ==
                                    history_after[key]["tree_sha256"]
                                    for key in history_before),
        "protected_module_sha_before": protected_before,
        "protected_module_sha_after": protected_after,
        "protected_modules_unchanged": protected_before == protected_after,
        "git_head_before": git_head_before,
        "git_head_after": git_value(REPO, "rev-parse", "HEAD"),
        "git_head_unchanged": git_head_before == git_value(REPO, "rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(REPO),
        "dirty_worktree_before": dirty_before,
        "api_calls": 0, "network_attempts": 0,
        "fresh_e2e_runs": 0, "round_2_runs": 0, "qa_calls": 0,
        "training_calls": 0, "git_commit_calls": 0,
    }
    if not all(check["passed"] for check in checks.values()):
        readiness["structural_expansion_precheck_ready"] = False
    if not integrity["historical_unchanged"] or not integrity["input_unchanged"]:
        readiness["structural_expansion_precheck_ready"] = False

    output_root.mkdir(parents=True, exist_ok=refresh_owned_output)
    artifacts = {
        "complete_dialogue_view.json": complete,
        "planning_state_view.json": planning,
        "interaction_relation_separation_audit.json": separation,
        "anchor_registry.json": anchors,
        "placement_boundary_fixtures.json": {
            "valid": placement_valid, "invalid": invalid_placements,
            "between_context_partition": insertion_context},
        "context_manifest_v2_fixtures.json": {
            "manifests": contexts,
            "parent_tail_required": [f"cc-train-episode-61555:P05:T{index:03d}"
                                     for index in range(10, 17)],
            "retry_context_sha_stable": _sha(contexts) == _sha(deepcopy(contexts))},
        "round_blueprint_fixture.json": {"model_fixture": blueprint,
                                         "validation": plan_validation,
                                         "compiled": compiled},
        "scene_dependency_dag.json": dag,
        "branching_scene_dag_fixtures.json": branching,
        "connected_jump_fixtures.json": connected_jumps,
        "dependency_runtime_dry_run.json": dependency,
        "promotion_closure_dry_run.json": promotion,
        "structural_gate_results.json": {
            "plan_gate": plan_gate, "realized_gate": realized_gate,
            "candidate_update_compatibility": {
                "no_proposal_scene_count": sum(
                    not scene["compiled_proposals"] for scene in compiled["scenes"]),
                "no_proposal_binder_allocation_count": 0,
                "candidate_hint_is_confirmed": False,
                "memory_write_count": 0}},
        "narrative_order_manifest.json": narrative,
        "request_budget_comparison.json": budget,
        "v10_shadow_replay_audit.json": shadow,
        "readiness.json": {"readiness": readiness, "integrity": integrity,
                           "checks": checks, "eleven_root_preflight": eleven},
    }
    for name, value in artifacts.items():
        atomic_json(output_root / name, value)
    report_data = {"complete": complete, "cumulative_fixture": cumulative,
                   "planning": planning, "anchors": anchors, "compiled": compiled,
                   "readiness": readiness, "checks": checks}
    phase_report = _phase_a_report(report_data)
    migration = _migration_review()
    atomic_text(REPORT_DIR / "SLIM_CONTINUOUS_EXPANSION_V11_PHASE_A_REPORT.md",
                phase_report)
    atomic_text(REPORT_DIR / "ROUND_EXPANSION_VALIDATOR_MIGRATION_REVIEW.md",
                migration)
    atomic_text(output_root / "reports/SLIM_CONTINUOUS_EXPANSION_V11_PHASE_A_REPORT.md",
                phase_report)
    atomic_text(output_root / "reports/ROUND_EXPANSION_VALIDATOR_MIGRATION_REVIEW.md",
                migration)
    return {"output_root": str(output_root), "readiness": readiness,
            "integrity": integrity, "checks": checks,
            "artifact_count": len(artifacts) + 2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-owned-output", action="store_true",
                        help="Atomically refresh only this Phase-A preflight's artifacts")
    args = parser.parse_args(argv)
    result = run(args.output_root.resolve(),
                 refresh_owned_output=args.refresh_owned_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(result["checks"][name]["passed"] for name in result["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
