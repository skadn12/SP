"""Fixed-manifest, zero-API final-freeze preflight for Slim Pipeline v7."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from ..core import sha256_file
from .append_only_memory import apply_scene_commit
from .claim_validator_v2 import validate_proposal_claims
from .context_compiler import compile_scene_context
from .controlled_expansion import compile_action_menu_v3, validate_action_menu_v3
from .dialogue_quality import validate_dialogue_quality
from .evidence_scope import (build_committer_scope_prompt,
                             validate_committer_scope_output)
from .inputs import load_locomo_parent_read_only, load_unified_root_read_only
from .interaction_links import compile_source_interaction_links
from .planner_v3 import (PLANNER_SCHEMA_VERSION_V3, build_planner_v3_prompt,
                         build_planner_v3_retry_prompt, compile_scene_plan_v3,
                         validate_planner_v3)
from .planner_contract_v3_1 import (
    compile_allowed_proposal_bundles, compile_planner_action_view,
    validate_planner_action_view,
)
from .promotion import assemble_child_surfaces
from .realizer_v3 import (build_claim_only_v3_prompt, build_realizer_v3_prompt,
                          validate_claim_only_realizer_shape,
                          validate_full_realizer_shape)
from .source_memory_initializer_v1 import (authoritative_affordance_view,
                                           initialize_source_world_memory)
from .transactional_binder import (validate_scene_local_evidence_refs,
                                   build_allocation_plan,
                                   commit_scene_transaction)
from .turn_budget import compile_turn_budget


OFFLINE_PREFLIGHT_VERSION = "slim_pipeline_v7_offline_preflight_v1"


def _sessions(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for turn in turns:
        grouped[str(turn.get("session_id") or "parent")].append(turn)
    return [{"session_id": key, "turns": value} for key, value in sorted(grouped.items())]


def _content(action: Mapping[str, Any]) -> dict[str, Any]:
    operation = action["operation"]
    fixtures: dict[str, dict[str, Any]] = {
        "add_person": {"display_role": "a person connected to the situation", "scene_presence": "mentioned_only"},
        "add_relation": {"source_ref": action["allowed_existing_entity_refs"][0],
                         "target_ref": action["allowed_existing_entity_refs"][1],
                         "relation_type": "knows"},
        "add_event": {"description": "They encounter a new practical development."},
        "create_state": {"state_category": "commitment", "state_dimension": "follow through",
                         "initial_value": "considering"},
        "add_shared_task": {"task_description": "Prepare one concrete next step together."},
        "open_thread": {"description": "They need to revisit one unresolved practical question."},
        "add_third_party_report": {"reported_content": "A third person reports a relevant development."},
        "add_information_asymmetry": {"information": "One participant learns a relevant detail first."},
        "create_conflict": {"conflict_description": "They disagree about how to proceed."},
        "create_misunderstanding": {"misunderstanding": "They interpret the same situation differently."},
        "introduce_external_constraint": {"constraint": "A schedule limitation narrows their options."},
        "add_role_handoff": {"source_ref": action["allowed_existing_entity_refs"][0],
                             "target_ref": action["allowed_existing_entity_refs"][1],
                             "handoff_description": "Responsibility moves to the second participant."},
        "update_state": {"new_value": "reconsidered"},
        "change_trust": {"new_value": "more cautious"},
        "change_authority": {"new_value": "shared authority"},
        "resolve_conflict": {"resolution": "They choose a mutually acceptable next step."},
        "close_thread": {"closure": "They settle the open question."},
        "supersede": {"target_type": "state", "authoritative_version": action.get("authoritative_version") or 1,
                      "replacement_payload": {"value": "replacement"}, "supersede_reason": "new verified information"},
    }
    frozen = set((action.get("frozen_payload") or {}).keys())
    return {key: value for key, value in fixtures[operation].items() if key not in frozen}


def _fixture_plan(action: Mapping[str, Any], entity_refs: Sequence[str]) -> dict[str, Any]:
    speakers = [ref for ref in entity_refs if ref in action["allowed_existing_entity_refs"]][:2]
    if len(speakers) < 2:
        raise ValueError("preflight_requires_two_authoritative_speakers")
    speaker_rows = [{"participant_ref": speakers[0], "scene_role": "primary",
                     "intents": ["raise a concrete concern", "seek a usable next step"]},
                    {"participant_ref": speakers[1], "scene_role": "supporting",
                     "intents": ["respond with a grounded option"]}]
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION_V3,
        "scene_budget": {"minimum": 2, "target": 3, "maximum": 4},
        "round_arc": {"starting_situation": "A practical issue is still active.",
                      "central_tension": "The participants need a clearer next step.",
                      "ending_direction": "They leave with a concrete direction."},
        "scene_goals": [{
            "goal": "Develop one concrete extension without changing source facts.",
            "speaker_participants": speaker_rows,
            "interaction_beats": ["The primary raises the issue.", "The supporting speaker challenges one assumption.",
                                  "They agree on one concrete direction."],
            "continuity_requirements": ["Respect authoritative context."],
            "proposal_selection": {
                "selection_type": "single_action",
                "action_id": action["action_id"],
                "proposed_content": _content(action),
            },
            "desired_outcome": "One proposal is expressed clearly enough for independent review.",
        }, {
            "goal": "Clarify how they will communicate next.",
            "speaker_participants": speaker_rows,
            "interaction_beats": ["They clarify expectations.", "They compare options.", "They close the exchange naturally."],
            "continuity_requirements": ["Do not assert a new persistent fact."],
            "proposal_selection": {"selection_type": "none"},
            "desired_outcome": "The dialogue closes without an additional memory proposal.",
        }],
    }


def _dialogue(scene: Mapping[str, Any], budget: Mapping[str, Any]) -> list[dict[str, str]]:
    speakers = [row["participant_ref"] for row in scene["speaker_participants"]]
    return [{"turn_ref": f"{scene['scene_ref']}:T{index:03d}",
             "speaker_ref": speakers[(index - 1) % len(speakers)],
             "text": f"Offline fixture utterance {index} advances the scene without copying source text."}
            for index in range(1, budget["target"] + 1)]


def run_fixed_manifest_preflight(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("roots") or []
    if len(entries) != 11 or len({row.get("root_id") for row in entries}) != 11:
        return {"preflight_version": OFFLINE_PREFLIGHT_VERSION, "ready": False,
                "failure": "fixed_manifest_requires_exactly_11_unique_roots", "roots": []}
    results: list[dict[str, Any]] = []
    for entry in entries:
        path = Path(entry["input_path"])
        input_sha = sha256_file(path)
        if input_sha != entry["expected_sha256"]:
            results.append({"root_id": entry["root_id"], "passed": False,
                            "failure": "manifest_input_sha_mismatch"})
            continue
        if entry["source_kind"] == "unified_root":
            loaded = load_unified_root_read_only(path)
        else:
            loaded = load_locomo_parent_read_only(
                path, Path(entry["attestation_path"]),
                line_number=int(entry["source_line_number"]),
                expected_identity=str(entry["expected_identity"]))
        root, parent = loaded["root"], loaded["parent"]
        if root["root_id"] != entry["root_id"]:
            raise ValueError("preflight_loaded_root_identity_mismatch")
        source = initialize_source_world_memory(root, parent)
        append_memory = source["memory"]
        affordance_memory = authoritative_affordance_view(append_memory)
        speaker_to_entity = source["speaker_to_entity_ref"]
        source_links = compile_source_interaction_links(
            root_id=root["root_id"], sessions=_sessions(parent["dialogue"]["turns"]),
            speaker_to_entity=speaker_to_entity)
        menu = compile_action_menu_v3(affordance_memory, root_id=root["root_id"],
                                      interaction_links=source_links)
        menu_validation = validate_action_menu_v3(menu)
        if not menu["actions"]:
            results.append({"root_id": root["root_id"], "passed": False,
                            "failure": "empty_action_menu"})
            continue
        entity_refs = [row["entity_ref"] for row in append_memory["entities"]]
        action_view = compile_planner_action_view(
            menu, role_table=append_memory["entities"])
        selectable_ids = {
            row["action_id"] for row in action_view["actions"]
            if row.get("selection_policy") == "single_allowed"
        }
        action = next((row for row in menu["actions"]
                       if row["action_id"] in selectable_ids), None)
        if action is None:
            results.append({"root_id": root["root_id"], "passed": False,
                            "failure": "no_single_allowed_preflight_action"})
            continue
        bundles = compile_allowed_proposal_bundles(menu)
        action_view_validation = validate_planner_action_view(menu, action_view)
        plan = _fixture_plan(action, entity_refs)
        planner_validation = validate_planner_v3(
            plan, action_menu=menu, authoritative_entity_refs=set(entity_refs),
            planner_action_view=action_view, allowed_bundles=bundles)
        compiled = compile_scene_plan_v3(
            plan, action_menu=menu, round_index=1, allowed_bundles=bundles)
        scene = compiled["scenes"][0]
        context = compile_scene_context(
            scene=scene, actions=[action], authoritative_memory=affordance_memory,
            parent_turns=parent["dialogue"]["turns"], interaction_links=source_links)
        planner_prompt = build_planner_v3_prompt(
            root_id=root["root_id"], role_table=append_memory["entities"],
            action_menu=menu, context_summary=context,
            planner_action_view=action_view, allowed_bundles=bundles)
        planner_retry_prompt = build_planner_v3_retry_prompt(
            root_id=root["root_id"], frozen_action_menu=menu,
            frozen_role_table=append_memory["entities"], frozen_context=context,
            frozen_scene_budget={"minimum": 2, "target": 3, "maximum": 4},
            errors=["offline_fixture_error"],
            relevant_action_ids=[action["action_id"]],
            frozen_authoritative_memory_sha256="m" * 64,
            frozen_interaction_links_sha256="i" * 64,
            frozen_round_context_sha256="c" * 64,
            frozen_planner_action_view=action_view,
            frozen_allowed_bundles=bundles)
        budget = compile_turn_budget(scene)
        dialogue = _dialogue(scene, budget)
        realizer = {"dialogue_turns": dialogue, "proposal_claims": [{
            "proposal_ref": scene["compiled_proposals"][0]["proposal_ref"],
            "realization_status": "realized",
            "candidate_evidence_turn_refs": [dialogue[-1]["turn_ref"]],
        }]}
        full_shape = validate_full_realizer_shape(realizer)
        quality = validate_dialogue_quality(
            scene_ref=scene["scene_ref"], dialogue_turns=dialogue,
            speaker_participants=scene["speaker_participants"], turn_budget=budget,
            parent_utterances=[row["text"] for row in parent["dialogue"]["turns"]])
        claims = validate_proposal_claims(
            scene_ref=scene["scene_ref"], proposal_claims=realizer["proposal_claims"],
            compiled_proposals=scene["compiled_proposals"], dialogue_turns=dialogue)
        committer = {"proposal_reviews": [{
            "proposal_ref": scene["compiled_proposals"][0]["proposal_ref"],
            "review_decision": "confirmed", "verification_status": "semantically_verified",
            "accepted_evidence_turn_refs": [dialogue[-1]["turn_ref"]], "reason_code": "supported",
        }], "scope_status": "clean", "world_consistency_status": "clean"}
        committer_validation = validate_committer_scope_output(
            committer, compiled_proposals=scene["compiled_proposals"],
            proposal_claims=realizer["proposal_claims"])
        realizer_prompt = build_realizer_v3_prompt(scene=scene, context=context, turn_budget=budget)
        claim_prompt = build_claim_only_v3_prompt(
            scene_ref=scene["scene_ref"], frozen_dialogue_turns=dialogue,
            compiled_proposals=scene["compiled_proposals"], errors=["offline fixture"])
        claim_only_shape = validate_claim_only_realizer_shape(
            {"proposal_claims": realizer["proposal_claims"]})
        committer_prompt = build_committer_scope_prompt(
            compiled_proposals=scene["compiled_proposals"], dialogue_turns=dialogue,
            proposal_claims=realizer["proposal_claims"], authoritative_world=append_memory)
        accepted_by_proposal = {committer["proposal_reviews"][0]["proposal_ref"]:
                                committer["proposal_reviews"][0]["accepted_evidence_turn_refs"]}
        binding = validate_scene_local_evidence_refs(
            scene_ref=scene["scene_ref"],
            dialogue_turns=dialogue, accepted_evidence_by_proposal=accepted_by_proposal)
        allocation = build_allocation_plan(
            root_id=root["root_id"], round_index=1, scene_index=1,
            compiled_proposals=scene["compiled_proposals"],
            high_water_marks=append_memory["high_water_marks"])
        committed_memory = apply_scene_commit(
            append_memory, scene_ref=scene["scene_ref"], round_index=1, scene_index=1,
            compiled_proposals=scene["compiled_proposals"], proposal_claims=realizer["proposal_claims"],
            committer_output=committer, allocation_plan=allocation,
            evidence_final_turn_refs=binding["evidence_final_turn_refs"],
            evidence_text_sha256=binding["evidence_text_sha256"])
        allocated_refs = [value["memory_ref"] for value in allocation["proposal_allocations"].values()]
        scene_result = {
            "scene_ref": scene["scene_ref"], "quality_valid": quality["valid"],
            "claims_valid": claims["valid"], "committer_contract_valid": committer_validation["valid"],
            "compiled_proposals": scene["compiled_proposals"], "committer_output": committer,
            "scene_commit_result": {"commit_ready": True},
            "bound_scene": {"scene_ref": scene["scene_ref"], "dialogue_turns": dialogue},
            "committed_memory_refs": allocated_refs,
        }
        surfaces = assemble_child_surfaces(
            child_id=f"{root['root_id']}:offline-child", root_id=root["root_id"],
            scene_results=[scene_result])
        with TemporaryDirectory(prefix="slim-v7-preflight-") as temp:
            first_tx = commit_scene_transaction(
                run_root=Path(temp), run_id="offline", scene_plan=scene,
                proposal_fingerprints=[row["proposal_fingerprint"] for row in scene["compiled_proposals"]],
                allocation_plan=allocation, artifacts={"world_memory.json": committed_memory,
                                                        "promotion_manifest.json": surfaces["promotion_manifest.json"]})
            second_tx = commit_scene_transaction(
                run_root=Path(temp), run_id="offline", scene_plan=scene,
                proposal_fingerprints=[row["proposal_fingerprint"] for row in scene["compiled_proposals"]],
                allocation_plan=allocation, artifacts={"world_memory.json": committed_memory})
        checks = {
            "menu": menu_validation["valid"],
            "planner_action_view": action_view_validation["valid"],
            "planner_prompt_hides_internal_action": '"frozen_payload"' not in planner_prompt and
                                                      '"payload_schema":' not in planner_prompt,
            "planner_retry_hides_internal_action": '"frozen_payload"' not in planner_retry_prompt and
                                                     '"payload_schema":' not in planner_retry_prompt,
            "planner": planner_validation["valid"],
            "full_realizer_shape": full_shape["valid"], "claim_only_shape": claim_only_shape["valid"],
            "quality": quality["valid"], "claims": claims["valid"],
            "committer": committer_validation["valid"],
            "append_only_commit": len(committed_memory["authoritative_records"]) ==
                                  len(append_memory["authoritative_records"]) + 1,
            "transaction": first_tx["status"] == "COMMITTED" and second_tx["idempotent_resume"],
            "promotion": surfaces["child_dialogue.json"] == surfaces["child_dialogue_promotable.json"],
            "same_round_text_isolation": not context["same_round_scene_text_included"],
        }
        results.append({
            "root_id": root["root_id"], "input_sha256": input_sha,
            "passed": all(checks.values()), "checks": checks,
            "authoritative_memory_records": len(append_memory["authoritative_records"]),
            "interaction_links": len(source_links), "action_count": len(menu["actions"]),
            "action_distribution": dict(Counter(row["operation_class"] for row in menu["actions"])),
            "create_action_deficit": menu["create_action_deficit"],
            "planner_prompt_token_estimate": (len(planner_prompt) + 3) // 4,
            "planner_retry_prompt_token_estimate": (len(planner_retry_prompt) + 3) // 4,
            "planner_action_view_count": len(action_view["actions"]),
            "planner_bundle_count": len(bundles["bundles"]),
            "realizer_prompt_token_estimate": (len(realizer_prompt) + 3) // 4,
            "claim_retry_prompt_token_estimate": (len(claim_prompt) + 3) // 4,
            "committer_prompt_token_estimate": (len(committer_prompt) + 3) // 4,
            "context_parent_turn_count": len(context["selected_context_refs"]),
            "outbound_network_attempt_count": 0, "api_client_call_count": 0,
        })
    return {
        "preflight_version": OFFLINE_PREFLIGHT_VERSION,
        "manifest_path": str(manifest_path), "root_count": len(results),
        "ready": len(results) == 11 and all(row.get("passed") for row in results),
        "api_client_call_count": 0, "outbound_network_attempt_count": 0,
        "roots": results,
    }


def directory_manifest_sha(directory: Path) -> tuple[str, list[dict[str, str]]]:
    entries = [{"relative_path": str(path.relative_to(directory)), "file_sha256": sha256_file(path)}
               for path in sorted(path for path in Path(directory).rglob("*") if path.is_file())]
    material = b"".join(row["relative_path"].encode("utf-8") + b"\0" +
                        row["file_sha256"].encode("ascii") + b"\0" for row in entries)
    return hashlib.sha256(material).hexdigest(), entries
