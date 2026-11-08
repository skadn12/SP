"""Sequential A then B expansion with checked cumulative root contexts."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "h4_r02_dualroute_feedback_v1"
sys.path.insert(0, str(LEGACY))
spec = importlib.util.spec_from_file_location("r02_v4_legacy_contract", LEGACY / "runner.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.path.insert(0, str(HERE))
import duplicate_check
import realization_review
import source_inventory
import source_pool
import card_selection
import planner_contract
import purpose_contract
import latest_state as state_contract
import event_state as event_contract
import evidence_refs

VERSION = "h4_sequential_ab_v32"
PARENT_RUN = HERE.parent / "h4_sequential_ab_v24/runs/three_roots_a3b3_01"
SOURCE_SPECS = {
    **base.SOURCE_SPECS,
    "cc-train-episode-135466": ("07", "31e7c751a3b599a11b8d8c96520949f9bdd963c1778353a24089ed5f3addb14d"),
    "cc-train-episode-138343": ("08", "0e04ee3a45b004ab5bcb0fd64c16cbddcc577bf5eb0677ce11579bbc97740a2e"),
}
ROOTS = tuple(SOURCE_SPECS)
DEFAULT_ROOTS = ("cc-train-episode-104814",)
PURPOSE_ATTEMPTS = 1
MODEL = base.dataset_cards.MODEL
STATE_ATTEMPTS = 2
EVENT_STATE_ATTEMPTS = 2
SCENE_ATTEMPTS = 2
PLAN_ATTEMPTS = 2
TRANSPORT_RETRIES = 1
REVIEW_RETRIES = 1
CARD_EXTRACTION_ATTEMPTS = 2
MIN_HTTP_GAP_SECONDS = 35.0
ROUTES = base.ROUTES
LOCK_PATH = HERE.parent / "h4_r02_source_unique_v2/runs/.unique_event_live.lock"

PLANNER = """Plan ONE scene slot after this root's complete cumulative_history.
The current run has three sequential scene slots per route; this request handles
only target_scene_id. Return zero or one scene using the route-specific schema
below. Report genuine shortfall rather than inventing accepted history.

A: identify an existing event and add a substantive action, decision or outcome.
B: FIRST establish the supplied dataset event's independent occasion, trigger and
main purpose in scene.goal from the card's core interaction. Preserve a
card-faithful purpose_input occasion; correct only draft details that change the
interaction or conflict with history. Then adapt participants and timing to the
ENTIRE cumulative_history, including all accepted A and prior B scenes. Ask what
new problem or activity this scene addresses in its own right. History constrains
identity, relationships, commitments and time; unfinished old tasks do not supply
B's main goal. Shared people, themes and motivations are allowed. A new activity
may be motivated by an old event, but another booking or logistics step whose
only purpose is to complete the old trip/interview/activity remains its substep.
Do not require a completely unrelated subject, and do not merely relabel an event.
A simple expression of thanks, interview conversation or farewell may be the
activity on a concrete new occasion; urgency, conflict or a broken promise is
not required. This still needs its own main objective and substantive new
dialogue rather than replay of a completed milestone or an old goal's logistics.

Only scene actors may speak. They may explicitly report other participants'
actions without adding speaking turns for them. Such participants keep their
own card roles; the reporting speaker is not the actor of a reported action.
Keep requested, planned, agreed, attempted and completed states
distinct. Every development must be realized in the new dialogue. Adapt the card's
seed premise or scaffold relations into this root; no source identity, quotation
or outcome is an already-existing root fact. The previous-round feedback guides
clarity and coverage, not replacement of B's supplied event with an old task.
Conversation appends after accepted history; event time must follow explicit text.
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_parent_a(parent, root):
    """Verify exact inherited A artifacts; never promote historical B verdicts."""
    parent = Path(parent).resolve()
    required = [parent / name for name in ("MANIFEST.json", "REPORT.json", "FINAL_AUDIT.json", "MERGE_MANIFEST.json")]
    base.require(all(path.is_file() for path in required), "parent_v24_not_complete_and_audited")
    audit, manifest, report = (base.read(parent / name) for name in ("FINAL_AUDIT.json", "MANIFEST.json", "REPORT.json"))
    base.require(manifest["version"] == "h4_sequential_ab_v24", "parent_version_invalid")
    base.require(audit["status"] == "PASS" and not audit["failures"] and Path(audit["run"]).resolve() == parent,
                 "parent_v24_audit_not_pass")
    base.verify_bindings(manifest["files"])
    folder = parent / "roots" / root / "A"
    result = base.read(folder / "RESULT.json")
    base.require(result in report["routes"] and result["accepted_scenes"] == 2,
                 "parent_A2_report_binding_invalid")
    original = base.read(parent / "sources" / f"{root}.json")
    cumulative = base.read(folder / "cumulative_dialogue.json")
    history = deepcopy(original["turns"])
    paths = set(required + [parent / "sources" / f"{root}.json"])
    paths.update(path for path in folder.rglob("*") if path.is_file())
    for entry in result["scene_metadata"]:
        scene_id = entry["scene_id"]
        realization = base.read(folder / f"{scene_id}.json")
        history = base.append_scene(history, root, ROUTES[0], realization)
        attempts = base.read(folder / f"{scene_id}.ATTEMPTS.json")
        accepted = next(row for row in reversed(attempts) if row["status"] == "PASS")
        realization_dir, review_dir = Path(accepted["request"]), Path(accepted["review_request"])
        base.require(realization_dir.is_relative_to(parent / "requests") and review_dir.is_relative_to(parent / "requests"),
                     "parent_A_request_outside_run")
        base.require(base.read(realization_dir / "ACCEPTANCE.json") == accepted,
                     "parent_A_acceptance_binding_invalid")
        base.require(base.read(review_dir / "REVIEW.json")["status"] == "PASS", "parent_A_review_not_pass")
        raw_scene = parse_object(base.read(realization_dir / "RESPONSE.json")["text"])
        base.require(raw_scene == realization, "parent_A_dialogue_differs_from_model")
    # Bind all A requests so the accepted plan/state/reviewer chains remain reviewable.
    for directory in (parent / "requests").iterdir():
        proof_path = directory / "PROOF.json"
        if proof_path.is_file():
            proof = base.read(proof_path)
            if proof.get("root_id") == root and proof.get("route") == ROUTES[0]:
                paths.update(path for path in directory.rglob("*") if path.is_file())
    base.require(history == cumulative["turns"] and result["output_history_sha256"] == base.digest(history),
                 "parent_A_accepted_chain_changed")
    base.require(all(t["turn_ref"].startswith("R02:A:") for t in history[len(original["turns"]):]),
                 "parent_A_contains_non_A_additions")
    return {"parent": parent, "root_id": root, "paths": sorted(paths), "result": result,
            "cumulative_path": folder / "cumulative_dialogue.json", "original": original}


def import_parent_a(run, root, inputs):
    base.require(base.read(run / "sources" / f"{root}.json") == inputs["original"], "imported_A_frozen_base_mismatch")
    target = run / "roots" / root / "A/cumulative_dialogue.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inputs["cumulative_path"], target)
    record = {"root_id": root, "parent_run": str(inputs["parent"]), "accepted_scenes": 2,
              "source_path": str(inputs["cumulative_path"]), "local_path": str(target),
              "source_sha256": base.digest(inputs["cumulative_path"].read_bytes()),
              "history_sha256": inputs["result"]["output_history_sha256"],
              "new_A_calls": 0, "semantic_certified": False,
              "acceptance_scope": "exact inherited A2 with original model requests and reviews; no B imported",
              "files": base.bindings(inputs["paths"])}
    base.write(run / "INHERITED_A.json", record)


def verify_inherited_a(run):
    record = base.read(run / "INHERITED_A.json")
    base.verify_bindings(record["files"])
    local, original = Path(record["local_path"]), Path(record["source_path"])
    base.require(local.read_bytes() == original.read_bytes()
                 and base.digest(local.read_bytes()) == record["source_sha256"], "inherited_A_bytes_changed")
    base.require(base.digest(base.read(local)["turns"]) == record["history_sha256"], "inherited_A_history_changed")
    return record


def propose_purpose(caller, run, root, scene_id, card, history):
    speakers = sorted({turn["speaker"] for turn in history})
    payload = purpose_contract.build_request(root, scene_id, card, speakers, MODEL)
    projection = purpose_contract.project_feedback(run / "feedback", root, base.feedback)
    base.require(projection == base.read(run / "PURPOSE_FEEDBACK_PROJECTION.json"), "purpose_feedback_projection_changed")
    suffix = purpose_contract.feedback_suffix(projection)
    before = payload["messages"][0]["content"]
    payload["messages"][0]["content"] += suffix
    receipt = {"feedback_kind": "purpose_guidance_projection", "full_feedback_appended": False,
               "projection_sha256": projection["projected_feedback_sha256"],
               "source_sha256": projection["source_sha256"], "suffix_sha256": base.digest(suffix.encode()),
               "before_user_text_sha256": base.digest(before.encode()),
               "request_payload_sha256": base.digest(payload), "current_generation_feedback_used": False}
    guidance = planner_contract.mapping_context(card)["card_guidance"]
    proof = {"root_id": root, "route": ROUTES[1], "scene_id": scene_id, "role": "purpose_proposer",
             "dataset_card_id": card["card_id"], "dataset_card_sha256": base.digest(guidance),
             "allowed_speakers": speakers, "feedback_receipt": receipt,
             "input_scope": "card_allowed_speakers_frozen_feedback_only",
             "history_or_state_supplied": False, "source_context_sha256": base.digest(history),
             "branch_base_sha256": base.digest(route_base_history(run, root, ROUTES[1]))}
    raw, directory = caller.call(f"{root}-B-{scene_id}-purpose-1", payload, proof)
    try:
        purpose = purpose_contract.validate_purpose(raw, root, ROUTES[1], scene_id, card)
    except (ValueError, TypeError) as error:
        base.write(directory / "VALIDATION.json", {"status": "REJECT", "reason": str(error),
                   "errors": getattr(error, "errors", [])})
        return None
    base.write(directory / "VALIDATION.json", {"status": "PASS", "purpose_input_sha256": base.digest(purpose),
                                               "semantic_certified": False})
    return purpose


def transport_inputs():
    """Freeze public provider configuration and imported local transport code.

    Loading these factories makes no HTTP request and does not read credentials.
    Credential contents are intentionally not copied into artifacts.
    """
    profile, _factory, _annotate, _capacity, _dev = base.transport()
    worktree = (base.PROJECT / "worktrees/h4-writer-coverage-v1").resolve()
    code = set()
    for module in tuple(sys.modules.values()):
        module_path = getattr(module, "__file__", None)
        if not module_path:
            continue
        path = Path(module_path).resolve()
        if path.suffix == ".py" and path.is_relative_to(worktree) and path.is_file():
            code.add(path)
    base.require(code, "transport_code_binding_missing")
    tokenizer_files = sorted(path for path in base.TOKENIZER.rglob("*") if path.is_file())
    base.require(tokenizer_files, "capacity_tokenizer_binding_missing")
    return profile(), sorted(code), tokenizer_files


def route_base_history(run, root, route):
    frozen = base.read(run / "sources" / f"{root}.json")["turns"]
    if route == ROUTES[0]:
        return frozen
    a_path = run / "roots" / root / "A/cumulative_dialogue.json"
    if not a_path.exists():
        return frozen
    accepted_a = base.read(a_path)["turns"]
    base.require(accepted_a[:len(frozen)] == frozen, "accepted_A_frozen_prefix_changed")
    base.require(all(t.get("turn_ref", "").startswith("R02:A:")
                     for t in accepted_a[len(frozen):]), "accepted_A_contains_non_A_additions")
    return accepted_a


def branch_history_ok(run, root, route, history):
    route_base = route_base_history(run, root, route)
    base.require(history[:len(route_base)] == route_base, "route_accepted_prefix_changed")
    letter = "A" if route == ROUTES[0] else "B"
    suffix = history[len(route_base):]
    base.require(all(t.get("turn_ref", "").startswith(f"R02:{letter}:") for t in suffix),
                 "unexpected_route_in_new_additions")
    base.require(len({t["turn_ref"] for t in history}) == len(history), "duplicate_branch_turn_ref")
    return route_base


def history_proof(run, root, route, history):
    route_base = branch_history_ok(run, root, route, history)
    frozen = base.read(run / "sources" / f"{root}.json")["turns"]
    inherited_path = run / "INHERITED_A.json"
    inherited = base.read(inherited_path) if inherited_path.exists() else None
    return {"inherited_A_provenance_sha256": base.digest(inherited) if inherited else None,
            "branch_base_sha256": base.digest(route_base),
            "frozen_base_sha256": base.digest(frozen),
            "history_sha256": base.digest(history), "history_turns": len(history),
            "sequential_history_verified": True,
            "accepted_A_in_history": route == ROUTES[1] and len(route_base) > len(frozen),
            "other_branch_in_generator_context": route == ROUTES[1] and len(route_base) > len(frozen),
            "history_policy": "A=frozen;B=frozen+accepted_A;own_accepted_scenes_append"}


def make_request(run, root, route, role, history, cards, scene_id, *, scene=None, repair=None,
                 latest_state=None, event_state=None, purpose_input=None):
    base.require(route in ROUTES and role in ("planner", "realizer"), "invalid_request_role")
    frozen = branch_history_ok(run, root, route, history)
    base.require((route == ROUTES[0] and not cards) or (route == ROUTES[1] and len(cards) == 1),
                 "one_card_per_B_slot_required")
    context = {"root_id": root, "route": route, "role": role, "target_scene_id": scene_id,
               "allowed_speakers": sorted({t["speaker"] for t in frozen}),
               "branch_base_sha256": base.digest(frozen), "cumulative_history": history,
               "dataset_cards": cards, "scene_plan": scene}
    base.require(latest_state is not None, "latest_state_required")
    state_contract.validate_state(latest_state, root, route, scene_id, history)
    context["latest_state"] = latest_state
    if purpose_input is not None:
        purpose_contract.validate_purpose(purpose_input, root, route, scene_id, cards[0])
        context["purpose_input"] = deepcopy(purpose_input)
    if event_state is not None:
        context["event_state"] = event_state
    if cards:
        context["card_mapping_context"] = planner_contract.mapping_context(cards[0], scene)
    if repair:
        context["targeted_repair"] = repair
    prompt = base.COMMON + (PLANNER + planner_contract.planner_contract_prompt(route) if role == "planner" else base.REALIZER)
    if route == ROUTES[0]:
        prompt += ("\nROUTE DUTY A: substantively advance the selected existing event. "
                   "Establish new action, decision, correction or outcome without replaying "
                   "completed actions. Do not replace this duty with an unrelated new event.\n")
    else:
        prompt += ("\nROUTE DUTY B: create a distinct dataset-guided event with its own concrete "
                   "purpose and occasion. Complete cumulative_history includes accepted A when "
                   "present, and is binding history. Sharing people or motivation is allowed. "
                   "Adding transport, booking or other steps to the same old trip, interview, "
                   "meeting or activity does not count as the new event. Make the distinction "
                   "natural and explicit; do not merely rename an old event.\n")
    if purpose_input is not None:
        prompt += ("\nPROVISIONAL PURPOSE: the card's core interaction takes priority over "
                   "purpose_input. This draft is not history, evidence or a binding goal. "
                   "Preserve a card-faithful draft's concrete independent occasion and interaction. "
                   "Correct details only where they change the card's actor/action/object/state or "
                   "actually conflict with history. Missing historical mention of an activity or "
                   "occupation is not a conflict; an established relationship does not restrict "
                   "people to their previous activities. New dialogue may establish compatible "
                   "background or explicitly report other participants' actions. Adjust participation "
                   "or timing to resolve a real conflict, without replacing the card event with an "
                   "old task. Discard a draft's replacement event or unsupported claim about what "
                   "old history already established, then reconstruct a card-faithful occasion. ")
        prompt += ("If no card-faithful independent event can fit this history, return no scene "
                   "with a clear shortfall_reason. Use the existing bounded replanning to "
                   "address card mismatch findings; disagreement with the draft alone is not "
                   "a failure. "
                   "scene.goal remains the only plan goal field.\n" if role == "planner" else
                   "Realize the supplied scene_plan's developments after event scanning, "
                   "preserving its card-faithful occasion and goal through actual dialogue. "
                   "Do not reintroduce purpose draft details absent from scene_plan.\n")
    prompt += state_contract.PLAN_STATE_PROMPT if role == "planner" else (
        "\nUse latest_state together with the complete cumulative_history and scene_plan.state_transition. "
        "The snapshot is fallible; later history remains authoritative. Do not replay completed "
        "actions, assume unresolved conditions, or confuse people, times, pickup and destination. "
        "Make any new occasion or trip explicit relative to historical milestones. "
        "Preserve evidence_basis: reported includes explicitly reported intentions; inferred "
        "and uncertain entries are qualified interpretations, never unconditional completed-history "
        "premises. New dialogue may establish later progress on previously unresolved events; "
        "do not claim that this later progress was already completed in the old history.\n")
    if event_state is not None:
        prompt += ("\nEVENT HISTORY: event_state traces the actual planned events across history segments. "
                   "Its timeline includes updates and uncertainties missed by the initial summary. "
                   "Revisit its exact citations and the full history; preserve completed milestones, "
                   "participants, time and location roles. Do not confuse arrival with completion of "
                   "orientation, or a new occasion with a repeated first occurrence. If revising a plan, "
                   "resolve every revise finding using new developments coherent with this history. "
                   "This is a fallible model extraction, never authority over the original history. "
                   "Preserve inferred/uncertain qualifications. Resolve binding revise findings; "
                   "soft-only findings are diagnostic and do not establish historical facts.\n")
        prompt += ("SCAN OBSERVATIONS are search diagnostics only: they record that the scan did not "
                   "locate a relevant historical record in the supplied history. They are not evidence "
                   "that an event never happened, must not be converted into timeline facts, and do not "
                   "automatically block or pass a plan. Only exact timeline citations and the full history "
                   "can constrain historical claims.\n")
        prompt += ("NONHISTORICAL_ENTRIES preserve rows relocated by an explicit model repair choice. "
                   "Their nested status/statement fields are the original rejected scan data, NOT "
                   "historical facts. kind=planned_development belongs to the proposed scene, and "
                   "kind=not_found is only a search observation. Do not cite or promote either as history.\n")
    if cards:
        if role == "planner":
            prompt += "\nCARD MAPPING CONTRACT:\n" + planner_contract.PLANNER_CONTRACT_PROMPT
        else:
            prompt += ("\nInstantiate the assigned card's event in the new dialogue. Preserve the mapped "
                       "steps, dependency relations, actors, conditions and action states; a map alone "
                       "does not realize them. Do not substitute an unrelated old event. "
                       "Return only the Realizer response schema, not a mapping or a review.\n")
    prompt += "\nThe complete accepted cumulative_history is authoritative, including accepted A for B. Add the route's required event progress; do not copy old blocks.\n"
    prompt += "\nINPUT_JSON\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    payload = {"model": MODEL, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
               "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]}
    identity = base.read(run / "feedback/COMPILED_FEEDBACK.json")["target_solver_identity_sha256"]
    payload, receipt = base.feedback.append_feedback_to_request(
        payload, run / "feedback", root, route, role, target_solver_model=base.TARGET,
        expected_target_identity_sha256=identity)
    proof = {"root_id": root, "route": route, "role": role, "scene_id": scene_id,
             **history_proof(run, root, route, history), "feedback_receipt": receipt,
             "latest_state_sha256": base.digest(latest_state),
             "event_state_sha256": base.digest(event_state),
             "purpose_input_sha256": base.digest(purpose_input),
             "payload_sha256": base.digest(payload)}
    return payload, proof


def validate_plan(raw, root, route, history, cards, scene_id, *, latest_state=None):
    """Validate the actual route-specific object; never fabricate old anchors."""
    plan = parse_object(raw)
    base.require(route in ROUTES, "invalid_plan_route")
    planner_contract.validate_planner_schema(plan, route)
    plan = evidence_refs.materialize_citations(plan, history)
    base.require(plan["root_id"] == root and plan["route"] == route, "planner_identity_mismatch")
    scenes = plan.get("scenes")
    base.require(isinstance(scenes, list) and len(scenes) <= 1, "one_scene_slot_only")
    base.require(isinstance(plan["shortfall_reason"], str), "planner_shortfall_reason_invalid")
    base.require(plan["shortfall_reason"] == "" if scenes else base.text_value(plan["shortfall_reason"]),
                 "planner_shortfall_reason_inconsistent")
    if scenes:
        scene = scenes[0]
        base.require(isinstance(scene, dict), "scene_not_object")
        base.require(scene["scene_id"] == scene_id, "target_scene_id_mismatch")
        speakers = {turn["speaker"] for turn in history}
        actors = scene["actors"]
        base.require(isinstance(actors, list) and 2 <= len(actors) <= len(speakers)
                     and all(isinstance(actor, str) and actor in speakers for actor in actors)
                     and len(set(actors)) == len(actors), "scene_actors_invalid")
        turns = {turn["turn_ref"]: turn for turn in history}
        anchors = scene["source_anchors"]
        minimum = 1 if route == ROUTES[0] else 0
        base.require(isinstance(anchors, list) and minimum <= len(anchors) <= 3,
                     "scene_anchors_missing_or_invalid")
        for anchor in anchors:
            base.require(isinstance(anchor, dict) and set(anchor) == {"turn_ref", "quote"}, "anchor_schema_invalid")
            base.require(isinstance(anchor["turn_ref"], str) and anchor["turn_ref"] in turns, "anchor_ref_missing")
            base.require(base.text_value(anchor["quote"]) and anchor["quote"] in turns[anchor["turn_ref"]]["text"],
                         "anchor_quote_not_exact")
        base.require(base.text_value(scene["goal"]), "scene_goal_missing")
        base.require(isinstance(scene["developments"], list) and 2 <= len(scene["developments"]) <= 5
                     and all(base.text_value(value) for value in scene["developments"]), "scene_developments_invalid")
        definition = scene["event_definition"]
        base.require(base.text_value(definition["trigger"]),
                     "event_definition_trigger_invalid")
        base.require(definition["history_relation"] == ("continuation" if route == ROUTES[0] else "distinct_event"),
                     "event_definition_declared_route_invalid")
        base.validate_topic_ids(scene["feedback_topic_ids"])
        if route == ROUTES[0]:
            base.require(not cards and scene["dataset_card_id"] is None and scene["card_development_links"] == [],
                         "old_route_must_not_use_dataset_card")
        else:
            base.require(len(cards) == 1 and scene["dataset_card_id"] == cards[0]["card_id"],
                         "new_route_card_invalid_or_reused")
        if latest_state is not None:
            base.require(latest_state.get("route") == route, "latest_state_route_mismatch")
            state_contract.validate_transition(scene, latest_state)
        if route == ROUTES[1]:
            mapping_errors = planner_contract.collect_mapping_errors(scene, cards[0])
            if mapping_errors:
                raise planner_contract.MappingValidationError(mapping_errors)
    return plan


def validate_card_plan_fidelity(scene, card):
    """Reject invalid mappings with the complete structured error list."""
    return planner_contract.validate_mapping(scene, card)


def invalid_speaker_details(turns, allowed_speakers):
    allowed = set(allowed_speakers)
    return [{"turn_position": index,
             "speaker": turn.get("speaker") if isinstance(turn, dict) else None}
            for index, turn in enumerate(turns, 1)
            if not isinstance(turn, dict) or turn.get("speaker") not in allowed]


def classify_review_failure(review):
    """Route explicit semantic misses to the realizer; malformed reviews stay with reviewer."""
    if review.get("schema_valid") is False:
        return "reviewer_failure"
    failures = review.get("failures", [])
    if any(isinstance(item, str) and (
            item.startswith("review_verdict_reject:") or
            item.startswith("review_route_fulfillment_reject:") or
            (item.startswith(("development[", "card_step[", "card_dependency[")) and ":not_realized:" in item) or
            (item.startswith(("state_fact[", "event_constraint[")) and ":contradicted:" in item))
           for item in failures):
        return "realizer_repair"
    return "reviewer_failure"


def parse_object(raw):
    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            base.require(key not in value, "duplicate_json_key:" + key)
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=unique_keys) if isinstance(raw, str) else raw
    base.require(isinstance(value, dict), "response_not_object")
    return value


class LocalRequestError(RuntimeError):
    """A local request/configuration failure; no provider shortfall is inferred."""


class Caller:
    """Retry one explicit transient provider status with the identical payload."""
    def __init__(self, run):
        self.run = run
        self.limit = base.read(run / "MANIFEST.json")["http_max"]
        profile, factory, annotate, capacity, dev = base.transport()
        self.profile, self.factory = profile(), factory
        base.require(self.profile == base.read(run / "provider_profile.json"), "provider_profile_changed")
        self.annotate, self.capacity = annotate, capacity
        self.tokenizer = dev.tokenizer_from(str(base.TOKENIZER))
        self.http_counter = 0
        self.request_counter = 0
        self.last_http_completed = None

    @staticmethod
    def error_metadata(error):
        metadata = getattr(error, "metadata", None)
        if isinstance(metadata, dict):
            return deepcopy(metadata)
        response = getattr(error, "response", None)
        status = getattr(error, "status_code", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        return {"http_status": status} if status is not None else {}

    @staticmethod
    def http_status(metadata):
        for key in ("http_status", "status_code"):
            value = metadata.get(key)
            if type(value) is int:
                return value
        return None

    def call(self, label, payload, proof):
        base.require(payload["model"] == MODEL, "unexpected_model")
        self.request_counter += 1
        directory = self.run / "requests" / f"{self.request_counter:03d}-{label}"
        base.write(directory / "REQUEST.json", payload)
        base.write(directory / "RESERVATION.json", {"request_number": self.request_counter,
            "http_max": self.limit, "max_attempts": TRANSPORT_RETRIES + 1,
            "retryable_statuses": [503, 504], "status": "RESERVED"})
        def local_error(code, error=None):
            base.write(directory / "LOCAL_ERROR.json", {
                "error_code": code, "error_type": type(error).__name__ if error else None,
                "http_attempted": False, "failure_category": "local_request_failure",
                "raw_exception_text_saved": False})
            raise LocalRequestError(code) from None

        parameters = self.profile.get("request_parameters", {})
        expected = {"model": self.profile.get("model"), "messages": payload.get("messages"),
                    **{key: value for key, value in parameters.items() if value is not None}}
        if payload.get("max_tokens") != self.profile.get("output_limit"):
            local_error("request_profile_output_budget_mismatch")
        if payload != expected:
            local_error("request_profile_payload_mismatch")
        try:
            capacity = self.annotate(self.capacity(self.profile, self.tokenizer, payload["messages"], payload["max_tokens"]))
        except Exception as error:
            local_error("request_capacity_check_failed", error)
        if capacity["status"] != "PASS":
            local_error("capacity_blocked_no_truncation")
        base.write(directory / "PROOF.json", {**proof, "capacity": capacity, "payload_sha256": base.digest(payload)})
        attempts = []
        for attempt in range(1, TRANSPORT_RETRIES + 2):
            attempt_path = directory / ("HTTP_ATTEMPT.json" if attempt == 1 else f"HTTP_ATTEMPT_{attempt:02d}.json")
            result_path = directory / f"HTTP_RESULT_{attempt:02d}.json"
            status = None
            waited_seconds = 0.0
            if self.last_http_completed is not None:
                waited_seconds = max(0.0, MIN_HTTP_GAP_SECONDS - (time.monotonic() - self.last_http_completed))
                if waited_seconds:
                    time.sleep(waited_seconds)

            http_started = False

            def on_attempt(actual):
                nonlocal http_started
                base.require(actual == payload, "actual_payload_differs_from_frozen_request")
                base.require(self.http_counter < self.limit, "global_http_budget_exhausted")
                self.http_counter += 1
                base.write(attempt_path, {"actual_payload_equals_frozen": True,
                    "payload_sha256": base.digest(actual), "requested_model": MODEL,
                    "http_attempt_number": self.http_counter, "request_attempt": attempt,
                    "same_payload_retry": attempt > 1, "transport_retries": attempt - 1})
                base.write(directory / f"THROTTLE_{attempt:02d}.json", {
                    "minimum_gap_after_previous_response_seconds": MIN_HTTP_GAP_SECONDS,
                    "waited_seconds": waited_seconds, "http_started_at_utc": now()})
                http_started = True

            try:
                response = self.factory(self.profile, on_http_attempt=on_attempt).complete(
                    prompt=payload["messages"][0]["content"], max_tokens=payload["max_tokens"], json_object=True)
                self.last_http_completed = time.monotonic()
                metadata = getattr(response, "metadata", {}) or {}
                status = self.http_status(metadata)
                # Keep each received body, including incomplete/invalid responses,
                # before validation. HTTP_RESULT is written exactly once below.
                base.write(directory / f"HTTP_RESPONSE_{attempt:02d}.json",
                           {"text": response.raw_text, "metadata": metadata})
                if status not in (None, 200):
                    raise RuntimeError("provider_http_status_invalid")
                base.require(metadata.get("returned_model") == MODEL, "response_model_mismatch")
                base.require(metadata.get("finish_reason") == "stop", "response_not_complete")
            except Exception as error:
                if not http_started:
                    local_error("pre_http_transport_contract_failure", error)
                if isinstance(error, (AssertionError, KeyError, TypeError, AttributeError)):
                    raise
                if isinstance(error, ValueError) and str(error) in (
                        "actual_payload_differs_from_frozen_request", "global_http_budget_exhausted"):
                    raise
                self.last_http_completed = time.monotonic()
                metadata = self.error_metadata(error)
                error_status = self.http_status(metadata)
                if error_status is None:
                    error_status = status
                if attempt_path.exists():
                    base.write(result_path, {"http_status": error_status, "outcome": "error",
                                             "error_type": type(error).__name__})
                attempts.append({"attempt": attempt, "error_type": type(error).__name__,
                                 "http_status": error_status,
                                 "completion_status": metadata.get("completion_status"),
                                 "retrying": error_status in (503, 504) and attempt <= TRANSPORT_RETRIES})
                if error_status in (503, 504) and attempt <= TRANSPORT_RETRIES:
                    continue
                base.write(directory / "ERROR.json", {"error_type": type(error).__name__,
                    "metadata": {**metadata, "http_status": error_status}, "attempts": attempts,
                    "automatic_retries": attempt - 1, "retry_policy": "one_retry_only_for_http_503_504"})
                raise RuntimeError("provider_request_failed_after_retry_policy") from None
            else:
                if attempt_path.exists():
                    base.write(result_path, {"http_status": status, "outcome": "response"})
                base.write(directory / "RESPONSE.json", {"text": response.raw_text, "metadata": metadata,
                    "attempts": attempts + [{"attempt": attempt, "http_status": status, "retrying": False}]})
                return response.raw_text, directory
        base.write(directory / "ERROR.json", {"error_type": "retry_loop_exhausted", "attempts": attempts})
        raise RuntimeError("provider_request_failed_after_retry_policy")


def review_request(root, route, scene, history, realization, other_turns, card=None, repair=None,
                   latest_state=None, event_state=None, purpose_input=None):
    prompt = realization_review.build_review_prompt(root, route, scene, history, realization, other_turns,
                                                     dataset_card=card, latest_state=latest_state,
                                                     event_state=event_state, purpose_input=purpose_input,
                                                     require_actor_checks=(route == ROUTES[1]),
                                                     require_goal_checks=(route == ROUTES[1]))
    if repair:
        prompt += "\nREVIEW_FORMAT_REPAIR:\n" + json.dumps(repair, ensure_ascii=False)
    payload = {"model": MODEL, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
               "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]}
    proof = {"root_id": root, "route": route, "role": "scene_reviewer", "scene_id": scene["scene_id"],
             "purpose_input_sha256": base.digest(purpose_input),
             "independent_request_without_prior_verdict": repair is None,
             "review_format_repair": repair is not None,
             "require_actor_checks": route == ROUTES[1],
             "require_goal_checks": route == ROUTES[1],
             "history_policy": "complete_current_root_including_accepted_A_for_B",
             "history_turns": len(history),
             "latest_state_sha256": base.digest(latest_state),
             "event_state_sha256": base.digest(event_state),
             "projected_event_state_sha256": base.digest(realization_review.review_event_context(event_state)),
             "scene_plan_sha256": base.digest(scene),
             "projected_scene_plan_sha256": base.digest(realization_review.review_plan_context(scene)),
             "history_sha256": base.digest(history), "realization_sha256": base.digest(realization)}
    return payload, proof


def repair_for_duplicate(checked, realization):
    # Only echo rejected output's own locations. Never copy the reference corpus
    # into the generator, including on retries.
    return {"instruction": "Rewrite the rejected scene with substantive new developments. Do not just paraphrase copied turns.",
            "failure_codes": checked["reasons"],
            "copied_new_turn_positions": [m["new_position"] for m in checked["matches"] if m["substantive"]],
            "previous_response": realization}


def repair_for_review(review, realization):
    parsed = review.get("parsed") or {}
    checks = parsed.get("checks") if isinstance(parsed.get("checks"), list) else []
    unmet = [c["development_index"] for c in checks
             if isinstance(c, dict) and c.get("realized") is False
             and type(c.get("development_index")) is int and c["development_index"] > 0]
    unmet_card = [{"kind": c["kind"], "id": c["id"]}
                  for c in parsed.get("card_checks", [])
                  if isinstance(c, dict) and c.get("realized") is False]
    soft_facts = {c["fact_id"] for c in review.get("soft_findings", []) if "fact_id" in c}
    soft_events = {c["constraint_id"] for c in review.get("soft_findings", []) if "constraint_id" in c}
    contradicted_facts = [c["fact_id"] for c in parsed.get("state_checks", [])
                          if isinstance(c, dict) and c.get("verdict") == "contradicted"
                          and c["fact_id"] not in soft_facts]
    contradicted_events = [c["constraint_id"] for c in parsed.get("event_checks", [])
                           if isinstance(c, dict) and c.get("verdict") == "contradicted"
                           and c["constraint_id"] not in soft_events]
    # Free-form reviewer prose may quote another branch. Keep it in the audit,
    # but return only structured findings and the already-seen rejected scene.
    codes = []
    base.require(classify_review_failure(review) == "realizer_repair", "reviewer_failure_not_repairable")
    for field, ok in (("new_event_or_progress", True), ("consistent_with_history", True),
                      ("duplicates_existing_event", False)):
        if (type(parsed.get(field)) is bool and parsed[field] is not ok
                and not (field == "consistent_with_history"
                         and review.get("history_consistency_status") == "UNRESOLVED_SOFT_FINDINGS")):
            codes.append(field.upper())
    return {"instruction": "Rewrite this scene. Realize every planned development at its stated action status. For A, substantively advance the selected existing event. For B, instantiate the seed's core premise and selected mapped elements, or the scaffold's complete mapped structure, as a distinct event. Unselected seed steps are optional. Re-read all history and the cited state facts. Resolve contradicted state checks without replaying completed actions or confusing people, times and places; do not change historical facts.",
            "failure_codes": codes, "unmet_development_indices": unmet,
            "route_fulfillment": parsed.get("route_fulfillment"),
            "unmet_card_elements": unmet_card, "contradicted_state_fact_ids": contradicted_facts,
            "contradicted_event_constraint_ids": contradicted_events,
            "previous_response": realization}


def produce_scene(caller, run, root, route, history, cards, scene, other_turns, *, latest_state=None,
                  event_state=None, purpose_input=None):
    repair = None
    scene_id = scene["scene_id"]
    letter = "A" if route == ROUTES[0] else "B"
    audits = []
    for attempt in range(1, SCENE_ATTEMPTS + 1):
        payload, proof = make_request(run, root, route, "realizer", history, cards, scene_id,
                                     scene=scene, repair=repair, latest_state=latest_state, event_state=event_state,
                                     purpose_input=purpose_input)
        raw, directory = caller.call(f"{root}-{letter}-{scene_id}-realizer-{attempt}", payload, proof)
        invalid_speakers = []
        try:
            response = parse_object(raw)
            turns = response.get("turns")
            invalid_speakers = invalid_speaker_details(turns, scene.get("actors", [])) if isinstance(turns, list) else []
            if invalid_speakers:
                raise ValueError("turn_speaker_invalid:" + json.dumps(invalid_speakers, ensure_ascii=False))
            realization = base.validate_realization(response, root, route, scene)
        except (ValueError, TypeError) as error:
            audit = {"status": "REJECT", "stage": "mechanical", "reason": str(error)}
            base.write(directory / "ACCEPTANCE.json", audit); audits.append(audit)
            repair = {"instruction": "Fix this specific response contract error, including every listed turn position and speaker.",
                      "failure": str(error), "invalid_speakers": invalid_speakers,
                      "previous_response": raw}
            continue
        if realization["status"] == "no_scene":
            audit = {"status": "NO_SCENE", "reason": realization["no_scene_reason"]}
            base.write(directory / "ACCEPTANCE.json", audit)
            return None, [*audits, audit]
        checked = duplicate_check.inspect_duplicates(realization["turns"],
            {"branch_history": history, "other_branch_accepted_additions": other_turns})
        base.write(directory / "DUPLICATE_CHECK.json", checked)
        if checked["status"] != "PASS":
            audit = {"status": "REJECT", "stage": "duplicate", "reasons": checked["reasons"],
                     "request": str(directory)}
            base.write(directory / "ACCEPTANCE.json", audit); audits.append(audit)
            repair = repair_for_duplicate(checked, realization)
            continue
        card = cards[0] if route == ROUTES[1] and cards else None
        review, review_dir, raw_review, review_candidate = None, None, None, None
        review_repair = None
        for review_attempt in range(1, REVIEW_RETRIES + 2):
            payload, proof = review_request(root, route, scene, history, realization, other_turns,
                                            card=card, repair=review_repair, latest_state=latest_state,
                                            event_state=event_state, purpose_input=purpose_input)
            raw_review, review_dir = caller.call(
                f"{root}-{letter}-{scene_id}-review-{attempt}-{review_attempt}", payload, proof)
            merge_error = None
            try:
                if review_repair is None:
                    review_candidate = raw_review
                else:
                    merged = realization_review.merge_review_repair(review_candidate, raw_review, review_repair)
                    base.write(review_dir / "FORMAT_REPAIR_MERGE.json", {
                        "status": "PASS", "provenance": "program_merge_of_model_repair_only_no_default_verdicts",
                        "previous_candidate_sha256": base.digest(review_candidate),
                        "raw_repair_sha256": base.digest(raw_review),
                        "repair_spec_sha256": base.digest(review_repair), "candidate": merged})
                    review_candidate = merged
            except ValueError as error:
                merge_error = str(error)
                base.write(review_dir / "FORMAT_REPAIR_MERGE.json", {
                    "status": "REJECT", "reason": str(error),
                    "previous_candidate_sha256": base.digest(review_candidate),
                    "raw_repair_sha256": base.digest(raw_review),
                    "repair_spec_sha256": base.digest(review_repair)})
                review = {"status": "REJECT", "schema_valid": False, "schema_failures": [str(error)],
                          "failures": [str(error)], "parsed": None,
                          "semantic_certified": False, "failure_category": "audit_incomplete"}
            if merge_error is None:
                review = realization_review.validate_review(review_candidate, root, route, scene, realization,
                                                            dataset_card=card, latest_state=latest_state, history=history,
                                                            event_state=event_state,
                                                            require_actor_checks=(route == ROUTES[1]),
                                                            require_goal_checks=(route == ROUTES[1]))
            base.write(review_dir / "REVIEW.json", review)
            if review.get("schema_valid") is not False:
                break
            if review_attempt <= REVIEW_RETRIES:
                review_repair = realization_review.build_format_repair(
                    review, review_candidate, scene, history, realization,
                    dataset_card=card, latest_state=latest_state, event_state=event_state,
                    require_actor_checks=(route == ROUTES[1]), require_goal_checks=(route == ROUTES[1]))
                base.write(review_dir / "FORMAT_REPAIR_SPEC.json", review_repair)
        audit = {"status": review["status"], "stage": "qwen_plan_review", "request": str(directory),
                 "review_request": str(review_dir), "failures": review["failures"],
                 "schema_valid": review.get("schema_valid"), "schema_failures": review.get("schema_failures", []),
                 "soft_findings": review.get("soft_findings", []),
                 "empty_consistent_findings": review.get("empty_consistent_findings", []),
                 "history_consistency_status": review.get("history_consistency_status"),
                 "route_fulfillment": (review.get("parsed") or {}).get("route_fulfillment"),
                 "semantic_certified": False}
        reviewer_failure = review["status"] != "PASS" and classify_review_failure(review) == "reviewer_failure"
        if reviewer_failure:
            audit["status"] = "AUDIT_INCOMPLETE"
            audit["failure_category"] = "audit_incomplete"
            audit["scene_content_failure_established"] = False
        base.write(directory / "ACCEPTANCE.json", audit); audits.append(audit)
        if review["status"] == "PASS":
            return realization, audits
        if reviewer_failure:
            return None, audits
        role_failure = (review.get("parsed") or {}).get("route_fulfillment") or {}
        if route == ROUTES[1] and role_failure and not event_contract.route_fulfilled(role_failure, route):
            audits[-1]["failure_category"] = "plan_role_rejected_after_realization"
            return None, audits
        repair = repair_for_review(review, realization)
    return None, audits


def extract_latest_state(caller, run, root, route, history, cards, scene_id, *, purpose_input=None):
    frozen = branch_history_ok(run, root, route, history)
    letter = "A" if route == ROUTES[0] else "B"
    repair = None
    for attempt in range(1, STATE_ATTEMPTS + 1):
        payload = state_contract.build_state_request(root, route, scene_id, history, cards, repair=repair, purpose_input=purpose_input)
        proof = {"root_id": root, "route": route, "role": "state_extractor", "scene_id": scene_id,
                 **history_proof(run, root, route, history),
                 "purpose_input_sha256": base.digest(purpose_input),
                 "validation_scope": "schema_and_literal_citations_only"}
        raw, directory = caller.call(f"{root}-{letter}-{scene_id}-state-{attempt}", payload, proof)
        candidate = raw
        try:
            if repair is not None and repair.get("repair_mode") == "reference_patch":
                merge_record = {"previous_raw_sha256": base.digest(repair["previous_raw"]),
                    "raw_patch_sha256": base.digest(raw), "repair_spec_sha256": base.digest(repair)}
                try:
                    candidate = state_contract.merge_state_repair(repair["previous_raw"], raw, repair, history)
                except ValueError as error:
                    base.write(directory / "FORMAT_REPAIR_MERGE.json", {
                        **merge_record, "status": "REJECT", "reason": str(error)})
                    base.write(directory / "VALIDATION.json", {
                        "status": "REJECT", "reason": str(error), "errors": [], "merge_failed": True})
                    continue
                base.write(directory / "FORMAT_REPAIR_MERGE.json", {
                    **merge_record, "status": "PASS", "candidate": candidate,
                    "changed_paths": repair["required_paths"], "semantic_certified": False})
            snapshot = state_contract.validate_state(candidate, root, route, scene_id, history)
            base.write(directory / "VALIDATION.json", {
                "status": "PASS", "semantic_certified": False,
                "latest_state_sha256": base.digest(snapshot)})
            return snapshot
        except (ValueError, TypeError) as error:
            repair = state_contract.build_state_repair(error, candidate, history=history)
            base.write(directory / "VALIDATION.json", {
                "status": "REJECT", "reason": str(error), "errors": repair["errors"]})
    return None


def extract_event_state(caller, run, root, route, history, cards, scene, snapshot, plan_attempt, *, purpose_input=None):
    frozen = branch_history_ok(run, root, route, history)
    letter = "A" if route == ROUTES[0] else "B"
    repair = None
    for attempt in range(1, EVENT_STATE_ATTEMPTS + 1):
        payload = event_contract.build_request(root, route, scene, history, cards, snapshot, repair=repair, purpose_input=purpose_input,
                                               require_goal_checks=(route == ROUTES[1]))
        proof = {"root_id": root, "route": route, "role": "event_state_extractor", "scene_id": scene["scene_id"],
                 "plan_attempt": plan_attempt, "scene_plan_sha256": base.digest(scene),
                 "projected_scene_plan_sha256": base.digest(event_contract.scan_plan_context(scene)),
                 "purpose_input_sha256": base.digest(purpose_input),
                 **history_proof(run, root, route, history),
                 "latest_state_sha256": base.digest(snapshot),
                 "require_goal_checks": route == ROUTES[1],
                 "validation_scope": "schema_coverage_literal_citations_and_explicit_goal_policy_not_semantic_certification"}
        raw, directory = caller.call(
            f"{root}-{letter}-{scene['scene_id']}-event-state-{plan_attempt}-{attempt}", payload, proof)
        candidate = raw
        try:
            if repair is not None and repair.get("repair_mode") == "timeline_entry_patch":
                merge_record = {"previous_raw_sha256": base.digest(repair["previous_raw"]),
                    "raw_patch_sha256": base.digest(raw), "repair_spec_sha256": base.digest(repair)}
                try:
                    candidate = event_contract.merge_event_repair(repair["previous_raw"], raw, repair, history)
                except ValueError as error:
                    base.write(directory / "FORMAT_REPAIR_MERGE.json", {
                        **merge_record, "status": "REJECT", "reason": str(error)})
                    base.write(directory / "VALIDATION.json", {
                        "status": "REJECT", "reason": str(error), "errors": [], "merge_failed": True})
                    continue
                base.write(directory / "FORMAT_REPAIR_MERGE.json", {
                    **merge_record, "status": "PASS", "candidate": candidate,
                    "semantic_certified": False})
                base.write(directory / "TIMELINE_PATCH_PROVENANCE.json",
                    event_contract.event_repair_provenance(repair["previous_raw"], raw, repair, candidate))
            if repair is not None:
                event_contract.validate_repair_preservation(repair["previous_raw"], candidate, repair)
                base.write(directory / "FORMAT_REPAIR_PRESERVATION.json", {
                    "status": "PASS", "previous_raw_sha256": base.digest(repair["previous_raw"]),
                    "raw_repair_sha256": base.digest(candidate), "repair_spec_sha256": base.digest(repair),
                    "unchanged_negative_findings_required": True})
            result = event_contract.validate(candidate, root, route, scene, history, require_goal_checks=(route == ROUTES[1]))
            base.write(directory / "EVIDENCE_PROVENANCE.json",
                       evidence_refs.citation_provenance(event_contract._parse(candidate), history))
            base.write(directory / "ROUTE_DECISION_PROVENANCE.json",
                       event_contract.route_decision_provenance(
                           event_contract._parse(candidate)["route_fulfillment"], route,
                           timeline=result["timeline"], require_goal_checks=(route == ROUTES[1])))
            base.write(directory / "DERIVED_COVERAGE.json", event_contract.derive_coverage(result, history))
            base.write(directory / "VALIDATION.json", {
                "status": "PASS", "semantic_certified": False,
                "requires_replan": event_contract.requires_replan(result),
                "plan_diagnostics": event_contract.plan_diagnostics(result),
                "event_state_sha256": base.digest(result)})
            return result
        except (ValueError, TypeError) as error:
            repair = event_contract.build_repair(error, candidate, history=history)
            base.write(directory / "VALIDATION.json", {
                "status": "REJECT", "reason": str(error), "errors": repair["errors"]})
    return None


def run_root(caller, run, root, cards, *, target_scenes=1, routes=None):
    frozen = base.read(run / "sources" / f"{root}.json")["turns"]
    selected_routes = tuple(routes) if routes is not None else ROUTES
    base.require(selected_routes in (ROUTES, (ROUTES[1],)), "invalid_route_selection")
    accepted_by_branch = {"A": [], "B": []}
    results = []
    for route, letter in zip(ROUTES, ("A", "B")):
        if route not in selected_routes:
            continue
        route_base = route_base_history(run, root, route)
        history = deepcopy(route_base)
        # Accepted A is already historical input to B, never a hidden parallel corpus.
        other = []
        folder = run / "roots" / root / letter
        base.write(folder / "INPUT.json", {"root_id": root, "route": route, "turns": history,
                                           **history_proof(run, root, route, history),
                                           "input_turns": len(route_base),
                                           "accepted_A_source_path": str(run / "roots" / root / "A/cumulative_dialogue.json")
                                               if letter == "B" and (run / "roots" / root / "A/cumulative_dialogue.json").exists() else None})
        slots = target_scenes if letter == "A" else min(target_scenes, len(cards))
        accepted, failures = [], []
        for number in range(1, slots + 1):
            scene_id = f"S{number:02d}"
            card = cards[number - 1] if letter == "B" else None
            if letter == "B" and card is None:
                failures.append({"scene_id": scene_id, "reason": "reserved_source_card_extraction_failed"})
                continue
            guidance = [base.dataset_cards.build_realization_guidance(card)] if card else []
            repair, plan, event_snapshot = None, None, None
            plan_failure = "plan_rejected_after_retry"
            try:
                purpose = propose_purpose(caller, run, root, scene_id, guidance[0], history) if letter == "B" else None
                if letter == "B" and purpose is None:
                    failures.append({"scene_id": scene_id, "reason": "purpose_contract_failure",
                                     "failure_category": "plan_failure", "scene_content_failure_established": False})
                    continue
                if purpose is not None:
                    base.write(folder / f"{scene_id}.PURPOSE.json", purpose)
                snapshot = extract_latest_state(caller, run, root, route, history, guidance, scene_id, purpose_input=purpose)
                if snapshot is None:
                    failures.append({"scene_id": scene_id, "reason": "state_audit_incomplete_after_retry",
                                     "failure_category": "audit_incomplete", "scene_content_failure_established": False})
                    continue
                base.write(folder / f"{scene_id}.LATEST_STATE.json", snapshot)
                for attempt in range(1, PLAN_ATTEMPTS + 1):
                    payload, proof = make_request(run, root, route, "planner", history, guidance, scene_id,
                                                 repair=repair, latest_state=snapshot, event_state=event_snapshot, purpose_input=purpose)
                    raw, directory = caller.call(f"{root}-{letter}-{scene_id}-planner-{attempt}", payload, proof)
                    try:
                        candidate = validate_plan(raw, root, route, history, guidance, scene_id, latest_state=snapshot)
                        base.write(directory / "VALIDATION.json", {"status": "PASS"})
                        base.write(directory / "PLAN_CITATION_PROVENANCE.json",
                                   evidence_refs.citation_provenance(parse_object(raw), history))
                    except (ValueError, TypeError) as error:
                        plan_failure = "plan_rejected_after_retry"
                        repair = planner_contract.build_repair_feedback(
                            error, raw, guidance[0] if guidance else None, route=route)
                        base.write(directory / "VALIDATION.json", {
                            "status": "REJECT", "reason": str(error), "errors": repair["errors"]})
                        continue
                    base.write(directory / "CANDIDATE_PLAN.json", candidate)
                    if not candidate["scenes"]:
                        plan = candidate
                        break
                    event_snapshot = extract_event_state(caller, run, root, route, history, guidance,
                                                         candidate["scenes"][0], snapshot, attempt, purpose_input=purpose)
                    if event_snapshot is None:
                        plan_failure = "event_state_audit_incomplete_after_retry"
                        break
                    replan = event_contract.requires_replan(event_snapshot)
                    base.write(directory / "EVENT_STATE_DECISION.json", {
                        "status": "REPLAN" if replan else "PASS",
                        "event_state_sha256": base.digest(event_snapshot),
                        "plan_diagnostics": event_contract.plan_diagnostics(event_snapshot)})
                    if replan:
                        plan_failure = "event_state_requires_replan_after_limit"
                        repair = {"instruction": "Revise the complete plan to resolve every revise finding. "
                                  "The historical timeline and uncertainty constrain the new event; do not rewrite "
                                  "history or repeat an already completed first occurrence. Keep the assigned "
                                  "card: realize a seed's core premise and selected mappings, or a scaffold's "
                                  "complete structure. Unselected seed steps are optional. The revised plan will be scanned again.",
                                  "failure": "event_state_requires_replan", "previous_response": candidate,
                                  "plan_checks": event_contract.binding_plan_checks(event_snapshot)}
                        continue
                    plan = candidate
                    base.write(folder / f"{scene_id}.EVENT_STATE.json", event_snapshot)
                    base.write(folder / f"{scene_id}.EVENT_COVERAGE.json",
                               event_contract.derive_coverage(event_snapshot, history))
                    break
                if plan is None:
                    failures.append({"scene_id": scene_id, "reason": plan_failure,
                                     "failure_category": "audit_incomplete" if "audit_incomplete" in plan_failure else "plan_failure",
                                     "scene_content_failure_established": False}); continue
                base.write(folder / f"{scene_id}.PLAN.json", plan)
                if not plan["scenes"]:
                    failures.append({"scene_id": scene_id, "reason": plan["shortfall_reason"]}); continue
                realization, audits = produce_scene(caller, run, root, route, history, guidance,
                                                    plan["scenes"][0], other, latest_state=snapshot,
                                                    event_state=event_snapshot, purpose_input=purpose)
                base.write(folder / f"{scene_id}.ATTEMPTS.json", audits)
                if realization is None:
                    incomplete = bool(audits and audits[-1].get("failure_category") == "audit_incomplete")
                    failures.append({"scene_id": scene_id,
                        "reason": "audit_incomplete" if incomplete else "no_accepted_scene",
                        "failure_category": "audit_incomplete" if incomplete else "scene_not_accepted",
                        "scene_content_failure_established": not incomplete and bool(audits), "attempts": audits}); continue
                base.write(folder / f"{scene_id}.json", realization)
                history = base.append_scene(history, root, route, realization)
                accepted_by_branch[letter] = history[len(route_base):]
                accepted.append({"scene_id": scene_id, "dataset_card_id": card["card_id"] if card else None,
                                 "turns": len(realization["turns"]), "duplicate_check": "PASS", "qwen_review": "PASS",
                                 "purpose_input_sha256": base.digest(purpose),
                                 "latest_state_sha256": base.digest(snapshot), "qwen_state_checks": "PASS",
                                 "event_state_sha256": base.digest(event_snapshot), "qwen_event_checks": "PASS",
                                 "soft_findings": audits[-1].get("soft_findings", []),
                                 "empty_consistent_findings": audits[-1].get("empty_consistent_findings", []),
                                 "history_consistency_status": audits[-1].get("history_consistency_status"),
                                 "route_fulfillment": audits[-1].get("route_fulfillment"),
                                 "plan_diagnostics": event_contract.plan_diagnostics(event_snapshot),
                                 "semantic_certified": False})
            except RuntimeError as error:
                if str(error) != "provider_request_failed_after_retry_policy":
                    raise
                failures.append({"scene_id": scene_id, "reason": str(error),
                                 "per_request_automatic_retry_limit": TRANSPORT_RETRIES,
                                 "failure_category": "transport_failure", "remaining_slots_continue": True})
                continue
        branch_history_ok(run, root, route, history)
        result = {"root_id": root, "route": route, "target_scenes": target_scenes,
                  "available_slots": slots, "accepted_scenes": len(accepted),
                  "scene_metadata": accepted, "failures": failures,
                  "shortfall_scenes": target_scenes - len(accepted),
                  "added_turns": len(history) - len(route_base), "input_turns": len(route_base),
                  "branch_base_sha256": base.digest(route_base),
                  "route_input_history_sha256": base.digest(route_base),
                  "frozen_base_sha256": base.digest(frozen),
                  "output_history_sha256": base.digest(history),
                  "accepted_A_in_input": letter == "B" and len(route_base) > len(frozen),
                  "other_branch_in_generator_context": letter == "B" and len(route_base) > len(frozen),
                  "no_available_guidance": letter == "B" and slots == 0,
                  "acceptance": "event_history_scan_and_exact_duplicate_check_and_same_qwen_event_state_card_plan_review",
                  "semantic_certified": False}
        base.write(folder / "RESULT.json", result)
        base.write(folder / "cumulative_dialogue.json", {"root_id": root, "route": route, "turns": history,
            "version": VERSION, "branch_base_sha256": base.digest(route_base), "model": MODEL,
            "frozen_base_sha256": base.digest(frozen),
            "sequential_root_history": True, "accepted_A_in_input": letter == "B" and len(route_base) > len(frozen)})
        results.append(result)
    merged_path = run / "sources_merged" / f"{root}.json"
    original = base.read(run / "sources" / f"{root}.json")
    inherited = base.read(run / "INHERITED_A.json") if (run / "INHERITED_A.json").exists() else None
    inherited_count = inherited["accepted_scenes"] if inherited else 0
    source = {"root_id": root, "turns": history, "turn_count": len(history), "version": VERSION, "model": MODEL,
              "frozen_source_metadata": {key:value for key,value in original.items() if key!='turns'},
              "accepted_scene_order": list(dict.fromkeys(t['turn_ref'].rsplit(':',1)[0] for t in history[len(frozen):])),
              "source_context_policy": "frozen_base_then_accepted_A_then_accepted_B",
              "base_occurrences": 1, "frozen_base_sha256": base.digest(frozen),
              "accepted_scenes": inherited_count + sum(row["accepted_scenes"] for row in results),
              "inherited_A_scenes": inherited_count, "new_accepted_scenes": sum(row["accepted_scenes"] for row in results),
              "scene_shortfall": {"A" if row["route"] == ROUTES[0] else "B": row["shortfall_scenes"]
                                  for row in results},
              "generation_is_cumulative_not_cross_scene_event_order_proof": True}
    base.write(merged_path, source)
    base.write(run / "roots" / root / "BRANCHES.json", {
        "root_id": root, "shared_base_sha256": base.digest(frozen), "independent_branches": False,
        "sequential_routes": True, "sequence": "frozen_base -> accepted_A -> accepted_B",
        "dialogues": {letter: str(run / "roots" / root / letter / "cumulative_dialogue.json")
                      for route, letter in zip(ROUTES, ("A", "B")) if route in selected_routes},
        "merged_dialogue": str(merged_path), "merged_sha256": base.digest(merged_path.read_bytes()),
        "downstream_requires_merged_root_source": True})
    return results


def prepare(run, *, scenes=3, roots=None, routes=None):
    base.require(not run.exists(), "run_already_exists")
    base.require(type(scenes) is int and 1 <= scenes <= 3, "scenes_outside_1_3")
    base.require(run.parent == HERE / "runs", "run_must_be_in_v32_inventory")
    roots = list(DEFAULT_ROOTS if roots is None else roots)
    routes = tuple((ROUTES[1],) if routes is None else routes)
    base.require(roots and len(set(roots)) == len(roots) and all(root in ROOTS for root in roots),
                 "invalid_root_selection")
    base.require(routes in (ROUTES, (ROUTES[1],)), "invalid_route_selection")
    base.require(roots == list(DEFAULT_ROOTS) and routes == (ROUTES[1],) and scenes == 3,
                 "v32_bounded_probe_requires_104814_B3")
    parent_inputs = checked_parent_a(PARENT_RUN, roots[0])
    # Arbitrary manifests could relabel a consumed event and recompute their
    # own hashes. Only use the source-bound local freezer for this version.
    prior = source_inventory.collect_prior()
    consumed = {source_inventory.event_key(row) for row in prior["accepted_uses"]}
    sources = source_pool.freeze_unused_source_subset(
        consumed, count=scenes * len(roots)
    )
    expected = sources["manifest_sha256"]
    base.require(base.dataset_cards._sha(base.dataset_cards._canonical(
        {k: v for k, v in sources.items() if k != "manifest_sha256"})) == expected, "dataset_manifest_hash_mismatch")
    for row in sources["files"]:
        for key, digest_key in (("local_path", "local_sha256"), ("source_receipt_path", "source_receipt_sha256")):
            base.require(base.digest(Path(row[key]).read_bytes()) == row[digest_key], "dataset_file_changed")
    source_ids = [item.get("source_id") for item in sources["items"]]
    base.require(all(isinstance(x, str) and x for x in source_ids)
                 and len(source_ids) == len(set(source_ids)), "source_ids_missing_or_duplicate")
    file_hashes = {row["local_sha256"] for row in sources["files"]}
    base.require(all(item["source_file_sha256"] in file_hashes for item in sources["items"]),
                 "source_item_not_bound_to_verified_file")
    allocation = source_inventory.legacy.allocate(sources, roots, prior, scenes)
    run.mkdir(parents=True)
    base.feedback.compile_feedback(run / "feedback", roots)
    base.write(run / "PURPOSE_FEEDBACK_PROJECTION.json",
               purpose_contract.project_feedback(run / "feedback", roots[0], base.feedback))
    original_sources = []
    for root in roots:
        shard, sha = SOURCE_SPECS[root]
        path = base.PROJECT / f"restart-n5-runs/h4_unexpanded122_memory_event_k3_v1/shards/shard-{shard}/expansion/roots/{root}/promotion/promotable_expanded_dialogue.json"
        base.require(base.digest(path.read_bytes()) == sha, "frozen_R01_source_changed")
        base.write(run / "sources" / f"{root}.json", base.read(path)); original_sources.append(path)
    import_parent_a(run, roots[0], parent_inputs)
    base.write(run / "dataset_sources.json", sources)
    base.write(run / "UNIQUE_EVENT_ALLOCATION.json", allocation)
    previews = []
    for root in roots:
        history = route_base_history(run, root, ROUTES[1])
        for route in routes:
            mock_cards = [] if route == ROUTES[0] else [{
                "card_id": "OFFLINE_PREVIEW_ONLY", "guidance_level": "seed", "actors": [],
                "steps": [{"step_id": "s1", "actor_id": "actor_1",
                           "action": "OFFLINE_PREVIEW_ONLY", "state": "proposed"}],
                "dependencies": [], "confusion_axes": [], "open_ends": [],
                "downgrade_reason": "OFFLINE_PREVIEW_ONLY"}]
            mock_purpose = None if route == ROUTES[0] else {
                "root_id": root, "route": route, "scene_id": "S01",
                "dataset_card_id": mock_cards[0]["card_id"],
                "trigger": "OFFLINE_PREVIEW_ONLY", "goal": "OFFLINE_PREVIEW_ONLY"}
            mock_state = {"root_id": root, "route": route, "scene_id": "S01", "facts": [{
                "fact_id": "f1", "entity": "OFFLINE_PREVIEW_ONLY", "event": "OFFLINE_PREVIEW_ONLY",
                "status": "uncertain", "evidence_basis": "uncertain", "statement": "OFFLINE_PREVIEW_ONLY",
                "evidence": [{"turn_ref": history[-1]["turn_ref"], "quote": history[-1]["text"]}]}],
                "conflicts": [], "unknowns": []}
            state_payload = state_contract.build_state_request(root, route, "S01", history, mock_cards,
                                                               purpose_input=mock_purpose)
            base.write(run / "previews" / root / f"{route}.state_extractor.json", {
                "payload": state_payload, "proof": {"mock_only_offline_preview": True,
                "role": "state_extractor", "purpose_input_sha256": base.digest(mock_purpose),
                "live_B_history_will_include_accepted_A": route == ROUTES[1]}})
            for role in ("planner", "realizer"):
                scene = None
                if role == "realizer":
                    scene = {"scene_id": "S01", "actors": sorted({turn["speaker"] for turn in history}),
                        "source_anchors": [{"turn_ref": history[-1]["turn_ref"], "quote": history[-1]["text"][:500]}]
                            if route == ROUTES[0] else [],
                        "goal": "OFFLINE_PREVIEW_ONLY", "developments": ["OFFLINE_PREVIEW_STEP_1", "OFFLINE_PREVIEW_STEP_2"],
                        "dataset_card_id": mock_cards[0]["card_id"] if mock_cards else None,
                        "card_development_links": [{"card_step_ids": ["s1"], "dependency_indices": [],
                                                    "development_indices": [1, 2]}] if mock_cards else [],
                        "feedback_topic_ids": [],
                        "event_definition": {"trigger": "OFFLINE_PREVIEW_ONLY",
                                             "history_relation": "continuation" if route == ROUTES[0] else "distinct_event"}}
                if scene:
                    scene["state_transition"] = {"basis_fact_ids": ["f1"] if route == ROUTES[0] else [],
                        "new_event": "OFFLINE_PREVIEW_ONLY", "temporal_placement": "OFFLINE_PREVIEW_ONLY"}
                    if route == ROUTES[1]:
                        scene["state_transition"]["basis_reason"] = "OFFLINE_PREVIEW_ONLY: card defines the new goal."
                    validate_plan({"root_id": root, "route": route, "scenes": [scene], "shortfall_reason": ""},
                                  root, route, history, mock_cards, "S01", latest_state=mock_state)
                payload, proof = make_request(run, root, route, role, history, mock_cards, "S01",
                                             scene=scene, latest_state=mock_state, purpose_input=mock_purpose)
                proof["mock_only_offline_preview"] = True
                base.write(run / "previews" / root / f"{route}.{role}.json", {"payload": payload, "proof": proof})
                previews.append(proof)
    files = list(HERE.glob("*.py")) + list(LEGACY.glob("*.py")) + [HERE / "POLICY.md"]
    provider_profile, transport_code, tokenizer_files = transport_inputs()
    base.write(run / "provider_profile.json", provider_profile)
    files += transport_code + tokenizer_files
    files += [HERE.parent / "h4_r02_source_unique_v2/unique_events.py", *original_sources, *parent_inputs["paths"]]
    files += [Path(x["path"]) for x in prior["prior_evidence"]]
    files += [Path(row[k]) for row in sources["files"] for k in ("local_path", "source_receipt_path")]
    files += list(run.rglob("*.json")) + list(run.rglob("*.txt"))
    logical_budget = (((scenes * len(roots) if ROUTES[0] in routes else 0) + allocation["assigned_slots"])
                      * (STATE_ATTEMPTS + PLAN_ATTEMPTS * (1 + EVENT_STATE_ATTEMPTS)
                         + SCENE_ATTEMPTS * (REVIEW_RETRIES + 2))
                      + allocation["assigned_slots"] * (CARD_EXTRACTION_ATTEMPTS + PURPOSE_ATTEMPTS))
    maximum = logical_budget * (TRANSPORT_RETRIES + 1)
    base.require(maximum <= 540, "bounded_probe_budget_exceeded")
    manifest = {"version": VERSION, "model": MODEL, "review_model": MODEL, "target_solver_model": base.TARGET,
        "roots": roots, "selected_routes": list(routes), "scenes_per_route_target": scenes,
        "sequence": "P+R01 -> A acceptance -> P+R01+accepted_A -> B acceptance -> one cumulative root",
        "route_context": "Every B planner/state scan/realizer/repair/reviewer sees full frozen base plus accepted A; failed A scenes excluded",
        "purpose_attempts_per_slot": PURPOSE_ATTEMPTS, "purpose_input_scope": "card_allowed_speakers_frozen_feedback_only",
        "inherited_A": base.read(run / "INHERITED_A.json"),
        "planner_contract_version": "v25_single_goal_purpose_then_full_history",
        "history_explanation_field": "state_transition.basis_reason",
        "guidance_policy": "seed_premise_selected_steps;scaffold_all_steps_and_dependencies",
        "route_roles": {"A": "advance_existing_event", "B": "distinct_dataset_guided_event_not_old_event_logistics"},
        "max_planner_attempts": PLAN_ATTEMPTS, "max_realizer_attempts": SCENE_ATTEMPTS,
        "max_state_extraction_attempts": STATE_ATTEMPTS,
        "max_event_state_attempts_per_plan": EVENT_STATE_ATTEMPTS,
        "state_model": MODEL, "state_version": state_contract.STATE_VERSION,
        "evidence_basis": "reported|inferred|uncertain; reported plans remain planned; soft-only conflicts are diagnostics",
        "question_generation_stage": "not launched; MERGE_MANIFEST schema is v17-loader-compatible; requires new sequential-context prompt adapter before future multi_select/ordering generation",
        "event_coverage_version": event_contract.EVENT_STATE_VERSION,
        "B_goal_checks_required": True, "B_actor_checks_required": True,
        "semantic_validation_scope": "same_Qwen_explicit_per_goal_and_actor_judgments_with_mechanical_policy;not_certification",
        "cited_segment_none_reason": "preserve_nonempty_text_as_diagnostic_only",
        "state_pipeline": "Qwen card-first purpose -> full-history state -> Planner -> card/purpose/history scan / bounded replan -> Realizer -> same-Qwen review",
        "automatic_transport_retries": TRANSPORT_RETRIES,
        "minimum_http_gap_after_response_seconds": MIN_HTTP_GAP_SECONDS,
        "transport_dependency_paths": [str(path) for path in transport_code],
        "capacity_tokenizer_paths": [str(path) for path in tokenizer_files],
        "provider_profile_path": str(run / "provider_profile.json"),
        "credential_contents_frozen": False,
        "retryable_transport_statuses": [503, 504],
        "card_extraction_attempts_per_source": CARD_EXTRACTION_ATTEMPTS,
        "review_schema_retries": REVIEW_RETRIES, "http_max": maximum, "files": base.bindings(files),
        "live_ready": ROUTES[0] in routes or allocation["assigned_slots"] > 0,
        "route_readiness": {"A": ROUTES[0] in routes, "B": allocation["assigned_slots"] > 0},
        "blocking_reason": None if allocation["assigned_slots"] else "B_NO_UNUSED_DATASET_EVENTS",
        "prepared_at_utc": now(), "generation_calls": 0, "solver_calls": 0, "training_started": False}
    base.write(run / "MANIFEST.json", manifest)
    snapshots = []
    for index, row in enumerate(manifest["files"]):
        path = Path(row["path"])
        if path.suffix != ".py":
            continue
        destination = run / "execution_code" / f"{index:03d}_{path.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        base.require(base.digest(destination.read_bytes()) == row["sha256"], "snapshot_hash_mismatch")
        snapshots.append({"original_path": row["path"], "snapshot_path": str(destination),
                          "sha256": row["sha256"]})
    base.write(run / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    result = {"status": "PREPARED" if allocation["assigned_slots"] else (
                  "PREPARED_A_ONLY_NO_UNUSED_DATASET_EVENTS" if ROUTES[0] in routes else "BLOCKED_NO_UNUSED_DATASET_EVENTS"),
              "assigned_B_slots": allocation["assigned_slots"], "excluded_source_ids": allocation["excluded_prior_source_ids"],
              "sequential_route_previews": len(previews), "preview_B_uses_bound_accepted_A": True,
              "all_feedback_receipts_present": all(p["feedback_receipt"] for p in previews),
              "live_ready": manifest["live_ready"], "route_readiness": manifest["route_readiness"], "http_calls": 0}
    base.write(run / "PREPARE_CHECKS.json", result)
    return result


def verify(run, *, live_execution=False):
    manifest = base.read(run / "MANIFEST.json")
    base.require(manifest["version"] == VERSION, "wrong_version")
    base.verify_bindings(manifest["files"])
    verify_inherited_a(run)
    bound = {row["path"] for row in manifest["files"]}
    base.require(manifest.get("transport_dependency_paths") and manifest.get("capacity_tokenizer_paths"),
                 "transport_binding_manifest_missing")
    required = [*manifest["transport_dependency_paths"], *manifest["capacity_tokenizer_paths"],
                str(run / "provider_profile.json")]
    base.require(all(str(Path(path).resolve()) in bound for path in required), "transport_inputs_not_bound")
    profile, _factory, _annotate, _capacity, _dev = base.transport()
    base.require(profile() == base.read(run / "provider_profile.json"), "provider_profile_changed")
    allocation = base.read(run / "UNIQUE_EVENT_ALLOCATION.json")
    source_inventory.legacy.verify_allocation(allocation, base.read(run / "dataset_sources.json"), manifest["roots"])
    return manifest, allocation


def live(run):
    with LOCK_PATH.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest, allocation = verify(run, live_execution=True)
        base.require(manifest["live_ready"], "NO_UNUSED_DATASET_EVENTS")
        base.require(not (run / "LIVE_STARTED.json").exists(), "already_started_no_automatic_replay")
        source_inventory.reserve(run, allocation["assignments"], source_inventory.collect_prior())
        base.write(run / "LIVE_STARTED.json", {"version": VERSION, "started_at": now(), "http_max": manifest["http_max"]})
        caller = Caller(run)
        sources = base.read(run / "dataset_sources.json")
        selected_ids = {x["source_id"] for rows in allocation["assignments"].values() for x in rows}
        cards, rejected, extraction_audits = [], [], []
        # One source per extraction avoids cross-batch card-ID collisions.
        for item in sources["items"]:
            if item["source_id"] not in selected_ids:
                continue
            subset = base.dataset_cards.select_source_subset(sources, [item["source_id"]])
            repair, selected = None, None
            source_audits = []
            for extract_attempt in range(1, CARD_EXTRACTION_ATTEMPTS + 1):
                payload = card_selection.build_extraction_request(subset, repair=repair)
                try:
                    raw, directory = caller.call(
                        f"extract-{item['source_id'].replace(':', '-')}-{extract_attempt}", payload,
                        {"role": "dataset_extractor", "source_id": item["source_id"],
                         "source_event_identity": source_inventory.event_key(item)})
                except RuntimeError as error:
                    source_audits.append({"attempt": extract_attempt, "status": "TRANSPORT_FAILURE",
                                          "reason": str(error)})
                    break
                checked = card_selection.select_cards(raw, subset)
                base.write(directory / "VALIDATION.json", checked)
                source_audits.append({"attempt": extract_attempt, "request": str(directory),
                                      "accepted_cards": len(checked["accepted_cards"]),
                                      "retryable": checked["retryable"]})
                rejected.extend(checked["rejected_records"])
                if checked["accepted_cards"]:
                    selected = checked["accepted_cards"][0]
                    cards.append(selected)
                    break
                if not checked["retryable"]:
                    break
                repair = checked["repair_feedback"]
            if selected is None:
                rejected.append({"scope": "source", "source_id": item["source_id"],
                                 "reason": "no_usable_card_after_bounded_extraction",
                                 "attempts": source_audits})
            extraction_audits.append({"source_id": item["source_id"], "attempts": source_audits,
                                      "selected_card_id": selected["card_id"] if selected else None})
        base.write(run / "dataset_cards_result.json", {"accepted_cards": cards,
            "rejected_records": rejected, "source_attempts": extraction_audits,
            "one_card_per_source": True, "selection_policy": card_selection.SELECTION_POLICY})
        by_source = {c["source_id"]: c for c in cards}
        # Preserve reservation slots even if an earlier extraction fails.
        rows, uses = [], []
        for root in manifest["roots"]:
            assigned = allocation["assignments"][root]
            # Keep missing slots as None. A later valid card retains its original
            # reserved S02/S03 identity and can still be used in this branch.
            root_cards = [by_source.get(x["source_id"]) for x in assigned]
            source_inventory.verify_cards_for_root([c for c in root_cards if c is not None], allocation, root)
            missing = [f"S{i:02d}" for i, card in enumerate(root_cards, 1) if card is None]
            if missing:
                base.write(run / "roots" / root / "CARD_SHORTFALL.json", {
                    "reason": "card_extraction_incomplete_only_missing_slots_skipped", "missing_scene_ids": missing})
            root_rows = run_root(caller, run, root, root_cards,
                                 target_scenes=manifest["scenes_per_route_target"], routes=manifest["selected_routes"])
            rows.extend(root_rows)
            b_row = next(row for row in root_rows if row["route"] == ROUTES[1])
            for scene in b_row["scene_metadata"]:
                card = root_cards[int(scene["scene_id"][1:]) - 1]
                key = source_inventory.event_key(card)
                uses.append({"dataset": key[0], "split": key[1], "item_id": key[2], "source_id": card["source_id"],
                             "source_item_sha256": card["lineage"]["source_item_sha256"],
                             "root_id": root, "scene_id": scene["scene_id"], "run": str(run)})
        base.verify_bindings(manifest["files"])
        actual_http = len(list((run / "requests").glob("*/HTTP_ATTEMPT*.json")))
        inherited = base.read(run / "INHERITED_A.json")
        output = {"version": VERSION,
                  "inherited_A_scenes": inherited["accepted_scenes"], "new_A_calls": 0,
                  "status": "COMPLETE_WITH_SHORTFALL" if any(r["shortfall_scenes"] for r in rows) else "COMPLETE",
                  "routes": rows, "accepted_uses": uses,
                  "accepted_scenes": sum(r["accepted_scenes"] for r in rows), "http_attempts": actual_http,
                  "http_request_reservations": caller.request_counter,
                  "http_attempt_log_count": caller.http_counter,
                  "branches_remain_separate": False, "sequence": "base+accepted_A+accepted_B",
                  "model": MODEL, "question_calls": 0, "solver_calls": 0,
                  "training_started": False, "semantic_certified": False}
        base.write(run / "REPORT.json", output)
        source_rows = []
        for root in manifest["roots"]:
            path = run / "sources_merged" / f"{root}.json"
            root_rows = [row for row in rows if row["root_id"] == root]
            source_rows.append({"root_id": root, "path": str(path),
                "sha256": base.digest(path.read_bytes()), "status": "PASS",
                "accepted_scenes": inherited["accepted_scenes"] + sum(row["accepted_scenes"] for row in root_rows),
                "new_accepted_scenes": sum(row["accepted_scenes"] for row in root_rows),
                "shortfall_scenes": sum(row["shortfall_scenes"] for row in root_rows),
                "accepted_A_scenes": inherited["accepted_scenes"],
                "inherited_A_provenance_sha256": base.digest(inherited),
                "accepted_B_scenes": sum(row["accepted_scenes"] for row in root_rows if row["route"] == ROUTES[1]),
                "pass_scope": "frozen_prefix_and_only_accepted_sequential_scenes;shortfall_is_explicit",
                "semantic_certified": False})
        merge_files = [run / "REPORT.json", run / "SOURCE_RESERVATIONS.json", run / "INHERITED_A.json",
                       Path(inherited["local_path"])]
        merge_files += [run / "roots" / root / "BRANCHES.json" for root in manifest["roots"]]
        merge_files += [Path(row["path"]) for row in source_rows]
        base.write(run / "MERGE_MANIFEST.json", {"version": VERSION, "status": "PASS",
            "created_at_utc": now(), "sequence": "frozen_base -> accepted_A -> accepted_B",
            "source_rows": source_rows, "files": base.bindings(merge_files),
            "partial_generation_allowed": True, "semantic_certified": False,
            "question_calls": 0, "solver_calls": 0, "training_started": False})
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify", "live"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--roots", nargs="+", choices=ROOTS)
    parser.add_argument("--routes", choices=("both", "B"), default="B")
    args = parser.parse_args()
    run = args.run.resolve()
    result = prepare(run, scenes=args.scenes, roots=args.roots,
                     routes=(ROUTES[1],) if args.routes == "B" else ROUTES) if args.action == "prepare" else (
        live(run) if args.action == "live" else {"status": "PASS", "live_ready": verify(run)[0]["live_ready"]})
    print(json.dumps(result, ensure_ascii=False, indent=2))
