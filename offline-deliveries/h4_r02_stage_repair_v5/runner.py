"""Independent R02 branches with bounded Qwen scene repair and immutable output."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

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

VERSION = "h4_r02_stage_repair_v5"
MODEL = base.dataset_cards.MODEL
SCENE_ATTEMPTS = 2
PLAN_ATTEMPTS = 2
TRANSPORT_RETRIES = 1
REVIEW_RETRIES = 1
CARD_EXTRACTION_ATTEMPTS = 2
ROUTES = base.ROUTES
LOCK_PATH = HERE.parent / "h4_r02_source_unique_v2/runs/.unique_event_live.lock"

PLANNER = """Plan at most ONE new scene after this branch's cumulative_history.
This is one scene slot, not a request to fill three scenes. Return exactly
{root_id,route,scenes,shortfall_reason}. scenes has zero or one entry. For zero
scenes explain why; for one scene shortfall_reason may be empty.
The scene has exactly {scene_id,actors,source_anchors,goal,developments,
dataset_card_id,card_development_links,feedback_topic_ids}. scene_id MUST equal target_scene_id.
actors: at least two distinct allowed_speakers. source_anchors: 1-3 exact
{turn_ref,quote} entries from cumulative_history. goal: concrete short string.
developments: 2-5 concrete achievable new changes, decisions or outcomes.
Only scene actors may speak directly in the later dialogue. Make every planned
development expressible by these actors. They may accurately report a third
party's action, but never require that third party to speak or join this scene.
feedback_topic_ids: at most four IDs from supplied previous-round feedback.
For old_event_continuation, progress an existing event; dataset_card_id is null
and card_development_links is empty.
For dataset_new_event, introduce a distinct new event from the single supplied
dataset card; dataset_card_id MUST be its card_id. Transfer a scaffold's
relationships/conditions or a seed's premise, never source identities as facts.
card_development_links is a list of {card_step_ids,dependency_indices,
development_indices}. Map every card step and every dependency to one or more
1-based scene development indices. A dependency link must point to a development
that expresses its relation, not merely repeat the two actions. For a seed with
no dependencies, dependency_indices must be empty and the premise still needs
at least one mapped development.
Keep proposals, agreements, attempts and outcomes distinct. A plan need not
force all actions to finish, but every listed development must appear in the
new dialogue. Do not merely rephrase old events or create reassurance filler.
The other expansion branch is not part of this branch's history.
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def branch_history_ok(run, root, route, history):
    frozen = base.read(run / "sources" / f"{root}.json")["turns"]
    base.require(history[:len(frozen)] == frozen, "frozen_branch_prefix_changed")
    letter = "A" if route == ROUTES[0] else "B"
    suffix = history[len(frozen):]
    base.require(all(t.get("turn_ref", "").startswith(f"R02:{letter}:") for t in suffix),
                 "other_branch_leaked_into_history")
    base.require(len({t["turn_ref"] for t in history}) == len(history), "duplicate_branch_turn_ref")
    return frozen


def make_request(run, root, route, role, history, cards, scene_id, *, scene=None, repair=None):
    base.require(route in ROUTES and role in ("planner", "realizer"), "invalid_request_role")
    frozen = branch_history_ok(run, root, route, history)
    base.require((route == ROUTES[0] and not cards) or (route == ROUTES[1] and len(cards) == 1),
                 "one_card_per_B_slot_required")
    context = {"root_id": root, "route": route, "role": role, "target_scene_id": scene_id,
               "allowed_speakers": sorted({t["speaker"] for t in frozen}),
               "branch_base_sha256": base.digest(frozen), "cumulative_history": history,
               "dataset_cards": cards, "scene_plan": scene}
    if repair:
        context["targeted_repair"] = repair
    prompt = base.COMMON + (PLANNER if role == "planner" else base.REALIZER)
    prompt += "\nOnly this branch's accepted history is authoritative. Add new progress; do not copy old blocks.\n"
    prompt += "\nINPUT_JSON\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    payload = {"model": MODEL, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
               "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]}
    identity = base.read(run / "feedback/COMPILED_FEEDBACK.json")["target_solver_identity_sha256"]
    payload, receipt = base.feedback.append_feedback_to_request(
        payload, run / "feedback", root, route, role, target_solver_model=base.TARGET,
        expected_target_identity_sha256=identity)
    proof = {"root_id": root, "route": route, "role": role, "scene_id": scene_id,
             "branch_base_sha256": base.digest(frozen), "history_sha256": base.digest(history),
             "history_turns": len(history), "branch_isolation_verified": True,
             "other_branch_in_generator_context": False, "feedback_receipt": receipt,
             "payload_sha256": base.digest(payload)}
    return payload, proof


def validate_plan(raw, root, route, history, cards, scene_id):
    plan = parse_object(raw)
    scenes = plan.get("scenes")
    base.require(isinstance(scenes, list) and len(scenes) <= 1, "one_scene_slot_only")
    if scenes:
        base.require(isinstance(scenes[0], dict), "scene_not_object")
        base.require(scenes[0].get("scene_id") == scene_id, "target_scene_id_mismatch")
        if route == ROUTES[1]:
            validate_card_plan_fidelity(scenes[0], cards[0])
    # Reuse legacy structural/evidence checks on a validation-only view. Never
    # change the actual model response or its accepted plan.
    view = deepcopy(plan)
    if scenes:
        view["scenes"][0]["scene_id"] = "S01"
        base.require(isinstance(view.get("shortfall_reason"), str), "shortfall_reason_invalid")
        view["shortfall_reason"] = view["shortfall_reason"] or "single-scene slot"
    if scenes:
        view["scenes"][0].pop("card_development_links", None)
    base.validate_plan(view, root, route, history, cards)
    return plan


def validate_card_plan_fidelity(scene, card):
    """Require an explicit, complete mapping from the assigned card to plan steps."""
    base.require(scene.get("dataset_card_id") == card.get("card_id"), "dataset_card_id_mismatch")
    links = scene.get("card_development_links")
    base.require(isinstance(links, list) and links, "card_development_links_missing")
    developments = scene.get("developments")
    base.require(isinstance(developments, list) and developments, "scene_developments_missing")
    steps = card.get("steps")
    dependencies = card.get("dependencies")
    base.require(isinstance(steps, list) and steps, "card_steps_missing")
    base.require(isinstance(dependencies, list), "card_dependencies_invalid")
    step_ids = {step.get("step_id") for step in steps if isinstance(step, dict)}
    base.require(len(step_ids) == len(steps) and None not in step_ids, "card_step_ids_invalid")
    link_step_ids, link_dependency_indices, mapped_developments = set(), set(), set()
    for link in links:
        base.require(isinstance(link, dict) and set(link) == {
            "card_step_ids", "dependency_indices", "development_indices"
        }, "card_development_link_fields_invalid")
        linked_steps = link["card_step_ids"]
        linked_dependencies = link["dependency_indices"]
        linked_developments = link["development_indices"]
        base.require(isinstance(linked_steps, list) and linked_steps
                     and all(isinstance(x, str) and x in step_ids for x in linked_steps),
                     "card_development_link_steps_invalid")
        base.require(isinstance(linked_dependencies, list)
                     and all(type(x) is int and 1 <= x <= len(dependencies) for x in linked_dependencies),
                     "card_development_link_dependencies_invalid")
        base.require(isinstance(linked_developments, list) and linked_developments
                     and all(type(x) is int and 1 <= x <= len(developments) for x in linked_developments),
                     "card_development_link_indices_invalid")
        base.require(not (link_step_ids & set(linked_steps)), "card_step_mapped_more_than_once")
        base.require(not (link_dependency_indices & set(linked_dependencies)), "card_dependency_mapped_more_than_once")
        link_step_ids.update(linked_steps)
        link_dependency_indices.update(linked_dependencies)
        mapped_developments.update(linked_developments)
    base.require(link_step_ids == step_ids, "card_steps_not_fully_mapped")
    base.require(link_dependency_indices == set(range(1, len(dependencies) + 1)),
                 "card_dependencies_not_fully_mapped")
    base.require(mapped_developments, "card_premise_not_mapped_to_development")
    return True


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
            (item.startswith("development[") and ":not_realized:" in item))
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


class Caller:
    """Retry one explicit transient provider status with the identical payload."""
    def __init__(self, run):
        self.run = run
        self.limit = base.read(run / "MANIFEST.json")["http_max"]
        profile, factory, annotate, capacity, dev = base.transport()
        self.profile, self.factory = profile(), factory
        self.annotate, self.capacity = annotate, capacity
        self.tokenizer = dev.tokenizer_from(str(base.TOKENIZER))
        self.http_counter = 0
        self.request_counter = 0

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
        capacity = self.annotate(self.capacity(self.profile, self.tokenizer, payload["messages"], 4096))
        base.require(capacity["status"] == "PASS", "capacity_blocked_no_truncation")
        self.request_counter += 1
        directory = self.run / "requests" / f"{self.request_counter:03d}-{label}"
        base.write(directory / "REQUEST.json", payload)
        base.write(directory / "PROOF.json", {**proof, "capacity": capacity, "payload_sha256": base.digest(payload)})
        base.write(directory / "RESERVATION.json", {"request_number": self.request_counter,
            "http_max": self.limit, "max_attempts": TRANSPORT_RETRIES + 1,
            "retryable_statuses": [503, 504], "status": "RESERVED"})
        attempts = []
        for attempt in range(1, TRANSPORT_RETRIES + 2):
            attempt_path = directory / ("HTTP_ATTEMPT.json" if attempt == 1 else f"HTTP_ATTEMPT_{attempt:02d}.json")
            result_path = directory / f"HTTP_RESULT_{attempt:02d}.json"
            status = None

            def on_attempt(actual):
                base.require(actual == payload, "actual_payload_differs_from_frozen_request")
                base.require(self.http_counter < self.limit, "global_http_budget_exhausted")
                self.http_counter += 1
                base.write(attempt_path, {"actual_payload_equals_frozen": True,
                    "payload_sha256": base.digest(actual), "requested_model": MODEL,
                    "http_attempt_number": self.http_counter, "request_attempt": attempt,
                    "same_payload_retry": attempt > 1, "transport_retries": attempt - 1})

            try:
                response = self.factory(self.profile, on_http_attempt=on_attempt).complete(
                    prompt=payload["messages"][0]["content"], max_tokens=4096, json_object=True)
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


def review_request(root, route, scene, history, realization, other_turns, card=None, repair=None):
    prompt = realization_review.build_review_prompt(root, route, scene, history, realization, other_turns,
                                                     dataset_card=card)
    if repair:
        prompt += "\nREVIEW_FORMAT_REPAIR:\n" + json.dumps(repair, ensure_ascii=False)
    payload = {"model": MODEL, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
               "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]}
    proof = {"root_id": root, "route": route, "role": "scene_reviewer", "scene_id": scene["scene_id"],
             "independent_request_without_prior_verdict": repair is None,
             "review_format_repair": repair is not None, "other_branch_review_only": True,
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
    # Free-form reviewer prose may quote another branch. Keep it in the audit,
    # but return only structured findings and the already-seen rejected scene.
    codes = []
    base.require(classify_review_failure(review) == "realizer_repair", "reviewer_failure_not_repairable")
    for field, ok in (("new_event_or_progress", True), ("consistent_with_history", True),
                      ("duplicates_existing_event", False)):
        if type(parsed.get(field)) is bool and parsed[field] is not ok:
            codes.append(field.upper())
    return {"instruction": "Rewrite this scene. Realize every planned development at its stated action status; add a distinct event/progress.",
            "failure_codes": codes, "unmet_development_indices": unmet, "previous_response": realization}


def produce_scene(caller, run, root, route, history, cards, scene, other_turns):
    repair = None
    scene_id = scene["scene_id"]
    letter = "A" if route == ROUTES[0] else "B"
    audits = []
    for attempt in range(1, SCENE_ATTEMPTS + 1):
        payload, proof = make_request(run, root, route, "realizer", history, cards, scene_id, scene=scene, repair=repair)
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
        review, review_dir, raw_review = None, None, None
        review_repair = None
        for review_attempt in range(1, REVIEW_RETRIES + 2):
            payload, proof = review_request(root, route, scene, history, realization, other_turns,
                                            card=card, repair=review_repair)
            raw_review, review_dir = caller.call(
                f"{root}-{letter}-{scene_id}-review-{attempt}-{review_attempt}", payload, proof)
            review = realization_review.validate_review(raw_review, root, route, scene, realization)
            base.write(review_dir / "REVIEW.json", review)
            if review.get("schema_valid") is not False:
                break
            review_repair = {"failure_codes": review.get("schema_failures", review.get("failures", [])),
                             "previous_response": raw_review,
                             "instruction": "Return the exact required JSON schema. Fix only these format, identity, or citation errors; do not change the scene being reviewed."}
        audit = {"status": review["status"], "stage": "qwen_plan_review", "request": str(directory),
                 "review_request": str(review_dir), "failures": review["failures"],
                 "schema_valid": review.get("schema_valid"), "schema_failures": review.get("schema_failures", [])}
        reviewer_failure = review["status"] != "PASS" and classify_review_failure(review) == "reviewer_failure"
        if reviewer_failure:
            audit["status"] = "REVIEWER_FAILURE_NO_DIALOGUE_REWRITE"
        base.write(directory / "ACCEPTANCE.json", audit); audits.append(audit)
        if review["status"] == "PASS":
            return realization, audits
        if reviewer_failure:
            return None, audits
        repair = repair_for_review(review, realization)
    return None, audits


def run_root(caller, run, root, cards, *, target_scenes=3):
    frozen = base.read(run / "sources" / f"{root}.json")["turns"]
    accepted_by_branch = {"A": [], "B": []}
    results = []
    for route, letter in zip(ROUTES, ("A", "B")):
        history = deepcopy(frozen)  # Never inherit the other branch's additions.
        other = accepted_by_branch["B" if letter == "A" else "A"]
        folder = run / "roots" / root / letter
        base.write(folder / "INPUT.json", {"root_id": root, "route": route, "turns": history,
                                           "branch_base_sha256": base.digest(frozen)})
        slots = target_scenes if letter == "A" else min(target_scenes, len(cards))
        accepted, failures = [], []
        for number in range(1, slots + 1):
            scene_id = f"S{number:02d}"
            card = cards[number - 1] if letter == "B" else None
            if letter == "B" and card is None:
                failures.append({"scene_id": scene_id, "reason": "reserved_source_card_extraction_failed"})
                continue
            guidance = [base.dataset_cards.build_realization_guidance(card)] if card else []
            repair, plan = None, None
            try:
                for attempt in range(1, PLAN_ATTEMPTS + 1):
                    payload, proof = make_request(run, root, route, "planner", history, guidance, scene_id, repair=repair)
                    raw, directory = caller.call(f"{root}-{letter}-{scene_id}-planner-{attempt}", payload, proof)
                    try:
                        plan = validate_plan(raw, root, route, history, guidance, scene_id)
                        base.write(directory / "VALIDATION.json", {"status": "PASS"})
                        break
                    except (ValueError, TypeError) as error:
                        base.write(directory / "VALIDATION.json", {"status": "REJECT", "reason": str(error)})
                        repair = {"failure": str(error), "previous_response": raw,
                                  "instruction": "Return one scene for this slot and this card, fixing the reported error."}
                if plan is None:
                    failures.append({"scene_id": scene_id, "reason": "plan_rejected_after_retry"}); continue
                base.write(folder / f"{scene_id}.PLAN.json", plan)
                if not plan["scenes"]:
                    failures.append({"scene_id": scene_id, "reason": plan["shortfall_reason"]}); continue
                realization, audits = produce_scene(caller, run, root, route, history, guidance, plan["scenes"][0], other)
                base.write(folder / f"{scene_id}.ATTEMPTS.json", audits)
                if realization is None:
                    failures.append({"scene_id": scene_id, "reason": "no_accepted_scene", "attempts": audits}); continue
                base.write(folder / f"{scene_id}.json", realization)
                history = base.append_scene(history, root, route, realization)
                accepted_by_branch[letter] = history[len(frozen):]
                accepted.append({"scene_id": scene_id, "dataset_card_id": card["card_id"] if card else None,
                                 "turns": len(realization["turns"]), "duplicate_check": "PASS", "qwen_review": "PASS"})
            except RuntimeError as error:
                failures.append({"scene_id": scene_id, "reason": str(error),
                                 "per_request_automatic_retry_limit": TRANSPORT_RETRIES})
                break
        branch_history_ok(run, root, route, history)
        result = {"root_id": root, "route": route, "target_scenes": target_scenes,
                  "available_slots": slots, "accepted_scenes": len(accepted),
                  "scene_metadata": accepted, "failures": failures,
                  "shortfall_scenes": target_scenes - len(accepted),
                  "added_turns": len(history) - len(frozen), "input_turns": len(frozen),
                  "branch_base_sha256": base.digest(frozen), "output_history_sha256": base.digest(history),
                  "other_branch_in_generator_context": False,
                  "no_available_guidance": letter == "B" and slots == 0,
                  "acceptance": "exact_duplicate_check_and_same_qwen_plan_review",
                  "semantic_certified": False}
        base.write(folder / "RESULT.json", result)
        base.write(folder / "cumulative_dialogue.json", {"root_id": root, "route": route, "turns": history,
            "version": VERSION, "branch_base_sha256": base.digest(frozen), "model": MODEL,
            "branch_independent": True, "must_not_concatenate_with_other_branch": True})
        results.append(result)
    base.write(run / "roots" / root / "BRANCHES.json", {
        "root_id": root, "shared_base_sha256": base.digest(frozen), "independent_branches": True,
        "dialogues": {letter: str(run / "roots" / root / letter / "cumulative_dialogue.json") for letter in ("A", "B")},
        "merged_dialogue": None, "downstream_requires_explicit_branch_source": True})
    return results


def prepare(run, *, scenes=3):
    base.require(not run.exists(), "run_already_exists")
    base.require(type(scenes) is int and 1 <= scenes <= 3, "scenes_outside_1_3")
    base.require(run.parent == HERE / "runs", "run_must_be_in_v5_inventory")
    # Arbitrary manifests could relabel a consumed event and recompute their
    # own hashes. Only use the source-bound local freezer for this version.
    prior = source_inventory.collect_prior()
    consumed = {source_inventory.event_key(row) for row in prior["accepted_uses"]}
    sources = source_pool.freeze_unused_source_subset(
        consumed, count=scenes * len(base.ROOTS)
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
    allocation = source_inventory.legacy.allocate(sources, base.ROOTS, prior, scenes)
    run.mkdir(parents=True)
    base.feedback.compile_feedback(run / "feedback", base.ROOTS)
    original_sources = []
    for root, (shard, sha) in base.SOURCE_SPECS.items():
        path = base.PROJECT / f"restart-n5-runs/h4_unexpanded122_memory_event_k3_v1/shards/shard-{shard}/expansion/roots/{root}/promotion/promotable_expanded_dialogue.json"
        base.require(base.digest(path.read_bytes()) == sha, "frozen_R01_source_changed")
        base.write(run / "sources" / f"{root}.json", base.read(path)); original_sources.append(path)
    base.write(run / "dataset_sources.json", sources)
    base.write(run / "UNIQUE_EVENT_ALLOCATION.json", allocation)
    previews = []
    for root in base.ROOTS:
        history = base.read(run / "sources" / f"{root}.json")["turns"]
        for route in ROUTES:
            mock_cards = [] if route == ROUTES[0] else [{"card_id": "OFFLINE_PREVIEW_ONLY", "guidance_level": "seed"}]
            for role in ("planner", "realizer"):
                scene = {"scene_id": "S01", "developments": ["OFFLINE_PREVIEW_ONLY"]} if role == "realizer" else None
                payload, proof = make_request(run, root, route, role, history, mock_cards, "S01", scene=scene)
                proof["mock_only_offline_preview"] = True
                base.write(run / "previews" / root / f"{route}.{role}.json", {"payload": payload, "proof": proof})
                previews.append(proof)
    files = list(HERE.glob("*.py")) + list(LEGACY.glob("*.py"))
    files += [HERE.parent / "h4_r02_source_unique_v2/unique_events.py", *original_sources]
    files += [Path(x["path"]) for x in prior["prior_evidence"]]
    files += [Path(row[k]) for row in sources["files"] for k in ("local_path", "source_receipt_path")]
    files += list(run.rglob("*.json")) + list(run.rglob("*.txt"))
    logical_budget = ((scenes * len(base.ROOTS) + allocation["assigned_slots"])
                      * (PLAN_ATTEMPTS + SCENE_ATTEMPTS * (REVIEW_RETRIES + 2))
                      + allocation["assigned_slots"] * CARD_EXTRACTION_ATTEMPTS)
    maximum = logical_budget * (TRANSPORT_RETRIES + 1)
    manifest = {"version": VERSION, "model": MODEL, "review_model": MODEL, "target_solver_model": base.TARGET,
        "roots": base.ROOTS, "scenes_per_route_target": scenes,
        "sequence": "P+R01 -> A; same frozen P+R01 -> B; outputs remain separate",
        "other_branch_context": "checker/reviewer only; never generator, including repair",
        "max_planner_attempts": PLAN_ATTEMPTS, "max_realizer_attempts": SCENE_ATTEMPTS,
        "automatic_transport_retries": TRANSPORT_RETRIES,
        "retryable_transport_statuses": [503, 504],
        "card_extraction_attempts_per_source": CARD_EXTRACTION_ATTEMPTS,
        "review_schema_retries": REVIEW_RETRIES, "http_max": maximum, "files": base.bindings(files),
        "live_ready": True,
        "route_readiness": {"A": True, "B": allocation["assigned_slots"] > 0},
        "blocking_reason": None if allocation["assigned_slots"] else "B_NO_UNUSED_DATASET_EVENTS",
        "prepared_at_utc": now(), "generation_calls": 0, "solver_calls": 0, "training_started": False}
    base.write(run / "MANIFEST.json", manifest)
    result = {"status": "PREPARED" if allocation["assigned_slots"] else "PREPARED_A_ONLY_NO_UNUSED_DATASET_EVENTS",
              "assigned_B_slots": allocation["assigned_slots"], "excluded_source_ids": allocation["excluded_prior_source_ids"],
              "independent_branch_previews": len(previews), "all_feedback_receipts_present": all(p["feedback_receipt"] for p in previews),
              "live_ready": manifest["live_ready"], "route_readiness": manifest["route_readiness"], "http_calls": 0}
    base.write(run / "PREPARE_CHECKS.json", result)
    return result


def verify(run, *, live_execution=False):
    manifest = base.read(run / "MANIFEST.json")
    base.require(manifest["version"] == VERSION, "wrong_version")
    if not live_execution and (run / "CODE_SNAPSHOT_INDEX.json").exists() and (run / "REPORT.json").exists():
        from audit_archive import verify_archive
        verify_archive(run)
    else:
        base.verify_bindings(manifest["files"])
    allocation = base.read(run / "UNIQUE_EVENT_ALLOCATION.json")
    source_inventory.legacy.verify_allocation(allocation, base.read(run / "dataset_sources.json"), base.ROOTS)
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
        for root in base.ROOTS:
            assigned = allocation["assignments"][root]
            # Keep missing slots as None. A later valid card retains its original
            # reserved S02/S03 identity and can still be used in this branch.
            root_cards = [by_source.get(x["source_id"]) for x in assigned]
            source_inventory.verify_cards_for_root([c for c in root_cards if c is not None], allocation, root)
            missing = [f"S{i:02d}" for i, card in enumerate(root_cards, 1) if card is None]
            if missing:
                base.write(run / "roots" / root / "CARD_SHORTFALL.json", {
                    "reason": "card_extraction_incomplete_only_missing_slots_skipped", "missing_scene_ids": missing})
            root_rows = run_root(caller, run, root, root_cards, target_scenes=manifest["scenes_per_route_target"])
            rows.extend(root_rows)
            for scene in root_rows[1]["scene_metadata"]:
                card = root_cards[int(scene["scene_id"][1:]) - 1]
                key = source_inventory.event_key(card)
                uses.append({"dataset": key[0], "split": key[1], "item_id": key[2], "source_id": card["source_id"],
                             "source_item_sha256": card["lineage"]["source_item_sha256"],
                             "root_id": root, "scene_id": scene["scene_id"], "run": str(run)})
        base.verify_bindings(manifest["files"])
        actual_http = len(list((run / "requests").glob("*/HTTP_ATTEMPT*.json")))
        output = {"version": VERSION,
                  "status": "COMPLETE_WITH_SHORTFALL" if any(r["shortfall_scenes"] for r in rows) else "COMPLETE",
                  "routes": rows, "accepted_uses": uses,
                  "accepted_scenes": sum(r["accepted_scenes"] for r in rows), "http_attempts": actual_http,
                  "http_request_reservations": caller.request_counter,
                  "http_attempt_log_count": caller.http_counter,
                  "branches_remain_separate": True, "model": MODEL, "question_calls": 0, "solver_calls": 0,
                  "training_started": False, "semantic_certified": False}
        base.write(run / "REPORT.json", output)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify", "live"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=3)
    args = parser.parse_args()
    run = args.run.resolve()
    result = prepare(run, scenes=args.scenes) if args.action == "prepare" else (
        live(run) if args.action == "live" else {"status": "PASS", "live_ready": verify(run)[0]["live_ready"]})
    print(json.dumps(result, ensure_ascii=False, indent=2))
