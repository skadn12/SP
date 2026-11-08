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

VERSION = "h4_r02_independent_routes_v4"
MODEL = base.dataset_cards.MODEL
SCENE_ATTEMPTS = 2
PLAN_ATTEMPTS = 2
ROUTES = base.ROUTES
LOCK_PATH = HERE.parent / "h4_r02_source_unique_v2/runs/.unique_event_live.lock"

PLANNER = """Plan at most ONE new scene after this branch's cumulative_history.
This is one scene slot, not a request to fill three scenes. Return exactly
{root_id,route,scenes,shortfall_reason}. scenes has zero or one entry. For zero
scenes explain why; for one scene shortfall_reason may be empty.
The scene has exactly {scene_id,actors,source_anchors,goal,developments,
dataset_card_id,feedback_topic_ids}. scene_id MUST equal target_scene_id.
actors: at least two distinct allowed_speakers. source_anchors: 1-3 exact
{turn_ref,quote} entries from cumulative_history. goal: concrete short string.
developments: 2-5 concrete achievable new changes, decisions or outcomes.
feedback_topic_ids: at most four IDs from supplied previous-round feedback.
For old_event_continuation, progress an existing event; dataset_card_id is null.
For dataset_new_event, introduce a distinct new event from the single supplied
dataset card; dataset_card_id MUST be its card_id. Transfer a scaffold's
relationships/conditions or a seed's premise, never source identities as facts.
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
    # Reuse legacy structural/evidence checks on a validation-only view. Never
    # change the actual model response or its accepted plan.
    view = deepcopy(plan)
    if scenes:
        view["scenes"][0]["scene_id"] = "S01"
        base.require(isinstance(view.get("shortfall_reason"), str), "shortfall_reason_invalid")
        view["shortfall_reason"] = view["shortfall_reason"] or "single-scene slot"
    base.validate_plan(view, root, route, history, cards)
    return plan


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
    """No transport retries; successful but rejected scenes use a new request."""
    def __init__(self, run):
        self.run = run
        self.limit = base.read(run / "MANIFEST.json")["http_max"]
        profile, factory, annotate, capacity, dev = base.transport()
        self.profile, self.factory = profile(), factory
        self.annotate, self.capacity = annotate, capacity
        self.tokenizer = dev.tokenizer_from(str(base.TOKENIZER))
        self.http_counter = 0

    def call(self, label, payload, proof):
        base.require(self.http_counter < self.limit, "global_http_budget_exhausted")
        base.require(payload["model"] == MODEL, "unexpected_model")
        capacity = self.annotate(self.capacity(self.profile, self.tokenizer, payload["messages"], 4096))
        base.require(capacity["status"] == "PASS", "capacity_blocked_no_truncation")
        self.http_counter += 1
        directory = self.run / "requests" / f"{self.http_counter:03d}-{label}"
        base.write(directory / "REQUEST.json", payload)
        base.write(directory / "PROOF.json", {**proof, "capacity": capacity, "payload_sha256": base.digest(payload)})
        base.write(directory / "RESERVATION.json", {"http_number": self.http_counter, "http_max": self.limit,
                                                     "status": "RESERVED_NO_AUTOMATIC_REPLAY"})
        def on_attempt(actual):
            base.require(actual == payload, "actual_payload_differs_from_frozen_request")
            base.write(directory / "HTTP_ATTEMPT.json", {"actual_payload_equals_frozen": True,
                "payload_sha256": base.digest(actual), "requested_model": MODEL, "transport_retries": 0})
        try:
            response = self.factory(self.profile, on_http_attempt=on_attempt).complete(
                prompt=payload["messages"][0]["content"], max_tokens=4096, json_object=True)
            metadata = getattr(response, "metadata", {}) or {}
            base.write(directory / "RESPONSE.json", {"text": response.raw_text, "metadata": metadata})
            base.require(metadata.get("returned_model") == MODEL, "response_model_mismatch")
            base.require(metadata.get("finish_reason") == "stop", "response_not_complete")
            return response.raw_text, directory
        except Exception as error:
            base.write(directory / "ERROR.json", {"error_type": type(error).__name__, "no_automatic_retry": True})
            raise RuntimeError("provider_request_failed_no_automatic_retry") from None


def review_request(root, route, scene, history, realization, other_turns):
    prompt = realization_review.build_review_prompt(root, route, scene, history, realization, other_turns)
    payload = {"model": MODEL, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
               "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]}
    proof = {"root_id": root, "route": route, "role": "scene_reviewer", "scene_id": scene["scene_id"],
             "independent_request_without_prior_verdict": True, "other_branch_review_only": True,
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
    if any(not f.startswith(("review_verdict_reject:", "development["))
           or ":not_realized:" not in f and f.startswith("development[") for f in review["failures"]):
        codes.append("REVIEW_SCHEMA_OR_EVIDENCE_INVALID")
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
        try:
            realization = base.validate_realization(parse_object(raw), root, route, scene)
        except (ValueError, TypeError) as error:
            audit = {"status": "REJECT", "stage": "mechanical", "reason": str(error)}
            base.write(directory / "ACCEPTANCE.json", audit); audits.append(audit)
            repair = {"instruction": "Fix this specific response contract error.", "failure": str(error), "previous_response": raw}
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
        payload, proof = review_request(root, route, scene, history, realization, other_turns)
        raw_review, review_dir = caller.call(f"{root}-{letter}-{scene_id}-review-{attempt}", payload, proof)
        review = realization_review.validate_review(raw_review, root, route, scene, realization)
        base.write(review_dir / "REVIEW.json", review)
        audit = {"status": review["status"], "stage": "qwen_plan_review", "request": str(directory),
                 "review_request": str(review_dir), "failures": review["failures"]}
        base.write(directory / "ACCEPTANCE.json", audit); audits.append(audit)
        if review["status"] == "PASS":
            return realization, audits
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
                failures.append({"scene_id": scene_id, "reason": str(error), "automatic_transport_retry": False})
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
    base.require(run.parent == HERE / "runs", "run_must_be_in_v4_inventory")
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
    maximum = (scenes * len(base.ROOTS) + allocation["assigned_slots"]) * (PLAN_ATTEMPTS + 2 * SCENE_ATTEMPTS) + len(sources["items"])
    manifest = {"version": VERSION, "model": MODEL, "review_model": MODEL, "target_solver_model": base.TARGET,
        "roots": base.ROOTS, "scenes_per_route_target": scenes,
        "sequence": "P+R01 -> A; same frozen P+R01 -> B; outputs remain separate",
        "other_branch_context": "checker/reviewer only; never generator, including repair",
        "max_planner_attempts": PLAN_ATTEMPTS, "max_realizer_attempts": SCENE_ATTEMPTS,
        "automatic_transport_retries": 0, "http_max": maximum, "files": base.bindings(files),
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
        cards, rejected = [], []
        # One source per extraction avoids cross-batch card-ID collisions.
        for item in sources["items"]:
            if item["source_id"] not in selected_ids:
                continue
            subset = base.dataset_cards.select_source_subset(sources, [item["source_id"]])
            payload = base.dataset_cards.build_extraction_request(subset)
            try:
                raw, directory = caller.call(f"extract-{len(cards)+len(rejected):02d}", payload,
                                             {"role": "dataset_extractor", "source_id": item["source_id"]})
                checked = base.dataset_cards.validate_extracted_cards(raw, subset)
                base.write(directory / "VALIDATION.json", checked)
                cards.extend(checked["accepted_cards"]); rejected.extend(checked["rejected_records"])
            except RuntimeError as error:
                rejected.append({"source_id": item["source_id"], "reason": str(error)})
        base.write(run / "dataset_cards_result.json", {"accepted_cards": cards, "rejected_records": rejected})
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
        actual_http = len(list((run / "requests").glob("*/HTTP_ATTEMPT.json")))
        output = {"version": VERSION,
                  "status": "COMPLETE_WITH_SHORTFALL" if any(r["shortfall_scenes"] for r in rows) else "COMPLETE",
                  "routes": rows, "accepted_uses": uses,
                  "accepted_scenes": sum(r["accepted_scenes"] for r in rows), "http_attempts": actual_http,
                  "http_request_reservations": caller.http_counter,
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
