"""Collect provisional A/B dialogue without making audit completeness an admission gate."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil

import runner as r
import candidate_contract as contract
import candidate_metrics as metrics
import grounded_task

HERE = Path(__file__).resolve().parent
VERSION = "h4_candidate_collection_v51"
CODE_SNAPSHOT = "h4_sequential_ab_v57_root194259_v1"
ROOTS = tuple(f"cc-train-episode-{value}" for value in ("138343", "135466", "210046", "194259"))
ROOT = ROOTS[-1]
BASELINE_POLICY = "immutable_exact_original_plus_R01_no_inherited_A_or_B"
TRANSPORT_RETRY_POLICY = "one_identical_payload_retry_only_for_http_503_504"
SOURCE_DIR = HERE.parent / "h4_external_event_source_refresh_v1"
IDENTITY_KEYS = ("root_id", "route", "scene_id")
SKIPPED = {name: "SKIPPED_BY_CANDIDATE_POLICY" for name in ("purpose", "latest_state", "event_scanner")}


def identity(root, route, scene_id):
    return dict(zip(IDENTITY_KEYS, (root, route, scene_id)))


def payload(prompt):
    return {"model": r.MODEL, "temperature": .25, "max_tokens": 4096,
            "enable_thinking": False, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}


def request(run, root, route, scene_id, role, history, baseline, card=None, plan=None,
            realization=None, repair=None):
    r.base.require(history[:len(baseline)] == baseline, "formal_baseline_prefix_changed")
    context = {**identity(root, route, scene_id), "allowed_speakers": sorted({t["speaker"] for t in baseline}),
        "cumulative_history": history, "formal_baseline_turns": len(baseline),
        "history_policy": "exact_formal_prefix_then_provisional_candidate_dialogue",
        "dataset_card": card, "advisory_plan": plan, "new_scene": realization,
        "mechanical_repair": repair}
    common = r.base.COMMON + """
The suffix after formal_baseline_turns is provisional collected dialogue, NOT admitted
training data. Maintain narrative continuity with its actual words, without treating
any plan or reviewer summary as an established fact. Only allowed_speakers may speak;
other people may act in explicitly attributed reports, keeping their own roles.
A advances an existing event through a concrete action, decision, or outcome.
B adapts the supplied source card's core interaction into a distinct occasion with
its own objective. Related people/themes are allowed; do not replace the core
interaction with another logistics step of an old task. Preserve actor, action,
object and action status: considering/agreeing is not completing. A seed's optional
steps and every draft development need not all occur. Add substantive progress,
not repeated farewells or a recap. Full dialogue and source evidence take priority.
"""
    if role == "planner":
        instruction = """Provide one advisory plan, with JSON fields root_id, route,
scene_id, actors (allowed speakers), goal (string), developments (list of strings),
dataset_card_id (the supplied ID for B, null for A), source_anchors (list of exact
{turn_ref,quote} historical citations, at least one for A), actor_mapping (list of
{card_actor_id,root_actor}, required for B's card actors; root_actor can be an
explicitly described non-speaking participant). A plan is a draft, not history.
"""
    elif role == "realizer":
        instruction = """Write the new scene. Aim for 12-16 natural conversational turns;
length is guidance. Use an advisory plan only where faithful to the full history
and card. If the plan is absent, compose directly from those inputs. Return JSON:
{root_id,route,scene_id,status:"scene",turns:[{speaker,text}],dataset_card_id,
feedback_topic_ids:[],no_scene_reason:""}. If no coherent scene can be written,
return status:"no_scene", turns:[], and explain no_scene_reason. Do not output a
review or a summary in place of dialogue.
"""
    else:
        instruction = """Independently diagnose the NEW dialogue, not the plan's intended
outcome. Use new-scene references NEW:T001 etc. Do not require every planned step,
old-goal comparison, or optional source step. An explicit supported decision or
interaction counts as progress without requiring subsequent physical completion.
Return JSON {root_id,route,scene_id,verdict:"acceptable"|"problem"|"uncertain",
core_checks:{source_core:{status,reason,evidence_refs},actor_actions:{status,reason,evidence_refs},
history_consistency:{status,reason,evidence_refs},actual_progress:{status,reason,evidence_refs}},
plan_observations:[],summary:""}. Every status is "ok", "problem", or "uncertain";
evidence_refs is a list of exact history/NEW references. For A, source_core checks
continuation of its existing event. Report a core problem only for an explicit
content problem. Incomplete plan coverage is an observation, not automatically a
core problem. Unsupported or ambiguous claims are uncertain. Cite what the new
dialogue actually says; do not fill gaps using a draft or imagined outcome.
"""
    data = payload(common + instruction + "\nCANDIDATE_INPUT_JSON\n" + json.dumps(context, ensure_ascii=False))
    receipt = None
    if role in ("planner", "realizer"):
        feedback = r.base.read(run / "feedback/COMPILED_FEEDBACK.json")
        data, receipt = r.base.feedback.append_feedback_to_request(data, run / "feedback", root, route,
            role, target_solver_model=r.base.TARGET,
            expected_target_identity_sha256=feedback["target_solver_identity_sha256"])
    proof = {**identity(root, route, scene_id), "role": role, "history_sha256": r.base.digest(history),
        "history_turns": len(history), "formal_baseline_sha256": r.base.digest(baseline),
        "formal_baseline_turns": len(baseline), "history_policy": context["history_policy"],
        "card_sha256": r.base.digest(card), "plan_sha256": r.base.digest(plan),
        "realization_sha256": r.base.digest(realization), "feedback_receipt": receipt,
        "training_pool_eligible": False, "semantic_certified": False}
    return grounded_task.append_current_task(data, proof, root=root, route=route,
        scene_id=scene_id, role=role, history=history, card=card, plan=plan,
        realization=realization, repair=repair)


def validate_plan(raw, expected, history, card):
    plan = r.parse_object(raw)
    r.base.require(all(plan.get(k) == v for k, v in expected.items()), "plan_identity_mismatch")
    allowed = {t["speaker"] for t in history}
    actors = plan.get("actors")
    r.base.require(isinstance(actors, list) and actors and all(isinstance(x, str) and x in allowed for x in actors),
                   "plan_actor_invalid")
    r.base.require(isinstance(plan.get("goal"), str) and plan["goal"].strip(), "plan_goal_missing")
    r.base.require(isinstance(plan.get("developments"), list) and
                   all(isinstance(x, str) and x.strip() for x in plan["developments"]), "plan_developments_invalid")
    r.base.require(plan.get("dataset_card_id") == (card["card_id"] if card else None), "plan_card_mismatch")
    anchors = plan.get("source_anchors", [])
    r.base.require(isinstance(anchors, list), "plan_anchor_invalid")
    refs = {t["turn_ref"]: t["text"] for t in history}
    for anchor in anchors:
        r.base.require(isinstance(anchor, dict) and anchor.get("turn_ref") in refs
            and isinstance(anchor.get("quote"), str) and anchor["quote"].strip()
            and anchor["quote"] in refs[anchor["turn_ref"]], "plan_anchor_invalid")
    r.base.require(card is not None or anchors, "A_plan_anchor_missing")
    if card:
        mappings = plan.get("actor_mapping", [])
        r.base.require(isinstance(mappings, list) and all(isinstance(x, dict) for x in mappings), "plan_mapping_invalid")
        ids = {actor["actor_id"] for actor in card.get("actors", [])}
        r.base.require({x.get("card_actor_id") for x in mappings} == ids and len(mappings) == len(ids)
                       and all(isinstance(x.get("root_actor"), str) and x["root_actor"].strip() for x in mappings),
                       "plan_mapping_invalid")
    return plan


def validate_dialogue(raw, expected, allowed_speakers, card):
    obj = r.parse_object(raw)
    r.base.require(all(obj.get(k) == v for k, v in expected.items()), "dialogue_identity_mismatch")
    r.base.require(obj.get("status") in ("scene", "no_scene"), "dialogue_status_invalid")
    turns = obj.get("turns")
    r.base.require(isinstance(turns, list), "dialogue_turns_invalid")
    if obj["status"] == "no_scene":
        r.base.require(not turns, "no_scene_contains_dialogue")
        return obj, ["model_returned_no_scene"]
    r.base.require(turns, "dialogue_empty")
    for turn in turns:
        r.base.require(isinstance(turn, dict) and isinstance(turn.get("speaker"), str)
                       and turn["speaker"] in allowed_speakers, "dialogue_speaker_invalid")
        r.base.require(isinstance(turn.get("text"), str) and turn["text"].strip(), "dialogue_text_invalid")
    if "dataset_card_id" in obj:
        r.base.require(obj["dataset_card_id"] == (card["card_id"] if card else None), "dialogue_card_mismatch")
    warnings = []
    if not 12 <= len(turns) <= 16:
        warnings.append("turn_count_outside_suggested_12_16")
    for key in ("feedback_topic_ids", "no_scene_reason", "dataset_card_id"):
        if key not in obj:
            warnings.append("auxiliary_field_missing:" + key)
    return obj, warnings


def append_candidate(history, root, route, realization):
    letter = "A" if route == r.ROUTES[0] else "B"
    ref = f"CANDIDATE:{letter}:{realization['scene_id']}"
    extra = [{"turn_ref": f"{ref}:T{i:03d}", "session_ref": f"{root}:{ref}", "scene_ref": ref,
        "speaker": t["speaker"], "text": t["text"], "origin": "provisional_candidate",
        "training_pool_eligible": False} for i, t in enumerate(realization["turns"], 1)]
    combined = deepcopy(history) + extra
    r.base.require(len({t["turn_ref"] for t in combined}) == len(combined), "candidate_ref_collision")
    return combined


def inspect_realization(raw, expected, history, baseline, card):
    """Report all readable duplicate evidence even when identity is invalid."""
    realization, failures, warnings, duplicate = None, [], [], None
    try:
        realization = r.parse_object(raw)
        realization, warnings = validate_dialogue(raw, expected, {t['speaker'] for t in baseline}, card)
        if realization['status'] == 'no_scene':
            failures.append('model_returned_no_scene')
    except (ValueError, TypeError, KeyError) as error:
        failures.append(str(error))
    turns = realization.get('turns') if isinstance(realization, dict) else None
    if isinstance(turns, list) and turns and all(isinstance(t, dict) and
        isinstance(t.get('speaker'), str) and isinstance(t.get('text'), str) for t in turns):
        duplicate = r.duplicate_check.inspect_duplicates(turns, {'complete_candidate_history': history})
    return realization, failures, warnings, duplicate


def collect_slot(caller, run, root, route, scene_id, history, baseline, card=None):
    """One optional plan, up to two mechanical realizations, one advisory review."""
    expected = identity(root, route, scene_id)
    letter = "A" if route == r.ROUTES[0] else "B"
    folder = run / "candidates" / root / letter / scene_id
    r.base.write(folder / "INPUT.json", {**expected, "history": history,
        "baseline_sha256": r.base.digest(baseline), "card": card, "skipped_stages": SKIPPED})
    plan, plan_issue = None, None
    try:
        raw, directory = caller.call(f"{root}-{letter}-{scene_id}-planner", *request(
            run, root, route, scene_id, "planner", history, baseline, card))
        try:
            plan = validate_plan(raw, expected, history, card)
        except (ValueError, TypeError, KeyError) as error:
            plan_issue = str(error)
        r.base.write(directory / "ADVISORY.json", {"usable": plan is not None, "issue": plan_issue,
            "blocks_generation": False, "semantic_certified": False})
    except RuntimeError as error:
        plan_issue = str(error)
    r.base.write(folder / "PLAN_ADVISORY.json", {"plan": plan, "issue": plan_issue, "blocks_generation": False})
    records, repair = [], None
    for attempt in (1, 2):
        data, proof = request(run, root, route, scene_id, "realizer", history, baseline, card, plan, repair=repair)
        try:
            raw, directory = caller.call(f"{root}-{letter}-{scene_id}-realizer-{attempt}", data, proof)
        except RuntimeError as error:
            r.base.write(folder / f"TRANSPORT_FAILURE_{attempt}.json", {"stage": "realizer", "error": str(error)})
            break
        realization, failures, warnings, dup = inspect_realization(raw, expected, history, baseline, card)
        # Persist generated text before attempting any diagnostic call.
        r.base.write(folder / f"GENERATED_{attempt}.json", {"raw": raw, "realization": realization,
            "request": str(directory), "mechanical_failures": failures, "quality_warnings": warnings,
            "duplicate_result": dup, "training_pool_eligible": False})
        review_raw, review_error, review_directory = None, None, None
        if not failures and dup and dup["status"] == "PASS":
            try:
                review_raw, review_directory = caller.call(f"{root}-{letter}-{scene_id}-review", *request(
                    run, root, route, scene_id, "reviewer", history, baseline, card, plan, realization))
            except RuntimeError as error:
                review_error = str(error)
        decision = contract.decide_candidate(mechanical_failures=failures, duplicate_result=dup,
            review_raw=review_raw, review_error=review_error, expected_identity=expected)
        record = {"candidate_id": f"{root}/{letter}/{scene_id}/{attempt}", "root_id": root,
            "route": route, "scene_id": scene_id, "disposition": decision["disposition"],
            "realization": realization, "raw_review": review_raw, "review_error": review_error,
            "review_request": str(review_directory) if review_directory else None,
            "request": str(directory), "provenance": "fresh_collection", "decision": decision,
            "quality_warnings": warnings, "mechanical_failures": failures, "duplicate_result": dup,
            "training_pool_eligible": False, "semantic_certified": False}
        r.base.write(folder / f"RECORD_{attempt}.json", record)
        records.append(record)
        if decision["disposition"] != "hard_reject":
            if decision["continue_candidate_history"]:
                history = append_candidate(history, root, route, realization)
            break
        if realization and realization.get("status") == "no_scene":
            break
        repair = {"instruction": "Fix only the mechanical/duplicate issues, preserving source and history.",
                  "failures": failures, "duplicate_reasons": dup.get("reasons", []) if dup else [],
                  "previous_response": raw}
    return records, history


def prepare(run, *, root=ROOT, routes=r.ROUTES, scenes=3, source_dir=SOURCE_DIR):
    run, source_dir = Path(run).resolve(), Path(source_dir).resolve()
    r.base.require(run.parent == HERE / "runs" and not run.exists(), "invalid_or_existing_run")
    r.base.require(routes in ((r.ROUTES[0],), (r.ROUTES[1],), r.ROUTES) and 1 <= scenes <= 3, "invalid_scope")
    r.base.require(root in ROOTS, "invalid_root_selection")
    source = r.frozen_original_r01(root)
    prior = r.source_inventory.collect_prior()
    count = scenes if r.ROUTES[1] in routes else 0
    sources = r.source_pool.freeze_unused_source_subset(
        {r.source_inventory.event_key(x) for x in prior["accepted_uses"]}, count=max(1, count), source_dir=source_dir)
    allocation = (r.source_inventory.legacy.allocate(sources, [root], prior, count) if count else
        {"schema_version": "v41.no_B_sources", "roots": [root], "assignments": {root: []},
         "assigned_slots": 0, "prior": prior})
    r.base.require(allocation["assigned_slots"] == count, "insufficient_unused_sources")
    run.mkdir(parents=True)
    r.base.feedback.compile_feedback(run / "feedback", [root])
    r.base.write(run / "sources" / f"{root}.json", source["original"])
    baseline = source["original"]["turns"]
    r.base.write(run / "FORMAL_BASELINE.json", {"root_id": root, "turns": baseline,
        "history_sha256": r.base.digest(baseline), "policy": BASELINE_POLICY,
        "original_source_path": str(source["path"]), "original_source_sha256": source["sha256"]})
    r.base.write(run / "dataset_sources.json", sources)
    r.base.write(run / "UNIQUE_EVENT_ALLOCATION.json", allocation)
    profile, transport, tokenizer = r.transport_inputs()
    r.base.write(run / "provider_profile.json", profile)
    r.base.write(run / "PREVIEW.json", {"root_id": root, "routes": list(routes),
        "formal_history_sha256": r.base.digest(baseline), "formal_history_turns": len(baseline),
        "source_ids": [item['source_id'] for item in allocation['assignments'][root]],
        "current_task_contract": 'v43_original_evidence_final_task',
        "preview_only_card_not_yet_extracted": True, "executable_model_request": False,
        "live_request_requires_full_validated_source_card": True})
    # Per B slot: two card attempts, one optional plan, two mechanical realization
    # attempts, one diagnostic review. No semantic or report-schema retry loop.
    logical = len(routes) * scenes * 4 + count * 2
    files = list(HERE.glob("*.py")) + [HERE / "POLICY.md", HERE / "BASELINE_FILES.json",
        HERE / "V51_CODE_BASELINE.json", HERE / "ROOT_SELECTION.json"]
    files += [Path(r.base.read(HERE / "ROOT_SELECTION.json")["source_manifest"]["path"])]
    files += list(r.LEGACY.glob("*.py")) + transport + tokenizer + [source["path"]]
    files += [Path(x["path"]) for x in prior["prior_evidence"]]
    files += [Path(x[k]) for x in sources["files"] for k in ("local_path", "source_receipt_path")]
    files += [HERE.parent / "h4_r02_source_unique_v2/unique_events.py"]
    files += [p for p in run.rglob("*") if p.is_file()]
    manifest = {"version": VERSION, "code_snapshot": CODE_SNAPSHOT,
        "entrypoint": "candidate_runner.py", "root_id": root,
        "routes": list(routes), "scenes_per_route": scenes, "model": r.MODEL,
        "source_pool_directory": str(source_dir), "source_policy": "first_unused_stratified_no_replacement",
        "scope": "fresh_collection", "logical_max": logical, "http_max": logical * 2,
        "baseline_policy": BASELINE_POLICY, "original_source_path": str(source["path"]),
        "original_source_sha256": source["sha256"], "transport_retry_policy": TRANSPORT_RETRY_POLICY,
        "max_card_attempts": 2, "max_plan_attempts": 1, "max_realizer_attempts": 2, "max_review_attempts": 1,
        "skipped_stages": SKIPPED, "minimum_http_gap_after_response_seconds": 35,
        "current_task_contract": "v43_original_evidence_final_task",
        "card_context_policy": "full_validated_card_with_original_quotes",
        "training_pool_eligible": False, "training_started": False,
        "formal_history_sha256": r.base.digest(baseline), "files": r.base.bindings(files), "prepared_at": r.now()}
    r.base.write(run / "MANIFEST.json", manifest)
    # Reviewable immutable code snapshot, while verify also detects code changes.
    snapshots = []
    for i, row in enumerate(manifest["files"]):
        if Path(row["path"]).suffix == ".py":
            destination = run / "execution_code" / f"{i:04d}_{Path(row['path']).name}"
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(row["path"], destination)
            snapshots.append({"path": str(destination), "sha256": row["sha256"]})
    r.base.write(run / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    return {"status": "PREPARED", "run": str(run), "root_id": root, "assigned_B_slots": count,
            "logical_max": logical, "http_max": logical * 2, "generation_calls": 0, "sources_reserved": False}


def verify(run):
    run = Path(run).resolve()
    r.base.require(run.parent == HERE / "runs", "run_outside_current_source_inventory")
    manifest = r.base.read(run / "MANIFEST.json")
    r.base.require(manifest["version"] == VERSION and manifest["scope"] == "fresh_collection", "candidate_manifest_invalid")
    r.base.require(manifest.get("code_snapshot") == CODE_SNAPSHOT, "candidate_code_snapshot_changed")
    r.base.require(manifest["training_pool_eligible"] is False, "automatic_training_admission_forbidden")
    root = manifest["root_id"]
    r.base.require(root in ROOTS, "invalid_root_selection")
    routes, scenes = tuple(manifest["routes"]), manifest["scenes_per_route"]
    r.base.require(routes in ((r.ROUTES[0],), (r.ROUTES[1],), r.ROUTES)
                   and type(scenes) is int and 1 <= scenes <= 3, "candidate_scope_changed")
    count = scenes if r.ROUTES[1] in routes else 0
    logical = len(routes) * scenes * 4 + count * 2
    r.base.require(manifest["logical_max"] == logical and manifest["http_max"] == logical * 2
                   and manifest["model"] == r.MODEL
                   and manifest["minimum_http_gap_after_response_seconds"] == 35
                   and manifest["skipped_stages"] == SKIPPED
                   and manifest.get("baseline_policy") == BASELINE_POLICY
                   and manifest.get("transport_retry_policy") == TRANSPORT_RETRY_POLICY
                   and manifest.get('current_task_contract') == 'v43_original_evidence_final_task'
                   and manifest.get('card_context_policy') == 'full_validated_card_with_original_quotes',
                   "candidate_policy_changed")
    r.base.verify_bindings(manifest["files"])
    bound = {Path(x["path"]).resolve() for x in manifest["files"]}
    required = {run / name for name in ("FORMAL_BASELINE.json", "dataset_sources.json",
        "UNIQUE_EVENT_ALLOCATION.json", "provider_profile.json", "PREVIEW.json")}
    required.add(run / "sources" / f"{root}.json")
    source = r.frozen_original_r01(root)
    required.add(source["path"])
    required.update(HERE.glob("*.py"))
    r.base.require(required <= bound, "current_run_or_execution_inputs_not_bound")
    r.base.verify_bindings(r.base.read(HERE / "BASELINE_FILES.json")["files"])
    r.base.verify_bindings(r.base.read(HERE / "V51_CODE_BASELINE.json")["files"])
    r.base.verify_bindings(r.base.read(run / "EXECUTION_CODE_SNAPSHOT.json")["files"])
    r.base.require(not (run / "INHERITED_A.json").exists(), "inherited_A_forbidden")
    baseline_record = r.base.read(run / "FORMAL_BASELINE.json")
    baseline = baseline_record["turns"]
    r.base.require(r.base.read(run / "sources" / f"{root}.json") == source["original"]
                   and baseline == source["original"]["turns"]
                   and r.base.digest(baseline) == manifest["formal_history_sha256"],
                   "candidate_formal_baseline_changed")
    r.base.require(manifest["original_source_path"] == str(source["path"])
                   and manifest["original_source_sha256"] == source["sha256"]
                   and baseline_record == {"root_id": root, "turns": baseline,
                       "history_sha256": r.base.digest(baseline), "policy": BASELINE_POLICY,
                       "original_source_path": str(source["path"]), "original_source_sha256": source["sha256"]},
                   "candidate_original_source_binding_changed")
    sources = r.base.read(run / "dataset_sources.json")
    allocation = r.base.read(run / "UNIQUE_EVENT_ALLOCATION.json")
    consumed = {r.source_inventory.event_key(x) for x in allocation["prior"]["accepted_uses"]}
    expected_sources = r.source_pool.freeze_unused_source_subset(consumed, count=max(1, count),
        source_dir=Path(manifest["source_pool_directory"]))
    r.base.require(sources == expected_sources, "candidate_source_selection_changed")
    r.base.require(allocation["assigned_slots"] == count, "candidate_source_slot_count_changed")
    if r.ROUTES[1] in manifest["routes"]:
        r.source_inventory.legacy.verify_allocation(allocation, sources, [root])
    else:
        r.base.require(allocation == {"schema_version": "v41.no_B_sources", "roots": [root],
            "assignments": {root: []}, "assigned_slots": 0, "prior": allocation["prior"]}, "A_only_allocation_changed")
    profile, *_ = r.base.transport()
    r.base.require(profile() == r.base.read(run / "provider_profile.json"), "provider_profile_changed")
    return manifest, allocation


def extract_cards(caller, run, allocation):
    root = r.base.read(run / "MANIFEST.json")["root_id"]
    sources = r.base.read(run / "dataset_sources.json")
    cards = []
    for item in allocation["assignments"][root]:
        subset = r.base.dataset_cards.select_source_subset(sources, [item["source_id"]])
        selected, repair, attempts = None, None, []
        for attempt in (1, 2):
            try:
                raw, directory = caller.call(f"extract-{item['source_id'].replace(':', '-')}-{attempt}",
                    r.card_selection.build_extraction_request(subset, repair=repair),
                    {"role": "dataset_extractor", "source_id": item["source_id"], "source_item_sha256": item["source_item_sha256"]})
            except RuntimeError as error:
                attempts.append({"transport_error": str(error)})
                break
            checked = r.card_selection.select_cards(raw, subset)
            r.base.write(directory / "VALIDATION.json", checked)
            attempts.append({"request": str(directory), "selected": bool(checked["accepted_cards"])})
            if checked["accepted_cards"]:
                selected = checked["accepted_cards"][0]
                r.source_inventory.verify_cards_for_root([selected], allocation, root)
                break
            if not checked["retryable"]:
                break
            repair = checked["repair_feedback"]
        cards.append(selected)
        r.base.write(run / "extractions" / f"{len(cards):02d}.json", {"source_id": item["source_id"],
            "card": selected, "attempts": attempts, "source_remains_consumed": True})
    return cards


def token_counter():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(r.base.TOKENIZER), local_files_only=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def live(run):
    from candidate_transport import Caller
    run = Path(run).resolve()
    with r.LOCK_PATH.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest, allocation = verify(run)
        r.base.require(not (run / "LIVE_STARTED.json").exists(), "already_started_no_automatic_replay")
        r.source_inventory.reserve(run, allocation["assignments"], r.source_inventory.collect_prior())
        r.base.write(run / "LIVE_STARTED.json", {"version": VERSION, "started_at": r.now(), "scope": "fresh_collection"})
        caller = Caller(run)
        root = manifest["root_id"]
        cards = []
        baseline = r.base.read(run / "FORMAL_BASELINE.json")["turns"]
        history, records, shortfalls = deepcopy(baseline), [], []
        for route in manifest["routes"]:
            if route == r.ROUTES[1]:
                cards = extract_cards(caller, run, allocation)
            for number in range(1, manifest["scenes_per_route"] + 1):
                card = cards[number - 1] if route == r.ROUTES[1] else None
                if route == r.ROUTES[1] and card is None:
                    shortfalls.append({"route": route, "scene_id": f"S{number:02d}", "reason": "source_card_unavailable"})
                    continue
                rows, history = collect_slot(caller, run, root, route, f"S{number:02d}", history, baseline,
                    card)
                records.extend(rows)
                if not any(x["disposition"] == "retained_pending_review" for x in rows):
                    shortfalls.append({"route": route, "scene_id": f"S{number:02d}", "reason": "no_retained_pending_candidate"})
        r.base.write(run / "CANDIDATE_RECORDS.json", records)
        r.base.write(run / "CANDIDATE_HISTORY.json", {"root_id": root, "turns": history, "formal_baseline_turns": len(baseline),
            "history_policy": "formal_prefix_then_provisional_retained_dialogue", "training_pool_eligible": False})
        summary = metrics.summarize_records(records, count_text_tokens=token_counter(), baseline_turns=baseline)
        report = {"version": VERSION, "root_id": root, "scope": "fresh_collection", "metrics": summary, "shortfalls": shortfalls,
            "logical_calls": caller.request_counter, "http_attempts": caller.http_counter,
            "training_pool_eligible": False, "training_started": False, "question_calls": 0, "solver_calls": 0}
        r.base.write(run / "CANDIDATE_REPORT.json", report)
        verify(run)
        files = [p for p in run.rglob("*") if p.is_file()]
        r.base.write(run / "COLLECTION_BINDINGS.json", {"files": r.base.bindings(files), "semantic_certified": False})
        return report


def audit(run):
    """Replay artifact bindings, raw dialogue decisions, cumulative history and yield."""
    run = Path(run).resolve()
    manifest, allocation = verify(run)
    root = manifest["root_id"]
    r.base.verify_bindings(r.base.read(run / "COLLECTION_BINDINGS.json")["files"])
    records = r.base.read(run / "CANDIDATE_RECORDS.json")
    baseline = r.base.read(run / "FORMAL_BASELINE.json")["turns"]
    reservations = r.base.read(run / "SOURCE_RESERVATIONS.json")
    r.base.require(reservations["reservations"] == r.source_inventory._assignment_rows(run, allocation["assignments"]),
                   "candidate_reservation_changed")
    cards = []
    for i, assigned in enumerate(allocation["assignments"][root], 1):
        extracted = r.base.read(run / "extractions" / f"{i:02d}.json")
        r.base.require(extracted["source_id"] == assigned["source_id"], "candidate_extraction_source_changed")
        if extracted["card"]:
            r.source_inventory.verify_cards_for_root([extracted["card"]], allocation, root)
        cards.append(extracted["card"])
    history = deepcopy(baseline)
    for row in records:
        r.base.require(row["root_id"] == root, "candidate_record_root_changed")
        realization_raw = r.base.read(Path(row["request"]) / "RESPONSE.json")["text"]
        try:
            realization = r.parse_object(realization_raw)
        except (ValueError, TypeError):
            realization = None
        r.base.require(realization == row["realization"], "candidate_not_raw_model_output")
        folder = run / "candidates" / root / ("A" if row["route"] == r.ROUTES[0] else "B") / row["scene_id"]
        slot = r.base.read(folder / "INPUT.json")
        r.base.require(slot["history"] == history, "candidate_history_chain_changed")
        if row["route"] == r.ROUTES[1]:
            source_card = cards[int(row["scene_id"][1:]) - 1]
            r.base.require(source_card is not None and slot["card"] == source_card,
                           "candidate_card_binding_changed")
        else:
            r.base.require(slot["card"] is None, "A_candidate_has_B_card")
        raw_review = (r.base.read(Path(row["review_request"]) / "RESPONSE.json")["text"]
                      if row["review_request"] else None)
        r.base.require(raw_review == row["raw_review"], "raw_review_changed")
        expected = identity(root, row["route"], row["scene_id"])
        _parsed, failures, warnings, duplicate = inspect_realization(
            realization_raw, expected, history, baseline, slot['card'])
        decision = contract.decide_candidate(mechanical_failures=failures, duplicate_result=duplicate,
            review_raw=raw_review, review_error=row["review_error"], expected_identity=expected)
        r.base.require(decision == row["decision"] and decision["disposition"] == row["disposition"], "candidate_decision_changed")
        if decision["continue_candidate_history"]:
            history = append_candidate(history, root, row["route"], realization)
    r.base.require(history == r.base.read(run / "CANDIDATE_HISTORY.json")["turns"], "candidate_cumulative_changed")
    report = r.base.read(run / "CANDIDATE_REPORT.json")
    requests = list((run / "requests").glob("*/REQUEST.json"))
    attempts = list((run / "requests").glob("*/HTTP_ATTEMPT*.json"))
    r.base.require(report["logical_calls"] == len(requests) <= manifest["logical_max"]
                   and report["http_attempts"] == len(attempts) <= manifest["http_max"], "candidate_budget_receipts_changed")
    from candidate_request_audit import audit_requests
    audit_requests(run, records, baseline)
    metrics.verify_summary(report["metrics"], records, count_text_tokens=token_counter(), baseline_turns=baseline)
    return {"status": "ARTIFACT_REPLAY_VERIFIED", "semantic_certified": False,
            "training_pool_eligible": False, "metrics": report["metrics"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "live", "audit"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--routes", choices=("A", "B", "both"), default="both")
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--root", choices=ROOTS, default=ROOT)
    args = parser.parse_args()
    if args.command == "prepare":
        routes = r.ROUTES if args.routes == "both" else (r.ROUTES[args.routes == "B"],)
        result = prepare(args.run, root=args.root, routes=routes, scenes=args.scenes)
    elif args.command == "verify":
        manifest, allocation = verify(args.run)
        result = {"status": "PREPARED_BINDINGS_VERIFIED", "logical_max": manifest["logical_max"],
                  "http_max": manifest["http_max"], "assigned_B_slots": allocation["assigned_slots"]}
    else:
        result = globals()[args.command](args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
