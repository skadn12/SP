"""Generate Qwen questions from checked, cumulative v17 root contexts.

prepare/verify are offline. Only generate can call the provider; neither phase
calls a Solver, expands dialogue, or starts training. Historical scripts and
artifacts are imported read-only. A and B are generation provenance, never
separate downstream context pools here.

Input MERGE_MANIFEST.json has source_rows containing root_id, path, sha256,
accepted_scenes and status=PASS (or top-level status=PASS). Paths may be absolute
or relative to the merge manifest. Non-PASS rows are reported, never inferred
to have passed. A checked base-only source is permitted. Manifest files bindings
are verified when provided; source and manifest bytes are frozen again locally.
Only frozen previous-training-round feedback enters requests. Current Solver
scores, answers and Gold never enter the previous-question inventory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import re
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
QA = DELIVERIES / "h4_feedback_training_pool_v15_qa"
sys.path.insert(0, str(QA))
import qa as v15

legacy, base = v15.legacy, v15.base
read, write, require = base.read, base.write, base.require
VERSION = "h4_root_merged_context_v17_generation_v1"
MODEL, TARGET = v15.MODEL, v15.TARGET
TYPES = ("multi_select", "ordering")
INTERVAL = 35
DEFAULT_PRIOR = (DELIVERIES / "h4_r02_existing_source_yield_v16/runs/"
                 "same_sources_additional_01_training_pool/TRAINING_POOL.jsonl")

MERGED_INSTRUCTION = """
CHECKED MERGED ROOT CONTEXT:
All source_turns belong to one cumulative root context. A and B were generated
independently from the same frozen base, then checked before merge. Both routes
are available as evidence; neither route is a separate question pool.
Scene storage order does not prove real-world cross-scene event chronology.
Ordering questions retain the explicit conversation-order scope: when the
statements APPEAR IN THE SUPPLIED DIALOGUE. Do not silently replace it with
real-world chronological inference. Preserve narrow coherent question scope.
Prior question inventory contains no current Solver scores, answers or Gold.
Supported paraphrases are allowed as training augmentation and are not rejected
merely for testing a previously used fact. Prefer diverse supported wording and
coverage where available, but do not impose a new semantic novelty condition.
Avoid an exact copy of an entire six-statement set; partial overlap is allowed.
New R02 evidence is a priority and diagnostic, not an admission requirement.
"""


def _resolve(path, parent):
    value = Path(path)
    return (value if value.is_absolute() else parent / value).resolve()


def checked_sources(path, requested_roots=None):
    manifest = read(path)
    for binding in manifest.get("files", []):
        bound = _resolve(binding["path"], path.parent)
        require(base.digest(bound.read_bytes()) == binding["sha256"], "merge_manifest_binding_changed:" + str(bound))
    rows = manifest.get("source_rows")
    require(isinstance(rows, list) and rows, "merge_source_rows_missing")
    ids = [row.get("root_id") for row in rows]
    require(all(isinstance(root, str) and root for root in ids), "invalid_merge_root_id")
    require(len(set(ids)) == len(ids), "multiple_merge_sources_per_root")
    requested = set(requested_roots or ids)
    require(requested <= set(ids), "requested_root_not_in_merge_manifest")
    sources, excluded = [], []
    for row in rows:
        root = row["root_id"]
        if root not in requested:
            continue
        status = row.get("status", row.get("merge_status", manifest.get("status")))
        if status != "PASS":
            excluded.append({"root_id": root, "status": status,
                             "reason": "MERGE_NOT_EXPLICITLY_PASS"})
            continue
        source_path = _resolve(row.get("path", row.get("source_path", "")), path.parent)
        expected_sha = row.get("sha256", row.get("source_sha256"))
        require(source_path.is_file(), "merged_source_missing:" + root)
        require(base.digest(source_path.read_bytes()) == expected_sha, "merged_source_hash_mismatch:" + root)
        source = read(source_path)
        require(source.get("root_id") == root, "merged_source_root_mismatch:" + root)
        turns = source.get("turns")
        require(isinstance(turns, list) and turns, "merged_source_empty:" + root)
        require(all(isinstance(turn, dict) and all(isinstance(turn.get(k), str) and turn[k].strip()
                for k in ("turn_ref", "speaker", "text")) for turn in turns), "merged_turn_schema:" + root)
        require(len({turn["turn_ref"] for turn in turns}) == len(turns), "merged_duplicate_turn_refs:" + root)
        count = row.get("accepted_scenes")
        require(type(count) is int and count >= 0, "accepted_scene_count_missing:" + root)
        sources.append({"root_id": root, "path": source_path, "sha256": expected_sha,
                        "accepted_scenes": count, "source": source, "merge_record": deepcopy(row)})
    require(sources, "no_explicitly_passed_merged_sources")
    return manifest, sources, excluded


def _prior_pool_rows(path, root):
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        obj = read(path)
        rows = obj if isinstance(obj, list) else obj.get("questions", obj.get("candidates", []))
    result = []
    for original in rows:
        question = original.get("original_selected_question", original)
        if question.get("root_id") != root:
            continue
        values = question.get("options") or question.get("items")
        require(isinstance(values, dict) and all(isinstance(t, str) for t in values.values()),
                "prior_pool_statement_text_missing:" + str(path))
        result.append({"candidate_id": question["candidate_id"], "root_id": root,
                       "question_type": question["question_type"], "question": question["question"],
                       "facts": list(values.values()), "hard_dedup": True,
                       "history_status": "EXPLICIT_PRIOR_POOL_EXACT_SET_ONLY",
                       "source_path": str(path), "source_sha256": base.digest(path.read_bytes())})
    return result


def prompt_prior(rows):
    result = legacy.prompt_prior(rows)
    for row in result:
        row["dedup_action"] = ("avoid_exact_six_statement_set_only_paraphrases_allowed"
                               if row["dedup_action"] == "avoid_already_selected" else "warning_only")
    return result


def make_payload(run, root, qtype, context):
    common = legacy.COMMON.replace(
        "It includes P, R01, and recovered R02 A/B turns.",
        "It includes the frozen base and all accepted, checked A/B turns in one merged root.")
    common = common.replace(
        "Do not create another copy of\nthe same fact set by changing wording, negation or display labels.",
        "Supported paraphrases are allowed; avoid exact six-statement copies.")
    require("Do not create another copy" not in common, "legacy_paraphrase_ban_not_removed")
    instructions = common + MERGED_INSTRUCTION + (legacy.MULTI if qtype == "multi_select" else legacy.ORDERING)
    return legacy.payload_with_feedback(run, root, instructions, context,
        min(8192, 1024 + 2048 * len(context["candidate_ids"])))


def _dependency_paths():
    profile, factory, annotate, capacity, dev = base.transport()
    paths = {Path(__file__).resolve(), Path(legacy.__file__), Path(base.__file__),
             Path(v15.__file__), Path(v15.dedup.__file__), Path(legacy.feedback.__file__),
             Path(legacy.multi.__file__), Path(legacy.ordering_gate.__file__), legacy.ordering_gate.PROMPTS_PATH}
    paths.update(QA.glob("*.py"))
    for obj in (profile, factory, annotate, capacity, dev):
        value = inspect.getsourcefile(obj) or inspect.getfile(obj)
        if value:
            paths.add(Path(value))
    paths.update(path for path in base.TOKENIZER.rglob("*") if path.is_file())
    return profile(), paths


def prepare(run, merge_manifest, *, roots=None, slots=4, prior_pools=()):
    require(not run.exists(), "generation_run_already_exists")
    require(type(slots) is int and 1 <= slots <= 8, "slots_outside_1_8")
    merge, sources, excluded = checked_sources(merge_manifest, roots)
    feedback_lock = read(legacy.FEEDBACK / "source_lock.json")
    require(feedback_lock.get("status") == "EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY", "feedback_round_not_frozen")
    require(feedback_lock.get("merge_current_generation_feedback") is False, "current_feedback_not_allowed")
    require(feedback_lock["model_identity_source"]["sha256"] == v15.evaluator.TRAINED_AUDIT_SHA256,
            "feedback_and_solver_checkpoint_differ")
    provider, dependencies = _dependency_paths()
    frozen = set(dependencies) | {merge_manifest}
    for path in sorted(legacy.FEEDBACK.rglob("*")):
        if path.is_file():
            target = run / "feedback" / path.relative_to(legacy.FEEDBACK)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            frozen.update((path, target))
    prior_paths = [path.resolve() for path in prior_pools]
    if DEFAULT_PRIOR.exists() and DEFAULT_PRIOR.resolve() not in prior_paths:
        prior_paths.insert(0, DEFAULT_PRIOR.resolve())
    require(all(path.is_file() for path in prior_paths), "explicit_prior_pool_missing")
    source_rows = []
    for row in sources:
        root = row["root_id"]
        target = run / "sources" / (root + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
        prior, paths = v15.prior_candidates(root)
        for path in prior_paths:
            prior += _prior_pool_rows(path, root)
        write(run / "prior" / (root + ".json"), prior)
        legacy.feedback.build_feedback_suffix(run / "feedback", root, "writer", TARGET)
        frozen.update([row["path"], target, run / "prior" / (root + ".json"), *paths, *prior_paths])
        source_rows.append({"root_id": root, "source_path": str(target.resolve()),
            "source_sha256": row["sha256"], "merge_source_path": str(row["path"]),
            "accepted_scenes": row["accepted_scenes"], "turn_count": len(row["source"]["turns"]),
            "merge_record": row["merge_record"], "qa_status": "READY_CHECKED_MERGED_ROOT"})
    write(run / "provider_profile.json", provider)
    frozen.add(run / "provider_profile.json")
    manifest = {"version": VERSION, "prepared_at_utc": v15.now(),
        "merge_manifest_path": str(merge_manifest), "merge_manifest_sha256": base.digest(merge_manifest.read_bytes()),
        "roots": [row["root_id"] for row in source_rows], "source_rows": source_rows,
        "excluded_merge_rows": excluded, "source_mode": "one_checked_cumulative_context_per_root",
        "branch_separated_question_pools": False, "generator_model": MODEL, "reviewer_model": MODEL,
        "solver_model": TARGET, "question_types": list(TYPES), "candidate_slots_per_type_per_root": slots,
        "budget": v15.budget(len(sources), slots), "minimum_seconds_after_call_completion": INTERVAL,
        "automatic_transport_retries": 0, "mechanical_repair_attempts_max": 1,
        "semantic_multi_review": False, "ordering_independent_qwen_review": True,
        "ordering_anchor_contract": "unique_exact_nonoverlapping_clause_spans_v1",
        "ordering_scope": "conversation_order_not_real_world_event_chronology",
        "paraphrases_allowed": True, "semantic_novelty_gate": False,
        "dedup_hard_scope": "exact_entire_six_statement_sets_only",
        "new_scene_evidence_required": False, "codex_question_edits": False,
        "feedback_round": "previous_training_round_only", "current_solver_outputs_in_prompts": False,
        "full_policy_version": v15.POLICY, "selection_rule": "complete_valid_full_set and 0 < full_partial_mean < 1",
        "full_slots": {"multi_select": "6 random permutations x 3 full rollouts",
                         "ordering": "12 random permutations x 1 full rollout"},
        "ground_calls": 0, "training_calls": 0, "training_started": False,
        "merged_instruction": MERGED_INSTRUCTION, "files": base.bindings(frozen)}
    write(run / "MANIFEST.json", manifest)
    verify(run)
    return {key: manifest[key] for key in ("version", "roots", "source_rows", "excluded_merge_rows", "budget")}


def verify(run):
    manifest = read(run / "MANIFEST.json")
    require(manifest["version"] == VERSION and manifest["merged_instruction"] == MERGED_INSTRUCTION,
            "generation_version_or_prompt_changed")
    require(manifest["paraphrases_allowed"] is True and manifest["semantic_multi_review"] is False,
            "question_policy_changed")
    require(manifest["ground_calls"] == manifest["training_calls"] == 0, "unrequested_stage")
    base.verify_bindings(manifest["files"])
    checked_sources(Path(manifest["merge_manifest_path"]), manifest["roots"])
    return manifest


def generate(run):
    manifest = verify(run)
    require(not (run / "STARTED.json").exists(), "generation_already_started_no_replay")
    legacy.MAX_HTTP = manifest["budget"]["http_attempts_max"]
    write(run / "STARTED.json", {"started_at_utc": v15.now(), "http_max": legacy.MAX_HTTP})
    caller = legacy.Caller(run)
    last_end = None

    def call(label, payload, proof):
        nonlocal last_end
        while last_end is not None and time.monotonic() - last_end < INTERVAL:
            time.sleep(min(INTERVAL, INTERVAL - (time.monotonic() - last_end)))
        started, started_at = time.monotonic(), v15.now()
        result = caller.call(label, payload, proof)
        ended = time.monotonic()
        write(result[1] / "TIMING.json", {"started_at_utc": started_at, "finished_at_utc": v15.now(),
            "interval_since_previous_call_completion_seconds": None if last_end is None else started - last_end,
            "elapsed_seconds": ended - started})
        last_end = ended
        return result

    ready, mechanical, outcomes = [], [], []
    for source_row in manifest["source_rows"]:
        root, current = source_row["root_id"], []
        source_path = Path(source_row["source_path"])
        source, prior = read(source_path), read(run / "prior" / (root + ".json"))
        for qtype in TYPES:
            code = "multi" if qtype == "multi_select" else "order"
            slots = manifest["candidate_slots_per_type_per_root"]
            tag = re.sub(r"[^a-zA-Z0-9_-]", "_", run.name)
            ids = [f"{root}-v17-{tag}-merged-{code}-b01-{index:02d}" for index in range(1, slots + 1)]
            context = {"root_id": root, "context_mode": "checked_merged_root",
                "candidate_ids": ids, "source_turns": source["turns"],
                "previous_candidates_dedup_only": prompt_prior(prior + legacy.compact_questions(current)),
                "priority_new_scene_refs": sorted({t.get("scene_ref", t["turn_ref"].rsplit(":", 1)[0])
                    for t in source["turns"] if t["turn_ref"].startswith("R02:")}),
                "new_scene_evidence_required": False, "paraphrase_augmentation_allowed": True}
            accepted, rejected, abstentions = [], [], []
            pending = ids[:]
            for attempt in (1, 2):
                payload, receipt = make_payload(run, root, qtype, context)
                raw, directory, metadata = call(f"{root.split('-')[-1]}-merged-{code}-generate-{attempt}", payload,
                    {"root_id": root, "source_mode": "checked_merged_root", "role": "question_generator",
                     "question_type": qtype, "candidate_ids": pending,
                     "source_sha256": source_row["source_sha256"], "feedback_receipt": receipt})
                if raw is None:
                    rejected = [{"candidate_id": qid, "errors": [read(directory / "ERROR.json")["error_type"]]}
                                for qid in pending]
                    break
                valid, rejected, abstained, envelope_error = legacy.validate_draft(
                    raw, pending, root, qtype, source, source_path)
                evidence_key = "option_evidence" if qtype == "multi_select" else "item_evidence"
                for question in valid:
                    refs = [entry.get("turn_ref", "") for entry in question[evidence_key].values() if isinstance(entry, dict)]
                    question.update(generator_model=MODEL, generation_request_path=str(directory / "REQUEST.json"),
                        generation_response_path=str(directory / "RESPONSE.json"), raw_question_content_preserved=True,
                        expansion_route="merged_root", expansion_branch="MERGED",
                        source_context_version=VERSION, merged_source_manifest_path=manifest["merge_manifest_path"],
                        new_scene_reference_diagnostic=any(ref.startswith("R02:") for ref in refs),
                        evidence_routes_diagnostic=sorted({ref.split(":")[1] for ref in refs
                            if ref.startswith("R02:") and len(ref.split(":")) > 1}))
                accepted += valid
                abstentions += abstained
                write(directory / "VALIDATION.json", {"accepted_ids": [q["candidate_id"] for q in valid],
                    "rejected": rejected, "abstentions": abstained, "envelope_error": envelope_error,
                    "semantic_review": False, "new_scene_evidence_check": "diagnostic_only"})
                if not rejected:
                    break
                pending = [row["candidate_id"] for row in rejected]
                repair = {"instruction": "Repair only listed mechanical/source-binding errors; preserve accepted slots.",
                          "failures": legacy.repair_failures(rejected)}
                if envelope_error:
                    repair["envelope_error"] = envelope_error
                context = {**context, "candidate_ids": pending, "mechanical_repair": repair,
                    "previous_candidates_dedup_only": prompt_prior(prior + legacy.compact_questions(current + accepted))}
            accepted, duplicates = v15.dedup.deduplicate(accepted, prior, current)
            rejected += duplicates
            current += accepted
            mechanical += accepted
            outcome = {"root_id": root, "question_type": qtype, "candidate_slots": slots,
                "mechanical_pass": len(accepted), "mechanical_rejected": rejected, "abstentions": abstentions,
                "review_rejected": [], "actual_generation_requests": attempt}
            reviewed = []
            if qtype == "ordering":
                for candidate in accepted:
                    payload = {"model": MODEL, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
                        "response_format": {"type": "json_object"}, "messages": [{"role": "user",
                        "content": legacy.ordering_gate.build_review_prompt(candidate, source)}]}
                    payload, receipt = legacy.feedback.append_feedback_to_request(payload, run / "feedback", root, "writer", TARGET)
                    raw, directory, metadata = call(f"{root.split('-')[-1]}-merged-review-{candidate['candidate_id'].split('-')[-1]}",
                        payload, {"root_id": root, "role": "ordering_reviewer", "candidate_id": candidate["candidate_id"],
                        "feedback_receipt": receipt, "independent_request_without_prior_verdicts": True,
                        "source_sha256": source_row["source_sha256"]})
                    if raw is None:
                        outcome["review_rejected"].append({"candidate_id": candidate["candidate_id"], "reasons": ["review_request_failed"]})
                        continue
                    checked = legacy.ordering_gate.review_response(raw, [candidate], source,
                        directory / "RESPONSE.json", directory / "REQUEST.json", metadata)
                    write(directory / "REVIEW.json", checked)
                    reviewed += checked["accepted"]
                    outcome["review_rejected"] += checked["rejected"]
            else:
                reviewed = accepted
            ready += reviewed
            outcome["ready_for_solver"] = len(reviewed)
            outcomes.append(outcome)
            write(run / "outcomes" / f"{root}.{qtype}.json", outcome)
            print(json.dumps({"event": "merged_root_type_done", "root": root,
                              "question_type": qtype, "ready_for_solver": len(reviewed)}), flush=True)
    verify(run)
    for directory in sorted((run / "requests").glob("*")):
        payload, proof = read(directory / "REQUEST.json"), read(directory / "PROOF.json")
        suffix = legacy.feedback.build_feedback_suffix(run / "feedback", proof["root_id"], "writer", TARGET)
        require(payload["messages"][0]["content"].endswith(suffix), "feedback_missing_from_actual_payload")
        require(proof["payload_sha256"] == base.digest(payload), "frozen_payload_changed")
        require(proof["feedback_receipt"]["feedback_appended"] is True, "feedback_receipt_missing")
    write(run / "MECHANICAL_POOL.json", {"questions": mechanical})
    write(run / "SOLVER_POOL.json", {"version": VERSION, "candidate_count": len(ready), "questions": ready,
        "provenance": {"requested_model": MODEL, "returned_model": MODEL, "codex_question_edits": False,
            "multi_select_semantic_review": False, "ordering_review_owner": MODEL,
            "source_mode": "one_checked_cumulative_context_per_root",
            "source_context_version": VERSION, "feedback_target_solver": TARGET},
        "ground_calls": 0, "training_started": False})
    report = {"version": VERSION, "http_attempts": len(list((run / "requests").glob("*/HTTP_ATTEMPT.json"))),
        "requests_prepared": caller.counter, "candidate_slots": manifest["budget"]["candidate_slots"],
        "mechanical_pass": len(mechanical), "ready_for_solver": len(ready), "roots": outcomes,
        "feedback_in_every_request": True, "semantic_novelty_gate": False, "paraphrases_allowed": True,
        "no_current_solver_feedback_in_generation": True, "ground_calls": 0, "training_calls": 0}
    write(run / "REPORT.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "generate"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--merge-manifest", type=Path)
    parser.add_argument("--roots", nargs="+")
    parser.add_argument("--slots", type=int, default=4, help="candidate slots per type/root; not a selected quota")
    parser.add_argument("--prior-pool", action="append", type=Path, default=[])
    args = parser.parse_args()
    run = args.run.resolve()
    require(run.is_relative_to(HERE / "runs"), "run_must_be_in_v17_runs")
    if args.action == "prepare":
        require(args.merge_manifest is not None, "prepare_requires_merge_manifest")
        result = prepare(run, args.merge_manifest.resolve(), roots=args.roots, slots=args.slots, prior_pools=args.prior_pool)
    elif args.action == "verify":
        manifest = verify(run)
        result = {"status": "PASS", "roots": manifest["roots"], "budget": manifest["budget"]}
    else:
        result = generate(run)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
