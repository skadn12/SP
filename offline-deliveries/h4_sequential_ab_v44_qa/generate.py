"""Bounded Qwen QA probe over full v43 provisional candidate history.

prepare/verify are offline. generate creates two multi-select and two ordering
slots using the established v15 contract; scenes remain provisional throughout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path

import collection_source
import prompt_rules
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
QA = DELIVERIES / "h4_feedback_training_pool_v15_qa"
sys.path.insert(0, str(QA))
import qa as v15

legacy, base = v15.legacy, v15.base
read, write, require = base.read, base.write, base.require
VERSION = "h4_sequential_ab_v44_generation_v1"
EXPANSION_VERSION = "h4_candidate_collection_v43"
MODEL, TARGET = v15.MODEL, v15.TARGET
TYPES = ("multi_select", "ordering")
INTERVAL = 35
DEFAULT_PRIOR = (DELIVERIES / "h4_root_merged_context_v17/runs/"
                 "merged_latest_02/training_pool/TRAINING_POOL.jsonl")

MERGED_INSTRUCTION = """
FULL CUMULATIVE ROOT CONTEXT:
All source_turns belong to one root: the complete formal baseline followed by
retained provisional B dialogue. The supplied dialogue is the evidence for this
QA probe; its provisional suffix has NOT been admitted to training. Do not claim
that the source scenes are approved, externally factual, or semantically certified.
Use the actual supplied words, without importing source cards, plans or reviews.
Favor questions grounded in the new CANDIDATE:B scenes so this small probe tests
what the added dialogue contributes. Keep each question coherent; abstain rather
than inventing a fact to fill a slot. New-scene evidence is diagnostic, not a new
mechanical admission gate. The complete source remains available, without truncation.
Scene storage order does not prove real-world cross-scene event chronology.
Ordering questions retain the explicit conversation-order scope: when the
statements APPEAR IN THE SUPPLIED DIALOGUE. Do not silently replace it with
real-world chronological inference. Preserve narrow coherent question scope.
Prior question inventory contains no current Solver scores, answers or Gold.
Supported paraphrases and partial overlap are allowed as training augmentation.
Avoid an exact copy of an entire six-statement set.
"""


def checked_sources(path, requested_roots=None):
    row = collection_source.checked_source(path, base, requested_roots=requested_roots)
    return read(path.parent / "COLLECTION_BINDINGS.json"), [row], []


def checked_expansion(history_path):
    return history_path.parent / "COLLECTION_BINDINGS.json", history_path.parent / "MANIFEST.json"


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
        "It includes the full formal baseline and retained provisional dialogue in one root.")
    common = common.replace(
        "Do not create another copy of\nthe same fact set by changing wording, negation or display labels.",
        "Supported paraphrases are allowed; avoid exact six-statement copies.")
    common = common.replace("Prioritize new R02 developments", "Prioritize newly supplied CANDIDATE developments")
    require("Do not create another copy" not in common, "legacy_paraphrase_ban_not_removed")
    instructions = (common + MERGED_INSTRUCTION
                    + (legacy.MULTI if qtype == "multi_select" else legacy.ORDERING)
                    + prompt_rules.generation_guidance(qtype))
    return legacy.payload_with_feedback(run, root, instructions, context,
        min(8192, 1024 + 2048 * len(context["candidate_ids"])))


def make_review_payload(run, root, candidate, source):
    original = legacy.ordering_gate.build_review_prompt(candidate, source)
    prefix, separator, data = original.partition("\nDATA:\n")
    require(bool(separator), "ordering_review_data_marker_missing")
    content = prefix + prompt_rules.review_guidance() + separator + data
    payload = {"model": MODEL, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}]}
    return legacy.feedback.append_feedback_to_request(payload, run / "feedback", root, "writer", TARGET)


def _dependency_paths():
    profile, factory, annotate, capacity, dev = base.transport()
    paths = {Path(__file__).resolve(), Path(legacy.__file__), Path(base.__file__),
             Path(v15.__file__), Path(v15.dedup.__file__), Path(legacy.feedback.__file__),
             Path(legacy.multi.__file__), Path(legacy.ordering_gate.__file__), legacy.ordering_gate.PROMPTS_PATH}
    paths.update(QA.glob("*.py"))
    paths.update(HERE.glob("*.py"))
    for obj in (profile, factory, annotate, capacity, dev):
        value = inspect.getsourcefile(obj) or inspect.getfile(obj)
        if value:
            paths.add(Path(value))
    paths.update(path for path in base.TOKENIZER.rglob("*") if path.is_file())
    return profile(), paths


def prepare(run, merge_manifest, *, roots=None, slots=2, prior_pools=()):
    require(not run.exists(), "generation_run_already_exists")
    require(type(slots) is int and 1 <= slots <= 8, "slots_outside_1_8")
    merge, sources, excluded = checked_sources(merge_manifest, roots)
    feedback_lock = read(legacy.FEEDBACK / "source_lock.json")
    require(feedback_lock.get("status") == "EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY", "feedback_round_not_frozen")
    require(feedback_lock.get("merge_current_generation_feedback") is False, "current_feedback_not_allowed")
    require(feedback_lock["model_identity_source"]["sha256"] == v15.evaluator.TRAINED_AUDIT_SHA256,
            "feedback_and_solver_checkpoint_differ")
    provider, dependencies = _dependency_paths()
    audit_path, upstream_path = checked_expansion(merge_manifest)
    frozen = set(dependencies) | {merge_manifest, audit_path, upstream_path}
    frozen.update(Path(binding["path"]).resolve() for binding in merge.get("files", []))
    for path in sorted(legacy.FEEDBACK.rglob("*")):
        if path.is_file():
            target = run / "feedback" / path.relative_to(legacy.FEEDBACK)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            frozen.update((path, target))
    prior_paths = [path.resolve() for path in prior_pools]
    if DEFAULT_PRIOR.resolve() not in prior_paths:
        prior_paths.insert(0, DEFAULT_PRIOR.resolve())
    require(all(path.is_file() for path in prior_paths), "explicit_prior_pool_missing")
    source_rows = []
    for row in sources:
        root = row["root_id"]
        target = run / "sources" / (root + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        write(target, row["source"])
        prior, paths = v15.prior_candidates(root)
        for path in prior_paths:
            prior += _prior_pool_rows(path, root)
        write(run / "prior" / (root + ".json"), prior)
        legacy.feedback.build_feedback_suffix(run / "feedback", root, "writer", TARGET)
        frozen.update([row["path"], target, run / "prior" / (root + ".json"), *paths, *prior_paths])
        source_rows.append({"root_id": root, "source_path": str(target.resolve()),
            "source_sha256": base.digest(target.read_bytes()), "merge_source_path": str(row["path"]),
            "candidate_history_sha256": row["sha256"], "candidate_source_training_pool_eligible": False,
            "retained_candidate_scenes": row["retained_candidate_scenes"],
            "new_admitted_scenes": row["new_admitted_scenes"], "turn_count": len(row["source"]["turns"]),
            "merge_record": row["merge_record"], "qa_status": "READY_PROVISIONAL_CANDIDATE_HISTORY"})
    write(run / "provider_profile.json", provider)
    frozen.add(run / "provider_profile.json")
    write(run / "UPSTREAM_ARTIFACT_AUDIT.json", sources[0]["audit"])
    frozen.add(run / "UPSTREAM_ARTIFACT_AUDIT.json")
    manifest = {"version": VERSION, "prepared_at_utc": v15.now(),
        "merge_manifest_path": str(merge_manifest),
        "candidate_history_path": str(merge_manifest), "source_training_pool_eligible": False,
        "source_semantic_certified": False, "upstream_scene_promotion": False,
        "training_allowed": False, "source_admission_status": "PROVISIONAL_CANDIDATE_HISTORY",
        "upstream_admitted_scenes": 0, "merge_manifest_sha256": base.digest(merge_manifest.read_bytes()),
        "expansion_final_audit_path": str(audit_path), "expansion_final_audit_sha256": base.digest(audit_path.read_bytes()),
        "expansion_manifest_path": str(upstream_path), "expansion_manifest_sha256": base.digest(upstream_path.read_bytes()),
        "roots": [row["root_id"] for row in source_rows], "source_rows": source_rows,
        "excluded_merge_rows": excluded, "source_mode": "one_full_provisional_candidate_context_per_root",
        "branch_separated_question_pools": False, "generator_model": MODEL, "reviewer_model": MODEL,
        "solver_model": TARGET, "question_types": list(TYPES), "candidate_slots_per_type_per_root": slots,
        "budget": v15.budget(len(sources), slots), "minimum_seconds_after_call_completion": INTERVAL,
        "automatic_transport_retries": 0, "mechanical_repair_attempts_max": 1,
        "semantic_multi_review": False, "ordering_independent_qwen_review": True,
        "ordering_anchor_contract": "unique_exact_nonoverlapping_clause_spans_v1",
        "ordering_scope": "conversation_order_not_real_world_event_chronology",
        "paraphrases_allowed": True, "semantic_novelty_gate": False,
        "dedup_hard_scope": "exact_entire_six_statement_sets_only",
        "new_scene_evidence_required": False, "new_scene_evidence_priority": True, "codex_question_edits": False,
        "feedback_round": "previous_training_round_only", "current_solver_outputs_in_prompts": False,
        "full_policy_version": v15.POLICY, "selection_rule": "complete_valid_full_set and 0 < full_partial_mean < 1",
        "full_slots": {"multi_select": "6 random permutations x 3 full rollouts",
                         "ordering": "12 random permutations x 1 full rollout"},
        "ground_calls": 0, "training_calls": 0, "training_started": False,
        "merged_instruction": MERGED_INSTRUCTION,
        "prompt_repair_contract": prompt_rules.CONTRACT, "files": base.bindings(frozen)}
    write(run / "MANIFEST.json", manifest)
    verify(run)
    return {key: manifest[key] for key in ("version", "roots", "source_rows", "excluded_merge_rows", "budget")}


def verify(run):
    manifest = read(run / "MANIFEST.json")
    require(manifest["version"] == VERSION and manifest["merged_instruction"] == MERGED_INSTRUCTION,
            "generation_version_or_prompt_changed")
    require(manifest.get("prompt_repair_contract") == prompt_rules.CONTRACT,
            "generation_prompt_repair_contract_changed")
    require(manifest["paraphrases_allowed"] is True and manifest["semantic_multi_review"] is False,
            "question_policy_changed")
    require(manifest["ground_calls"] == manifest["training_calls"] == 0, "unrequested_stage")
    require(manifest["source_training_pool_eligible"] is False
            and manifest["source_semantic_certified"] is False
            and manifest["upstream_scene_promotion"] is False
            and manifest["training_allowed"] is False
            and manifest["source_admission_status"] == "PROVISIONAL_CANDIDATE_HISTORY"
            and manifest["upstream_admitted_scenes"] == 0, "candidate_status_promoted")
    require(manifest["budget"] == v15.budget(len(manifest["roots"]), manifest["candidate_slots_per_type_per_root"]), "budget_changed")
    base.verify_bindings(manifest["files"])
    _, sources, _ = checked_sources(Path(manifest["candidate_history_path"]), manifest["roots"])
    require(len(sources) == len(manifest["source_rows"]), "candidate_source_count_changed")
    for source, row in zip(sources, manifest["source_rows"]):
        local = Path(row["source_path"])
        require(read(local) == source["source"] and base.digest(local.read_bytes()) == row["source_sha256"],
                "candidate_local_source_changed")
    return manifest


def _generate_locked(run):
    manifest = verify(run)
    require(not (run / "STARTED.json").exists(), "generation_already_started_no_replay")
    from qa_transport import Caller
    write(run / "STARTED.json", {"started_at_utc": v15.now(), "http_max": manifest["budget"]["http_attempts_max"]})
    caller = Caller(run)
    call = caller.call

    ready, mechanical, outcomes = [], [], []
    for source_row in manifest["source_rows"]:
        root, current = source_row["root_id"], []
        source_path = Path(source_row["source_path"])
        source, prior = read(source_path), read(run / "prior" / (root + ".json"))
        for qtype in TYPES:
            code = "multi" if qtype == "multi_select" else "order"
            slots = manifest["candidate_slots_per_type_per_root"]
            tag = re.sub(r"[^a-zA-Z0-9_-]", "_", run.name)
            ids = [f"{root}-v44-{tag}-merged-{code}-b01-{index:02d}" for index in range(1, slots + 1)]
            context = {"root_id": root, "context_mode": "full_provisional_candidate_root",
                "candidate_ids": ids, "source_turns": source["turns"],
                "previous_candidates_dedup_only": prompt_prior(prior + legacy.compact_questions(current)),
                "priority_new_scene_refs": sorted({t.get("scene_ref", t["turn_ref"].rsplit(":", 1)[0])
                    for t in source["turns"] if t["turn_ref"].startswith("CANDIDATE:")}),
                "new_scene_evidence_required": False, "paraphrase_augmentation_allowed": True}
            accepted, rejected, abstentions = [], [], []
            pending = ids[:]
            for attempt in (1, 2):
                payload, receipt = make_payload(run, root, qtype, context)
                raw, directory, metadata = call(f"{root.split('-')[-1]}-merged-{code}-generate-{attempt}", payload,
                    {"root_id": root, "source_mode": "full_provisional_candidate_root", "role": "question_generator",
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
                        source_training_pool_eligible=False, source_semantic_certified=False,
                        upstream_scene_promotion=False, training_allowed=False,
                        source_admission_status="PROVISIONAL_CANDIDATE_HISTORY", upstream_admitted_scenes=0,
                        new_scene_reference_diagnostic=any(ref.startswith("CANDIDATE:") for ref in refs),
                        evidence_routes_diagnostic=sorted({ref.split(":")[1] for ref in refs
                            if ref.startswith(("CANDIDATE:", "R02:")) and len(ref.split(":")) > 1}))
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
                    payload, receipt = make_review_payload(run, root, candidate, source)
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
            "source_mode": "one_full_provisional_candidate_context_per_root",
            "source_context_version": VERSION, "feedback_target_solver": TARGET},
        "ground_calls": 0, "training_started": False, "source_training_pool_eligible": False,
        "upstream_scene_promotion": False})
    report = {"version": VERSION, "http_attempts": len(list((run / "requests").glob("*/HTTP_ATTEMPT.json"))),
        "requests_prepared": caller.counter, "candidate_slots": manifest["budget"]["candidate_slots"],
        "mechanical_pass": len(mechanical), "ready_for_solver": len(ready), "roots": outcomes,
        "feedback_in_every_request": True, "semantic_novelty_gate": False, "paraphrases_allowed": True,
        "no_current_solver_feedback_in_generation": True, "ground_calls": 0, "training_calls": 0}
    write(run / "REPORT.json", report)
    return report


def generate(run):
    from qa_transport import shared_lock
    with shared_lock():
        return _generate_locked(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "generate"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--candidate-history", "--merge-manifest", dest="merge_manifest", type=Path)
    parser.add_argument("--roots", nargs="+")
    parser.add_argument("--slots", type=int, default=2, help="candidate slots per type/root; not a selected quota")
    parser.add_argument("--prior-pool", action="append", type=Path, default=[])
    args = parser.parse_args()
    run = args.run.resolve()
    require(run.is_relative_to(HERE / "runs"), "run_must_be_in_v44_qa_runs")
    if args.action == "prepare":
        require(args.merge_manifest is not None, "prepare_requires_candidate_history")
        result = prepare(run, args.merge_manifest.resolve(), roots=args.roots, slots=args.slots, prior_pools=args.prior_pool)
    elif args.action == "verify":
        manifest = verify(run)
        result = {"status": "PASS", "roots": manifest["roots"], "budget": manifest["budget"]}
    else:
        result = generate(run)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
