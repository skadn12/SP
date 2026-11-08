"""One bounded recovery of five frozen HTTP 429/50602 requests, no other replay.

prepare makes a derived QA run without inference. live repeats each failed
payload once and independently reviews newly generated ordering candidates.
Use ordinary qa.py Solver/delivery and audit_run.py commands afterward.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import qa

HERE = Path(__file__).resolve().parent
VERSION = "v15_http429_50602_single_retry_v1"
EXPECTED = {
    ("A", "cc-train-episode-138343", "ordering_reviewer", "03"),
    ("A", "cc-train-episode-138343", "ordering_reviewer", "04"),
    ("B", "cc-train-episode-135466", "ordering_reviewer", "04"),
    ("B", "cc-train-episode-138343", "question_generator", "multi_select"),
    ("B", "cc-train-episode-138343", "question_generator", "ordering"),
}
MIN_INTERVAL_SECONDS = 35
MAX_HTTP = 9


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def scope_from_parent(parent):
    parent = Path(parent).resolve()
    manifest = read(parent / "MANIFEST.json")
    require(manifest["version"] == qa.VERSION and manifest["solver_model"] == qa.TARGET,
            "parent_version_or_solver_mismatch")
    failed, bound, observed = [], {}, set()
    def bind(path):
        bound[str(Path(path).resolve())] = sha(path)
    bind(parent / "MANIFEST.json")
    for branch in qa.ROUTES:
        generation = parent / branch / "generation"
        for name in ("MANIFEST.json", "MECHANICAL_POOL.json", "SOLVER_POOL.json", "REPORT.json"):
            bind(generation / name)
        mechanical = {q["candidate_id"]: q for q in read(generation / "MECHANICAL_POOL.json")["questions"]}
        successful = {q["candidate_id"] for q in read(generation / "SOLVER_POOL.json")["questions"]}
        for error_path in sorted((generation / "requests").glob("*/ERROR.json")):
            error = read(error_path)
            meta = error.get("metadata", {})
            if meta.get("http_status") != 429:
                continue
            raw_error = json.loads(meta.get("raw_response_text", "{}"))
            require(raw_error.get("code") == 50602 and "TPM limit reached" in raw_error.get("message", ""),
                    "429_is_not_frozen_tpm_failure")
            folder = error_path.parent
            require(not (folder / "RESPONSE.json").exists(), "successful_response_must_not_be_retried")
            proof, payload = read(folder / "PROOF.json"), read(folder / "REQUEST.json")
            qids = proof.get("candidate_ids", [proof.get("candidate_id")])
            discriminator = proof.get("question_type") if proof["role"] == "question_generator" else qids[0].rsplit("-", 1)[1]
            key = (branch, proof["root_id"], proof["role"], discriminator)
            require(key in EXPECTED and key not in observed, "unexpected_or_duplicate_retry_scope")
            observed.add(key)
            require(payload["model"] == qa.MODEL and qa.base.digest(payload) == proof["payload_sha256"],
                    "parent_payload_binding_changed")
            require(not successful.intersection(qids), "already_ready_or_evaluated_candidate_in_retry_scope")
            require(len(qids) == (4 if proof["role"] == "question_generator" else 1), "retry_slot_count_changed")
            candidate = mechanical[qids[0]] if proof["role"] == "ordering_reviewer" else None
            if candidate:
                require(candidate["question_type"] == "ordering", "review_retry_type_invalid")
            for path in folder.iterdir():
                if path.is_file():
                    bind(path)
            failed.append({"branch": branch, "root_id": proof["root_id"], "role": proof["role"],
                           "question_type": proof.get("question_type", "ordering"), "candidate_ids": qids,
                           "parent_request_dir": str(folder), "parent_request_sha256": sha(folder / "REQUEST.json"),
                           "payload_sha256": proof["payload_sha256"], "payload": payload,
                           "parent_proof": proof, "parent_mechanical_candidate": candidate})
    require(observed == EXPECTED and len(failed) == 5, "not_exactly_the_five_authorized_tpm_failures")
    return {"version": VERSION, "parent_run": str(parent), "roots": manifest["roots"],
            "expansion_run": manifest["expansion_run"], "requests": failed,
            "parent_files": [{"path": path, "sha256": digest} for path, digest in sorted(bound.items())],
            "retry_http_max": 5, "new_ordering_review_http_max": 4, "total_http_max": MAX_HTTP,
            "minimum_seconds_after_previous_call_completion": MIN_INTERVAL_SECONDS,
            "candidate_slots": 11, "new_candidate_ids_allowed": False,
            "generation_repairs_allowed": False, "training_started": False}


def scope_counts(scope, branch):
    counts = Counter()
    for entry in scope["requests"]:
        if entry["branch"] == branch:
            counts[(entry["root_id"], entry["question_type"])] += len(entry["candidate_ids"])
    return counts


def prepare(parent, run):
    parent, run = Path(parent).resolve(), Path(run).resolve()
    require(not run.exists() and parent != run, "derived_run_already_exists_or_is_parent")
    scope = scope_from_parent(parent)
    qa.prepare(Path(scope["expansion_run"]), run, slots=4)
    write(run / "RETRY_SCOPE.json", scope)
    # Bind the added recovery implementation and immutable parent request data.
    additions = [*scope["parent_files"],
                 {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
                 {"path": str(run / "RETRY_SCOPE.json"), "sha256": sha(run / "RETRY_SCOPE.json")}]
    for branch in qa.ROUTES:
        path = run / branch / "generation/MANIFEST.json"
        manifest = read(path)
        parent_manifest = read(parent / branch / "generation/MANIFEST.json")
        require([{k: v for k, v in row.items() if k != "source_path"} for row in manifest["source_rows"]]
                == [{k: v for k, v in row.items() if k != "source_path"} for row in parent_manifest["source_rows"]],
                "retry_source_inventory_changed")
        counts = scope_counts(scope, branch)
        request_count = sum(e["branch"] == branch for e in scope["requests"])
        extra_review = sum(len(e["candidate_ids"]) for e in scope["requests"]
                           if e["branch"] == branch and e["role"] == "question_generator" and e["question_type"] == "ordering")
        manifest.update(transport_recovery=VERSION, parent_run=str(parent), candidate_slots_per_type_per_root=0,
                        recovery_candidate_slots=[{"root_id": root, "question_type": kind, "candidate_slots": count}
                                                  for (root, kind), count in counts.items()])
        manifest["budget"] = {"candidate_slots": sum(counts.values()), "initial_generation_requests":
            sum(e["branch"] == branch and e["role"] == "question_generator" for e in scope["requests"]),
            "mechanical_repair_requests_max": 0, "ordering_review_requests_max": request_count + extra_review,
            "http_attempts_max": request_count + extra_review,
            "solver_full_calls_max": sum(count * (18 if kind == "multi_select" else 12) for (_, kind), count in counts.items())}
        manifest["files"] += additions
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    path = run / "MANIFEST.json"
    experiment = read(path)
    experiment.update(transport_recovery=VERSION, parent_run=str(parent), candidate_slots_total=11,
                      max_http_attempts=MAX_HTTP, max_solver_full_calls=156,
                      slots_per_root_route_type=0, recovery_scope_path=str(run / "RETRY_SCOPE.json"),
                      budget_per_route={branch: read(run / branch / "generation/MANIFEST.json")["budget"] for branch in qa.ROUTES})
    experiment["files"] += additions
    path.write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n")
    verify(run)
    return {"status": "PREPARED_NO_HTTP", "run": str(run), "retry_requests": 5,
            "recovery_candidate_slots": 11, "max_http": MAX_HTTP, "max_full_solver_calls": 156}


def verify(run):
    scope = read(run / "RETRY_SCOPE.json")
    require(scope["version"] == VERSION and len(scope["requests"]) == 5, "retry_scope_version_invalid")
    qa.base.verify_bindings(read(run / "MANIFEST.json")["files"])
    for entry in scope["requests"]:
        source = Path(entry["parent_request_dir"]) / "REQUEST.json"
        require(sha(source) == entry["parent_request_sha256"] and read(source) == entry["payload"], "retry_payload_changed")
        generation = run / entry["branch"] / "generation"
        suffix = qa.legacy.feedback.build_feedback_suffix(generation / "feedback", entry["root_id"], "writer", qa.TARGET)
        require(entry["payload"]["messages"][0]["content"].endswith(suffix), "retry_feedback_suffix_differs_from_parent")
    return scope


class PacedCalls:
    def __init__(self, run, *, monotonic=time.monotonic, sleep=time.sleep):
        self.run, self.monotonic, self.sleep = run, monotonic, sleep
        self.last_end = None
        self.rows = []

    def call(self, caller, label, payload, proof):
        require(len(self.rows) < MAX_HTTP, "global_retry_http_budget_exhausted")
        while self.last_end is not None and self.monotonic() - self.last_end < MIN_INTERVAL_SECONDS:
            self.sleep(min(35, MIN_INTERVAL_SECONDS - (self.monotonic() - self.last_end)))
        start = self.monotonic()
        began = now()
        raw, folder, metadata = caller.call(label, payload, proof)
        ended = self.monotonic()
        timing = {"call_sequence": len(self.rows) + 1, "started_at_utc": began, "finished_at_utc": now(),
                  "interval_since_previous_call_completion_seconds": None if self.last_end is None else start - self.last_end,
                  "elapsed_seconds": ended - start, "http_attempted": (folder / "HTTP_ATTEMPT.json").exists(),
                  "payload_sha256": qa.base.digest(payload), "request_directory": str(folder)}
        write(folder / "TIMING.json", timing)
        self.rows.append(timing)
        self.last_end = ended
        return raw, folder, metadata


def retry_proof(entry):
    proof = deepcopy(entry["parent_proof"])
    for key in ("payload_sha256", "capacity"):
        proof.pop(key, None)
    proof.update(transport_recovery=VERSION, parent_request_path=str(Path(entry["parent_request_dir"]) / "REQUEST.json"),
                 parent_request_sha256=entry["parent_request_sha256"], retry_number=1)
    return proof


def reject_parent_success(candidates, parent_questions):
    accepted, rejected = [], []
    for candidate in candidates:
        matches = [old for old in parent_questions if old["root_id"] == candidate["root_id"]
                   and old["question_type"] == candidate["question_type"] and qa.signature(old) == qa.signature(candidate)]
        if matches:
            rejected.append({"candidate_id": candidate["candidate_id"], "errors": ["duplicate_parent_success_statement_set"],
                             "matches": [{"candidate_id": q["candidate_id"]} for q in matches], "draft": candidate})
        else:
            accepted.append(candidate)
    return accepted, rejected


def live(run):
    run = Path(run).resolve()
    scope = verify(run)
    require(not (run / "RETRY_STARTED.json").exists(), "retry_already_started_no_automatic_replay")
    write(run / "RETRY_STARTED.json", {"started_at_utc": now(), "scope_sha256": sha(run / "RETRY_SCOPE.json")})
    paced = PacedCalls(run)
    parent = Path(scope["parent_run"])
    parent_ready = [q for branch in qa.ROUTES for q in read(parent / branch / "generation/SOLVER_POOL.json")["questions"]]
    try:
        for branch, route in qa.ROUTES.items():
            generation = run / branch / "generation"
            manifest = read(generation / "MANIFEST.json")
            qa.base.verify_bindings(manifest["files"])
            qa.legacy.MAX_HTTP = manifest["budget"]["http_attempts_max"]
            write(generation / "STARTED.json", {"started_at": now(), "http_max": qa.legacy.MAX_HTTP, "transport_recovery": VERSION})
            caller = qa.legacy.Caller(generation)
            counts = scope_counts(scope, branch)
            outcomes = {(root, kind): {"root_id": root, "route": route, "branch": branch, "question_type": kind,
                        "candidate_slots": counts[(root, kind)], "mechanical_pass": 0, "mechanical_rejected": [],
                        "abstentions": [], "review_rejected": [], "ready_for_solver": 0,
                        "qa_status": "RECOVERY_SCOPE" if counts[(root, kind)] else "OUTSIDE_429_RECOVERY_SCOPE",
                        "actual_generation_requests": 0, "actual_retry_requests": 0}
                        for root in scope["roots"] for kind in qa.TYPES}
            mechanical, ready = [], []
            for entry in scope["requests"]:
                if entry["branch"] != branch:
                    continue
                root, kind = entry["root_id"], entry["question_type"]
                outcome = outcomes[(root, kind)]
                source_path = generation / "sources" / (root + ".json")
                source = read(source_path)
                raw, folder, metadata = paced.call(caller, "429-once-" + Path(entry["parent_request_dir"]).name,
                                                     deepcopy(entry["payload"]), retry_proof(entry))
                outcome["actual_retry_requests"] += 1
                if entry["role"] == "ordering_reviewer":
                    candidate = deepcopy(entry["parent_mechanical_candidate"])
                    mechanical.append(candidate)
                    outcome["mechanical_pass"] += 1
                    if raw is None:
                        outcome["review_rejected"].append({"candidate_id": candidate["candidate_id"],
                            "reasons": [read(folder / "ERROR.json")["error_type"]]})
                    else:
                        checked = qa.legacy.ordering_gate.review_response(raw, [candidate], source,
                            folder / "RESPONSE.json", folder / "REQUEST.json", metadata)
                        write(folder / "REVIEW.json", checked)
                        accepted, duplicates = reject_parent_success(checked["accepted"], parent_ready)
                        ready += accepted
                        outcome["ready_for_solver"] += len(accepted)
                        outcome["review_rejected"] += checked["rejected"] + [
                            {"candidate_id": d["candidate_id"], "reasons": d["errors"]} for d in duplicates]
                    continue
                outcome["actual_generation_requests"] += 1
                if raw is None:
                    outcome["mechanical_rejected"] += [{"candidate_id": qid, "errors": [read(folder / "ERROR.json")["error_type"]]}
                                                       for qid in entry["candidate_ids"]]
                    continue
                valid, rejected, abstained, envelope_error = qa.legacy.validate_draft(
                    raw, entry["candidate_ids"], root, kind, source, source_path)
                branch_valid = []
                for candidate in valid:
                    evidence = candidate["option_evidence" if kind == "multi_select" else "item_evidence"]
                    has_new = any(isinstance(value, dict) and str(value.get("turn_ref", "")).startswith("R02:" + branch + ":")
                                  for value in evidence.values())
                    candidate["new_branch_reference_diagnostic"] = has_new
                    if kind == "ordering" and not has_new:
                        rejected.append({"candidate_id": candidate["candidate_id"], "errors": ["no_new_branch_evidence"], "draft": candidate})
                    else:
                        candidate.update(generator_model=qa.MODEL, generation_request_path=str(folder / "REQUEST.json"),
                            generation_response_path=str(folder / "RESPONSE.json"), raw_question_content_preserved=True,
                            expansion_route=route, expansion_branch=branch)
                        branch_valid.append(candidate)
                accepted, duplicate_rows = qa.dedup.deduplicate(branch_valid, read(generation / "prior" / (root + ".json")),
                                                              [q for q in mechanical if q["root_id"] == root])
                accepted, parent_duplicates = reject_parent_success(accepted, parent_ready)
                rejected += duplicate_rows + parent_duplicates
                write(folder / "VALIDATION.json", {"accepted_ids": [q["candidate_id"] for q in accepted],
                    "rejected": rejected, "abstentions": abstained, "envelope_error": envelope_error,
                    "semantic_review": False, "additional_generation_repair": False})
                mechanical += accepted
                outcome["mechanical_pass"] += len(accepted)
                outcome["mechanical_rejected"] += rejected
                outcome["abstentions"] += abstained
                if kind == "multi_select":
                    ready += accepted
                    outcome["ready_for_solver"] += len(accepted)
                    continue
                for candidate in accepted:
                    payload = {"model": qa.MODEL, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
                        "response_format": {"type": "json_object"}, "messages": [{"role": "user",
                        "content": qa.legacy.ordering_gate.build_review_prompt(candidate, source)}]}
                    payload, receipt = qa.legacy.feedback.append_feedback_to_request(payload, generation / "feedback", root, "writer", qa.TARGET)
                    reviewed_raw, review_folder, reviewed_metadata = paced.call(caller,
                        root.split("-")[-1] + "-" + branch + "-recovered-order-review-" + candidate["candidate_id"].split("-")[-1], payload,
                        {"root_id": root, "route": route, "branch": branch, "role": "ordering_reviewer",
                         "candidate_id": candidate["candidate_id"], "feedback_receipt": receipt,
                         "source_sha256": sha(source_path), "independent_request_without_prior_verdicts": True,
                         "transport_recovery": VERSION, "necessary_review_of_recovered_generation": True})
                    if reviewed_raw is None:
                        outcome["review_rejected"].append({"candidate_id": candidate["candidate_id"],
                            "reasons": [read(review_folder / "ERROR.json")["error_type"]]})
                    else:
                        checked = qa.legacy.ordering_gate.review_response(reviewed_raw, [candidate], source,
                            review_folder / "RESPONSE.json", review_folder / "REQUEST.json", reviewed_metadata)
                        write(review_folder / "REVIEW.json", checked)
                        ready += checked["accepted"]
                        outcome["ready_for_solver"] += len(checked["accepted"])
                        outcome["review_rejected"] += checked["rejected"]
            require(not {q["candidate_id"] for q in ready}.intersection(q["candidate_id"] for q in parent_ready), "parent_success_replayed")
            write(generation / "MECHANICAL_POOL.json", {"questions": mechanical})
            write(generation / "SOLVER_POOL.json", {"version": qa.VERSION, "candidate_count": len(ready), "questions": ready,
                "provenance": {"requested_model": qa.MODEL, "returned_model": qa.MODEL, "codex_question_edits": False,
                    "multi_select_semantic_review": False, "ordering_review_owner": qa.MODEL,
                    "source_round": "P+R01+R02_" + branch, "expansion_route": route, "feedback_target_solver": qa.TARGET},
                "ground_calls": 0, "training_started": False})
            for (root, kind), outcome in outcomes.items():
                write(generation / "outcomes" / (root + "." + kind + ".json"), outcome)
            report = {"version": qa.VERSION, "transport_recovery": VERSION, "branch": branch, "route": route,
                "http_attempts": len(list((generation / "requests").glob("*/HTTP_ATTEMPT.json"))),
                "requests_prepared": caller.counter, "candidate_slots": sum(counts.values()),
                "mechanical_pass": len(mechanical), "ready_for_solver": len(ready), "roots": list(outcomes.values()),
                "feedback_in_every_request": True, "semantic_dedup_certified": False,
                "expansion_shortfall_no_qa_roots": [r["root_id"] for r in manifest["source_rows"] if not r["accepted_scenes"]],
                "no_current_solver_feedback_in_generation": True, "ground_calls": 0, "training_calls": 0}
            qa.base.verify_bindings(manifest["files"])
            write(generation / "REPORT.json", report)
        verify(run)
        write(run / "RETRY_COMPLETE.json", {"status": "COMPLETE", "request_count": len(paced.rows),
            "actual_http_attempts": sum(row["http_attempted"] for row in paced.rows), "maximum_http": MAX_HTTP,
            "retry_payloads": 5, "additional_generation_repairs": 0, "ground_calls": 0, "training_calls": 0,
            "finished_at_utc": now()})
    finally:
        write(run / "TRANSPORT_TIMING.json", {"minimum_interval_seconds": MIN_INTERVAL_SECONDS, "calls": paced.rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "live"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.parent is not None, "parent_required")
        print(json.dumps(prepare(args.parent, args.run), ensure_ascii=False))
    elif args.action == "verify":
        scope = verify(args.run.resolve())
        print(json.dumps({"status": "VERIFIED", "retry_requests": len(scope["requests"])}))
    else:
        live(args.run)


if __name__ == "__main__":
    main()
