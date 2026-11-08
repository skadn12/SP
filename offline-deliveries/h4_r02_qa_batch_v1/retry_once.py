"""Replay frozen successful requests; retry eligible parent transport failures once.

Preparation and tests are offline. Live uses the unchanged generator pipeline
with a restricted caller. New downstream calls are permitted only for candidate
slots whose generator transport request was recovered in this retry run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("r02_qa_retry_generator", HERE / "generate.py")
gen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)
base = gen.base
ORIGINAL_CALLER = gen.Caller
VERSION = "h4_r02_qa_transport_retry_once_v1"
MAX_NEW_HTTP = 12
MAX_COMBINED_HTTP = 24


def stage(directory: Path) -> str:
    return directory.name.split("-", 1)[1]


def parent_inventory(parent: Path) -> tuple[list[dict], int]:
    """Read and bind completed responses, including mechanically rejected drafts."""
    report = base.read(parent / "REPORT.json")
    base.verify_bindings(base.read(parent / "MANIFEST.json")["files"])
    rows, hashes, stages = [], set(), set()
    for directory in sorted((parent / "requests").iterdir()):
        payload = base.read(directory / "REQUEST.json")
        proof = base.read(directory / "PROOF.json")
        fingerprint = base.digest(payload)
        base.require(proof["payload_sha256"] == fingerprint, "parent_payload_proof_mismatch")
        base.require(fingerprint not in hashes and stage(directory) not in stages, "parent_duplicate_request")
        hashes.add(fingerprint); stages.add(stage(directory))
        attempted = (directory / "HTTP_ATTEMPT.json").exists()
        if attempted:
            attempt = base.read(directory / "HTTP_ATTEMPT.json")
            base.require(attempt["actual_payload_equals_frozen"] is True
                         and attempt["payload_sha256"] == fingerprint, "parent_actual_payload_unverified")
        error = base.read(directory / "ERROR.json") if (directory / "ERROR.json").exists() else None
        response = base.read(directory / "RESPONSE.json") if (directory / "RESPONSE.json").exists() else None
        if response is not None and error is None:
            metadata = response.get("metadata") or {}
            base.require(attempted and metadata.get("returned_model") == gen.MODEL
                         and metadata.get("finish_reason") == "stop", "parent_response_not_complete")
            disposition = "REUSE_SUCCESSFUL_HTTP"
        else:
            base.require(error is not None, "parent_request_incomplete")
            metadata = error.get("metadata") or {}
            http_status = metadata.get("http_status")
            unknown_completion = (error.get("error_type") == "ComparisonTransportError"
                                  and http_status is None and metadata.get("http_attempt_count") == 1
                                  and metadata.get("completion_status") == "UNKNOWN")
            disposition = ("RETRY_TRANSPORT_ONCE" if attempted and response is None and (http_status in (503, 504) or unknown_completion)
                           else "REUSE_NONRETRYABLE_FAILURE")
        rows.append({"stage": stage(directory), "directory": str(directory.resolve()),
                     "payload_sha256": fingerprint, "disposition": disposition,
                     "root_id": proof["root_id"], "role": proof["role"],
                     "question_type": proof.get("question_type"),
                     "candidate_ids": proof.get("candidate_ids", [proof.get("candidate_id")]),
                     "http_status": (error.get("metadata") or {}).get("http_status") if error else None,
                     "prior_completion_unknown": bool(error and (error.get("metadata") or {}).get("completion_status") == "UNKNOWN"),
                     "http_attempted": attempted})
    actual = sum(row["http_attempted"] for row in rows)
    base.require(actual == report["http_attempts"], "parent_http_count_mismatch")
    return rows, actual


def prepare(parent: Path, run: Path) -> dict:
    base.require(not run.exists(), "retry_run_already_exists")
    inventory, old_http = parent_inventory(parent)
    eligible = [row for row in inventory if row["disposition"] == "RETRY_TRANSPORT_ONCE"]
    base.require(eligible, "no_parent_retryable_transport_requests")
    budget = min(MAX_NEW_HTTP, MAX_COMBINED_HTTP - old_http)
    base.require(budget >= len(eligible), "retry_http_budget_insufficient")
    manifest = deepcopy(base.read(parent / "MANIFEST.json"))
    run.mkdir(parents=True)
    frozen = [Path(__file__), HERE / "generate.py", HERE / "ordering_gate.py"]
    frozen.extend(path for path in parent.rglob("*") if path.is_file())
    for name in ("sources", "prior", "feedback"):
        shutil.copytree(parent / name, run / name)
        frozen.extend(path for path in (run / name).rglob("*") if path.is_file())
    shutil.copyfile(parent / "provider_profile.json", run / "provider_profile.json")
    frozen.append(run / "provider_profile.json")
    policy = {"version": VERSION, "parent_run": str(parent.resolve()), "parent_http_attempts": old_http,
              "max_new_http_attempts": budget, "max_combined_http_attempts": MAX_COMBINED_HTTP,
              "retryable_http_statuses": [503, 504], "attempts_per_failed_payload": 1,
              "retry_transmitted_unknown_comparison_transport_error": True,
              "unknown_prior_service_completion_or_cost_not_asserted": True,
              "successful_http_replayed_even_if_mechanical_reject": True,
              "new_downstream_only_after_generator_recovery": True,
              "no_new_candidate_slots": True, "requests": inventory}
    base.write(run / "RETRY_POLICY.json", policy)
    frozen.append(run / "RETRY_POLICY.json")
    manifest["parent_run"] = str(parent.resolve())
    manifest["retry_version"] = VERSION
    manifest["max_new_http_attempts"] = budget
    manifest["files"] += base.bindings(frozen)
    base.write(run / "MANIFEST.json", manifest)
    return {"status": "PREPARED_NO_HTTP", "parent_http_attempts": old_http,
            "retryable_requests": [row["stage"] for row in eligible], "max_new_http_attempts": budget,
            "run": str(run.resolve())}


class ReplayCaller(ORIGINAL_CALLER):
    def __init__(self, run):
        super().__init__(run)
        self.policy = base.read(run / "RETRY_POLICY.json")
        self.cache = {row["payload_sha256"]: row for row in self.policy["requests"]}
        self.old_stages = {row["stage"]: row for row in self.policy["requests"]}
        self.retried: set[str] = set()
        self.recovered: dict[tuple[str, str], set[str]] = {}
        self.new_http = 0

    def _reuse(self, label, payload, proof, cached):
        base.require(self.counter < gen.MAX_HTTP, "logical_request_budget_exhausted")
        original = Path(cached["directory"])
        prior_payload = base.read(original / "REQUEST.json")
        base.require(prior_payload == payload, "reuse_payload_mismatch")
        self.counter += 1
        directory = self.run / "requests" / f"{self.counter:03d}-{label}"
        old_proof = base.read(original / "PROOF.json")
        for key in ("root_id", "role", "question_type", "candidate_id", "candidate_ids", "source_sha256", "feedback_receipt"):
            base.require(old_proof.get(key) == proof.get(key), "reuse_identity_mismatch:" + key)
        base.write(directory / "REQUEST.json", payload)
        base.write(directory / "PROOF.json", {**proof, "payload_sha256": base.digest(payload),
                                               "capacity": old_proof["capacity"]})
        base.write(directory / "PROFILE.json", base.read(original / "PROFILE.json"))
        base.write(directory / "REUSE.json", {"parent_request": str(original), "disposition": cached["disposition"],
            "payload_sha256": base.digest(payload), "parent_response_sha256":
                base.digest((original / "RESPONSE.json").read_bytes()) if (original / "RESPONSE.json").exists() else None,
            "new_http_attempts": 0})
        if cached["disposition"] == "REUSE_SUCCESSFUL_HTTP":
            response = base.read(original / "RESPONSE.json")
            base.write(directory / "RESPONSE.json", response)
            print(json.dumps({"event": "reused_response", "request": directory.name, "parent": original.name}), flush=True)
            return response["text"], directory, response["metadata"]
        base.write(directory / "ERROR.json", base.read(original / "ERROR.json"))
        return None, directory, {}

    def _new_allowed(self, label, proof):
        base.require(label not in self.old_stages, "existing_stage_payload_changed_no_regeneration")
        qtype = proof.get("question_type") if proof["role"] == "question_generator" else "ordering"
        ids = set(proof.get("candidate_ids", [proof.get("candidate_id")]))
        allowed = self.recovered.get((proof["root_id"], qtype), set())
        base.require(ids and None not in ids and ids <= allowed, "new_request_outside_recovered_slots")
        base.require(proof["role"] == "ordering_reviewer" or
                     (proof["role"] == "question_generator" and label.endswith("generate-2")),
                     "new_request_is_not_required_downstream_stage")

    def _frozen_prior_request(self, label, payload, proof):
        """Keep a prior ordering branch frozen when recovered multi facts grow.

        Only the deduplication context may differ. Source, slot IDs, task,
        repair errors and feedback must remain exactly identical. The old
        request and its actual receipt are recorded as the executed/reused
        request; the newly proposed context is recorded separately.
        """
        prior = self.old_stages[label]
        directory = Path(prior["directory"])
        old_payload, old_proof = base.read(directory / "REQUEST.json"), base.read(directory / "PROOF.json")
        base.require(proof.get("role") == old_proof.get("role") == "question_generator"
                     and proof.get("question_type") == old_proof.get("question_type") == "ordering",
                     "existing_stage_payload_changed_no_regeneration")
        for key in ("root_id", "candidate_ids", "source_sha256"):
            base.require(proof.get(key) == old_proof.get(key), "frozen_prior_identity_changed:" + key)
        def unpack(value):
            rest = deepcopy(value)
            base.require(isinstance(rest.get("messages"), list), "frozen_prior_message_schema")
            messages = rest.pop("messages")
            base.require(len(messages) == 1 and messages[0]["role"] == "user", "frozen_prior_message_schema")
            base.require("\nINPUT_JSON\n" in messages[0]["content"], "frozen_prior_message_schema")
            prefix, tail = messages[0]["content"].split("\nINPUT_JSON\n", 1)
            context, end = json.JSONDecoder().raw_decode(tail)
            previous = context.pop("previous_candidates_dedup_only")
            return rest, prefix, context, previous, tail[end:]
        old = unpack(old_payload); new = unpack(payload)
        base.require(old[:3] == new[:3] and old[4] == new[4], "frozen_prior_change_outside_dedup_context")
        recovered = self.recovered.get((proof["root_id"], "multi_select"), set())
        added = [row for row in new[3] if row.get("candidate_id") in recovered]
        unchanged = [row for row in new[3] if row.get("candidate_id") not in recovered]
        base.require(added and unchanged == old[3], "frozen_prior_change_not_recovered_multiselect")
        note = {"action": "FROZEN_PRIOR_CONTEXT_REPLAY", "parent_request": str(directory),
                "proposed_payload_sha256": base.digest(payload), "actual_frozen_payload_sha256": base.digest(old_payload),
                "recovered_multiselect_ids_not_visible_to_old_ordering": [row["candidate_id"] for row in added],
                "frozen_successful_or_mechanically_rejected_ordering_output_preserved": True,
                "cross_type_semantic_novelty_certified": False, "new_http_for_context_change": 0}
        return old_payload, old_proof, note

    def call(self, label, payload, proof):
        fingerprint = base.digest(payload)
        cached = self.cache.get(fingerprint)
        note = None
        if cached is None and label in self.old_stages:
            payload, proof, note = self._frozen_prior_request(label, payload, proof)
            fingerprint = base.digest(payload); cached = self.cache[fingerprint]
        if cached and cached["disposition"] != "RETRY_TRANSPORT_ONCE":
            result = self._reuse(label, payload, proof, cached)
            if note:
                base.write(result[1] / "FROZEN_PRIOR_CONTEXT_REPLAY.json", note)
            return result
        if cached:
            base.require(fingerprint not in self.retried, "failed_payload_already_retried")
            base.require(label == cached["stage"], "retry_stage_identity_changed")
            self.retried.add(fingerprint)
            action = "RETRY_PARENT_TRANSPORT_ONCE"
        else:
            self._new_allowed(label, proof)
            action = "NEW_DOWNSTREAM_OF_RECOVERED_GENERATION"
        base.require(self.new_http < self.policy["max_new_http_attempts"], "new_http_budget_exhausted")
        raw, directory, metadata = super().call(label, payload, proof)
        attempted = (directory / "HTTP_ATTEMPT.json").exists()
        self.new_http += int(attempted)
        base.write(directory / "RETRY_ACTION.json", {"action": action, "payload_sha256": fingerprint,
            "parent_request": cached["directory"] if cached else None, "new_http_attempted": attempted,
            "new_http_attempts_so_far": self.new_http})
        if note:
            base.write(directory / "FROZEN_PRIOR_CONTEXT_REPLAY.json", note)
        if raw is not None and proof["role"] == "question_generator":
            self.recovered.setdefault((proof["root_id"], proof["question_type"]), set()).update(proof["candidate_ids"])
        return raw, directory, metadata


CONTENT_FIELDS = ("candidate_id", "root_id", "question_type", "topic", "chronology_kind", "question",
                  "options", "items", "gold", "option_evidence", "item_evidence", "order_anchor", "support_refs")


def verify_preserved(parent: Path, run: Path) -> dict:
    counts = {}
    for name in ("MECHANICAL_POOL.json", "SOLVER_POOL.json"):
        before = base.read(parent / name)["questions"]
        after = {row["candidate_id"]: row for row in base.read(run / name)["questions"]}
        for row in before:
            base.require(row["candidate_id"] in after, "parent_success_missing:" + row["candidate_id"])
            other = after[row["candidate_id"]]
            base.require(all(row.get(key) == other.get(key) for key in CONTENT_FIELDS),
                         "parent_success_content_changed:" + row["candidate_id"])
        counts[name] = len(before)
    return {"status": "PASS", "preserved_parent_candidates": counts}


def live(run: Path) -> dict:
    policy = base.read(run / "RETRY_POLICY.json")
    parent = Path(policy["parent_run"])
    gen.Caller = ReplayCaller
    try:
        report = gen.live(run)
    finally:
        gen.Caller = ORIGINAL_CALLER
    preserved = verify_preserved(parent, run)
    attempts = list((run / "requests").glob("*/HTTP_ATTEMPT.json"))
    base.require(len(attempts) == report["http_attempts"] <= policy["max_new_http_attempts"], "new_http_count_invalid")
    eligible = {row["payload_sha256"] for row in policy["requests"] if row["disposition"] == "RETRY_TRANSPORT_ONCE"}
    retried = [base.read(path) for path in (run / "requests").glob("*/RETRY_ACTION.json")
               if base.read(path)["action"] == "RETRY_PARENT_TRANSPORT_ONCE"]
    base.require({row["payload_sha256"] for row in retried} == eligible and len(retried) == len(eligible),
                 "retry_payload_set_not_exact")
    result = {"version": VERSION, "status": "COMPLETE", "parent_run": str(parent),
              "parent_http_attempts": policy["parent_http_attempts"], "new_http_attempts": len(attempts),
              "combined_http_attempts": policy["parent_http_attempts"] + len(attempts),
              "successful_http_reuses": len(list((run / "requests").glob("*/REUSE.json"))),
              "retried_parent_failures": len(retried), "preservation": preserved,
              "frozen_prior_context_replays": len(list((run / "requests").glob("*/FROZEN_PRIOR_CONTEXT_REPLAY.json"))),
              "cross_type_semantic_novelty_certified": False,
              "ready_for_solver": report["ready_for_solver"], "training_calls": 0}
    base.write(run / "RETRY_REPORT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "live"))
    parser.add_argument("--parent", type=Path, default=HERE / "runs/pilot_02")
    parser.add_argument("--run", type=Path, default=HERE / "runs/pilot_02_transport_retry_01")
    args = parser.parse_args()
    result = prepare(args.parent.resolve(), args.run.resolve()) if args.action == "prepare" else live(args.run.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
