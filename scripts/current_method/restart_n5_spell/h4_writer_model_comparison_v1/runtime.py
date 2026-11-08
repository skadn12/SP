"""Paired model scheduling around the shared TEMPORAL Writer, never a second generator."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..h4_r02_feedback_ablation_p0_p1_v1.qa_run import exclusive_run
from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from .transport import ProfileClient, ComparisonTransportError, build_payload, resolve_credential

POLICY = {"version": "ordering_temporal_contract_v1", "node_count": 4}
ARMS = ("QWEN14B_TEMPORAL", "STRONG_TEMPORAL")
KNOWN_REJECTIONS = {400, 401, 402, 403, 404, 405, 406, 413, 415, 422, 429}


class ExecutionStopped(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []


def jsonl(path, values):
    atomic_text(Path(path), "".join(canonical_json(x) + "\n" for x in values))


def directory(run, row):
    return Path(run) / "arms" / row["arm_id"] / "qa/shards/shard-00/roots" / row["root_id"]


def stop(run, reason):
    # Reasons are fixed codes, never exception repr or authentication material.
    atomic_json(Path(run) / "audit/execution_stop.json", {"reason": reason, "automatic_resend": False})
    raise ExecutionStopped(reason)


def ensure_safe(run):
    path = Path(run) / "audit/execution_stop.json"
    if path.exists():
        raise ExecutionStopped("EXISTING_STOP:" + read(path)["reason"])


def audit_inventory(run):
    """Deleting/moving a reservation cannot restore the call budget on resume."""
    run = Path(run)
    events = rows(run / "audit/logical_calls.jsonl")
    paths = list((run / "arms").glob("*/qa/shards/shard-00/roots/*/ledger/requests/*.json"))
    identities = [x.get("identity") for x in events]
    if len(paths) != len(events) or len(identities) != len(set(identities)) or len(paths) > 24:
        stop(run, "RESERVATION_INVENTORY_MISMATCH")
    for event in events:
        expected = directory(run, event) / "ledger/requests" / (event["request_id"] + ".json")
        if not expected.is_file():
            stop(run, "MISSING_OR_MOVED_RESERVATION")
        state = read(expected)
        metadata = state.get("request_metadata", {})
        if (metadata.get("global_request_id") != event["identity"] or
                any(metadata.get(k) != event.get(k) for k in ("arm_id", "root_id", "slot_id", "stage"))):
            stop(run, "RESERVATION_IDENTITY_MISMATCH")
    primaries = [x["order"] for x in events if x["stage"] == "primary"]
    if primaries != list(range(1, len(primaries) + 1)):
        stop(run, "PRIMARY_EXECUTION_ORDER_MISMATCH")
    for index, event in enumerate(events):
        if event["stage"] == "repair" and (index == 0 or events[index - 1]["stage"] != "primary" or
                events[index - 1]["order"] != event["order"]):
            stop(run, "REPAIR_EXECUTION_ORDER_MISMATCH")
    http = rows(run / "audit/http_events.jsonl")
    def http_identity(row):
        return tuple(row.get(k) for k in ("arm_id", "root_id", "slot_id", "stage"))
    actual = [http_identity(x) for x in http]
    expected = []
    for path in (run / "arms").glob("*/qa/shards/shard-00/roots/*/ledger/transport_outcomes/*.json"):
        outcome = read(path)
        if outcome.get("transport_attempt_count") == 1:
            expected.append(http_identity(outcome))
    if len(actual) > 24 or len(actual) != len(set(actual)) or set(actual) != set(expected):
        stop(run, "HTTP_ATTEMPT_INVENTORY_MISMATCH")


class ModelLedger:
    """Scope the original ledger to one arm/root/intent with one replacing repair."""
    def __init__(self, run, row, profile, *, mode, verifier, capacity):
        self.run, self.row, self.profile = Path(run), row, profile
        self.root = directory(run, row)
        self.delegate = RequestLedger(self.root, sha256_file(self.run / "audit/frozen_manifest.json"), mode)
        self.mode, self.verifier, self.capacity = mode, verifier, capacity
        self.parsed, self.current_stage, self.http_seen = {}, None, False
        self.ids = {stage: f"{row['root_id']}-contract_guided-{verb}-{row['slot_id']}"
                    for stage, verb in (("primary", "generator"), ("repair", "repair"))}

    def before_send(self, payload):
        self.verifier()
        ensure_safe(self.run)
        if self.current_stage is None or self.http_seen:
            stop(self.run, "UNBOUND_OR_REPEATED_HTTP_ATTEMPT")
        request_id = self.ids[self.current_stage]
        frozen = read(self.root / "raw/requests" / (request_id + ".json"))
        if payload != frozen["payload"]:
            stop(self.run, "SENT_PAYLOAD_DIFFERS_FROM_FROZEN_REQUEST")
        events = rows(self.run / "audit/http_events.jsonl")
        if len(events) >= 24:
            stop(self.run, "HTTP_BUDGET_EXHAUSTED")
        event = {**self.row, "stage": self.current_stage, "request_id": request_id,
                 "event": "http_attempt_started", "attempt": 1,
                 "payload_sha256": sha256_bytes(canonical_json(payload).encode())}
        events.append(event)
        jsonl(self.run / "audit/http_events.jsonl", events)
        self.http_seen = True

    def invoke(self, **kwargs):
        try:
            return self._invoke(**kwargs)
        except BaseException:
            if not (self.run / "audit/execution_stop.json").exists():
                atomic_json(self.run / "audit/execution_stop.json", {
                    "reason": "REQUEST_VALIDATION_OR_UNKNOWN_COMPLETION_STOP", "automatic_resend": False})
            raise

    def _invoke(self, **kwargs):
        ensure_safe(self.run)
        self.verifier()
        stage = next((s for s, rid in self.ids.items() if rid == kwargs.get("request_id")), None)
        if stage is None:
            stop(self.run, "REQUEST_OUTSIDE_PLANNED_SLOT")
        rid = self.ids[stage]
        if stage == "repair" and "primary" not in self.parsed:
            stop(self.run, "REPAIR_BEFORE_PRIMARY_RESULT")
        prompt = (self.root / "raw/prompts" / (rid + ".txt")).read_text()
        payload = build_payload(self.profile, prompt)
        if stage == "primary":
            location = self.run / "policy_snapshot/primary_requests" / self.row["arm_id"] / self.row["root_id"] / self.row["slot_id"]
            expected = read(location / "request.json")["payload"]
            if payload != expected or prompt.encode() != (location / "prompt.txt").read_bytes():
                stop(self.run, "PRIMARY_REQUEST_DRIFT")
        else:
            from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_repair_prompt
            shared = self.run / "inputs/shared" / self.row["root_id"]
            expected = build_repair_prompt(read(shared / "expanded_dialogue_clean.json"),
                read(shared / "contracts.json")[self.row["raw_index"]], self.parsed["primary"]["raw_parsed"],
                "contract_guided", intent_guidance=True, ordering_policy=POLICY)["prompt"]
            if prompt != expected:
                stop(self.run, "REPAIR_DOES_NOT_USE_OWN_PRIMARY_AND_ORIGINAL_TEMPLATE")
        metadata = kwargs["request_metadata"]
        if (metadata.get("max_tokens") != self.profile["output_limit"] or
                metadata.get("temperature") != self.profile["request_parameters"].get("temperature") or
                kwargs["backend_identity"].get("model") != self.profile["model"] or
                kwargs["backend_identity"].get("base_url") != self.profile["base_url"]):
            stop(self.run, "SHARED_WRITER_PARAMETER_IDENTITY_DRIFT")
        identity = "::".join((self.row["arm_id"], self.row["root_id"], self.row["slot_id"], stage))
        self.capacity(self.row["arm_id"], payload["messages"], self.profile["output_limit"], identity)
        self.current_stage, self.http_seen = stage, False
        atomic_json(self.root / "raw/requests" / (rid + ".json"), {
            "payload": payload, "endpoint": self.profile["endpoint"], "stage": stage})
        current = self.delegate.read(rid)
        if current and current.get("status") != "completed":
            stop(self.run, "UNFINISHED_RESERVATION_NO_AUTOMATIC_RESEND")
        reservations = list((self.run / "arms").glob("*/qa/shards/shard-00/roots/*/ledger/requests/*.json"))
        events = rows(self.run / "audit/logical_calls.jsonl")
        if len(reservations) != len(events):
            stop(self.run, "RESERVATION_INVENTORY_MISMATCH")
        if not current:
            if len(reservations) >= 24:
                stop(self.run, "LOGICAL_CALL_BUDGET_EXHAUSTED")
            events.append({**self.row, "stage": stage, "request_id": rid, "identity": identity})
            jsonl(self.run / "audit/logical_calls.jsonl", events)
        call = kwargs["call"]

        def observed():
            try:
                response = call()
            except BaseException as error:
                detail = error.metadata if isinstance(error, ComparisonTransportError) else {}
                http_status = detail.get("http_status", getattr(error, "code", None))
                status = "KNOWN_HTTP_REJECTION" if http_status in KNOWN_REJECTIONS else "UNKNOWN_COMPLETION"
                atomic_json(self.root / "ledger/transport_outcomes" / (rid + ".json"), {
                    **detail, **self.row, "stage": stage, "request_id": rid,
                    "service_completion_status": status, "http_status": http_status,
                    "transport_attempt_count": int(self.http_seen), "error_type": type(error).__name__})
                atomic_json(self.run / "audit/execution_stop.json", {
                    "reason": "TECHNICAL_FAILURE_NO_RESEND_OR_REPAIR",
                    "service_completion_status": status, "http_status": http_status,
                    "automatic_resend": False, "content_repair_permitted": False,
                    **self.row, "stage": stage, "request_id": rid})
                raise
            atomic_json(self.root / "ledger/transport_outcomes" / (rid + ".json"), {
                **response.metadata, **self.row, "stage": stage, "request_id": rid,
                "service_completion_status": "COMPLETED", "transport_attempt_count": int(self.http_seen)})
            return response

        kwargs.update(call=observed, budget_limit=6, total_budget_limit=6,
                      request_metadata={**metadata, **self.row, "stage": stage,
                          "global_request_id": identity, "ordering_policy": POLICY,
                          "source": "contract_guided", "intent_guidance": True,
                          "payload_sha256": sha256_bytes(canonical_json(payload).encode())},
                      backend_identity={**kwargs["backend_identity"],
                          "endpoint": self.profile["endpoint"], "profile_sha256": sha256_bytes(canonical_json(self.profile).encode())})
        return self.delegate.invoke(**kwargs)

    def after_call(self, *, stage, request_id, raw=None, raw_text="", parse_error=None):
        self.parsed[stage] = {"raw_parsed": raw, "parse_error": parse_error}
        atomic_json(self.root / "stage_calls" / self.row["slot_id"] / (stage + ".json"), {
            **self.row, "stage": stage, "request_id": request_id, **self.parsed[stage],
            "raw_text_sha256": sha256_bytes(raw_text.encode())})
        ensure_safe(self.run)
        if not self.delegate.read(request_id):
            stop(self.run, "NO_RESERVATION_AFTER_SHARED_CALL")


def finalize_completed_pools(run, pools, slot_outcomes):
    """Finalize only returned batches, also when a later technical failure stops the run."""
    finals, outcomes = [], []
    for (arm, root_id), pool in pools.items():
        root = directory(run, {"arm_id": arm, "root_id": root_id})
        outcome = generation.finalize_writer_candidates(root_dir=root, root_id=root_id,
            candidates=pool["candidates"], generation_records=pool["records"],
            expanded=read(run / "inputs/shared" / root_id / "expanded_dialogue_clean.json"),
            ordering_policy=POLICY, ordering_context={"run_id": run.name, "arm_id": arm, "root_id": root_id})
        outcomes.append({**outcome, "arm_id": arm})
        for candidate in read(root / "candidates_valid.json"):
            finals.append({**candidate, **pool["identities"][candidate["candidate_id"]]})
    jsonl(run / "reports/runtime_final_candidates.jsonl", finals)
    atomic_json(run / "reports/slot_outcomes.json", slot_outcomes)
    return outcomes


def execute_schedule(run, *, client_factory: Callable, verifier: Callable,
                     capacity: Callable, backend_mode="mock"):
    """Same shared Writer stage for both arms. Factories are injected in offline tests."""
    run = Path(run)
    with exclusive_run(run):
        verifier()
        ensure_safe(run)
        audit_inventory(run)
        schedule, profiles = read(run / "schedule.json"), read(run / "profiles.json")
        from .constants import SCHEDULE
        if schedule != SCHEDULE or len(schedule) != 12:
            stop(run, "SCHEDULE_DRIFT")
        contract = read(run / "execution_contract.json")
        for key, expected in (("planned_primary_calls", 12), ("maximum_repair_calls", 12),
                              ("maximum_logical_calls", 24), ("maximum_http_attempts", 24)):
            if contract.get(key) != expected:
                stop(run, "EXECUTION_BUDGET_DRIFT")
        stages, pools, slot_outcomes = [], {}, []
        try:
            for row in schedule:
                root = directory(run, row)
                shared = run / "inputs/shared" / row["root_id"]
                expanded, intents = read(shared / "expanded_dialogue_clean.json"), read(shared / "contracts.json")
                if [x.get("intent_id") for x in intents] != ["intent_1", "intent_2", "intent_3"]:
                    stop(run, "INTENT_INVENTORY_DRIFT")
                checkpoint = root / "cells" / (row["slot_id"] + ".json")
                if checkpoint.exists():
                    value = read(checkpoint)
                    marker = checkpoint.with_suffix(".sha256")
                    if not marker.exists() or marker.read_text().strip() != sha256_file(checkpoint):
                        stop(run, "CELL_CHECKPOINT_DRIFT")
                    if value["row"] != row:
                        stop(run, "CELL_IDENTITY_DRIFT")
                    for path, sha in value["artifacts"].items():
                        if not Path(path).is_file() or sha256_file(Path(path)) != sha:
                            stop(run, "CELL_ARTIFACT_DRIFT")
                    result, parsed = value["result"], value["parsed"]
                else:
                    profile = profiles[row["arm_id"]]
                    ledger = ModelLedger(run, row, profile, mode=backend_mode, verifier=verifier, capacity=capacity)
                    client = client_factory(row, profile, ledger.before_send)
                    result = generation.execute_writer_batch(expanded=expanded,
                        contract=intents[row["raw_index"]], root_dir=root, root_id=row["root_id"],
                        ledger=ledger, client=client, source="contract_guided", intent_guidance=True,
                        slot_suffix="", hooks=None, after_call=ledger.after_call,
                        ordering_policy=POLICY,
                        ordering_context={"run_id": run.name, "arm_id": row["arm_id"], "root_id": row["root_id"]},
                        writer_max_tokens=profile["output_limit"],
                        writer_temperature=profile["request_parameters"].get("temperature"))
                    ensure_safe(run)
                    parsed = ledger.parsed
                    artifacts = {}
                    for record in result["generation_records"]:
                        rid = record["request_id"]
                        for folder, suffix in (("ledger/requests", ".json"), ("ledger/transport_outcomes", ".json"),
                                               ("raw/backend_responses", ".txt"), ("raw/prompts", ".txt"),
                                               ("raw/requests", ".json")):
                            path = root / folder / (rid + suffix)
                            if path.is_file():
                                artifacts[str(path)] = sha256_file(path)
                    atomic_json(checkpoint, {"row": row, "result": result, "parsed": parsed, "artifacts": artifacts})
                    atomic_text(checkpoint.with_suffix(".sha256"), sha256_file(checkpoint) + "\n")
                for index, record in enumerate(result["generation_records"]):
                    stage = "repair" if record.get("repair_of") else "primary"
                    stages.append({**record, **row, "stage": stage, "request_id": record["request_id"],
                                   **parsed.get(stage, {}), "replaced_by_repair": index == 0 and len(result["generation_records"]) == 2})
                slot_outcomes.append({**row, "repair": len(result["generation_records"]) == 2,
                    "whole_batch_rejected": not bool(result["candidates"]),
                    "candidate_count_before_final_validator": len(result["candidates"])})
                pool = pools.setdefault((row["arm_id"], row["root_id"]), {"candidates": [], "records": [], "identities": {}})
                pool["records"].extend(result["generation_records"])
                for candidate in result["candidates"]:
                    pool["candidates"].append(candidate)
                    pool["identities"][candidate["candidate_id"]] = {**row,
                        "stage": "repair" if len(result["generation_records"]) == 2 else "primary"}
                jsonl(run / "reports/runtime_stage_records.jsonl", stages)
            outcomes = finalize_completed_pools(run, pools, slot_outcomes)
            calls = rows(run / "audit/logical_calls.jsonl")
            result = {"status": "LIVE_COMPLETED" if backend_mode == "live" else "MOCK_COMPLETED",
                      "primary_calls": sum(x["stage"] == "primary" for x in calls),
                      "repair_calls": sum(x["stage"] == "repair" for x in calls),
                      "logical_calls": len(calls), "http_attempts": len(rows(run / "audit/http_events.jsonl")),
                      "external_calls": len(rows(run / "audit/http_events.jsonl")) if backend_mode == "live" else 0,
                      "root_arm_outcomes": outcomes, "content_status": "PENDING_OFFLINE_DIAGNOSTICS"}
            atomic_json(run / "reports/runtime_summary.json", result)
            return result
        except BaseException:
            # Preserve raw outcomes even if a failure interrupts the shared batch before it returns.
            for path in sorted((run / "arms").glob("*/qa/shards/shard-00/roots/*/stage_calls/*/*.json")):
                value = read(path)
                if not any(all(x.get(k) == value.get(k) for k in ("arm_id", "root_id", "slot_id", "stage")) for x in stages):
                    stages.append(value)
            jsonl(run / "reports/runtime_stage_records.jsonl", stages)
            outcomes = finalize_completed_pools(run, pools, slot_outcomes)
            calls = rows(run / "audit/logical_calls.jsonl")
            http_count = len(rows(run / "audit/http_events.jsonl"))
            atomic_json(run / "reports/runtime_summary.json", {"status": "STOPPED",
                "content_status": "PENDING_OFFLINE_DIAGNOSTICS", "logical_calls": len(calls),
                "primary_calls": sum(x["stage"] == "primary" for x in calls),
                "repair_calls": sum(x["stage"] == "repair" for x in calls),
                "http_attempts": http_count, "external_calls": http_count if backend_mode == "live" else 0,
                "completed_primary_slots": len(slot_outcomes), "root_arm_outcomes": outcomes,
                "stop": read(run / "audit/execution_stop.json") if (run / "audit/execution_stop.json").exists() else None})
            raise


def execute_live(run, *, authorized=False):
    """Single prepared entry; cannot discover services or issue smoke calls."""
    if not authorized:
        raise ExecutionStopped("EXPLICIT_LIVE_BUDGET_AUTHORIZATION_REQUIRED")
    from .prepare import preflight, verify, check_capacity
    run = Path(run)
    check = preflight(run, require_live=True)
    if check.get("status") != "PASS":
        raise ExecutionStopped("LIVE_PREFLIGHT_NOT_READY")
    profiles = read(run / "profiles.json")
    credentials = {}
    for arm, profile in profiles.items():
        supplied = os.environ.get(profile.get("api_key_env", ""))
        if supplied:
            credentials[arm] = supplied
        elif profile.get("credential_config_path") or profile.get("credential_format") == "challenger_json":
            # The existing Qwen secret stays in its original config; never copy or log it.
            credentials[arm] = read(profile.get("credential_config_path") or profile["credential_file"])["api"]["api_key"]
        else:
            credentials[arm] = resolve_credential(profile)
    try:
        return execute_schedule(run,
            client_factory=lambda row, profile, callback: ProfileClient(profile, credentials[row["arm_id"]], before_send=callback),
            verifier=lambda: verify(run),
            capacity=lambda arm, messages, maximum, identity: check_capacity(run, arm, messages, maximum, identity),
            backend_mode="live")
    finally:
        from .reporting import report
        report(run)
