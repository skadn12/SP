"""Frozen Writer-only schedule around the original client, ledger and stage.

Importing never sends. execute_schedule is the offline fake-client seam; only
execute_live constructs the network client, behind the explicitly authorized CLI.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import urllib.error
import urllib.request
from unittest.mock import patch

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..h4_r02_feedback_ablation_p0_p1_v1.qa_run import (
    GuardrailStop, ObservedClient, append, exclusive_run, now,
)
from .constants import ARMS, ROOTS
from .guidance import quality_suffix, repair_spec

MAX_OUTPUT = 8192
PRIMARY_LIMIT, REPAIR_LIMIT, TOTAL_LIMIT = 12, 12, 24
KNOWN_REJECTIONS = frozenset((400, 401, 402, 403, 404, 405, 406, 413, 415, 422, 429))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()] if Path(path).is_file() else []


def root_path(run, arm, root):
    return Path(run) / "arms" / arm / "qa/shards/shard-00/roots" / root


def request_id(row, stage):
    verb = "generator" if stage == "primary" else "repair"
    return f"{row['root_id']}-contract_guided-{verb}-{row['slot_id']}"


def global_id(row, stage):
    return "::".join((row["arm_id"], row["root_id"], row["slot_id"], stage, request_id(row, stage)))


def expected_schedule():
    result = []
    for root_index, root in enumerate(ROOTS):
        for slot in range(1, 4):
            arms = ARMS if (root_index + slot) % 2 else ARMS[::-1]
            for arm in arms:
                result.append({"order": len(result) + 1, "root_id": root, "arm_id": arm,
                               "raw_slot": slot, "raw_index": slot - 1,
                               "intent_id": f"intent_{slot}", "slot_id": f"intent_{slot}"})
    for row in result:
        row["request_id"] = request_id(row, "primary")
    return result


def frozen_schedule(run):
    actual = read(Path(run) / "schedule.json")
    expected = expected_schedule()
    if (not isinstance(actual, list) or len(actual) != 12 or
            any(any(row.get(k) != value for k, value in wanted.items())
                for row, wanted in zip(actual, expected))):
        raise GuardrailStop("writer_schedule_drift")
    return actual


def stage_policy(run, row):
    """Match the frozen manifest to arguments actually passed to the shared stage."""
    suffix = quality_suffix(run, row["arm_id"])
    policy = {"source": "contract_guided", "intent_guidance": True,
              "slot_suffix_sha256": sha256_bytes(suffix.encode("utf-8")), "hooks": None}
    manifest = rows(Path(run) / "slot_call_manifest.jsonl")
    if len(manifest) != PRIMARY_LIMIT:
        raise GuardrailStop("slot_call_manifest_inventory_drift")
    matches = [item for item in manifest if all(item.get(key) == row[key]
               for key in ("arm_id", "root_id", "slot_id"))]
    if len(matches) != 1 or any(matches[0].get(key) != value for key, value in policy.items()):
        raise GuardrailStop("slot_manifest_actual_stage_arguments_drift")
    prompt = (Path(run) / "policy_snapshot/primary_requests" / row["arm_id"] /
              row["root_id"] / row["slot_id"] / "prompt.txt")
    if matches[0].get("prompt_sha256") != sha256_file(prompt):
        raise GuardrailStop("slot_manifest_primary_prompt_drift")
    return policy


def cell_path(run, row):
    return root_path(run, row["arm_id"], row["root_id"]) / "cells" / (row["slot_id"] + ".json")


def record_guardrail(run, reason, row=None, stage=None):
    append(Path(run) / "audit/guardrail_events.jsonl", {
        "at_utc": now(), "reason": str(reason), "status": "BLOCKED", "sticky": True,
        "automatic_resend": False, "arm_id": (row or {}).get("arm_id"),
        "root_id": (row or {}).get("root_id"), "slot_id": (row or {}).get("slot_id"), "stage": stage})


def assert_safe(run):
    events = rows(Path(run) / "audit/guardrail_events.jsonl")
    if events:
        raise GuardrailStop("sticky_guardrail:" + events[0]["reason"])


def _failure_status(error):
    return ("KNOWN_HTTP_REJECTION" if isinstance(error, urllib.error.HTTPError)
            and error.code in KNOWN_REJECTIONS else "UNKNOWN_COMPLETION")


def validate_execution_contract(run):
    """No CLI/config budget override can enlarge this frozen experiment."""
    from .constants import GENERATION
    contract = read(Path(run) / "execution_contract.json")
    expected = {"generation": GENERATION, "roots": list(ROOTS), "arms": list(ARMS),
                "planned_primary_calls": PRIMARY_LIMIT, "maximum_repair_calls": REPAIR_LIMIT,
                "maximum_logical_calls": TOTAL_LIMIT,
                "per_arm": {"primary": 6, "repair": 6, "total": 12},
                "per_root_arm": {"primary": 3, "repair": 3, "total": 6},
                "per_cell": {"primary": 1, "repair": 1}, "context_reserve_tokens": 512}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise GuardrailStop("execution_budget_or_configuration_drift")


class SendingGuardClient:
    """Inspect every original HTTP send, including unchanged transport retries."""
    def __init__(self, delegate, run, row, verifier, tokens):
        self.delegate, self.config = delegate, delegate.config
        self.run, self.row, self.verifier, self.tokens = run, row, verifier, tokens
        self.stage, self.prompt_sha = None, None

    def bind(self, stage, prompt_sha):
        self.stage, self.prompt_sha = stage, prompt_sha

    def complete(self, *, prompt, max_tokens, json_object):
        if (self.stage not in ("primary", "repair") or max_tokens != MAX_OUTPUT or
                json_object is not True or sha256_bytes(prompt.encode()) != self.prompt_sha):
            raise GuardrailStop("unbound_or_changed_http_request")
        expected_payload = {"model": "Qwen/Qwen3-14B",
            "messages": [{"role": "user", "content": prompt}], "temperature": 0.1,
            "max_tokens": MAX_OUTPUT, "enable_thinking": False,
            "response_format": {"type": "json_object"}}
        original_urlopen = urllib.request.urlopen
        def guarded_urlopen(request, *args, **kwargs):
            try:
                assert_safe(self.run)
                self.verifier()
                validate_execution_contract(self.run)
                if (request.full_url != "https://api.siliconflow.cn/v1/chat/completions" or
                        request.get_method() != "POST" or
                        json.loads(request.data.decode("utf-8")) != expected_payload or
                        kwargs.get("timeout") != 180):
                    raise GuardrailStop("actual_http_payload_or_timeout_drift")
                self.tokens.check(prompt, MAX_OUTPUT, global_id(self.row, self.stage))
            except BaseException as error:
                record_guardrail(self.run, error, self.row, self.stage)
                raise
            return original_urlopen(request, *args, **kwargs)
        with patch("urllib.request.urlopen", guarded_urlopen):
            return self.delegate.complete(prompt=prompt, max_tokens=max_tokens, json_object=json_object)


def _checkpoint_valid(run, row):
    path = cell_path(run, row)
    matches = [e for e in rows(Path(run) / "audit/cell_completion_events.jsonl")
               if e.get("order") == row["order"]]
    if not path.exists() and not matches:
        return False
    if len(matches) != 1 or not path.exists() or sha256_file(path) != matches[0]["sha256"]:
        raise GuardrailStop("cell_checkpoint_drift")
    checkpoint = read(path)
    if checkpoint.get("schedule_row") != row:
        raise GuardrailStop("cell_checkpoint_identity_drift")
    for artifact, sha in checkpoint.get("request_artifacts", {}).items():
        if not Path(artifact).is_file() or sha256_file(Path(artifact)) != sha:
            raise GuardrailStop("cell_request_artifact_drift:" + artifact)
    return True


def audit_reservations(run):
    """Unknown reservations, missing logs and moved states stop recovery."""
    assert_safe(run)
    schedule = frozen_schedule(run)
    manifest_sha = sha256_file(Path(run) / "audit/frozen_manifest.json")
    schedule_sha = sha256_file(Path(run) / "schedule.json")
    by_identity = {global_id(row, stage): (row, stage)
                   for row in schedule for stage in ("primary", "repair")}
    states = {}
    for path in sorted(Path(run).glob("**/ledger/requests/*.json")):
        state = read(path)
        metadata = state.get("request_metadata") or {}
        identity = metadata.get("global_request_id")
        if identity not in by_identity or identity in states:
            raise GuardrailStop("request_inventory_identity_drift")
        row, stage = by_identity[identity]
        policy = stage_policy(run, row)
        root = root_path(run, row["arm_id"], row["root_id"])
        if path.resolve() != (root / "ledger/requests" / (request_id(row, stage) + ".json")).resolve():
            raise GuardrailStop("request_directory_scope_drift")
        if (metadata.get("schedule_row") != row or metadata.get("stage") != stage or
                any(metadata.get(key) != value for key, value in policy.items()) or
                metadata.get("max_tokens") != MAX_OUTPUT or metadata.get("temperature") != 0.1 or
                state.get("request_id") != request_id(row, stage) or
                state.get("intent") != ("h4_k3_candidate_generator" if stage == "primary"
                                        else "h4_k3_candidate_schema_repair") or
                state.get("attempt") != 1 or state.get("budget_limit") != 6 or
                state.get("total_budget_limit") != 6 or state.get("budget_class") != "fresh_qa_generation" or
                state.get("manifest_sha256") != manifest_sha or
                metadata.get("schedule_sha256") != schedule_sha or
                state.get("status") not in ("reserved", "completed") or
                state.get("backend_mode") not in ("mock", "live") or
                state.get("request_metadata_sha256") != sha256_bytes(canonical_json(metadata).encode())):
            raise GuardrailStop("request_reservation_contract_drift")
        backend = state.get("backend_identity") or {}
        if (backend.get("model") != "Qwen/Qwen3-14B" or
                backend.get("base_url") != "https://api.siliconflow.cn/v1" or
                backend.get("temperature") != 0.1 or
                state.get("backend_identity_sha256") != sha256_bytes(canonical_json(backend).encode())):
            raise GuardrailStop("reservation_backend_identity_drift")
        outcome_path = root / "ledger/transport_outcomes" / (state["request_id"] + ".json")
        if not outcome_path.exists():
            raise GuardrailStop("unknown_completion_missing_transport_outcome")
        outcome = read(outcome_path)
        if (outcome.get("global_request_id") != identity or
                outcome.get("prompt_sha256") != state.get("prompt_sha256") or
                outcome.get("max_tokens") != MAX_OUTPUT):
            raise GuardrailStop("transport_identity_drift")
        if state.get("status") == "completed":
            if outcome.get("service_completion_status") != "COMPLETED":
                raise GuardrailStop("ledger_transport_completion_disagreement")
            raw = root / "raw/backend_responses" / (state["request_id"] + ".txt")
            if (Path(state.get("raw_response_path", "")).resolve() != raw.resolve() or
                    not raw.exists() or sha256_file(raw) != state.get("raw_response_sha256")):
                raise GuardrailStop("completed_raw_response_drift")
        elif outcome.get("service_completion_status") != "KNOWN_HTTP_REJECTION":
            raise GuardrailStop("unknown_completion_no_automatic_resend_or_repair")
        elif outcome.get("http_status") not in KNOWN_REJECTIONS:
            raise GuardrailStop("known_rejection_without_explicit_http_status")
        if state.get("backend_mode") == "live" and not outcome.get("transport_attempt_count"):
            raise GuardrailStop("live_transport_attempts_missing")
        states[identity] = state
    events = [e for e in rows(Path(run) / "audit/call_events.jsonl") if e.get("event") == "new_logical_call"]
    if (len(events) != len(states) or len(events) > TOTAL_LIMIT or
            {e.get("global_request_id") for e in events} != set(states)):
        raise GuardrailStop("logical_reservation_event_inventory_drift")
    primaries, repairs = [], []
    for i, event in enumerate(events):
        row, stage = by_identity[event["global_request_id"]]
        if stage == "primary":
            primaries.append(row["order"])
        else:
            repairs.append(row["order"])
            if i == 0 or events[i - 1]["global_request_id"] != global_id(row, "primary"):
                raise GuardrailStop("repair_not_immediately_after_own_primary")
    if primaries != list(range(1, len(primaries) + 1)):
        raise GuardrailStop("primary_schedule_order_drift")
    if len(primaries) > PRIMARY_LIMIT or len(repairs) > REPAIR_LIMIT:
        raise GuardrailStop("logical_call_budget_exceeded")
    return states


class BoundedLedger:
    """One primary and at most one replacement repair for one frozen slot."""
    def __init__(self, run, row, *, manifest_sha256, verifier, token_preflight,
                 backend_mode="mock", observed_client=None):
        self.run, self.row = Path(run), dict(row)
        self.root = root_path(run, row["arm_id"], row["root_id"])
        self.delegate = RequestLedger(self.root, manifest_sha256, backend_mode)
        self.verifier, self.tokens = verifier, token_preflight
        self.observed_client = observed_client
        self.schedule_sha = sha256_file(self.run / "schedule.json")
        self._ordinary_failure = None
        self.after_call_results = {}

    def _event(self, event, stage, **extra):
        append(self.run / "audit/call_events.jsonl", {
            "event": event, "at_utc": now(), **self.row, "stage": stage,
            "request_id": request_id(self.row, stage), "global_request_id": global_id(self.row, stage), **extra})

    def invoke(self, **kwargs):
        self._ordinary_failure = None
        try:
            return self._invoke(**kwargs)
        except BaseException as error:
            if error is not self._ordinary_failure:
                record_guardrail(self.run, error, self.row)
            raise

    def _invoke(self, **kwargs):
        assert_safe(self.run)
        rid = kwargs.get("request_id")
        stage = next((s for s in ("primary", "repair") if rid == request_id(self.row, s)), None)
        meta = kwargs.get("request_metadata") or {}
        backend = kwargs.get("backend_identity") or {}
        if (stage is None or kwargs.get("attempt") != 1 or
                kwargs.get("intent") != ("h4_k3_candidate_generator" if stage == "primary" else
                                         "h4_k3_candidate_schema_repair") or
                kwargs.get("budget_class") != "fresh_qa_generation" or
                meta.get("max_tokens") != MAX_OUTPUT or meta.get("temperature") != 0.1 or
                meta.get("request_id") != rid or "seed" in meta or
                backend.get("model") != "Qwen/Qwen3-14B" or
                backend.get("base_url") != "https://api.siliconflow.cn/v1" or
                backend.get("temperature") != 0.1):
            raise GuardrailStop("request_role_or_configuration_drift")
        self.verifier()
        if sha256_file(self.run / "schedule.json") != self.schedule_sha:
            raise GuardrailStop("schedule_changed_during_execution")
        states = audit_reservations(self.run)
        prompt_path = self.root / "raw/prompts" / (rid + ".txt")
        prompt = prompt_path.read_text(encoding="utf-8")
        if sha256_bytes(prompt.encode()) != kwargs.get("prompt_sha256"):
            raise GuardrailStop("actual_prompt_sha_drift")
        if stage == "primary":
            frozen = (self.run / "policy_snapshot/primary_requests" / self.row["arm_id"] /
                      self.row["root_id"] / self.row["slot_id"] / "prompt.txt")
            if prompt.encode() != frozen.read_bytes():
                raise GuardrailStop("primary_prompt_not_exact_frozen_bytes")
        else:
            if "primary" not in self.after_call_results:
                raise GuardrailStop("repair_without_same_cell_primary_parse")
            shared = self.run / "inputs/shared" / self.row["root_id"]
            expected = repair_spec(read(shared / "expanded_dialogue_clean.json"),
                read(shared / "contracts.json")[self.row["raw_index"]],
                self.after_call_results["primary"]["raw_parsed"], self.row["arm_id"], self.run)
            if prompt != expected["prompt"]:
                raise GuardrailStop("repair_prompt_not_original_builder_plus_frozen_suffix")
        policy = stage_policy(self.run, self.row)
        identity = global_id(self.row, stage)
        self.tokens.check(prompt, MAX_OUTPUT, identity)
        metadata = {**meta, "arm_id": self.row["arm_id"], "root_id": self.row["root_id"],
                    "slot_id": self.row["slot_id"], "stage": stage, "global_request_id": identity,
                    "schedule_row": self.row, "schedule_sha256": self.schedule_sha, **policy}
        kwargs.update(request_metadata=metadata, budget_limit=6, total_budget_limit=6)
        existing = self.delegate.read(rid)
        if existing:
            expected = {"intent": kwargs["intent"], "attempt": 1, "prompt_sha256": kwargs["prompt_sha256"],
                        "manifest_sha256": self.delegate.manifest_sha256,
                        "backend_mode": self.delegate.backend_mode, "budget_class": "fresh_qa_generation",
                        "request_metadata_sha256": sha256_bytes(canonical_json(metadata).encode()),
                        "backend_identity_sha256": sha256_bytes(canonical_json(backend).encode())}
            if any(existing.get(k) != v for k, v in expected.items()):
                raise GuardrailStop("completed_cache_identity_drift")
            if existing["status"] != "completed":
                self._event("known_failure_replayed_no_send", stage)
                failure = read(self.root / "ledger/transport_outcomes" / (rid + ".json"))
                # Reproduce the original _call_json error text without resending.
                error_class = type(failure["error_type"], (Exception,), {})
                self._ordinary_failure = error_class(failure["error_message"])
                raise self._ordinary_failure
            result = self.delegate.invoke(**kwargs)
            self._event("completed_cache_reused_no_send", stage)
            self.tokens.verify_usage(identity, result.metadata.get("usage"))
            return result
        if len(states) >= TOTAL_LIMIT:
            raise GuardrailStop("total_logical_budget_exhausted")
        for previous in frozen_schedule(self.run)[:self.row["order"] - 1]:
            if not _checkpoint_valid(self.run, previous):
                raise GuardrailStop("previous_scheduled_cell_not_complete")
        if _checkpoint_valid(self.run, self.row):
            raise GuardrailStop("completed_cell_cannot_add_requests")
        if stage == "primary":
            if sum(s["request_metadata"]["stage"] == "primary" for s in states.values()) != self.row["order"] - 1:
                raise GuardrailStop("primary_outside_fixed_order")
        elif global_id(self.row, "primary") not in states:
            raise GuardrailStop("repair_without_conclusive_primary")
        if self.observed_client is not None:
            self.observed_client.event_path = self.root / "ledger/http_events" / (rid + ".jsonl")
            self.observed_client.delegate.bind(stage, kwargs["prompt_sha256"])
        original_call = kwargs["call"]
        def observed_call():
            response, error = None, None
            try:
                response = original_call()
                return response
            except BaseException as caught:
                error = caught
                raise
            finally:
                transport = rows(self.root / "ledger/http_events" / (rid + ".jsonl"))
                attempts = sum(e.get("event") == "http_attempt_started" for e in transport)
                for event in transport:
                    append(self.run / "audit/http_events.jsonl", {**event, **self.row,
                        "stage": stage, "request_id": rid, "global_request_id": identity})
                outcome = {"global_request_id": identity, "prompt_sha256": kwargs["prompt_sha256"],
                           "max_tokens": MAX_OUTPUT, "service_completion_status":
                           "COMPLETED" if response is not None else _failure_status(error),
                           "completion_received": response is not None,
                           "transport_attempt_count": attempts,
                           "http_status": getattr(error, "code", None),
                           "error_type": type(error).__name__ if error is not None else None,
                           "error_message": str(error) if error is not None else None,
                           "usage": response.metadata.get("usage") if response is not None else None,
                           "at_utc": now()}
                atomic_json(self.root / "ledger/transport_outcomes" / (rid + ".json"), outcome)
                if outcome["service_completion_status"] == "UNKNOWN_COMPLETION":
                    record_guardrail(self.run, "unknown_service_completion:" + str(error), self.row, stage)
        kwargs["call"] = observed_call
        self._event("new_logical_call", stage, experiment_call_index=len(states) + 1)
        try:
            result = self.delegate.invoke(**kwargs)
        except Exception as error:
            reservation = self.delegate.read(rid)
            if not reservation:
                raise GuardrailStop("ledger_failed_before_reservation") from error
            if reservation.get("status") == "completed":
                raise GuardrailStop("ledger_failed_after_recording_completion") from error
            self._event("logical_call_failed", stage, service_completion_status=_failure_status(error))
            self._ordinary_failure = error
            raise
        self._event("logical_call_completed", stage, usage=result.metadata.get("usage"))
        self.tokens.verify_usage(identity, result.metadata.get("usage"))
        return result

    def after_call(self, *, stage, request_id, raw=None, raw_text="", parse_error=None):
        """Mandatory immediately after _call_json, before a possible repair."""
        self.after_call_results[stage] = {"raw_parsed": raw, "parse_error": parse_error}
        try:
            policy = stage_policy(self.run, self.row)
        except BaseException as error:
            record_guardrail(self.run, error, self.row, stage)
            raise
        # Retain even a swallowed failure before raising; partial-run diagnostics
        # can enumerate all scheduled cells and this stage without inventing QA.
        atomic_json(self.root / "stage_calls" / self.row["slot_id"] / (stage + ".json"), {
            **self.row, "stage": stage, "request_id": request_id,
            **policy,
            "contract_alignment_analysis": "AVAILABLE_FOR_OFFLINE_CONTENT_REVIEW",
            "global_request_id": global_id(self.row, stage),
            **self.after_call_results[stage], "raw_text_sha256": sha256_bytes(raw_text.encode()),
            "raw_response_path": str(self.root / "raw/responses" / (request_id + ".json.txt"))})
        assert_safe(self.run)
        audit_reservations(self.run)
        if request_id != globals()["request_id"](self.row, stage) or not self.delegate.read(request_id):
            record_guardrail(self.run, "stage_returned_without_matching_reservation", self.row, stage)
            raise GuardrailStop("stage_returned_without_matching_reservation")


def _write_jsonl(path, records):
    atomic_text(path, "".join(canonical_json(row) + "\n" for row in records))


def _candidate_sidecar(run, row, stage, candidate):
    return {**candidate, "arm_id": row["arm_id"], "root_id": row["root_id"],
            "slot_id": row["slot_id"], "raw_slot": row["raw_slot"], "stage": stage,
            "analysis_identity": [row["arm_id"], row["root_id"], row["slot_id"], stage,
                                  candidate.get("candidate_id")],
            **stage_policy(run, row),
            "contract_alignment_analysis": "AVAILABLE_FOR_OFFLINE_CONTENT_REVIEW"}


def execute_schedule(run, *, client_factory, verifier, token_preflight, backend_mode="mock"):
    """Use injected clients. Synthetic tests must use a temporary directory."""
    from ..h4_qa_multiroot_k3_t08_probe_v1.generation import (
        execute_writer_batch, finalize_writer_candidates,
    )
    run = Path(run)
    with exclusive_run(run):
        verifier()
        validate_execution_contract(run)
        schedule = frozen_schedule(run)
        for row in schedule:
            stage_policy(run, row)
        manifest_sha = sha256_file(run / "audit/frozen_manifest.json")
        assert_safe(run)
        audit_reservations(run)
        stage_rows, final_rows, outcomes = [], [], []
        by_root_arm = {}
        for row in schedule:
            arm, root_id = row["arm_id"], row["root_id"]
            root = root_path(run, arm, root_id)
            shared = run / "inputs/shared" / root_id
            expanded, contracts = read(shared / "expanded_dialogue_clean.json"), read(shared / "contracts.json")
            if len(contracts) != 3 or [c.get("intent_id") for c in contracts] != ["intent_1", "intent_2", "intent_3"]:
                raise GuardrailStop("six_original_contract_slots_required")
            contract = contracts[row["raw_index"]]
            if _checkpoint_valid(run, row):
                checkpoint = read(cell_path(run, row))
                result, parsed = checkpoint["result"], checkpoint.get("after_call_results", {})
            else:
                client = client_factory(row)
                observed = (ObservedClient(SendingGuardClient(client, run, row, verifier, token_preflight),
                                            run / "audit/http_events.jsonl", arm, root_id)
                            if backend_mode == "live" else None)
                ledger = BoundedLedger(run, row, manifest_sha256=manifest_sha, verifier=verifier,
                                       token_preflight=token_preflight, backend_mode=backend_mode,
                                       observed_client=observed)
                result = execute_writer_batch(expanded=expanded, contract=contract, root_dir=root,
                    root_id=root_id, ledger=ledger, client=observed or client, source="contract_guided",
                    intent_guidance=True, slot_suffix=quality_suffix(run, arm), hooks=None,
                    after_call=ledger.after_call)
                assert_safe(run)
                audit_reservations(run)
                parsed = ledger.after_call_results
                artifact_hashes = {}
                for record in result["generation_records"]:
                    rid = record["request_id"]
                    for rel in ("ledger/requests/" + rid + ".json", "ledger/transport_outcomes/" + rid + ".json",
                                "raw/backend_responses/" + rid + ".txt", "raw/prompts/" + rid + ".txt",
                                "raw/responses/" + rid + ".json.txt"):
                        path = root / rel
                        if path.is_file():
                            artifact_hashes[str(path)] = sha256_file(path)
                atomic_json(cell_path(run, row), {"schedule_row": row, "result": result,
                    "after_call_results": parsed, "request_artifacts": artifact_hashes})
                append(run / "audit/cell_completion_events.jsonl", {"order": row["order"],
                       "path": str(cell_path(run, row)), "sha256": sha256_file(cell_path(run, row)), "at_utc": now()})
            bucket = by_root_arm.setdefault((arm, root_id), {"candidates": [], "records": [], "analysis": {}})
            for index, record in enumerate(result["generation_records"]):
                stage = "repair" if record.get("repair_of") else "primary"
                stage_rows.append({**record, **row, "request_id": record["request_id"], "stage": stage,
                                   **stage_policy(run, row),
                                   "contract_alignment_analysis": "AVAILABLE_FOR_OFFLINE_CONTENT_REVIEW",
                                   **parsed.get(stage, {}), "global_request_id": global_id(row, stage),
                                   "raw_response_path": str(root / "raw/responses" / (record["request_id"] + ".json.txt")),
                                   "replaced_by_repair": index == 0 and len(result["generation_records"]) == 2})
            chosen_stage = "repair" if len(result["generation_records"]) == 2 else "primary"
            bucket["records"].extend(result["generation_records"])
            bucket["candidates"].extend(result["candidates"])
            for candidate in result["candidates"]:
                bucket["analysis"].setdefault(candidate["candidate_id"], []).append((dict(row), chosen_stage))
            _write_jsonl(run / "reports/runtime_stage_records.jsonl", stage_rows)
        for (arm, root_id), bucket in by_root_arm.items():
            root = root_path(run, arm, root_id)
            expanded = read(run / "inputs/shared" / root_id / "expanded_dialogue_clean.json")
            outcome = finalize_writer_candidates(root_dir=root, root_id=root_id,
                candidates=bucket["candidates"], generation_records=bucket["records"], expanded=expanded)
            outcomes.append({**outcome, "arm_id": arm})
            for candidate in read(root / "candidates_valid.json"):
                row, stage = bucket["analysis"][candidate["candidate_id"]].pop(0)
                final_rows.append(_candidate_sidecar(run, row, stage, candidate))
        _write_jsonl(run / "reports/runtime_final_candidates.jsonl", final_rows)
        states = audit_reservations(run)
        counts = Counter(state["request_metadata"]["stage"] for state in states.values())
        result = {"status": "COMPLETED", "primary_calls": counts["primary"], "repair_calls": counts["repair"],
                  "logical_calls": len(states), "root_arm_outcomes": outcomes,
                  "backend_mode": backend_mode, "external_calls": 0 if backend_mode == "mock" else len(states)}
        atomic_json(run / "reports/runtime_summary.json", result)
        return result


def execute_live(run, config_path, *, verify_frozen=None):
    """Future live entrypoint; preparation exercises it only with mocked transport."""
    from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
    from ..h4_r02_feedback_transfer_p2_2_v1.token_preflight import TokenPreflight
    if verify_frozen is None:
        from .dependencies import verify_frozen
    run, config_path = Path(run), Path(config_path)
    manifest = verify_frozen(run, config_path)
    validate_execution_contract(run)
    preflight = read(run / "audit/preflight_report.json")
    if preflight.get("status") not in ("PASS", "READY_FOR_AUTHORIZATION"):
        raise GuardrailStop("prepared_preflight_not_passed")
    config = replace(load_challenger_config(config_path), temperature=0.1,
                     enable_thinking=False, timeout_seconds=180)
    if config.model != "Qwen/Qwen3-14B" or config.base_url != "https://api.siliconflow.cn/v1":
        raise GuardrailStop("writer_backend_identity_drift")
    tokens = TokenPreflight(run)
    tokens.report_path = run / "audit/live_token_report.json"
    from .dependencies import repository_execution_guard
    with repository_execution_guard(manifest["code"]):
        return execute_schedule(run, client_factory=lambda row: OpenAICompatibleChallengerClient(config),
            verifier=lambda: verify_frozen(run, config_path), token_preflight=tokens, backend_mode="live")
