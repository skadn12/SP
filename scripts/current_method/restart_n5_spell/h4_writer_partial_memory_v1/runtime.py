"""Bounded formal entry using the original Writer and one-attempt transport.

Preparation never invokes this module's live client. A started scope cannot be
resumed or resent: explicit new authorization and a new prepared run are needed.
"""
from __future__ import annotations

from collections import Counter
import fcntl
import json
import os
from pathlib import Path

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes
from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_writer_model_comparison_v1.prepare import _capacity
from ..h4_writer_model_comparison_v1.transport import (
    ComparisonTransportError, ProfileClient, build_payload, resolve_credential,
)
from .admission import admit_stage, normalize_stage

POLICY = {"version": "ordering_temporal_contract_v1", "node_count": 4}
BUDGETS = {"memory_assist": {"primary": 12, "repair": 12, "http": 24},
           "memory_only": {"primary": 2, "repair": 2, "http": 4},
           "writer_repair_comparison": {"primary": 12, "repair": 12, "http": 24}}


def slot_policies(run, row):
    if row["scope"] != "writer_repair_comparison":
        return POLICY, "partial_v1"
    from .repair_trial import ARM_POLICIES
    expected = ARM_POLICIES.get(row["arm_id"])
    declared = read(Path(run) / "execution_contract.json").get("arm_policies", {}).get(row["arm_id"])
    if not expected or declared != expected:
        raise ValueError("WRITER_REPAIR_ARM_POLICY_DRIFT")
    return expected["ordering_policy"], expected["admission_policy"]


class ExecutionStopped(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def write_rows(path, values):
    atomic_text(path, "".join(canonical_json(value) + "\n" for value in values))


class SlotLedger:
    def __init__(self, run, work, row, profile, root, expanded, contract, allow_unknown):
        self.run, self.work, self.row, self.profile = run, work, row, profile
        self.root, self.expanded, self.contract = root, expanded, contract
        self.allow_unknown = allow_unknown
        self.ordering_policy, self.admission_policy = slot_policies(run, row)
        self.current = None
        self.http_seen = False
        self.stopped = None
        self.parsed = {}

    def stop(self, reason, detail=None):
        self.stopped = reason
        atomic_json(self.work / "audit/execution_stop.json", {
            **self.row, "stage": self.current, "reason": reason,
            "detail": detail or {}, "automatic_retry": False, "content_repair_permitted": False})
        raise ExecutionStopped(reason)

    def before_send(self, payload):
        if self.stopped or self.current is None or self.http_seen:
            self.stop("UNBOUND_OR_REPEATED_HTTP_ATTEMPT")
        from .prepare import verify
        verify(self.run)
        if payload != self.payload:
            self.stop("HTTP_PAYLOAD_DRIFT")
        events = rows(self.work / "audit/http_events.jsonl")
        if len(events) >= BUDGETS[self.row["scope"]]["http"]:
            self.stop("HTTP_BUDGET_EXHAUSTED")
        events.append({**self.row, "stage": self.current, "request_id": self.request_id,
                       "event": "http_attempt_started", "attempt": 1,
                       "payload_sha256": sha256_bytes(canonical_json(payload).encode())})
        write_rows(self.work / "audit/http_events.jsonl", events)
        self.http_seen = True

    def invoke(self, **kwargs):
        try:
            return self._invoke(**kwargs)
        except BaseException as error:
            if self.stopped is None:
                self.stop("REQUEST_VALIDATION_OR_TECHNICAL_FAILURE", {"error_type": type(error).__name__})
            raise

    def _invoke(self, **kwargs):
        if self.stopped:
            raise ExecutionStopped(self.stopped)
        from .prepare import verify
        verify(self.run)
        rid = kwargs["request_id"]
        primary_id = f'{self.row["root_id"]}-contract_guided-generator-{self.row["slot_id"]}'
        repair_id = f'{self.row["root_id"]}-contract_guided-repair-{self.row["slot_id"]}'
        stage = "primary" if rid == primary_id else "repair" if rid == repair_id else None
        if stage is None or stage in self.parsed or (stage == "repair" and "primary" not in self.parsed):
            self.stop("REQUEST_OUTSIDE_FROZEN_SLOT")
        self.current, self.request_id, self.http_seen = stage, rid, False
        prompt = (self.root / "raw/prompts" / (rid + ".txt")).read_text()
        self.payload = build_payload(self.profile, prompt)
        if stage == "primary":
            folder = self.row["arm_id"] if self.row["scope"] == "writer_repair_comparison" else self.row["source_mode"]
            path = self.run / "policy_snapshot/primary_requests" / folder / self.row["root_id"] / self.row["slot_id"]
            if self.payload != read(path / "request.json")["payload"] or prompt != (path / "prompt.txt").read_text():
                self.stop("PRIMARY_REQUEST_DRIFT")
        else:
            from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_repair_prompt
            extra = {}
            if self.admission_policy == "partial_grounded_v2":
                # Recompute the validated primary abstention from its frozen raw
                # response, just as the shared generator does. Never trust an
                # unvalidated top-level abstention when checking repair drift.
                primary = self.parsed["primary"]
                normalized = normalize_stage(primary["raw"], intent_id=self.row["slot_id"],
                    source="contract_guided", contract=self.contract, expanded=self.expanded,
                    root_id=self.row["root_id"], stage="primary", request_id=primary["request_id"],
                    parse_error=primary["parse_error"], ordering_policy=self.ordering_policy,
                    ordering_context={"run_id": self.run.name + ":" + self.row["scope"],
                        "arm_id": self.row["arm_id"], "root_id": self.row["root_id"],
                        "root_dir": self.root, "stage": "primary", "request_id": primary["request_id"],
                        "raw_sha256": primary["raw_text_sha256"]},
                    admission_policy=self.admission_policy)
                extra["primary_abstentions"] = normalized.get("effective_abstentions", [])
            expected = build_repair_prompt(self.expanded, self.contract,
                self.parsed["primary"]["raw"], "contract_guided", ordering_policy=self.ordering_policy,
                **extra)["prompt"]
            if prompt != expected:
                self.stop("REPAIR_TEMPLATE_OR_OWN_PRIMARY_DRIFT")
        metadata = kwargs["request_metadata"]
        if (metadata.get("max_tokens") != 8192 or metadata.get("temperature") != 0.1
                or kwargs["backend_identity"].get("model") != self.profile["model"]
                or kwargs["backend_identity"].get("base_url") != self.profile["base_url"]):
            self.stop("SHARED_WRITER_PARAMETERS_DRIFT")
        capacity = _capacity(self.profile, self.payload["messages"], 8192, rid)
        atomic_json(self.root / "capacity" / (rid + ".json"), capacity)
        if capacity["status"] != "PASS" and not (capacity["status"] == "UNKNOWN" and self.allow_unknown):
            self.stop("CAPACITY_" + capacity["status"])
        logical = rows(self.work / "audit/logical_calls.jsonl")
        budget = BUDGETS[self.row["scope"]]
        if len(logical) >= budget["http"] or sum(x["stage"] == stage for x in logical) >= budget[stage]:
            self.stop("LOGICAL_BUDGET_EXHAUSTED")
        logical.append({**self.row, "stage": stage, "request_id": rid,
                        "status": "RESERVED_PERMANENTLY", "attempt": 1})
        write_rows(self.work / "audit/logical_calls.jsonl", logical)
        atomic_json(self.root / "raw/requests" / (rid + ".json"), {
            "payload": self.payload, "endpoint": self.profile["endpoint"], "stage": stage})
        try:
            response = kwargs["call"]()
        except BaseException as error:
            detail = error.metadata if isinstance(error, ComparisonTransportError) else {}
            atomic_json(self.root / "transport_outcomes" / (rid + ".json"), {
                **detail, **self.row, "stage": stage, "request_id": rid,
                "http_attempts": int(self.http_seen), "completion_status": "TECHNICAL_FAILURE",
                "error_type": type(error).__name__})
            self.stop("TECHNICAL_FAILURE_NO_RETRY_NO_REPAIR", {"error_type": type(error).__name__})
        detail = dict(response.metadata)
        atomic_json(self.root / "transport_outcomes" / (rid + ".json"), {
            **detail, **self.row, "stage": stage, "request_id": rid,
            "returned_model": response.model_identifier, "http_attempts": int(self.http_seen)})
        # Preserve content even when a missing/nonterminal finish state blocks the flow.
        atomic_text(self.root / "raw/responses" / (rid + ".json.txt"), response.raw_text)
        if not self.http_seen:
            self.stop("HTTP_CALLBACK_MISSING")
        if detail.get("finish_reason") != "stop":
            self.stop("UNKNOWN_OR_NONTERMINAL_FINISH_REASON")
        return response

    def after_call(self, *, stage, request_id, raw, raw_text, parse_error):
        if self.stopped:
            raise ExecutionStopped(self.stopped)
        self.parsed[stage] = {"raw": raw, "parse_error": parse_error,
                              "raw_text_sha256": sha256_bytes(raw_text.encode()), "request_id": request_id}
        atomic_json(self.root / "stage_calls" / self.row["slot_id"] / (stage + ".json"), {
            **self.row, "stage": stage, **self.parsed[stage]})


def execute(run, scope, *, authorized=False, allow_unknown_service=False, client_factory=None):
    if not authorized:
        raise RuntimeError("EXPLICIT_SCOPE_AND_BUDGET_AUTHORIZATION_REQUIRED")
    if scope not in BUDGETS:
        raise ValueError("UNKNOWN_SCOPE")
    run = Path(run).resolve()
    from .prepare import verify
    verify(run)
    profile = read(run / "profile.json")
    parameters = {"temperature": 0.1, "max_tokens": 8192, "response_format": {"type": "json_object"}}
    if scope == "writer_repair_comparison":
        from .repair_trial import validate_schedule
        validate_schedule(read(run / "schedule.json"))
        parameters["enable_thinking"] = False
        if (profile.get("provider"), profile.get("model"), profile.get("endpoint")) != (
                "siliconflow", "Qwen/Qwen3-14B", "https://api.siliconflow.cn/v1/chat/completions"):
            raise ValueError("WRITER_REPAIR_FROZEN_MODEL_REQUIRED")
    if profile.get("request_parameters") != parameters:
        raise ValueError("FIXED_WRITER_PARAMETERS_REQUIRED")
    if profile.get("user_service_exception"):
        raise ValueError("OLD_EXPERIMENT_EXCEPTION_CANNOT_AUTHORIZE_NEW_RUN")
    schedule = [row for row in read(run / "schedule.json") if row["scope"] == scope]
    if len(schedule) != BUDGETS[scope]["primary"]:
        raise ValueError("FROZEN_PRIMARY_COUNT_MISMATCH")
    if not allow_unknown_service and profile.get("unresolved"):
        raise RuntimeError("UNVERIFIED_SERVICE_CONFIGURATION_REQUIRES_EXPLICIT_EXCEPTION")
    credential = None
    if client_factory is None:
        if scope == "writer_repair_comparison" and profile.get("credential_format") == "challenger_json":
            # Same credential resolution as the existing paired Writer runtime;
            # secret values never enter manifests, payloads, or log messages.
            credential = os.environ.get(profile.get("api_key_env", ""))
            if not credential:
                credential = read(profile.get("credential_config_path") or profile["credential_file"])["api"]["api_key"]
        else:
            credential = resolve_credential(profile)
    work = run / "executions" / scope
    work.mkdir(parents=True, exist_ok=True)
    with (work / "execution.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (work / "audit/execution_started.json").exists():
            raise ExecutionStopped("STARTED_SCOPE_CANNOT_AUTOMATICALLY_RESUME_OR_RESEND")
        atomic_json(work / "audit/execution_started.json", {"scope": scope,
            "budget": BUDGETS[scope], "allow_unknown_service": allow_unknown_service,
            "backend": "MOCK" if client_factory is not None else "LIVE",
            "reasoning": "REQUESTED_DISABLED_ACTUAL_UNKNOWN" if scope == "writer_repair_comparison" else "UNKNOWN_SERVICE_DEFAULT",
            "automatic_retry": False})
        pools, slot_outcomes, failure = {}, [], None
        for row in schedule:
            folder = run / "inputs" / row["source_mode"] / row["root_id"]
            expanded = read(folder / "expanded.json")
            contract = read(folder / "contracts.json")[row["raw_index"]]
            root = work / "arms" / row["arm_id"] / "roots" / row["root_id"]
            context = {"run_id": run.name + ":" + scope, "arm_id": row["arm_id"]}
            ledger = SlotLedger(run, work, row, profile, root, expanded, contract, allow_unknown_service)
            ordering_policy, admission_policy = ledger.ordering_policy, ledger.admission_policy
            client = (client_factory(row, profile, ledger.before_send) if client_factory is not None else
                      ProfileClient(profile, credential=credential, before_send=ledger.before_send))
            key = (row["arm_id"], row["root_id"])
            pool = pools.setdefault(key, {"candidates": [], "records": [], "expanded": expanded,
                                         "root": root, "context": context,
                                         "ordering_policy": ordering_policy, "admission_policy": admission_policy})
            try:
                result = generation.execute_writer_batch(expanded=expanded, contract=contract,
                    root_dir=root, root_id=row["root_id"], ledger=ledger, client=client,
                    after_call=ledger.after_call, ordering_policy=ordering_policy, ordering_context=context,
                    writer_max_tokens=8192, writer_temperature=0.1, admission_policy=admission_policy)
                pool["candidates"].extend(result["candidates"])
                pool["records"].extend(result["generation_records"])
                slot_outcomes.append({**row, "status": "COMPLETED", "candidate_count": len(result["candidates"]),
                    "repair_requested": result["repair_requested"], "admission_summary": result["admission_summary"]})
            except BaseException as error:
                failure = type(error).__name__ + ":" + str(error) if isinstance(error, ExecutionStopped) else type(error).__name__
                # Preserve an already completed primary even if its repair failed technically.
                retained = []
                inherited_abstentions = []
                acceptance = None
                for stage, parsed in ledger.parsed.items():
                    ordering_context = {**context, "root_id": row["root_id"], "root_dir": root,
                        "stage": stage, "request_id": parsed["request_id"], "raw_sha256": parsed["raw_text_sha256"]}
                    normalized = normalize_stage(parsed["raw"], intent_id=row["slot_id"], source="contract_guided",
                        contract=contract, expanded=expanded, root_id=row["root_id"], stage=stage,
                        request_id=parsed["request_id"], parse_error=parsed["parse_error"],
                        ordering_policy=ordering_policy, ordering_context=ordering_context,
                        **({"admission_policy": admission_policy, "inherited_abstentions": inherited_abstentions}
                           if admission_policy == "partial_grounded_v2" else {}))
                    retained, acceptance = admit_stage(normalized, retained=retained,
                        **({"admission_policy": admission_policy} if admission_policy == "partial_grounded_v2" else {}))
                    if stage == "primary":
                        inherited_abstentions = normalized.get("effective_abstentions", [])
                    pool["records"].append({"request_id": parsed["request_id"], "stage": stage,
                                             "validation": normalized, "minimum_acceptance": acceptance})
                pool["candidates"].extend(retained)
                slot_outcomes.append({**row, "status": "STOPPED", "reason": failure,
                                      "candidate_count": len(retained), "admission_summary": acceptance})
                break
        finals, outcomes = [], []
        for (arm, root_id), pool in pools.items():
            if not pool["candidates"] and not pool["records"]:
                continue
            outcome = generation.finalize_writer_candidates(root_dir=pool["root"], root_id=root_id,
                candidates=pool["candidates"], generation_records=pool["records"], expanded=pool["expanded"],
                ordering_policy=pool["ordering_policy"], ordering_context=pool["context"], admission_policy=pool["admission_policy"])
            outcomes.append({"arm_id": arm, **outcome})
            finals.extend({"arm_id": arm, **candidate} for candidate in read(pool["root"] / "candidates_valid.json"))
        all_calls = [read(path) for path in sorted((work / "arms").glob("*/roots/*/transport_outcomes/*.json"))]
        raw_stages = []
        for pool in pools.values():
            for path in sorted((pool["root"] / "stage_calls").glob("*/*.json")):
                raw_stages.append(read(path))
        counts = Counter(row["stage"] for row in rows(work / "audit/logical_calls.jsonl"))
        summary = {"status": "STOPPED" if failure else "COMPLETED", "scope": scope,
            "primary_calls": counts["primary"], "repair_calls": counts["repair"],
            "logical_calls": sum(counts.values()), "http_attempts": len(rows(work / "audit/http_events.jsonl")),
            "unattempted_primary_slots": len(schedule)-counts["primary"], "technical_failure": failure,
            "program_accepted": len(finals), "semantic_diagnostic": "NOT_RUN",
            "memory_support": "NOT_ASSESSED" if scope == "memory_only" else "NOT_APPLICABLE",
            "transcript_support": "NOT_ASSESSED",
            "actual_money": "UNKNOWN", "root_outcomes": outcomes,
            "automatic_retries": 0, "external_calls": 0 if client_factory is not None else len(rows(work / "audit/http_events.jsonl"))}
        if scope == "writer_repair_comparison":
            from .repair_trial import grouped_metrics
            metrics = grouped_metrics(schedule, finals, raw_stages, all_calls, slot_outcomes,
                                      logical_calls=rows(work / "audit/logical_calls.jsonl"))
            atomic_json(work / "reports/root_arm_metrics.json", metrics)
            summary["root_arm_metrics"] = "reports/root_arm_metrics.json"
        write_rows(work / "reports/final_candidates.jsonl", finals)
        write_rows(work / "reports/raw_stage_candidates.jsonl", raw_stages)
        write_rows(work / "reports/all_call_costs.jsonl", all_calls)
        atomic_json(work / "reports/slot_outcomes.json", slot_outcomes)
        atomic_json(work / "reports/summary.json", summary)
        if failure:
            raise ExecutionStopped(failure)
        return summary
