"""P0 pipeline adapter using the existing global12 ledger and proxy route.

The question logic and four-stage orchestration are inherited from frozen v2.
This version changes only execution authorization, attempt accounting, and the
single-attempt transport binding. Construction defaults to dry-run.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import uuid

from . import base_qa_executor_v2 as v2
from . import base_qa_global_budget_adapter_v3 as budget
from . import base_qa_pipeline_v1 as qa
from . import base_qa_proxy_transport_v3 as proxy
from . import qa_api_transport_v2 as transport_api

VERSION = budget.VERSION
PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = budget.ARTIFACTS
PROFILE = v2.PROFILE
ROOTS = budget.ROOTS
STAGES = budget.STAGES
OUTPUT_LIMIT = budget.MAX_OUTPUT
INPUT_LIMITS = v2.INPUT_LIMITS
LIVE_EXECUTION_ENABLED = False


def _read_json(path):
    return transport_api._strict_json(Path(path).read_bytes())


def _freeze_state(path, value):
    v2.freeze(path, "EXPERIMENT", value)


def initialize(state_dir, selection_path, arm="G"):
    """Freeze a new v3 manifest from the already reviewed three-root selection."""
    if arm not in ("F", "G"):
        raise v2.ExecutionHold("invalid F/G arm")
    state = Path(state_dir).resolve()
    selection_path = Path(selection_path).resolve()
    selection = qa.read(selection_path)
    selected = {}
    for root_id in ROOTS:
        rows = [row for row in selection["roots"] if row.get("root_id") == root_id]
        if len(rows) != 1:
            raise v2.ExecutionHold("exactly one reviewed source per approved root required")
        row = rows[0]
        binding = qa.legacy.file_binding(row["path"])
        if row.get("sha256") != binding["sha256"] or row.get("bytes") != binding["bytes"]:
            raise v2.ExecutionHold("selected source binding drift")
        qa.source_from(binding, root_id)
        warnings_path = row.get("warnings_path")
        selected[root_id] = {"source_binding": binding,
            "warnings_binding": qa.legacy.file_binding(warnings_path) if warnings_path else None}

    state.mkdir(parents=True, exist_ok=False)
    plan = ARTIFACTS.parent / "base_qa_p0_20261027_v1" / "API_PLAN.json"
    takeover = budget.CANONICAL_DIR / "BUDGET_TAKEOVER.json"
    manifest = {"version": VERSION, "experiment_id": VERSION,
        "state_directory": str(state), "ledger_path": str(budget.CANONICAL_LEDGER.resolve()),
        "budget_guard_binding": qa.legacy.file_binding(takeover),
        "roots": list(ROOTS), "stages": list(STAGES), "maximum_attempts": budget.GLOBAL_LIMIT,
        "max_output_tokens": OUTPUT_LIMIT, "input_limits": INPUT_LIMITS,
        "arm": arm, "feedback_enabled": False, "language": "en",
        "selection_binding": qa.legacy.file_binding(selection_path),
        "sources": selected, "implementation_bindings": bindings(),
        "api_plan_binding": qa.legacy.file_binding(plan),
        "network_authorization": "REQUIRES_SEPARATE_APPROVAL_AND_VERIFIED_CAPABILITY",
        "retry_attempts": 0, "fallback_attempts": 0,
        "request_ledger": "EXISTING_GLOBAL12_APPEND_ONLY"}
    _freeze_state(state, manifest)
    return manifest


def bindings():
    paths = [Path(__file__), Path(budget.__file__), Path(proxy.__file__),
             Path(transport_api.__file__), Path(v2.__file__),
             Path(__file__).with_name("qa_api_budget_v2.py"), PROFILE,
             ARTIFACTS.parent / "base_qa_p0_20261027_v1" / "API_PLAN.json"]
    return qa.implementation_bindings() + [qa.legacy.file_binding(path) for path in paths]


class ExecutorV3(v2.Executor):
    def __init__(self, state_dir, *, mode="dry-run", approval_path=None,
                 capability_path=None, token_counter=None, transport=None):
        if mode not in ("dry-run", "live"):
            raise v2.ExecutionHold("v3 executor supports dry-run or explicitly gated live mode")
        self.state = Path(state_dir).resolve()
        self.mode = mode
        self.manifest = qa.read(self.state / "EXPERIMENT.json")
        self.manifest_sha = qa.legacy.fingerprint(self.manifest)
        self.transport = None
        self.counter = None
        self.ledger = None
        self.capability = None
        self.live_evidence_bindings = []
        self._check_frozen()
        if mode == "dry-run":
            if transport is not None or token_counter is not None:
                raise v2.ExecutionHold("dry-run cannot receive a transport or token counter")
            return
        if not LIVE_EXECUTION_ENABLED:
            raise v2.ExecutionHold("v3 live execution is disabled pending separate capability review")
        if transport is not None or token_counter is not None:
            raise v2.ExecutionHold("live transport and token counting are owned by v3")
        self._authorize(approval_path, capability_path)
        self.ledger = budget.GlobalBudgetAdapter(mode="live", state_dir=self.state)
        try:
            self.counter = v2.LocalTokenizerCounter(qa.read(PROFILE)["tokenizer_directory"])
            self.transport = proxy.ProxyTransportV3(self.ledger, mode="live",
                credential_file=Path(qa.read(PROFILE)["credential_file"]), capability=self.capability)
            self.transport.validate_configuration()
        except Exception:
            self.ledger.close()
            self.ledger = None
            self.transport = None
            raise v2.ExecutionHold("live preflight failed before request reservation") from None

    def _check_frozen(self):
        manifest = self.manifest
        if v2.frozen(self.state, "EXPERIMENT") != manifest:
            raise v2.ExecutionHold("experiment manifest drift")
        if (manifest.get("version") != VERSION or manifest.get("roots") != list(ROOTS) or
                manifest.get("stages") != list(STAGES) or manifest.get("maximum_attempts") != 12 or
                manifest.get("max_output_tokens") != OUTPUT_LIMIT or
                manifest.get("input_limits") != INPUT_LIMITS):
            raise v2.ExecutionHold("v3 execution policy drift")
        if (manifest.get("state_directory") != str(self.state) or
                manifest.get("ledger_path") != str(budget.CANONICAL_LEDGER.resolve()) or
                manifest.get("request_ledger") != "EXISTING_GLOBAL12_APPEND_ONLY"):
            raise v2.ExecutionHold("authoritative ledger path drift")
        if (manifest.get("feedback_enabled") is not False or manifest.get("language") != "en" or
                manifest.get("arm") not in ("F", "G") or manifest.get("retry_attempts") != 0 or
                manifest.get("fallback_attempts") != 0):
            raise v2.ExecutionHold("P0 experiment policy drift")
        if manifest.get("implementation_bindings") != bindings():
            raise v2.ExecutionHold("v3 implementation bindings drift")
        for binding in (manifest.get("selection_binding"), manifest.get("api_plan_binding"),
                        manifest.get("budget_guard_binding")):
            if not isinstance(binding, dict) or qa.legacy.file_binding(binding["path"]) != binding:
                raise v2.ExecutionHold("source plan or budget guard binding drift")
        if manifest["budget_guard_binding"]["path"] != str((budget.CANONICAL_DIR / "BUDGET_TAKEOVER.json").resolve()):
            raise v2.ExecutionHold("unexpected global budget guard")
        for root_id, row in manifest["sources"].items():
            if root_id not in ROOTS:
                raise v2.ExecutionHold("unapproved source root")
            for binding in (row["source_binding"], row["warnings_binding"]):
                if binding and qa.legacy.file_binding(binding["path"]) != binding:
                    raise v2.ExecutionHold("source or warning file drift")
        for binding in getattr(self, "live_evidence_bindings", []):
            if qa.legacy.file_binding(binding["path"]) != binding:
                raise v2.ExecutionHold("approval/capability/evidence drift")

    def _authorize(self, approval_path, capability_path):
        if approval_path is None or capability_path is None:
            raise v2.ExecutionHold("live remains HOLD without explicit approval and verified capabilities")
        approval, capability = qa.read(approval_path), qa.read(capability_path)
        profile = qa.read(PROFILE)
        evidence = capability.get("evidence_binding")
        required_approval = (approval.get("approved") is True and
            approval.get("experiment_sha256") == self.manifest_sha and
            approval.get("max_attempts") == GLOBAL_LIMIT_FOR_AUTH and
            approval.get("max_output_tokens") == OUTPUT_LIMIT and
            approval.get("roots") == list(ROOTS) and
            isinstance(approval.get("approval_reference"), str) and approval["approval_reference"].strip() and
            approval.get("capability_binding") == qa.legacy.file_binding(capability_path))
        if not required_approval:
            raise v2.ExecutionHold("live approval missing, mismatched or incomplete")
        if (capability.get("tokenizer_compatible") is not True or
                not isinstance(evidence, dict) or qa.legacy.file_binding(evidence["path"]) != evidence or
                capability.get("max_tokens_hard_limit_confirmed") is not True or
                capability.get("reasoning_in_output_budget_confirmed") is not True or
                capability.get("max_output_tokens") != OUTPUT_LIMIT or
                capability.get("output_parameter") != "max_tokens" or
                capability.get("endpoint") != transport_api.ENDPOINT or
                capability.get("request_model") != transport_api.MODEL or
                capability.get("response_model") != transport_api.EXPECTED_MODEL or
                type(capability.get("context_window_tokens")) is not int or
                capability["context_window_tokens"] <= OUTPUT_LIMIT):
            raise v2.ExecutionHold("verified tokenizer, 4096 output semantics and context evidence required")
        try:
            transport_api.validate_capability(capability)
        except transport_api.TransportError as error:
            raise v2.ExecutionHold(error.reason) from None
        if profile.get("endpoint") != transport_api.ENDPOINT or profile.get("request_model") != transport_api.MODEL:
            raise v2.ExecutionHold("school model profile drift")
        self.capability = copy.deepcopy(capability)
        self.live_evidence_bindings = [qa.legacy.file_binding(approval_path),
                                       qa.legacy.file_binding(capability_path), evidence]

    def _call(self, root_id, stage, packet, metadata, run):
        self._check_frozen()
        metadata = {**metadata, "packet_sha256": qa.legacy.fingerprint(packet)}
        self._stage_gate(root_id, stage, packet, metadata, run)
        body = transport_api.build_body(packet, metadata)
        transport_api.validate_body(body)
        input_tokens = self.counter(body)
        v2.required(type(input_tokens) is int and input_tokens > 0,
                    "valid local input token count required")
        v2.required(input_tokens <= INPUT_LIMITS[stage], "stage input token budget exceeded")
        v2.required(input_tokens + OUTPUT_LIMIT <= self.capability["context_window_tokens"],
                    "request plus output exceeds verified service context")
        self.transport.validate_configuration()

        v2.write(run / (stage + "_REQUEST.json"), body)
        request_binding = {"body_sha256": qa.legacy.fingerprint(body),
            "packet_sha256": qa.legacy.fingerprint(packet), "input_tokens": input_tokens,
            "max_output_tokens": OUTPUT_LIMIT, "max_tokens": OUTPUT_LIMIT,
            "endpoint": transport_api.ENDPOINT, "model": transport_api.MODEL,
            "root_id": root_id, "manifest_sha256": self.manifest_sha,
            "attempt_label": str(uuid.uuid4()), "stage": stage}
        v2.write(run / (stage + "_REQUEST_BINDING.json"), request_binding)
        try:
            value, receipt, _ = self.ledger.execute_stage(
                root_id, stage, copy.deepcopy(body), self.transport)
        except BaseException as error:
            v2.write(run / (stage + "_ERROR.json"), {
                "error_type": type(error).__name__,
                "reason_code": getattr(error, "reason", "LOCAL_OR_TRANSPORT_FAILURE"),
                "uncertain": bool(getattr(error, "uncertain", True)),
                "attempt_consumed": True, "retry_allowed": False})
            raise v2.ExecutionHold("REQUEST_FAILED_NO_RETRY") from None
        try:
            v2.write(run / (stage + "_TRANSPORT_RECEIPT.json"), receipt)
            v2.write(run / (stage + "_PARSED_RESPONSE.json"), value)
            return value, receipt
        except BaseException as error:
            v2.write(run / (stage + "_ERROR.json"), {
                "error_type": type(error).__name__,
                "reason_code": "RESPONSE_ARTIFACT_WRITE_FAILED",
                "uncertain": True, "attempt_consumed": True,
                "retry_allowed": False})
            raise v2.ExecutionHold("REQUEST_FAILED_NO_RETRY") from None


GLOBAL_LIMIT_FOR_AUTH = budget.GLOBAL_LIMIT
