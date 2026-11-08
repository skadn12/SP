"""V4 candidate executor: one frozen root, four ordered stages, live disabled."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import uuid

from . import base_qa_executor_v2 as v2
from . import base_qa_global_budget_adapter_v4 as budget
from . import base_qa_global_budget_adapter_v3 as budget_v3
from . import base_qa_pipeline_v1 as qa
from . import base_qa_proxy_transport_v3 as proxy
from . import qa_api_transport_v2 as transport_api

VERSION = budget.VERSION
ROOT_ID = budget.ROOT_ID
ROOTS = budget.ROOTS
STAGES = budget.STAGES
GLOBAL_LIMIT = budget.GLOBAL_LIMIT
ROOT_ATTEMPT_LIMIT = budget.ROOT_ATTEMPT_LIMIT
OUTPUT_LIMIT = budget.MAX_OUTPUT
INPUT_LIMITS = dict(v2.INPUT_LIMITS)
PROFILE = v2.PROFILE
ARTIFACTS = budget.ARTIFACTS
API_PLAN = ARTIFACTS.parent / "base_qa_p0_20261027_v1" / "API_PLAN.json"
DEFAULT_SELECTION = (ARTIFACTS.parent / "natural_question_forms_fg_15root_extension_20261026_v1"
                     / "SOURCE_SELECTION_FINAL.json")
LIVE_EXECUTION_ENABLED = False


def _baseline_digest(rows):
    return qa.legacy.fingerprint(rows)


def initialize(state_dir, selection_path=DEFAULT_SELECTION, arm="G"):
    if arm not in ("F", "G"):
        raise v2.ExecutionHold("invalid F/G arm")
    state = Path(state_dir).resolve()
    selection_path = Path(selection_path).resolve()
    selection = qa.read(selection_path)
    rows = [row for row in selection.get("roots", []) if row.get("root_id") == ROOT_ID]
    if len(rows) != 1:
        raise v2.ExecutionHold("V4 requires exactly one reviewed 109164 source")
    selected = rows[0]
    source_binding = qa.legacy.file_binding(selected["path"])
    if selected.get("sha256") != source_binding["sha256"] or selected.get("bytes") != source_binding["bytes"]:
        raise v2.ExecutionHold("selected V4 source binding drift")
    qa.source_from(source_binding, ROOT_ID)
    warnings_path = selected.get("warnings_path")
    warnings_binding = qa.legacy.file_binding(warnings_path) if warnings_path else None
    with budget.GlobalBudgetAdapterV4(mode="inspect") as ledger:
        snapshot = ledger.snapshot()
    if snapshot["used"] != 4 or snapshot["remaining"] != 8 or snapshot["v4_used"] != 0:
        raise v2.ExecutionHold("V4 initialization requires canonical global ledger at 4/12 with no V4 attempts")
    baseline = [budget.baseline_row(row) for row in snapshot["attempts"]]
    if any(row["purpose"].startswith(budget.NAMESPACE) for row in snapshot["attempts"]):
        raise v2.ExecutionHold("V4 namespace already present in the starting ledger")
    takeover = budget.CANONICAL_DIR / "BUDGET_TAKEOVER.json"
    manifest = {
        "version": VERSION, "experiment_id": VERSION, "state_directory": str(state),
        "ledger_path": str(budget.CANONICAL_LEDGER.resolve()),
        "budget_guard_binding": qa.legacy.file_binding(takeover),
        "roots": [ROOT_ID], "stages": list(STAGES),
        "root_attempt_limit": ROOT_ATTEMPT_LIMIT, "maximum_attempts": ROOT_ATTEMPT_LIMIT,
        "global_limit": GLOBAL_LIMIT, "starting_global_used": snapshot["used"],
        "starting_global_remaining": snapshot["remaining"],
        "baseline_attempts": baseline, "baseline_digest": _baseline_digest(baseline),
        "max_output_tokens": OUTPUT_LIMIT, "input_limits": INPUT_LIMITS,
        "arm": arm, "feedback_enabled": False, "language": "en",
        "candidate_opportunities_per_root": 1, "selection_binding": qa.legacy.file_binding(selection_path),
        "sources": {ROOT_ID: {"source_binding": source_binding, "warnings_binding": warnings_binding}},
        "implementation_bindings": bindings(), "api_plan_binding": qa.legacy.file_binding(API_PLAN),
        "network_authorization": "REQUIRES_SEPARATE_APPROVAL_AND_VERIFIED_SERVICE_CAPACITY",
        "retry_attempts": 0, "fallback_attempts": 0,
        "request_ledger": "EXISTING_GLOBAL12_APPEND_ONLY",
    }
    state.mkdir(parents=True, exist_ok=False)
    v2.freeze(state, "EXPERIMENT", manifest)
    return manifest


def bindings():
    paths = [Path(__file__), Path(budget.__file__), Path(budget_v3.__file__),
             Path(proxy.__file__), Path(transport_api.__file__), Path(v2.__file__),
             Path(__file__).with_name("base_qa_context_preflight_v4.py"),
             Path(__file__).with_name("qa_api_budget_v2.py"), PROFILE, API_PLAN]
    return qa.implementation_bindings() + [qa.legacy.file_binding(path) for path in paths]


class ExecutorV4(v2.Executor):
    def __init__(self, state_dir, *, mode="dry-run", approval_path=None, capability_path=None,
                 token_counter=None, transport=None, ledger=None, context_window_tokens=None):
        if mode not in ("dry-run", "mock", "live"):
            raise v2.ExecutionHold("V4 supports dry-run, isolated mock or gated live mode")
        self.state = Path(state_dir).resolve()
        self.mode = mode
        self.manifest = qa.read(self.state / "EXPERIMENT.json")
        self.manifest_sha = qa.legacy.fingerprint(self.manifest)
        self.transport = None
        self.counter = None
        self.ledger = None
        self.capability = None
        self.live_evidence_bindings = []
        self.context_window_tokens = context_window_tokens
        self._check_frozen()
        if mode == "dry-run":
            if any(value is not None for value in (token_counter, transport, ledger, context_window_tokens)):
                raise v2.ExecutionHold("dry-run cannot receive request dependencies")
            return
        if mode == "mock":
            if not isinstance(ledger, budget.GlobalBudgetAdapterV4) or ledger.mode != "mock":
                raise v2.ExecutionHold("mock mode requires an isolated V4 budget fixture")
            if transport is None or not callable(token_counter):
                raise v2.ExecutionHold("mock mode requires explicit offline transport and counter")
            if ledger.state_dir != self.state:
                raise v2.ExecutionHold("mock ledger state must bind the experiment directory")
            self.ledger, self.transport, self.counter = ledger, transport, token_counter
            return
        if not LIVE_EXECUTION_ENABLED:
            raise v2.ExecutionHold("V4 live execution is disabled pending verified context and output semantics")
        self._authorize(approval_path, capability_path)
        self.ledger = budget.GlobalBudgetAdapterV4(
            mode="live", state_dir=self.state, baseline_attempts=self.manifest["baseline_attempts"])
        try:
            self.counter = v2.LocalTokenizerCounter(qa.read(PROFILE)["tokenizer_directory"])
            self.transport = proxy.ProxyTransportV3(
                self.ledger, mode="live", credential_file=Path(qa.read(PROFILE)["credential_file"]),
                capability=self.capability)
            self.transport.validate_configuration()
        except Exception:
            self.ledger.close()
            self.ledger = None
            self.transport = None
            raise v2.ExecutionHold("V4 live preflight failed before request reservation") from None

    def _check_frozen(self):
        manifest = self.manifest
        if v2.frozen(self.state, "EXPERIMENT") != manifest:
            raise v2.ExecutionHold("V4 manifest drift")
        if (manifest.get("version") != VERSION or manifest.get("roots") != [ROOT_ID] or
                manifest.get("stages") != list(STAGES) or
                manifest.get("root_attempt_limit") != 4 or manifest.get("maximum_attempts") != 4 or
                manifest.get("global_limit") != 12 or manifest.get("starting_global_used") != 4 or
                manifest.get("starting_global_remaining") != 8 or
                manifest.get("max_output_tokens") != OUTPUT_LIMIT or
                manifest.get("input_limits") != INPUT_LIMITS):
            raise v2.ExecutionHold("V4 single-root budget policy drift")
        if (manifest.get("state_directory") != str(self.state) or
                manifest.get("ledger_path") != str(budget.CANONICAL_LEDGER.resolve()) or
                manifest.get("request_ledger") != "EXISTING_GLOBAL12_APPEND_ONLY"):
            raise v2.ExecutionHold("V4 authoritative ledger binding drift")
        if (manifest.get("feedback_enabled") is not False or manifest.get("language") != "en" or
                manifest.get("retry_attempts") != 0 or manifest.get("fallback_attempts") != 0 or
                manifest.get("candidate_opportunities_per_root") != 1):
            raise v2.ExecutionHold("V4 execution policy drift")
        if set(manifest.get("sources", {})) != {ROOT_ID}:
            raise v2.ExecutionHold("V4 source scope must contain exactly one root")
        if manifest.get("implementation_bindings") != bindings():
            raise v2.ExecutionHold("V4 implementation or prompt binding drift")
        for key in ("selection_binding", "api_plan_binding", "budget_guard_binding"):
            binding = manifest.get(key)
            if not isinstance(binding, dict) or qa.legacy.file_binding(binding["path"]) != binding:
                raise v2.ExecutionHold("V4 plan or ledger guard binding drift")
        source = manifest["sources"][ROOT_ID]
        for binding in (source.get("source_binding"), source.get("warnings_binding")):
            if binding and qa.legacy.file_binding(binding["path"]) != binding:
                raise v2.ExecutionHold("V4 source or warnings binding drift")
        if manifest.get("baseline_digest") != _baseline_digest(manifest.get("baseline_attempts", [])):
            raise v2.ExecutionHold("V4 starting ledger baseline drift")
        for binding in getattr(self, "live_evidence_bindings", []):
            if qa.legacy.file_binding(binding["path"]) != binding:
                raise v2.ExecutionHold("V4 approval or capability evidence drift")

    def _authorize(self, approval_path, capability_path):
        if approval_path is None or capability_path is None:
            raise v2.ExecutionHold("V4 live requires separate approval and service evidence")
        approval, capability = qa.read(approval_path), qa.read(capability_path)
        evidence = capability.get("evidence_binding")
        if not (approval.get("approved") is True and
                approval.get("experiment_sha256") == self.manifest_sha and
                approval.get("roots") == [ROOT_ID] and
                approval.get("max_attempts") == 4 and approval.get("global_max_attempts") == 12 and
                approval.get("max_output_tokens") == OUTPUT_LIMIT and
                approval.get("capability_binding") == qa.legacy.file_binding(capability_path) and
                isinstance(approval.get("approval_reference"), str) and approval["approval_reference"].strip()):
            raise v2.ExecutionHold("V4 approval must bind only 109164 and at most four requests")
        if (capability.get("tokenizer_compatible") is not True or
                capability.get("max_tokens_hard_limit_confirmed") is not True or
                capability.get("reasoning_in_output_budget_confirmed") is not True or
                capability.get("output_parameter") != "max_tokens" or
                capability.get("max_output_tokens") != OUTPUT_LIMIT or
                capability.get("endpoint") != transport_api.ENDPOINT or
                capability.get("request_model") != transport_api.MODEL or
                capability.get("response_model") != transport_api.EXPECTED_MODEL or
                type(capability.get("context_window_tokens")) is not int or
                capability["context_window_tokens"] <= max(INPUT_LIMITS.values()) + OUTPUT_LIMIT or
                not isinstance(evidence, dict) or qa.legacy.file_binding(evidence["path"]) != evidence):
            raise v2.ExecutionHold("verified service tokenizer, output and context evidence required")
        try:
            transport_api.validate_capability(capability)
        except transport_api.TransportError as error:
            raise v2.ExecutionHold(error.reason) from None
        self.capability = copy.deepcopy(capability)
        self.context_window_tokens = capability["context_window_tokens"]
        self.live_evidence_bindings = [qa.legacy.file_binding(approval_path),
                                       qa.legacy.file_binding(capability_path), evidence]

    def _source(self, root_id):
        if root_id != ROOT_ID:
            raise v2.ExecutionHold("V4 permits only cc-train-episode-109164")
        row = self.manifest["sources"][ROOT_ID]
        source = qa.source_from(row["source_binding"], root_id)
        warnings = qa.read(row["warnings_binding"]["path"]) if row["warnings_binding"] else []
        return source, warnings

    def _comparison_packet(self, run, source, warnings):
        candidate, relation = v2.frozen(run, "CANDIDATE"), v2.frozen(run, "RELATION")
        blind, packet = v2.frozen(run, "BLIND_SOLUTION"), v2.frozen(run, "BLIND_PACKET")
        v2.required(not qa.validate_blind(candidate, source, packet, blind),
                    "invalid blind response cannot be compared")
        v2.required(qa.same_answer(candidate["question_type"], blind["independent_gold"], candidate["gold"]),
                    "Gold disagreement cannot be compared")
        return {"stage": "COMPARE_AFTER_BLIND_FREEZE", "instruction": qa.instruction("comparison"),
            "source": qa.source_view(source), "source_warnings": warnings, "candidate": candidate,
            "relation": relation, "blind_solution": blind, "required_checks": list(qa.CHECKS),
            "necessity_check_mode": "STATIC_COUNTERFACTUAL_REVIEW_NOT_BLIND_ABLATION",
            "removal_contexts": [{"group_id": group["group_id"], "removed_turn_refs": group["turn_refs"],
                "remaining_turn_refs": [turn["turn_ref"] for turn in source["turns"]
                    if turn["turn_ref"] not in group["turn_refs"]],
                "single_group_turn_refs": [turn["turn_ref"] for turn in source["turns"]
                    if turn["turn_ref"] in group["turn_refs"]]}
                for group in relation["necessity_groups"]]}

    def _stage_gate(self, root_id, stage, packet, metadata, run):
        if root_id != ROOT_ID or stage not in STAGES:
            raise v2.ExecutionHold("V4 root or stage outside the frozen single-root experiment")
        v2.required(self.mode in ("mock", "live") and self.ledger is not None,
                    "V4 stage execution requires an attached budget adapter")
        v2.required(Path(run).resolve() == self.state / "roots" / ROOT_ID,
                    "V4 stage directory drift")
        names = {"DISCOVERY": "DISCOVERY_PACKET", "AUTHOR": "AUTHOR_PACKET",
                 "BLIND": "BLIND_PACKET", "COMPARE": "COMPARISON_PACKET"}
        v2.required(packet == v2.frozen(run, names[stage]), "V4 packet differs from frozen stage packet")
        source, warnings = self._source(ROOT_ID)
        author_id, reviewer_id = "p0-v4-author-" + ROOT_ID, "p0-v4-reviewer-" + ROOT_ID
        if stage == "DISCOVERY":
            expected, expected_meta = self._discovery_packet(source, warnings), {"author_id": author_id}
        else:
            relation = v2.frozen(run, "RELATION")
            v2.required(relation.get("author_id") == author_id and
                        not qa.validate_relation(relation, source), "invalid frozen V4 relation")
            if stage == "AUTHOR":
                expected = self._author_packet(source, warnings, relation)
                expected_meta = {"author_id": author_id, "relation_sha256": qa.legacy.fingerprint(relation)}
            else:
                candidate = v2.frozen(run, "CANDIDATE")
                v2.required(qa.validate_candidate(candidate, relation, source)["mechanical_pass"],
                            "invalid frozen V4 candidate")
                if stage == "BLIND":
                    expected = self._blind_packet(source, warnings, candidate)
                    expected_meta = {"reviewer_id": reviewer_id, "model_id": transport_api.EXPECTED_MODEL}
                else:
                    expected = self._comparison_packet(run, source, warnings)
                    blind = v2.frozen(run, "BLIND_SOLUTION")
                    v2.required(blind.get("reviewer_id") == reviewer_id and
                                blind.get("model_id") == transport_api.EXPECTED_MODEL,
                                "V4 blind identity drift")
                    expected_meta = {"reviewer_id": reviewer_id,
                        "blind_sha256": qa.legacy.fingerprint(blind),
                        "candidate_sha256": qa.legacy.fingerprint(candidate)}
        expected_meta["packet_sha256"] = qa.legacy.fingerprint(expected)
        v2.required(packet == expected and metadata == expected_meta,
                    "V4 stage packet or metadata policy drift")

    def _call(self, root_id, stage, packet, metadata, run):
        self._check_frozen()
        if root_id != ROOT_ID or stage not in STAGES:
            raise v2.ExecutionHold("V4 request scope violation")
        metadata = {**metadata, "packet_sha256": qa.legacy.fingerprint(packet)}
        self._stage_gate(root_id, stage, packet, metadata, run)
        body = transport_api.build_body(packet, metadata)
        transport_api.validate_body(body)
        input_tokens = self.counter(body)
        v2.required(type(input_tokens) is int and input_tokens > 0,
                    "V4 requires a positive local request token count")
        v2.required(input_tokens <= INPUT_LIMITS[stage], "V4 stage input token limit exceeded")
        if self.mode == "live":
            v2.required(type(self.context_window_tokens) is int and
                        input_tokens + OUTPUT_LIMIT <= self.context_window_tokens,
                        "request plus output exceeds verified service context")
        elif self.context_window_tokens is not None:
            v2.required(type(self.context_window_tokens) is int and
                        input_tokens + OUTPUT_LIMIT <= self.context_window_tokens,
                        "mock request exceeds configured context limit")
        self.transport.validate_configuration() if hasattr(self.transport, "validate_configuration") else None
        v2.write(run / (stage + "_REQUEST.json"), body)
        binding = {"body_sha256": qa.legacy.fingerprint(body),
            "packet_sha256": qa.legacy.fingerprint(packet), "input_tokens_local": input_tokens,
            "max_tokens": OUTPUT_LIMIT, "max_output_tokens": OUTPUT_LIMIT,
            "endpoint": transport_api.ENDPOINT, "model": transport_api.MODEL,
            "root_id": ROOT_ID, "manifest_sha256": self.manifest_sha, "stage": stage,
            "retry_attempts": 0}
        v2.write(run / (stage + "_REQUEST_BINDING.json"), binding)
        try:
            value, receipt, _ = self.ledger.execute_stage(root_id, stage, copy.deepcopy(body), self.transport)
        except BaseException as error:
            v2.write(run / (stage + "_ERROR.json"), {
                "error_type": type(error).__name__, "reason_code": getattr(error, "reason", "REQUEST_FAILED"),
                "uncertain": bool(getattr(error, "uncertain", True)), "attempt_consumed": True,
                "retry_allowed": False})
            raise v2.ExecutionHold("V4 request failed; no retry") from None
        try:
            v2.write(run / (stage + "_TRANSPORT_RECEIPT.json"), receipt)
            v2.write(run / (stage + "_PARSED_RESPONSE.json"), value)
            return value, receipt
        except BaseException:
            v2.write(run / (stage + "_ERROR.json"), {
                "reason_code": "RESPONSE_ARTIFACT_WRITE_FAILED", "uncertain": True,
                "attempt_consumed": True, "retry_allowed": False})
            raise v2.ExecutionHold("V4 response freeze failed; no retry") from None

    def run_root(self, root_id):
        if root_id != ROOT_ID:
            raise v2.ExecutionHold("V4 permits only cc-train-episode-109164")
        self._check_frozen()
        if self.mode == "dry-run":
            return {"version": VERSION, "mode": "dry-run", "root_id": ROOT_ID,
                    "status": "DRY_RUN_ONLY", "planned_stages": list(STAGES),
                    "max_attempts": ROOT_ATTEMPT_LIMIT, "model_calls": 0}
        try:
            self.ledger.claim_root(ROOT_ID)
        except Exception as error:
            return {"root_id": ROOT_ID, "status": "HOLD", "reason": type(error).__name__,
                    "model_calls": 0, "retry_allowed": False}
        run = self.state / "roots" / ROOT_ID
        author_id, reviewer_id = "p0-v4-author-" + ROOT_ID, "p0-v4-reviewer-" + ROOT_ID
        try:
            run.mkdir(parents=True, exist_ok=False)
            source, warnings = self._source(ROOT_ID)
            discovery = self._discovery_packet(source, warnings)
            v2.freeze(run, "DISCOVERY_PACKET", discovery)
            relation, _ = self._call(ROOT_ID, "DISCOVERY", discovery, {"author_id": author_id}, run)
            if relation.get("status") == "NO_SAFE_QA":
                v2.required(qa.text(relation.get("reason")), "NO_SAFE_QA requires a reason")
                v2.freeze(run, "RELATION", relation)
                self.ledger.finish_root(ROOT_ID, "NO_SAFE_QA", relation["reason"])
                return {"root_id": ROOT_ID, "status": "NO_SAFE_QA", "new_questions": 0}
            v2.required(relation.get("author_id") == author_id and
                        not qa.validate_relation(relation, source), "invalid V4 relation; stop")
            v2.freeze(run, "RELATION", relation)
            relation = v2.frozen(run, "RELATION")
            author_packet = self._author_packet(source, warnings, relation)
            v2.freeze(run, "AUTHOR_PACKET", author_packet)
            candidate, _ = self._call(ROOT_ID, "AUTHOR", author_packet,
                {"author_id": author_id, "relation_sha256": qa.legacy.fingerprint(relation)}, run)
            audit = qa.validate_candidate(candidate, relation, source)
            v2.required(audit["mechanical_pass"], "invalid V4 candidate; no repair call")
            v2.freeze(run, "CANDIDATE", candidate)
            v2.write(run / "MECHANICAL_AUDIT.json", audit)
            candidate = v2.frozen(run, "CANDIDATE")
            blind_packet = self._blind_packet(source, warnings, candidate)
            v2.freeze(run, "BLIND_PACKET", blind_packet)
            blind, receipt = self._call(ROOT_ID, "BLIND", blind_packet,
                {"reviewer_id": reviewer_id, "model_id": transport_api.EXPECTED_MODEL}, run)
            v2.required(blind.get("reviewer_id") == reviewer_id and
                        blind.get("model_id") in (None, receipt["returned_model"]),
                        "V4 reviewer identity mismatch")
            blind["model_id"] = receipt["returned_model"]
            v2.freeze(run, "BLIND_SOLUTION", blind)
            blind = v2.frozen(run, "BLIND_SOLUTION")
            errors = qa.validate_blind(candidate, source, blind_packet, blind)
            v2.required(not errors, "invalid or ambiguous V4 blind response")
            v2.required(qa.same_answer(candidate["question_type"], blind["independent_gold"], candidate["gold"]),
                        "V4 Gold disagreement; stop before comparison")
            comparison_packet = self._comparison_packet(run, source, warnings)
            v2.freeze(run, "COMPARISON_PACKET", comparison_packet)
            comparison, _ = self._call(ROOT_ID, "COMPARE", comparison_packet,
                {"reviewer_id": reviewer_id, "blind_sha256": qa.legacy.fingerprint(blind),
                 "candidate_sha256": qa.legacy.fingerprint(candidate)}, run)
            v2.freeze(run, "COMPARISON_REVIEW", comparison)
            result = qa.validate_review(candidate, relation, source, blind_packet, blind, comparison)
            v2.freeze(run, "DECISION", result)
            if result["admitted"]:
                v2.freeze(run, "FROZEN_QA", {"candidate": candidate, "relation": relation,
                    "blind_solution_sha256": qa.legacy.fingerprint(blind), "decision": result})
            self.ledger.finish_root(ROOT_ID, result["decision"], ";".join(result["errors"]))
            return {"root_id": ROOT_ID, "status": result["decision"],
                    "new_questions": int(result["admitted"]), "empirical_quality_established": False}
        except BaseException as error:
            reason = str(error) if isinstance(error, v2.ExecutionHold) else "LOCAL_FAILURE:" + type(error).__name__
            try:
                self.ledger.finish_root(ROOT_ID, "HOLD", reason)
            except Exception:
                pass
            if run.is_dir() and not (run / "ROOT_HOLD.json").exists():
                v2.write(run / "ROOT_HOLD.json", {"status": "HOLD", "reason": reason,
                    "retry_allowed": False, "new_candidate_opportunity_allowed": False})
            return {"root_id": ROOT_ID, "status": "HOLD", "reason": reason,
                    "new_questions": 0, "retry_allowed": False}
