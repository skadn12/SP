"""Versioned, fail-closed API execution adapter for the frozen P0 v1 pipeline.

Default dry-run never constructs an HTTP transport. Mock and live ledgers cannot
mix. Live requires explicit approval plus verified service capability evidence.
No Solver, training, expansion, legacy authoring, or automatic retry path exists.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import uuid

from . import base_qa_pipeline_v1 as qa
from .qa_api_budget_v2 import Ledger, BudgetError, ROOTS, STAGES, MAX_ATTEMPTS
from . import qa_api_transport_v2 as transport_api


VERSION = "base_qa_executor_v2_20261029_v1"
PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT / "research-protocols/selfplay_week_20261019" / VERSION
PROFILE = ARTIFACTS / "EXECUTION_PROFILE.json"
OUTPUT_LIMIT = 4096
INPUT_LIMITS = {"DISCOVERY": 20000, "AUTHOR": 24000, "BLIND": 22000, "COMPARE": 32768}


class ExecutionHold(RuntimeError):
    pass


def required(condition, reason):
    if not condition:
        raise ExecutionHold(reason)


def bindings():
    paths = [Path(__file__), Path(transport_api.__file__),
             Path(__file__).with_name("qa_api_budget_v2.py"),
             Path(__file__).with_name("question_presentation_gate.py"), PROFILE,
             ARTIFACTS / "CALL_BUDGET_V2.json"]
    return qa.implementation_bindings() + [qa.legacy.file_binding(p) for p in paths]


def write(path, value):
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def freeze(run, name, value):
    write(run / (name + ".json"), value)
    write(run / (name + "_FREEZE.json"), {"sha256": qa.legacy.fingerprint(value), "version": VERSION})


def frozen(run, name):
    return qa._frozen(run, name + ".json", name + "_FREEZE.json")


def initialize(state_dir, selection_path, arm="G"):
    """Create an exclusive experiment directory without credentials or network."""
    required(arm in ("F", "G"), "invalid F/G arm")
    state_dir, selection_path = Path(state_dir).resolve(), Path(selection_path).resolve()
    selection = qa.read(selection_path)
    selected = {}
    for root_id in ROOTS:
        rows = [r for r in selection["roots"] if r["root_id"] == root_id]
        required(len(rows) == 1, "exactly one reviewed source per approved root required")
        row = rows[0]
        binding = qa.legacy.file_binding(row["path"])
        required(row.get("sha256") == binding["sha256"] and row.get("bytes") == binding["bytes"],
                 "selected source binding drift")
        qa.source_from(binding, root_id)
        warnings_path = row.get("warnings_path")
        selected[root_id] = {"source_binding": binding,
            "warnings_binding": qa.legacy.file_binding(warnings_path) if warnings_path else None}
    state_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"version": VERSION, "experiment_id": VERSION,
        "state_directory": str(state_dir), "ledger_path": str(state_dir / "attempts.sqlite3"),
        "roots": list(ROOTS), "stages": list(STAGES), "maximum_attempts": MAX_ATTEMPTS,
        "max_output_tokens": OUTPUT_LIMIT, "input_limits": INPUT_LIMITS,
        "arm": arm, "feedback_enabled": False, "language": "en",
        "selection_binding": qa.legacy.file_binding(selection_path),
        "sources": selected, "implementation_bindings": bindings(),
        "api_plan_binding": qa.legacy.file_binding(ARTIFACTS.parent /
            "base_qa_p0_20261027_v1/API_PLAN.json"),
        "network_authorization": "REQUIRES_SEPARATE_EXPLICIT_APPROVAL_AND_CAPABILITY",
        "retry_attempts": 0, "fallback_attempts": 0}
    freeze(state_dir, "EXPERIMENT", manifest)
    return manifest


class LocalTokenizerCounter:
    """Tokenizer only, local files only; never load model weights or fetch files."""
    def __init__(self, directory):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                                       trust_remote_code=False)

    def __call__(self, body):
        return len(self.tokenizer.apply_chat_template(body["messages"], tokenize=True,
                                                       add_generation_prompt=True))


class Executor:
    def __init__(self, state_dir, *, mode="dry-run", transport=None, token_counter=None,
                 authorization=None, capability=None):
        required(mode in ("dry-run", "mock", "live"), "unknown execution mode")
        self.state = Path(state_dir).resolve()
        self.mode, self.transport, self.counter = mode, transport, token_counter
        self.manifest = frozen(self.state, "EXPERIMENT")
        self.manifest_sha = qa.legacy.fingerprint(self.manifest)
        self._check_frozen()
        self.ledger = None
        self.capability = None
        self.live_evidence_bindings = []
        if mode == "dry-run":
            return
        if mode == "mock":
            required(getattr(transport, "is_mock", False) is True, "mock transport required")
            required(callable(token_counter), "mock input counter required")
        else:
            self._authorize(authorization, capability)
            required(transport is None, "live transport cannot be replaced by a custom caller")
            self.transport = transport_api.SingleAttemptHTTPTransport(
                credential_file=Path(qa.read(PROFILE)["credential_file"]), live=True,
                capability=self.capability, timeout=60)
            try:
                self.counter = LocalTokenizerCounter(qa.read(PROFILE)["tokenizer_directory"])
            except Exception:
                raise ExecutionHold("local-only input tokenizer unavailable: HOLD") from None
        self.ledger = Ledger(self.manifest["ledger_path"], self.manifest["experiment_id"],
                             self.manifest_sha, mode=mode)

    def _check_frozen(self):
        m = self.manifest
        required(frozen(self.state, "EXPERIMENT") == m, "experiment drift")
        required(m["version"] == VERSION and m["roots"] == list(ROOTS) and
                 m["stages"] == list(STAGES), "execution path differs from approved P0 flow")
        required(m["maximum_attempts"] == 12 and m["max_output_tokens"] == 4096 and
                 m["input_limits"] == INPUT_LIMITS, "budget parameters differ")
        required(m["state_directory"] == str(self.state) and
                 m["ledger_path"] == str(self.state / "attempts.sqlite3"), "ledger path drift")
        required(m["feedback_enabled"] is False and m["language"] == "en" and
                 m["arm"] in ("F", "G"), "P0 experiment policy drift")
        required(m["implementation_bindings"] == bindings(), "code/profile/prompt drift")
        for b in [m["selection_binding"], m["api_plan_binding"]]:
            required(qa.legacy.file_binding(b["path"]) == b, "frozen plan/selection drift")
        for row in m["sources"].values():
            for b in (row["source_binding"], row["warnings_binding"]):
                if b:
                    required(qa.legacy.file_binding(b["path"]) == b, "source/warning drift")
        for b in getattr(self, "live_evidence_bindings", []):
            required(qa.legacy.file_binding(b["path"]) == b, "approval/capability/evidence drift")

    def _authorize(self, approval_path, capability_path):
        required(approval_path is not None and capability_path is not None,
                 "live remains HOLD without explicit approval and verified capabilities")
        approval, capability = qa.read(approval_path), qa.read(capability_path)
        profile = qa.read(PROFILE)
        required(str(self.state) == profile["live_state_directory"],
                 "live requires the canonical experiment ledger directory")
        required(approval.get("approved") is True and
                 approval.get("experiment_sha256") == self.manifest_sha and
                 approval.get("max_attempts") == 12 and approval.get("max_output_tokens") == 4096 and
                 approval.get("roots") == list(ROOTS) and qa.text(approval.get("approval_reference")),
                 "live approval missing, mismatched or incomplete")
        required(approval.get("capability_binding") == qa.legacy.file_binding(capability_path),
                 "capability is not bound by approval")
        required(capability.get("tokenizer_compatible") is True,
                 "service/local tokenizer compatibility is unverified")
        evidence = capability.get("evidence_binding")
        required(isinstance(evidence, dict) and qa.legacy.file_binding(evidence["path"]) == evidence,
                 "service output/context capability needs bound evidence")
        required(capability.get("max_tokens_hard_limit_confirmed") is True and
                 capability.get("reasoning_in_output_budget_confirmed") is True and
                 capability.get("max_output_tokens") == 4096 and
                 capability.get("output_parameter") == "max_tokens" and
                 capability.get("endpoint") == transport_api.ENDPOINT and
                 capability.get("request_model") == transport_api.MODEL and
                 capability.get("response_model") == transport_api.EXPECTED_MODEL,
                 "service cannot reliably enforce the requested4096 output budget: HOLD")
        required(isinstance(capability.get("context_window_tokens"), int) and
                 not isinstance(capability["context_window_tokens"], bool) and
                 capability["context_window_tokens"] > 4096,
                 "verified service context limit required")
        self.capability = copy.deepcopy(capability)
        self.capability["evidence_ref"] = evidence["path"]
        # Validate the whole transport contract before opening a ledger or
        # reserving any attempt. This is not a service availability probe.
        transport_api.validate_capability(self.capability)
        self.live_evidence_bindings = [qa.legacy.file_binding(approval_path),
                                      qa.legacy.file_binding(capability_path), evidence]

    def _source(self, root_id):
        row = self.manifest["sources"][root_id]
        source = qa.source_from(row["source_binding"], root_id)
        warnings = qa.read(row["warnings_binding"]["path"]) if row["warnings_binding"] else []
        return source, warnings

    def _discovery_packet(self, source, warnings):
        return {"stage": "DISCOVER_RELATION", "policy": qa.POLICY,
            "instruction": qa.instruction("discovery"), "source": qa.source_view(source),
            "source_warnings": warnings}

    def _author_packet(self, source, warnings, relation):
        return {"stage": "WRITE_NATURAL_QUESTION", "policy": qa.POLICY,
            "instruction": qa.instruction("author"), "source": qa.source_view(source),
            "source_warnings": warnings, "relation": relation,
            "relation_sha256": qa.legacy.fingerprint(relation)}

    def _blind_packet(self, source, warnings, candidate):
        packet = qa.blind_packet(candidate, source)
        # An author-controlled candidate identifier is unnecessary in a blind
        # solve. Use a neutral source-derived identifier, never an answer hash.
        packet["candidate_id"] = "p0-v2-blind-" + source["root_id"]
        packet["source_warnings"] = warnings
        required(set(packet) == {"stage", "language", "instruction", "candidate_id", "source",
            "question_type", "question", "options", "source_warnings"}, "unexpected blind payload fields")
        return packet

    def _comparison_packet(self, run, source, warnings):
        candidate, relation = frozen(run, "CANDIDATE"), frozen(run, "RELATION")
        blind, packet = frozen(run, "BLIND_SOLUTION"), frozen(run, "BLIND_PACKET")
        required(not qa.validate_blind(candidate, source, packet, blind),
                 "invalid/ambiguous/insufficient independent result")
        required(qa.same_answer(candidate["question_type"], blind["independent_gold"], candidate["gold"]),
                 "GOLD_DISAGREEMENT_HOLD_NO_COMPARISON")
        return {"stage": "COMPARE_AFTER_BLIND_FREEZE", "instruction": qa.instruction("comparison"),
            "source": qa.source_view(source), "source_warnings": warnings, "candidate": candidate,
            "relation": relation, "blind_solution": blind, "required_checks": list(qa.CHECKS),
            "necessity_check_mode": "STATIC_COUNTERFACTUAL_REVIEW_NOT_BLIND_ABLATION",
            "removal_contexts": [{"group_id": g["group_id"], "removed_turn_refs": g["turn_refs"],
                "remaining_turn_refs": [t["turn_ref"] for t in source["turns"] if t["turn_ref"] not in g["turn_refs"]],
                "single_group_turn_refs": [t["turn_ref"] for t in source["turns"] if t["turn_ref"] in g["turn_refs"]]}
                for g in relation["necessity_groups"]]}

    def _stage_gate(self, root_id, stage, packet, metadata, run):
        """Recheck the frozen path even when the internal caller is invoked directly."""
        required(self.mode in ("mock", "live") and self.ledger is not None,
                 "dry-run has no callable transport")
        required(root_id in ROOTS and stage in STAGES, "unknown root or stage")
        required(Path(run).resolve() == self.state / "roots" / root_id, "stage directory drift")
        name = {"DISCOVERY": "DISCOVERY_PACKET", "AUTHOR": "AUTHOR_PACKET",
                "BLIND": "BLIND_PACKET", "COMPARE": "COMPARISON_PACKET"}[stage]
        required(packet == frozen(run, name), "request differs from its frozen stage packet")
        source, warnings = self._source(root_id)
        author_id, reviewer_id = "p0-v2-author-" + root_id, "p0-v2-reviewer-" + root_id
        if stage == "DISCOVERY":
            expected, expected_meta = self._discovery_packet(source, warnings), {"author_id": author_id}
        else:
            relation = frozen(run, "RELATION")
            required(relation.get("author_id") == author_id and not qa.validate_relation(relation, source),
                     "invalid frozen relation/proof")
            if stage == "AUTHOR":
                expected = self._author_packet(source, warnings, relation)
                expected_meta = {"author_id": author_id, "relation_sha256": qa.legacy.fingerprint(relation)}
            else:
                candidate = frozen(run, "CANDIDATE")
                required(qa.validate_candidate(candidate, relation, source)["mechanical_pass"],
                         "invalid frozen candidate")
                if stage == "BLIND":
                    expected = self._blind_packet(source, warnings, candidate)
                    expected_meta = {"reviewer_id": reviewer_id, "model_id": transport_api.EXPECTED_MODEL}
                else:
                    # Requires the committed blind freeze and local agreement;
                    # a private _call cannot bypass the orchestration guard.
                    expected = self._comparison_packet(run, source, warnings)
                    blind = frozen(run, "BLIND_SOLUTION")
                    required(blind.get("reviewer_id") == reviewer_id and
                             blind.get("model_id") == transport_api.EXPECTED_MODEL, "blind identity drift")
                    expected_meta = {"reviewer_id": reviewer_id,
                        "blind_sha256": qa.legacy.fingerprint(blind),
                        "candidate_sha256": qa.legacy.fingerprint(candidate)}
        expected_meta["packet_sha256"] = qa.legacy.fingerprint(expected)
        required(packet == expected and metadata == expected_meta, "stage packet or metadata policy drift")

    def _call(self, root_id, stage, packet, metadata, run):
        self._check_frozen()
        metadata = {**metadata, "packet_sha256": qa.legacy.fingerprint(packet)}
        self._stage_gate(root_id, stage, packet, metadata, run)
        body = transport_api.build_body(packet, metadata)
        transport_api.validate_body(body)
        input_tokens = self.counter(body)
        required(isinstance(input_tokens, int) and not isinstance(input_tokens, bool) and input_tokens > 0,
                 "valid local input token count required")
        required(input_tokens <= INPUT_LIMITS[stage], "stage input token budget exceeded")
        if self.mode == "live":
            required(input_tokens + 4096 <= self.capability["context_window_tokens"],
                     "request plus output exceeds verified service context")
        # Every HTTP attempt is preceded by an immutable body and durable reservation.
        write(run / (stage + "_REQUEST.json"), body)
        request_binding = {"body_sha256": qa.legacy.fingerprint(body),
            "packet_sha256": qa.legacy.fingerprint(packet), "input_tokens": input_tokens,
            "max_output_tokens": 4096, "max_tokens": 4096, "endpoint": transport_api.ENDPOINT,
            "model": transport_api.MODEL, "root_id": root_id, "manifest_sha256": self.manifest_sha,
            "local_attempt_id": str(uuid.uuid4()),
            "mode": self.mode, "stage": stage}
        write(run / (stage + "_REQUEST_BINDING.json"), request_binding)
        attempt = self.ledger.reserve(root_id, stage, request_binding)
        try:
            # No loop, retry decorator, SDK, redirects, fallback or repair request.
            envelope = self.transport.send(copy.deepcopy(body))
            write(run / (stage + "_RAW_RESPONSE.json"), envelope)
            value, receipt = transport_api.parse_completion(envelope)
            write(run / (stage + "_TRANSPORT_RECEIPT.json"), receipt)
            write(run / (stage + "_PARSED_RESPONSE.json"), value)
            ledger_receipt = {"response_model": receipt["returned_model"],
                "response_sha256": receipt["raw_response_sha256"],
                "request_id": receipt["request_id"], "http_status": receipt["http_status"],
                "finish_reason": receipt["finish_reason"],
                "usage": {k: receipt["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}}
            self.ledger.settle(attempt, "SUCCESS", ledger_receipt)
            return value, receipt
        except BaseException as exc:
            # KeyboardInterrupt/SystemExit also leave a consumed durable attempt.
            failure = {"error_type": type(exc).__name__, "uncertain": True,
                       "attempt_consumed": True, "retry_allowed": False}
            if isinstance(exc, transport_api.TransportError):
                failure["reason_code"] = exc.reason
            try:
                write(run / (stage + "_ERROR.json"), failure)
            except Exception:
                pass
            try:
                self.ledger.settle(attempt, "UNCERTAIN", {"error_type": type(exc).__name__,
                    "uncertain": True})
            except Exception:
                pass
            raise ExecutionHold("REQUEST_FAILED_NO_RETRY:" + failure.get("reason_code", type(exc).__name__)) from None

    def run_root(self, root_id):
        required(root_id in ROOTS, "root outside approved three-root experiment")
        self._check_frozen()
        if self.mode == "dry-run":
            return {"version": VERSION, "mode": "dry-run", "root_id": root_id,
                    "status": "DRY_RUN_ONLY", "planned_stages": list(STAGES),
                    "model_calls": 0, "ledger_attempts_reserved": 0,
                    "message": "No transport, credentials, token counter or model is invoked."}
        # One durable opportunity across all processes; active/crashed claims cannot retry.
        self.ledger.claim_root(root_id)
        run = self.state / "roots" / root_id
        author_id, reviewer_id = "p0-v2-author-" + root_id, "p0-v2-reviewer-" + root_id
        try:
            run.mkdir(parents=True, exist_ok=False)
            source, warnings = self._source(root_id)
            discovery = self._discovery_packet(source, warnings)
            freeze(run, "DISCOVERY_PACKET", discovery)
            relation, _ = self._call(root_id, "DISCOVERY", discovery,
                {"author_id": author_id}, run)
            if relation.get("status") == "NO_SAFE_QA":
                required(qa.text(relation.get("reason")), "NO_SAFE_QA requires reason")
                freeze(run, "RELATION", relation)
                self.ledger.finish_root(root_id, "NO_SAFE_QA", relation["reason"])
                return {"root_id": root_id, "status": "NO_SAFE_QA", "new_questions": 0}
            required(relation.get("author_id") == author_id, "author identity drift")
            required(not qa.validate_relation(relation, source), "invalid discovered relation/proof")
            freeze(run, "RELATION", relation)
            relation = frozen(run, "RELATION")
            author_packet = self._author_packet(source, warnings, relation)
            freeze(run, "AUTHOR_PACKET", author_packet)
            candidate, _ = self._call(root_id, "AUTHOR", author_packet,
                {"author_id": author_id, "relation_sha256": qa.legacy.fingerprint(relation)}, run)
            audit = qa.validate_candidate(candidate, relation, source)
            required(audit["mechanical_pass"], "invalid candidate: HOLD without author repair")
            freeze(run, "CANDIDATE", candidate)
            write(run / "MECHANICAL_AUDIT.json", audit)
            candidate = frozen(run, "CANDIDATE")
            packet = self._blind_packet(source, warnings, candidate)
            freeze(run, "BLIND_PACKET", packet)
            blind, receipt = self._call(root_id, "BLIND", packet,
                {"packet_sha256": qa.legacy.fingerprint(packet), "reviewer_id": reviewer_id,
                 "model_id": transport_api.EXPECTED_MODEL}, run)
            # Preserve the raw model answer. Only model identity is transport-owned.
            required(blind.get("reviewer_id") == reviewer_id, "reviewer identity drift")
            required(blind.get("model_id") in (None, receipt["returned_model"]),
                     "reviewer model self-report contradicts transport")
            blind["model_id"] = receipt["returned_model"]
            freeze(run, "BLIND_SOLUTION", blind)
            blind = frozen(run, "BLIND_SOLUTION")
            errors = qa.validate_blind(candidate, source, packet, blind)
            required(not errors, "invalid/ambiguous/insufficient independent result")
            required(qa.same_answer(candidate["question_type"], blind["independent_gold"], candidate["gold"]),
                     "GOLD_DISAGREEMENT_HOLD_NO_COMPARISON")
            # Do not even construct comparison material before these freeze/validity checks.
            comparison_packet = self._comparison_packet(run, source, warnings)
            freeze(run, "COMPARISON_PACKET", comparison_packet)
            comparison, _ = self._call(root_id, "COMPARE", comparison_packet,
                {"reviewer_id": reviewer_id, "blind_sha256": qa.legacy.fingerprint(blind),
                 "candidate_sha256": qa.legacy.fingerprint(candidate)}, run)
            freeze(run, "COMPARISON_REVIEW", comparison)
            result = qa.validate_review(candidate, relation, source, packet, blind, comparison)
            freeze(run, "DECISION", result)
            if result["admitted"]:
                freeze(run, "FROZEN_QA", {"candidate": candidate, "relation": relation,
                    "blind_solution_sha256": qa.legacy.fingerprint(blind), "decision": result})
            self.ledger.finish_root(root_id, result["decision"], ";".join(result["errors"]))
            return {"root_id": root_id, "status": result["decision"], "new_questions": int(result["admitted"]),
                    "empirical_quality_established": False}
        except BaseException as exc:
            reason = str(exc) if isinstance(exc, ExecutionHold) else "LOCAL_FAILURE:" + type(exc).__name__
            self.ledger.stop_root(root_id, reason)
            if run.is_dir() and not (run / "ROOT_HOLD.json").exists():
                write(run / "ROOT_HOLD.json", {"status": "HOLD", "reason": reason,
                    "retry_allowed": False, "new_candidate_opportunity_allowed": False})
            return {"root_id": root_id, "status": "HOLD", "reason": reason, "new_questions": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--arm", choices=("F", "G"), default="G")
    p = sub.add_parser("run")
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--root", choices=ROOTS, required=True)
    p.add_argument("--live", action="store_true", help="Still requires separate approval/capability evidence")
    p.add_argument("--authorization", type=Path)
    p.add_argument("--capability", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = initialize(args.state_dir, args.selection, args.arm)
            print(json.dumps({"status": "INITIALIZED_OFFLINE", "experiment_id": result["experiment_id"], "model_calls": 0}))
        else:
            executor = Executor(args.state_dir, mode="live" if args.live else "dry-run",
                                authorization=args.authorization, capability=args.capability)
            result = executor.run_root(args.root)
            print(json.dumps(result, ensure_ascii=False))
            return 2 if result["status"] == "HOLD" else 0
    except (ExecutionHold, BudgetError, transport_api.TransportError, ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "HOLD", "error_type": type(exc).__name__, "model_calls_started_by_this_error_path": 0}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
