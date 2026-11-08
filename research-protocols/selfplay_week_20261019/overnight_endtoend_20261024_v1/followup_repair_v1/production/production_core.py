"""Shared, fail-closed source production primitives for a NEW protocol version.

No model calls or filesystem writes occur at import. Frozen OET1 data is input
only. Quotes are checked against source text / assigned native utterances only.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading

SCENES = [f"OET1:{route}:S{i:02}" for route in "AB" for i in range(1, 4)]
CHECKS = ["time", "location", "actor_identity", "plan_vs_completion",
          "cross_scene_continuity", "semantic_repetition", "naturalness", "event_fidelity"]
LOCK = threading.RLock()


class ProductionError(ValueError):
    def __init__(self, code, **details):
        self.code, self.details = code, details
        super().__init__(code + ": " + json.dumps(details, ensure_ascii=False, sort_keys=True))


def require(condition, code, **details):
    if not condition:
        raise ProductionError(code, **details)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def verify(binding):
    require(bind(binding["path"]) == {k: binding[k] for k in ("path", "sha256", "bytes")},
            "BINDING_CHANGED", path=binding["path"])
    return read(binding["path"])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return bind(path)


def safe_log(path, event_type, *, emit=True, **metadata):
    """One canonical timestamp; caller fields cannot collide with logger fields."""
    row = dict(metadata)
    if "at_utc" in row:
        row["caller_at_utc"] = row.pop("at_utc")
    if "kind" in row:
        row["caller_kind"] = row.pop("kind")
    row.update(at_utc=now(), kind=event_type)
    line = json.dumps(row, ensure_ascii=False)
    with LOCK:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if emit:
            print(line, flush=True)
    return row


def validate_zero_change(value, *, schema="count", field="question_admission_changes"):
    """Explicit schemas avoid treating integer 0 as the Boolean singleton False."""
    if schema == "count":
        require(type(value) is int and value >= 0, "INVALID_NONNEGATIVE_COUNT", field=field)
        require(value == 0, "NONZERO_CHANGE_COUNT", field=field, actual=value)
    elif schema == "boolean":
        require(type(value) is bool, "INVALID_BOOLEAN", field=field)
        require(value is False, "CHANGE_FLAG_TRUE", field=field)
    else:
        raise ProductionError("UNKNOWN_FIELD_SCHEMA", schema=schema)
    return True


def make_turns(response, packet, previous, scene):
    require(response.get("all_source_turns_read") is True, "AUTHOR_FULL_SOURCE_READ_REQUIRED")
    count = response.get("source_turn_count")
    require(type(count) is int and count == len(previous["turns"]), "AUTHOR_SOURCE_COUNT_MISMATCH",
            expected=len(previous["turns"]), reported=count)
    if SCENES.index(scene) >= 3:
        require(response.get("native_record_read") is True, "AUTHOR_NATIVE_READ_REQUIRED")
    turns = response.get("turns")
    require(isinstance(turns, list) and len(turns) == 14, "DRAFT_TURN_COUNT_MISMATCH")
    result = []
    for index, turn in enumerate(turns, 1):
        require(turn.get("speaker") in packet["speakers"], "DRAFT_UNKNOWN_SPEAKER", index=index)
        require(isinstance(turn.get("text"), str) and bool(turn["text"].strip()), "DRAFT_EMPTY_TEXT", index=index)
        result.append({"turn_ref": f"{scene}:T{index:03}", "scene_ref": scene,
                       "session_ref": scene, "speaker": turn["speaker"], "text": turn["text"]})
    return result


def build_cumulative(packet, previous, draft, scene, previous_binding, draft_binding):
    """Compute all namespace counts from exact arrays before first presentation."""
    require(scene in SCENES, "UNKNOWN_SCENE", scene=scene)
    index = SCENES.index(scene)
    prior, append = previous["turns"], draft["turns"]
    initial = packet["full_initial_source"]["turns"]
    require(len(prior) == len(initial) + 14 * index, "PREDECESSOR_STAGE_COUNT_MISMATCH",
            expected=len(initial) + 14 * index, actual=len(prior))
    require(prior[:len(initial)] == initial, "INITIAL_SOURCE_PREFIX_CHANGED")
    require(draft.get("root_id") == packet["root_id"] and draft.get("scene_ref") == scene,
            "DRAFT_SCOPE_MISMATCH")
    require(draft.get("input_source_binding") == previous_binding, "DRAFT_PREDECESSOR_BINDING_CHANGED")
    require(len(append) == 14, "DRAFT_TURN_COUNT_MISMATCH")
    if index >= 3:
        require(draft.get("native_event_binding") == packet["native_events"][index - 3]["native_event_binding"],
                "DRAFT_NATIVE_BINDING_CHANGED")
    for n, turn in enumerate(append, 1):
        require(turn.get("turn_ref") == f"{scene}:T{n:03}" and turn.get("scene_ref") == scene,
                "DRAFT_TURN_METADATA_INVALID", index=n)
        require(turn.get("speaker") in packet["speakers"] and isinstance(turn.get("text"), str)
                and bool(turn["text"].strip()), "DRAFT_TURN_CONTENT_INVALID", index=n)
    refs = [turn["turn_ref"] for turn in prior + append]
    require(len(refs) == len(set(refs)), "DUPLICATE_SOURCE_REFS")
    all_turns = deepcopy(prior + append)
    source = {**deepcopy(previous), "schema_version": "oet1_followup_cumulative_source_v1",
              "root_id": packet["root_id"], "condition": packet["condition"], "namespace": "OET1",
              "scene_ref": scene, "input_source_binding": previous_binding, "draft_binding": draft_binding,
              "turns": all_turns, "previous_turn_count": len(prior), "new_turn_count": len(append),
              "oet1_initial_turn_count": len(initial), "oet1_added_turn_count": len(all_turns) - len(initial),
              "oet1_total_turn_count": len(all_turns),
              "oet1_initial_historical_extension_turn_count": packet["full_initial_source"].get("cumulative_new_turn_count"),
              "mechanical_metadata_version": "followup_v1_exact_arrays_before_review"}
    baseline = previous.get("baseline_turn_count")
    if type(baseline) is int:
        source["cumulative_new_turn_count"] = len(all_turns) - baseline
    proof = {"status": "PASS_MECHANICAL_PREFIX_ONLY", "semantic_admission_inferred": False,
             "exact_previous_turn_prefix": all_turns[:len(prior)] == prior,
             "exact_draft_turn_suffix": all_turns[len(prior):] == append,
             "previous_turn_count": len(prior), "new_turn_count": len(append),
             "total_turn_count": len(all_turns), "initial_turn_count": len(initial),
             "oet1_added_turn_count": len(all_turns) - len(initial),
             "previous_binding": previous_binding, "draft_binding": draft_binding}
    return source, proof


class EvidenceResolver:
    """Only this scene's hash-bound assigned native record can provide evidence."""
    ARRAY_ALIAS = re.compile(r"assigned_native_event\.native_record\.turns\[(0|[1-9][0-9]*)\]\Z")

    def __init__(self, packet, source, scene):
        require(scene in SCENES, "UNKNOWN_SCENE", scene=scene)
        self.source = {}
        self.native = {}
        self.native_order = []
        self.aliases = {}
        self.native_binding = None
        for turn in source["turns"]:
            ref = turn["turn_ref"]
            require(isinstance(ref, str) and ref not in self.source, "DUPLICATE_SOURCE_REFS")
            self.source[ref] = turn
        index = SCENES.index(scene)
        if index >= 3:
            assigned = packet["native_events"][index - 3]
            self.native_binding = assigned["native_event_binding"]
            record = verify(self.native_binding)
            require(record == assigned["native_record"], "PACKET_NATIVE_RECORD_DIFFERS_FROM_BOUND_RECORD")
            dialogue_id = record["dialogue_id"]
            require(isinstance(dialogue_id, str) and bool(dialogue_id), "INVALID_NATIVE_DIALOGUE_ID")
            for turn in record["turns"]:
                canonical = dialogue_id + ":" + str(turn["turn_id"])
                alias = dialogue_id + ":turn_id=" + str(turn["turn_id"])
                require(canonical not in self.native and canonical not in self.source
                        and alias not in self.source and alias not in self.aliases, "NONUNIQUE_NATIVE_REFERENCE")
                require(isinstance(turn.get("utterance"), str) and bool(turn["utterance"]), "INVALID_NATIVE_UTTERANCE")
                self.native[canonical] = turn
                self.native_order.append(canonical)
                self.aliases[alias] = canonical

    def resolve(self, ref):
        require(isinstance(ref, str), "EVIDENCE_REF_NOT_STRING")
        if ref in self.source:
            turn = self.source[ref]
            return ref, "cumulative_source", turn, turn["text"]
        canonical = self.aliases.get(ref, ref)
        match = self.ARRAY_ALIAS.fullmatch(ref)
        if match:
            index = int(match.group(1))
            require(index < len(self.native_order), "NATIVE_ARRAY_INDEX_OUT_OF_RANGE", reference=ref)
            canonical = self.native_order[index]
        require(canonical in self.native, "UNKNOWN_OR_UNAUTHORIZED_EVIDENCE_REF", reference=ref)
        turn = self.native[canonical]
        return canonical, "assigned_native_record", turn, turn["utterance"]

    def prompt_contract(self):
        return {"source_references": "Use exact source.turns[].turn_ref",
                "native_references": list(self.native),
                "native_quote_domain": "assigned_native_event.native_record.turns[].utterance ONLY; never frames/state/metadata",
                "accepted_compatibility_aliases": ["<assigned dialogue_id>:turn_id=<turn_id>",
                                                    "assigned_native_event.native_record.turns[<zero-based array index>]"],
                "native_event_binding": self.native_binding,
                "source_turn_count_excludes_native_turns": True}


def validate_review(response, packet, source, scene, *, require_ready=True):
    """Validate unchanged raw review, then separately enforce semantic admission."""
    require(response.get("full_source_read") is True, "REVIEW_FULL_SOURCE_READ_REQUIRED")
    count = response.get("source_turn_count")
    require(type(count) is int and count == len(source["turns"]), "REVIEW_SOURCE_COUNT_MISMATCH",
            expected=len(source["turns"]), reported=count)
    require(response.get("independent_of_source_author") is True, "REVIEW_INDEPENDENCE_REQUIRED")
    require(type(response.get("source_or_gold_ambiguity_failure")) is bool, "REVIEW_AMBIGUITY_BOOLEAN_REQUIRED")
    checks = response.get("checks", {})
    require(isinstance(checks, dict) and set(CHECKS) <= set(checks), "REVIEW_REQUIRED_DIMENSIONS_MISSING")
    resolver = EvidenceResolver(packet, source, scene)
    checked = []
    for dimension, check in checks.items():
        require(isinstance(check.get("reason"), str) and bool(check["reason"].strip())
                and isinstance(check.get("evidence"), list) and bool(check["evidence"]),
                "EMPTY_REVIEW_CHECK", dimension=dimension)
        require(check.get("status") in ("PASS", "WARNING", "PASS_WITH_WARNING", "FAIL", "HOLD"),
                "INVALID_REVIEW_CHECK_STATUS", dimension=dimension)
        for evidence in check["evidence"]:
            raw_ref = evidence.get("turn_ref")
            canonical, domain, turn, text = resolver.resolve(raw_ref)
            quote = evidence.get("quote")
            require(isinstance(quote, str) and bool(quote), "EMPTY_EVIDENCE_QUOTE", reference=raw_ref)
            require(quote in text, "QUOTE_NOT_EXACT_CONTIGUOUS_SUBSTRING",
                    reference=raw_ref, canonical_reference=canonical, domain=domain)
            if "speaker" in evidence:
                require(evidence["speaker"] == turn["speaker"], "EVIDENCE_SPEAKER_MISMATCH", reference=raw_ref)
            checked.append({"dimension": dimension, "raw_turn_ref": raw_ref, "canonical_turn_ref": canonical,
                            "domain": domain, "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
                            "exact_contiguous_substring": True})
    findings = response.get("findings")
    codes = response.get("failure_codes")
    require(isinstance(findings, list) and isinstance(codes, list), "REVIEW_FINDINGS_OR_CODES_MISSING")
    for finding in findings:
        require(finding.get("severity") in ("WARNING", "BLOCKING"), "INVALID_FINDING_SEVERITY")
        for ref in finding.get("turn_refs", []):
            resolver.resolve(ref)
    semantic_failures = []
    if response.get("decision") != "READY":
        semantic_failures.append("DECISION_NOT_READY")
    if response["source_or_gold_ambiguity_failure"]:
        semantic_failures.append("SOURCE_OR_GOLD_AMBIGUITY")
    if any(check["status"] not in ("PASS", "WARNING", "PASS_WITH_WARNING") for check in checks.values()):
        semantic_failures.append("FAILED_REVIEW_DIMENSION")
    if any(finding["severity"] == "BLOCKING" for finding in findings):
        semantic_failures.append("BLOCKING_FINDING")
    if codes:
        semantic_failures.append("NONEMPTY_FAILURE_CODES")
    receipt = {"schema_version": "oet1_followup_review_validation_v1", "scene_ref": scene,
               "status": "PASS_EVIDENCE_AND_ADMISSION" if not semantic_failures else "PASS_EVIDENCE_ONLY_REJECT_SEMANTIC_ADMISSION",
               "source_turn_count": len(resolver.source), "native_turn_count": len(resolver.native),
               "native_event_binding": resolver.native_binding, "checked_quotes": checked,
               "quote_counts_by_domain": dict(Counter(item["domain"] for item in checked)),
               "semantic_admitted": not semantic_failures, "semantic_failures": semantic_failures,
               "raw_response_rewritten": False, "checker_binding": bind(__file__)}
    if require_ready:
        require(not semantic_failures, "REVIEW_SEMANTIC_ADMISSION_REJECTED", failures=semantic_failures,
                decision=response.get("decision"), failure_codes=codes)
    return receipt
