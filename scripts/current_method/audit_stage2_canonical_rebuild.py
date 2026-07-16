#!/usr/bin/env python3
"""Fail-closed, read-only audit for a canonical Stage2 graph rebuild.

The auditor never repairs, rewrites, or admits a row.  It cross-checks the
immutable Stage2 source, evidence-only extraction manifests, admitted graphs,
and Stage3 parent payloads.  The only optional write is the explicit JSON
audit report requested with ``--output``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import build_stage3_parent_v2  # noqa: E402
from scripts.current_method import semantic_graph_v2  # noqa: E402


AUDIT_SCHEMA_VERSION = "stage2_canonical_rebuild_audit_v2"
REBUILD_SCHEMA_VERSION = "stage2_canonical_graph_rebuild_v3"
PARENT_SCHEMA_VERSION = "stage3_parent_v3"

DEFAULT_SOURCE = ROOT / (
    "base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/"
    "train_mix/frontier_train.jsonl"
)
DEFAULT_REBUILD_DIR = ROOT / (
    "base_restart_current_method_validation/"
    "172_stage2_canonical_graph_rebuild_v3_semantic_audited_final"
)

REQUIRED_FILES = (
    "rebuild_summary.json",
    "extraction_requests.jsonl",
    "ineligible_source_rows.jsonl",
    "semantic_audit_exclusions_applied.jsonl",
    "source_event_exclusions.jsonl",
    "rejected_events.jsonl",
    "stage2_semantic_graphs.jsonl",
    "stage3_parent_v3.jsonl",
    "stage3_parent_v3_rejected.jsonl",
)

# These are payload fields that would expose an old task, answer, solver
# judgement, or legacy graph to the next challenger.  Audit/provenance booleans
# such as ``old_qa_reused`` are intentionally not in this exact-key denylist.
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "answer",
        "challenger_output",
        "correct_answer",
        "correct_option",
        "expanded_graph",
        "frontier_label",
        "gold",
        "gold_answer",
        "gold_label",
        "graph_update",
        "old_qa",
        "option_map",
        "options",
        "qa",
        "raw_qa",
        "solver_eval",
        "state_chain",
        "state_chain_nodes",
    }
)
FORBIDDEN_REQUEST_KEYS = FORBIDDEN_PAYLOAD_KEYS | frozenset(
    {
        "entity_hint",
        "slot_hint",
        "status_hint",
        "time_index_hint",
        "value_hint",
    }
)

LEGACY_QA_TEXT_RE = re.compile(
    r"raw[\s_-]*qa|\bQ\s*:\s*.+?\bA\s*:\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
REQUEST_QA_TEXT_RE = re.compile(
    r"raw[\s_-]*qa|(?:^|\s)[QA]\s*:\s*|\b(?:question|answer)\s*:\s*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AuditExpectations:
    graph_count: int = 77
    parent_count: int = 77
    canonical_event_count: int = 385
    source_event_count: int = 556
    preexcluded_source_event_count: int = 171
    semantic_audit_exclusion_count: int = 18
    fallback_event_count: int = 0
    ineligible_source_row_count: int = 1
    eligible_rejection_count: int = 0
    minimum_events_per_row: int = 2
    lineage_count: int = 6


class AuditCollector:
    def __init__(self) -> None:
        self.failures: list[dict[str, Any]] = []

    def require(
        self,
        condition: bool,
        code: str,
        path: str,
        message: str,
        **details: Any,
    ) -> None:
        if condition:
            return
        item: dict[str, Any] = {
            "code": code,
            "path": path,
            "message": message,
        }
        if details:
            item["details"] = details
        self.failures.append(item)


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def iter_mappings(value: Any, path: str = "$") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from iter_mappings(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_mappings(child, f"{path}[{index}]")


def forbidden_key_paths(value: Any, forbidden: frozenset[str]) -> list[str]:
    found: list[str] = []
    for path, mapping in iter_mappings(value):
        for key in mapping:
            if str(key).casefold() in forbidden:
                found.append(f"{path}.{key}")
    return sorted(found)


def source_graph_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    graph = row.get("graph_update")
    events = graph.get("events") if isinstance(graph, dict) else None
    if not isinstance(events, list) or not all(isinstance(item, dict) for item in events):
        return []
    return copy.deepcopy(events)


def event_provenance_source_id(event: dict[str, Any]) -> str:
    provenance = event.get("provenance")
    if not isinstance(provenance, dict):
        return ""
    return normalize_space(provenance.get("source_event_id"))


def parent_source_sample_id(parent: dict[str, Any]) -> str:
    snapshot = parent.get("source_snapshot")
    if isinstance(snapshot, dict):
        value = normalize_space(snapshot.get("source_sample_id"))
        if value:
            return value
    metadata = parent.get("metadata")
    if isinstance(metadata, dict):
        return normalize_space(metadata.get("source_sample_id"))
    return ""


def check_exact_count(
    collector: AuditCollector,
    *,
    observed: int,
    expected: int,
    code: str,
    path: str,
    label: str,
) -> None:
    collector.require(
        observed == expected,
        code,
        path,
        f"Expected exactly {expected} {label}, observed {observed}.",
        expected=expected,
        observed=observed,
    )


def audit_rebuild(
    *,
    source_path: Path,
    rebuild_dir: Path,
    expectations: AuditExpectations = AuditExpectations(),
    forbidden_jsonl: Iterable[Path] = (),
) -> dict[str, Any]:
    """Return a complete audit report without modifying any input artifact."""

    source_path = source_path.resolve()
    rebuild_dir = rebuild_dir.resolve()
    forbidden_paths = [path.resolve() for path in forbidden_jsonl]
    collector = AuditCollector()

    missing = [name for name in REQUIRED_FILES if not (rebuild_dir / name).is_file()]
    collector.require(
        source_path.is_file(),
        "source_file_missing",
        str(source_path),
        "The immutable Stage2 source JSONL is missing.",
    )
    collector.require(
        not missing,
        "required_rebuild_files_missing",
        str(rebuild_dir),
        "One or more required rebuild artifacts are missing.",
        filenames=missing,
    )
    missing_forbidden = [str(path) for path in forbidden_paths if not path.is_file()]
    collector.require(
        not missing_forbidden,
        "forbidden_jsonl_missing",
        "$.forbidden_jsonl",
        "A requested heldout exact-overlap input is missing.",
        paths=missing_forbidden,
    )
    if collector.failures:
        return _final_report(
            collector,
            source_path=source_path,
            rebuild_dir=rebuild_dir,
            forbidden_paths=forbidden_paths,
            expectations=expectations,
            observed={},
        )

    try:
        source_rows = read_jsonl(source_path)
        summary = read_json(rebuild_dir / "rebuild_summary.json")
        requests = read_jsonl(rebuild_dir / "extraction_requests.jsonl")
        ineligible_rows = read_jsonl(rebuild_dir / "ineligible_source_rows.jsonl")
        semantic_audit_exclusions = read_jsonl(
            rebuild_dir / "semantic_audit_exclusions_applied.jsonl"
        )
        exclusions = read_jsonl(rebuild_dir / "source_event_exclusions.jsonl")
        rejected_events = read_jsonl(rebuild_dir / "rejected_events.jsonl")
        graphs = read_jsonl(rebuild_dir / "stage2_semantic_graphs.jsonl")
        parents = read_jsonl(rebuild_dir / "stage3_parent_v3.jsonl")
        parent_rejections = read_jsonl(rebuild_dir / "stage3_parent_v3_rejected.jsonl")
    except Exception as exc:  # Fail closed while still returning machine-readable JSON.
        collector.require(
            False,
            "artifact_parse_failed",
            str(rebuild_dir),
            "A required artifact could not be parsed.",
            error=f"{type(exc).__name__}: {exc}",
        )
        return _final_report(
            collector,
            source_path=source_path,
            rebuild_dir=rebuild_dir,
            forbidden_paths=forbidden_paths,
            expectations=expectations,
            observed={},
        )

    source_by_sample: dict[str, dict[str, Any]] = {}
    source_event_owner: dict[str, str] = {}
    source_event_ids: list[str] = []
    for row_index, row in enumerate(source_rows):
        sample_id = normalize_space(row.get("sample_id") or row.get("source_id"))
        collector.require(
            bool(sample_id),
            "source_sample_id_missing",
            f"$.source[{row_index}]",
            "Every source row must have a stable sample identity.",
        )
        collector.require(
            sample_id not in source_by_sample,
            "duplicate_source_sample_id",
            f"$.source[{row_index}]",
            "Source sample identities must be unique.",
            sample_id=sample_id,
        )
        if sample_id:
            source_by_sample[sample_id] = row
        events = source_graph_events(row)
        collector.require(
            bool(events),
            "source_graph_events_missing",
            f"$.source[{row_index}].graph_update.events",
            "Each source row must expose its legacy graph events for disposition auditing.",
        )
        for event_index, event in enumerate(events):
            event_id = normalize_space(event.get("event_id"))
            collector.require(
                bool(event_id),
                "source_event_id_missing",
                f"$.source[{row_index}].graph_update.events[{event_index}]",
                "Every legacy source event must have an event_id.",
            )
            collector.require(
                event_id not in source_event_owner,
                "duplicate_source_event_id",
                f"$.source[{row_index}].graph_update.events[{event_index}].event_id",
                "Legacy source event IDs must be globally unique.",
                event_id=event_id,
            )
            if event_id:
                source_event_ids.append(event_id)
                source_event_owner[event_id] = sample_id

    ineligible_by_request: dict[str, dict[str, Any]] = {}
    ineligible_by_sample: dict[str, dict[str, Any]] = {}
    for row_index, ineligible in enumerate(ineligible_rows):
        path = f"$.ineligible_source_rows[{row_index}]"
        request_id = normalize_space(ineligible.get("request_id"))
        sample_id = normalize_space(ineligible.get("source_sample_id"))
        collector.require(
            ineligible.get("schema_version") == REBUILD_SCHEMA_VERSION,
            "ineligible_source_row_schema_mismatch",
            f"{path}.schema_version",
            "Ineligible source rows must use the frozen rebuild schema.",
        )
        collector.require(
            ineligible.get("reason") == "no_clean_linked_semantic_events",
            "invalid_ineligible_source_row_reason",
            f"{path}.reason",
            "Only a source row with no clean linked semantic events may be ineligible.",
            observed=ineligible.get("reason"),
        )
        collector.require(
            ineligible.get("dialogue_fallback_used") is False,
            "ineligible_source_row_used_dialogue_fallback",
            f"{path}.dialogue_fallback_used",
            "Formal v3 reconstruction must not rescue an ineligible row with dialogue fallback.",
            observed=ineligible.get("dialogue_fallback_used"),
        )
        collector.require(
            ineligible.get("disposition") == "source_row_quarantined",
            "invalid_ineligible_source_row_disposition",
            f"{path}.disposition",
            "An ineligible source row must be explicitly quarantined.",
            observed=ineligible.get("disposition"),
        )
        collector.require(
            bool(request_id) and request_id not in ineligible_by_request,
            "duplicate_or_empty_ineligible_request_id",
            f"{path}.request_id",
            "Ineligible request IDs must be non-empty and unique.",
            request_id=request_id,
        )
        collector.require(
            sample_id in source_by_sample and sample_id not in ineligible_by_sample,
            "ineligible_source_mapping_invalid",
            f"{path}.source_sample_id",
            "Each ineligible record must map one-to-one to an immutable source row.",
            source_sample_id=sample_id,
        )
        if request_id:
            ineligible_by_request[request_id] = ineligible
        if sample_id:
            ineligible_by_sample[sample_id] = ineligible

    check_exact_count(
        collector,
        observed=len(ineligible_rows),
        expected=expectations.ineligible_source_row_count,
        code="unexpected_ineligible_source_row_count",
        path="$.ineligible_source_rows",
        label="ineligible source rows",
    )

    request_by_id: dict[str, dict[str, Any]] = {}
    request_by_sample: dict[str, dict[str, Any]] = {}
    reextracted_ids: list[str] = []
    fallback_ids: list[str] = []
    request_event_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, request in enumerate(requests):
        path = f"$.extraction_requests[{row_index}]"
        request_id = normalize_space(request.get("request_id"))
        sample_id = normalize_space(request.get("source_sample_id"))
        collector.require(
            request.get("schema_version") == REBUILD_SCHEMA_VERSION,
            "request_schema_mismatch",
            f"{path}.schema_version",
            "Extraction requests must use the frozen rebuild schema.",
        )
        collector.require(
            bool(request_id) and request_id not in request_by_id,
            "duplicate_or_empty_request_id",
            f"{path}.request_id",
            "Request IDs must be non-empty and unique.",
            request_id=request_id,
        )
        collector.require(
            sample_id in source_by_sample and sample_id not in request_by_sample,
            "request_source_mapping_invalid",
            f"{path}.source_sample_id",
            "Each request must map one-to-one to an immutable source row.",
            source_sample_id=sample_id,
        )
        if request_id:
            request_by_id[request_id] = request
        if sample_id:
            request_by_sample[sample_id] = request

        leaked_keys = forbidden_key_paths(request, FORBIDDEN_REQUEST_KEYS)
        collector.require(
            not leaked_keys,
            "forbidden_request_payload_key",
            path,
            "An extraction request contains a legacy semantic hint or QA payload key.",
            paths=leaked_keys,
        )
        events = request.get("events")
        ineligible = ineligible_by_request.get(request_id)
        collector.require(
            isinstance(events, list)
            and (bool(events) or isinstance(ineligible, dict)),
            "request_events_missing",
            f"{path}.events",
            "Every eligible request must contain evidence-only events; only an audited ineligible row may be empty.",
        )
        if isinstance(ineligible, dict):
            collector.require(
                isinstance(events, list)
                and not events
                and int(request.get("fallback_event_count") or 0) == 0
                and normalize_space(ineligible.get("source_sample_id")) == sample_id,
                "ineligible_request_contract_invalid",
                path,
                "An ineligible request must be empty, use no fallback, and match its quarantine record.",
            )
        for event_index, event in enumerate(events if isinstance(events, list) else []):
            event_path = f"{path}.events[{event_index}]"
            if not isinstance(event, dict):
                collector.require(False, "invalid_request_event", event_path, "Request event must be an object.")
                continue
            source_event_id = normalize_space(event.get("source_event_id"))
            reconstruction_source = normalize_space(event.get("reconstruction_source"))
            collector.require(
                reconstruction_source == "source_event_evidence_link",
                "unsupported_reconstruction_source",
                f"{event_path}.reconstruction_source",
                "Formal v3 request events must come only from linked source-event evidence; dialogue fallback is forbidden.",
                value=reconstruction_source,
            )
            collector.require(
                bool(source_event_id) and (request_id, source_event_id) not in request_event_by_pair,
                "duplicate_or_empty_requested_source_event_id",
                f"{event_path}.source_event_id",
                "Requested source-event IDs must be non-empty and unique within a request.",
                source_event_id=source_event_id,
            )
            request_event_by_pair[(request_id, source_event_id)] = event
            if reconstruction_source == "source_event_evidence_link":
                reextracted_ids.append(source_event_id)
                collector.require(
                    source_event_owner.get(source_event_id) == sample_id,
                    "reextracted_event_owner_mismatch",
                    f"{event_path}.source_event_id",
                    "A source-linked request event must belong to the request's source row.",
                    source_event_id=source_event_id,
                    expected_owner=source_event_owner.get(source_event_id),
                    observed_owner=sample_id,
                )
            elif reconstruction_source == "clean_dialogue_fallback":
                fallback_ids.append(source_event_id)
                collector.require(
                    source_event_id not in source_event_owner,
                    "fallback_reuses_legacy_event_id",
                    f"{event_path}.source_event_id",
                    "Fallback events must be counted separately from the legacy-event disposition.",
                    source_event_id=source_event_id,
                )
            evidence = event.get("evidence")
            collector.require(
                isinstance(evidence, list) and bool(evidence),
                "request_event_evidence_missing",
                f"{event_path}.evidence",
                "Every requested event must carry clean evidence turns.",
            )
            evidence_ids: set[str] = set()
            for evidence_index, evidence_item in enumerate(evidence if isinstance(evidence, list) else []):
                evidence_path = f"{event_path}.evidence[{evidence_index}]"
                if not isinstance(evidence_item, dict):
                    collector.require(False, "invalid_request_evidence", evidence_path, "Evidence must be an object.")
                    continue
                turn_id = normalize_space(evidence_item.get("turn_id"))
                text = normalize_space(evidence_item.get("text"))
                collector.require(
                    bool(turn_id) and turn_id not in evidence_ids and bool(text),
                    "invalid_request_evidence_turn",
                    evidence_path,
                    "Evidence turn IDs/text must be non-empty and IDs unique per requested event.",
                    turn_id=turn_id,
                )
                if turn_id:
                    evidence_ids.add(turn_id)
                collector.require(
                    REQUEST_QA_TEXT_RE.search(text) is None,
                    "legacy_qa_in_request_evidence",
                    f"{evidence_path}.text",
                    "Legacy QA scaffold text reached an evidence-only model request.",
                    turn_id=turn_id,
                )

    empty_request_ids = {
        request_id
        for request_id, request in request_by_id.items()
        if isinstance(request.get("events"), list) and not request.get("events")
    }
    collector.require(
        empty_request_ids == set(ineligible_by_request),
        "ineligible_request_coverage_mismatch",
        "$.ineligible_source_rows",
        "Empty extraction requests and ineligible source-row records must correspond exactly.",
        empty_requests_without_record=sorted(
            empty_request_ids - set(ineligible_by_request)
        ),
        records_without_empty_request=sorted(
            set(ineligible_by_request) - empty_request_ids
        ),
    )
    collector.require(
        set(request_by_sample) == set(source_by_sample),
        "source_request_coverage_mismatch",
        "$.extraction_requests",
        "Every immutable source row must have exactly one extraction request, including quarantined ineligible rows.",
        sources_without_request=sorted(set(source_by_sample) - set(request_by_sample)),
        requests_without_source=sorted(set(request_by_sample) - set(source_by_sample)),
    )

    semantic_audit_by_source_id: dict[str, dict[str, Any]] = {}
    for row_index, semantic_exclusion in enumerate(semantic_audit_exclusions):
        path = f"$.semantic_audit_exclusions_applied[{row_index}]"
        event_id = normalize_space(semantic_exclusion.get("source_event_id"))
        request_id = normalize_space(semantic_exclusion.get("request_id"))
        sample_id = normalize_space(semantic_exclusion.get("source_sample_id"))
        collector.require(
            semantic_exclusion.get("schema_version")
            == "stage2_semantic_audit_exclusion_v1",
            "semantic_audit_exclusion_schema_mismatch",
            f"{path}.schema_version",
            "Applied semantic-audit exclusions must use the frozen v1 schema.",
        )
        collector.require(
            semantic_exclusion.get("decision") == "quarantine",
            "semantic_audit_exclusion_decision_invalid",
            f"{path}.decision",
            "Every applied semantic-audit decision must be quarantine.",
            observed=semantic_exclusion.get("decision"),
        )
        collector.require(
            semantic_exclusion.get("disposition")
            == "quarantined_before_canonical_admission",
            "semantic_audit_exclusion_disposition_invalid",
            f"{path}.disposition",
            "Every semantic-audit exclusion must record that quarantine was applied before canonical admission.",
            observed=semantic_exclusion.get("disposition"),
        )
        collector.require(
            bool(event_id) and event_id not in semantic_audit_by_source_id,
            "duplicate_or_empty_semantic_audit_source_event_id",
            f"{path}.source_event_id",
            "Applied semantic-audit source-event IDs must be non-empty and unique.",
            source_event_id=event_id,
        )
        collector.require(
            source_event_owner.get(event_id) == sample_id
            and request_by_id.get(request_id) is request_by_sample.get(sample_id),
            "semantic_audit_exclusion_source_mapping_invalid",
            path,
            "Each semantic-audit exclusion must map to the owning source row and its extraction request.",
            source_event_id=event_id,
            request_id=request_id,
            source_sample_id=sample_id,
        )
        if event_id:
            semantic_audit_by_source_id[event_id] = semantic_exclusion
    check_exact_count(
        collector,
        observed=len(semantic_audit_exclusions),
        expected=expectations.semantic_audit_exclusion_count,
        code="unexpected_semantic_audit_exclusion_count",
        path="$.semantic_audit_exclusions_applied",
        label="applied semantic-audit exclusions",
    )

    excluded_ids: list[str] = []
    semantic_source_exclusion_by_id: dict[str, dict[str, Any]] = {}
    for row_index, exclusion in enumerate(exclusions):
        path = f"$.source_event_exclusions[{row_index}]"
        event_id = normalize_space(exclusion.get("source_event_id"))
        sample_id = normalize_space(exclusion.get("source_sample_id"))
        request_id = normalize_space(exclusion.get("request_id"))
        excluded_ids.append(event_id)
        collector.require(
            exclusion.get("schema_version") == REBUILD_SCHEMA_VERSION,
            "exclusion_schema_mismatch",
            f"{path}.schema_version",
            "Source exclusions must use the frozen rebuild schema.",
        )
        collector.require(
            exclusion.get("disposition") == "quarantined_before_model",
            "exclusion_not_quarantined",
            f"{path}.disposition",
            "Every excluded legacy source event must be explicitly quarantined before model use.",
        )
        collector.require(
            source_event_owner.get(event_id) == sample_id,
            "excluded_event_owner_mismatch",
            f"{path}.source_event_id",
            "An excluded event must belong to the declared source row.",
            source_event_id=event_id,
            source_sample_id=sample_id,
        )
        collector.require(
            request_by_id.get(request_id) is request_by_sample.get(sample_id),
            "excluded_event_request_mismatch",
            path,
            "An exclusion must refer to the request for the same source row.",
        )
        if exclusion.get("reason") == "independent_semantic_audit_exclusion":
            collector.require(
                event_id not in semantic_source_exclusion_by_id,
                "duplicate_semantic_audit_source_exclusion",
                path,
                "A semantic-audit source event may appear only once in the source exclusion ledger.",
                source_event_id=event_id,
            )
            if event_id:
                semantic_source_exclusion_by_id[event_id] = exclusion

    collector.require(
        set(semantic_source_exclusion_by_id) == set(semantic_audit_by_source_id),
        "semantic_audit_exclusion_ledger_mismatch",
        "$.semantic_audit_exclusion_ledger",
        "Applied semantic-audit exclusions and independent source-exclusion records must correspond one-to-one.",
        applied_without_source_exclusion=sorted(
            set(semantic_audit_by_source_id) - set(semantic_source_exclusion_by_id)
        ),
        source_exclusion_without_applied_record=sorted(
            set(semantic_source_exclusion_by_id) - set(semantic_audit_by_source_id)
        ),
    )
    for event_id in sorted(
        set(semantic_source_exclusion_by_id) & set(semantic_audit_by_source_id)
    ):
        applied = semantic_audit_by_source_id[event_id]
        source_exclusion = semantic_source_exclusion_by_id[event_id]
        collector.require(
            normalize_space(applied.get("request_id"))
            == normalize_space(source_exclusion.get("request_id"))
            and normalize_space(applied.get("source_sample_id"))
            == normalize_space(source_exclusion.get("source_sample_id"))
            and int(applied.get("source_line_number") or 0)
            == int(source_exclusion.get("source_line_number") or 0)
            and normalize_space(applied.get("reason"))
            == normalize_space(source_exclusion.get("semantic_audit_reason"))
            and normalize_space(applied.get("detail"))
            == normalize_space(source_exclusion.get("semantic_audit_detail")),
            "semantic_audit_exclusion_payload_mismatch",
            f"$.semantic_audit_exclusion_ledger[{event_id}]",
            "The applied semantic decision and source exclusion must carry identical provenance and audit rationale.",
        )

    source_id_set = set(source_event_ids)
    reextracted_set = set(reextracted_ids)
    excluded_set = set(excluded_ids)
    collector.require(
        len(reextracted_ids) == len(reextracted_set),
        "duplicate_reextracted_source_event_disposition",
        "$.extraction_requests",
        "A legacy event was selected for re-extraction more than once.",
    )
    collector.require(
        len(excluded_ids) == len(excluded_set),
        "duplicate_quarantined_source_event_disposition",
        "$.source_event_exclusions",
        "A legacy event was quarantined more than once.",
    )
    collector.require(
        not (reextracted_set & excluded_set),
        "source_event_has_multiple_dispositions",
        "$.legacy_event_disposition",
        "A legacy event cannot be both re-extracted and quarantined.",
        event_ids=sorted(reextracted_set & excluded_set),
    )
    disposition_ids = reextracted_set | excluded_set
    collector.require(
        disposition_ids == source_id_set,
        "incomplete_legacy_event_disposition",
        "$.legacy_event_disposition",
        "Every legacy event must have exactly one disposition: re-extracted or quarantined.",
        missing=sorted(source_id_set - disposition_ids),
        unexpected=sorted(disposition_ids - source_id_set),
    )
    collector.require(
        len(fallback_ids) == len(set(fallback_ids)),
        "duplicate_fallback_event_id",
        "$.extraction_requests",
        "Fallback event IDs must be unique.",
    )
    check_exact_count(
        collector,
        observed=len(excluded_ids),
        expected=expectations.preexcluded_source_event_count,
        code="unexpected_preexcluded_source_event_count",
        path="$.source_event_exclusions",
        label="pre-model quarantined source events",
    )
    for request_id, ineligible in ineligible_by_request.items():
        path = f"$.ineligible_source_rows[{request_id}]"
        sample_id = normalize_space(ineligible.get("source_sample_id"))
        request = request_by_id.get(request_id) or {}
        owned_source_ids = {
            event_id
            for event_id, owner in source_event_owner.items()
            if owner == sample_id
        }
        skipped_source_ids = {
            normalize_space(item.get("source_event_id"))
            for item in (request.get("skipped_events") or [])
            if isinstance(item, dict) and normalize_space(item.get("source_event_id"))
        }
        collector.require(
            int(ineligible.get("source_event_count") or 0)
            == len(owned_source_ids)
            == int(ineligible.get("excluded_source_event_count") or 0),
            "ineligible_source_event_count_mismatch",
            path,
            "An ineligible row must declare and exclude every legacy event owned by its source row.",
            owned_source_event_count=len(owned_source_ids),
            declared_source_event_count=ineligible.get("source_event_count"),
            declared_excluded_source_event_count=ineligible.get(
                "excluded_source_event_count"
            ),
        )
        collector.require(
            skipped_source_ids == owned_source_ids
            and (excluded_set & owned_source_ids) == owned_source_ids
            and not (reextracted_set & owned_source_ids),
            "ineligible_source_event_disposition_mismatch",
            path,
            "Every event from an ineligible row must be skipped and quarantined, never re-extracted.",
            missing_from_request_skips=sorted(owned_source_ids - skipped_source_ids),
            missing_from_exclusions=sorted(owned_source_ids - excluded_set),
            unexpectedly_reextracted=sorted(owned_source_ids & reextracted_set),
        )

    graph_by_sample: dict[str, dict[str, Any]] = {}
    graph_ids: list[str] = []
    canonical_event_ids: list[str] = []
    canonical_source_event_ids: list[str] = []
    canonical_event_count = 0
    dependency_count = 0
    dangling_or_nonprior_dependency_count = 0
    for row_index, graph_row in enumerate(graphs):
        path = f"$.stage2_semantic_graphs[{row_index}]"
        sample_id = normalize_space(graph_row.get("source_sample_id"))
        request_id = normalize_space(graph_row.get("request_id"))
        graph = graph_row.get("semantic_parent_graph")
        collector.require(
            graph_row.get("schema_version") == REBUILD_SCHEMA_VERSION,
            "canonical_row_schema_mismatch",
            f"{path}.schema_version",
            "Canonical graph rows must use the frozen rebuild schema.",
        )
        collector.require(
            sample_id in request_by_sample and sample_id not in graph_by_sample,
            "canonical_graph_source_mapping_invalid",
            f"{path}.source_sample_id",
            "Canonical graph rows must map one-to-one to extraction requests.",
            source_sample_id=sample_id,
        )
        collector.require(
            request_by_id.get(request_id) is request_by_sample.get(sample_id),
            "canonical_graph_request_mapping_invalid",
            f"{path}.request_id",
            "Canonical graph request/source identities are inconsistent.",
        )
        if sample_id:
            graph_by_sample[sample_id] = graph_row
        collector.require(
            isinstance(graph, dict),
            "canonical_graph_missing",
            f"{path}.semantic_parent_graph",
            "A canonical row must contain a graph object.",
        )
        if not isinstance(graph, dict):
            continue
        collector.require(
            graph.get("schema_version") == semantic_graph_v2.SCHEMA_VERSION,
            "semantic_graph_schema_mismatch",
            f"{path}.semantic_parent_graph.schema_version",
            "Canonical graphs must use semantic_graph_v2.",
        )
        graph_id = normalize_space(graph.get("graph_id"))
        graph_ids.append(graph_id)
        events = graph.get("events")
        collector.require(
            isinstance(events, list),
            "canonical_events_invalid",
            f"{path}.semantic_parent_graph.events",
            "Canonical graph events must be an array.",
        )
        events = events if isinstance(events, list) else []
        canonical_event_count += len(events)
        event_position_by_id = {
            normalize_space(event.get("event_id")): event_index
            for event_index, event in enumerate(events)
            if isinstance(event, dict) and normalize_space(event.get("event_id"))
        }
        collector.require(
            len(events) >= expectations.minimum_events_per_row,
            "canonical_graph_below_minimum_events",
            f"{path}.semantic_parent_graph.events",
            "Every canonical parent graph must satisfy the minimum event floor.",
            minimum=expectations.minimum_events_per_row,
            observed=len(events),
        )
        collector.require(
            graph.get("event_count") == len(events),
            "canonical_graph_event_count_mismatch",
            f"{path}.semantic_parent_graph.event_count",
            "Declared canonical event_count must equal the event array length.",
        )
        coverage = graph_row.get("coverage")
        collector.require(
            isinstance(coverage, dict)
            and coverage.get("complete_for_eligible_evidence") is True
            and int(coverage.get("rejected_eligible_event_count") or 0) == 0
            and coverage.get("old_graph_events_inherited") is False
            and coverage.get("legacy_semantic_labels_exposed_to_model") is False,
            "canonical_coverage_not_complete_or_clean",
            f"{path}.coverage",
            "Canonical admission requires complete evidence coverage and no legacy-label exposure/inheritance.",
        )
        requested = request_by_id.get(request_id, {}).get("events")
        requested = requested if isinstance(requested, list) else []
        requested_ids = {normalize_space(item.get("source_event_id")) for item in requested if isinstance(item, dict)}
        accepted_source_ids: list[str] = []
        for event_index, event in enumerate(events):
            event_path = f"{path}.semantic_parent_graph.events[{event_index}]"
            if not isinstance(event, dict):
                collector.require(False, "invalid_canonical_event", event_path, "Canonical event must be an object.")
                continue
            issues = semantic_graph_v2.validate_event(event, path=event_path)
            collector.require(
                not issues,
                "canonical_event_schema_invalid",
                event_path,
                "Canonical event failed semantic_graph_v2 validation.",
                issues=issues,
            )
            event_id = normalize_space(event.get("event_id"))
            canonical_event_ids.append(event_id)
            dependencies = event.get("depends_on")
            dependencies = dependencies if isinstance(dependencies, list) else []
            provenance = (
                event.get("provenance")
                if isinstance(event.get("provenance"), dict)
                else {}
            )
            for dependency_index, dependency_id_value in enumerate(dependencies):
                dependency_count += 1
                dependency_id = normalize_space(dependency_id_value)
                dependency_position = event_position_by_id.get(dependency_id)
                dependency_event = (
                    events[dependency_position]
                    if isinstance(dependency_position, int)
                    and 0 <= dependency_position < len(events)
                    and isinstance(events[dependency_position], dict)
                    else None
                )
                try:
                    dependency_time = int((dependency_event or {}).get("time_index"))
                    event_time = int(event.get("time_index"))
                except (TypeError, ValueError):
                    dependency_time = event_time = 0
                dependency_is_prior = bool(
                    dependency_id
                    and isinstance(dependency_position, int)
                    and dependency_position < event_index
                    and dependency_time < event_time
                )
                if not dependency_is_prior:
                    dangling_or_nonprior_dependency_count += 1
                collector.require(
                    dependency_is_prior,
                    "canonical_dependency_dangling_or_nonprior",
                    f"{event_path}.depends_on[{dependency_index}]",
                    "Every dependency must resolve to an earlier event in the same graph by both array order and time_index.",
                    dependency_event_id=dependency_id,
                    dependency_position=dependency_position,
                    event_position=event_index,
                    dependency_time_index=(
                        (dependency_event or {}).get("time_index")
                    ),
                    event_time_index=event.get("time_index"),
                )
            if dependencies and provenance.get("deterministic_terminal_recovery") is True:
                collector.require(
                    bool(normalize_space(provenance.get("dependency_source"))),
                    "terminal_dependency_source_provenance_missing",
                    f"{event_path}.provenance.dependency_source",
                    "A terminal recovery dependency must record its deterministic source provenance.",
                )
            source_event_id = event_provenance_source_id(event)
            accepted_source_ids.append(source_event_id)
            canonical_source_event_ids.append(source_event_id)
            request_event = request_event_by_pair.get((request_id, source_event_id))
            collector.require(
                request_event is not None,
                "canonical_event_not_requested",
                f"{event_path}.provenance.source_event_id",
                "Every admitted canonical event must correspond to one evidence-only request event.",
                source_event_id=source_event_id,
            )
            requested_evidence_ids = {
                normalize_space(item.get("turn_id"))
                for item in ((request_event or {}).get("evidence") or [])
                if isinstance(item, dict)
            }
            observed_evidence_ids = {
                normalize_space(item) for item in (event.get("evidence_turn_ids") or [])
            }
            collector.require(
                bool(observed_evidence_ids) and observed_evidence_ids == requested_evidence_ids,
                "canonical_evidence_refs_mismatch_request",
                f"{event_path}.evidence_turn_ids",
                "Canonical evidence refs must exactly match the evidence-only request refs.",
                expected=sorted(requested_evidence_ids),
                observed=sorted(observed_evidence_ids),
            )
        collector.require(
            len(accepted_source_ids) == len(set(accepted_source_ids))
            and set(accepted_source_ids) == requested_ids,
            "canonical_row_not_atomic_for_request",
            f"{path}.semantic_parent_graph.events",
            "Canonical row admission must include every requested event exactly once.",
            missing=sorted(requested_ids - set(accepted_source_ids)),
            unexpected=sorted(set(accepted_source_ids) - requested_ids),
        )

    collector.require(
        len(graph_ids) == len(set(graph_ids)) and all(graph_ids),
        "duplicate_or_empty_canonical_graph_id",
        "$.stage2_semantic_graphs",
        "Canonical graph IDs must be globally unique and non-empty.",
    )
    collector.require(
        len(canonical_event_ids) == len(set(canonical_event_ids)) and all(canonical_event_ids),
        "duplicate_or_empty_canonical_event_id",
        "$.stage2_semantic_graphs",
        "Canonical event IDs must be globally unique and non-empty.",
    )
    eligible_request_samples = {
        normalize_space(request.get("source_sample_id"))
        for request in requests
        if isinstance(request.get("events"), list) and bool(request.get("events"))
    }
    collector.require(
        set(graph_by_sample) == eligible_request_samples,
        "eligible_request_graph_coverage_mismatch",
        "$.stage2_semantic_graphs",
        "Every eligible request must produce exactly one canonical graph, and no ineligible request may produce one.",
        eligible_requests_without_graph=sorted(
            eligible_request_samples - set(graph_by_sample)
        ),
        graphs_without_eligible_request=sorted(
            set(graph_by_sample) - eligible_request_samples
        ),
    )
    collector.require(
        len(canonical_source_event_ids) == len(set(canonical_source_event_ids))
        and set(canonical_source_event_ids) == reextracted_set,
        "canonical_reextracted_event_coverage_mismatch",
        "$.legacy_event_disposition",
        "Every re-extracted source event must appear exactly once in canonical graphs.",
        reextracted_without_canonical=sorted(
            reextracted_set - set(canonical_source_event_ids)
        ),
        canonical_without_reextracted=sorted(
            set(canonical_source_event_ids) - reextracted_set
        ),
    )
    collector.require(
        canonical_event_count + len(excluded_ids) == len(source_event_ids),
        "canonical_preexcluded_ledger_mismatch",
        "$.legacy_event_disposition",
        "The formal v3 ledger must equal canonical events plus pre-model exclusions.",
        canonical_event_count=canonical_event_count,
        preexcluded_source_event_count=len(excluded_ids),
        source_event_count=len(source_event_ids),
    )

    parent_ids: list[str] = []
    parent_source_ids: list[str] = []
    parent_graph_ids: list[str] = []
    dialogue_hashes: list[str] = []
    lineages: set[str] = set()
    heldout_matches: list[dict[str, Any]] = []
    forbidden_index: dict[str, Any] | None = None
    if forbidden_paths:
        forbidden_index, _ = build_stage3_parent_v2.load_forbidden_identifiers(forbidden_paths)
    for row_index, parent in enumerate(parents):
        path = f"$.stage3_parent_v3[{row_index}]"
        sample_id = normalize_space(parent.get("sample_id"))
        source_id = normalize_space(parent.get("source_id"))
        source_sample_id = parent_source_sample_id(parent)
        graph_row = graph_by_sample.get(source_sample_id)
        graph = parent.get("semantic_parent_graph")
        parent_ids.append(sample_id)
        parent_source_ids.append(source_id)
        collector.require(
            parent.get("schema_version") == PARENT_SCHEMA_VERSION,
            "parent_schema_mismatch",
            f"{path}.schema_version",
            "Parent rows must use stage3_parent_v3.",
        )
        collector.require(
            bool(sample_id) and sample_id == source_id,
            "parent_sample_source_id_mismatch",
            path,
            "Parent sample_id/source_id must be the same non-empty rebuilt identity.",
            sample_id=sample_id,
            source_id=source_id,
        )
        collector.require(
            graph_row is not None,
            "parent_missing_canonical_graph_row",
            f"{path}.source_snapshot.source_sample_id",
            "Each parent must map one-to-one to a canonical graph row.",
            source_sample_id=source_sample_id,
        )
        leaked_keys = forbidden_key_paths(parent, FORBIDDEN_PAYLOAD_KEYS)
        collector.require(
            not leaked_keys,
            "forbidden_parent_payload_key",
            path,
            "A Stage3 parent contains an old QA/eval/legacy-graph payload key.",
            paths=leaked_keys,
        )
        collector.require(
            isinstance(graph, dict)
            and graph.get("schema_version") == semantic_graph_v2.SCHEMA_VERSION,
            "parent_semantic_graph_schema_mismatch",
            f"{path}.semantic_parent_graph",
            "Parent semantic graphs must use semantic_graph_v2.",
        )
        graph = graph if isinstance(graph, dict) else {}
        parent_graph_ids.append(normalize_space(graph.get("graph_id")))
        parent_events = graph.get("events")
        parent_events = parent_events if isinstance(parent_events, list) else []
        canonical_events = (
            ((graph_row or {}).get("semantic_parent_graph") or {}).get("events") or []
        )
        collector.require(
            parent_events == canonical_events,
            "parent_canonical_graph_events_mismatch",
            f"{path}.semantic_parent_graph.events",
            "Parent graph events must byte-semantically equal the admitted canonical events.",
        )
        collector.require(
            len(parent_events) >= expectations.minimum_events_per_row
            and graph.get("event_count") == len(parent_events)
            and graph.get("empty_parent_snapshot_allowed") is False,
            "parent_graph_event_contract_invalid",
            f"{path}.semantic_parent_graph",
            "Parent graph count/floor/nonempty contract is invalid.",
        )
        expected_graph_hash = sha256_json(parent_events)
        snapshot = parent.get("source_snapshot")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        collector.require(
            graph.get("graph_sha256") == expected_graph_hash
            and snapshot.get("reconstructed_graph_sha256") == expected_graph_hash,
            "parent_graph_hash_mismatch",
            f"{path}.semantic_parent_graph.graph_sha256",
            "Parent graph hashes must match the exact canonical event payload.",
            expected=expected_graph_hash,
        )
        dialogue = parent.get("dialogue")
        dialogue_turns = dialogue.get("turns") if isinstance(dialogue, dict) else None
        turns = parent.get("turns")
        collector.require(
            isinstance(turns, list) and bool(turns) and turns == dialogue_turns,
            "parent_dialogue_turns_invalid",
            f"{path}.turns",
            "Top-level and dialogue turn arrays must be identical and non-empty.",
        )
        turns = turns if isinstance(turns, list) else []
        turn_ids: list[str] = []
        qa_turn_ids: list[str] = []
        padding_turn_ids: list[str] = []
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                collector.require(False, "invalid_parent_turn", f"{path}.turns[{turn_index}]", "Dialogue turn must be an object.")
                continue
            turn_id = normalize_space(turn.get("turn_id") or turn.get("id"))
            turn_ids.append(turn_id)
            text = normalize_space(turn.get("text") or turn.get("content") or turn.get("utterance"))
            if LEGACY_QA_TEXT_RE.search(text):
                qa_turn_ids.append(turn_id or f"index:{turn_index}")
            if build_stage3_parent_v2.known_deterministic_padding_turn(turn):
                padding_turn_ids.append(turn_id or f"index:{turn_index}")
        collector.require(
            len(turn_ids) == len(set(turn_ids)) and all(turn_ids),
            "duplicate_or_empty_parent_turn_id",
            f"{path}.turns",
            "Parent dialogue turn IDs must be unique and non-empty.",
        )
        collector.require(
            not qa_turn_ids,
            "legacy_qa_dialogue_not_cleaned",
            f"{path}.turns",
            "Legacy raw-QA/Q:A scaffold turns remain in the parent dialogue.",
            turn_ids=qa_turn_ids,
        )
        collector.require(
            not padding_turn_ids,
            "deterministic_padding_not_cleaned",
            f"{path}.turns",
            "Known historical deterministic padding remains in the parent dialogue.",
            turn_ids=padding_turn_ids,
        )
        retained_turn_ids = set(turn_ids)
        dangling = sorted(
            {
                normalize_space(turn_id)
                for event in parent_events
                if isinstance(event, dict)
                for turn_id in (event.get("evidence_turn_ids") or [])
                if normalize_space(turn_id) not in retained_turn_ids
            }
        )
        collector.require(
            not dangling,
            "parent_graph_dangling_evidence_ref",
            f"{path}.semantic_parent_graph.events",
            "Every graph evidence ref must resolve to a retained clean parent turn.",
            turn_ids=dangling,
        )
        dialogue_hash = normalize_space(
            dialogue.get("dialogue_hash") if isinstance(dialogue, dict) else ""
        )
        dialogue_hashes.append(dialogue_hash)
        lineage = normalize_space(parent.get("lineage_group"))
        if lineage:
            lineages.add(lineage)
        collector.require(
            bool(lineage),
            "parent_lineage_missing",
            f"{path}.lineage_group",
            "Every parent must retain a non-empty lineage group.",
        )
        if forbidden_index is not None:
            parent_match = build_stage3_parent_v2.exact_forbidden_matches(
                build_stage3_parent_v2.exact_identifiers(parent),
                forbidden_index["exact"],
            )
            source_match = build_stage3_parent_v2.exact_forbidden_matches(
                build_stage3_parent_v2.exact_identifiers(source_by_sample.get(source_sample_id, {})),
                forbidden_index["exact"],
            )
            if parent_match or source_match:
                heldout_matches.append(
                    {
                        "parent_sample_id": sample_id,
                        "source_sample_id": source_sample_id,
                        "parent_matches": parent_match,
                        "source_matches": source_match,
                    }
                )

    collector.require(
        len(parent_ids) == len(set(parent_ids)) and all(parent_ids),
        "duplicate_or_empty_parent_id",
        "$.stage3_parent_v3",
        "Parent sample IDs must be globally unique and non-empty.",
    )
    collector.require(
        len(parent_source_ids) == len(set(parent_source_ids)) and all(parent_source_ids),
        "duplicate_or_empty_parent_source_id",
        "$.stage3_parent_v3",
        "Parent source IDs must be globally unique and non-empty.",
    )
    collector.require(
        len(parent_graph_ids) == len(set(parent_graph_ids)) and all(parent_graph_ids),
        "duplicate_or_empty_parent_graph_id",
        "$.stage3_parent_v3",
        "Parent graph IDs must be globally unique and non-empty.",
    )
    collector.require(
        len(dialogue_hashes) == len(set(dialogue_hashes)) and all(dialogue_hashes),
        "duplicate_or_empty_parent_dialogue_hash",
        "$.stage3_parent_v3",
        "Parent dialogue hashes must be globally unique and non-empty.",
    )
    collector.require(
        set(graph_by_sample) == {parent_source_sample_id(parent) for parent in parents},
        "parent_graph_source_coverage_mismatch",
        "$.stage3_parent_v3",
        "Canonical graph rows and Stage3 parent rows must cover the same source samples.",
        graphs_without_parent=sorted(set(graph_by_sample) - {parent_source_sample_id(parent) for parent in parents}),
        parents_without_graph=sorted({parent_source_sample_id(parent) for parent in parents} - set(graph_by_sample)),
    )
    collector.require(
        not heldout_matches,
        "heldout_exact_identity_overlap",
        "$.heldout_exact_overlap",
        "A rebuilt parent/source has an exact identity overlap with an optional heldout input.",
        matches=heldout_matches,
    )

    check_exact_count(
        collector,
        observed=len(graphs),
        expected=expectations.graph_count,
        code="unexpected_canonical_graph_count",
        path="$.stage2_semantic_graphs",
        label="canonical graph rows",
    )
    check_exact_count(
        collector,
        observed=len(parents),
        expected=expectations.parent_count,
        code="unexpected_parent_count",
        path="$.stage3_parent_v3",
        label="Stage3 parent rows",
    )
    check_exact_count(
        collector,
        observed=canonical_event_count,
        expected=expectations.canonical_event_count,
        code="unexpected_canonical_event_count",
        path="$.stage2_semantic_graphs[*].semantic_parent_graph.events",
        label="canonical events",
    )
    check_exact_count(
        collector,
        observed=len(source_event_ids),
        expected=expectations.source_event_count,
        code="unexpected_source_event_count",
        path="$.source[*].graph_update.events",
        label="legacy source events",
    )
    check_exact_count(
        collector,
        observed=len(fallback_ids),
        expected=expectations.fallback_event_count,
        code="unexpected_fallback_event_count",
        path="$.extraction_requests[*].events",
        label="clean-dialogue fallback events",
    )
    check_exact_count(
        collector,
        observed=len(lineages),
        expected=expectations.lineage_count,
        code="unexpected_lineage_count",
        path="$.stage3_parent_v3[*].lineage_group",
        label="unique lineage groups",
    )
    collector.require(
        not rejected_events and not parent_rejections,
        "rebuild_contains_rejections",
        str(rebuild_dir),
        "The final admitted rebuild must not contain event or parent rejection records.",
        rejected_event_records=len(rejected_events),
        parent_rejection_records=len(parent_rejections),
    )
    check_exact_count(
        collector,
        observed=len(rejected_events),
        expected=expectations.eligible_rejection_count,
        code="unexpected_eligible_rejection_count",
        path="$.rejected_events",
        label="eligible event/row rejection records",
    )

    expected_summary = {
        "input_row_count": (
            expectations.graph_count + expectations.ineligible_source_row_count
        ),
        "selected_request_count": (
            expectations.graph_count + expectations.ineligible_source_row_count
        ),
        "eligible_request_count": expectations.graph_count,
        "eligible_selected_request_count": expectations.graph_count,
        "accepted_canonical_event_count": expectations.canonical_event_count,
        "excluded_source_event_count": expectations.preexcluded_source_event_count,
        "fallback_event_count": expectations.fallback_event_count,
        "semantic_audit_exclusion_count": (
            expectations.semantic_audit_exclusion_count
        ),
        "semantic_audit_exclusion_schema_version": (
            "stage2_semantic_audit_exclusion_v1"
        ),
        "ineligible_source_row_count": expectations.ineligible_source_row_count,
        "canonical_graph_count": expectations.graph_count,
        "row_atomic_rejected_graph_count": expectations.eligible_rejection_count,
        "rejection_record_count": expectations.eligible_rejection_count,
        "parent_v3_count": expectations.parent_count,
        "parent_v3_rejected_count": 0,
    }
    for key, expected_value in expected_summary.items():
        collector.require(
            summary.get(key) == expected_value,
            "rebuild_summary_mismatch",
            f"$.rebuild_summary.{key}",
            "Persisted rebuild summary does not match the independently observed artifacts.",
            expected=expected_value,
            observed=summary.get(key),
        )
    for key, expected_value in {
        "complete_selection": True,
        "legacy_qa_in_model_prompt": False,
        "legacy_semantic_label_fields_in_model_request": False,
        "old_graph_events_inherited": False,
        "row_atomic_admission": True,
    }.items():
        collector.require(
            summary.get(key) is expected_value,
            "unsafe_rebuild_summary_flag",
            f"$.rebuild_summary.{key}",
            "Persisted safety/completeness summary flag has an unsafe value.",
            expected=expected_value,
            observed=summary.get(key),
        )

    observed = {
        "source_row_count": len(source_rows),
        "request_count": len(requests),
        "eligible_request_count": len(eligible_request_samples),
        "ineligible_source_row_count": len(ineligible_rows),
        "canonical_graph_count": len(graphs),
        "parent_count": len(parents),
        "source_event_count": len(source_event_ids),
        "reextracted_source_event_count": len(reextracted_ids),
        "quarantined_source_event_count": len(excluded_ids),
        "legacy_event_disposition_count": len(disposition_ids),
        "fallback_event_count": len(fallback_ids),
        "canonical_event_count": canonical_event_count,
        "canonical_plus_preexcluded_source_event_count": (
            canonical_event_count + len(excluded_ids)
        ),
        "semantic_audit_exclusion_count": len(semantic_audit_exclusions),
        "semantic_audit_source_exclusion_count": len(
            semantic_source_exclusion_by_id
        ),
        "dependency_count": dependency_count,
        "dangling_or_nonprior_dependency_count": (
            dangling_or_nonprior_dependency_count
        ),
        "unique_lineage_count": len(lineages),
        "lineages": sorted(lineages),
        "event_rejection_record_count": len(rejected_events),
        "parent_rejection_record_count": len(parent_rejections),
        "heldout_file_count": len(forbidden_paths),
        "heldout_exact_overlap_count": len(heldout_matches),
    }
    return _final_report(
        collector,
        source_path=source_path,
        rebuild_dir=rebuild_dir,
        forbidden_paths=forbidden_paths,
        expectations=expectations,
        observed=observed,
    )


def _final_report(
    collector: AuditCollector,
    *,
    source_path: Path,
    rebuild_dir: Path,
    forbidden_paths: list[Path],
    expectations: AuditExpectations,
    observed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "pass": not collector.failures,
        "source": str(source_path),
        "rebuild_dir": str(rebuild_dir),
        "forbidden_jsonl": [str(path) for path in forbidden_paths],
        "expectations": asdict(expectations),
        "observed": observed,
        "failure_count": len(collector.failures),
        "failures": collector.failures,
        "fail_closed": True,
        "inputs_mutated": False,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strictly audit canonical Stage2 graphs and Stage3 parents without repairing them."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rebuild-dir", type=Path, default=DEFAULT_REBUILD_DIR)
    parser.add_argument(
        "--forbidden-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Optional heldout JSONL; exact sample/question/dialogue identity overlap is fatal.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. Input artifacts are never modified.",
    )
    args = parser.parse_args(argv)
    try:
        report = audit_rebuild(
            source_path=args.source,
            rebuild_dir=args.rebuild_dir,
            forbidden_jsonl=args.forbidden_jsonl,
        )
    except Exception as exc:  # Last-resort fail-closed CLI boundary.
        report = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "pass": False,
            "source": str(args.source.resolve()),
            "rebuild_dir": str(args.rebuild_dir.resolve()),
            "failure_count": 1,
            "failures": [
                {
                    "code": "unexpected_audit_exception",
                    "path": "$",
                    "message": "Audit terminated unexpectedly and therefore failed closed.",
                    "details": {"error": f"{type(exc).__name__}: {exc}"},
                }
            ],
            "fail_closed": True,
            "inputs_mutated": False,
        }
    if args.output is not None:
        write_report(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
