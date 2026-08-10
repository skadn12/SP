"""Stateful Graph v2 contracts for restart-n5 dialogue expansion.

This module deliberately does not modify the historical Graph v1 protocol.  In
v1, an assigned stage was copied into ``completed_event_stage_ids`` after a
scene passed its surface checks.  V2 separates assignment, evidence-bearing
stage updates, state-instance transitions, and handoff accounting.

The API Realizer may propose updates, but only the deterministic functions in
this module can make them authoritative.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


STATEFUL_GRAPH_EXPANSION_VERSION = "stateful_graph_expansion_v4"
STATEFUL_GRAPH_OUTLINE_SCHEMA_VERSION = "stateful_graph_outline_v4"
STATEFUL_COMPLETION_CONTRACT_VERSION = "structured_canonical_v1"
STATEFUL_GRAPH_SCENE_CONTRACT_VERSION = "stateful_graph_scene_contract_v4"
STATEFUL_GRAPH_LEDGER_VERSION = "stateful_graph_ledger_v2"
STATEFUL_GRAPH_REALIZER_PROMPT_VERSION = "stateful_graph_realizer_v2"
STATEFUL_TERMINAL_MARKER_NORMALIZATION_VERSION = (
    "terminal_marker_apostrophe_normalization_v1")
STATEFUL_HANDOFF_WHITELIST_VERSION = "stateful_handoff_exact_whitelist_v1"
STATEFUL_CROSS_ARC_CONFLICT_GATE_VERSION = (
    "cross_arc_stage_conflict_gate_v1")
LEGACY_MIGRATION_VERSION = "graph_v1_completion_reclassification_v1"

STAGE_STATUSES = {
    "assigned", "mentioned", "in_progress", "completed", "failed",
    "carry_forward",
}
TERMINAL_STAGE_STATUSES = {"completed", "failed"}
HANDOFF_STATUSES = {"pending", "resolved", "carry_forward"}

# Four ordered four-stage arcs over ten scenes.  This preserves the original
# outline-Graph mainline while changing only the meaning of completion.
STANDARD_STAGE_SCHEDULE: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((0, 1),),
    ((0, 2), (1, 0)),
    ((0, 3), (1, 1)),
    ((1, 2), (2, 0)),
    ((2, 1),),
    ((1, 3), (2, 2), (2, 3)),
    ((3, 0),),
    ((3, 1), (3, 2)),
    ((3, 3),),
)

_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
COMPLETION_STATUSES = {"achieved", "not_achieved"}
COMPLETION_TEMPORALITIES = {"current_or_past", "future"}
COMPLETION_CONDITIONALITIES = {"unconditional", "conditional"}
STAGE_ROLES = {"base_fact_completion", "derived_state"}
MAX_CANONICAL_MARKER_WORDS = 18
MAX_CANONICAL_MARKER_CHARS = 160

# The canonical marker contract deliberately checks only an obvious matrix
# prefix.  It must not infer completion by scanning arbitrary subordinate
# natural language for modal or conditional words.
_NONTERMINAL_CANONICAL_PREFIX = re.compile(
    r"^\s*(?:(?:if|when|once|tomorrow)\b|"
    r"(?:i|we)\s+(?:can|will|would|could|should|may|might)\b|"
    r"(?:i|we)['\u2019](?:ll|d)\b|"
    r"(?:i|we)\s+(?:am|are)\s+going\s+to\b)",
    re.IGNORECASE,
)
_CANONICAL_FIRST_PERSON_PREFIX = re.compile(
    r"^\s*(?:i(?:\s|['\u2019](?:ve|m)\b)|"
    r"we(?:\s|['\u2019](?:ve|re)\b))", re.IGNORECASE)
_AMBIGUOUS_CANONICAL_PREFIX = re.compile(
    r"^\s*(?:it(?:['\u2019]s|\s+is)|that(?:['\u2019]s|\s+is)|"
    r"this\s+is|they\b)", re.IGNORECASE)


def terminal_evidence_is_future_or_conditional(text: str) -> bool:
    """Reject only an overt future/conditional matrix-clause prefix.

    Structured completion fields carry the authoritative temporality and
    conditionality.  This limited lexical check only catches contradictions
    such as ``I will decide`` or ``If I decide``.  Embedded wording such as
    ``I reflected on what might happen`` is not scanned or reinterpreted.
    """
    return _NONTERMINAL_CANONICAL_PREFIX.search(str(text)) is not None


def canonical_marker_is_short_and_speakable(text: Any) -> bool:
    """Check only format/direct-speech properties, never infer semantics."""
    if not isinstance(text, str) or not text.strip() or "\n" in text:
        return False
    marker = text.strip()
    words = re.findall(r"\b[\w'\u2019-]+\b", marker)
    return (len(marker) <= MAX_CANONICAL_MARKER_CHARS and
            len(words) <= MAX_CANONICAL_MARKER_WORDS and
            len(re.findall(r"[.!?]", marker)) <= 1 and
            _CANONICAL_FIRST_PERSON_PREFIX.search(marker) is not None and
            _AMBIGUOUS_CANONICAL_PREFIX.search(marker) is None)


STATEFUL_REALIZER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "dialogue_turns", "used_parent_anchor_ids",
        "advanced_event_arc_ids", "candidate_stage_updates",
        "candidate_handoff_updates", "durable_assertion_candidates",
        "local_scene_facts",
    ],
    "properties": {
        "dialogue_turns": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["speaker", "text"],
                "properties": {
                    "speaker": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                },
            },
        },
        "used_parent_anchor_ids": {
            "type": "array", "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "advanced_event_arc_ids": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "candidate_stage_updates": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "stage_id", "status", "result_state",
                    "evidence_turn_index", "result_speaker",
                ],
                "properties": {
                    "stage_id": {"type": "string", "minLength": 1},
                    "status": {"enum": [
                        "mentioned", "in_progress", "completed", "failed",
                        "carry_forward",
                    ]},
                    "result_state": {"type": ["string", "null"]},
                    "evidence_turn_index": {"type": ["integer", "null"]},
                    "result_speaker": {"type": ["string", "null"]},
                },
            },
        },
        "candidate_handoff_updates": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "handoff_id", "status", "evidence_turn_index",
                    "result_speaker",
                ],
                "properties": {
                    "handoff_id": {"type": "string", "minLength": 1},
                    "status": {"enum": ["resolved", "carry_forward"]},
                    "evidence_turn_index": {"type": ["integer", "null"]},
                    "result_speaker": {"type": ["string", "null"]},
                },
            },
        },
        "durable_assertion_candidates": {"type": "array"},
        "local_scene_facts": {
            "type": "array", "items": {"type": "string"},
        },
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    rows = [
        {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
        for path in sorted(root.rglob("*")) if path.is_file()
    ]
    return canonical_sha256(rows)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _unique_nonempty_strings(value: Any) -> bool:
    return (isinstance(value, list) and bool(value) and
            all(_nonempty(item) for item in value) and
            len(value) == len(set(value)))


def _stage_id(event_id: str, stage_index: int) -> str:
    return f"{event_id}:stage_{stage_index:02d}"


def _normalized_state_value(value: Any) -> str:
    """Canonicalize an explicit state label without semantic inference."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _normalized_marker(value: Any) -> str:
    """Normalize only surface punctuation/spacing for exact repetition."""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def cross_arc_stage_conflict_gate(arcs: Any) -> dict[str, Any]:
    """Reject duplicate base completions and unchained derived states.

    This gate intentionally does no topic similarity or LLM inference.  The
    Planner supplies one stable ``event_identity_key`` per underlying fact;
    the program then enforces exact state/predecessor structure globally.
    """
    diagnostics: list[dict[str, Any]] = []
    if not isinstance(arcs, list):
        return {
            "version": STATEFUL_CROSS_ARC_CONFLICT_GATE_VERSION,
            "valid": False,
            "errors": ["cross_arc_arcs_not_array"],
            "diagnostics": [], "conflicting_stage_refs": [],
            "conflicting_arc_ids": [],
        }
    rows: list[dict[str, Any]] = []
    by_ref: dict[str, dict[str, Any]] = {}
    for arc_position, arc in enumerate(arcs, 1):
        event_id = (str(arc.get("event_id")) if isinstance(arc, Mapping) and
                    arc.get("event_id") else f"E{arc_position}")
        stages = arc.get("stages") if isinstance(arc, Mapping) else None
        if not isinstance(stages, list):
            diagnostics.append({"error": "cross_arc_stages_not_array",
                                "arc_id": event_id})
            continue
        for stage_position, stage in enumerate(stages, 1):
            stage_ref = _stage_id(event_id, stage_position)
            if not isinstance(stage, Mapping):
                diagnostics.append({"error": "cross_arc_stage_not_object",
                                    "conflicting_stage": stage_ref})
                continue
            row = {
                "stage_ref": stage_ref, "arc_id": event_id,
                "event_identity_key": stage.get("event_identity_key"),
                "stage_role": stage.get("stage_role"),
                "predecessor_stage_ref": stage.get("predecessor_stage_ref"),
                "pre_state": stage.get("pre_state"),
                "target_post_state": stage.get("target_post_state"),
                "markers": list(stage.get("completion_evidence_phrases") or []),
            }
            rows.append(row)
            by_ref[stage_ref] = row

    prior_by_identity: dict[str, list[dict[str, Any]]] = {}
    seen_markers: dict[tuple[str, str], str] = {}
    for row in rows:
        ref = row["stage_ref"]
        key = str(row.get("event_identity_key") or "")
        role = row.get("stage_role")
        predecessor_ref = row.get("predecessor_stage_ref")
        if not key or not _ID.fullmatch(key):
            diagnostics.append({"error": "event_identity_key_invalid",
                                "conflicting_stage": ref,
                                "event_identity_key": key})
            continue
        prior_rows = prior_by_identity.setdefault(key, [])
        if role not in STAGE_ROLES:
            diagnostics.append({"error": "stage_role_invalid",
                                "conflicting_stage": ref,
                                "event_identity_key": key})
        elif role == "base_fact_completion":
            if predecessor_ref is not None:
                diagnostics.append({
                    "error": "base_fact_has_predecessor",
                    "event_identity_key": key,
                    "conflicting_stage": ref,
                    "required_action": "remove_predecessor_or_convert_to_derived_state",
                })
            if prior_rows:
                diagnostics.append({
                    "error": "duplicate_completion_obligation",
                    "event_identity_key": key,
                    "existing_stage": prior_rows[0]["stage_ref"],
                    "conflicting_stage": ref,
                    "required_action": "convert_to_derived_state_or_remove_duplicate",
                })
        elif role == "derived_state":
            predecessor = by_ref.get(str(predecessor_ref or ""))
            if (predecessor is None or predecessor not in prior_rows or
                    predecessor.get("event_identity_key") != key):
                diagnostics.append({
                    "error": "derived_state_predecessor_invalid",
                    "event_identity_key": key,
                    "conflicting_stage": ref,
                    "reported_predecessor_stage_ref": predecessor_ref,
                    "required_action": "reference_an_earlier_stage_with_the_same_event_identity",
                })
            elif (_normalized_state_value(row.get("pre_state")) !=
                  _normalized_state_value(predecessor.get("target_post_state"))):
                diagnostics.append({
                    "error": "derived_state_pre_state_mismatch",
                    "event_identity_key": key,
                    "existing_stage": predecessor["stage_ref"],
                    "conflicting_stage": ref,
                    "expected_pre_state": predecessor.get("target_post_state"),
                    "reported_pre_state": row.get("pre_state"),
                    "required_action": "consume_the_predecessor_target_state",
                })
        for prior_row in prior_rows:
            if (_normalized_state_value(row.get("target_post_state")) ==
                    _normalized_state_value(prior_row.get("target_post_state"))):
                diagnostics.append({
                    "error": "duplicate_completion_obligation",
                    "event_identity_key": key,
                    "existing_stage": prior_row["stage_ref"],
                    "conflicting_stage": ref,
                    "required_action": "convert_to_derived_state_or_remove_duplicate",
                })
                break
        for marker in row["markers"]:
            normalized = _normalized_marker(marker)
            marker_key = (key, normalized)
            if normalized and marker_key in seen_markers:
                diagnostics.append({
                    "error": "repeated_base_fact_marker",
                    "event_identity_key": key,
                    "existing_stage": seen_markers[marker_key],
                    "conflicting_stage": ref,
                    "required_action": "write_a_marker_for_the_new_derived_state",
                })
            elif normalized:
                seen_markers[marker_key] = ref
        prior_rows.append(row)

    # Stable de-duplication keeps reports deterministic.
    unique = {canonical_json(row): row for row in diagnostics}
    diagnostics = [unique[key] for key in sorted(unique)]
    conflicting = sorted({str(row.get("conflicting_stage"))
                          for row in diagnostics
                          if row.get("conflicting_stage")})
    conflict_arcs = sorted({ref.split(":", 1)[0] for ref in conflicting})
    return {
        "version": STATEFUL_CROSS_ARC_CONFLICT_GATE_VERSION,
        "valid": not diagnostics,
        "errors": sorted({str(row["error"]) for row in diagnostics}),
        "diagnostics": diagnostics,
        "conflicting_stage_refs": conflicting,
        "conflicting_arc_ids": conflict_arcs,
    }


def _materialize_stage(event_id: str, stage_index: int,
                       stage: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(stage))
    row["stage_id"] = _stage_id(event_id, stage_index)
    row["event_id"] = event_id
    row["outcome_handoffs"] = [
        {**dict(handoff),
         "handoff_id": f"{row['stage_id']}:handoff_{position:02d}"}
        for position, handoff in enumerate(row.get("outcome_handoffs") or [], 1)
    ]
    return row


def validate_stateful_outline(
    outline: Any, *, prior_state_instances: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Validate typed state transitions without judging story difficulty."""
    if not isinstance(outline, dict):
        return ["stateful_outline_not_object"]
    errors: list[str] = []
    if outline.get("schema_version") != STATEFUL_GRAPH_OUTLINE_SCHEMA_VERSION:
        errors.append("stateful_outline_schema_version_invalid")
    if (outline.get("completion_contract_version") !=
            STATEFUL_COMPLETION_CONTRACT_VERSION):
        errors.append("completion_contract_version_invalid")
    arcs = outline.get("planned_event_arcs")
    scenes = outline.get("scene_plan")
    if not isinstance(arcs, list) or len(arcs) != 4:
        errors.append("stateful_outline_requires_four_arcs")
        arcs = []
    if not isinstance(scenes, list) or len(scenes) != 10:
        errors.append("stateful_outline_requires_ten_scenes")
        scenes = []
    if scenes:
        ids = [scene.get("scene_id") for scene in scenes
               if isinstance(scene, dict)]
        if ids != [f"S{index}" for index in range(1, 11)]:
            errors.append("stateful_scene_ids_not_monotonic")
        for scene in scenes:
            if (not isinstance(scene, dict) or
                    not _unique_nonempty_strings(scene.get("participants")) or
                    not isinstance(scene.get("required_parent_anchor_ids"), list) or
                    any(not _nonempty(value) for value in
                        scene.get("required_parent_anchor_ids") or []) or
                    not _nonempty(scene.get("purpose")) or
                    not _nonempty(scene.get("session_location"))):
                errors.append("stateful_scene_schema_invalid")

    prior = {str(key): dict(value) for key, value in
             (prior_state_instances or {}).items()}
    simulated = {key: str(value.get("current_state"))
                 for key, value in prior.items()}
    seen_stage_ids: set[str] = set()
    seen_instances: dict[str, tuple[str, int, str | None]] = {}
    terminal_instances: set[str] = {
        key for key, value in prior.items() if value.get("terminal") is True
    }

    for arc_position, arc in enumerate(arcs, 1):
        if not isinstance(arc, dict) or not _nonempty(arc.get("event_id")):
            errors.append("stateful_arc_schema_invalid")
            continue
        event_id = str(arc["event_id"])
        stages = arc.get("stages")
        if not isinstance(stages, list) or len(stages) != 4:
            errors.append("stateful_outline_requires_four_stages_per_arc")
            continue
        for stage_position, stage in enumerate(stages, 1):
            stage_id = _stage_id(event_id, stage_position)
            if stage_id in seen_stage_ids:
                errors.append("duplicate_stage_id")
            seen_stage_ids.add(stage_id)
            if not isinstance(stage, dict):
                errors.append("stateful_stage_schema_invalid")
                continue
            required = {
                "description", "state_instance_id", "state_key", "object_id",
                "event_identity_key", "stage_role", "predecessor_stage_ref",
                "version", "pre_state", "target_post_state",
                "allowed_post_states", "terminal_post_states",
                "completion_event_type", "mandatory_resolution",
                "authorized_result_speakers", "completion_evidence_phrases",
                "completion_status", "completion_temporality",
                "completion_conditionality",
                "predecessor_state_instance_id", "outcome_handoffs",
            }
            if set(stage) != required:
                errors.append("stateful_stage_fields_invalid")
                continue
            strings = [
                stage.get("description"), stage.get("state_instance_id"),
                stage.get("state_key"), stage.get("object_id"),
                stage.get("event_identity_key"),
                stage.get("pre_state"), stage.get("target_post_state"),
                stage.get("completion_event_type"),
            ]
            if any(not _nonempty(value) for value in strings):
                errors.append("stateful_stage_empty_value")
                continue
            instance_id = str(stage["state_instance_id"])
            if (not _ID.fullmatch(instance_id) or
                    not _ID.fullmatch(str(stage["state_key"])) or
                    not _ID.fullmatch(str(stage["object_id"])) or
                    not _ID.fullmatch(str(stage["event_identity_key"]))):
                errors.append("stateful_identifier_invalid")
            if stage.get("stage_role") not in STAGE_ROLES:
                errors.append("stage_role_invalid")
            if (stage.get("stage_role") == "base_fact_completion" and
                    stage.get("predecessor_stage_ref") is not None):
                errors.append("base_fact_has_predecessor")
            if (stage.get("stage_role") == "derived_state" and
                    not _nonempty(stage.get("predecessor_stage_ref"))):
                errors.append("derived_state_predecessor_invalid")
            if not isinstance(stage.get("version"), int) or stage["version"] < 1:
                errors.append("stateful_version_invalid")
            expected_instance_id = (
                f"{stage['object_id']}:{stage['state_key']}:v{stage['version']}"
            )
            if instance_id != expected_instance_id:
                errors.append("state_instance_id_not_canonical")
            allowed = stage.get("allowed_post_states")
            terminal = stage.get("terminal_post_states")
            if not _unique_nonempty_strings(allowed):
                errors.append("allowed_post_states_invalid")
                allowed = []
            if (not isinstance(terminal, list) or
                    any(not _nonempty(item) for item in terminal) or
                    len(terminal) != len(set(terminal)) or
                    not set(terminal) <= set(allowed)):
                errors.append("terminal_post_states_invalid")
                terminal = []
            if stage.get("target_post_state") not in allowed:
                errors.append("target_post_state_not_allowed")
            if stage.get("target_post_state") not in terminal:
                errors.append("target_post_state_not_terminal")
            if stage.get("pre_state") == stage.get("target_post_state"):
                errors.append("state_transition_is_noop")
            if not isinstance(stage.get("mandatory_resolution"), bool):
                errors.append("mandatory_resolution_not_boolean")
            if not _unique_nonempty_strings(stage.get("authorized_result_speakers")):
                errors.append("authorized_result_speakers_invalid")
            if not _unique_nonempty_strings(stage.get("completion_evidence_phrases")):
                errors.append("completion_evidence_phrases_invalid")
            if stage.get("completion_status") not in COMPLETION_STATUSES:
                errors.append("completion_status_invalid")
            if (stage.get("completion_temporality") not in
                    COMPLETION_TEMPORALITIES):
                errors.append("completion_temporality_invalid")
            if (stage.get("completion_conditionality") not in
                    COMPLETION_CONDITIONALITIES):
                errors.append("completion_conditionality_invalid")
            if (stage.get("mandatory_resolution") is True and
                    stage.get("completion_status") != "achieved"):
                errors.append("mandatory_terminal_completion_not_achieved")
            if (stage.get("mandatory_resolution") is True and
                    stage.get("completion_temporality") != "current_or_past"):
                errors.append("mandatory_terminal_completion_not_current_or_past")
            if (stage.get("mandatory_resolution") is True and
                    stage.get("completion_conditionality") != "unconditional"):
                errors.append("mandatory_terminal_completion_not_unconditional")
            if (stage.get("completion_status") == "achieved" and
                    stage.get("completion_temporality") == "future"):
                errors.append("completion_contract_inconsistent")
            if (stage.get("completion_status") == "achieved" and
                    stage.get("completion_conditionality") == "conditional"):
                errors.append("completion_contract_inconsistent")
            phrases = stage.get("completion_evidence_phrases") or []
            if any(not canonical_marker_is_short_and_speakable(value)
                   for value in phrases):
                errors.append("completion_marker_not_canonical")
            if any(terminal_evidence_is_future_or_conditional(value)
                   for value in phrases if isinstance(value, str)):
                errors.append("completion_marker_contract_inconsistent")
            predecessor = stage.get("predecessor_state_instance_id")
            if predecessor is not None and not _nonempty(predecessor):
                errors.append("predecessor_state_instance_id_invalid")
            if not isinstance(stage.get("outcome_handoffs"), list):
                errors.append("outcome_handoffs_invalid")
            else:
                for handoff in stage["outcome_handoffs"]:
                    handoff_fields = {
                        "handoff_key", "description", "owner",
                        "created_on_post_state", "resolution_evidence_phrases",
                        "authorized_result_speakers",
                    }
                    if (not isinstance(handoff, dict) or
                            set(handoff) != handoff_fields or
                            any(not _nonempty(handoff.get(key)) for key in (
                                "handoff_key", "description", "owner",
                                "created_on_post_state")) or
                            handoff.get("created_on_post_state") not in allowed or
                            not _unique_nonempty_strings(
                                handoff.get("resolution_evidence_phrases")) or
                            not _unique_nonempty_strings(
                                handoff.get("authorized_result_speakers"))):
                        errors.append("outcome_handoff_schema_invalid")

            # State continuity is deterministic.  A later transition on the
            # same instance must consume the exact target state of the earlier
            # transition.  A terminal instance may not be reopened.
            expected_pre = simulated.get(instance_id)
            if instance_id in terminal_instances:
                errors.append("completed_state_instance_reopened")
            if expected_pre is not None and stage.get("pre_state") != expected_pre:
                errors.append("pre_state_ledger_mismatch")
            if expected_pre is None:
                simulated[instance_id] = str(stage.get("pre_state"))
            simulated[instance_id] = str(stage.get("target_post_state"))
            if stage.get("target_post_state") in set(terminal):
                terminal_instances.add(instance_id)

            previous = seen_instances.get(instance_id)
            identity = (str(stage.get("object_id")), int(stage.get("version") or 0),
                        predecessor)
            if previous is not None and previous != identity:
                errors.append("state_instance_identity_drift")
            seen_instances[instance_id] = identity
            if predecessor is not None:
                predecessor_row = prior.get(str(predecessor))
                if predecessor_row is None:
                    errors.append("revision_predecessor_missing")
                else:
                    predecessor_version = int(predecessor_row.get("version") or 0)
                    if stage.get("version") != predecessor_version + 1:
                        errors.append("revision_version_not_monotonic")
                    if predecessor_row.get("terminal") is not True:
                        errors.append("revision_predecessor_not_terminal")

    conflict_gate = cross_arc_stage_conflict_gate(arcs)
    errors.extend(f"cross_arc:{value}" for value in conflict_gate["errors"])
    return sorted(set(errors))


def compile_stateful_scene_contracts(
    outline: Mapping[str, Any], *,
    prior_state_instances: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    errors = validate_stateful_outline(
        outline, prior_state_instances=prior_state_instances)
    if errors:
        raise ValueError("stateful_outline_invalid:" + ",".join(errors))
    arcs = list(outline["planned_event_arcs"])
    scenes = list(outline["scene_plan"])
    anchor_index = {str(row["anchor_id"]): dict(row)
                    for row in outline.get("parent_anchors") or []}
    contracts: list[dict[str, Any]] = []
    for scene_position, (scene, schedule) in enumerate(
            zip(scenes, STANDARD_STAGE_SCHEDULE), 1):
        stages = []
        for arc_position, stage_position in schedule:
            arc = arcs[arc_position]
            stages.append(_materialize_stage(
                str(arc["event_id"]), stage_position + 1,
                arc["stages"][stage_position]))
        required_anchor_ids = list(dict.fromkeys([
            *list(scene.get("required_parent_anchor_ids") or []),
            *(anchor_id for arc_position, _ in schedule
              for anchor_id in arcs[arc_position].get("parent_anchor_ids") or []),
        ]))
        target_token_range = list(scene.get("target_dialogue_token_range") or
                                  scene.get("acceptable_token_range") or
                                  [240, 520])
        target_turn_range = list(scene.get("target_turn_range") or
                                 scene.get("target_turns") or [4, 12])
        if (len(target_token_range) != 2 or
                int(target_token_range[0]) > int(target_token_range[1])):
            raise ValueError("stateful_scene_token_range_invalid")
        if (len(target_turn_range) != 2 or int(target_turn_range[0]) < 1 or
                int(target_turn_range[0]) > int(target_turn_range[1])):
            raise ValueError("stateful_scene_turn_range_invalid")
        contract = {
            "contract_version": STATEFUL_GRAPH_SCENE_CONTRACT_VERSION,
            "completion_contract_version":
                STATEFUL_COMPLETION_CONTRACT_VERSION,
            "scene_id": str(scene["scene_id"]),
            "scene_index": scene_position,
            "purpose": str(scene.get("purpose") or ""),
            "required_participants": list(scene.get("participants") or []),
            "allowed_speakers": list(scene.get("participants") or []),
            "required_parent_anchor_ids": required_anchor_ids,
            "parent_anchor_evidence": [anchor_index[anchor_id]
                                       for anchor_id in required_anchor_ids
                                       if anchor_id in anchor_index],
            "session_location": str(scene.get("session_location") or
                                    scene.get("location") or ""),
            "target_dialogue_token_range": [int(target_token_range[0]),
                                              int(target_token_range[1])],
            "hard_dialogue_token_range": [
                max(1, int(int(target_token_range[0]) * 0.80)),
                int(int(target_token_range[1]) * 1.20),
            ],
            "target_turn_range": [int(target_turn_range[0]),
                                  int(target_turn_range[1])],
            "hard_turn_range": [max(1, int(target_turn_range[0]) - 2),
                                int(target_turn_range[1]) + 3],
            "assigned_event_stages": stages,
            "assigned_event_stage_ids": [stage["stage_id"] for stage in stages],
            "assigned_event_arc_ids": list(dict.fromkeys(
                stage["event_id"] for stage in stages)),
            "is_final_scene": scene_position == len(scenes),
        }
        contract["contract_sha256"] = canonical_sha256(contract)
        contracts.append(contract)
    return contracts


def initialize_stateful_ledger(
    contracts: Sequence[Mapping[str, Any]], *,
    prior_state_instances: Mapping[str, Mapping[str, Any]] | None = None,
    prior_handoffs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    ledger = {
        "ledger_version": STATEFUL_GRAPH_LEDGER_VERSION,
        "state_instances": deepcopy(dict(prior_state_instances or {})),
        "stage_states": {},
        "handoffs": deepcopy(dict(prior_handoffs or {})),
        "accepted_scene_ids": [],
        "transition_history": [],
    }
    for contract in contracts:
        for stage in contract["assigned_event_stages"]:
            instance_id = stage["state_instance_id"]
            existing = ledger["state_instances"].get(instance_id)
            if existing is None:
                ledger["state_instances"][instance_id] = {
                    "state_instance_id": instance_id,
                    "state_key": stage["state_key"],
                    "object_id": stage["object_id"],
                    "version": stage["version"],
                    "predecessor_state_instance_id": stage[
                        "predecessor_state_instance_id"],
                    "current_state": stage["pre_state"],
                    "terminal": False,
                }
            ledger["stage_states"][stage["stage_id"]] = {
                "stage_id": stage["stage_id"],
                "scene_id": contract["scene_id"],
                "state_instance_id": instance_id,
                "mandatory_resolution": stage["mandatory_resolution"],
                "status": "assigned",
                "result_state": None,
                "evidence_turn_index": None,
                "result_speaker": None,
            }
    ledger["ledger_sha256"] = canonical_sha256({
        key: value for key, value in ledger.items() if key != "ledger_sha256"
    })
    return ledger


def validate_stateful_realizer_schema(parsed: Any) -> list[str]:
    if not isinstance(parsed, dict):
        return ["stateful_realizer_schema_invalid"]
    required = set(STATEFUL_REALIZER_JSON_SCHEMA["required"])
    if set(parsed) != required:
        return ["stateful_realizer_fields_invalid"]
    errors: list[str] = []
    turns = parsed.get("dialogue_turns")
    if (not isinstance(turns, list) or not turns or any(
            not isinstance(turn, dict) or set(turn) != {"speaker", "text"} or
            not _nonempty(turn.get("speaker")) or not _nonempty(turn.get("text"))
            for turn in turns)):
        errors.append("stateful_dialogue_turns_invalid")
    for key in ("used_parent_anchor_ids", "advanced_event_arc_ids"):
        values = parsed.get(key)
        if (not isinstance(values, list) or
                any(not _nonempty(value) for value in values) or
                len(values) != len(set(values))):
            errors.append("stateful_realizer_id_array_invalid")
    stage_updates = parsed.get("candidate_stage_updates")
    if not isinstance(stage_updates, list) or not stage_updates:
        errors.append("candidate_stage_updates_invalid")
    else:
        fields = {"stage_id", "status", "result_state",
                  "evidence_turn_index", "result_speaker"}
        for update in stage_updates:
            if (not isinstance(update, dict) or set(update) != fields or
                    not _nonempty(update.get("stage_id")) or
                    update.get("status") not in STAGE_STATUSES - {"assigned"}):
                errors.append("candidate_stage_update_schema_invalid")
    handoff_updates = parsed.get("candidate_handoff_updates")
    if not isinstance(handoff_updates, list):
        errors.append("candidate_handoff_updates_invalid")
    else:
        fields = {"handoff_id", "status", "evidence_turn_index",
                  "result_speaker"}
        for update in handoff_updates:
            if (not isinstance(update, dict) or set(update) != fields or
                    not _nonempty(update.get("handoff_id")) or
                    update.get("status") not in {"resolved", "carry_forward"}):
                errors.append("candidate_handoff_update_schema_invalid")
    if not isinstance(parsed.get("durable_assertion_candidates"), list):
        errors.append("durable_assertion_candidates_invalid")
    facts = parsed.get("local_scene_facts")
    if (not isinstance(facts, list) or
            any(not _nonempty(value) for value in facts)):
        errors.append("local_scene_facts_invalid")
    return sorted(set(errors))


def _indexed_turn(turns: Sequence[Mapping[str, Any]], index: Any) -> Mapping[str, Any] | None:
    if not isinstance(index, int) or isinstance(index, bool):
        return None
    if index < 0 or index >= len(turns):
        return None
    turn = turns[index]
    return turn if isinstance(turn, Mapping) else None


_APOSTROPHE_TRANSLATION = str.maketrans({
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u02bc": "'",  # modifier letter apostrophe
    "\uff07": "'",  # fullwidth apostrophe
})


def normalize_terminal_marker_text(text: str) -> str:
    """Normalize typography only; do not rewrite marker words or semantics."""
    return unicodedata.normalize("NFKC", str(text)).translate(
        _APOSTROPHE_TRANSLATION).casefold()


def _contains_phrase(text: str, phrases: Sequence[str]) -> bool:
    folded = normalize_terminal_marker_text(text)
    return any(normalize_terminal_marker_text(phrase) in folded
               for phrase in phrases)


def _has_terminal_marker_sentence(text: str, phrases: Sequence[str]) -> bool:
    """Require one marker-bearing sentence that is not future/conditional."""
    for sentence in re.split(r"(?<=[.!?])\s+", str(text)):
        if (_contains_phrase(sentence, phrases) and
                not terminal_evidence_is_future_or_conditional(sentence)):
            return True
    return False


def terminal_evidence_phrase_present(
    text: str, phrases: Sequence[str],
) -> bool:
    """Public read-only marker check used by stage-specific diagnostics."""
    return _has_terminal_marker_sentence(text, phrases)


def evidence_phrase_present(
    text: str, phrases: Sequence[str],
) -> bool:
    """Return whether a marker occurs, without weakening terminal semantics."""
    return _contains_phrase(text, phrases)


def _outcome_handoff_records(stage: Mapping[str, Any], result_state: str) -> list[dict[str, Any]]:
    records = []
    for position, handoff in enumerate(stage.get("outcome_handoffs") or [], 1):
        if handoff["created_on_post_state"] != result_state:
            continue
        records.append({
            "handoff_id": str(handoff.get("handoff_id") or
                               f"{stage['stage_id']}:handoff_{position:02d}"),
            "origin_stage_id": stage["stage_id"],
            "state_instance_id": stage["state_instance_id"],
            "handoff_key": handoff["handoff_key"],
            "description": handoff["description"],
            "owner": handoff["owner"],
            "status": "pending",
            "resolution_evidence_phrases": list(
                handoff["resolution_evidence_phrases"]),
            "authorized_result_speakers": list(
                handoff["authorized_result_speakers"]),
            "evidence_turn_index": None,
            "result_speaker": None,
        })
    return records


def validate_candidate_stage_updates(
    parsed: Mapping[str, Any], contract: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate API proposals; return accepted updates without mutating ledger."""
    errors: list[str] = []
    turns = parsed.get("dialogue_turns") or []
    updates = parsed.get("candidate_stage_updates") or []
    stage_index = {stage["stage_id"]: stage
                   for stage in contract["assigned_event_stages"]}
    by_id: dict[str, Mapping[str, Any]] = {}
    for update in updates:
        stage_id = str(update.get("stage_id") or "")
        if stage_id in by_id:
            errors.append("duplicate_stage_update")
        by_id[stage_id] = update
    if set(by_id) != set(stage_index):
        errors.append("assigned_stage_update_set_mismatch")

    accepted: list[dict[str, Any]] = []
    prospective_handoffs: list[dict[str, Any]] = []
    for stage_id, stage in stage_index.items():
        update = by_id.get(stage_id)
        if update is None:
            continue
        existing = (ledger.get("stage_states") or {}).get(stage_id)
        instance = (ledger.get("state_instances") or {}).get(
            stage["state_instance_id"])
        if existing is None or instance is None:
            errors.append("stage_or_state_instance_missing")
            continue
        if existing.get("status") in TERMINAL_STAGE_STATUSES:
            errors.append("terminal_stage_repeated")
        if instance.get("terminal") is True:
            errors.append("completed_state_instance_reopened")
        if instance.get("current_state") != stage["pre_state"]:
            errors.append("pre_state_ledger_mismatch")
        status = update.get("status")
        evidence = _indexed_turn(turns, update.get("evidence_turn_index"))
        speaker = update.get("result_speaker")
        if status == "carry_forward":
            if (update.get("result_state") is not None or
                    update.get("evidence_turn_index") is not None or
                    update.get("result_speaker") is not None):
                errors.append("carry_forward_must_not_claim_evidence")
        else:
            if evidence is None:
                errors.append("stage_evidence_turn_invalid")
            elif speaker != evidence.get("speaker"):
                errors.append("stage_evidence_speaker_mismatch")
            if speaker not in stage["authorized_result_speakers"]:
                errors.append("stage_result_speaker_unauthorized")
        if status in TERMINAL_STAGE_STATUSES:
            if update.get("result_state") != stage["target_post_state"]:
                errors.append("stage_result_state_mismatch")
            if evidence is not None:
                text = str(evidence.get("text") or "")
                if not _contains_phrase(text, stage["completion_evidence_phrases"]):
                    errors.append("terminal_evidence_marker_missing")
                elif not _has_terminal_marker_sentence(
                        text, stage["completion_evidence_phrases"]):
                    errors.append("terminal_evidence_is_future_or_conditional")
            if update.get("result_state") not in stage["allowed_post_states"]:
                errors.append("stage_result_not_allowed")
            prospective_handoffs.extend(_outcome_handoff_records(
                stage, str(update.get("result_state"))))
        else:
            if update.get("result_state") is not None:
                errors.append("nonterminal_stage_cannot_change_state")
        accepted.append(deepcopy(dict(update)))
    return {"errors": sorted(set(errors)),
            "accepted_stage_updates": accepted if not errors else [],
            "prospective_handoffs": prospective_handoffs if not errors else []}


def validate_candidate_handoff_updates(
    parsed: Mapping[str, Any], ledger: Mapping[str, Any], *,
    prospective_handoffs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    errors: list[str] = []
    turns = parsed.get("dialogue_turns") or []
    available = {str(key): deepcopy(dict(value)) for key, value in
                 (ledger.get("handoffs") or {}).items()}
    available.update({str(row["handoff_id"]): deepcopy(dict(row))
                      for row in prospective_handoffs})
    updates = parsed.get("candidate_handoff_updates") or []
    by_id: dict[str, Mapping[str, Any]] = {}
    for update in updates:
        handoff_id = str(update.get("handoff_id") or "")
        if handoff_id in by_id:
            errors.append("duplicate_handoff_update")
        by_id[handoff_id] = update
    accepted: list[dict[str, Any]] = []
    for handoff_id, update in by_id.items():
        record = available.get(handoff_id)
        if record is None:
            errors.append("unknown_handoff_id")
            continue
        if record.get("status") == "resolved":
            errors.append("resolved_handoff_reopened")
        status = update.get("status")
        if status == "carry_forward":
            if (update.get("evidence_turn_index") is not None or
                    update.get("result_speaker") is not None):
                errors.append("handoff_carry_forward_must_not_claim_evidence")
        elif status == "resolved":
            evidence = _indexed_turn(turns, update.get("evidence_turn_index"))
            if evidence is None:
                errors.append("handoff_evidence_turn_invalid")
            else:
                if update.get("result_speaker") != evidence.get("speaker"):
                    errors.append("handoff_evidence_speaker_mismatch")
                evidence_text = str(evidence.get("text") or "")
                if not _contains_phrase(
                        evidence_text, record["resolution_evidence_phrases"]):
                    errors.append("handoff_resolution_marker_missing")
                elif not _has_terminal_marker_sentence(
                        evidence_text, record["resolution_evidence_phrases"]):
                    errors.append("handoff_evidence_is_future_or_conditional")
            if update.get("result_speaker") not in record[
                    "authorized_result_speakers"]:
                errors.append("handoff_result_speaker_unauthorized")
        accepted.append(deepcopy(dict(update)))
    return {"errors": sorted(set(errors)),
            "accepted_handoff_updates": accepted if not errors else []}


def apply_authoritative_scene_updates(
    ledger: Mapping[str, Any], contract: Mapping[str, Any],
    stage_updates: Sequence[Mapping[str, Any]],
    handoff_updates: Sequence[Mapping[str, Any]],
    prospective_handoffs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply already-validated updates and return a new immutable-style ledger."""
    result = deepcopy(dict(ledger))
    stage_index = {stage["stage_id"]: stage
                   for stage in contract["assigned_event_stages"]}
    for update in stage_updates:
        stage_id = str(update["stage_id"])
        stage = stage_index[stage_id]
        state = result["stage_states"][stage_id]
        state.update(deepcopy(dict(update)))
        if update["status"] in TERMINAL_STAGE_STATUSES:
            instance = result["state_instances"][stage["state_instance_id"]]
            old_state = instance["current_state"]
            instance["current_state"] = update["result_state"]
            instance["terminal"] = update["result_state"] in set(
                stage["terminal_post_states"])
            result["transition_history"].append({
                "scene_id": contract["scene_id"], "stage_id": stage_id,
                "state_instance_id": stage["state_instance_id"],
                "pre_state": old_state, "post_state": update["result_state"],
                "status": update["status"],
            })
    for handoff in prospective_handoffs:
        handoff_id = str(handoff["handoff_id"])
        if handoff_id in result["handoffs"]:
            raise ValueError("duplicate_created_handoff_id")
        result["handoffs"][handoff_id] = deepcopy(dict(handoff))
    for update in handoff_updates:
        record = result["handoffs"][str(update["handoff_id"])]
        record.update(deepcopy(dict(update)))
    if contract["scene_id"] not in result["accepted_scene_ids"]:
        result["accepted_scene_ids"].append(contract["scene_id"])
    result["ledger_sha256"] = canonical_sha256({
        key: value for key, value in result.items() if key != "ledger_sha256"
    })
    return result


def validate_and_apply_stateful_scene(
    parsed: Mapping[str, Any], contract: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    errors = validate_stateful_realizer_schema(parsed)
    if set(parsed.get("advanced_event_arc_ids") or []) != set(
            contract["assigned_event_arc_ids"]):
        errors.append("stateful_assigned_event_arcs_incomplete")
    if not set(contract["required_parent_anchor_ids"]) <= set(
            parsed.get("used_parent_anchor_ids") or []):
        errors.append("stateful_required_parent_anchor_missing")
    speakers = {turn.get("speaker") for turn in parsed.get("dialogue_turns") or []
                if isinstance(turn, dict)}
    if not set(contract["required_participants"]) <= speakers:
        errors.append("stateful_required_participant_missing")
    if not speakers <= set(contract.get("allowed_speakers") or
                           contract["required_participants"]):
        errors.append("stateful_unauthorized_speaker")
    stage_result = validate_candidate_stage_updates(parsed, contract, ledger)
    errors.extend(stage_result["errors"])
    handoff_result = validate_candidate_handoff_updates(
        parsed, ledger,
        prospective_handoffs=stage_result["prospective_handoffs"])
    errors.extend(handoff_result["errors"])
    if contract.get("is_final_scene") is True:
        required_handoffs = {
            str(handoff_id) for handoff_id, row in
            (ledger.get("handoffs") or {}).items()
            if row.get("status") != "resolved"
        }
        required_handoffs.update(
            str(row["handoff_id"])
            for row in stage_result["prospective_handoffs"])
        reported_handoffs = {
            str(row.get("handoff_id")) for row in
            handoff_result["accepted_handoff_updates"]
        }
        if not required_handoffs <= reported_handoffs:
            errors.append("final_scene_handoff_accounting_incomplete")
    if errors:
        return {"decision": "reject_stateful_scene",
                "errors": sorted(set(errors)), "ledger": deepcopy(dict(ledger))}
    next_ledger = apply_authoritative_scene_updates(
        ledger, contract, stage_result["accepted_stage_updates"],
        handoff_result["accepted_handoff_updates"],
        stage_result["prospective_handoffs"])
    return {"decision": "accept_stateful_scene", "errors": [],
            "accepted_stage_updates": stage_result["accepted_stage_updates"],
            "accepted_handoff_updates": handoff_result[
                "accepted_handoff_updates"],
            "created_handoffs": stage_result["prospective_handoffs"],
            "ledger": next_ledger}


def stateful_completion_report(
    contracts: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any],
) -> dict[str, Any]:
    expected_scene_ids = [str(contract["scene_id"]) for contract in contracts]
    expected_stage_ids = [str(stage["stage_id"]) for contract in contracts
                          for stage in contract["assigned_event_stages"]]
    accepted_scene_ids = list(ledger.get("accepted_scene_ids") or [])
    stage_states = ledger.get("stage_states") or {}
    statuses = {stage_id: (stage_states.get(stage_id) or {}).get("status")
                for stage_id in expected_stage_ids}
    mandatory = [stage_id for stage_id in expected_stage_ids
                 if (stage_states.get(stage_id) or {}).get(
                     "mandatory_resolution") is True]
    unresolved_handoffs = [
        handoff_id for handoff_id, row in (ledger.get("handoffs") or {}).items()
        if row.get("status") == "pending"
    ]
    layers = {
        "scene_realization_complete": accepted_scene_ids == expected_scene_ids,
        "stage_coverage_complete": all(
            status in STAGE_STATUSES - {"assigned"} for status in statuses.values()),
        "semantic_transition_complete": all(
            statuses.get(stage_id) in TERMINAL_STAGE_STATUSES
            for stage_id in mandatory),
        "handoff_accounting_complete": not unresolved_handoffs,
    }
    counts = {status: sum(value == status for value in statuses.values())
              for status in sorted(STAGE_STATUSES)}
    counts["resolved_handoffs"] = sum(
        row.get("status") == "resolved"
        for row in (ledger.get("handoffs") or {}).values())
    counts["carry_forward_handoffs"] = sum(
        row.get("status") == "carry_forward"
        for row in (ledger.get("handoffs") or {}).values())
    counts["pending_handoffs"] = len(unresolved_handoffs)
    overall = (layers["scene_realization_complete"] and
               layers["semantic_transition_complete"] and
               layers["handoff_accounting_complete"])
    return {
        "graph_expansion_version": STATEFUL_GRAPH_EXPANSION_VERSION,
        "ledger_version": STATEFUL_GRAPH_LEDGER_VERSION,
        "layers": layers, "counts": counts,
        "expected_scene_ids": expected_scene_ids,
        "accepted_scene_ids": accepted_scene_ids,
        "expected_stage_ids": expected_stage_ids,
        "stage_status_by_id": statuses,
        "unresolved_handoff_ids": sorted(unresolved_handoffs),
        "complete": overall,
    }


def build_stateful_graph_realizer_prompt(
    *, contract: Mapping[str, Any], recent_sessions: Sequence[Mapping[str, Any]],
    active_handoffs: Sequence[Mapping[str, Any]],
) -> str:
    """Render the v2 Realizer contract; no Questioner or answer data appears."""
    allowed_handoff_ids = sorted(
        str(row["handoff_id"]) for row in active_handoffs
        if row.get("handoff_id"))
    rules = [
        "Realize exactly one frozen Graph scene as natural dialogue.",
        "Return one JSON object matching the schema exactly; no Markdown.",
        "Assigned never means completed. Return one candidate_stage_updates row for every assigned stage.",
        "Use mentioned or in_progress when the target state has not actually been reached.",
        "Use carry_forward when the scene deliberately leaves the stage unresolved; do not attach evidence or a result state.",
        "Use completed or failed only when one evidence turn explicitly establishes the frozen target_post_state.",
        "A terminal evidence turn must use an authorized speaker and contain one supplied completion_evidence_phrase.",
        "A plan is terminal only when the frozen target state explicitly represents plan creation or finalization; otherwise plans, intentions, readiness statements, future conditions, and waiting states are nonterminal.",
        "Do not output or paraphrase bookkeeping fields in the dialogue.",
        "Use only allowed_speakers, and make every required_participant speak meaningfully.",
        "The dialogue_turns text itself must meet target_dialogue_token_range and target_turn_range from the contract; metadata does not count toward either range.",
        "Never return fewer turns or less dialogue text than the hard ranges in the contract. Develop the scene naturally instead of compressing it into a short exchange.",
        "Do not invent stage IDs, handoff IDs, state values, speakers, locations, questions, options, or answers.",
        "You may only use a handoff_id listed in allowed_handoff_ids. Never invent, rename, normalize, or approximately match a handoff identifier.",
        "If allowed_handoff_ids is empty, candidate_handoff_updates must be an empty array; do not output a handoff object with a null, omitted, or fabricated handoff_id.",
        "candidate_handoff_updates may resolve or explicitly carry forward only supplied active handoffs. Outcome-created handoffs are applied by the program and must not be echoed as candidate updates in the same scene.",
    ]
    payload = {"scene_contract": contract,
               "recent_accepted_sessions": list(recent_sessions)[-2:],
               "active_handoffs": list(active_handoffs),
               "allowed_handoff_ids": allowed_handoff_ids}
    return "\n\n".join([
        f"PROMPT_VERSION: {STATEFUL_GRAPH_REALIZER_PROMPT_VERSION}",
        "RULES\n- " + "\n- ".join(rules),
        "FROZEN INPUT\n" + json.dumps(payload, ensure_ascii=False,
                                        sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            STATEFUL_REALIZER_JSON_SCHEMA, ensure_ascii=False,
            sort_keys=True, indent=2),
    ])


def legacy_round_reclassification(round_root: Path) -> dict[str, Any]:
    """Reclassify one Graph v1 round without mutating any historical artifact."""
    outline_path = round_root / "frozen/expansion_outline.json"
    contracts_path = round_root / "frozen/graph_scene_contracts.json"
    hooks_path = round_root / "unresolved_hooks.json"
    session_paths = sorted((round_root / "accepted").glob("session_*.json"))
    for path in (outline_path, contracts_path, hooks_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not session_paths:
        raise RuntimeError("legacy_round_has_no_accepted_sessions")
    outline = json.loads(outline_path.read_text())
    contracts_blob = json.loads(contracts_path.read_text())
    contracts = contracts_blob.get("contracts") or contracts_blob
    sessions = [json.loads(path.read_text()) for path in session_paths]
    hooks = json.loads(hooks_path.read_text())
    expected_stages = [
        stage_id for contract in contracts
        for stage_id in contract.get("assigned_event_stage_ids") or []
    ]
    reported_completed = {
        str(stage_id) for session in sessions
        for stage_id in session.get("completed_event_stage_ids") or []
    }
    stage_rows = [{
        "stage_id": stage_id,
        "legacy_reported_status": (
            "completed" if stage_id in reported_completed else "not_reported"),
        "reclassified_status": "unknown_legacy",
        "reason": "graph_v1_copied_assigned_stage_ids_into_completed_event_stage_ids",
        "eligible_as_v2_state_seed": False,
    } for stage_id in expected_stages]
    handoffs = [{
        "handoff_id": str(row.get("hook_id")),
        "description": str(row.get("description") or ""),
        "status": "pending",
        "source": "legacy_unresolved_hooks",
    } for row in hooks]
    hook_tracking = []
    for handoff in handoffs:
        hook_id = handoff["handoff_id"]
        created_in = [str(session.get("session_id") or path.stem)
                      for session, path in zip(sessions, session_paths)
                      if hook_id in {str(row.get("hook_id")) for row in
                                     session.get("new_unresolved_hooks") or []
                                     if isinstance(row, dict)}]
        continued_in = [str(session.get("session_id") or path.stem)
                        for session, path in zip(sessions, session_paths)
                        if hook_id in set(session.get(
                            "continued_unresolved_hook_ids") or [])]
        resolved_in = [str(session.get("session_id") or path.stem)
                       for session, path in zip(sessions, session_paths)
                       if hook_id in set(session.get("resolved_hook_ids") or [])]
        hook_tracking.append({"handoff_id": hook_id,
                              "created_in_sessions": created_in,
                              "continued_in_sessions": continued_in,
                              "resolved_in_sessions": resolved_in})
    layers = {
        "scene_realization_complete": len(sessions) == len(contracts),
        "stage_coverage_complete": reported_completed == set(expected_stages),
        "semantic_transition_complete": False,
        "handoff_accounting_complete": False,
    }
    return {
        "migration_version": LEGACY_MIGRATION_VERSION,
        "source_round_root": str(round_root.resolve()),
        "source_files": {
            str(path.relative_to(round_root)): file_sha256(path)
            for path in [outline_path, contracts_path, hooks_path, *session_paths]
        },
        "source_outline_schema_version": outline.get("schema_version"),
        "accepted_scene_count": len(sessions),
        "expected_stage_count": len(expected_stages),
        "legacy_reported_completed_stage_count": len(reported_completed),
        "v2_authoritative_completed_stage_count": 0,
        "stage_states": stage_rows,
        "handoffs": handoffs,
        "legacy_hook_tracking": hook_tracking,
        "layers": layers,
        "legacy_round_complete_reclassified": False,
        "v2_seed_policy": (
            "Do not seed a new state ledger from legacy completed_event_stage_ids; "
            "only explicitly audited state transitions may be imported."
        ),
    }
