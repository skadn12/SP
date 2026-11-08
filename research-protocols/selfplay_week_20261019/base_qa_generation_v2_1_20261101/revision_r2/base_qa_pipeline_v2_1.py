"""Versioned V2 core repairs for source admission, option audits and blind review.

The module is offline. It makes no model, network, Solver or training calls.
Semantic quality still requires independent review of the complete source.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

try:
    from . import base_qa_pipeline_v1 as v1
    from . import base_qa_pipeline_v2 as v2
except ImportError:
    import base_qa_pipeline_v1 as v1
    import base_qa_pipeline_v2 as v2

VERSION = "base_qa_generation_v2_1r2"
BLIND_SCHEMA = "base_qa_blind_response_v2_1"
BLIND_FIELDS = frozenset({
    "schema_version", "candidate_id", "packet_sha256", "source_scope_sha256",
    "reviewer_id", "model_id", "reviewer_mode", "decision", "independent_gold",
    "evidence", "reasoning_steps", "ambiguities", "source_limitations",
    "isolation_receipt",
})
RECEIPT_SCHEMA = "base_qa_material_admission_v2_1"
ISOLATION_SCHEMA = "base_qa_orchestration_receipt_v2_1"
PROVENANCE_TYPES = frozenset({
    "original_root_dialogue", "old_route_expansion", "new_event_seed_migration",
    "codex_synthetic_extension",
})
_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:\.\d+)?")
_NUMBER_RE = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
_CUE_RE = re.compile(
    r"\b(?:only|exactly|minimum|maximum|least|most|smallest|largest|"
    r"earliest|latest|shortest|longest|never|always|still|already|"
    r"cancelled|canceled|possibly|perhaps|at least|at most|no longer)\b",
    re.IGNORECASE,
)
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "before", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "is", "it",
    "of", "on", "or", "the", "their", "there", "this", "to", "was", "were",
    "what", "when", "which", "who", "with",
})


def _canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_sha256(source, binding):
    if isinstance(binding, dict) and _is_sha256(binding.get("sha256")):
        return binding["sha256"]
    return hashlib.sha256(_canonical_json_bytes(source)).hexdigest()


def _verify_source_binding(source, binding):
    if not isinstance(binding, dict):
        return True, None
    if not isinstance(binding.get("path"), str):
        return False, "SOURCE_BINDING_PATH_MISSING"
    try:
        raw = Path(binding["path"]).read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        if binding.get("sha256") != actual_hash or binding.get("bytes") != len(raw):
            return False, "SOURCE_BINDING_DRIFT"
        if json.loads(raw.decode("utf-8")) != source:
            return False, "SOURCE_BINDING_OBJECT_MISMATCH"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False, "SOURCE_BINDING_UNREADABLE"
    return True, None


def _normalized_provenance(value):
    if not isinstance(value, str):
        return None
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "source_grounded": "original_root_dialogue",
        "original_root_dialogue": "original_root_dialogue",
        "old_route_expansion": "old_route_expansion",
        "legacy_expansion": "old_route_expansion",
        "new_event_seed_migration": "new_event_seed_migration",
        "seed_migration": "new_event_seed_migration",
        "codex_session_generated": "codex_synthetic_extension",
        "codex_synthetic_extension": "codex_synthetic_extension",
        "generated_extension": "generated_extension_unclassified",
    }
    return aliases.get(value)


def _verified_json_binding(binding, expected_object):
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        return False, "RECEIPT_BINDING_MISSING"
    try:
        raw = Path(binding["path"]).read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        if binding.get("sha256") != actual_hash or binding.get("bytes") != len(raw):
            return False, "RECEIPT_BINDING_DRIFT"
        decoded = json.loads(raw.decode("utf-8"))
        if decoded != expected_object:
            return False, "RECEIPT_OBJECT_MISMATCH"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False, "RECEIPT_BINDING_UNREADABLE"
    return True, None


def _read_verified_json_binding(binding):
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        return None, "REVIEW_RECORD_BINDING_MISSING"
    try:
        raw = Path(binding["path"]).read_bytes()
        if binding.get("sha256") != hashlib.sha256(raw).hexdigest() or binding.get("bytes") != len(raw):
            return None, "REVIEW_RECORD_BINDING_DRIFT"
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None, "REVIEW_RECORD_BINDING_UNREADABLE"
    if not isinstance(value, dict):
        return None, "REVIEW_RECORD_OBJECT_REQUIRED"
    return value, None


def source_manifest(source, source_binding=None, admission=None,
                    admission_binding=None, challenger_id=None):
    """Build a traceable source scope; no inline/self-authored PASS is trusted.

    A formal-use grant must come from a separately frozen, source-hash-bound
    material-admission receipt with complete turn coverage and a distinct reviewer.
    """
    errors = []
    source_bound, source_binding_error = _verify_source_binding(source, source_binding)
    if not source_bound:
        errors.append(source_binding_error)
    source_sha = _source_sha256(source, source_binding)
    turns = source.get("turns") if isinstance(source, dict) else None
    if not isinstance(turns, list) or not turns:
        turns = []
        errors.append("SOURCE_TURNS_INVALID")
    refs = [t.get("turn_ref") for t in turns if isinstance(t, dict)]
    if len(refs) != len(turns) or any(not isinstance(r, str) or not r for r in refs):
        errors.append("SOURCE_TURN_REF_INVALID")
    if len(set(refs)) != len(refs):
        errors.append("SOURCE_TURN_REF_DUPLICATE")

    receipt_errors = []
    receipt_ok = isinstance(admission, dict)
    if not receipt_ok:
        receipt_errors.append("SOURCE_ADMISSION_MISSING")
    else:
        if source_binding is None:
            receipt_errors.append("SOURCE_BINDING_REQUIRED_FOR_ADMISSION")
        bound, binding_error = _verified_json_binding(admission_binding, admission)
        if not bound:
            receipt_errors.append(binding_error)
        if admission.get("schema_version") != RECEIPT_SCHEMA:
            receipt_errors.append("SOURCE_ADMISSION_SCHEMA")
        if admission.get("review_scope") != "formal_qa_material_admission":
            receipt_errors.append("SOURCE_ADMISSION_SCOPE")
        if admission.get("decision") != "READY":
            receipt_errors.append("SOURCE_ADMISSION_NOT_READY")
        if admission.get("source_sha256") != source_sha:
            receipt_errors.append("SOURCE_ADMISSION_SOURCE_HASH_MISMATCH")
        expected_source_id = source.get("source_id", source.get("root_id"))
        if admission.get("source_id") != expected_source_id:
            receipt_errors.append("SOURCE_ADMISSION_SOURCE_ID_MISMATCH")
        reviewer_id = admission.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            receipt_errors.append("SOURCE_ADMISSION_REVIEWER_MISSING")
        if not isinstance(challenger_id, str) or not challenger_id.strip():
            receipt_errors.append("CHALLENGER_ID_REQUIRED")
        elif reviewer_id == challenger_id:
            receipt_errors.append("SOURCE_ADMISSION_SELF_APPROVAL")

    admitted_rows = {}
    if isinstance(admission, dict) and isinstance(admission.get("turns"), list):
        for row in admission["turns"]:
            if not isinstance(row, dict):
                receipt_errors.append("SOURCE_ADMISSION_ROW_INVALID")
                continue
            turn_id = row.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id or turn_id in admitted_rows:
                receipt_errors.append("SOURCE_ADMISSION_ROW_ID_INVALID")
                continue
            admitted_rows[turn_id] = row
    elif isinstance(admission, dict):
        receipt_errors.append("SOURCE_ADMISSION_TURNS_INVALID")
    if set(admitted_rows) != set(refs):
        receipt_errors.append("SOURCE_ADMISSION_COVERAGE_MISMATCH")

    source_id = source.get("source_id", source.get("root_id")) if isinstance(source, dict) else None
    manifest_rows = []
    known_hash = bool(source_binding is None or _is_sha256(source_sha))
    receipt_global_valid = not receipt_errors and not errors and known_hash
    version_default = (source.get("source_version") if isinstance(source, dict) else None) or source_sha
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        turn_id = turn.get("turn_ref")
        row = admitted_rows.get(turn_id, {})
        raw_provenance = turn.get("provenance_type", turn.get("origin"))
        raw_class = _normalized_provenance(raw_provenance)
        declared_class = _normalized_provenance(row.get("provenance_type"))
        if raw_class == "generated_extension_unclassified":
            provenance = declared_class if declared_class in {
                "old_route_expansion", "new_event_seed_migration"} else raw_class
        else:
            provenance = raw_class
        version = turn.get("source_version") or version_default
        reviewer_id = row.get("reviewer_id")
        row_errors = []
        if not receipt_global_valid:
            row_errors.extend(receipt_errors or errors or ["SOURCE_ADMISSION_INVALID"])
        if provenance not in PROVENANCE_TYPES:
            row_errors.append("PROVENANCE_UNKNOWN_OR_UNCLASSIFIED")
        if declared_class != provenance:
            row_errors.append("PROVENANCE_CLASS_MISMATCH")
        if row.get("source_id", source_id) != source_id:
            row_errors.append("SOURCE_ID_MISMATCH")
        if row.get("source_version") != version:
            row_errors.append("SOURCE_VERSION_MISMATCH")
        if row.get("review_status") not in ("PASS", "WARN", "HOLD"):
            row_errors.append("MATERIAL_REVIEW_STATUS_INVALID")
        if not isinstance(row.get("formal_qa_allowed"), bool):
            row_errors.append("FORMAL_QA_ALLOWANCE_MISSING")
        if not isinstance(row.get("review_record"), str) or not row.get("review_record", "").strip():
            row_errors.append("MATERIAL_REVIEW_RECORD_MISSING")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            row_errors.append("MATERIAL_REVIEWER_MISSING")
        if isinstance(admission, dict) and reviewer_id != admission.get("reviewer_id"):
            row_errors.append("MATERIAL_REVIEWER_MISMATCH")
        if reviewer_id == challenger_id:
            row_errors.append("MATERIAL_SELF_APPROVAL")
        record, record_error = _read_verified_json_binding(row.get("review_record_binding"))
        if record_error:
            row_errors.append(record_error)
        elif row.get("review_record") != row["review_record_binding"].get("path"):
            row_errors.append("MATERIAL_REVIEW_RECORD_PATH_MISMATCH")
        else:
            expected_record = {
                "schema_version": "base_qa_turn_material_review_v2_1",
                "source_sha256": source_sha,
                "source_id": source_id,
                "turn_id": turn_id,
                "provenance_type": provenance,
                "source_version": version,
                "reviewer_id": reviewer_id,
                "review_status": row.get("review_status"),
                "formal_qa_allowed": row.get("formal_qa_allowed"),
            }
            if any(record.get(key) != value for key, value in expected_record.items()):
                row_errors.append("MATERIAL_REVIEW_RECORD_CONTENT_MISMATCH")
        eligible = (not row_errors and row.get("review_status") == "PASS"
                    and row.get("formal_qa_allowed") is True)
        manifest_rows.append({
            "source_id": row.get("source_id", source_id),
            "turn_id": turn_id,
            "source_provenance_label": raw_provenance if isinstance(raw_provenance, str) else "unknown",
            "provenance_type": provenance or "unknown",
            "source_version": version,
            "material_review_status": row.get("review_status", "UNVERIFIED"),
            "material_review_record": row.get("review_record"),
            "material_reviewer_id": reviewer_id,
            "formal_qa_allowed": eligible,
            "eligibility_errors": sorted(set(row_errors)),
        })
    admission_complete = bool(manifest_rows) and receipt_global_valid and all(
        not row["eligibility_errors"] for row in manifest_rows)
    row_admission_errors = [error for row in manifest_rows for error in row["eligibility_errors"]]
    return {
        "schema_version": VERSION + ":source_scope",
        "root_id": source.get("root_id") if isinstance(source, dict) else None,
        "source_id": source_id,
        "source_binding": source_binding,
        "source_sha256": source_sha,
        "admission_binding": admission_binding,
        "admission_sha256": (admission_binding or {}).get("sha256"),
        "admission_decision": admission.get("decision") if isinstance(admission, dict) else "UNVERIFIED",
        "admission_review_scope": admission.get("review_scope") if isinstance(admission, dict) else "UNVERIFIED",
        "admission_errors": sorted(set(receipt_errors + errors + row_admission_errors)),
        "turns": manifest_rows,
        "formal_qa_eligible_turn_refs": sorted(r["turn_id"] for r in manifest_rows if r["formal_qa_allowed"]),
        "source_admission_complete": admission_complete,
        "formal_qa_eligible": bool(manifest_rows) and all(r["formal_qa_allowed"] for r in manifest_rows),
        "eligibility_policy": "complete hash-bound independent formal QA material admission; unknown or unapproved is ineligible",
    }


def formal_evidence_gate(source_scope, turn_refs):
    if not isinstance(turn_refs, (list, tuple, set)) or any(not isinstance(ref, str) for ref in turn_refs):
        return {"eligible": False, "errors": ["EVIDENCE_REFS_INVALID"], "ineligible_turn_refs": []}
    if not isinstance(source_scope, dict):
        return {"eligible": False, "errors": ["SOURCE_SCOPE_INVALID"], "ineligible_turn_refs": []}
    if not turn_refs:
        return {"eligible": False, "errors": ["EVIDENCE_REFS_EMPTY"], "ineligible_turn_refs": []}
    rows = {row.get("turn_id"): row for row in source_scope.get("turns", [])
            if isinstance(row, dict) and isinstance(row.get("turn_id"), str)}
    missing = sorted(set(turn_refs) - set(rows))
    denied = sorted(ref for ref in set(turn_refs) & set(rows)
                    if rows[ref].get("formal_qa_allowed") is not True)
    errors = []
    if not source_scope.get("source_admission_complete"):
        errors.append("SOURCE_SCOPE_ADMISSION_INCOMPLETE")
    if missing:
        errors.append("EVIDENCE_TURN_NOT_IN_SOURCE")
    if denied:
        errors.append("EVIDENCE_TURN_NOT_FORMALLY_ALLOWED")
    return {"eligible": not errors, "errors": errors,
            "ineligible_turn_refs": sorted(set(missing + denied))}


def _text_features(value):
    words = _WORD_RE.findall(value)
    tokens = {w.lower() for w in words if w.lower() not in _STOP_WORDS and not w.isdigit()}
    numbers = [m.group(0).replace(",", "") for m in _NUMBER_RE.finditer(value)]
    try:
        numeric_values = [float(n) for n in numbers]
    except ValueError:
        numeric_values = []
    return {
        "word_count": len(words),
        "character_count": len(value),
        "content_tokens": tokens,
        "numbers": numbers,
        "numeric_values": numeric_values,
        "digit_widths": [len(re.sub(r"\D", "", n)) for n in numbers],
        "decimal_places": [len(n.partition(".")[2]) for n in numbers],
        "semantic_cues": sorted(set(m.group(0).lower() for m in _CUE_RE.finditer(value))),
        "clause_markers": len(re.findall(r"[,;:]|\b(?:and|or|but|while|because)\b", value, re.I)),
        "punctuation_count": len(re.findall(r"[^\w\s]", value)),
    }


def style_audit(candidate):
    """Detect option-only signals and cross-option outliers; flags require review."""
    options = candidate.get("options") if isinstance(candidate, dict) else None
    if not isinstance(options, dict) or len(options) < 2 or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in options.items()):
        return {"decision": "HOLD_INVALID_OPTIONS", "flags": ["OPTIONS_INVALID"],
                "features": {}, "static_shortcuts_proved_absent": False}
    question = candidate.get("question", "")
    question = question if isinstance(question, str) else ""
    gold = candidate.get("gold", [])
    if (not isinstance(gold, list) or not gold
            or any(not isinstance(key, str) for key in gold)
            or len(gold) != len(set(gold)) or not set(gold) <= set(options)):
        return {"decision": "HOLD_INVALID_GOLD", "flags": ["GOLD_INVALID"],
                "features": {}, "static_shortcuts_proved_absent": False}
    features = {key: _text_features(value) for key, value in options.items()}
    flags = []
    risks = []
    counts = {key: row["word_count"] for key, row in features.items()}
    char_counts = {key: row["character_count"] for key, row in features.items()}
    question_type = candidate.get("question_type")
    if question_type not in ("single_choice", "multi_select", "ordering"):
        return {"decision": "HOLD_INVALID_QUESTION_TYPE", "flags": ["QUESTION_TYPE_INVALID"],
                "features": {}, "static_shortcuts_proved_absent": False}
    gold_set = set(gold)
    distractor_set = set(options) - gold_set
    if min(counts.values()) and max(counts.values()) > 1.6 * min(counts.values()):
        flags.append("OPTION_WORD_LENGTH_OUTLIER")
    if min(char_counts.values()) and max(char_counts.values()) > 1.7 * min(char_counts.values()):
        flags.append("OPTION_CHARACTER_LENGTH_OUTLIER")
    # Features that identify exactly the correct option set can reveal Gold in
    # single-choice and multi-select items, even when no one option is extreme.
    scalar_features = {
        "word_count": {key: row["word_count"] for key, row in features.items()},
        "character_count": {key: row["character_count"] for key, row in features.items()},
        "clause_markers": {key: row["clause_markers"] for key, row in features.items()},
        "punctuation_count": {key: row["punctuation_count"] for key, row in features.items()},
    }
    for feature_name, values in scalar_features.items():
        groups = {}
        for key, value in values.items():
            groups.setdefault(value, set()).add(key)
        if any(members == gold_set or members == distractor_set
               for members in groups.values()) and len(set(values.values())) > 1:
            flags.append("GOLD_SET_" + feature_name.upper() + "_PATTERN")
            risks.append({"feature": feature_name, "pattern": "matches_gold_or_distractor_set"})

    all_cues = set().union(*(set(row["semantic_cues"]) for row in features.values()))
    for cue in sorted(all_cues):
        holders = {key for key, row in features.items() if cue in row["semantic_cues"]}
        if holders == gold_set or holders == distractor_set:
            flags.append("GOLD_SET_SEMANTIC_CUE_PATTERN")
            risks.append({"cue": cue, "feature": "matches_gold_or_distractor_set"})
            if len(gold_set) == 1 and holders == gold_set:
                flags.append("GOLD_UNIQUE_SEMANTIC_CUE")
        elif holders and (holders <= gold_set or holders <= distractor_set):
            flags.append("GOLD_OPTION_UNIQUE_SEMANTIC_CUE")
            risks.append({"cue": cue, "feature": "confined_to_one_answer_class"})

    question_tokens = {w.lower() for w in _WORD_RE.findall(question)} - _STOP_WORDS
    overlaps = {key: len(row["content_tokens"] & question_tokens)
                for key, row in features.items()}
    if any(overlaps[key] >= 2 and overlaps[key] >
           max((overlaps[other] for other in distractor_set), default=0) + 1
           for key in gold_set):
        flags.append("GOLD_STEM_LEXICAL_OVERLAP_OUTLIER")

    # Analyze each number independently, preserving every option's complete
    # numeric signature so times, ranges and quantities are not skipped.
    numeric_rows = {key: row["numeric_values"] for key, row in features.items()}
    width_rows = {key: row["digit_widths"] for key, row in features.items()}
    decimal_rows = {key: row["decimal_places"] for key, row in features.items()}
    signatures = {key: tuple(width_rows[key]) for key in options}
    if len(set(signatures.values())) > 1 and any(
            {key for key, signature in signatures.items() if signature == value} in
            (gold_set, distractor_set) for value in set(signatures.values())):
        flags.append("GOLD_SET_NUMERIC_WIDTH_PATTERN")
        if len(gold_set) == 1 and signatures[next(iter(gold_set))] not in {
                signature for key, signature in signatures.items() if key in distractor_set}:
            flags.append("GOLD_UNIQUE_NUMERIC_WIDTH")
    precision_signatures = {key: tuple(decimal_rows[key]) for key in options}
    if len(set(precision_signatures.values())) > 1 and any(
            {key for key, signature in precision_signatures.items() if signature == value} in
            (gold_set, distractor_set) for value in set(precision_signatures.values())):
        flags.append("GOLD_SET_NUMERIC_PRECISION_PATTERN")
    max_number_count = max((len(values) for values in numeric_rows.values()), default=0)
    for position in range(max_number_count):
        present = {key for key, values in numeric_rows.items() if len(values) > position}
        if present == gold_set or present == distractor_set:
            if max_number_count:
                flags.append("GOLD_SET_NUMERIC_COUNT_PATTERN")
        if len(present) != len(options):
            continue
        values = {key: numeric_rows[key][position] for key in options}
        distinct = len(set(values.values())) == len(options)
        low_cue = re.search(r"\b(minimum|least|smallest|lowest|shortest|earliest)\b", question, re.I)
        high_cue = re.search(r"\b(maximum|most|largest|highest|longest|latest)\b", question, re.I)
        if distinct:
            low_gold = {key for key, value in values.items() if value == min(values.values())}
            high_gold = {key for key, value in values.items() if value == max(values.values())}
            if (low_gold == gold_set and low_cue) or (high_gold == gold_set and high_cue):
                flags.append("GOLD_EXTREME_MATCHES_STEM")
            gold_values = [values[key] for key in gold_set]
            distractor_values = [values[key] for key in distractor_set]
            if (len(gold_set) == 1 and distractor_values and
                    (all(value > gold_values[0] for value in distractor_values)
                     or all(value < gold_values[0] for value in distractor_values))):
                flags.append("GOLD_NUMERIC_DIRECTION_OUTLIER")
            elif (distractor_values and
                  (max(gold_values) < min(distractor_values)
                   or max(distractor_values) < min(gold_values))):
                flags.append("GOLD_SET_NUMERIC_CLASS_SEPARATION")
        widths_at_position = {key: width_rows[key][position]
                              for key in options if len(width_rows[key]) > position}
        if len(widths_at_position) == len(options) and len(set(widths_at_position.values())) > 1:
            width_group = {key for key, width in widths_at_position.items()
                           if width == widths_at_position[next(iter(gold_set))]}
            if width_group == gold_set or width_group == distractor_set:
                flags.append("GOLD_SET_NUMERIC_WIDTH_PATTERN")
                if len(gold_set) == 1 and width_group == gold_set:
                    flags.append("GOLD_UNIQUE_NUMERIC_WIDTH")
        precision_at_position = {key: decimal_rows[key][position]
                                 for key in options if len(decimal_rows[key]) > position}
        if (len(precision_at_position) == len(options)
                and len(set(precision_at_position.values())) > 1):
            gold_precision = {precision_at_position[key] for key in gold_set}
            if len(gold_precision) == 1:
                matching = {key for key, places in precision_at_position.items()
                            if places in gold_precision}
                if matching == gold_set or matching == distractor_set:
                    flags.append("GOLD_SET_NUMERIC_PRECISION_PATTERN")
        ordered = sorted(values.values())
        adjacent_gaps = [b - a for a, b in zip(ordered, ordered[1:])]
        nonzero = [gap for gap in adjacent_gaps if gap > 0]
        if len(nonzero) >= 3 and max(nonzero) > 2.5 * (sum(nonzero) / len(nonzero)):
            flags.append("NUMERIC_RANGE_GAP_OUTLIER")

    flags = sorted(set(flags))
    if flags:
        decision = "HOLD_REQUIRES_OPTION_ONLY_REVIEW"
    else:
        decision = "NO_STATIC_TRIGGER_REVIEW_STILL_REQUIRED"
    return {
        "decision": decision,
        "flags": flags,
        "features": {key: {k: v for k, v in row.items() if k != "content_tokens"}
                     for key, row in features.items()},
        "word_counts": counts,
        "character_counts": char_counts,
        "numeric_risks": risks,
        "static_shortcuts_proved_absent": False,
        "scope": "feature screening only; option-only counterfactual review remains human/model semantic judgment",
    }


def blind_instruction():
    return (Path(__file__).with_name("base_qa_prompts_v2_1") / "blind.txt").read_text()


def construction_instruction():
    return (Path(__file__).with_name("base_qa_prompts_v2_1") / "construction.txt").read_text()


def blind_packet(candidate, source, source_scope):
    """Return the public-only packet plus trusted hashes for a fresh reviewer context."""
    public, private = v1.prepare_display(candidate, seed="independent-review-v2.1")
    displayed = {private["display_to_canonical"][label]: value
                 for label, value in public["options"].items()}
    payload = {
        "stage": "INDEPENDENT_SOLVE",
        "instruction": blind_instruction(),
        "candidate_id": candidate["candidate_id"],
        "reviewer_mode": "same_model_blind",
        "source": v2.source_view(source, source_scope),
        "source_scope": source_scope,
        "question_type": candidate["question_type"],
        "question": candidate["question"],
        "options": displayed,
    }
    packet_hash = fingerprint(payload)
    scope_hash = fingerprint(source_scope)
    return {
        "packet": payload,
        "packet_sha256": packet_hash,
        "source_scope_sha256": scope_hash,
    }


def build_orchestration_receipt(candidate_id, packet_sha256, source_scope_sha256,
                                reviewer_id, context_id, issuer_id):
    """Bind a platform-issued fresh-context receipt to the public review packet."""
    return {
        "schema_version": ISOLATION_SCHEMA,
        "candidate_id": candidate_id,
        "packet_sha256": packet_sha256,
        "source_scope_sha256": source_scope_sha256,
        "reviewer_id": reviewer_id,
        "context_id": context_id,
        "context_mode": "fresh_context_no_history",
        "author_material_visible": False,
        "issuer_id": issuer_id,
    }


class _DuplicateKey(ValueError):
    pass


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonstandard JSON constant: " + value)


def _valid_text_list(value, field, errors, allow_empty=False):
    if not isinstance(value, list):
        errors.append(field + ":TYPE")
        return
    if not allow_empty and not value:
        errors.append(field + ":EMPTY")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(field + ":ITEM_TYPE")


def parse_blind_response(raw_bytes, expected_packet_hash, expected_scope_hash,
                         candidate, source, source_scope=None,
                         orchestration_receipt=None):
    """Strict, no-repair parser. It is total over malformed byte payloads."""
    raw_hash = hashlib.sha256(raw_bytes).hexdigest() if isinstance(raw_bytes, bytes) else None
    errors = []
    response = None
    if not isinstance(raw_bytes, bytes):
        return {"status": "INVALID", "raw_sha256": None, "response": None,
                "errors": ["RAW_INPUT:TYPE"]}
    if (not isinstance(candidate, dict) or not isinstance(candidate.get("options"), dict)
            or not isinstance(source, dict) or not isinstance(source.get("turns"), list)
            or not isinstance(source_scope, dict)):
        return {"status": "INVALID", "raw_sha256": raw_hash, "response": None,
                "errors": ["CONTEXT:INVALID"]}
    if fingerprint(source_scope) != expected_scope_hash:
        errors.append("SOURCE_SCOPE:HASH_CONTENT_MISMATCH")
    if source_scope.get("source_id") != source.get("source_id", source.get("root_id")):
        errors.append("SOURCE_SCOPE:SOURCE_ID_MISMATCH")
    bound, binding_error = _verify_source_binding(source, source_scope.get("source_binding"))
    if not bound:
        errors.append(binding_error)
    if _source_sha256(source, source_scope.get("source_binding")) != source_scope.get("source_sha256"):
        errors.append("SOURCE_SCOPE:SOURCE_HASH_MISMATCH")
    try:
        decoded = raw_bytes.decode("utf-8", errors="strict")
        response = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys,
                              parse_constant=_reject_constant)
    except _DuplicateKey:
        return {"status": "INVALID", "raw_sha256": raw_hash, "response": None,
                "errors": ["JSON:DUPLICATE_KEY"]}
    except UnicodeDecodeError:
        return {"status": "INVALID", "raw_sha256": raw_hash, "response": None,
                "errors": ["JSON:INVALID_UTF8"]}
    except (json.JSONDecodeError, ValueError):
        return {"status": "INVALID", "raw_sha256": raw_hash, "response": None,
                "errors": ["JSON:UNPARSEABLE"]}
    except Exception:
        return {"status": "INVALID", "raw_sha256": raw_hash, "response": None,
                "errors": ["JSON:UNPARSEABLE"]}
    if not isinstance(response, dict):
        return {"status": "INVALID", "raw_sha256": raw_hash, "response": None,
                "errors": ["TOP_LEVEL:OBJECT_REQUIRED"]}
    missing = sorted(BLIND_FIELDS - set(response))
    extra = sorted(set(response) - BLIND_FIELDS)
    if missing:
        errors.extend("FIELD_MISSING:" + field for field in missing)
    if extra:
        errors.extend("FIELD_EXTRA:" + field for field in extra)
    scalar_types = {
        "schema_version": str, "candidate_id": str, "packet_sha256": str,
        "source_scope_sha256": str, "reviewer_id": str, "model_id": str,
        "reviewer_mode": str, "decision": str,
    }
    for field, expected_type in scalar_types.items():
        if field in response and (not isinstance(response[field], expected_type)
                                  or not response[field].strip()):
            errors.append("FIELD_TYPE:" + field)
    if response.get("schema_version") != BLIND_SCHEMA:
        errors.append("SCHEMA_VERSION:MISMATCH")
    if response.get("candidate_id") != candidate.get("candidate_id"):
        errors.append("CANDIDATE_ID:MISMATCH")
    if not _is_sha256(expected_packet_hash) or response.get("packet_sha256") != expected_packet_hash:
        errors.append("PACKET_HASH:MISMATCH")
    if not _is_sha256(expected_scope_hash) or response.get("source_scope_sha256") != expected_scope_hash:
        errors.append("SOURCE_SCOPE_HASH:MISMATCH")
    if response.get("reviewer_mode") != "same_model_blind":
        errors.append("REVIEWER_MODE:INVALID")
    if response.get("reviewer_id") == candidate.get("author_id"):
        errors.append("REVIEWER_ID:AUTHOR_COLLISION")
    if response.get("decision") not in ("SOLVED", "HOLD"):
        errors.append("DECISION:INVALID")
    options = candidate.get("options") if isinstance(candidate, dict) else {}
    if not isinstance(options, dict):
        options = {}
        errors.append("CANDIDATE_OPTIONS:INVALID")
    answer = response.get("independent_gold")
    if not isinstance(answer, list) or any(not isinstance(item, str) for item in answer):
        errors.append("INDEPENDENT_GOLD:TYPE")
    elif len(answer) != len(set(answer)) or not set(answer) <= set(options):
        errors.append("INDEPENDENT_GOLD:INVALID_IDS")
    elif response.get("decision") == "SOLVED":
        kind = candidate.get("question_type")
        if not answer:
            errors.append("INDEPENDENT_GOLD:EMPTY_SOLVED")
        elif kind == "single_choice" and len(answer) != 1:
            errors.append("INDEPENDENT_GOLD:CARDINALITY")
        elif kind == "ordering" and set(answer) != set(options):
            errors.append("INDEPENDENT_GOLD:ORDERING_COVERAGE")
    _valid_text_list(response.get("reasoning_steps"), "REASONING_STEPS", errors)
    _valid_text_list(response.get("ambiguities"), "AMBIGUITIES", errors, allow_empty=True)
    _valid_text_list(response.get("source_limitations"), "SOURCE_LIMITATIONS", errors, allow_empty=True)
    if response.get("decision") == "SOLVED" and response.get("ambiguities") != []:
        errors.append("AMBIGUITIES:UNRESOLVED_SOLVED")
    evidence = response.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("EVIDENCE:NONEMPTY_LIST_REQUIRED")
    else:
        by_ref = {}
        for turn in source["turns"]:
            if isinstance(turn, dict) and isinstance(turn.get("turn_ref"), str):
                by_ref[turn["turn_ref"]] = turn
        cited_refs = []
        for index, item in enumerate(evidence):
            prefix = "EVIDENCE:" + str(index)
            if not isinstance(item, dict):
                errors.append(prefix + ":OBJECT_REQUIRED")
                continue
            if set(item) != {"turn_ref", "speaker", "quote"}:
                errors.append(prefix + ":FIELDS_INVALID")
                continue
            if any(not isinstance(item.get(k), str) or not item[k].strip()
                   for k in ("turn_ref", "speaker", "quote")):
                errors.append(prefix + ":FIELD_TYPE")
                continue
            turn = by_ref.get(item["turn_ref"])
            cited_refs.append(item["turn_ref"])
            text = turn.get("text") if isinstance(turn, dict) else None
            if (not turn or turn.get("speaker") != item["speaker"]
                    or not isinstance(text, str) or item["quote"] not in text):
                errors.append(prefix + ":NOT_EXACT_SOURCE_EVIDENCE")
        if len(cited_refs) != len(set(cited_refs)):
            errors.append("EVIDENCE:DUPLICATE_TURN_REF")
        gate = formal_evidence_gate(source_scope, cited_refs)
        if not gate["eligible"]:
            errors.extend("EVIDENCE:NOT_FORMALLY_ADMITTED:" + error for error in gate["errors"])
            errors.extend("EVIDENCE:TURN_NOT_FORMALLY_ALLOWED:" + ref
                          for ref in gate["ineligible_turn_refs"])
    receipt = response.get("isolation_receipt")
    if not isinstance(receipt, dict):
        errors.append("ISOLATION_RECEIPT:OBJECT_REQUIRED")
    else:
        if not isinstance(orchestration_receipt, dict):
            errors.append("ISOLATION_RECEIPT:ORCHESTRATION_RECEIPT_REQUIRED")
        elif set(orchestration_receipt) != {
                "schema_version", "candidate_id", "packet_sha256", "source_scope_sha256",
                "reviewer_id", "context_id", "context_mode", "author_material_visible", "issuer_id"}:
            errors.append("ISOLATION_RECEIPT:ORCHESTRATION_FIELDS_INVALID")
        else:
            if orchestration_receipt.get("schema_version") != ISOLATION_SCHEMA:
                errors.append("ISOLATION_RECEIPT:SCHEMA")
            if orchestration_receipt.get("candidate_id") != candidate.get("candidate_id"):
                errors.append("ISOLATION_RECEIPT:CANDIDATE_ID")
            if orchestration_receipt.get("packet_sha256") != expected_packet_hash:
                errors.append("ISOLATION_RECEIPT:PACKET_HASH")
            if orchestration_receipt.get("source_scope_sha256") != expected_scope_hash:
                errors.append("ISOLATION_RECEIPT:SOURCE_SCOPE_HASH")
            if orchestration_receipt.get("reviewer_id") != response.get("reviewer_id"):
                errors.append("ISOLATION_RECEIPT:REVIEWER_ID")
            if orchestration_receipt.get("issuer_id") == response.get("reviewer_id"):
                errors.append("ISOLATION_RECEIPT:ISSUER_COLLISION")
            if not isinstance(orchestration_receipt.get("issuer_id"), str) or not orchestration_receipt.get("issuer_id", "").strip():
                errors.append("ISOLATION_RECEIPT:ISSUER_MISSING")
            if receipt != orchestration_receipt:
                errors.append("ISOLATION_RECEIPT:RESPONSE_MISMATCH")
        if set(receipt) != {
                "schema_version", "candidate_id", "packet_sha256", "source_scope_sha256",
                "reviewer_id", "context_id", "context_mode", "author_material_visible", "issuer_id"}:
            errors.append("ISOLATION_RECEIPT:FIELDS_INVALID")
        if receipt.get("schema_version") != ISOLATION_SCHEMA:
            errors.append("ISOLATION_RECEIPT:SCHEMA")
        if receipt.get("candidate_id") != candidate.get("candidate_id"):
            errors.append("ISOLATION_RECEIPT:CANDIDATE_ID")
        if receipt.get("context_mode") != "fresh_context_no_history":
            errors.append("ISOLATION_RECEIPT:CONTEXT_MODE")
        if receipt.get("author_material_visible") is not False:
            errors.append("ISOLATION_RECEIPT:VISIBILITY")
        if not isinstance(receipt.get("context_id"), str) or not receipt.get("context_id", "").strip():
            errors.append("ISOLATION_RECEIPT:CONTEXT_ID")
        if not isinstance(receipt.get("issuer_id"), str) or not receipt.get("issuer_id", "").strip():
            errors.append("ISOLATION_RECEIPT:ISSUER_MISSING")
    errors = sorted(set(errors))
    return {"status": "VALID" if not errors else "INVALID", "raw_sha256": raw_hash,
            "response": response, "errors": errors}


def _write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def freeze_author_candidate(run_dir, candidate, expected_packet_hash, expected_scope_hash):
    """Freeze the complete private candidate and Gold before independent review."""
    if (not isinstance(candidate, dict) or not _is_sha256(expected_packet_hash)
            or not _is_sha256(expected_scope_hash)):
        raise ValueError("candidate and valid review packet hashes are required")
    run_dir = Path(run_dir)
    _write_json_exclusive(run_dir / "AUTHOR_CANDIDATE.json", candidate)
    _write_json_exclusive(run_dir / "AUTHOR_CANDIDATE_FREEZE.json", {
        "version": VERSION,
        "schema_version": VERSION + ":author_candidate_freeze",
        "candidate_sha256": fingerprint(candidate),
        "packet_sha256": expected_packet_hash,
        "source_scope_sha256": expected_scope_hash,
    })


def _load_frozen_author_candidate(run_dir, expected_packet_hash, expected_scope_hash):
    try:
        candidate = json.loads((run_dir / "AUTHOR_CANDIDATE.json").read_text(encoding="utf-8"))
        freeze = json.loads((run_dir / "AUTHOR_CANDIDATE_FREEZE.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, ["AUTHOR_CANDIDATE_FREEZE_MISSING_OR_UNREADABLE"]
    errors = []
    if not isinstance(candidate, dict):
        errors.append("AUTHOR_CANDIDATE_OBJECT_REQUIRED")
    if freeze.get("version") != VERSION or freeze.get("schema_version") != VERSION + ":author_candidate_freeze":
        errors.append("AUTHOR_CANDIDATE_FREEZE_SCHEMA")
    if freeze.get("candidate_sha256") != fingerprint(candidate):
        errors.append("AUTHOR_CANDIDATE_FREEZE_HASH_MISMATCH")
    if freeze.get("packet_sha256") != expected_packet_hash:
        errors.append("AUTHOR_CANDIDATE_PACKET_HASH_MISMATCH")
    if freeze.get("source_scope_sha256") != expected_scope_hash:
        errors.append("AUTHOR_CANDIDATE_SCOPE_HASH_MISMATCH")
    return (candidate if not errors else None), errors


def freeze_blind_response(run_dir, raw_bytes, expected_packet_hash,
                          expected_scope_hash, candidate, source, source_scope,
                          orchestration_receipt):
    """Persist exact reviewer bytes before parse; invalid payloads remain auditable."""
    run_dir = Path(run_dir)
    raw_hash = hashlib.sha256(raw_bytes).hexdigest() if isinstance(raw_bytes, bytes) else None
    with (run_dir / "BLIND_RAW.bin").open("xb") as handle:
        if isinstance(raw_bytes, bytes):
            handle.write(raw_bytes)
    _write_json_exclusive(run_dir / "BLIND_RAW_FREEZE.json", {
        "version": VERSION,
        "schema_version": VERSION + ":raw_freeze",
        "sha256": raw_hash,
        "bytes": len(raw_bytes) if isinstance(raw_bytes, bytes) else None,
        "encoding": "exact input bytes; not normalized",
    })
    parsed = parse_blind_response(raw_bytes, expected_packet_hash, expected_scope_hash,
                                  candidate, source, source_scope, orchestration_receipt)
    frozen_candidate, candidate_freeze_errors = _load_frozen_author_candidate(
        run_dir, expected_packet_hash, expected_scope_hash)
    if frozen_candidate != candidate:
        candidate_freeze_errors.append("AUTHOR_CANDIDATE_INPUT_MISMATCH")
    if candidate_freeze_errors:
        parsed["status"] = "INVALID"
        parsed["errors"] = sorted(set(parsed.get("errors", []) + candidate_freeze_errors))
    _write_json_exclusive(run_dir / "BLIND_PARSE.json", parsed)
    if parsed["status"] == "VALID":
        _write_json_exclusive(run_dir / "BLIND_SOLUTION.json", parsed["response"])
        parse_sha = hashlib.sha256((run_dir / "BLIND_PARSE.json").read_bytes()).hexdigest()
        solution_sha = hashlib.sha256((run_dir / "BLIND_SOLUTION.json").read_bytes()).hexdigest()
        _write_json_exclusive(run_dir / "BLIND_SOLUTION_FREEZE.json", {
            "version": VERSION,
            "schema_version": VERSION + ":solution_freeze",
            "sha256": fingerprint(parsed["response"]),
            "solution_file_sha256": solution_sha,
            "parse_file_sha256": parse_sha,
            "raw_sha256": parsed["raw_sha256"],
        })
    return parsed


def blind_disposition(run_dir, expected_packet_hash, expected_scope_hash, source,
                      source_scope, orchestration_receipt):
    """Re-parse the frozen raw answer and compare Gold only after freeze verification."""
    errors = []
    run_dir = Path(run_dir) if isinstance(run_dir, (str, Path)) else None
    if run_dir is None:
        return {"decision": "HOLD", "errors": ["BLIND_FREEZE_DIRECTORY_REQUIRED"],
                "gold_compared": False}
    try:
        candidate, candidate_freeze_errors = _load_frozen_author_candidate(
            run_dir, expected_packet_hash, expected_scope_hash)
        raw = (run_dir / "BLIND_RAW.bin").read_bytes()
        raw_freeze = json.loads((run_dir / "BLIND_RAW_FREEZE.json").read_text(encoding="utf-8"))
        frozen_parse = json.loads((run_dir / "BLIND_PARSE.json").read_text(encoding="utf-8"))
        frozen_solution = json.loads((run_dir / "BLIND_SOLUTION.json").read_text(encoding="utf-8"))
        solution_freeze = json.loads((run_dir / "BLIND_SOLUTION_FREEZE.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"decision": "HOLD", "errors": ["BLIND_FREEZE_MISSING_OR_UNREADABLE"],
                "gold_compared": False}
    if candidate_freeze_errors:
        return {"decision": "HOLD", "errors": candidate_freeze_errors,
                "gold_compared": False}
    if not isinstance(candidate, dict) or not isinstance(candidate.get("gold"), list):
        return {"decision": "HOLD", "errors": ["AUTHOR_CANDIDATE_GOLD_INVALID"],
                "gold_compared": False}
    raw_hash = hashlib.sha256(raw).hexdigest()
    parse_hash = hashlib.sha256((run_dir / "BLIND_PARSE.json").read_bytes()).hexdigest()
    solution_file_hash = hashlib.sha256((run_dir / "BLIND_SOLUTION.json").read_bytes()).hexdigest()
    if raw_freeze.get("version") != VERSION or raw_freeze.get("sha256") != raw_hash:
        errors.append("BLIND_RAW_FREEZE_MISMATCH")
    if raw_freeze.get("schema_version") != VERSION + ":raw_freeze":
        errors.append("BLIND_RAW_FREEZE_SCHEMA")
    if raw_freeze.get("bytes") != len(raw):
        errors.append("BLIND_RAW_BYTE_COUNT_MISMATCH")
    if solution_freeze.get("version") != VERSION:
        errors.append("BLIND_SOLUTION_FREEZE_VERSION")
    if solution_freeze.get("schema_version") != VERSION + ":solution_freeze":
        errors.append("BLIND_SOLUTION_FREEZE_SCHEMA")
    if solution_freeze.get("raw_sha256") != raw_hash:
        errors.append("BLIND_SOLUTION_RAW_HASH_MISMATCH")
    if solution_freeze.get("parse_file_sha256") != parse_hash:
        errors.append("BLIND_PARSE_FREEZE_MISMATCH")
    if solution_freeze.get("solution_file_sha256") != solution_file_hash:
        errors.append("BLIND_SOLUTION_FILE_HASH_MISMATCH")
    if solution_freeze.get("sha256") != fingerprint(frozen_solution):
        errors.append("BLIND_SOLUTION_CONTENT_HASH_MISMATCH")
    reparsed = parse_blind_response(raw, expected_packet_hash, expected_scope_hash,
                                    candidate, source, source_scope,
                                    orchestration_receipt)
    if reparsed.get("status") != "VALID":
        errors.append("BLIND_RAW_REPARSE_INVALID")
    if frozen_parse != reparsed:
        errors.append("BLIND_PARSE_CONTENT_MISMATCH")
    if reparsed.get("response") != frozen_solution:
        errors.append("BLIND_SOLUTION_CONTENT_MISMATCH")
    if errors:
        return {"decision": "HOLD", "errors": sorted(set(errors)), "gold_compared": False}
    if reparsed.get("status") != "VALID":
        errors.append("BLIND_PARSE_INVALID")
        return {"decision": "HOLD", "errors": errors, "gold_compared": False}
    response = reparsed.get("response")
    if not isinstance(response, dict) or response.get("decision") != "SOLVED":
        return {"decision": "HOLD", "errors": ["REVIEWER_HOLD"], "gold_compared": False}
    author_gold = candidate["gold"]
    question_type = candidate.get("question_type")
    if not isinstance(response.get("independent_gold"), list):
        return {"decision": "HOLD", "errors": ["GOLD_SCHEMA_INVALID"], "gold_compared": False}
    if not v1.same_answer(question_type, author_gold, response["independent_gold"]):
        errors.append("REVIEWER_GOLD_DISAGREEMENT")
        return {"decision": "HOLD", "errors": errors, "gold_compared": True,
                "author_gold": author_gold, "reviewer_gold": response["independent_gold"]}
    return {"decision": "REVIEWER_MATCH", "errors": [], "gold_compared": True,
            "author_gold": author_gold, "reviewer_gold": response["independent_gold"]}


def implementation_bindings():
    paths = [Path(__file__), Path(v2.__file__), Path(v1.__file__),
             Path(v1.legacy.__file__),
             *sorted(Path(__file__).with_name("base_qa_prompts_v2_1").glob("*.txt"))]
    return [v1.legacy.file_binding(path) for path in paths]
