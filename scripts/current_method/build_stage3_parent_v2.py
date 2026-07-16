#!/usr/bin/env python3
"""Build leakage-safe Stage3 challenger parents from Stage2 JSONL rows.

The source files are treated as immutable snapshots.  Output rows are rebuilt
from a small allowlist instead of copying Stage2 samples and deleting fields;
this prevents old QA answers, solver evaluations, and legacy graphs from
silently becoming challenger inputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from scripts.current_method import semantic_graph_v2
except ModuleNotFoundError:  # Direct execution from scripts/current_method.
    import semantic_graph_v2  # type: ignore[no-redef]


SCHEMA_VERSION = "stage3_parent_v2"
SEMANTIC_GRAPH_VERSION = "semantic_graph_v2"
TRAIN_SPLITS = {"train", "training", "train_candidate"}
FORBIDDEN_SPLITS = {
    "dev",
    "development",
    "eval",
    "evaluation",
    "heldout",
    "holdout",
    "test",
    "validation",
    "val",
}
SPLIT_KEYS = {"split", "source_split", "dataset_split", "frontier_from_split", "heldout_split"}
TRAINING_USE_KEYS = {"training_use", "raw_training_use"}
FORBIDDEN_MATCH_MODES = {"lineage", "exact"}
# In historical Stage2 rows, top-level source_id is a reusable raw LoCoMo chunk
# identity rather than the generated sample identity.  It belongs in lineage
# auditing, not in the exact-sample hard gate.
EXACT_IDENTITY_KEYS = ("sample_id", "question_id", "dialogue_hash")
RAW_QA_RE = re.compile(r"raw[\s_-]*qa", flags=re.IGNORECASE)
QA_SURFACE_RE = re.compile(r"(?:^|\s)[QA]\s*:\s*", flags=re.IGNORECASE)
# Dialogue cleanup deliberately uses a stricter surface than graph/provenance
# rejection.  A natural utterance may contain a question mark or an isolated
# "A:" token; the historical scaffold always contains either a raw_qa marker
# or a Q:/A: pair in the same turn.
QA_PAIR_SURFACE_RE = re.compile(
    r"\bQ\s*:\s*.+?\bA\s*:\s*",
    flags=re.IGNORECASE | re.DOTALL,
)

LEGACY_QA_DIALOGUE_CLEANUP_VERSION = "historical_raw_qa_dialogue_cleanup_v1"

DETERMINISTIC_PADDING_CLEANUP_VERSION = "historical_deterministic_padding_cleanup_v1"
# Exact text emitted by ensure_dialogue_padding and
# ensure_append_length_guard_margin in native_api_graph_dialogue_expansion.py.
# Keep this tuple ordered so its audit hash is stable across processes.
KNOWN_DETERMINISTIC_PADDING_TEXTS = (
    "Let's review the latest updates carefully before deciding.",
    "Okay, that helps me keep the details straight.",
    "I will rely only on what was clearly said in this conversation.",
    "Good; anything else can wait until later.",
    (
        "Before we leave it there, I want to restate the practical picture in ordinary terms. "
        "The earlier part of the conversation is background, and these later updates are the ones I will use for the next decision. "
        "If someone asks why, I will point to the newest correction rather than the older assumption."
    ),
    (
        "That helps. I am separating what someone merely heard from what the group actually settled after the correction. "
        "The newer detail changes how I would act, while the older version is useful only as context. "
        "I will not treat a canceled or uncertain plan as if it were still confirmed."
    ),
    (
        "One more check: the final takeaway should come from the latest clear update, not from the first thing mentioned. "
        "When the conversation changes direction, the later clarification controls the answer. "
        "The old note can explain why the correction mattered, but it should not override the corrected plan."
    ),
    (
        "Agreed. I will keep the timing straight and remember which details were limited, delayed, canceled, or restored. "
        "That way the final answer follows the actual sequence of updates. "
        "Anything that was only a report stays weaker than the detail that was directly confirmed later."
    ),
)
_KNOWN_DETERMINISTIC_PADDING_TEXT_SET = frozenset(KNOWN_DETERMINISTIC_PADDING_TEXTS)


class DeterministicPaddingEvidenceReferenceError(ValueError):
    """Raised when cleanup would remove evidence retained by the parent graph."""


class LegacyQADialogueEvidenceReferenceError(ValueError):
    """Raised when QA-scaffold cleanup would remove retained graph evidence."""

    def __init__(self, source_label: str, turn_ids: Iterable[str]) -> None:
        self.turn_ids = sorted(set(turn_ids))
        super().__init__(
            f"{source_label}: legacy QA dialogue cleanup would remove retained "
            f"semantic graph evidence turn IDs: {self.turn_ids}"
        )


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def walk_mappings(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from walk_mappings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from walk_mappings(nested)


def scalar_values(value: Any) -> Iterator[str]:
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from scalar_values(item)
    elif value is not None and not isinstance(value, (dict, list)):
        text = normalize_text(value)
        if text:
            yield text


def training_rejection_reasons(row: dict[str, Any]) -> list[str]:
    """Return reasons that a row is not an affirmative train candidate."""

    reasons: list[str] = []
    observed_train_marker = False
    for mapping in walk_mappings(row):
        for raw_key, raw_value in mapping.items():
            key = str(raw_key).lower()
            if key in SPLIT_KEYS:
                for value in scalar_values(raw_value):
                    split = value.lower()
                    if split in TRAIN_SPLITS:
                        observed_train_marker = True
                    elif split in FORBIDDEN_SPLITS or key == "heldout_split":
                        reasons.append(f"forbidden_split:{key}={split}")
                    else:
                        reasons.append(f"non_train_split:{key}={split}")
            elif key in TRAINING_USE_KEYS:
                for value in scalar_values(raw_value):
                    use = value.lower()
                    if use == "train_candidate":
                        observed_train_marker = True
                    else:
                        reasons.append(f"training_use_not_train_candidate:{use}")
            elif key == "allowed_train_seed" and raw_value is False:
                reasons.append("allowed_train_seed_false")
            elif key == "heldout_contamination" and raw_value is True:
                reasons.append("heldout_contamination_true")
            elif key == "forbidden_reasons" and isinstance(raw_value, list) and raw_value:
                reasons.append("source_forbidden_reasons_present")
    if not observed_train_marker:
        reasons.append("missing_train_candidate_evidence")
    return sorted(set(reasons))


def exact_identifiers(row: dict[str, Any]) -> dict[str, set[str]]:
    """Return only explicitly comparable identities, never cross categories."""

    dialogue = row.get("dialogue") if isinstance(row.get("dialogue"), dict) else {}
    values: dict[str, Any] = {
        "sample_id": row.get("sample_id"),
        "question_id": row.get("question_id"),
        "dialogue_hash": row.get("dialogue_hash") or dialogue.get("dialogue_hash"),
    }
    return {key: set(scalar_values(value)) for key, value in values.items()}


IDENTITY_SKIP_KEYS = {
    "expanded_graph",
    "graph",
    "graph_update",
    "old_qa",
    "qa",
    "semantic_parent_graph",
    "solver_eval",
    "state_chain",
    "state_chain_nodes",
}


def walk_lineage_mappings(value: Any) -> Iterator[dict[str, Any]]:
    """Walk provenance-like mappings without collecting graph event lineage IDs."""

    if isinstance(value, dict):
        yield value
        for key, nested in value.items():
            if str(key).lower() not in IDENTITY_SKIP_KEYS:
                yield from walk_lineage_mappings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from walk_lineage_mappings(nested)


def lineage_identifier_key(key: str) -> bool:
    lower = key.lower()
    if lower in {"sample_id", "source_id", "lineage_id", "lineage_group"}:
        return True
    return any(token in lower for token in ("parent", "ancestor", "root", "chunk", "group", "lineage")) and (
        lower.endswith("_id") or lower.endswith("_ids") or lower.endswith("_group")
    )


def lineage_identifiers(row: dict[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    for mapping in walk_lineage_mappings(row):
        for key, value in mapping.items():
            if lineage_identifier_key(str(key)):
                identifiers.update(scalar_values(value))
    return identifiers


def load_forbidden_identifiers(paths: Iterable[Path]) -> tuple[dict[str, Any], int]:
    index: dict[str, Any] = {
        "exact": {key: set() for key in EXACT_IDENTITY_KEYS},
        "lineage": set(),
    }
    row_count = 0
    for path in paths:
        for _, row in iter_jsonl(path):
            row_count += 1
            found_exact = exact_identifiers(row)
            for key in EXACT_IDENTITY_KEYS:
                index["exact"][key].update(found_exact[key])
            index["lineage"].update(lineage_identifiers(row))
    return index, row_count


def exact_forbidden_matches(
    candidate: dict[str, set[str]], forbidden: dict[str, set[str]]
) -> dict[str, list[str]]:
    return {
        key: sorted(candidate[key] & forbidden[key])
        for key in EXACT_IDENTITY_KEYS
        if candidate[key] & forbidden[key]
    }


def lineage_forbidden_matches(candidate: set[str], forbidden: set[str]) -> list[str]:
    return sorted(candidate & forbidden)


def dialogue_and_turns(row: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    dialogue = row.get("dialogue")
    turns: Any = None
    if isinstance(dialogue, dict):
        turns = dialogue.get("turns")
    elif isinstance(dialogue, list):
        turns = dialogue
    if not isinstance(turns, list):
        turns = row.get("turns")
    if not isinstance(turns, list) or not turns or not all(isinstance(turn, dict) for turn in turns):
        return None, []
    if dialogue is None:
        dialogue = {"turns": copy.deepcopy(turns)}
    return copy.deepcopy(dialogue), copy.deepcopy(turns)


def dialogue_content_hash(turns: list[dict[str, Any]]) -> str:
    content = []
    for turn in turns:
        content.append(
            {
                "speaker": normalize_text(turn.get("speaker") or turn.get("role")).casefold(),
                "session": normalize_text(
                    turn.get("session_id") or turn.get("session") or turn.get("date")
                ).casefold(),
                "text": normalize_text(
                    turn.get("text") or turn.get("content") or turn.get("utterance")
                ).casefold(),
            }
        )
    return sha256_payload(content)


def deterministic_padding_signature_sha256() -> str:
    """Return a stable fingerprint of the exact cleanup allowlist."""

    return sha256_payload(
        {
            "cleanup_version": DETERMINISTIC_PADDING_CLEANUP_VERSION,
            "texts": KNOWN_DETERMINISTIC_PADDING_TEXTS,
        }
    )


def known_deterministic_padding_text(value: Any) -> bool:
    """Return whether value exactly equals a historical deterministic filler text."""

    return isinstance(value, str) and value in _KNOWN_DETERMINISTIC_PADDING_TEXT_SET


def known_deterministic_padding_turn(turn: Any) -> bool:
    """Exact, reusable detector for formal-runner parent admission checks."""

    return isinstance(turn, dict) and known_deterministic_padding_text(turn.get("text"))


def detect_historical_deterministic_padding(
    turns: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return stable audit records for turns matching the exact filler allowlist."""

    detected: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if not known_deterministic_padding_turn(turn):
            continue
        detected.append(
            {
                "turn_index": index,
                "turn_id": copy.deepcopy(turn.get("turn_id")),
                "speaker": copy.deepcopy(turn.get("speaker")),
                "text_sha256": sha256_payload(turn["text"]),
            }
        )
    return detected


def _turn_id_set(turns: Iterable[dict[str, Any]]) -> set[str]:
    return {
        turn_id
        for turn in turns
        if (turn_id := normalize_text(turn.get("turn_id")))
    }


def _retained_evidence_turn_ids(events: Iterable[dict[str, Any]]) -> set[str]:
    evidence_turn_ids: set[str] = set()
    for event in events:
        raw_ids = event.get("evidence_turn_ids")
        if not isinstance(raw_ids, list):
            continue
        evidence_turn_ids.update(
            turn_id
            for value in raw_ids
            if (turn_id := normalize_text(value))
        )
    return evidence_turn_ids


def _dialogue_turn_text(turn: dict[str, Any]) -> str:
    """Read the utterance from supported dialogue schemas without rewriting it."""

    for key in ("text", "content", "utterance"):
        value = turn.get(key)
        if isinstance(value, str):
            return value
    return ""


def legacy_qa_scaffolding_turn(turn: Any) -> bool:
    """Detect only the historical raw-QA turn surfaces.

    This is intentionally narrower than ``contains_legacy_qa_surface``.  The
    latter is a conservative graph/provenance gate; dialogue deletion requires
    an unambiguous scaffold marker so ordinary questions are retained.
    """

    if not isinstance(turn, dict):
        return False
    text = _dialogue_turn_text(turn)
    return bool(RAW_QA_RE.search(text) or QA_PAIR_SURFACE_RE.search(text))


def strip_legacy_qa_dialogue_scaffolding(
    dialogue: Any,
    turns: list[dict[str, Any]],
    retained_semantic_events: Iterable[dict[str, Any]],
    *,
    source_label: str = "source row",
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Remove deterministic old QA scaffold turns without damaging evidence.

    Retained turns are deep-copied verbatim, including ``turn_id``.  If a turn
    selected for deletion is evidence for any retained semantic event, cleanup
    fails closed instead of emitting a parent with a dangling evidence link.
    """

    pre_turns = copy.deepcopy(turns)
    removed_indexes = [
        index for index, turn in enumerate(pre_turns) if legacy_qa_scaffolding_turn(turn)
    ]
    removed_index_set = set(removed_indexes)
    removed_turns = [pre_turns[index] for index in removed_indexes]
    post_turns = [
        turn for index, turn in enumerate(pre_turns) if index not in removed_index_set
    ]
    removed_turn_ids = _turn_id_set(removed_turns)
    dangling_evidence = sorted(
        _retained_evidence_turn_ids(retained_semantic_events) & removed_turn_ids
    )
    if dangling_evidence:
        raise LegacyQADialogueEvidenceReferenceError(source_label, dangling_evidence)

    cleaned_dialogue = copy.deepcopy(dialogue)
    if removed_turns:
        if isinstance(cleaned_dialogue, dict):
            cleaned_dialogue["turns"] = copy.deepcopy(post_turns)
            if "dialogue_hash" in cleaned_dialogue:
                cleaned_dialogue["dialogue_hash"] = dialogue_content_hash(post_turns)
        elif isinstance(cleaned_dialogue, list):
            cleaned_dialogue = copy.deepcopy(post_turns)
        else:
            raise ValueError(
                f"{source_label}: unsupported dialogue representation during legacy QA cleanup"
            )

    audit = {
        "enabled": True,
        "cleanup_version": LEGACY_QA_DIALOGUE_CLEANUP_VERSION,
        "pre_turn_count": len(pre_turns),
        "post_turn_count": len(post_turns),
        "removed_turn_count": len(removed_turns),
        "removed_turn_indexes": removed_indexes,
        "removed_turn_ids": sorted(removed_turn_ids),
        "pre_turns_sha256": sha256_payload(pre_turns),
        "post_turns_sha256": sha256_payload(post_turns),
        "removed_turns_sha256": sha256_payload(removed_turns),
        "pre_dialogue_content_sha256": dialogue_content_hash(pre_turns),
        "post_dialogue_content_sha256": dialogue_content_hash(post_turns),
    }
    return cleaned_dialogue, copy.deepcopy(post_turns), audit


def strip_historical_deterministic_padding(
    dialogue: Any,
    turns: list[dict[str, Any]],
    retained_semantic_events: Iterable[dict[str, Any]],
    *,
    source_label: str = "source row",
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Remove only exact historical filler turns and fail on dangling evidence.

    Turn objects that remain are copied without modification, including their
    original turn_id values.  Both representations of the dialogue are rebuilt
    from the same cleaned list so top-level turns and dialogue.turns cannot
    diverge.
    """

    pre_turns = copy.deepcopy(turns)
    removed_turns = [turn for turn in pre_turns if known_deterministic_padding_turn(turn)]
    post_turns = [turn for turn in pre_turns if not known_deterministic_padding_turn(turn)]
    post_turn_ids = _turn_id_set(post_turns)
    removed_turn_ids = _turn_id_set(removed_turns) - post_turn_ids
    dangling_evidence = sorted(
        _retained_evidence_turn_ids(retained_semantic_events) & removed_turn_ids
    )
    if dangling_evidence:
        raise DeterministicPaddingEvidenceReferenceError(
            f"{source_label}: deterministic padding cleanup would remove retained "
            f"semantic graph evidence turn IDs: {dangling_evidence}"
        )

    cleaned_dialogue = copy.deepcopy(dialogue)
    if isinstance(cleaned_dialogue, dict):
        cleaned_dialogue["turns"] = copy.deepcopy(post_turns)
        if "dialogue_hash" in cleaned_dialogue:
            cleaned_dialogue["dialogue_hash"] = dialogue_content_hash(post_turns)
    elif isinstance(cleaned_dialogue, list):
        cleaned_dialogue = copy.deepcopy(post_turns)
    else:  # dialogue_and_turns normally prevents this, but keep the helper closed.
        raise ValueError(f"{source_label}: unsupported dialogue representation during cleanup")

    audit = {
        "enabled": True,
        "cleanup_version": DETERMINISTIC_PADDING_CLEANUP_VERSION,
        "known_padding_text_count": len(KNOWN_DETERMINISTIC_PADDING_TEXTS),
        "known_padding_signature_sha256": deterministic_padding_signature_sha256(),
        "pre_turn_count": len(pre_turns),
        "post_turn_count": len(post_turns),
        "removed_turn_count": len(removed_turns),
        "removed_turn_ids": sorted(removed_turn_ids),
        "pre_turns_sha256": sha256_payload(pre_turns),
        "post_turns_sha256": sha256_payload(post_turns),
        "removed_turns_sha256": sha256_payload(removed_turns),
        "pre_dialogue_content_sha256": dialogue_content_hash(pre_turns),
        "post_dialogue_content_sha256": dialogue_content_hash(post_turns),
    }
    return cleaned_dialogue, copy.deepcopy(post_turns), audit


def first_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def lineage_group(row: dict[str, Any]) -> str:
    candidates = (
        ("row.group_id", row.get("group_id")),
        ("metadata.group_id", first_nested(row, ("metadata", "group_id"))),
        ("metadata.root_seed_parent_id", first_nested(row, ("metadata", "root_seed_parent_id"))),
        ("seed_source.root_seed_parent_id", first_nested(row, ("seed_source", "root_seed_parent_id"))),
        ("seed_source.parent_sample_id", first_nested(row, ("seed_source", "parent_sample_id"))),
        ("seed.source.parent_sample_id", first_nested(row, ("seed", "source", "parent_sample_id"))),
        ("seed_source.parent_chunk_id", first_nested(row, ("seed_source", "parent_chunk_id"))),
        ("metadata.source_seed_id", first_nested(row, ("metadata", "source_seed_id"))),
        ("row.source_id", row.get("source_id")),
        ("row.sample_id", row.get("sample_id")),
    )
    for _, value in candidates:
        text = normalize_text(value)
        if text:
            return text
    return f"row_fingerprint:{sha256_payload(row)}"


def lineage_key(row: dict[str, Any]) -> str:
    """Backward-compatible alias for callers of the initial migration tool."""

    return lineage_group(row)


def contains_legacy_qa_surface(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            lower_key = str(key).lower()
            if lower_key in {"qa", "old_qa", "solver_eval", "frontier_label"}:
                return True
            if RAW_QA_RE.search(lower_key) or contains_legacy_qa_surface(nested):
                return True
        return False
    if isinstance(value, list):
        return any(contains_legacy_qa_surface(item) for item in value)
    if isinstance(value, str):
        return bool(RAW_QA_RE.search(value) or QA_SURFACE_RE.search(value) or "?" in value)
    return False


def source_graphs(row: dict[str, Any]) -> Iterator[Any]:
    for key in ("semantic_parent_graph", "graph_update", "graph", "expanded_graph"):
        graph = row.get(key)
        if isinstance(graph, (dict, list)):
            yield graph


def clean_semantic_events(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int, int]:
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_count = 0
    legacy_or_qa_removed = 0
    noncanonical_removed = 0
    for graph in source_graphs(row):
        events = graph.get("events") if isinstance(graph, dict) else graph
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            source_count += 1
            if contains_legacy_qa_surface(event):
                legacy_or_qa_removed += 1
                continue
            if semantic_graph_v2.validate_event(event):
                noncanonical_removed += 1
                continue
            fingerprint = sha256_payload(event)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            clean.append(copy.deepcopy(event))
    return clean, source_count, legacy_or_qa_removed, noncanonical_removed


def source_graph_hashes(row: dict[str, Any]) -> list[str]:
    hashes: set[str] = set()
    for mapping in walk_mappings(row):
        for key, value in mapping.items():
            if str(key).lower().endswith("graph_snapshot_hash"):
                hashes.update(scalar_values(value))
    return sorted(hashes)


DROP_PROVENANCE_KEYS = {
    "answer",
    "answers",
    "frontier_label",
    "gold",
    "gold_answer",
    "gold_answers",
    "gold_options",
    "graph",
    "graph_dialogue_alignment",
    "graph_update",
    "old_qa",
    "options",
    "qa",
    "question",
    "solver_eval",
    "state_chain",
    "state_chain_nodes",
}


def sanitize_provenance(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, nested in value.items():
            lower_key = str(key).lower()
            if lower_key in DROP_PROVENANCE_KEYS or RAW_QA_RE.search(lower_key):
                continue
            sanitized = sanitize_provenance(nested)
            if sanitized is not None:
                output[str(key)] = sanitized
        return output
    if isinstance(value, list):
        return [sanitized for item in value if (sanitized := sanitize_provenance(item)) is not None]
    if isinstance(value, str) and (RAW_QA_RE.search(value) or QA_SURFACE_RE.search(value)):
        return None
    return copy.deepcopy(value)


def build_provenance(row: dict[str, Any], source_path: Path, line_number: int) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "migration": "build_stage3_parent_v2",
        "source_path": str(source_path.resolve()),
        "source_line_number": line_number,
        "source_sample_id": normalize_text(row.get("sample_id")),
        "source_id": normalize_text(row.get("source_id")),
        "source_generation": normalize_text(row.get("generation_source") or row.get("source")),
    }
    for key in ("provenance", "seed_source", "seed"):
        if isinstance(row.get(key), (dict, list)):
            provenance[key] = sanitize_provenance(row[key])
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata:
        provenance["metadata"] = sanitize_provenance(metadata)
    return provenance


def build_parent(
    row: dict[str, Any],
    *,
    source_path: Path,
    line_number: int,
    dialogue: Any,
    turns: list[dict[str, Any]],
    content_hash: str,
    row_hash: str,
    lineage: str,
    forbidden_match_mode: str,
    forbidden_lineage_overlap: list[str],
    require_nonempty_semantic_parent_graph: bool = False,
    legacy_qa_dialogue_cleanup: dict[str, Any] | None = None,
    deterministic_padding_cleanup: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int, int, int]:
    events, source_event_count, legacy_or_qa_removed, noncanonical_removed = clean_semantic_events(row)
    dialogue_was_cleaned = bool(
        (legacy_qa_dialogue_cleanup or {}).get("removed_turn_count", 0)
        or (deterministic_padding_cleanup or {}).get("removed_turn_count", 0)
    )
    if not dialogue_was_cleaned:
        parent_identity_hash = row_hash
    else:
        parent_identity_hash = sha256_payload(
            {
                "source_row_sha256": row_hash,
                "legacy_qa_dialogue_cleanup_version": (
                    LEGACY_QA_DIALOGUE_CLEANUP_VERSION
                    if (legacy_qa_dialogue_cleanup or {}).get("removed_turn_count", 0)
                    else None
                ),
                "deterministic_padding_cleanup_version": (
                    DETERMINISTIC_PADDING_CLEANUP_VERSION
                    if (deterministic_padding_cleanup or {}).get("removed_turn_count", 0)
                    else None
                ),
                "post_cleanup_dialogue_content_sha256": content_hash,
            }
        )
    parent_id = f"stage3_parent_v2_{parent_identity_hash[:20]}"
    source_sample_id = normalize_text(row.get("sample_id") or row.get("source_id"))
    source_seed = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
    source_sample_name = normalize_text(
        source_seed.get("sample_name")
        or row.get("frontier_from_sample_name")
        or row.get("sample_name")
    )
    graph_hashes = source_graph_hashes(row)
    parent: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": parent_id,
        "source_id": parent_id,
        "split": "train",
        "source_split": "train",
        "training_use": "train_candidate",
        "lineage_group": lineage,
        "dialogue": dialogue,
        "turns": turns,
        "semantic_parent_graph": {
            "schema_version": SEMANTIC_GRAPH_VERSION,
            "graph_id": f"{parent_id}:semantic_parent_graph",
            "parent_graph_id": (
                graph_hashes[-1]
                if graph_hashes
                else f"stage2_source_snapshot:{row_hash[:20]}"
            ),
            "graph_role": "parent_snapshot",
            "events": events,
            "event_count": len(events),
            "empty_parent_snapshot_allowed": not require_nonempty_semantic_parent_graph,
        },
        "source_snapshot": {
            "source_path": str(source_path.resolve()),
            "source_line_number": line_number,
            "source_sample_id": source_sample_id,
            "row_sha256": row_hash,
            "dialogue_content_sha256": content_hash,
            "source_graph_snapshot_hashes": graph_hashes,
            "source_sample_name": source_sample_name,
        },
        "provenance": build_provenance(row, source_path, line_number),
        "seed_source": {
            "split": "train",
            "source_split": "train",
            "training_use": "train_candidate",
            "raw_training_use": "train_candidate",
            "allowed_train_seed": True,
            "parent_source": SCHEMA_VERSION,
            "parent_sample_id": source_sample_id,
            "sample_id": parent_id,
            "source_lineage_key": lineage,
            "lineage_group": lineage,
            "sample_name": source_sample_name,
        },
        "metadata": {
            "source_round": "stage2",
            "target_round": "stage3",
            "parent_schema_version": SCHEMA_VERSION,
            "semantic_graph_schema_version": SEMANTIC_GRAPH_VERSION,
            "source_sample_id": source_sample_id,
            "source_sample_name": source_sample_name,
            "source_lineage_key": lineage,
            "lineage_group": lineage,
            "forbidden_match_mode": forbidden_match_mode,
            "forbidden_lineage_overlap_audit": {
                "has_overlap": bool(forbidden_lineage_overlap),
                "identifiers": forbidden_lineage_overlap,
                "hard_rejected": False,
            },
            "source_snapshot_sha256": row_hash,
            "dialogue_content_sha256": content_hash,
            "source_semantic_event_count": source_event_count,
            "retained_semantic_event_count": len(events),
            "removed_legacy_or_qa_graph_event_count": legacy_or_qa_removed,
            "removed_noncanonical_graph_event_count": noncanonical_removed,
            "require_nonempty_semantic_parent_graph": require_nonempty_semantic_parent_graph,
        },
    }
    if legacy_qa_dialogue_cleanup is not None:
        parent["metadata"]["legacy_qa_dialogue_cleanup"] = copy.deepcopy(
            legacy_qa_dialogue_cleanup
        )
    if deterministic_padding_cleanup is not None:
        parent["metadata"]["deterministic_padding_cleanup"] = copy.deepcopy(
            deterministic_padding_cleanup
        )
    if dialogue_was_cleaned:
        parent["source_snapshot"]["parent_identity_sha256"] = parent_identity_hash
    if source_sample_name:
        parent["sample_name"] = source_sample_name
    # Preserve explicit conversation presentation fields when a source schema
    # stores them outside dialogue.  QA/eval/graph fields are never copied.
    for key in (
        "speaker",
        "speakers",
        "speaker_id",
        "session",
        "sessions",
        "session_id",
        "turn",
        "turn_id",
    ):
        if key in row:
            parent[key] = copy.deepcopy(row[key])
    return parent, source_event_count, legacy_or_qa_removed, noncanonical_removed


def rejected_record(
    row: dict[str, Any],
    *,
    source_path: Path,
    line_number: int,
    reasons: list[str],
    exact_matches: dict[str, list[str]] | None = None,
    lineage_overlap: list[str] | None = None,
    forbidden_match_mode: str,
) -> dict[str, Any]:
    return {
        "source_path": str(source_path.resolve()),
        "source_line_number": line_number,
        "source_sample_id": normalize_text(row.get("sample_id") or row.get("source_id")),
        "source_snapshot_sha256": sha256_payload(row),
        "reasons": sorted(set(reasons)),
        "forbidden_match_mode": forbidden_match_mode,
        "forbidden_exact_identity_matches": exact_matches or {},
        "forbidden_lineage_overlap_audit": {
            "has_overlap": bool(lineage_overlap),
            "identifiers": lineage_overlap or [],
            "hard_rejected": forbidden_match_mode == "lineage" and bool(lineage_overlap),
        },
    }


def migrate(
    input_paths: list[Path],
    forbidden_paths: list[Path],
    *,
    forbidden_match_mode: str = "lineage",
    strip_deterministic_padding: bool = False,
    require_nonempty_semantic_parent_graph: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if forbidden_match_mode not in FORBIDDEN_MATCH_MODES:
        raise ValueError(
            f"Unsupported forbidden match mode {forbidden_match_mode!r}; "
            f"expected one of {sorted(FORBIDDEN_MATCH_MODES)}"
        )
    forbidden, forbidden_row_count = load_forbidden_identifiers(forbidden_paths)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_content: dict[str, str] = {}
    seen_exact: dict[str, dict[str, str]] = {key: {} for key in EXACT_IDENTITY_KEYS}
    reason_counts: Counter[str] = Counter()
    lineage_group_counts: Counter[str] = Counter()
    input_count = 0
    source_event_count = 0
    retained_event_count = 0
    legacy_or_qa_removed_count = 0
    noncanonical_removed_count = 0
    exact_forbidden_match_row_count = 0
    lineage_overlap_row_count = 0
    accepted_lineage_overlap_row_count = 0
    lineage_overlap_occurrence_count = 0
    lineage_overlap_values: set[str] = set()
    padding_cleanup_processed_row_count = 0
    padding_cleanup_affected_row_count = 0
    padding_cleanup_removed_turn_count = 0
    padding_cleanup_pre_turn_count = 0
    padding_cleanup_post_turn_count = 0
    legacy_qa_cleanup_processed_row_count = 0
    legacy_qa_cleanup_affected_row_count = 0
    legacy_qa_cleanup_removed_turn_count = 0
    legacy_qa_cleanup_pre_turn_count = 0
    legacy_qa_cleanup_post_turn_count = 0
    legacy_qa_cleanup_blocked_row_count = 0
    empty_semantic_parent_graph_input_count = 0
    rejected_empty_semantic_parent_graph_count = 0

    for input_path in input_paths:
        for line_number, row in iter_jsonl(input_path):
            input_count += 1
            reasons = training_rejection_reasons(row)
            candidate_exact = exact_identifiers(row)
            exact_matches = exact_forbidden_matches(candidate_exact, forbidden["exact"])
            candidate_lineage = lineage_identifiers(row)
            lineage_overlap = lineage_forbidden_matches(candidate_lineage, forbidden["lineage"])
            if exact_matches:
                exact_forbidden_match_row_count += 1
            if lineage_overlap:
                lineage_overlap_row_count += 1
                lineage_overlap_occurrence_count += len(lineage_overlap)
                lineage_overlap_values.update(lineage_overlap)
            if forbidden_match_mode == "exact" and exact_matches:
                reasons.append("forbidden_exact_identity")
            elif forbidden_match_mode == "lineage" and lineage_overlap:
                reasons.append("forbidden_lineage_identifier")
            duplicate_exact_matches = {
                key: sorted(value for value in values if value in seen_exact[key])
                for key, values in candidate_exact.items()
                if any(value in seen_exact[key] for value in values)
            }
            if duplicate_exact_matches:
                reasons.append("duplicate_exact_identity")
            retained_events, _, _, _ = clean_semantic_events(row)
            if not retained_events:
                empty_semantic_parent_graph_input_count += 1
                if require_nonempty_semantic_parent_graph:
                    reasons.append("empty_semantic_parent_graph")
                    rejected_empty_semantic_parent_graph_count += 1
            dialogue, turns = dialogue_and_turns(row)
            legacy_qa_cleanup_audit: dict[str, Any] | None = None
            legacy_qa_cleanup_blocked_turn_ids: list[str] = []
            padding_cleanup_audit: dict[str, Any] | None = None
            if not turns:
                reasons.append("missing_or_invalid_dialogue")
                content_hash = ""
            else:
                if not reasons:
                    try:
                        dialogue, turns, legacy_qa_cleanup_audit = (
                            strip_legacy_qa_dialogue_scaffolding(
                                dialogue,
                                turns,
                                retained_events,
                                source_label=f"{input_path}:{line_number}",
                            )
                        )
                    except LegacyQADialogueEvidenceReferenceError as exc:
                        reasons.append(
                            "legacy_qa_dialogue_cleanup_would_remove_retained_evidence"
                        )
                        legacy_qa_cleanup_blocked_turn_ids = exc.turn_ids
                        legacy_qa_cleanup_blocked_row_count += 1
                    if not turns:
                        reasons.append("empty_dialogue_after_legacy_qa_cleanup")
                if strip_deterministic_padding and not reasons:
                    dialogue, turns, padding_cleanup_audit = strip_historical_deterministic_padding(
                        dialogue,
                        turns,
                        retained_events,
                        source_label=f"{input_path}:{line_number}",
                    )
                    if not turns:
                        reasons.append("empty_dialogue_after_deterministic_padding_cleanup")
                content_hash = dialogue_content_hash(turns)
            lineage = lineage_group(row)
            if content_hash and content_hash in seen_content:
                reasons.append("duplicate_dialogue_content")
            if reasons:
                record = rejected_record(
                    row,
                    source_path=input_path,
                    line_number=line_number,
                    reasons=reasons,
                    exact_matches=exact_matches,
                    lineage_overlap=lineage_overlap,
                    forbidden_match_mode=forbidden_match_mode,
                )
                if duplicate_exact_matches:
                    record["duplicate_exact_identity_matches"] = duplicate_exact_matches
                if legacy_qa_cleanup_blocked_turn_ids:
                    record["legacy_qa_dialogue_cleanup"] = {
                        "enabled": True,
                        "cleanup_version": LEGACY_QA_DIALOGUE_CLEANUP_VERSION,
                        "blocked_retained_evidence_turn_ids": legacy_qa_cleanup_blocked_turn_ids,
                    }
                rejected.append(record)
                reason_counts.update(record["reasons"])
                continue

            row_hash = sha256_payload(row)
            parent, raw_event_count, qa_removed, noncanonical_removed = build_parent(
                row,
                source_path=input_path,
                line_number=line_number,
                dialogue=dialogue,
                turns=turns,
                content_hash=content_hash,
                row_hash=row_hash,
                lineage=lineage,
                forbidden_match_mode=forbidden_match_mode,
                forbidden_lineage_overlap=lineage_overlap,
                require_nonempty_semantic_parent_graph=(
                    require_nonempty_semantic_parent_graph
                ),
                legacy_qa_dialogue_cleanup=legacy_qa_cleanup_audit,
                deterministic_padding_cleanup=padding_cleanup_audit,
            )
            accepted.append(parent)
            seen_content[content_hash] = parent["sample_id"]
            for key, values in candidate_exact.items():
                for value in values:
                    seen_exact[key][value] = parent["sample_id"]
            lineage_group_counts[lineage] += 1
            if lineage_overlap:
                accepted_lineage_overlap_row_count += 1
            source_event_count += raw_event_count
            retained_event_count += parent["semantic_parent_graph"]["event_count"]
            legacy_or_qa_removed_count += qa_removed
            noncanonical_removed_count += noncanonical_removed
            if legacy_qa_cleanup_audit is not None:
                legacy_qa_cleanup_processed_row_count += 1
                removed_turn_count = int(legacy_qa_cleanup_audit["removed_turn_count"])
                legacy_qa_cleanup_affected_row_count += int(removed_turn_count > 0)
                legacy_qa_cleanup_removed_turn_count += removed_turn_count
                legacy_qa_cleanup_pre_turn_count += int(
                    legacy_qa_cleanup_audit["pre_turn_count"]
                )
                legacy_qa_cleanup_post_turn_count += int(
                    legacy_qa_cleanup_audit["post_turn_count"]
                )
            if padding_cleanup_audit is not None:
                padding_cleanup_processed_row_count += 1
                removed_turn_count = int(padding_cleanup_audit["removed_turn_count"])
                padding_cleanup_affected_row_count += int(removed_turn_count > 0)
                padding_cleanup_removed_turn_count += removed_turn_count
                padding_cleanup_pre_turn_count += int(padding_cleanup_audit["pre_turn_count"])
                padding_cleanup_post_turn_count += int(padding_cleanup_audit["post_turn_count"])

    summary = {
        "schema_version": SCHEMA_VERSION,
        "forbidden_match_mode": forbidden_match_mode,
        "inputs": [str(path.resolve()) for path in input_paths],
        "forbidden_jsonl": [str(path.resolve()) for path in forbidden_paths],
        "input_count": input_count,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "forbidden_row_count": forbidden_row_count,
        "forbidden_exact_identifier_counts": {
            key: len(values) for key, values in forbidden["exact"].items()
        },
        "forbidden_lineage_identifier_count": len(forbidden["lineage"]),
        "exact_forbidden_match_row_count": exact_forbidden_match_row_count,
        "lineage_overlap_row_count": lineage_overlap_row_count,
        "accepted_lineage_overlap_row_count": accepted_lineage_overlap_row_count,
        "lineage_overlap_occurrence_count": lineage_overlap_occurrence_count,
        "lineage_overlap_unique_identifier_count": len(lineage_overlap_values),
        "lineage_overlap_identifiers": sorted(lineage_overlap_values),
        "lineage_group_count": len(lineage_group_counts),
        "lineage_groups_with_multiple_dialogues": sum(
            count > 1 for count in lineage_group_counts.values()
        ),
        "source_graph_event_count": source_event_count,
        "retained_semantic_event_count": retained_event_count,
        "removed_legacy_or_qa_graph_event_count": legacy_or_qa_removed_count,
        "removed_noncanonical_graph_event_count": noncanonical_removed_count,
        "removed_graph_event_count": source_event_count - retained_event_count,
        "require_nonempty_semantic_parent_graph": require_nonempty_semantic_parent_graph,
        "semantic_parent_graph_admission": {
            "require_nonempty": require_nonempty_semantic_parent_graph,
            "empty_input_row_count": empty_semantic_parent_graph_input_count,
            "rejected_empty_row_count": rejected_empty_semantic_parent_graph_count,
            "accepted_empty_row_count": sum(
                parent["semantic_parent_graph"]["event_count"] == 0
                for parent in accepted
            ),
            "fail_closed_requirement_satisfied": (
                not require_nonempty_semantic_parent_graph
                or all(
                    parent["semantic_parent_graph"]["event_count"] > 0
                    for parent in accepted
                )
            ),
        },
        "legacy_qa_dialogue_cleanup": {
            "enabled": True,
            "cleanup_version": LEGACY_QA_DIALOGUE_CLEANUP_VERSION,
            "processed_row_count": legacy_qa_cleanup_processed_row_count,
            "affected_row_count": legacy_qa_cleanup_affected_row_count,
            "blocked_retained_evidence_row_count": legacy_qa_cleanup_blocked_row_count,
            "removed_turn_count": legacy_qa_cleanup_removed_turn_count,
            "pre_turn_count": legacy_qa_cleanup_pre_turn_count,
            "post_turn_count": legacy_qa_cleanup_post_turn_count,
        },
        "deterministic_padding_cleanup": {
            "enabled": strip_deterministic_padding,
            "cleanup_version": DETERMINISTIC_PADDING_CLEANUP_VERSION,
            "known_padding_text_count": len(KNOWN_DETERMINISTIC_PADDING_TEXTS),
            "known_padding_signature_sha256": deterministic_padding_signature_sha256(),
            "processed_row_count": padding_cleanup_processed_row_count,
            "affected_row_count": padding_cleanup_affected_row_count,
            "removed_turn_count": padding_cleanup_removed_turn_count,
            "pre_turn_count": padding_cleanup_pre_turn_count,
            "post_turn_count": padding_cleanup_post_turn_count,
        },
        "source_files_modified": False,
    }
    return accepted, rejected, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate immutable Stage2 train candidates into leakage-safe Stage3 parent_v2 rows."
    )
    parser.add_argument("--input", type=Path, action="append", required=True, help="Stage2 JSONL; repeatable")
    parser.add_argument(
        "--forbidden-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Dev/heldout/test manifest used to exclude sample, parent, and group identities; repeatable",
    )
    parser.add_argument(
        "--forbidden-match-mode",
        choices=sorted(FORBIDDEN_MATCH_MODES),
        default="lineage",
        help=(
            "lineage (default) rejects any lineage identity overlap for compatibility; "
            "exact rejects only same-field generated sample/question/dialogue-hash matches and audits source/lineage overlap"
        ),
    )
    parser.add_argument(
        "--strip-historical-deterministic-padding",
        action="store_true",
        help=(
            "Remove only turns whose text exactly matches the historical deterministic filler "
            "emitted by native graph dialogue expansion. Disabled by default."
        ),
    )
    parser.add_argument(
        "--require-nonempty-semantic-parent-graph",
        action="store_true",
        help=(
            "Fail closed per row when all source graph events are removed as legacy/QA or "
            "noncanonical. Disabled by default for migration compatibility; enable for "
            "formal Stage3 parent construction."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    accepted, rejected, summary = migrate(
        args.input,
        args.forbidden_jsonl,
        forbidden_match_mode=args.forbidden_match_mode,
        strip_deterministic_padding=args.strip_historical_deterministic_padding,
        require_nonempty_semantic_parent_graph=(
            args.require_nonempty_semantic_parent_graph
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "parent_samples.jsonl", accepted)
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)
    summary.update(
        {
            "output_dir": str(args.output_dir.resolve()),
            "parent_samples": str((args.output_dir / "parent_samples.jsonl").resolve()),
            "rejected_samples": str((args.output_dir / "rejected.jsonl").resolve()),
        }
    )
    write_json(args.output_dir / "migration_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
