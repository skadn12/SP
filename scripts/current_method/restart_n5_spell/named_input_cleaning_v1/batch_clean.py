"""Deterministic cleaning for Conversation Chronicles and expanded inputs.

This module is deliberately independent from every expansion and QA entrypoint.
It changes only explicitly configured speaker aliases and writes no model or
API artifacts.  Ambiguous short tokens are reported instead of guessed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CLEANING_VERSION = "named_input_cleaning_batch_v1"
CC_SESSION_NAMES = ("first", "second", "third", "fourth", "fifth")

# These are the label families used by the source rows.  Singular and spelling
# variants are included because the dialogue text is not always consistent
# with the speaker array.
_PREFIX_FAMILIES: dict[str, tuple[str, ...]] = {
    "neighbor": ("Neighbor", "Neighbors"),
    "classmate": ("Classmate", "Classmates"),
    "co worker": ("Co-worker", "Co-workers", "Coworker", "Coworkers"),
    "speaker": ("Speaker",),
    "person": ("Person",),
}

_SUFFIXED_LABEL = re.compile(r"\s+([AB])$", re.IGNORECASE)
_SHORT_TOKEN = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")

# A bare token followed by one of these common verbal forms is much more
# likely to be an omitted speaker alias (``A said ...``) than an article.  We
# fail closed for these cases instead of silently leaving a possible alias in
# the cleaned input.  Less obvious cases remain review-worthy by design.
_LIKELY_ALIAS_VERBS = frozenset({
    "said", "says", "was", "were", "is", "are", "am", "be", "been",
    "had", "has", "have", "did", "does", "do", "wanted", "needed",
    "decided", "chose", "felt", "thought", "got", "went", "came",
    "asked", "told", "liked", "loved", "hated", "tried", "planned",
    "continued", "became", "mentioned", "suggested", "offered",
    "expressed", "experienced", "looked", "seemed", "could", "would",
    "will", "should", "might", "may",
})

# In source summaries, ``A and B`` is an explicit shorthand for the two
# declared speakers. It is safe to replace only when both sides are present;
# ordinary phrases such as ``A bakery`` do not match.
_SHORT_PAIR = re.compile(
    r"(?<![A-Za-z])(?:(?:Neighbors?|Classmates?|Co[- ]workers?|"
    r"Coworkers?|Speakers?|Persons?)\s+)?"
    r"(?P<left>[AB])(?P<between>\s+and\s+)"
    r"(?P<right>[AB])(?![A-Za-z])",
    re.IGNORECASE,
)

# A short token is treated as a direct address only when punctuation makes
# that reading explicit.  This deliberately does not match ordinary prose
# such as "A bakery" or "A bit".
_DIRECT_ADDRESS = re.compile(
    r"(?P<prefix>^|,\s+)(?P<label>[AB])(?P<suffix>\s*[,!.?])",
    re.IGNORECASE | re.MULTILINE,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prefix_variants(prefix: str) -> tuple[str, ...]:
    normalized = re.sub(r"[-_]", " ", prefix.strip().casefold())
    normalized = re.sub(r"\s+", " ", normalized)
    for key, variants in _PREFIX_FAMILIES.items():
        if normalized == key or normalized.rstrip("s") == key.rstrip("s"):
            return variants
    return (prefix.strip(),)


def build_replacement_plan(
    speaker_map: Mapping[str, str],
    bare_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a validated, deterministic replacement plan.

    ``speaker_map`` is intentionally explicit.  The order in a source file is
    not used to infer identity, which prevents a first-speaker B/A swap.
    """
    mapping = {str(old).strip(): str(new).strip()
               for old, new in speaker_map.items()}
    if len(mapping) != 2 or any(not old or not new for old, new in mapping.items()):
        raise ValueError("speaker_map_must_contain_two_nonempty_pairs")
    if len(set(mapping.values())) != 2:
        raise ValueError("target_names_must_be_unique_within_root")
    if any(name.casefold() in {"a", "b"} for name in mapping.values()):
        raise ValueError("target_names_must_be_concrete_not_A_or_B")

    # Non-suffixed role labels such as ``Parent``/``Child`` are valid speaker
    # field values but are also ordinary nouns in dialogue.  Keep them in the
    # exact speaker map only; textual replacement is enabled for the explicit
    # A/B label families below, where the source convention is unambiguous.
    alias_map: dict[str, str] = {}
    short_map: dict[str, str] = {}
    suffix_targets: dict[str, str] = {}
    for old, new in mapping.items():
        match = _SUFFIXED_LABEL.search(old)
        if not match:
            continue
        alias_map[old] = new
        suffix = match.group(1).upper()
        prefix = old[:match.start()].strip()
        previous = suffix_targets.get(suffix)
        if previous is not None and previous != new:
            raise ValueError(f"suffix_mapping_conflict:{suffix}")
        suffix_targets[suffix] = new
        for variant in _prefix_variants(prefix):
            alias_map[f"{variant} {suffix}"] = new
    if set(suffix_targets) == {"A", "B"}:
        short_map = dict(suffix_targets)
    if bare_aliases is not None:
        explicit_bare = {
            str(alias).strip().upper(): str(target).strip()
            for alias, target in bare_aliases.items()
        }
        if (set(explicit_bare) - {"A", "B"} or
                set(explicit_bare.values()) - set(mapping.values())):
            raise ValueError("bare_aliases_must_bind_A_or_B_to_target_name")
        if explicit_bare:
            short_map = explicit_bare

    lookup = {key.casefold(): value for key, value in alias_map.items()}
    alternatives = sorted(alias_map, key=len, reverse=True)
    # An empty alternation would match every position when a mapping uses
    # non-suffixed speaker labels (for example ``Parent``/``Child``).  Use a
    # never-match expression until an explicit textual alias is configured.
    full_pattern = (
        re.compile(
            r"(?<![A-Za-z])(?:" + "|".join(re.escape(item)
            for item in alternatives) + r")(?![A-Za-z])",
            re.IGNORECASE,
        ) if alternatives else re.compile(r"(?!x)x")
    )
    pair_prefixes = sorted({alias.rsplit(" ", 1)[0]
                            for alias in alias_map
                            if _SUFFIXED_LABEL.search(alias)}, key=len,
                           reverse=True)
    pair_pattern = re.compile(
        r"(?<![A-Za-z])(?:" + "|".join(re.escape(item)
        for item in pair_prefixes) + r")\s+(?P<left>[AB])"
        r"(?P<between>\s+and\s+)(?P<right>[AB])(?![A-Za-z])",
        re.IGNORECASE,
    ) if pair_prefixes else None
    return {
        "speaker_map": mapping,
        "alias_map": alias_map,
        "alias_lookup": lookup,
        "short_map": short_map,
        "full_pattern": full_pattern,
        "pair_pattern": pair_pattern,
    }


def _replace_full(text: str, plan: Mapping[str, Any], counts: dict[str, int]) -> str:
    pattern = plan["full_pattern"]
    lookup = plan["alias_lookup"]

    def replace(match: re.Match[str]) -> str:
        old = match.group(0)
        new = lookup.get(old.casefold())
        if new is None:
            return old
        counts["full_alias"] = counts.get("full_alias", 0) + 1
        counts[f"full:{old}"] = counts.get(f"full:{old}", 0) + 1
        return new

    return pattern.sub(replace, text)


def _replace_full_pairs(text: str, plan: Mapping[str, Any],
                        counts: dict[str, int]) -> str:
    pattern = plan.get("pair_pattern")
    short_map = plan["short_map"]
    if not short_map:
        return text

    def replace(match: re.Match[str]) -> str:
        left = short_map.get(match.group("left").upper())
        right = short_map.get(match.group("right").upper())
        if left is None or right is None:
            return match.group(0)
        counts["full_alias_pair"] = counts.get("full_alias_pair", 0) + 1
        return f"{left}{match.group('between')}{right}"

    result = pattern.sub(replace, text) if pattern is not None else text

    # Some source summaries omit the role prefix and use bare ``A and B``.
    # The pair is materially less ambiguous than an isolated short token, so
    # replace it only when both sides are present and both are explicitly
    # bound by the mapping.  Ordinary text such as ``A bakery`` cannot match.
    def replace_short_pair(match: re.Match[str]) -> str:
        left = short_map.get(match.group("left").upper())
        right = short_map.get(match.group("right").upper())
        if left is None or right is None:
            return match.group(0)
        counts["short_pair"] = counts.get("short_pair", 0) + 1
        return f"{left}{match.group('between')}{right}"

    return _SHORT_PAIR.sub(replace_short_pair, result)


def _replace_direct(text: str, plan: Mapping[str, Any], counts: dict[str, int]) -> str:
    short_map = plan["short_map"]
    if not short_map:
        return text

    def replace(match: re.Match[str]) -> str:
        label = match.group("label").upper()
        new = short_map.get(label)
        if new is None:
            return match.group(0)
        counts["direct_address"] = counts.get("direct_address", 0) + 1
        counts[f"direct:{label}"] = counts.get(f"direct:{label}", 0) + 1
        return f"{match.group('prefix')}{new}{match.group('suffix')}"

    return _DIRECT_ADDRESS.sub(replace, text)


def clean_text(
    text: str, plan: Mapping[str, Any], counts: dict[str, int],
    *, allow_direct_address: bool = True,
) -> str:
    result = _replace_full_pairs(str(text), plan, counts)
    result = _replace_full(result, plan, counts)
    if allow_direct_address:
        result = _replace_direct(result, plan, counts)
    return result


def _replace_speaker(value: Any, plan: Mapping[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    mapping = plan["speaker_map"]
    if value in mapping:
        return mapping[value]
    folded = value.casefold()
    for old, new in mapping.items():
        if old.casefold() == folded:
            return new
    return value


def _clean_metadata_strings(value: Any, plan: Mapping[str, Any],
                            counts: dict[str, int]) -> Any:
    if isinstance(value, str):
        return clean_text(value, plan, counts, allow_direct_address=True)
    if isinstance(value, list):
        return [_clean_metadata_strings(item, plan, counts) for item in value]
    if isinstance(value, dict):
        return {key: _clean_metadata_strings(item, plan, counts)
                for key, item in value.items()}
    return value


def _validate_cc_shape(value: Mapping[str, Any]) -> None:
    required = {"dataID", "relationship", "time_interval", "summary"}
    required.update(f"{name}_session_{field}"
                   for name in CC_SESSION_NAMES
                   for field in ("dialogue", "speakers"))
    missing = sorted(required - set(value))
    if missing:
        raise ValueError("conversation_chronicles_fields_missing:" + ",".join(missing))
    for name in CC_SESSION_NAMES:
        dialogue = value[f"{name}_session_dialogue"]
        speakers = value[f"{name}_session_speakers"]
        if not isinstance(dialogue, list) or not isinstance(speakers, list):
            raise ValueError(f"session_arrays_invalid:{name}")
        if len(dialogue) != len(speakers):
            raise ValueError(f"session_alignment_invalid:{name}")


def clean_conversation_chronicles_record(
    record: Mapping[str, Any], speaker_map: Mapping[str, str],
    bare_aliases: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    """Clean one raw Conversation Chronicles record without changing schema."""
    _validate_cc_shape(record)
    plan = build_replacement_plan(speaker_map, bare_aliases)
    cleaned = copy.deepcopy(dict(record))
    counts: dict[str, int] = {}

    # Relationship, intervals, and summaries are metadata.  Full labels are
    # safe to replace there; short A/B tokens are intentionally left alone.
    for key in ("relationship", "time_interval", "summary"):
        cleaned[key] = _clean_metadata_strings(cleaned[key], plan, counts)

    for name in CC_SESSION_NAMES:
        speaker_key = f"{name}_session_speakers"
        dialogue_key = f"{name}_session_dialogue"
        cleaned[speaker_key] = [
            _replace_speaker(value, plan)
            for value in cleaned[speaker_key]
        ]
        cleaned[dialogue_key] = [
            clean_text(value, plan, counts, allow_direct_address=True)
            for value in cleaned[dialogue_key]
        ]

    audit = {
        "format": "conversation_chronicles_record",
        "canonical_names": sorted(set(plan["speaker_map"].values())),
        "speaker_map": dict(plan["speaker_map"]),
        "alias_variants": sorted(plan["alias_map"]),
    }
    return cleaned, counts, audit


def clean_expanded_dialogue(
    expanded: Mapping[str, Any], speaker_map: Mapping[str, str],
    bare_aliases: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    """Clean a simple ``expanded_dialogue*.json`` snapshot for audit/replay."""
    if not isinstance(expanded, Mapping) or not isinstance(expanded.get("turns"), list):
        raise ValueError("expanded_dialogue_turns_missing")
    plan = build_replacement_plan(speaker_map, bare_aliases)
    cleaned = copy.deepcopy(dict(expanded))
    counts: dict[str, int] = {}

    # Clean only user-visible metadata strings.  Hashes and IDs are retained;
    # the complete hash is recomputed below when present.
    for key, value in list(cleaned.items()):
        if key in {"turns", "complete_dialogue_sha256"}:
            continue
        cleaned[key] = _clean_metadata_strings(value, plan, counts)
    new_turns: list[dict[str, Any]] = []
    for raw_turn in cleaned["turns"]:
        if not isinstance(raw_turn, Mapping):
            raise ValueError("expanded_dialogue_turn_invalid")
        turn = dict(raw_turn)
        turn["speaker"] = _replace_speaker(turn.get("speaker"), plan)
        if "speaker_label" in turn:
            turn["speaker_label"] = _replace_speaker(
                turn.get("speaker_label"), plan)
        turn["text"] = clean_text(turn.get("text", ""), plan, counts,
                                   allow_direct_address=True)
        new_turns.append(turn)
    cleaned["turns"] = new_turns
    cleaned["turn_count"] = len(new_turns)

    old_complete_hash = cleaned.get("complete_dialogue_sha256")
    if old_complete_hash is not None:
        body = copy.deepcopy(cleaned)
        body.pop("complete_dialogue_sha256", None)
        cleaned["complete_dialogue_sha256"] = sha256_bytes(
            canonical_json(body).encode("utf-8"))
    audit = {
        "format": "expanded_dialogue",
        "canonical_names": sorted(set(plan["speaker_map"].values())),
        "speaker_map": dict(plan["speaker_map"]),
        "alias_variants": sorted(plan["alias_map"]),
        "source_complete_dialogue_sha256": old_complete_hash,
        "cleaned_complete_dialogue_sha256": cleaned.get(
            "complete_dialogue_sha256"),
    }
    return cleaned, counts, audit


def clean_unified_dialogue_root(
    root: Mapping[str, Any], speaker_map: Mapping[str, str],
    bare_aliases: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    """Clean a unified root while preserving its input schema and provenance."""
    from ..input_adapter import dialogue_content_hash
    from ..unified_dialogue_root import validate_unified_dialogue_root

    if not isinstance(root, Mapping):
        raise ValueError("unified_root_not_object")
    before = validate_unified_dialogue_root(root)
    if not before["valid"]:
        raise ValueError("unified_root_invalid:" + ",".join(before["errors"]))
    plan = build_replacement_plan(speaker_map, bare_aliases)
    cleaned = copy.deepcopy(dict(root))
    counts: dict[str, int] = {}
    dialogue = cleaned["dialogue"]
    for turn in dialogue["turns"]:
        turn["speaker"] = _replace_speaker(turn.get("speaker"), plan)
        turn["text"] = clean_text(turn.get("text", ""), plan, counts)
    for session in cleaned.get("session_metadata") or []:
        session["summary"] = clean_text(session.get("summary", ""), plan, counts)
    source_metadata = cleaned.get("source_metadata")
    if isinstance(source_metadata, Mapping):
        source_metadata["relationship"] = _clean_metadata_strings(
            source_metadata.get("relationship", ""), plan, counts)
    dialogue["dialogue_hash"] = dialogue_content_hash(
        [dict(turn) for turn in dialogue["turns"]])
    after = validate_unified_dialogue_root(cleaned)
    if not after["valid"]:
        raise ValueError("cleaned_unified_root_invalid:" + ",".join(after["errors"]))
    return cleaned, counts, {
        "format": "unified_dialogue_root",
        "canonical_names": sorted(set(plan["speaker_map"].values())),
        "speaker_map": dict(plan["speaker_map"]),
        "alias_variants": sorted(plan["alias_map"]),
        "source_dialogue_hash": root["dialogue"]["dialogue_hash"],
        "cleaned_dialogue_hash": dialogue["dialogue_hash"],
    }


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)


def _dialogue_texts(value: Mapping[str, Any], fmt: str) -> list[str]:
    if fmt == "conversation_chronicles_record":
        texts: list[str] = []
        for name in CC_SESSION_NAMES:
            texts.extend(str(item) for item in value[f"{name}_session_dialogue"])
        return texts
    if fmt == "unified_dialogue_root":
        turns = (value.get("dialogue") or {}).get("turns") or []
    else:
        turns = value.get("turns", [])
    return [str(turn.get("text", "")) for turn in turns]


def _unresolved_short_tokens(text: str, *, enabled: bool) -> list[str]:
    if not enabled:
        return []
    unresolved: list[str] = []
    for match in _SHORT_TOKEN.finditer(text):
        token = match.group(1).upper()
        # Preserve ordinary article phrases while rejecting common subject-
        # verb shapes that would otherwise hide a bare speaker alias.
        next_index = match.end()
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1
        next_word_match = re.match(r"[A-Za-z]+", text[next_index:])
        next_word = (next_word_match.group(0).casefold()
                     if next_word_match else "")
        if token == "A" and next_word and next_word not in _LIKELY_ALIAS_VERBS:
            continue
        unresolved.append(match.group(0))
    return unresolved


def _full_alias_residuals(value: Any, plan: Mapping[str, Any]) -> list[str]:
    # Check only aliases declared for this root.  A generic ``role + A`` regex
    # would reject legitimate prose such as ``co-workers a lot`` in a
    # Neighbors dialogue.  The explicit alias map still covers all known
    # singular/plural/spelling variants for the configured source speakers.
    residuals: set[str] = set()
    for alias, target in plan["alias_map"].items():
        if alias.casefold() == target.casefold():
            continue
        pattern = re.compile(
            r"(?<![A-Za-z])" + re.escape(alias) + r"(?![A-Za-z])",
            re.IGNORECASE,
        )
        residuals.update(match.group(0) for text in _iter_strings(value)
                         for match in pattern.finditer(text))
    return sorted(residuals)


def _visible_strings(value: Mapping[str, Any], fmt: str) -> list[str]:
    """Return text/speaker fields, excluding IDs, hashes, and provenance."""
    visible: list[str] = []
    if fmt == "conversation_chronicles_record":
        visible.extend(str(value.get("relationship") or ""))
        visible.extend(str(item) for item in value.get("time_interval") or [])
        visible.extend(str(item) for item in value.get("summary") or [])
        for name in CC_SESSION_NAMES:
            visible.extend(str(item) for item in value[f"{name}_session_speakers"])
            visible.extend(str(item) for item in value[f"{name}_session_dialogue"])
    elif fmt == "unified_dialogue_root":
        visible.extend(str(row.get("speaker") or "")
                       for row in (value.get("dialogue") or {}).get("turns") or [])
        visible.extend(str(row.get("text") or "")
                       for row in (value.get("dialogue") or {}).get("turns") or [])
        visible.extend(str(row.get("summary") or "")
                       for row in value.get("session_metadata") or [])
        source_metadata = value.get("source_metadata")
        if isinstance(source_metadata, Mapping):
            visible.append(str(source_metadata.get("relationship") or ""))
    else:
        for row in value.get("turns") or []:
            visible.extend([str(row.get("speaker") or ""),
                            str(row.get("speaker_label") or ""),
                            str(row.get("text") or "")])
        # Expanded snapshots may carry summaries/relationships beside turns.
        # Include user-facing metadata while excluding technical identifiers,
        # hashes, and provenance paths that are not dialogue content.
        technical_keys = {
            "turns", "root_id", "sample_id", "lineage_group", "source",
            "provenance", "complete_dialogue_sha256",
        }
        for key, item in value.items():
            folded = str(key).casefold()
            if key in technical_keys or any(token in folded for token in (
                    "hash", "sha", "count", "index", "version", "path", "id")):
                continue
            visible.extend(_iter_strings(item))
    return visible


def validate_cleaned_record(
    original: Mapping[str, Any], cleaned: Mapping[str, Any],
    speaker_map: Mapping[str, str], *, fmt: str,
    bare_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Structural and residual checks; no semantic judgment is attempted."""
    plan = build_replacement_plan(speaker_map, bare_aliases)
    expected_names = set(plan["speaker_map"].values())
    errors: list[str] = []
    checks: dict[str, bool] = {}

    if fmt == "conversation_chronicles_record":
        try:
            _validate_cc_shape(original)
            _validate_cc_shape(cleaned)
            checks["session_lengths_preserved"] = all(
                len(original[f"{name}_session_dialogue"]) ==
                len(cleaned[f"{name}_session_dialogue"])
                and len(original[f"{name}_session_speakers"]) ==
                len(cleaned[f"{name}_session_speakers"])
                for name in CC_SESSION_NAMES)
            checks["schema_keys_preserved"] = set(original) == set(cleaned)
            speakers = [speaker for name in CC_SESSION_NAMES
                        for speaker in cleaned[f"{name}_session_speakers"]]
            checks["two_unique_concrete_speakers"] = (
                set(speakers) == expected_names and len(set(speakers)) == 2)
            texts = _dialogue_texts(cleaned, fmt)
        except Exception as error:
            checks.update({"session_lengths_preserved": False,
                           "schema_keys_preserved": False,
                           "two_unique_concrete_speakers": False})
            texts = []
            errors.append(f"shape:{type(error).__name__}:{error}")
    elif fmt == "unified_dialogue_root":
        original_dialogue = original.get("dialogue") or {}
        cleaned_dialogue = cleaned.get("dialogue") or {}
        original_turns = list(original_dialogue.get("turns") or [])
        cleaned_turns = list(cleaned_dialogue.get("turns") or [])
        checks["turn_count_preserved"] = len(original_turns) == len(cleaned_turns)
        immutable_fields = (
            "turn_id", "session_id", "partition", "source_id",
            "source_session_index", "source_turn_index",
        )
        checks["turn_refs_and_order_preserved"] = (
            len(original_turns) == len(cleaned_turns) and all(
                all(original_turns[index].get(field) ==
                    cleaned_turns[index].get(field) for field in immutable_fields)
                for index in range(len(original_turns))))
        checks["schema_keys_preserved"] = set(original) == set(cleaned)
        speakers = [str(turn.get("speaker", "")) for turn in cleaned_turns]
        checks["two_unique_concrete_speakers"] = (
            set(speakers) == expected_names and len(set(speakers)) == 2)
        original_sessions = original.get("session_metadata") or []
        cleaned_sessions = cleaned.get("session_metadata") or []
        checks["session_metadata_structure_preserved"] = (
            len(original_sessions) == len(cleaned_sessions) and all(
                {key: value for key, value in before.items() if key != "summary"} ==
                {key: value for key, value in after.items() if key != "summary"}
                for before, after in zip(original_sessions, cleaned_sessions)))
        checks["dialogue_hash_recomputed"] = (
            isinstance(cleaned_dialogue.get("dialogue_hash"), str) and
            cleaned_dialogue.get("dialogue_hash") !=
            original_dialogue.get("dialogue_hash"))
        texts = _dialogue_texts(cleaned, fmt)
    else:
        original_turns = list(original.get("turns") or [])
        cleaned_turns = list(cleaned.get("turns") or [])
        checks["turn_count_preserved"] = len(original_turns) == len(cleaned_turns)
        immutable_fields = ("turn_ref", "session_ref", "scene_ref", "origin")
        checks["turn_refs_and_order_preserved"] = (
            len(original_turns) == len(cleaned_turns) and all(
                all(original_turns[index].get(field) ==
                    cleaned_turns[index].get(field) for field in immutable_fields)
                for index in range(len(original_turns))))
        speakers = [str(turn.get("speaker", "")) for turn in cleaned_turns]
        checks["two_unique_concrete_speakers"] = (
            set(speakers) == expected_names and len(set(speakers)) == 2)
        checks["schema_keys_preserved"] = set(original) == set(cleaned)
        texts = _dialogue_texts(cleaned, fmt)

    visible = _visible_strings(cleaned, fmt)
    full_residuals = _full_alias_residuals({"visible": visible}, plan)
    direct_residuals = sorted({match.group(0)
                               for text in visible
                               for match in _DIRECT_ADDRESS.finditer(text)})
    unresolved_short = sorted({token for text in visible
                               for token in _unresolved_short_tokens(
                                   text, enabled=bool(plan["short_map"]))})
    checks["full_alias_residuals_empty"] = not full_residuals
    checks["direct_address_residuals_empty"] = not direct_residuals
    checks["unresolved_short_tokens_empty"] = not unresolved_short
    checks["names_unique"] = len(expected_names) == 2
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {
        "valid": not errors,
        "checks": checks,
        "errors": errors,
        "full_alias_residuals": full_residuals,
        "direct_address_residuals": direct_residuals,
        "unresolved_short_tokens": unresolved_short,
    }


def record_format(value: Mapping[str, Any]) -> str:
    if (isinstance(value.get("dialogue"), Mapping) and
            isinstance(value["dialogue"].get("turns"), list)):
        return "unified_dialogue_root"
    if isinstance(value.get("turns"), list):
        return "expanded_dialogue"
    if all(f"{name}_session_dialogue" in value for name in CC_SESSION_NAMES):
        return "conversation_chronicles_record"
    raise ValueError("unsupported_input_format")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _safe_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    if not result:
        raise ValueError("source_identity_cannot_form_path")
    return result


def _root_id(value: Mapping[str, Any], fmt: str) -> str:
    if fmt == "conversation_chronicles_record":
        source_id = str(value.get("dataID") or "").strip()
        if not source_id:
            raise ValueError("conversation_chronicles_data_id_missing")
        suffix = re.sub(r"[^a-z0-9]+", "-", source_id.casefold()).strip("-")
        return f"cc-train-{suffix}"
    root_id = str(value.get("root_id") or "").strip()
    if not root_id:
        raise ValueError("root_id_missing")
    return root_id


def _source_speakers(value: Mapping[str, Any], fmt: str) -> list[str]:
    values: list[Any]
    if fmt == "conversation_chronicles_record":
        values = [speaker for name in CC_SESSION_NAMES
                  for speaker in value[f"{name}_session_speakers"]]
    elif fmt == "unified_dialogue_root":
        values = [turn.get("speaker")
                  for turn in value["dialogue"]["turns"]]
    else:
        values = [turn.get("speaker") for turn in value["turns"]]
    speakers: list[str] = []
    for raw in values:
        speaker = str(raw or "").strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)
    if len(speakers) != 2:
        raise ValueError(f"expected_two_speakers:{speakers}")
    return speakers


def load_mapping(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping) and isinstance(value.get("mappings"), Mapping):
        value = value["mappings"]
    if not isinstance(value, Mapping) or not value:
        raise ValueError("mapping_file_must_be_nonempty_object")
    result: dict[str, dict[str, Any]] = {}
    for identity, entry in value.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"mapping_entry_must_be_object:{identity}")
        if "speaker_map" in entry or "speakers" in entry:
            raw_speakers = entry.get("speaker_map", entry.get("speakers"))
            if not isinstance(raw_speakers, Mapping):
                raise ValueError(f"speaker_map_must_be_object:{identity}")
            normalized = {str(old).strip(): str(new).strip()
                          for old, new in raw_speakers.items()}
            bare = entry.get("bare_aliases")
            if bare is not None and not isinstance(bare, Mapping):
                raise ValueError(f"bare_aliases_must_be_object:{identity}")
            normalized_bare = ({str(old).strip().upper(): str(new).strip()
                               for old, new in bare.items()}
                              if isinstance(bare, Mapping) else None)
        else:
            normalized = {str(old).strip(): str(new).strip()
                          for old, new in entry.items()}
            normalized_bare = None
        build_replacement_plan(normalized, normalized_bare)
        result[str(identity)] = {
            "speaker_map": normalized,
            "bare_aliases": normalized_bare,
        }
    return result


def _mapping_candidates(value: Mapping[str, Any], fmt: str) -> list[str]:
    root_id = _root_id(value, fmt)
    candidates = [root_id]
    if fmt == "conversation_chronicles_record":
        candidates.append(str(value.get("dataID") or ""))
    else:
        source = value.get("source")
        if isinstance(source, Mapping):
            candidates.append(str(source.get("source_id") or ""))
    if root_id.startswith("cc-train-"):
        candidates.append(root_id[len("cc-train-"):])
    return [item for item in dict.fromkeys(candidates) if item]


def _resolve_mapping(
    value: Mapping[str, Any], fmt: str,
    mappings: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, str], dict[str, str] | None]:
    mapping_key = next((key for key in _mapping_candidates(value, fmt)
                        if key in mappings), None)
    if mapping_key is None:
        raise KeyError("explicit_mapping_missing:" +
                       "|".join(_mapping_candidates(value, fmt)))
    entry = mappings[mapping_key]
    mapping = dict(entry["speaker_map"])
    bare_aliases = (dict(entry["bare_aliases"])
                    if entry.get("bare_aliases") is not None else None)
    speakers = _source_speakers(value, fmt)
    if set(mapping) != set(speakers):
        raise ValueError(
            f"mapping_source_speakers_mismatch:{mapping_key}:"
            f"expected={speakers}:configured={sorted(mapping)}")
    return mapping_key, mapping, bare_aliases


def _clean_dispatch(
    value: Mapping[str, Any], fmt: str, mapping: Mapping[str, str],
    bare_aliases: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    if fmt == "conversation_chronicles_record":
        return clean_conversation_chronicles_record(value, mapping, bare_aliases)
    if fmt == "unified_dialogue_root":
        return clean_unified_dialogue_root(value, mapping, bare_aliases)
    if fmt == "expanded_dialogue":
        return clean_expanded_dialogue(value, mapping, bare_aliases)
    raise ValueError(f"unsupported_input_format:{fmt}")


def _iter_input(path: Path):
    if path.is_dir():
        unified = sorted(path.glob("*/unified_root.json"))
        files = unified or sorted(path.glob("**/*.json"))
        for item in files:
            yield {
                "source_path": item.resolve(),
                "source_sha256": sha256_file(item),
                "source_record_sha256": sha256_file(item),
                "source_row_index": None,
                "value": json.loads(item.read_text(encoding="utf-8")),
            }
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".jsonl":
        with path.open("rb") as handle:
            for index, raw_line in enumerate(handle):
                if not raw_line.strip():
                    continue
                yield {
                    "source_path": path.resolve(),
                    "source_sha256": None,
                    "source_record_sha256": sha256_bytes(raw_line),
                    "source_row_index": index,
                    "value": json.loads(raw_line),
                }
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    values = value if isinstance(value, list) else [value]
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise ValueError(f"json_record_not_object:{index}")
        yield {
            "source_path": path.resolve(),
            "source_sha256": sha256_file(path),
            "source_record_sha256": sha256_bytes(
                canonical_json(item).encode("utf-8")),
            "source_row_index": index if isinstance(value, list) else None,
            "value": item,
        }


def _input_hash(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    entries = [{"path": str(item.relative_to(path)), "sha256": sha256_file(item)}
               for item in sorted(path.glob("**/*.json"))]
    return sha256_bytes(canonical_json(entries).encode("utf-8"))


def run_batch(
    *, input_path: Path, output_root: Path, mapping_path: Path,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Materialize a new named-input directory without changing its source."""
    input_path = input_path.resolve()
    output_root = output_root.resolve()
    mapping_path = mapping_path.resolve()
    if output_root.exists():
        raise FileExistsError(f"output_root_exists:{output_root}")
    mappings = load_mapping(mapping_path)
    input_sha_before = _input_hash(input_path)
    mapping_sha = sha256_file(mapping_path)
    selected: list[dict[str, Any]] = []
    matched_mapping_keys: set[str] = set()

    # Mapping coverage selects the batch for large JSONL files.  A missing
    # mapping never causes positional name inference.
    for source in _iter_input(input_path):
        value = source["value"]
        if not isinstance(value, Mapping):
            continue
        try:
            fmt = record_format(value)
            candidates = _mapping_candidates(value, fmt)
        except ValueError:
            continue
        if not any(key in mappings for key in candidates):
            continue
        mapping_key, mapping, bare_aliases = _resolve_mapping(value, fmt, mappings)
        matched_mapping_keys.add(mapping_key)
        root_id = _root_id(value, fmt)
        cleaned, counts, audit = _clean_dispatch(
            value, fmt, mapping, bare_aliases)
        repeated, repeated_counts, repeated_audit = _clean_dispatch(
            value, fmt, mapping, bare_aliases)
        validation = validate_cleaned_record(
            value, cleaned, mapping, fmt=fmt, bare_aliases=bare_aliases)
        repeatable = (
            canonical_json(cleaned) == canonical_json(repeated) and
            counts == repeated_counts and audit == repeated_audit)
        if not repeatable:
            validation = copy.deepcopy(validation)
            validation["valid"] = False
            validation["checks"]["repeatable"] = False
            validation["errors"].append("repeatable")
        else:
            validation["checks"]["repeatable"] = True
        selected.append({
            **source, "root_id": root_id, "format": fmt,
            "mapping_key": mapping_key, "mapping": mapping,
            "bare_aliases": bare_aliases,
            "cleaned": cleaned, "replacement_counts": counts,
            "audit": audit, "validation": validation,
            "repeatable": repeatable,
        })

    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(f"selected_count_mismatch:{len(selected)}/{expected_count}")
    if not selected:
        raise ValueError("no_mapped_records_found")
    root_ids = [row["root_id"] for row in selected]
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("duplicate_root_id_in_batch")

    output_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    raw_rows = [row for row in selected
                if row["format"] == "conversation_chronicles_record"]
    raw_row_positions = {id(row): index for index, row in enumerate(raw_rows)}
    cleaned_jsonl = output_root / "cleaned_records.jsonl"
    raw_line_bytes: list[bytes] = []
    if raw_rows:
        raw_line_bytes = [
            (canonical_json(row["cleaned"]) + "\n").encode("utf-8")
            for row in raw_rows]
        cleaned_jsonl.write_bytes(b"".join(raw_line_bytes))
    cleaned_jsonl_sha = sha256_file(cleaned_jsonl) if raw_rows else None

    for row in selected:
        root_dir = output_root / "roots" / _safe_id(row["root_id"])
        validation = row["validation"]
        output_path: Path | None = None
        normalized_validation: dict[str, Any] | None = None
        if validation["valid"]:
            if row["format"] == "conversation_chronicles_record":
                from ..conversation_chronicles_adapter import build_unified_root
                raw_index = raw_row_positions[id(row)]
                unified = build_unified_root(
                    row["cleaned"], source_path=cleaned_jsonl,
                    source_file_sha256=str(cleaned_jsonl_sha),
                    source_row_index=raw_index,
                    source_record_sha256=sha256_bytes(
                        raw_line_bytes[raw_index]))
                from ..unified_dialogue_root import validate_unified_dialogue_root
                normalized_validation = validate_unified_dialogue_root(
                    unified, expected_source_path=cleaned_jsonl,
                    expected_source_file_sha256=cleaned_jsonl_sha,
                    expected_session_count=5)
                output_path = root_dir / "unified_root.json"
                _write_json(output_path, unified)
                _write_json(root_dir / "cleaned_source_record.json", row["cleaned"])
            elif row["format"] == "unified_dialogue_root":
                output_path = root_dir / "unified_root.json"
                _write_json(output_path, row["cleaned"])
            else:
                output_path = root_dir / "expanded_dialogue.json"
                _write_json(output_path, row["cleaned"])
        record = {
            "root_id": row["root_id"], "format": row["format"],
            "source_path": str(row["source_path"]),
            "source_row_index": row["source_row_index"],
            "source_record_sha256": row["source_record_sha256"],
            "mapping_key": row["mapping_key"],
            "speaker_map": row["mapping"],
            "bare_aliases": row["bare_aliases"],
            "canonical_names": sorted(set(row["mapping"].values())),
            "replacement_counts": row["replacement_counts"],
            "repeatable": row["repeatable"],
            "validation": validation,
            "normalized_validation": normalized_validation,
            "output_path": str(output_path) if output_path else None,
            "output_sha256": sha256_file(output_path) if output_path else None,
        }
        _write_json(root_dir / "name_cleaning_report.json", record)
        records.append(record)

    input_sha_after = _input_hash(input_path)
    unused_mapping_keys = sorted(set(mappings) - matched_mapping_keys)
    failed = [row for row in records if not row["validation"]["valid"] or
              (row["normalized_validation"] is not None and
               not row["normalized_validation"]["valid"])]
    all_names = [name for row in records for name in row["canonical_names"]]
    global_checks = {
        "expected_count": expected_count is None or len(records) == expected_count,
        "all_records_valid": not failed,
        "all_repeatable": all(row["repeatable"] for row in records),
        "source_unchanged": input_sha_before == input_sha_after,
        "mapping_coverage_exact": not unused_mapping_keys,
        "root_ids_unique": len(root_ids) == len(set(root_ids)),
        "names_unique_within_each_root": all(
            len(row["canonical_names"]) == 2 and
            len(set(row["canonical_names"])) == 2 for row in records),
    }
    manifest = {
        "manifest_version": "named_input_cleaning_batch_manifest_v1",
        "cleaning_version": CLEANING_VERSION,
        "status": "passed" if all(global_checks.values()) else "failed",
        "input_path": str(input_path),
        "input_sha256_before": input_sha_before,
        "input_sha256_after": input_sha_after,
        "mapping_path": str(mapping_path),
        "mapping_sha256": mapping_sha,
        "output_root": str(output_root),
        "record_count": len(records),
        "expected_count": expected_count,
        "global_checks": global_checks,
        "unused_mapping_keys": unused_mapping_keys,
        "cross_root_name_collisions": sorted(
            name for name in set(all_names) if all_names.count(name) > 1),
        "cleaned_jsonl": str(cleaned_jsonl) if raw_rows else None,
        "cleaned_jsonl_sha256": cleaned_jsonl_sha,
        "roots_directory": str(output_root / "roots"),
        "records": records,
        "pipeline_integration": {
            "mode": "precleaned_input_directory",
            "cleaning_in_production_pipeline": False,
            "api_calls": 0, "solver_calls": 0,
            "planner_calls": 0, "realizer_calls": 0,
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    summary = [
        "# Named Input Cleaning Batch", "",
        f"- status: `{manifest['status']}`",
        f"- records: `{len(records)}`",
        f"- source unchanged: `{str(global_checks['source_unchanged']).lower()}`",
        f"- repeatable: `{str(global_checks['all_repeatable']).lower()}`",
        f"- cleaned roots: `{output_root / 'roots'}`", "",
        "This is an offline preprocessing artifact. No production expansion or QA "
        "entrypoint imports or invokes this cleaner.", "",
    ]
    (output_root / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    if manifest["status"] != "passed":
        raise RuntimeError(f"batch_cleaning_failed:{output_root / 'manifest.json'}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args(argv)
    result = run_batch(
        input_path=args.input, output_root=args.output_root,
        mapping_path=args.mapping, expected_count=args.expected_count)
    print(json.dumps({
        "status": result["status"], "record_count": result["record_count"],
        "output_root": result["output_root"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
