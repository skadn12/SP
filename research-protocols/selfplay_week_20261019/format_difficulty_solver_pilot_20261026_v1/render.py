"""Lossless, display-only conversion of existing conversational sources.

Only ``short_text`` and ``refs`` belong in a Solver request.  Citation mappings,
field offsets, and validation data are private conversion/audit artifacts.
No date is extracted from dialogue text.  Optional dates are read only from
explicit ISO calendar-date metadata on turns or in ``source.scene_metadata``.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any


CITATION_INSTRUCTION = (
    "Cite a turn as scene-id:Txxx using its complete Scene header and local "
    "turn label, for example R03:A:S01:T001 or P01:T001. "
    "Local turn labels restart in each scene. Date: unknown means no calendar "
    "date is available in the source metadata."
)
_REF_PATTERN = re.compile(
    r"(?:(?P<root>.+):)?(?P<scene>P\d+|R\d+(?::[A-Z])?:S\d+):T\d+\Z"
)
_SCENE_PATTERN = re.compile(r"(?:P\d+|R\d+(?::[A-Z])?:S\d+)\Z")
_DATE_KEYS = ("calendar_date", "scene_date", "date")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_scene(value: str, root_id: str) -> str:
    if value.startswith(root_id + ":"):
        value = value[len(root_id) + 1 :]
    parent = re.fullmatch(r"parent_session_(\d+)", value, re.IGNORECASE)
    if parent:
        value = "P" + parent.group(1)
    value = value.upper()
    if not _SCENE_PATTERN.fullmatch(value):
        raise ValueError(f"Unrecognized scene/session identifier: {value!r}")
    return value


def _scene_for_turn(turn: dict[str, Any], root_id: str) -> str:
    ref = turn.get("turn_ref")
    if not isinstance(ref, str):
        raise ValueError("Every turn must have a string turn_ref")
    match = _REF_PATTERN.fullmatch(ref)
    if not match:
        raise ValueError(f"Unrecognized complete turn_ref: {ref!r}")
    if match.group("root") not in (None, root_id):
        raise ValueError(f"turn_ref root differs from source root: {ref!r}")
    scene = match.group("scene")
    for key in ("scene_ref", "session_ref"):
        if turn.get(key) is not None:
            declared = _normalize_scene(turn[key], root_id)
            if declared != scene:
                raise ValueError(f"{key} disagrees with turn_ref: {ref!r}")
    return scene


def _explicit_dates(metadata: dict[str, Any], origin: str) -> list[tuple[str, str]]:
    result = []
    for key in _DATE_KEYS:
        value = metadata.get(key)
        if value in (None, "", "unknown"):
            continue
        if not isinstance(value, str):
            raise ValueError(f"{origin}.{key} must be an explicit ISO date")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{origin}.{key} is not an ISO calendar date") from exc
        if parsed.isoformat() != value:
            raise ValueError(f"{origin}.{key} must use YYYY-MM-DD")
        result.append((value, f"{origin}.{key}"))
    return result


def _scene_dates(source: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Read metadata only; relative or dialogue dates are never interpreted."""
    metadata = source.get("scene_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("scene_metadata, when supplied, must be keyed by scene ID")
    normalized_metadata = {}
    for key, value in metadata.items():
        scene = _normalize_scene(key, source["root_id"])
        if scene in normalized_metadata:
            raise ValueError(f"Duplicate scene_metadata entry: {scene}")
        if not isinstance(value, dict):
            raise ValueError(f"scene_metadata[{key!r}] must be an object")
        normalized_metadata[scene] = value
    known_scenes = {group["scene_id"] for group in groups}
    if set(normalized_metadata) - known_scenes:
        raise ValueError("scene_metadata names scenes that are absent from turns")
    dates = {}
    for group in groups:
        scene = group["scene_id"]
        candidates = _explicit_dates(
            normalized_metadata.get(scene, {}), f"scene_metadata.{scene}"
        )
        for index, turn in group["indexed_turns"]:
            candidates.extend(_explicit_dates(turn, f"turns[{index}]"))
        unique = {value for value, _ in candidates}
        if len(unique) > 1:
            raise ValueError(f"Conflicting explicit calendar dates in scene {scene}")
        dates[scene] = {
            "date": next(iter(unique)) if unique else "unknown",
            "date_metadata_fields": [origin for _, origin in candidates],
        }
    return dates


def restore_original(rendered: dict[str, Any]) -> str:
    """Recover the original display exactly using private Unicode field offsets."""
    short_text = rendered["short_text"]
    lines = []
    for alias in rendered["refs"]:
        span = rendered["turn_spans"][alias]
        speaker = short_text[span["speaker_start"] : span["speaker_end"]]
        text = short_text[span["text_start"] : span["text_end"]]
        lines.append(f"[{rendered['short_to_original'][alias]}] {speaker}: {text}")
    return "\n".join(lines)


def render_source(source: dict[str, Any]) -> dict[str, Any]:
    """Render one source without changing speakers, utterances, or turn order.

    Returns original_text, short_text, refs (in source order), short_to_original,
    scenes (per-scene statistics), turn_spans (private reversible Unicode
    offsets), and lossless_validation.  Noncontiguous repeated scenes,
    conflicting identifiers/dates, duplicate original refs, and malformed
    source fields fail explicitly rather than being repaired or regrouped.
    """
    root_id = source.get("root_id")
    if not isinstance(root_id, str) or not root_id or "\n" in root_id or "\r" in root_id:
        raise ValueError("source.root_id must be a nonempty single-line string")
    turns = source.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("source.turns must be a nonempty list")
    groups: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    seen_original_refs: set[str] = set()
    originals = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"turns[{index}] must be an object")
        for key in ("speaker", "text"):
            if not isinstance(turn.get(key), str):
                raise ValueError(f"turns[{index}].{key} must be a string")
        scene = _scene_for_turn(turn, root_id)
        if turn["turn_ref"] in seen_original_refs:
            raise ValueError(f"Duplicate turn_ref: {turn['turn_ref']}")
        seen_original_refs.add(turn["turn_ref"])
        if not groups or groups[-1]["scene_id"] != scene:
            if scene in seen_scenes:
                raise ValueError(f"Scene {scene} is noncontiguous; refusing to regroup turns")
            seen_scenes.add(scene)
            groups.append({"scene_id": scene, "indexed_turns": []})
        groups[-1]["indexed_turns"].append((index, turn))
        originals.append(f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}")
    date_metadata = _scene_dates(source, groups)

    pieces = [f"Source: {root_id}\n{CITATION_INSTRUCTION}\n"]
    offset = len(pieces[0])
    refs = []
    mapping = {}
    spans = {}
    scenes = []
    for group in groups:
        scene = group["scene_id"]
        scene_date = date_metadata[scene]
        header = f"\nScene {scene} | Date: {scene_date['date']}\n"
        pieces.append(header)
        offset += len(header)
        scene_start = offset
        for local_index, (source_index, turn) in enumerate(group["indexed_turns"], 1):
            local_label = f"T{local_index:03d}"
            alias = f"{scene}:{local_label}"
            if alias in mapping:
                raise ValueError(f"Duplicate short citation: {alias}")
            prefix = f"[{local_label}] "
            speaker_start = offset + len(prefix)
            speaker_end = speaker_start + len(turn["speaker"])
            text_start = speaker_end + len(": ")
            text_end = text_start + len(turn["text"])
            line = f"{prefix}{turn['speaker']}: {turn['text']}\n"
            pieces.append(line)
            offset += len(line)
            refs.append(alias)
            mapping[alias] = turn["turn_ref"]
            spans[alias] = {
                "source_turn_index": source_index,
                "speaker_start": speaker_start,
                "speaker_end": speaker_end,
                "text_start": text_start,
                "text_end": text_end,
            }
        indices = [index for index, _ in group["indexed_turns"]]
        scenes.append({
            "scene_id": scene,
            **scene_date,
            "turn_count": len(indices),
            "first_source_turn_index": indices[0],
            "last_source_turn_index": indices[-1],
            "first_short_ref": refs[-len(indices)],
            "last_short_ref": refs[-1],
            "rendered_turn_characters": offset - scene_start,
        })
    original_text = "\n".join(originals)
    result = {
        "original_text": original_text,
        "short_text": "".join(pieces),
        "refs": refs,
        "short_to_original": mapping,
        "scenes": scenes,
        "turn_spans": spans,
    }
    restored = restore_original(result)
    field_equality = all(
        result["short_text"][spans[alias]["speaker_start"] : spans[alias]["speaker_end"]]
        == turn["speaker"]
        and result["short_text"][spans[alias]["text_start"] : spans[alias]["text_end"]]
        == turn["text"]
        for alias, turn in zip(refs, turns, strict=True)
    )
    validation = {
        "passed": restored == original_text and field_equality,
        "turn_count": len(turns),
        "scene_count": len(scenes),
        "unknown_date_scene_count": sum(scene["date"] == "unknown" for scene in scenes),
        "unique_original_refs": len(seen_original_refs) == len(turns),
        "unique_short_refs": len(set(refs)) == len(turns),
        "contiguous_scenes": True,
        "speaker_text_order_preserved": field_equality,
        "original_text_roundtrip_equal": restored == original_text,
        "date_policy": "explicit ISO calendar-date metadata only; no dialogue inference",
        "offset_unit": "Python Unicode code points",
        "original_text_sha256": _sha256(original_text),
        "restored_original_text_sha256": _sha256(restored),
        "original_characters": len(original_text),
        "short_characters": len(result["short_text"]),
    }
    if not validation["passed"]:
        raise ValueError("Lossless render validation failed")
    result["lossless_validation"] = validation
    return result


def selfcheck(source_dir: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    """Audit actual frozen inputs; never modify them or run a model."""
    base = Path(__file__).resolve().parent
    source_dir = source_dir or base.parent / "feedback_closed_loop_20261024_v1" / "sources"
    checks = []
    for suffix in ("218519", "222937", "225575", "229686"):
        path = source_dir / f"cc-train-episode-{suffix}.json"
        before = path.read_bytes()
        converted = render_source(json.loads(before))
        after = path.read_bytes()
        if before != after:
            raise ValueError(f"Source changed during conversion audit: {path}")
        checks.append({
            "source_path": str(path),
            "source_sha256": hashlib.sha256(before).hexdigest(),
            "source_unchanged": before == after,
            **converted["lossless_validation"],
            "scenes": converted["scenes"],
        })
    audit = {
        "schema_version": "lossless_short_format_conversion_audit_v1",
        "all_passed": all(check["passed"] and check["source_unchanged"] for check in checks),
        "model_calls": 0,
        "sources": checks,
    }
    if output:
        output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--audit-output", type=Path, default=Path(__file__).with_name("CONVERSION_AUDIT.json"))
    args = parser.parse_args()
    audit = selfcheck(args.source_dir, args.audit_output)
    print(json.dumps({"all_passed": audit["all_passed"], "sources": len(audit["sources"]), "audit_output": str(args.audit_output)}))
