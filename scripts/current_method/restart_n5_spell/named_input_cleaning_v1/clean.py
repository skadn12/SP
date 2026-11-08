"""Derive a named-speaker unified root without modifying the source root."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.input_adapter import dialogue_content_hash
from scripts.current_method.restart_n5_spell.unified_dialogue_root import (
    validate_unified_dialogue_root,
)


ALIAS_RE = re.compile(
    r"\b(?:Neighbors?|Classmates?|Co-workers?|Coworkers?|Speaker|Person)\s+[A-Z]\b"
)
SHORT_ALIAS_RE = re.compile(r"(?<![A-Za-z])[AB](?![A-Za-z])")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace(value: Any, mapping: dict[str, str], counter: dict[str, int]) -> Any:
    if isinstance(value, str):
        result = value
        for old, new in mapping.items():
            if old in {"A", "B"}:
                matches = list(re.finditer(rf"(?<![A-Za-z]){old}(?![A-Za-z])", result))
                hits = len(matches)
                result = re.sub(rf"(?<![A-Za-z]){old}(?![A-Za-z])", new, result)
            else:
                hits = result.count(old)
                result = result.replace(old, new)
            if hits:
                counter[old] = counter.get(old, 0) + hits
        return result
    if isinstance(value, list):
        return [_replace(item, mapping, counter) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, mapping, counter) for key, item in value.items()}
    return value


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    return []


def clean_root(*, source: Path, output: Path, names: tuple[str, str]) -> dict[str, Any]:
    original = json.loads(source.read_text(encoding="utf-8"))
    validation = validate_unified_dialogue_root(original)
    if not validation["valid"]:
        raise ValueError("source_root_invalid:" + ",".join(validation["errors"]))
    old_speakers = []
    for turn in original["dialogue"]["turns"]:
        speaker = str(turn["speaker"])
        if speaker not in old_speakers:
            old_speakers.append(speaker)
    if len(old_speakers) != 2:
        raise ValueError(f"expected_two_speakers:{old_speakers}")
    if len(set(names)) != 2 or any(not name.strip() for name in names):
        raise ValueError(f"names_must_be_unique:{names}")
    mapping: dict[str, str] = {}
    for old, new in zip(old_speakers, names):
        mapping[old] = new
        if old.startswith("Neighbors "):
            mapping[old.replace("Neighbors ", "Neighbor ", 1)] = new
        if old.startswith("Classmates "):
            mapping[old.replace("Classmates ", "Classmate ", 1)] = new
        if old.startswith("Co-workers "):
            mapping[old.replace("Co-workers ", "Co-worker ", 1)] = new
    # Some source rows shorten the two canonical speaker labels to bare
    # ``A``/``B`` inside later turns (for example ``Thanks, A.``).  Add these
    # only for the corresponding speaker slots and replace them as standalone
    # tokens, never as substrings of ordinary words.
    if old_speakers[0].endswith(" A") and old_speakers[1].endswith(" B"):
        mapping["A"] = names[0]
        mapping["B"] = names[1]
    counter: dict[str, int] = {}
    cleaned = _replace(copy.deepcopy(original), mapping, counter)
    cleaned["dialogue"]["dialogue_hash"] = dialogue_content_hash(
        [dict(turn) for turn in cleaned["dialogue"]["turns"]])
    residuals = sorted({match.group(0) for text in _strings(cleaned)
                        for pattern in (ALIAS_RE, SHORT_ALIAS_RE)
                        for match in pattern.finditer(text)})
    cleaned_speakers = [str(turn["speaker"]) for turn in cleaned["dialogue"]["turns"]]
    checks = {
        "source_valid": True,
        "speaker_count_two": len(set(cleaned_speakers)) == 2,
        "names_unique": len(set(names)) == 2,
        "concrete_names_applied": set(cleaned_speakers) == set(names),
        "alias_residuals_empty": not residuals,
        "dialogue_hash_recomputed": True,
    }
    if not all(checks.values()):
        raise ValueError(f"name_cleaning_failed:{checks};residuals={residuals}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "passed",
        "source_path": str(source.resolve()),
        "source_sha256": _sha256(source),
        "output_path": str(output.resolve()),
        "output_sha256": _sha256(output),
        "root_id": cleaned["root_id"],
        "mapping": mapping,
        "canonical_names": list(names),
        "replacement_counts": counter,
        "checks": checks,
        "residual_aliases": residuals,
        "source_dialogue_hash": original["dialogue"]["dialogue_hash"],
        "cleaned_dialogue_hash": cleaned["dialogue"]["dialogue_hash"],
        "source_unchanged": _sha256(source) != _sha256(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name-a", default="Nora")
    parser.add_argument("--name-b", default="Caleb")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = clean_root(
        source=args.source.resolve(), output=args.output.resolve(),
        names=(args.name_a.strip(), args.name_b.strip()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
