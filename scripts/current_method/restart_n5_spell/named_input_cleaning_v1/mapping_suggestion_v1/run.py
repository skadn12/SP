"""Generate and validate concrete speaker names for selected roots.

The model is only a naming assistant.  It never rewrites a transcript.  The
result is accepted only after the deterministic batch cleaner's mapping
validator approves the exact source speaker set.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..batch_clean import build_replacement_plan, load_mapping
from ...core import atomic_json, atomic_text, sha256_file
from ...live_backends import OpenAICompatibleChallengerClient, load_challenger_config


RUN_VERSION = "named_input_mapping_suggestion_v1"
PROMPT_VERSION = "named_input_mapping_prompt_v1"
MAX_NAME_LENGTH = 80
ALIAS_RE = re.compile(
    r"^(?:[AB]|Speaker\s+[AB]|Person\s+[AB]|(?:Neighbor|Neighbors|"
    r"Classmate|Classmates|Co-worker|Co-workers|Coworker|Coworkers)\s+[AB])$",
    re.IGNORECASE,
)
FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"json_load_failed:{path}:{type(error).__name__}") from error


def _extract_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    fenced = FENCE_RE.fullmatch(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model_json_invalid") from error
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError as nested:
            raise ValueError("model_json_invalid") from nested
    if not isinstance(value, dict):
        raise ValueError("model_output_not_object")
    return value


def _validate_candidate(
    candidate: Mapping[str, Any], *, expected_speakers: list[str], root_id: str,
) -> dict[str, Any]:
    raw_map = candidate.get("speaker_map")
    if not isinstance(raw_map, Mapping):
        return {"valid": False, "errors": ["speaker_map_missing"]}
    mapping = {str(key).strip(): str(value).strip() for key, value in raw_map.items()}
    errors: list[str] = []
    if set(mapping) != set(expected_speakers):
        errors.append("speaker_set_mismatch")
    names = list(mapping.values())
    if any(not name for name in names):
        errors.append("name_empty")
    if len(set(name.casefold() for name in names)) != len(names):
        errors.append("names_not_unique_within_root")
    if any(len(name) > MAX_NAME_LENGTH for name in names):
        errors.append("name_too_long")
    if any(ALIAS_RE.fullmatch(name) for name in names):
        errors.append("alias_name_used")
    if any(name.casefold() in {"a", "b"} for name in names):
        errors.append("bare_alias_name_used")
    if not errors:
        try:
            build_replacement_plan(mapping)
        except (TypeError, ValueError) as error:
            errors.append(f"cleaner_mapping_rejected:{error}")
    return {
        "valid": not errors,
        "errors": errors,
        "root_id": root_id,
        "speaker_map": mapping,
    }


def _prompt(record: Mapping[str, Any]) -> str:
    speakers = [str(value) for value in record.get("source_speakers") or []]
    return (
        "You are assisting an offline dialogue preprocessing step. "
        "Assign two ordinary concrete personal names to the two speaker labels. "
        "Names are arbitrary identity markers, not inferred facts about the story. "
        "Keep the exact speaker labels as JSON keys. Return only one JSON object "
        "with this shape: {\"speaker_map\": {\"label1\": \"Name1\", "
        "\"label2\": \"Name2\"}}.\n"
        "Rules: use two different first names; do not use A, B, Speaker A/B, "
        "Person A/B, or relationship labels as names; do not add keys or prose.\n"
        f"root_id: {record.get('root_id')}\n"
        f"relationship: {record.get('relationship') or 'unknown'}\n"
        f"speaker_labels: {json.dumps(speakers, ensure_ascii=False)}\n"
    )


def _records(selection_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = _load_json(selection_path.resolve())
    if not isinstance(selection, Mapping):
        raise ValueError("selection_not_object")
    if selection.get("selection_version") != "named_input_sequential_selection_v1":
        raise ValueError("selection_version_invalid")
    rows = selection.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection_records_missing")
    normalized = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(normalized) != len(rows):
        raise ValueError("selection_record_not_object")
    return dict(selection), normalized


def run_generation(
    *, selection_path: Path, output_root: Path, api_config_path: Path,
    max_attempts: int = 2, temperature: float | None = None,
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    output_root = output_root.resolve()
    api_config_path = api_config_path.resolve()
    if output_root.exists():
        raise FileExistsError(f"mapping_run_root_exists:{output_root}")
    if max_attempts <= 0:
        raise ValueError("max_attempts_must_be_positive")
    selection, rows = _records(selection_path)
    config = load_challenger_config(api_config_path)
    if temperature is not None:
        config = replace(config, temperature=float(temperature))
    client = OpenAICompatibleChallengerClient(config)
    output_root.mkdir(parents=True, exist_ok=False)
    responses_root = output_root / "responses"
    records: list[dict[str, Any]] = []
    for row in rows:
        root_id = str(row["root_id"])
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", root_id)
        prompt = _prompt(row)
        prompt_path = output_root / "prompts" / f"{safe}.txt"
        atomic_text(prompt_path, prompt)
        accepted: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.complete(
                    prompt=prompt if attempt == 1 else prompt + (
                        "\nYour previous response was invalid. Return only the exact "
                        "JSON object requested above."),
                    max_tokens=256,
                    json_object=True,
                )
                raw = str(response.raw_text)
                response_path = responses_root / f"{safe}__attempt_{attempt:02d}.txt"
                atomic_text(response_path, raw)
                parsed = _extract_object(raw)
                validation = _validate_candidate(
                    parsed, expected_speakers=list(row["source_speakers"]),
                    root_id=root_id,
                )
                attempt_record = {
                    "attempt": attempt,
                    "status": "accepted" if validation["valid"] else "rejected",
                    "response_path": str(response_path),
                    "response_sha256": sha256_file(response_path),
                    "validation": validation,
                    "metadata": response.metadata,
                }
                attempts.append(attempt_record)
                if validation["valid"]:
                    accepted = validation
                    break
            except Exception as error:
                attempts.append({
                    "attempt": attempt,
                    "status": "technical_failure",
                    "error": f"{type(error).__name__}:{error}",
                })
        records.append({
            "root_id": root_id,
            "source_id": row.get("source_id"),
            "source_row_index": row.get("source_row_index"),
            "source_record_sha256": row.get("source_record_sha256"),
            "source_speakers": row.get("source_speakers"),
            "prompt_path": str(prompt_path),
            "status": "accepted" if accepted else "failed",
            "speaker_map": accepted.get("speaker_map") if accepted else None,
            "attempts": attempts,
        })
    accepted_rows = [row for row in records if row["status"] == "accepted"]
    mappings = {
        str(row["source_id"] or row["root_id"]): {
            "speaker_map": dict(row["speaker_map"]),
            "source_row_index": row["source_row_index"],
            "source_record_sha256": row["source_record_sha256"],
        }
        for row in accepted_rows
    }
    mapping_path = output_root / "name_mapping.json"
    atomic_json(mapping_path, {
        "schema_version": "named_input_cleaning_mapping_model_assisted_v1",
        "status": "passed" if len(accepted_rows) == len(records) else "partial",
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "api_config_sha256": sha256_file(api_config_path),
        "prompt_version": PROMPT_VERSION,
        "mappings": mappings,
    })
    # Re-load through the production cleaner's parser as the final contract.
    load_mapping(mapping_path)
    report = {
        "run_version": RUN_VERSION,
        "status": "completed" if len(accepted_rows) == len(records) else "partial",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "api_config_sha256": sha256_file(api_config_path),
        "model": config.model,
        "temperature": config.temperature,
        "root_count": len(records),
        "accepted_count": len(accepted_rows),
        "failed_count": len(records) - len(accepted_rows),
        "max_attempts": max_attempts,
        "records": records,
        "mapping_path": str(mapping_path),
        "api_calls": sum(len(row["attempts"]) for row in records),
        "cleaning_calls": 0,
        "transcript_writes": 0,
        "mainline_untouched": True,
    }
    atomic_json(output_root / "mapping_generation_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args(argv)
    report = run_generation(
        selection_path=args.selection, output_root=args.output_root,
        api_config_path=args.api_config, max_attempts=args.max_attempts,
        temperature=args.temperature,
    )
    print(json.dumps({
        "status": report["status"], "root_count": report["root_count"],
        "accepted_count": report["accepted_count"],
        "mapping_path": report["mapping_path"],
        "api_calls": report["api_calls"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
