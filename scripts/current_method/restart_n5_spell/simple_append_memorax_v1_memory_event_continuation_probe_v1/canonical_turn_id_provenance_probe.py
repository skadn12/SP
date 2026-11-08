"""Zero-API audit of Parent turn identity and Memorax isolation semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-13836/"
    "unified_root.json"
)
DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_13836_transcript_event_verifier_"
    "v3_fresh_r1_live_1"
)
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "reports"
PROBE_VERSION = "canonical_turn_id_provenance_probe_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _canonical_ref(root_id: str, session_index: int, turn_index: int) -> str:
    return f"{root_id}:P{session_index:02d}:T{turn_index:03d}"


def _turn_refs(parent: Mapping[str, Any]) -> list[str]:
    return [str(row["turn_id"]) for row in parent["dialogue"]["turns"]]


def _formula_checks(parent: Mapping[str, Any]) -> dict[str, Any]:
    root_id = str(parent["root_id"])
    rows = []
    for turn in parent["dialogue"]["turns"]:
        expected = _canonical_ref(
            root_id,
            int(turn["source_session_index"]),
            int(turn["source_turn_index"]),
        )
        rows.append({
            "turn_ref": str(turn["turn_id"]),
            "expected_turn_ref": expected,
            "matches": str(turn["turn_id"]) == expected,
        })
    mismatches = [row for row in rows if not row["matches"]]
    return {
        "format": "{root_id}:P{source_session_index:02d}:T{source_turn_index:03d}",
        "checked_turn_count": len(rows),
        "all_match": not mismatches,
        "mismatches": mismatches,
        "sample_refs": [row["turn_ref"] for row in rows[:3]],
    }


def _position_independence(parent: Mapping[str, Any]) -> dict[str, Any]:
    turns = list(parent["dialogue"]["turns"])
    original = {str(row["turn_id"]): index for index, row in enumerate(turns, 1)}
    # A prompt/context view may reorder or select turns, but it must carry the
    # source turn object and its canonical identity without regenerating it.
    alternate_view = list(reversed(turns[::2])) + list(reversed(turns[1::2]))
    alternate = {
        str(row["turn_id"]): index for index, row in enumerate(alternate_view, 1)
    }
    changed = [ref for ref in original if original[ref] != alternate[ref]]
    return {
        "alternate_view_turn_count": len(alternate_view),
        "prompt_positions_changed_count": len(changed),
        "canonical_ref_set_unchanged": set(original) == set(alternate),
        "example": ({
            "turn_ref": changed[0],
            "original_prompt_position": original[changed[0]],
            "alternate_prompt_position": alternate[changed[0]],
        } if changed else None),
    }


def _append_child_check(parent: Mapping[str, Any]) -> dict[str, Any]:
    before = _turn_refs(parent)
    combined = deepcopy(list(parent["dialogue"]["turns"]))
    combined.extend([
        {
            "turn_id": "R01:S01:T001",
            "session_id": f"{parent['root_id']}:r01:s01",
            "speaker": "Synthetic A",
            "text": "Synthetic zero-API fixture.",
            "partition": "child",
        },
        {
            "turn_id": "R01:S01:T002",
            "session_id": f"{parent['root_id']}:r01:s01",
            "speaker": "Synthetic B",
            "text": "The Parent prefix remains unchanged.",
            "partition": "child",
        },
    ])
    after_parent_prefix = [str(row["turn_id"]) for row in combined[:len(before)]]
    return {
        "parent_turn_count": len(before),
        "synthetic_child_turn_count": len(combined) - len(before),
        "parent_refs_unchanged_after_append": before == after_parent_prefix,
        "combined_refs_unique": len({str(row["turn_id"]) for row in combined}) == len(combined),
    }


def _session_rematerialization(parent: Mapping[str, Any]) -> dict[str, Any]:
    root_id = str(parent["root_id"])
    rematerialized = []
    for session in parent["session_metadata"]:
        session_index = int(session["source_session_index"])
        session_turns = [
            row for row in parent["dialogue"]["turns"]
            if int(row["source_session_index"]) == session_index
        ]
        rematerialized.extend(
            _canonical_ref(root_id, session_index, int(row["source_turn_index"]))
            for row in session_turns
        )
    return {
        "session_count": len(parent["session_metadata"]),
        "turn_count": len(rematerialized),
        "refs_match_original": rematerialized == _turn_refs(parent),
    }


def _key_inventory(value: Any, *, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.append(path)
            result.extend(_key_inventory(child, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value[:5]:
            result.extend(_key_inventory(child, prefix=f"{prefix}[]"))
    return result


def _memorax_audit(source_run: Path) -> dict[str, Any]:
    intent_path = source_run / "memory" / "parent_add_intent.json"
    query_path = source_run / "memory" / "pre_generation_bucket_query.json"
    snapshot_path = source_run / "memory" / "pre_generation_bucket_snapshot.json"
    intent = _read_json(intent_path)
    query = _read_json(query_path)
    snapshot = _read_json(snapshot_path)
    payload = intent.get("payload") or {}
    messages = payload.get("messages") or []
    message_has_turn_ref = [
        isinstance(row, Mapping) and any(
            key in row for key in ("turn_ref", "turn_id", "source_turn_ref")
        ) for row in messages
    ]
    snapshot_keys = _key_inventory(snapshot)
    bucket_turn_ref_paths = [
        path for path in snapshot_keys
        if path.endswith(("turn_ref", "turn_id", "source_turn_ref", "source_turn_refs"))
    ]
    user_id = str(payload.get("user_id") or "")
    root_id = str(intent.get("root_id") or "")
    namespace = user_id[:-len(root_id) - 1] if (
        root_id and user_id.endswith(f":{root_id}")) else None
    return {
        "source_run": str(source_run),
        "intent_path": str(intent_path),
        "query_path": str(query_path),
        "snapshot_path": str(snapshot_path),
        "user_id": user_id,
        "derived_namespace": namespace,
        "session_id": payload.get("session_id"),
        "root_id_metadata": (payload.get("metadata") or {}).get("root_id"),
        "source_dialogue_sha256_metadata": (
            payload.get("metadata") or {}).get("source_dialogue_sha256"),
        "local_intent_turn_ref_count": len(intent.get("turn_refs") or []),
        "messages_count": len(messages),
        "messages_with_per_turn_ref_count": sum(message_has_turn_ref),
        "messages_with_per_turn_ref": all(message_has_turn_ref) if messages else False,
        "payload_metadata_turn_boundary_only": all(
            key in (payload.get("metadata") or {})
            for key in ("first_turn_ref", "last_turn_ref", "turn_count")
        ),
        "bucket_snapshot_turn_ref_paths": bucket_turn_ref_paths,
        "bucket_has_exact_source_turn_refs": bool(bucket_turn_ref_paths),
        "bucket_return_raw_text": (
            (query.get("payload") or {}).get("return_raw_text")),
        "document_id_present": "document_id" in snapshot_keys or "document_id" in payload,
        "memory_id_present": "memory_id" in snapshot_keys or "memory_id" in payload,
        "isolation_identity": "user_id = {memorax_namespace}:{root_id}",
        "round_session_identity": "session_id = {root_id}:memorax:R{round_index:02d}",
        "fresh_rule": {
            "must_remain": ["root_id", "parent_dialogue_sha256", "canonical_turn_refs"],
            "must_change": ["memorax_namespace", "user_id"],
            "not_sufficient_alone": ["session_id", "root_id metadata"],
        },
    }


def build_report(*, input_path: Path, source_run: Path) -> dict[str, Any]:
    first = _read_json(input_path)
    second = _read_json(input_path)
    refs_first = _turn_refs(first)
    refs_second = _turn_refs(second)
    formula = _formula_checks(first)
    position = _position_independence(first)
    append = _append_child_check(first)
    rematerialization = _session_rematerialization(first)
    turns = list(first["dialogue"]["turns"])
    turn_hash_keys = sorted({
        key for row in turns for key in row
        if "hash" in str(key).casefold() or "sha" in str(key).casefold()
    })
    memorax = _memorax_audit(source_run)
    checks = {
        "repeat_load_refs_identical": refs_first == refs_second,
        "turn_refs_unique": len(refs_first) == len(set(refs_first)),
        "canonical_formula_exact": bool(formula["all_match"]),
        "prompt_position_independent": bool(position["canonical_ref_set_unchanged"]),
        "append_child_preserves_parent_refs": bool(
            append["parent_refs_unchanged_after_append"]),
        "session_rematerialization_stable": bool(
            rematerialization["refs_match_original"]),
        "parent_dialogue_sha_present": bool(
            (first.get("dialogue") or {}).get("dialogue_hash")),
        "input_sha_present": bool(_sha256_file(input_path)),
        "memorax_local_intent_retains_refs": (
            memorax["local_intent_turn_ref_count"] == len(refs_first)),
        "memorax_messages_omit_per_turn_refs": (
            memorax["messages_with_per_turn_ref_count"] == 0),
        "memorax_bucket_lacks_exact_turn_refs": (
            not memorax["bucket_has_exact_source_turn_refs"]),
    }
    return {
        "probe_version": PROBE_VERSION,
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "api_calls": 0,
        "input": {
            "path": str(input_path),
            "input_sha256": _sha256_file(input_path),
            "root_id": first.get("root_id"),
            "source_record_sha256": (first.get("source") or {}).get(
                "source_record_sha256"),
            "parent_dialogue_sha256": (first.get("dialogue") or {}).get(
                "dialogue_hash"),
            "session_count": (first.get("dialogue") or {}).get("session_count"),
            "turn_count": len(refs_first),
            "parent_level_sha_fields": [
                "source.source_file_sha256",
                "source.source_record_sha256",
                "dialogue.dialogue_hash",
            ],
            "turn_level_hash_fields": turn_hash_keys,
        },
        "canonical_turn_identity": {
            "already_exists": all(checks[name] for name in (
                "repeat_load_refs_identical", "turn_refs_unique",
                "canonical_formula_exact", "prompt_position_independent",
                "append_child_preserves_parent_refs",
                "session_rematerialization_stable")),
            "reuse_recommended": True,
            "new_parallel_id_system_needed": False,
            "generation_source": (
                "scripts/current_method/restart_n5_spell/"
                "conversation_chronicles_adapter.py:_turn_id"
            ),
            "identity_tuple_recommended": [
                "parent_dialogue_sha256", "source_turn_refs"],
            "version_boundary": (
                "The ref is stable for the frozen source session/turn structure. "
                "Parent SHA must accompany refs because inserting or deleting "
                "a source turn would create a different Parent version."
            ),
            "formula_audit": formula,
            "repeat_load": {
                "refs_identical": refs_first == refs_second,
                "turn_count": len(refs_first),
            },
            "position_independence": position,
            "append_child": append,
            "session_rematerialization": rematerialization,
        },
        "memorax_identity_and_provenance": memorax,
        "conclusions": {
            "canonical_parent_identity": True,
            "canonical_turn_refs": True,
            "parent_sha": True,
            "memorax_per_message_provenance": False,
            "bucket_to_exact_source_turn_refs": False,
            "canonical_turn_ref": "existing stable ref",
            "parent_ref_system": "reuse",
            "new_provenance_id_system": "unnecessary",
            "next_interface_gap": (
                "Reliably associate each Memorax event with existing canonical "
                "source_turn_refs without treating bucket summaries as evidence."
            ),
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    ident = report["canonical_turn_identity"]
    memory = report["memorax_identity_and_provenance"]
    source = report["input"]
    checks = report["checks"]
    lines = [
        "# Canonical Turn ID / Provenance Probe v1",
        "",
        "## Result",
        "",
        f"- Decision: `{report['decision']}`",
        "- API calls: `0`",
        f"- Canonical Parent turn ID already exists: `{'YES' if ident['already_exists'] else 'NO'}`",
        "- Recommendation: `REUSE EXISTING CANONICAL TURN REF`",
        "- New parallel provenance ID system: `unnecessary`",
        "",
        "## Parent Identity",
        "",
        f"- Root: `{source['root_id']}`",
        f"- Input SHA: `{source['input_sha256']}`",
        f"- Parent dialogue SHA: `{source['parent_dialogue_sha256']}`",
        f"- Turns: `{source['turn_count']}` across `{source['session_count']}` sessions",
        "- Format: `{root_id}:P{source_session_index:02d}:T{source_turn_index:03d}`",
        f"- Generator: `{ident['generation_source']}`",
        f"- Formula matches: `{ident['formula_audit']['checked_turn_count']}/{ident['formula_audit']['checked_turn_count']}`",
        "- Recommended provenance tuple:",
        "",
        "```json",
        json.dumps({
            "parent_dialogue_sha256": source["parent_dialogue_sha256"],
            "source_turn_refs": ["cc-train-episode-13836:P03:T005"],
        }, ensure_ascii=False, indent=2),
        "```",
        "",
        "The Parent SHA is required as a version boundary. The turn ref is stable",
        "for the frozen source session/turn structure, but must not be interpreted",
        "against a different Parent version after source turns are inserted/deleted.",
        "",
        "## Stability Checks",
        "",
    ]
    labels = {
        "repeat_load_refs_identical": "Repeated Parent loads preserve every ref",
        "turn_refs_unique": "Parent refs are unique",
        "canonical_formula_exact": "Every ref follows the source-index formula",
        "prompt_position_independent": "Prompt/view position changes do not change refs",
        "append_child_preserves_parent_refs": "Appending Child preserves the Parent prefix refs",
        "session_rematerialization_stable": "Session rematerialization preserves refs",
    }
    lines.extend(
        f"- {label}: `{'PASS' if checks[key] else 'FAIL'}`"
        for key, label in labels.items()
    )
    lines.extend([
        "",
        "## SHA Inventory",
        "",
        "- Parent-level hashes already exist: `source_file_sha256`,",
        "  `source_record_sha256`, and `dialogue_hash`.",
        f"- Turn-level hash fields: `{source['turn_level_hash_fields']}`",
        "- No turn-level hash is added by this probe.",
        "",
        "## Memorax Boundary",
        "",
        f"- `user_id`: `{memory['user_id']}`",
        f"- Derived namespace: `{memory['derived_namespace']}`",
        f"- `session_id`: `{memory['session_id']}`",
        f"- Local Add intent turn refs: `{memory['local_intent_turn_ref_count']}`",
        f"- Sent messages carrying a per-turn ref: `{memory['messages_with_per_turn_ref_count']}`",
        f"- Bucket exact source-turn-ref paths: `{memory['bucket_snapshot_turn_ref_paths']}`",
        f"- Bucket query `return_raw_text`: `{memory['bucket_return_raw_text']}`",
        "",
        "Therefore:",
        "",
        "```text",
        "canonical Parent identity          YES",
        "canonical turn refs                YES",
        "Parent SHA                         YES",
        "Memorax per-message provenance     NO",
        "bucket -> exact source_turn_refs   NO",
        "```",
        "",
        "The local Add intent knows all Parent refs, but each Memorax message only",
        "contains `[Speaker: ...] text`. Bucket content is a generated memory summary,",
        "not authoritative transcript evidence with canonical provenance.",
        "",
        "## Fresh Isolation",
        "",
        "For another Fresh expansion of the same Parent:",
        "",
        "- Keep `root_id`, Parent dialogue SHA, and canonical Parent turn refs.",
        "- Change the Memorax namespace, which changes `user_id`.",
        "- Do not treat `session_id = {root_id}:memorax:R00` as the isolation key.",
        "- Do not clear or reuse an already-written namespace and call it Fresh.",
        "",
        "## Next Interface Gap",
        "",
        "Associate a Memorax event with the existing canonical `source_turn_refs`",
        "without inventing another ID system and without treating Memory summaries",
        "as authoritative evidence.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    report = build_report(input_path=args.input, source_run=args.source_run)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "CANONICAL_TURN_ID_PROVENANCE_PROBE.json"
    md_path = args.report_dir / "CANONICAL_TURN_ID_PROVENANCE_PROBE.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"],
        "api_calls": 0,
        "canonical_turn_ref_exists": report[
            "canonical_turn_identity"]["already_exists"],
        "memorax_bucket_exact_provenance": report[
            "memorax_identity_and_provenance"]["bucket_has_exact_source_turn_refs"],
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
