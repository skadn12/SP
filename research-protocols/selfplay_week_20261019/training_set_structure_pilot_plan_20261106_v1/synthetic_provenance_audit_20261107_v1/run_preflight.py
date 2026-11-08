#!/usr/bin/env python3
"""Read-only provenance audit for the three R03 synthetic extensions.

This audit reads the frozen CANDIDATE_0 files, the canonical FULL_SOURCE files
named by execution_20261107_v2/PRIVATE_BINDINGS.json, and their frozen anonymous
SOURCE projections. It writes only its own receipt in this directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent
V2 = PLAN / "execution_20261107_v2"
BINDINGS = V2 / "PRIVATE_BINDINGS.json"
CANDIDATE_BASE = (
    PLAN.parent / "feedback_closed_loop_stable_v2_20261025_v1" / "sources"
)
ROOTS = (
    "cc-train-episode-25652",
    "cc-train-episode-75241",
    "cc-train-episode-77882",
)
EXPECTED_SCENES = [
    f"R03:{route}:S{scene:02d}"
    for route in ("A", "B")
    for scene in range(1, 4)
]
TURN_RE = re.compile(r"(R03):([AB]):(S[0-9]{2}):T([0-9]{3})\Z")
ANON_TURN_RE = re.compile(r"(S[0-9]{2}):T([0-9]{3})\Z")
EXPECTED_ORIGIN = "codex_synthetic_extension"


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    raw = path.read_bytes()
    obj = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=strict_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    return obj


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_binding(path: Path, expected_sha: str | None = None,
                   expected_bytes: int | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "bytes": len(raw),
        "sha256_matches_manifest": (
            actual_sha == expected_sha if expected_sha is not None else None
        ),
        "bytes_match_manifest": (
            len(raw) == expected_bytes if expected_bytes is not None else None
        ),
    }


def audit_root(root: str, binding: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    lineage_findings: list[str] = []
    expected_scenes = list(EXPECTED_SCENES)
    candidate_path = CANDIDATE_BASE / root / "CANDIDATE_0.json"
    canonical_path = Path(binding["canonical_path"])
    projection_path = V2 / binding["projection_path"]

    candidate_hash = record_binding(candidate_path)
    canonical_hash = record_binding(
        canonical_path, binding.get("canonical_sha256"),
        binding.get("canonical_bytes"),
    )
    projection_hash = record_binding(
        projection_path, binding.get("projection_sha256"),
    )
    candidate = read_json(candidate_path)
    canonical = read_json(canonical_path)
    projection = read_json(projection_path)

    if candidate.get("root_id") != root:
        errors.append("candidate_root_id_mismatch")
    if canonical.get("root_id") != root:
        errors.append("canonical_root_id_mismatch")
    if canonical.get("root_id") != binding.get("root_id", root):
        errors.append("manifest_root_id_mismatch")
    if canonical_hash["sha256_matches_manifest"] is False:
        errors.append("canonical_sha256_manifest_mismatch")
    if canonical_hash["bytes_match_manifest"] is False:
        errors.append("canonical_bytes_manifest_mismatch")
    if projection_hash["sha256_matches_manifest"] is False:
        errors.append("projection_sha256_manifest_mismatch")

    scenes = candidate.get("scenes")
    if not isinstance(scenes, list):
        scenes = []
        errors.append("candidate_scenes_not_list")
    seen_scene_refs = [s.get("scene_ref") for s in scenes if isinstance(s, dict)]
    if seen_scene_refs != expected_scenes:
        errors.append("candidate_scene_sequence_mismatch")

    candidate_turns: list[dict[str, Any]] = []
    scene_rows = []
    row_provenance = []
    for scene in scenes:
        if not isinstance(scene, dict):
            errors.append("candidate_scene_not_object")
            continue
        scene_ref = scene.get("scene_ref")
        expected_route = scene_ref.split(":")[1] if isinstance(scene_ref, str) and scene_ref.count(":") == 2 else None
        if scene.get("route") != expected_route:
            errors.append("candidate_route_mismatch:" + str(scene_ref))
        turns = scene.get("turns")
        if not isinstance(turns, list):
            turns = []
            errors.append("candidate_scene_turns_not_list:" + str(scene_ref))
        if len(turns) != 8:
            errors.append("candidate_scene_turn_count_not_8:" + str(scene_ref))
        refs = []
        for index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                errors.append("candidate_turn_not_object:" + str(scene_ref))
                continue
            turn_ref = turn.get("turn_ref")
            expected_ref = f"{scene_ref}:T{index:03d}"
            if turn_ref != expected_ref:
                errors.append("candidate_turn_ref_not_contiguous:" + str(turn_ref))
            parsed = TURN_RE.fullmatch(turn_ref) if isinstance(turn_ref, str) else None
            if not parsed:
                errors.append("candidate_turn_ref_unparseable:" + str(turn_ref))
                synthetic_scene_ref = None
            else:
                synthetic_scene_ref = ":".join(parsed.group(i) for i in (1, 2, 3))
            if turn.get("origin") != EXPECTED_ORIGIN:
                errors.append("candidate_origin_mismatch:" + str(turn_ref))
            if not isinstance(turn.get("speaker"), str) or not turn.get("speaker").strip():
                errors.append("candidate_speaker_missing:" + str(turn_ref))
            if not isinstance(turn.get("text"), str) or not turn.get("text").strip():
                errors.append("candidate_text_missing:" + str(turn_ref))
            if turn_ref in refs:
                errors.append("duplicate_turn_ref_in_scene:" + str(turn_ref))
            refs.append(turn_ref)
            candidate_turns.append(turn)
            session_value = turn.get("session_ref")
            scene_value = turn.get("scene_ref")
            unbound = not isinstance(session_value, str) or not session_value.strip()
            row_provenance.append({
                "turn_ref": turn_ref,
                "synthetic_scene_ref": synthetic_scene_ref,
                "origin": turn.get("origin"),
                "session_ref": {
                    "present": "session_ref" in turn,
                    "empty_or_missing": unbound,
                },
                "scene_ref_field": {
                    "present": "scene_ref" in turn,
                    "empty_or_missing": (
                        "scene_ref" not in turn
                        or scene_value is None
                        or (isinstance(scene_value, str) and not scene_value.strip())
                    ),
                },
                "provenance_status": (
                    "UNBOUND_SYNTHETIC_EXTENSION" if unbound
                    else "SESSION_REF_PRESENT_REQUIRES_VALIDATION"
                ),
            })
        scene_rows.append({
            "scene_ref": scene_ref,
            "route": scene.get("route"),
            "turn_count": len(turns),
            "turn_refs": refs,
        })

    if len(candidate_turns) != 48:
        errors.append("candidate_total_turn_count_not_48")
    candidate_keys = [tuple(sorted(t.keys())) for t in candidate_turns]
    candidate_key_counts = Counter(", ".join(keys) for keys in candidate_keys)

    canonical_turns = canonical.get("turns")
    if not isinstance(canonical_turns, list):
        canonical_turns = []
        errors.append("canonical_turns_not_list")
    extension_indexed = [
        (index, turn) for index, turn in enumerate(canonical_turns)
        if isinstance(turn, dict) and turn.get("origin") == EXPECTED_ORIGIN
    ]
    extension_turns = [turn for _, turn in extension_indexed]
    extension_indices = [index for index, _ in extension_indexed]
    if len(extension_turns) != 48:
        errors.append("canonical_synthetic_turn_count_not_48")
    if extension_indices and extension_indices != list(
        range(extension_indices[0], extension_indices[0] + len(extension_indices))
    ):
        errors.append("canonical_synthetic_turns_not_contiguous")
    if extension_indices and extension_indices[-1] != len(canonical_turns) - 1:
        errors.append("canonical_synthetic_turns_not_at_end")
    canonical_content = [
        {key: turn.get(key) for key in ("turn_ref", "speaker", "text", "origin")}
        for turn in extension_turns
    ]
    candidate_content = [
        {key: turn.get(key) for key in ("turn_ref", "speaker", "text", "origin")}
        for turn in candidate_turns
    ]
    if canonical_content != candidate_content:
        # The historical CANDIDATE_0 and the builder's manifest canonical
        # source are different artifacts. Keep that identity fact explicit;
        # do not silently treat either artifact as the other.
        lineage_findings.append("NON_BUILDER_CANDIDATE_DIFFERS_FROM_MANIFEST_CANONICAL")

    source_turns = projection.get("turns")
    if not isinstance(source_turns, list):
        source_turns = []
        errors.append("projection_turns_not_list")
    projected_synthetic = [
        turn for turn in source_turns
        if isinstance(turn, dict) and turn.get("origin") == EXPECTED_ORIGIN
    ]
    if len(projected_synthetic) != 48:
        errors.append("projection_synthetic_origin_count_not_48")
    if [
        {key: turn.get(key) for key in ("speaker", "text", "origin")}
        for turn in projected_synthetic
    ] != [
        {key: turn.get(key) for key in ("speaker", "text", "origin")}
        for turn in extension_turns
    ]:
        errors.append("projection_synthetic_content_order_mismatch")

    projected_groups: dict[str, list[int]] = defaultdict(list)
    for turn in projected_synthetic:
        anon_ref = turn.get("turn_ref")
        match = ANON_TURN_RE.fullmatch(anon_ref) if isinstance(anon_ref, str) else None
        if not match:
            errors.append("projected_synthetic_turn_ref_not_anonymous_session_ref:" + str(anon_ref))
            continue
        projected_groups[match.group(1)].append(int(match.group(2)))
    if len(projected_groups) != 6:
        errors.append("projection_synthetic_anonymous_session_count_not_6")
    for anon_session, numbers in projected_groups.items():
        if numbers != list(range(1, 9)):
            errors.append("projection_synthetic_anonymous_session_not_8_contiguous:" + anon_session)

    metadata = {
        key: projection.get(key)
        for key in (
            "projection_policy",
            "session_ref_policy",
            "session_ref_derivation",
            "turn_count",
            "session_count",
        )
    }
    origin_leak_count = sum(
        1 for turn in source_turns
        if isinstance(turn, dict) and "origin" in turn
    )
    origin_values = Counter(
        turn.get("origin") for turn in source_turns
        if isinstance(turn, dict) and isinstance(turn.get("origin"), str)
    )
    exposure = {
        "synthetic_origin_in_turn_rows": {
            "present": any(
                turn.get("origin") == EXPECTED_ORIGIN
                for turn in source_turns if isinstance(turn, dict)
            ),
            "synthetic_row_count": len(projected_synthetic),
            "all_rows_with_origin_count": origin_leak_count,
            "origin_value_counts": dict(sorted(origin_values.items())),
        },
        "synthetic_boundary_metadata": {
            "session_ref_policy": projection.get("session_ref_policy"),
            "session_ref_derivation": projection.get("session_ref_derivation"),
            "names_synthetic_origin": EXPECTED_ORIGIN in str(projection.get("session_ref_policy", "")),
            "names_R03_scene_pattern": "R03" in str(projection.get("session_ref_policy", ""))
            or "R03" in str(projection.get("session_ref_derivation", "")),
        },
        "finding": "SYNTHETIC_ORIGIN_AND_BOUNDARY_METADATA_EXPOSED",
        "scope": "The SOURCE.json projection retains per-turn origin labels and top-level synthetic-segment mapping metadata; anonymous turn_ref values themselves are remapped.",
    }

    return {
        "root_id": root,
        "source_code": binding.get("source_code"),
        "root_code": binding.get("root_code"),
        "input_bindings": {
            "candidate": candidate_hash,
            "canonical_full_source": canonical_hash,
            "anonymous_source_projection": projection_hash,
        },
        "artifact_identity": {
            "candidate_source_is_builder_canonical": (
                candidate_hash["sha256"] == canonical_hash["sha256"]
            ),
            "comparison_scope": (
                "Historical stable_v2 CANDIDATE_0 was compared with the"
                " execution manifest's canonical FULL_SOURCE; this audit does"
                " not promote the historical candidate into the builder input."
            ),
        },
        "structure": {
            "expected_scene_sequence": expected_scenes,
            "scene_count": len(scenes),
            "scenes": scene_rows,
            "candidate_synthetic_turn_count": len(candidate_turns),
            "candidate_turn_key_shapes": dict(candidate_key_counts),
            "canonical_synthetic_turn_count": len(extension_turns),
            "canonical_synthetic_indices_contiguous_and_terminal": (
                bool(extension_indices)
                and extension_indices
                == list(range(extension_indices[0], extension_indices[0] + 48))
                and extension_indices[-1] == len(canonical_turns) - 1
            ),
            "candidate_matches_canonical_extension": candidate_content == canonical_content,
            "projection_synthetic_turn_count": len(projected_synthetic),
            "projection_synthetic_anonymous_sessions": {
                key: values for key, values in sorted(projected_groups.items())
            },
        },
        "provenance": {
            "session_ref_present_count": sum(
                1 for row in row_provenance
                if not row["session_ref"]["empty_or_missing"]
            ),
            "session_ref_missing_or_empty_count": sum(
                1 for row in row_provenance
                if row["session_ref"]["empty_or_missing"]
            ),
            "scene_ref_field_present_count": sum(
                1 for row in row_provenance
                if row["scene_ref_field"]["present"]
            ),
            "scene_ref_missing_or_empty_count": sum(
                1 for row in row_provenance
                if row["scene_ref_field"]["empty_or_missing"]
            ),
            "rows": row_provenance,
            "admission_decision": (
                "BLOCK_CANONICAL_SESSION_BINDING_FOR_UNBOUND_SYNTHETIC_ROWS"
                if any(row["provenance_status"] == "UNBOUND_SYNTHETIC_EXTENSION"
                       for row in row_provenance)
                else "REQUIRES_SESSION_REF_VALIDATION"
            ),
        },
        "lineage_findings": lineage_findings,
        "projection_exposure": exposure,
        "projection_metadata": metadata,
        "checks": {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=HERE / "PROVENANCE_PREFLIGHT_RECEIPT.json",
        help="output JSON path (must not already exist)",
    )
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("refusing to overwrite existing report: " + str(args.report))
    bindings = read_json(BINDINGS)
    source_bindings = bindings.get("sources", {})
    root_results = []
    for root in ROOTS:
        binding = source_bindings.get(root)
        if not isinstance(binding, dict):
            root_results.append({
                "root_id": root,
                "checks": {"status": "FAIL", "errors": ["private_binding_missing"]},
            })
            continue
        result = audit_root(root, binding)
        root_results.append(result)

    structural_failures = [
        {"root_id": row["root_id"], "errors": row["checks"]["errors"]}
        for row in root_results if row.get("checks", {}).get("status") != "PASS"
    ]
    missing_session_rows = sum(
        row.get("provenance", {}).get("session_ref_missing_or_empty_count", 0)
        for row in root_results
    )
    report = {
        "schema_version": "synthetic_provenance_preflight_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_mode": "READ_ONLY",
        "scope": {
            "roots": list(ROOTS),
            "candidate_source": str(CANDIDATE_BASE),
            "canonical_and_projection_bindings": str(BINDINGS),
            "canonical_data_modified": False,
            "builder_rerun": False,
            "qa_data_modified": False,
        },
        "summary": {
            "root_count": len(root_results),
            "structural_status": "PASS" if not structural_failures else "FAIL",
            "structural_failures": structural_failures,
            "candidate_and_canonical_extension_turns": "48 per root expected",
            "scene_shape": "R03:A/B:S01-S03, 8 contiguous turns per scene",
            "session_ref_missing_or_empty_count": missing_session_rows,
            "provenance_status": (
                "UNBOUND_SYNTHETIC_EXTENSION" if missing_session_rows
                else "SESSION_REF_PRESENT_REQUIRES_VALIDATION"
            ),
            "canonical_binding_gate": (
                "BLOCKED" if missing_session_rows or structural_failures
                else "REQUIRES_SEPARATE_PROVENANCE_VALIDATION"
            ),
            "source_projection_leakage": "PRESENT",
            "lineage_findings": sorted({finding for row in root_results for finding in row.get("lineage_findings", [])}),
            "artifact_identity": "HISTORICAL_CANDIDATE_IS_NOT_BUILDER_CANONICAL",
        },
        "roots": root_results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("report:", args.report.resolve())
    if structural_failures:
        return 1
    if missing_session_rows:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
