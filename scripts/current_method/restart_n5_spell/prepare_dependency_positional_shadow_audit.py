#!/usr/bin/env python3
"""Create a zero-API positional shadow audit of the exhausted Stage-0 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .dependency_conditioned_child_canary import (
    normalize_dependency_batch,
    validate_dependency_batch_schema,
)
from .run_dependency_conditioned_child_canary import _source, tree_manifest_sha256


DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_dependency_conditioned_child_expansion_canary_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_dependency_conditioned_child_expansion_canary_v1_shadow_positional_audit_v1"
)

# Human-readable, zero-API semantic review.  These judgments do not change the
# source run and are never consumed by Stage 1.
SEMANTIC_REVIEW: dict[str, dict[str, dict[str, Any]]] = {
    "dependency-planner-batch-4x-a00": {
        "D1": {"parent_only_sufficient": False,
            "planned_child_only_sufficient": False, "combined_sufficient": False,
            "natural_relation": True, "minimal_parent_evidence": False,
            "conceptual_joint_shape": True,
            "notes": "The donation-to-meal progression has a possible two-sided shape, but the selected events/turn IDs do not bind the stated Maria operand and the 'deepening commitment' conclusion is not uniquely entailed."},
        "D2": {"parent_only_sufficient": False,
            "planned_child_only_sufficient": True, "combined_sufficient": True,
            "natural_relation": False, "minimal_parent_evidence": False,
            "conceptual_joint_shape": False,
            "notes": "The joint value simply repeats the planned collaborative event; the child event alone supplies it, and the explanation uses the forbidden vague 'foundation' bridge."},
        "D3": {"parent_only_sufficient": False,
            "planned_child_only_sufficient": True, "combined_sufficient": True,
            "natural_relation": False, "minimal_parent_evidence": False,
            "conceptual_joint_shape": False,
            "notes": "The joint value is the planned job-training event itself, so parent history is unnecessary."},
        "D4": {"parent_only_sufficient": False,
            "planned_child_only_sufficient": True, "combined_sufficient": True,
            "natural_relation": False, "minimal_parent_evidence": False,
            "conceptual_joint_shape": False,
            "notes": "The joint value is the planned gratitude-program event itself; parent history is unnecessary, and the referenced graph events do not exist in the authoritative projection."},
    },
    "dependency-planner-batch-4x-a01": {
        "D1": {"parent_only_sufficient": False,
            "planned_child_only_sufficient": False, "combined_sufficient": False,
            "natural_relation": False, "minimal_parent_evidence": False,
            "conceptual_joint_shape": False,
            "notes": "Volunteering plus aerial yoga does not uniquely establish that those activities enabled the proposed workshop; this is an unsupported causal bridge and the turn bundle is misbound."},
        "D2": {"parent_only_sufficient": False,
            "planned_child_only_sufficient": False, "combined_sufficient": False,
            "natural_relation": True, "minimal_parent_evidence": False,
            "conceptual_joint_shape": True,
            "notes": "Campaign networking followed by a stakeholder forum is a plausible two-sided progression, but the object is batch-schema incomplete, its turns are not P aliases and mostly do not bind the selected events, and 'validated by ability' is not uniquely entailed."},
        "D3": {"parent_only_sufficient": False,
            "planned_child_only_sufficient": False, "combined_sufficient": False,
            "natural_relation": False, "minimal_parent_evidence": False,
            "conceptual_joint_shape": False,
            "notes": "The selected event IDs do not exist in the authoritative projection and the proposed causal claim is not grounded by the cited parent bundle."},
        "D4": {"parent_only_sufficient": False,
            "planned_child_only_sufficient": False, "combined_sufficient": False,
            "natural_relation": False, "minimal_parent_evidence": False,
            "conceptual_joint_shape": False,
            "notes": "The selected events concern politics/education while the operand says kickboxing; the prediction of mentorship ability is not uniquely supported."},
    },
}


def _parse_wrapper(path: Path) -> tuple[dict[str, Any], str]:
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    raw_text = wrapper["raw_response"]
    return json.loads(raw_text), raw_text


def _assign_for_shadow(raw: dict[str, Any]) -> dict[str, Any]:
    rows = raw.get("dependencies")
    if not isinstance(rows, list):
        return raw
    return {"dependencies": [{**row, "dependency_id": f"D{index}"}
                              for index, row in enumerate(rows, 1)]}


def _final_judgment(review: dict[str, Any], structural_valid: bool) -> str:
    if not structural_valid:
        return "shadow_reject_structural_or_grounding"
    if review["parent_only_sufficient"]:
        return "parent_alone_sufficient"
    if review["planned_child_only_sufficient"]:
        return "planned_child_alone_sufficient"
    if not review["combined_sufficient"]:
        return "combined_not_supported"
    if not review["natural_relation"]:
        return "unnatural_relation"
    if not review["minimal_parent_evidence"]:
        return "nonminimal_parent_evidence"
    return "dependency_contract_pass"


def run(source_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output_root_exists:{output_root}")
    before = tree_manifest_sha256(source_root)
    source = _source()
    attempts = []
    for path in sorted((source_root / "raw/dependency_planner").glob("*.json")):
        raw, raw_text = _parse_wrapper(path)
        assigned = _assign_for_shadow(raw)
        schema = validate_dependency_batch_schema(assigned)
        normalized = None
        if schema["decision"] == "dependency_batch_accept":
            normalized = normalize_dependency_batch(assigned, source)
        local_rows = {row["dependency_id"]: row for row in
                      (normalized or {}).get("dependencies", [])}
        semantic = SEMANTIC_REVIEW[path.stem]
        items = []
        for index, row in enumerate(assigned.get("dependencies", []), 1):
            did = f"D{index}"
            local = local_rows.get(did)
            review = semantic[did]
            structural = bool(local and local.get("local_validation") == "accept")
            items.append({"dependency_id": did,
                "source_model_dependency_id": raw["dependencies"][index - 1].get(
                    "dependency_id"),
                "shadow_assignment": f"array_position_{index}_to_{did}",
                "row_schema_shape_valid": not any(reason.startswith(f"{did}:")
                    for reason in schema.get("reasons", [])),
                "batch_schema_decision_after_positional_assignment": schema["decision"],
                "local_validation": local.get("local_validation") if local else
                    "not_run_batch_schema_invalid",
                "local_validation_errors": local.get("local_validation_errors", [])
                    if local else [],
                "semantic_review": review,
                "shadow_final_judgment": _final_judgment(review, structural),
                "eligible_for_stage1": False})
        attempts.append({"attempt_id": path.stem, "source_raw_record": str(path),
            "source_raw_record_sha256": sha256_file(path),
            "source_raw_response_sha256": sha256_bytes(raw_text.encode()),
            "schema_decision_after_positional_assignment": schema["decision"],
            "schema_reasons": schema.get("reasons", []), "items": items})
    after = tree_manifest_sha256(source_root)
    report = {"audit_version": "dependency_stage0_positional_shadow_audit_v1",
        "diagnostic_only": True, "zero_api": True,
        "changes_source_run_formal_conclusion": False,
        "stage1_permitted": False,
        "assignment_rule": "array_position_1_to_4_maps_to_D1_to_D4",
        "source_run_root": str(source_root),
        "source_run_tree_sha256_before": before,
        "source_run_tree_sha256_after": after,
        "source_run_unchanged": before == after,
        "attempts": attempts,
        "summary": {"attempt_count": len(attempts), "dependency_rows_reviewed":
            sum(len(row["items"]) for row in attempts),
            "batch_schema_valid_after_assignment": sum(row[
                "schema_decision_after_positional_assignment"] ==
                "dependency_batch_accept" for row in attempts),
            "grounded_dependency_contract_pass": 0,
            "conceptual_joint_shape_only": sum(item["semantic_review"][
                "conceptual_joint_shape"] for attempt in attempts for item in attempt["items"]),
            "eligible_for_stage1": 0},
        "formal_source_conclusion_remains":
            "dependency_planner_schema_invalid_exhausted"}
    output_root.mkdir(parents=True)
    atomic_json(output_root / "reports/shadow_dependency_audit.json", report)
    atomic_json(output_root / "frozen/positional_shadow_assignments.json",
                {"assignment_rule": report["assignment_rule"],
                 "attempts": [{"attempt_id": row["attempt_id"],
                    "items": [{"dependency_id": item["dependency_id"],
                        "source_model_dependency_id": item["source_model_dependency_id"]}
                        for item in row["items"]]} for row in attempts]})
    lines = ["# Stage 0 positional shadow audit", "",
        "Diagnostic only. Zero API. The source run remains exhausted and no item may enter Stage 1.", ""]
    for attempt in attempts:
        lines += [f"## {attempt['attempt_id']}", "",
            f"Batch schema after positional assignment: `{attempt['schema_decision_after_positional_assignment']}`", ""]
        for item in attempt["items"]:
            lines += [f"### {item['dependency_id']}", "",
                f"Source model ID: `{item['source_model_dependency_id']}`", "",
                f"Local validation: `{item['local_validation']}`", "",
                f"Shadow judgment: `{item['shadow_final_judgment']}`", "",
                item["semantic_review"]["notes"], ""]
    atomic_text(output_root / "review/shadow_dependency_review.md", "\n".join(lines))
    atomic_json(output_root / "reports/artifact_sha256.json", {
        str(path.relative_to(output_root)): sha256_file(path)
        for path in sorted(output_root.rglob("*")) if path.is_file()})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = run(args.source_root.resolve(), args.output_root.resolve())
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
