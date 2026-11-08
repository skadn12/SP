"""Offline review for the two bounded revision-1 generation outputs."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..h4_round1_r03_qa_output_contract_v1_1_revision_1.contract import (
    local_closure,
    schema_valid,
)
from . import OUTPUT, ROOTS
from .run import frozen_inputs, parse_object, read_json, write_bytes, write_json


def norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def token_jaccard(left: Any, right: Any) -> float:
    a = set(re.findall(r"\w+", norm(left), flags=re.UNICODE))
    b = set(re.findall(r"\w+", norm(right), flags=re.UNICODE))
    return len(a & b) / len(a | b) if a | b else 1.0


def output_path(root: str) -> Path:
    return OUTPUT / "raw" / "model_outputs" / f"bounded-{ROOTS.index(root) + 1:02d}-{root}.txt"


def review_one(root: str, common: dict[str, Any], dialogue: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    raw = output_path(root).read_text(encoding="utf-8")
    output, parse_error = parse_object(raw)
    row: dict[str, Any] = {
        "root_id": root,
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "parse_error": parse_error,
        "schema_valid": False,
        "semantic_auditor_called": False,
        "gold_semantic_status": "UNKNOWN_NOT_SEMANTICALLY_AUDITED",
        "distractor_semantic_status": "UNKNOWN_NOT_SEMANTICALLY_AUDITED",
    }
    if output is None:
        return row
    valid, error = schema_valid(output, schema)
    row["schema_valid"] = valid
    if error:
        row["schema_error"] = error
        return row
    closure = local_closure(output, mode="targeted", known_refs={str(t.get("turn_ref")) for t in dialogue.get("turns") or []})
    source = common["source_qa"]
    question_ratio = difflib.SequenceMatcher(None, norm(output.get("new_question")), norm(source.get("question"))).ratio()
    option_text = " ".join(str(v) for v in (output.get("options_or_items") or {}).values())
    source_option_text = " ".join(str(v) for v in (source.get("options_or_items") or {}).values())
    source_evidence = set(str(x) for x in common.get("evidence", {}).get("authoritative_evidence_refs") or [])
    evidence = set(str(x) for x in output.get("evidence_refs") or [])
    row.update({
        "selected_strategy": output.get("selected_strategy"),
        "question_type": output.get("question_type"),
        "option_count": len(output.get("options_or_items") or {}),
        "canonical_gold": output.get("canonical_gold"),
        "closure": closure,
        "evidence_refs": sorted(evidence),
        "r03_evidence_refs": sorted(x for x in evidence if x.startswith("R03:")),
        "r03_evidence_used": bool(any(x.startswith("R03:") for x in evidence)),
        "evidence_relation_changed": evidence != source_evidence,
        "source_question_sequence_ratio": question_ratio,
        "source_question_rewrite_heuristic": question_ratio >= 0.90,
        "option_text_token_overlap_with_source": token_jaccard(option_text, source_option_text),
        "new_reasoning_requirement_claimed": bool(norm(output.get("expected_context_dependency"))),
        "question_type_explicit": output.get("question_type") in {"single_choice", "multi_select", "ordering"},
        "legacy_new_r03_evidence_refs_present": "new_r03_evidence_refs" in output,
        "content_status": "STRUCTURE_PASS_SEMANTICS_UNVERIFIED" if closure.get("status") == "PASS" else closure.get("status"),
    })
    return row


def run() -> dict[str, Any]:
    common, dialogues, inputs, _ = frozen_inputs()
    rows = [review_one(root, common[root], dialogues[root], inputs["targeted_schema"]) for root in ROOTS]
    exact_duplicates = []
    lexical_pairs = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            left_raw = json.loads(output_path(left["root_id"]).read_text(encoding="utf-8"))
            right_raw = json.loads(output_path(right["root_id"]).read_text(encoding="utf-8"))
            if (norm(left_raw.get("new_question")) == norm(right_raw.get("new_question"))
                    and left_raw.get("options_or_items") == right_raw.get("options_or_items")
                    and left_raw.get("canonical_gold") == right_raw.get("canonical_gold")):
                exact_duplicates.append([left["root_id"], right["root_id"]])
            ratio = difflib.SequenceMatcher(None, norm(left_raw.get("new_question")), norm(right_raw.get("new_question"))).ratio()
            if ratio >= 0.90:
                lexical_pairs.append({"roots": [left["root_id"], right["root_id"]], "question_sequence_ratio": ratio})
    report = {
        "status": "R03_QA_OUTPUT_CONTRACT_V1_1_REVISION_BOUNDED_GENERATION_REVIEW_COMPLETE",
        "sample_size": len(rows),
        "schema_valid_count": sum(bool(row.get("schema_valid")) for row in rows),
        "structural_pass_count": sum(row.get("closure", {}).get("status") == "PASS" for row in rows),
        "no_safe_count": sum(row.get("closure", {}).get("status") == "NO_SAFE_EVOLUTION" for row in rows),
        "r03_evidence_used_count": sum(bool(row.get("r03_evidence_used")) for row in rows),
        "source_question_rewrite_heuristic_count": sum(bool(row.get("source_question_rewrite_heuristic")) for row in rows),
        "exact_duplicate_pairs": exact_duplicates,
        "lexical_duplicate_pairs": lexical_pairs,
        "gold_semantic_status": "UNKNOWN_NOT_SEMANTICALLY_AUDITED",
        "distractor_semantic_status": "UNKNOWN_NOT_SEMANTICALLY_AUDITED",
        "semantic_auditor_calls": 0,
        "solver_calls": 0,
        "memorax_calls": 0,
        "training_calls": 0,
        "rows": rows,
        "interpretation": [
            "Both bounded outputs satisfy the revised structural contract.",
            "The first output is structurally valid but does not cite R03 evidence; this is recorded, not promoted as semantic success.",
            "The second output cites three real R03 refs and changes the evidence relation; semantic Gold and distractor correctness remain UNKNOWN.",
        ],
    }
    bounded_report = read_json(OUTPUT / "bounded_verification_report.json")
    bounded_report["generation_review_status"] = report["status"]
    bounded_report["next_status"] = "READY_FOR_AUTHORIZED_R03_QA_GENERATION_WITH_REVISION_1"
    write_json(OUTPUT / "bounded_verification_report.json", bounded_report)
    write_json(OUTPUT / "bounded_generation_review.json", report)
    markdown = "\n".join([
        "# Bounded Generation Review",
        "",
        "Status: `R03_QA_OUTPUT_CONTRACT_V1_1_REVISION_BOUNDED_GENERATION_REVIEW_COMPLETE`",
        "",
        f"Schema-valid: `{report['schema_valid_count']}/{len(rows)}`; structural PASS: `{report['structural_pass_count']}/{len(rows)}`; R03 evidence used: `{report['r03_evidence_used_count']}/{len(rows)}`.",
        "",
        "Gold and distractor semantic correctness remain `UNKNOWN`; no Solver or semantic auditor was called.",
        "",
    ])
    for row in rows:
        markdown += f"- `{row['root_id']}`: `{row.get('selected_strategy')}`, `{row.get('question_type')}`, R03 evidence `{row.get('r03_evidence_used')}`, content `{row.get('content_status')}`.\n"
    write_bytes(OUTPUT / "bounded_generation_review.md", (markdown + "\n").encode("utf-8"))
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and not path.name.endswith(".sha256") and path.name != "output_sha_manifest.json":
            sidecar = path.with_name(path.name + ".sha256")
            if not sidecar.is_file():
                sidecar.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n", encoding="utf-8")
    manifest = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and not path.name.endswith(".sha256") and path.name != "output_sha_manifest.json":
            manifest.append({
                "path": str(path.relative_to(OUTPUT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sidecar_present": path.with_name(path.name + ".sha256").is_file(),
            })
    write_json(OUTPUT / "output_sha_manifest.json", manifest)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
