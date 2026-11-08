"""Zero-API contract preflight for the isolated relation-aware Composer probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.h4_relation_guided_composition_probe_v1.gate import load_atomic_ir

from .composite_audit import audit_candidate_composite


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_fixture(candidate: dict[str, Any]) -> dict[str, Any]:
    qids = list(candidate["source_qids"])
    unique = candidate["unique_contribution_closure"]
    groups = [
        {"source_qa_id": qid, "turn_refs": list(unique[qid])}
        for qid in qids
    ]
    options = {
        "A": "The collaborators identified the thick glaze, added water, and obtained a pourable consistency.",
        "B": "The collaborators identified the thin glaze, added oil, and obtained a pourable consistency.",
        "C": "One collaborator identified the thick glaze, added water, and obtained a pourable consistency.",
        "D": "The collaborators identified the thick glaze, added water, and obtained a thick consistency.",
    }
    return {
        "candidate_id": candidate["candidate_id"],
        "required_qa_ids": qids,
        "target": candidate["target"],
        "question": "What joint revision changed the glaze from its initial problem to its usable final state?",
        "answer_dimension": "revision outcome",
        "gold_text": options["A"],
        "options": options,
        "correct_option": "A",
        "source_contributions": candidate["source_contributions"],
        "required_evidence_groups": groups,
        "required_evidence_turn_refs": sorted({ref for refs in unique.values() for ref in refs}),
        "distractor_mutations": {"B": qids[1], "C": qids[0], "D": qids[1]},
        "is_mechanical_conjunction": False,
    }


def run_preflight(*, atomic_path: Path, old_run: Path) -> dict[str, Any]:
    atomic = load_atomic_ir(atomic_path)
    by_id = {qa.qa_id: qa for qa in atomic}
    selector = _load(old_run / "selector_audit.json")
    candidates = [row for row in selector["candidates"] if row["candidate_id"] == "C02"]
    if len(candidates) != 1:
        raise ValueError("expected exactly one frozen C02 candidate")
    candidate = candidates[0]
    fixture = _canonical_fixture(candidate)
    result = audit_candidate_composite(fixture, candidate=candidate, atomic_by_id=by_id)
    old_c02 = _load(old_run / "candidates" / "C02" / "composer_response.json")
    old_result = audit_candidate_composite(old_c02, candidate=candidate, atomic_by_id=by_id)
    if not result["valid"]:
        raise AssertionError(f"canonical fixture unexpectedly failed: {result['errors']}")
    if old_result["valid"]:
        raise AssertionError("frozen malformed C02 unexpectedly passed after contract fix")
    return {
        "status": "passed",
        "api_calls": 0,
        "canonical_fixture_valid": result["valid"],
        "canonical_fixture_errors": result["errors"],
        "frozen_c02_valid_after_fix": old_result["valid"],
        "frozen_c02_errors_after_fix": old_result["errors"],
        "contract": {
            "required_qa_ids": "candidate source_qids, exact order",
            "required_evidence_groups_source_key": "source_qa_id",
            "required_evidence_groups_refs": "unique_contribution_closure",
            "required_evidence_turn_refs": "union of unique groups",
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(atomic_path=args.atomic, old_run=args.old_run)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
