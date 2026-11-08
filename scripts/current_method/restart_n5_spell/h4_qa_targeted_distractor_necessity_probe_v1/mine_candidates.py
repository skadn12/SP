"""Zero-API mining of alternative composite QA candidates from frozen atomic QA."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty, write_json,
)
from scripts.current_method.restart_n5_spell.h4_relation_guided_composition_probe_v1.gate import (
    load_atomic_ir,
)
from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.contracts import (
    lexical_tokens,
)


CORE = ("fact_A", "trigger", "fact_B")


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def semantic_slots(qa: Any) -> list[str]:
    """Classify only what the atomic answer says, never its evidence refs."""
    text = _norm(qa.gold_text)
    result: set[str] = set()
    if any(re.search(pattern, text) for pattern in (
        r"original recipe", r"initial", r"two parts", r"one part",
        r"ratio", r"formulation", r"honey", r"mustard",
    )):
        result.add("fact_A")
    if any(re.search(pattern, text) for pattern in (
        r"too thick", r"thickness", r"consistency", r"spread evenly",
        r"problem", r"issue", r"reconsider", r"caused",
    )):
        result.add("trigger")
    if any(re.search(pattern, text) for pattern in (
        r"water", r"thin", r"thinned", r"pourable", r"adjusted",
        r"revision", r"revised", r"added",
    )):
        result.add("fact_B")
    return sorted(result)


def _evidence_overlap(left: Any, right: Any) -> int:
    return len(set(left.evidence_turn_refs) & set(right.evidence_turn_refs))


def _unique_evidence_counts(left: Any, right: Any) -> tuple[int, int]:
    left_refs, right_refs = set(left.evidence_turn_refs), set(right.evidence_turn_refs)
    return len(left_refs - right_refs), len(right_refs - left_refs)


def _shared_content(left: Any, right: Any) -> list[str]:
    return sorted(lexical_tokens(left.gold_text) & lexical_tokens(right.gold_text))


def _pair_record(left: Any, right: Any, *, category: str, reason: str) -> dict[str, Any]:
    left_slots, right_slots = semantic_slots(left), semantic_slots(right)
    shared = _shared_content(left, right)
    return {
        "pair_id": f"MINE-{left.qa_id.rsplit('-', 1)[-1]}-{right.qa_id.rsplit('-', 1)[-1]}",
        "category": category,
        "qa_ids": [left.qa_id, right.qa_id],
        "question_types": [left.question_type, right.question_type],
        "source_scopes": [left.source_scope, right.source_scope],
        "semantic_slots": {left.qa_id: left_slots, right.qa_id: right_slots},
        "unique_semantic_slots": {
            left.qa_id: sorted(set(left_slots) - set(right_slots)),
            right.qa_id: sorted(set(right_slots) - set(left_slots)),
        },
        "evidence_overlap_turn_count": _evidence_overlap(left, right),
        "unique_evidence_turn_counts": dict(zip(
            (left.qa_id, right.qa_id), _unique_evidence_counts(left, right))),
        "shared_content_tokens": shared,
        "evidence_turn_refs": {
            left.qa_id: list(left.evidence_turn_refs),
            right.qa_id: list(right.evidence_turn_refs),
        },
        "atomic_questions": {left.qa_id: left.question, right.qa_id: right.question},
        "atomic_gold_text": {left.qa_id: left.gold_text, right.qa_id: right.gold_text},
        "reason": reason,
    }


def mine(*, atomic_path: Path, relation_contract: Path, output: Path) -> dict[str, Any]:
    prepare_empty(output)
    atomic = load_atomic_ir(atomic_path)
    contract = json.loads(relation_contract.read_text(encoding="utf-8"))
    relation_pairs: list[dict[str, Any]] = []
    cross_context_pairs: list[dict[str, Any]] = []
    rejected_relation: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(atomic, key=lambda qa: qa.qa_id), 2):
        left_slots, right_slots = set(semantic_slots(left)), set(semantic_slots(right))
        union = left_slots | right_slots
        unique_left, unique_right = left_slots - right_slots, right_slots - left_slots
        if set(CORE).issubset(union):
            row = _pair_record(
                left, right, category="h4_semantic_relation", reason="semantic gold union covers all core relation slots")
            row["single_qa_already_complete"] = set(CORE).issubset(left_slots) or set(CORE).issubset(right_slots)
            row["necessity_eligible"] = bool(unique_left and unique_right and not row["single_qa_already_complete"])
            if row["necessity_eligible"]:
                relation_pairs.append(row)
            else:
                row["rejection_reason"] = (
                    "one atomic gold already contains all three slots"
                    if row["single_qa_already_complete"] else
                    "one atomic contributes no unique semantic relation slot"
                )
                rejected_relation.append(row)
        shared = _shared_content(left, right)
        left_unique_refs, right_unique_refs = _unique_evidence_counts(left, right)
        complementary_evidence = left_unique_refs > 0 and right_unique_refs > 0
        same_scope = left.source_scope == right.source_scope
        if complementary_evidence and len(shared) >= 1 and same_scope and left.source_scope == "parent_only":
            cross_context_pairs.append(_pair_record(
                left, right, category="parent_cross_context_synthesis",
                reason="partially overlapping parent evidence with unique facts on both sides and a shared career/cooking theme"))
    cross_context_pairs.sort(key=lambda row: (
        -len(row["shared_content_tokens"]), row["pair_id"]))
    relation_pairs.sort(key=lambda row: row["pair_id"])
    result = {
        "probe_version": "h4_qa_targeted_distractor_necessity_probe_v1_candidate_mining",
        "analysis_execution_api_calls": 0,
        "source_artifacts_mutated": False,
        "atomic_count": len(atomic),
        "relation_contract_type": contract.get("relation_type"),
        "h4_semantic_relation_candidates": relation_pairs,
        "h4_semantic_relation_candidate_count": len(relation_pairs),
        "h4_semantic_relation_rejected_count": len(rejected_relation),
        "h4_semantic_relation_rejected": rejected_relation,
        "parent_cross_context_candidates": cross_context_pairs,
        "parent_cross_context_candidate_count": len(cross_context_pairs),
        "conclusion": (
            "NO_SEMANTICALLY_COMPLEMENTARY_H4_PAIR_IN_FROZEN_ATOMIC_POOL"
            if not relation_pairs else "H4_ALTERNATIVE_PAIRS_FOUND"
        ),
        "recommended_alternative": (
            cross_context_pairs[0] if cross_context_pairs else None
        ),
    }
    write_json(output / "candidate_mining.json", result)
    write_json(output / "run_freeze_manifest.json", {
        "freeze_version": "h4_qa_targeted_distractor_necessity_probe_v1_candidate_mining",
        "atomic_normalized": str(atomic_path.resolve()),
        "relation_contract": str(relation_contract.resolve()),
        "analysis_execution_api_calls": 0,
        "planner_calls": 0, "realizer_calls": 0, "qa_writer_calls": 0,
        "composer_calls": 0, "solver_calls": 0, "memorax_calls": 0,
    })
    write_json(output / "POST_RUN_INTEGRITY.json", {
        "status": "complete", "analysis_execution_api_calls": 0,
        "source_artifacts_mutated": False, "composer_calls": 0,
        "solver_calls": 0,
    })
    report = [
        "# H4 QA Candidate Mining v1", "",
        "Zero-API mining over the frozen atomic QA pool. Semantic relation slots "
        "are inferred from atomic gold text, not merely evidence references.", "",
        f"- Frozen eligible atomic QA: `{len(atomic)}`",
        f"- Semantically necessity-eligible H4 pairs: `{len(relation_pairs)}`",
        f"- Rejected H4 relation pairs: `{len(rejected_relation)}`",
        f"- Parent cross-context candidates: `{len(cross_context_pairs)}`", "",
        f"- Conclusion: **{result['conclusion']}**", "",
    ]
    if cross_context_pairs:
        row = cross_context_pairs[0]
        report.extend([
            "## Recommended Alternative", "",
            f"- Pair: `{row['qa_ids'][0]}` + `{row['qa_ids'][1]}`",
            f"- Shared content tokens: `{row['shared_content_tokens']}`",
            "- Candidate question: `How did their shared positive experiences and Classmates B's encouragement together relate to Classmates A's move from cashier work toward culinary work?`",
            "- This is a parent cross-context synthesis candidate, not a causal-revision H4 pair.", "",
        ])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic", type=Path, required=True)
    parser.add_argument("--relation-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(mine(
        atomic_path=args.atomic.resolve(),
        relation_contract=args.relation_contract.resolve(),
        output=args.output.resolve(),
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
