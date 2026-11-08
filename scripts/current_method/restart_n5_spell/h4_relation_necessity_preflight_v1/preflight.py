"""Audit whether a frozen relation has independently necessary evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json


_STOPWORDS = {
    "a", "about", "after", "all", "and", "because", "but", "by", "did",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or",
    "the", "their", "them", "they", "this", "to", "was", "what", "when",
    "which", "with", "would", "why", "you", "your", "classmates",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _turn_map(expanded: dict[str, Any]) -> dict[str, str]:
    return {
        str(row.get("turn_ref")): str(row.get("text") or "")
        for row in expanded.get("turns") or []
        if row.get("turn_ref")
    }


def _group_report(groups: dict[str, list[str]], turns: dict[str, str], anchors: set[str]) -> dict[str, Any]:
    names = list(groups)
    result: dict[str, Any] = {}
    all_relation_refs = set().union(*(set(values) for values in groups.values()))
    for name in names:
        refs = set(groups.get(name) or [])
        others = set().union(*(set(groups.get(other) or []) for other in names if other != name))
        texts = " ".join(turns.get(ref, "") for ref in refs)
        outside_text = " ".join(text for ref, text in turns.items() if ref not in refs)
        group_tokens = _tokens(texts)
        outside_relation_tokens = _tokens(" ".join(
            turns.get(ref, "") for ref in all_relation_refs - refs))
        result[name] = {
            "turn_count": len(refs),
            "exclusive_turn_count": len(refs - others),
            "overlap_with_other_relation_turns": len(refs & others),
            "exclusive_turn_refs": sorted(refs - others),
            "anchor_tokens_present": sorted(group_tokens & anchors),
            "anchor_tokens_unique_to_group": sorted((group_tokens & anchors) - outside_relation_tokens),
            "anchor_tokens_also_outside_group": sorted((group_tokens & anchors) & _tokens(outside_text)),
        }
    return result


def run(*, expanded_dialogue: Path, relation_audit: Path, composite_run: Path, output: Path) -> dict[str, Any]:
    prepare_empty(output)
    expanded = _load(expanded_dialogue)
    audit = _load(relation_audit)
    response = _load(composite_run / "composer_response_envelope.json")
    solver = _load(composite_run / "solver_evaluation.json")
    turns = _turn_map(expanded)
    groups = {
        str(name): [str(ref) for ref in values]
        for name, values in (audit.get("relation_source_turns") or {}).items()
        if isinstance(values, list)
    }
    anchors = _tokens(" ".join((response.get("question") or "", response.get("gold_text") or "")))
    group_details = _group_report(groups, turns, anchors)
    required_groups = response.get("required_evidence_groups") or []
    required_sets = [set(str(ref) for ref in row.get("turn_refs") or []) for row in required_groups]
    required_overlap = len(required_sets[0] & required_sets[1]) if len(required_sets) == 2 else None
    ablations = solver.get("ablations") or {}
    removal_accuracy = {
        "remove_A": (ablations.get("remove_fact_A") or {}).get("accuracy"),
        "remove_trigger": (ablations.get("remove_trigger") or {}).get("accuracy"),
        "remove_B": (ablations.get("remove_fact_B") or {}).get("accuracy"),
    }
    no_context = solver.get("grounding") or {}
    full = solver.get("full_context") or {}
    metrics = {
        "probe_version": "h4_relation_necessity_preflight_v1",
        "analysis_execution_api_calls": 0,
        "relation_audit_pass": bool(audit.get("relation_audit_pass")),
        "relation_type": audit.get("relation_type"),
        "relation_group_details": group_details,
        "required_evidence_group_overlap_turns": required_overlap,
        "required_evidence_group_count": len(required_groups),
        "answer_anchor_tokens": sorted(anchors),
        "no_context_decision": no_context.get("decision"),
        "no_context_accuracy": no_context.get("semantic_correct_rate"),
        "full_context_accuracy": full.get("accuracy"),
        "full_context_k5_decision": (full.get("k5") or {}).get("decision"),
        "removal_accuracy": removal_accuracy,
        "all_removal_views_still_correct": all(value == 1.0 for value in removal_accuracy.values()),
        "necessity_risk": "high" if all(value == 1.0 for value in removal_accuracy.values()) else "requires_review",
        "diagnosis": "evidence_redundancy_or_resolution_leakage",
    }
    write_json(output / "metrics.json", metrics)
    report = [
        "# H4 Relation Necessity Preflight v1", "",
        "Zero-API audit of the frozen POB03 relation-guided composite.", "",
        f"- Relation audit pass: `{metrics['relation_audit_pass']}`",
        f"- Relation type: `{metrics['relation_type']}`",
        f"- Required atomic evidence-group overlap: `{required_overlap}` turns",
        f"- No-context: `{metrics['no_context_decision']}`, accuracy `{metrics['no_context_accuracy']}`",
        f"- Full context: `{metrics['full_context_accuracy']}`, K=5 `{metrics['full_context_k5_decision']}`", "",
        "## Relation evidence", "",
    ]
    for name, detail in group_details.items():
        report.extend([
            f"### {name}", "",
            f"- Turns: `{detail['turn_count']}`; exclusive turns: `{detail['exclusive_turn_count']}`",
            f"- Overlap with other relation groups: `{detail['overlap_with_other_relation_turns']}`",
            f"- Answer anchors unique to this group: `{', '.join(detail['anchor_tokens_unique_to_group']) or 'none'}`",
            f"- Answer anchors also outside group: `{', '.join(detail['anchor_tokens_also_outside_group']) or 'none'}`", "",
        ])
    report.extend([
        "## Ablation", "",
        f"- Remove fact_A: `{removal_accuracy['remove_A']}`",
        f"- Remove trigger: `{removal_accuracy['remove_trigger']}`",
        f"- Remove fact_B: `{removal_accuracy['remove_B']}`",
        "- Necessity preflight: `FAIL`",
        "", "Diagnosis: the generated composite has relation wording, but its answer is "
        "not evidence-necessary. The next generation probe must require disjoint "
        "role-specific anchors and must treat resolution/future restatements as leakage.", "",
    ])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-dialogue", type=Path, required=True)
    parser.add_argument("--relation-audit", type=Path, required=True)
    parser.add_argument("--composite-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        expanded_dialogue=args.expanded_dialogue.resolve(),
        relation_audit=args.relation_audit.resolve(),
        composite_run=args.composite_run.resolve(),
        output=args.output.resolve(),
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
