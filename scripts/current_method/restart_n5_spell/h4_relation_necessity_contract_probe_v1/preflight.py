"""Audit whether a frozen child can support evidence-necessary QA."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json


ROLE_REFS = {
    "fact_A": [
        "R01:S01:T001", "R01:S01:T003", "R01:S01:T005", "R01:S01:T007", "R01:S01:T009",
    ],
    "trigger": ["R01:S02:T001"],
    "fact_B": ["R01:S02:T002", "R01:S02:T003", "R01:S02:T004"],
}
ANSWER_COMPONENTS = {
    "fact_A": ["two parts honey", "one part mustard"],
    "trigger": ["too thick", "spread evenly"],
    "fact_B": ["water", "thin"],
}
_STOPWORDS = {
    "a", "about", "after", "all", "and", "because", "but", "by", "for", "from",
    "how", "in", "into", "is", "it", "of", "on", "or", "the", "their", "this",
    "to", "was", "what", "when", "which", "with", "why", "would", "you",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _turn_map(expanded: dict[str, Any]) -> dict[str, str]:
    return {
        str(row["turn_ref"]): str(row.get("text") or "")
        for row in expanded.get("turns") or []
        if row.get("turn_ref")
    }


def _contains(text: str, phrase: str) -> bool:
    return phrase.casefold() in text.casefold()


def run(*, expanded_dialogue: Path, relation_audit: Path, output: Path) -> dict[str, Any]:
    prepare_empty(output)
    expanded = _load(expanded_dialogue)
    audit = _load(relation_audit)
    turns = _turn_map(expanded)
    child_refs = [ref for ref in turns if ref.startswith("R01:")]

    role_text = {
        role: " ".join(turns.get(ref, "") for ref in refs)
        for role, refs in ROLE_REFS.items()
    }
    role_overlap = {
        f"{left}+{right}": sorted(set(ROLE_REFS[left]) & set(ROLE_REFS[right]))
        for left, right in (("fact_A", "trigger"), ("fact_A", "fact_B"), ("trigger", "fact_B"))
    }
    role_support = {
        role: {
            phrase: _contains(role_text[role], phrase)
            for phrase in phrases
        }
        for role, phrases in ANSWER_COMPONENTS.items()
    }
    future_refs = [ref for ref in child_refs if ref.startswith("R01:S03:")]
    future_text = " ".join(turns.get(ref, "") for ref in future_refs)
    future_leakage = {
        phrase: _contains(future_text, phrase)
        for phrases in ANSWER_COMPONENTS.values()
        for phrase in phrases
    }
    required_refs = set().union(*(set(refs) for refs in ROLE_REFS.values()))
    full_context_refs = set(child_refs)
    full_context_has_resolution = any(ref.startswith("R01:S03:") for ref in full_context_refs)
    metrics = {
        "probe_version": "h4_relation_necessity_contract_probe_v1",
        "analysis_execution_api_calls": 0,
        "relation_audit_pass": bool(audit.get("relation_audit_pass")),
        "relation_type": audit.get("relation_type"),
        "role_refs": ROLE_REFS,
        "role_ref_overlap": role_overlap,
        "role_ref_disjoint": all(not refs for refs in role_overlap.values()),
        "role_answer_component_support": role_support,
        "all_components_supported": all(all(values.values()) for values in role_support.values()),
        "required_ref_count": len(required_refs),
        "child_turn_count": len(child_refs),
        "full_context_has_resolution": full_context_has_resolution,
        "future_resolution_answer_leakage": future_leakage,
        "future_leakage_present": any(future_leakage.values()),
        "necessity_contract_ready_for_full_context": bool(
            all(not refs for refs in role_overlap.values())
            and all(all(values.values()) for values in role_support.values())
            and not any(future_leakage.values())
        ),
        "diagnosis": (
            "future_resolution_leakage_blocks_full_context_necessity"
            if any(future_leakage.values())
            else "minimal_disjoint_contract_is_source_feasible"
        ),
    }
    write_json(output / "metrics.json", metrics)
    report = [
        "# H4 Necessity-First Relation Contract Probe v1", "",
        "Zero-API audit of a minimal relation contract over the frozen POB03 child.", "",
        f"- Relation audit pass: `{metrics['relation_audit_pass']}`",
        f"- Minimal role refs disjoint: `{metrics['role_ref_disjoint']}`",
        f"- All answer components supported by their role: `{metrics['all_components_supported']}`",
        f"- Full QA context includes resolution scene: `{full_context_has_resolution}`", "",
        "## Role evidence", "",
    ]
    for role, refs in ROLE_REFS.items():
        report.extend([
            f"### {role}", "",
            f"- Refs: `{', '.join(refs)}`",
            f"- Answer components: `{', '.join(ANSWER_COMPONENTS[role])}`",
            f"- Support: `{role_support[role]}`", "",
        ])
    report.extend(["## Future leakage", ""])
    for phrase, present in future_leakage.items():
        report.append(f"- `{phrase}` in S03 resolution: `{present}`")
    report.extend([
        "", "## Verdict", "",
        f"- Necessity contract ready for unchanged full context: `{metrics['necessity_contract_ready_for_full_context']}`",
        f"- Diagnosis: `{metrics['diagnosis']}`", "",
        "The minimal A/trigger/B windows are disjoint and locally supported, but the frozen S03 resolution repeats answer facts. A Composer/Solver run on the unchanged full context would therefore be unable to prove necessity. The next valid H4 probe needs a fresh canary whose resolution does not restate the target answer, or an explicitly pre-resolution QA-context condition labeled diagnostic-only.", "",
    ])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-dialogue", type=Path, required=True)
    parser.add_argument("--relation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        expanded_dialogue=args.expanded_dialogue.resolve(),
        relation_audit=args.relation_audit.resolve(),
        output=args.output.resolve(),
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
