"""Zero-API finalization of a completed relation-guided composition probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import write_json


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(run_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    metrics = _load(run_root / "metrics.json")
    solver = _load(run_root / "solver_evaluation.json")
    static = _load(run_root / "composer_static_audit.json")
    grounding = solver.get("grounding") or {}
    full = solver.get("full_context") or {}
    ablations = solver.get("ablations") or {}
    relation_binding = _load(run_root / "composer_response_envelope.json").get("relation_binding") or {}
    result = {
        "probe_version": "h4_relation_guided_composition_probe_v1_finalized",
        "analysis_execution_api_calls": 0,
        "current_pair_candidates": metrics.get("current_pair_candidates"),
        "relation_guided_candidates": metrics.get("relation_guided_candidates"),
        "current_composer_ready": metrics.get("current_composer_ready"),
        "relation_guided_composer_ready": metrics.get("relation_guided_composer_ready"),
        "composite_generated": int(static.get("valid") is True),
        "difficulty_qualified_composite": 0,
        "relation_used": bool(static.get("valid")),
        "fact_A_used": relation_binding.get("used_fact_A") is True,
        "trigger_used": relation_binding.get("used_trigger") is True,
        "fact_B_used": relation_binding.get("used_fact_B") is True,
        "no_context_grounding_decision": grounding.get("decision"),
        "no_context_accuracy": grounding.get("semantic_correct_rate"),
        "full_context_accuracy": full.get("accuracy"),
        "full_context_k5_decision": (full.get("k5") or {}).get("decision"),
        "remove_A_accuracy": (ablations.get("remove_fact_A") or {}).get("accuracy"),
        "remove_trigger_accuracy": (ablations.get("remove_trigger") or {}).get("accuracy"),
        "remove_B_accuracy": (ablations.get("remove_fact_B") or {}).get("accuracy"),
        "necessity_pass": False,
        "admitted": False,
        "conclusion": "COMPOSITION_GATE_CANDIDATE_BOTTLENECK_BUT_DIFFICULTY_NOT_VALIDATED",
        "reason": "Composer accepted relation binding, but no-context was all-correct and full/removal K5 views were all 5/5; the composite is easy and evidence necessity is unproven.",
    }
    write_json(output / "metrics.json", result)
    write_json(output / "POST_RUN_INTEGRITY.json", {
        "status": "complete",
        "analysis_execution_api_calls": 0,
        "source_run": str(run_root),
        "composer_static_audit": "pass",
        "no_context_grounding": result["no_context_grounding_decision"],
        "necessity_pass": False,
        "difficulty_qualified_composite": 0,
    })
    report = [
        "# H4 Relation-guided Composition Gate Isolation Probe v1: authoritative zero-API finalization", "",
        "Conclusion: **COMPOSITION_GATE_CANDIDATE_BOTTLENECK_BUT_DIFFICULTY_NOT_VALIDATED**", "",
        "This finalization reads the completed Composer and local Solver artifacts only. No API was called.", "",
        "| metric | Condition A: current gate | Condition B: relation-guided gate |", "|---|---:|---:|",
        f"| pair candidates | {result['current_pair_candidates']} | {result['relation_guided_candidates']} |",
        f"| Composer-ready | {result['current_composer_ready']} | {result['relation_guided_composer_ready']} |",
        f"| valid composite generated | 0 | {result['composite_generated']} |",
        f"| difficulty-qualified composite | 0 | {result['difficulty_qualified_composite']} |",
        f"| relation binding fact_A/trigger/fact_B | n/a | {result['fact_A_used']}/{result['trigger_used']}/{result['fact_B_used']} |",
        "", "## Difficulty", "",
        f"- No-context grounding: `{result['no_context_grounding_decision']}`; accuracy `{result['no_context_accuracy']}`",
        f"- Full context: `{result['full_context_accuracy']}`; K=5 decision `{result['full_context_k5_decision']}`",
        f"- Remove fact_A: `{result['remove_A_accuracy']}`",
        f"- Remove trigger: `{result['remove_trigger_accuracy']}`",
        f"- Remove fact_B: `{result['remove_B_accuracy']}`",
        "- Necessity pass: `False`",
        "- Admitted: `False`", "",
        "Interpretation: the current composition classifier is a candidate-generation bottleneck, "
        "because it marks all seven current candidates `uncertain` while the guided selector reaches Composer. "
        "But the generated composite is answerable without context and after every evidence removal, so this "
        "does not validate H4 difficulty or justify a mainline gate change.", "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_root.resolve(), args.output.resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
