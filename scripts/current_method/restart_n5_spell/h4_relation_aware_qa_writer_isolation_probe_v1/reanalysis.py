"""Zero-API reanalysis for a completed relation-aware Writer A/B probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json

from .attribution import compare, summarize_condition
from .probe import _run_composition


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(*, baseline_qa: Path, relation_aware_qa: Path, source_run: Path,
        relation_audit: Path, output: Path) -> dict:
    prepare_empty(output)
    relation = _load(relation_audit)
    baseline_composition = output / "baseline_composition_offline"
    relation_composition = output / "relation_aware_composition_offline"
    _run_composition(qa_run=baseline_qa, source_run=source_run,
                     output=baseline_composition)
    _run_composition(qa_run=relation_aware_qa, source_run=source_run,
                     output=relation_composition)
    baseline = summarize_condition(
        qa_run=baseline_qa, composition_run=baseline_composition,
        relation_audit=relation, condition="A_baseline")
    relation_aware = summarize_condition(
        qa_run=relation_aware_qa, composition_run=relation_composition,
        relation_audit=relation, condition="B_relation_aware")
    comparison = compare(baseline=baseline, relation_aware=relation_aware)
    result = {
        "probe_version": "h4_relation_aware_qa_writer_isolation_reanalysis_v1",
        "analysis_execution_api_calls": 0,
        "baseline": baseline,
        "relation_aware": relation_aware,
        "ab_comparison": comparison,
    }
    write_json(output / "metrics.json", result)
    (output / "report.md").write_text(
        "# H4 Relation-aware QA Writer Isolation Probe v1: zero-API reanalysis\n\n"
        f"Conclusion: **{comparison['conclusion']}**\n\n"
        "The original A/B QA and Solver artifacts were frozen. This reanalysis "
        "only repaired the evaluation-file adapter used by the offline composition "
        "probe; it did not call Writer, Solver, Composer, Planner, Realizer, or Memorax.\n\n"
        "## Comparison\n\n"
        "| metric | A: production Writer | B: relation-aware wrapper |\n"
        "|---|---:|---:|\n"
        f"| atomic generated | {baseline['atomic_generated']} | {relation_aware['atomic_generated']} |\n"
        f"| grounding pass | {baseline['grounding_pass']} | {relation_aware['grounding_pass']} |\n"
        f"| full-context K=5 stable | {baseline['k5_stable']} | {relation_aware['k5_stable']} |\n"
        f"| relation-targeted questions | {baseline['relation_targeted_questions']} | {relation_aware['relation_targeted_questions']} |\n"
        f"| complete fact_A+trigger+fact_B questions | {baseline['relation_complete_chain_questions']} | {relation_aware['relation_complete_chain_questions']} |\n"
        f"| relation-targeted stable | {baseline['relation_targeted_grounding_k5']} | {relation_aware['relation_targeted_grounding_k5']} |\n"
        f"| complete-chain stable | {baseline['relation_complete_chain_grounding_k5']} | {relation_aware['relation_complete_chain_grounding_k5']} |\n"
        f"| relation-targeted easy rejected | {baseline['relation_targeted_easy_rejected']} | {relation_aware['relation_targeted_easy_rejected']} |\n"
        f"| difficulty-qualified relation utilization | {baseline['difficulty_qualified_relation_utilization_rate']:.3f} | {relation_aware['difficulty_qualified_relation_utilization_rate']:.3f} |\n"
        f"| pair candidates | {baseline['pair_candidates']} | {relation_aware['pair_candidates']} |\n"
        f"| Composer-ready | {baseline['composer_ready']} | {relation_aware['composer_ready']} |\n"
        f"| admitted | {baseline['admitted']} | {relation_aware['admitted']} |\n\n"
        "## Interpretation\n\n"
        "The relation-aware wrapper increased relation-targeted stable questions "
        "and pair candidates, but every stable relation-targeted question was still "
        "easy-rejected. The conservative pair compiler classified all candidates as "
        "`uncertain`, so no Composer request was emitted. This separates relation "
        "coverage from difficulty: relation evidence was present and used, but no "
        "difficulty-qualified relation QA was produced.\n\n"
        "Necessity ablation remains not run because no Composer response exists.\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-qa", type=Path, required=True)
    parser.add_argument("--relation-aware-qa", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--relation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        baseline_qa=args.baseline_qa.resolve(),
        relation_aware_qa=args.relation_aware_qa.resolve(),
        source_run=args.source_run.resolve(),
        relation_audit=args.relation_audit.resolve(),
        output=args.output.resolve(),
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
