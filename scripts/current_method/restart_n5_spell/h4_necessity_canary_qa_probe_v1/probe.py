"""Isolated relation-aware QA Writer and unchanged QA replay for one canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json

from scripts.current_method.restart_n5_spell.h4_relation_aware_qa_writer_isolation_probe_v1.probe import (
    _call_writer,
    _run_replayed_qa,
)
from scripts.current_method.restart_n5_spell.h4_relation_aware_qa_writer_isolation_probe_v1.prompt import (
    build_relation_aware_writer_prompt,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(*, source_run: Path, relation_contract: Path, relation_audit: Path,
        config: Path, tokenizer: Path, solver: Path, output: Path) -> dict:
    prepare_empty(output)
    source = source_run / "qa_source"
    expanded = _load(source / "expanded_dialogue.json")
    contract = _load(relation_contract)
    audit = _load(relation_audit)
    prompt = build_relation_aware_writer_prompt(
        expanded=expanded, relation_contract=contract, relation_audit=audit)
    write_json(output / "relation_aware_writer_prompt.json", prompt)
    writer = _call_writer(prompt=prompt["prompt"], config_path=config, run_root=output)
    writer_path = output / "writer_response_envelope.json"
    qa_run = output / "qa_run"
    _run_replayed_qa(
        source_run=source, writer_output=writer_path, qa_run=qa_run,
        config=config, tokenizer=tokenizer, solver=solver)
    final_batch = _load(qa_run / "qa" / "final_mcq_batch.json")
    filters = [_load(path) for path in sorted((qa_run / "filter").glob("*.json"))]
    groups = audit.get("relation_source_turns") or {}
    rows = []
    for item, evaluation in zip(final_batch.get("items") or [], filters):
        refs = {str(ref) for ref in (item.get("evidence_turn_refs") or item.get("evidence_turn_ids") or [])}
        hits = sorted(key for key, values in groups.items() if refs & {str(value) for value in values})
        core = {"fact_A", "trigger", "fact_B"}
        rows.append({
            "qa_id": item.get("qa_id"),
            "question": item.get("question"),
            "relation_groups_hit": hits,
            "relation_targeted": len(set(hits) & core) >= 2,
            "relation_complete_chain": core.issubset(hits),
            "grounding_decision": (evaluation.get("grounding") or {}).get("decision"),
            "full_context_correct": evaluation.get("full_context_correct"),
            "full_context_total": evaluation.get("full_context_total"),
            "decision": evaluation.get("decision"),
        })
    targeted = [row for row in rows if row["relation_targeted"]]
    complete = [row for row in rows if row["relation_complete_chain"]]
    complete_stable = [row for row in complete if row["grounding_decision"] == "pass_grounding" and row["full_context_correct"] == 5 and row["full_context_total"] == 5]
    metrics = {
        "probe_version": "h4_necessity_canary_qa_probe_v1",
        "source_run": str(source_run.resolve()),
        "qa_run": str(qa_run.resolve()),
        "writer_api_calls": 1,
        "solver_api_calls": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "composer_calls": 0,
        "memorax_calls": 0,
        "atomic_generated": len(rows),
        "relation_targeted_questions": len(targeted),
        "relation_complete_chain_questions": len(complete),
        "relation_complete_chain_stable": len(complete_stable),
        "relation_utilization_rate": (len(complete_stable) / len(rows)) if rows else 0.0,
        "rows": rows,
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_freeze_manifest.json", {
        "freeze_version": "h4_necessity_canary_qa_probe_v1",
        "source_run": str(source_run.resolve()),
        "relation_contract": str(relation_contract.resolve()),
        "relation_audit": str(relation_audit.resolve()),
        "writer_api_calls": 1,
        "composer_calls": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "memorax_calls": 0,
    })
    write_json(output / "request_ledger.json", {
        "probe_version": "h4_necessity_canary_qa_probe_v1",
        "writer_requests": 1,
        "composer_requests": 0,
        "solver_requests": 0,
    })
    (output / "report.md").write_text("\n".join([
        "# H4 Necessity Canary QA Probe v1", "",
        "The relation-aware Writer wrapper and unchanged QA replay ran on the source-ready canary. Composer was not called.", "",
        f"- Atomic generated: `{len(rows)}`",
        f"- Relation-targeted: `{len(targeted)}`",
        f"- Complete fact_A+trigger+fact_B: `{len(complete)}`",
        f"- Complete-chain stable: `{len(complete_stable)}`", "",
        "Only complete-chain stable candidates are eligible for the next relation-guided composition stage.", "",
    ]), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--relation-contract", type=Path, required=True)
    parser.add_argument("--relation-audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        source_run=args.source_run.resolve(),
        relation_contract=args.relation_contract.resolve(),
        relation_audit=args.relation_audit.resolve(),
        config=args.config.resolve(), tokenizer=args.tokenizer.resolve(),
        solver=args.solver.resolve(), output=args.output.resolve(),
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
