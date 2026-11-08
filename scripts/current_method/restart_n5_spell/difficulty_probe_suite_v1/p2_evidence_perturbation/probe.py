from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..common.frozen_loader import DEFAULT_RUN, load_frozen_inputs
from ..common.io_utils import prepare_empty, read_json, write_json, write_jsonl
from ..common.live_adapter import LiveJSONClient, extract_json
from ..common.metrics import solver_summary
from ..common.solver_adapter import maybe_solver
from .validator import (attest_original_preservation, materialize_view,
                        validate_generation, validate_semantic_audit)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "p2_evidence_perturbation_v1"
TYPES = ("background", "competing_state", "entity_event_interference")


def _context_for(row: Mapping[str, Any], turns: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs = set(str(ref) for ref in row.get("evidence_turn_refs") or [])
    positions = [index for index, turn in enumerate(turns) if str(turn.get("turn_ref")) in refs]
    if not positions:
        return turns[:8]
    lo, hi = max(0, min(positions) - 2), min(len(turns), max(positions) + 3)
    return turns[lo:hi]


def _prompt(row: Mapping[str, Any], turns: list[Mapping[str, Any]], perturbation_type: str) -> str:
    type_instruction = {
        "background": "Add only environment/background or a related non-answer event.",
        "competing_state": "Add a historically true alternative, rejected option, temporary preference, or intermediate state before the unchanged final outcome.",
        "entity_event_interference": "Add a similar entity/event involving another existing speaker, while preserving the target entity's action and answer.",
    }[perturbation_type]
    enum_rule = {
        "background": "type must be background or related_event; relation_to_gold must be non_answer_background; temporal_relation must be concurrent_background; state_status must be background_only",
        "competing_state": "type must be alternative_state or intermediate_state; relation_to_gold must be historical_non_final_alternative; temporal_relation must be before_final_state; state_status must be historically_true_not_final",
        "entity_event_interference": "type must be entity_interference or related_event; relation_to_gold must be other_entity_only; temporal_relation must be concurrent_background or before_final_state; state_status must be other_entity_only",
    }[perturbation_type]
    context = [{"turn_ref": turn.get("turn_ref"), "speaker": turn.get("speaker"), "text": turn.get("text")} for turn in _context_for(row, turns)]
    return f"""You are generating one probe-local evidence perturbation. This is not paraphrase and must not create a new world state. {type_instruction}

Frozen QA:
{json.dumps({k: row.get(k) for k in ('question','options','gold_letter','gold_text','evidence_turn_refs','evidence_texts')}, ensure_ascii=False, indent=2)}

Local transcript context:
{json.dumps(context, ensure_ascii=False, indent=2)}

Insert exactly one short addition after an existing turn ref. For background and competing_state, use an existing canonical speaker exactly. For entity_event_interference, you may instead introduce one clearly separate new speaker whose exact name appears in the content; never rename or merge an existing entity. The original turns and evidence remain unchanged. {enum_rule}. Use only those literal enum values. The addition must have affects_answer=false and relation_to_gold must never say contradict/conflict. Never state the gold text as the new fact. Return only JSON:
{{
  "perturbation_type": "{perturbation_type}",
  "added_information": [{{"type":"one exact allowed enum","content":"...","relation_to_gold":"...","affects_answer":false,"speaker":"exact existing speaker","insert_after_turn_ref":"exact existing turn_ref","temporal_relation":"one exact allowed enum","state_status":"one exact allowed enum"}}],
  "preservation_claim": {{"answer_unchanged":true,"world_state_unchanged":true,"timeline_unchanged":true,"entity_bindings_unchanged":true,"original_evidence_sufficient":true}},
  "preservation_audit": {{"answer_preserved":true,"initial_state_preserved":true,"current_state_preserved":true,"final_outcome_preserved":true,"entity_bindings_preserved":true,"temporal_order_preserved":true,"original_evidence_still_sufficient":true,"contradiction_added":false,"answer_leakage_added":false,"decision_stage_changed":false,"added_fact_judgments":[{{"classification":"...","affects_answer":false,"reason":"..."}}]}}
}}"""


def _fallback(row: Mapping[str, Any], turns: list[Mapping[str, Any]], perturbation_type: str) -> dict[str, Any]:
    refs = [str(ref) for ref in row.get("evidence_turn_refs") or []]
    ref = refs[0] if refs else str(turns[0].get("turn_ref"))
    speaker = next((str(turn.get("speaker")) for turn in turns if str(turn.get("turn_ref")) == ref), str(turns[0].get("speaker")))
    if perturbation_type == "background":
        kind, content, relation, temporal, state = "background", "The surrounding discussion also included routine planning details.", "non_answer_background", "concurrent_background", "background_only"
    elif perturbation_type == "competing_state":
        kind, content, relation, temporal, state = "alternative_state", "Earlier in the discussion, a different option was briefly considered and left unresolved.", "historical_non_final_alternative", "before_final_state", "historically_true_not_final"
    else:
        kind, content, relation, temporal, state = "entity_interference", "Another participant mentioned a similar topic without changing the target participant's decision.", "other_entity_only", "concurrent_background", "other_entity_only"
    return {"perturbation_type": perturbation_type, "added_information": [{
        "type": kind, "content": content, "relation_to_gold": relation,
        "affects_answer": False, "speaker": speaker, "insert_after_turn_ref": ref,
        "temporal_relation": temporal, "state_status": state}],
        "preservation_claim": {"answer_unchanged": True, "world_state_unchanged": True,
            "timeline_unchanged": True, "entity_bindings_unchanged": True,
            "original_evidence_sufficient": True},
        "preservation_audit": {"answer_preserved": False, "initial_state_preserved": False,
            "current_state_preserved": False, "final_outcome_preserved": False,
            "entity_bindings_preserved": True, "temporal_order_preserved": True,
            "original_evidence_still_sufficient": True, "contradiction_added": False,
            "answer_leakage_added": False, "decision_stage_changed": False,
            "added_fact_judgments": [{"classification": kind, "affects_answer": False,
                                       "reason": "fallback is not accepted as semantic evidence"}]}}


def _generate(client: LiveJSONClient | None, prompt: str, row: Mapping[str, Any], turns: list[Mapping[str, Any]], perturbation_type: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if client is None:
        return _fallback(row, turns, perturbation_type), {"status": "fallback_no_live"}
    try:
        raw, meta = client.complete(prompt)
        value = extract_json(raw)
        if not isinstance(value, dict):
            raise ValueError("perturbation_top_level_not_object")
        return value, {"status": "pass", **meta}
    except Exception as exc:
        return _fallback(row, turns, perturbation_type), {"status": "failed_fallback", "error": f"{type(exc).__name__}:{exc}"}


def _audit_prompt(row: Mapping[str, Any], turns: list[Mapping[str, Any]], generation: Mapping[str, Any]) -> str:
    context = [{"turn_ref": turn.get("turn_ref"), "speaker": turn.get("speaker"), "text": turn.get("text")} for turn in _context_for(row, turns)]
    return f"""You are an independent semantic preservation auditor. The generator's preservation_claim and preservation_audit are untrusted and omitted. Judge whether the proposed additions preserve the frozen QA answer, initial/current/final state, entity bindings, chronology, decision stage, and sufficiency of the unchanged original evidence. An alternative must be clearly historical and non-final. An event involving another entity must not transfer the target action. Reject unstated false facts, contradictions, answer leakage, or random disconnected distractors.

FROZEN QA AND EVIDENCE
{json.dumps({k: row.get(k) for k in ('question','options','gold_letter','gold_text','evidence_turn_refs','evidence_texts')}, ensure_ascii=False, indent=2)}

LOCAL CONTEXT
{json.dumps(context, ensure_ascii=False, indent=2)}

EXPECTED PERTURBATION TYPE
{generation.get('perturbation_type')}

PROPOSED ADDITIONS
{json.dumps(generation.get('added_information') or [], ensure_ascii=False, indent=2)}

Return only JSON with exactly these fields:
{{"answer_preserved":true,"initial_state_preserved":true,"current_state_preserved":true,"final_outcome_preserved":true,"entity_bindings_preserved":true,"temporal_order_preserved":true,"original_evidence_still_sufficient":true,"contradiction_added":false,"answer_leakage_added":false,"decision_stage_changed":false,"added_fact_judgments":[{{"classification":"one exact classification enum","affects_answer":false,"reason":"..."}}],"reason":"..."}}

Classification must be one exact value. For background use background or related_event. For competing_state use alternative_state or intermediate_state. For entity_event_interference use entity_interference or related_event. Do not join labels with punctuation."""


def _independent_audit(client: LiveJSONClient | None, row: Mapping[str, Any], turns: list[Mapping[str, Any]], generation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if client is None:
        return {}, {"status": "not_run_no_live"}
    try:
        raw, meta = client.complete(_audit_prompt(row, turns, generation))
        value = extract_json(raw)
        if not isinstance(value, dict):
            raise ValueError("semantic_audit_top_level_not_object")
        return value, {"status": "pass", **meta}
    except Exception as exc:
        return {}, {"status": "failed", "error": f"{type(exc).__name__}:{exc}"}


def _evaluate_solver(
    solver: Any, qa: Mapping[str, Any], original: list[dict[str, Any]],
    perturbed: list[dict[str, Any]], perturbation_type: str,
) -> list[dict[str, Any]]:
    rows = []
    for condition, turns in (("original_full", original), ("perturbed_full", perturbed),
                             ("original_no_context", []), ("perturbed_no_context", []),
                             ("original_option_only", []), ("perturbed_option_only", [])):
        rows.extend([{**row, "condition": condition, "perturbation_type": perturbation_type}
                     for row in solver.run(qa, turns, mode=condition, requested=5)])
    return rows


def _metrics(records: list[dict[str, Any]], solver_rows: list[dict[str, Any]], api_calls: int, solver_error: str | None) -> dict[str, Any]:
    valid = [row for row in records if row.get("generation_audit", {}).get("valid") and row.get("semantic_audit", {}).get("valid") and row.get("original_preservation", {}).get("valid")]
    by_type = {}
    for kind in TYPES:
        subset = [row for row in valid if row.get("perturbation_type") == kind]
        by_type[kind] = {"generated": sum(row.get("perturbation_type") == kind for row in records), "valid": len(subset),
                         "valid_rate": len(subset) / max(1, sum(row.get("perturbation_type") == kind for row in records))}
        typed_solver = [row for row in solver_rows if row.get("perturbation_type") == kind]
        typed_original = solver_summary([row for row in typed_solver if row.get("condition") == "original_full"])
        typed_perturbed = solver_summary([row for row in typed_solver if row.get("condition") == "perturbed_full"])
        by_type[kind]["original_accuracy"] = typed_original
        by_type[kind]["perturbed_accuracy"] = typed_perturbed
        by_type[kind]["full_accuracy_drop"] = (
            typed_original.get("accuracy") - typed_perturbed.get("accuracy")
            if typed_original.get("accuracy") is not None and typed_perturbed.get("accuracy") is not None
            else None)
    def condition(name: str) -> dict[str, Any]:
        return solver_summary([row for row in solver_rows if row.get("condition") == name])
    original = condition("original_full"); perturbed = condition("perturbed_full")
    return {"probe": "H2/P2", "probe_version": PROBE_VERSION,
            "research_question": "Can controlled context perturbation increase QA difficulty without changing factual state?",
            "legacy_baseline": {"probe": "H2/P2 legacy", "name": "lexical robustness baseline", "preserved": True,
                                "path": "../p2_source_paraphrase/"},
            "generated_perturbations": len(records), "valid_perturbations": len(valid),
            "valid_perturbation_rate": len(valid) / max(1, len(records)), "type_distribution": dict(Counter(row.get("perturbation_type") for row in records)),
            "by_type": by_type, "original_accuracy": original, "perturbed_accuracy": perturbed,
            "no_context_accuracy": {"original": condition("original_no_context"), "perturbed": condition("perturbed_no_context")},
            "option_only_accuracy": {"original": condition("original_option_only"), "perturbed": condition("perturbed_option_only")},
            "full_accuracy_drop": (original.get("accuracy") - perturbed.get("accuracy")) if original.get("accuracy") is not None and perturbed.get("accuracy") is not None else None,
            "no_context_accuracy_drop": (condition("original_no_context").get("accuracy") - condition("perturbed_no_context").get("accuracy")) if condition("original_no_context").get("accuracy") is not None and condition("perturbed_no_context").get("accuracy") is not None else None,
            "api_calls": api_calls, "solver_error": solver_error,
            "verdict": ("PROMISING" if valid and original.get("accuracy") is not None and perturbed.get("accuracy") is not None and (original.get("accuracy") - perturbed.get("accuracy")) > .0 else
                        "MIXED" if valid else "INVALID_PROBE" if records else "INSUFFICIENT_DATA")}


def _report(metrics: Mapping[str, Any], records: list[Mapping[str, Any]]) -> str:
    lines = ["# P2 Evidence Perturbation / Context Augmentation", "", "The legacy P2 source-paraphrase result is preserved as a lexical robustness baseline. This is a second independent experiment.", "", f"Research question: {metrics['research_question']}", f"Verdict: **{metrics['verdict']}**", "", "## Per-type solver accuracy", "", "| Type | Valid | Original full | Perturbed full | Drop |", "|---|---:|---:|---:|---:|"]
    for kind in TYPES:
        row = metrics["by_type"][kind]
        original = row["original_accuracy"].get("accuracy")
        perturbed = row["perturbed_accuracy"].get("accuracy")
        drop = row.get("full_accuracy_drop")
        render = lambda value: "n/a" if value is None else f"{value:.3f}"
        lines.append(f"| {kind} | {row['valid']}/{row['generated']} | {render(original)} | {render(perturbed)} | {render(drop)} |")
    lines.extend(["", "## Metrics", "", "```json", json.dumps(dict(metrics), ensure_ascii=False, indent=2), "```", "", "## Failure examples", ""])
    failures = [row for row in records if not (row.get("generation_audit", {}).get("valid") and row.get("semantic_audit", {}).get("valid") and row.get("original_preservation", {}).get("valid"))]
    if not failures:
        lines.append("None in this run.")
    for row in failures[:8]:
        lines.append(f"- `{row.get('qa_id')}:{row.get('perturbation_type')}`: generation={row.get('generation_audit')}; semantic={row.get('semantic_audit')}; preservation={row.get('original_preservation')}")
    lines.extend(["", "## Interpretation", "", "A full-context accuracy drop is evidence only when preservation audits pass. If no drop occurs, the perturbation was insufficient; if audits fail, the data is invalid rather than difficult."])
    return "\n".join(lines) + "\n"


def run_probe(*, run_root: Path, config: Path | None = None, live: bool = False, max_items: int = 3) -> dict[str, Any]:
    prepare_empty(run_root)
    data = load_frozen_inputs(DEFAULT_RUN)
    client = None
    if live and config is not None:
        try:
            client = LiveJSONClient(config, max_tokens=1500, temperature=.25)
        except Exception as exc:
            (run_root / "live_init_error.txt").write_text(f"{type(exc).__name__}:{exc}\n", encoding="utf-8")
    solver = None
    solver_error = "not_loaded_without_live_generations" if not live else None
    turns = data["dialogue"]["turns"]
    eligible = [row for row in data["qa_rows"]
                if row["eligible_grounded"] and row["full_context_total"] == row["full_context_correct"]]
    selected = sorted(
        eligible,
        key=lambda row: (row.get("evaluation", {}).get("grounding", {}).get(
            "semantic_correct_rate", 1.0), str(row.get("qa_id"))),
    )[:max_items]
    records: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    evaluation_cases: list[tuple[Mapping[str, Any], list[dict[str, Any]], str]] = []
    try:
        for row in selected:
            for perturbation_type in TYPES:
                parsed, generation_meta = _generate(client, _prompt(row, turns, perturbation_type), row, turns, perturbation_type)
                parsed = parsed or {}
                generation_audit = validate_generation(parsed, perturbation_type=perturbation_type, qa=row,
                    valid_turn_refs=[str(turn.get("turn_ref")) for turn in turns], valid_speakers=[str(turn.get("speaker")) for turn in turns])
                independent_value, auditor_meta = _independent_audit(client, row, turns, parsed)
                semantic_audit = validate_semantic_audit(
                    independent_value, perturbation_type=perturbation_type,
                    expected_addition_count=len(parsed.get("added_information") or []))
                perturbed, added_refs = materialize_view(turns, qa_id=str(row["qa_id"]), perturbation_type=perturbation_type,
                    additions=parsed.get("added_information") if isinstance(parsed.get("added_information"), list) else [])
                preservation = attest_original_preservation(turns, perturbed, row.get("evidence_turn_refs") or [])
                record = {"qa_id": row["qa_id"], "question": row["question"], "options": row["options"], "gold_letter": row["gold_letter"], "gold_text": row["gold_text"],
                          "perturbation_type": perturbation_type, "generation_meta": generation_meta, "auditor_meta": auditor_meta,
                          "generation": parsed,
                          "generation_audit": generation_audit, "semantic_audit": semantic_audit, "original_preservation": preservation,
                          "added_turn_refs": added_refs, "original_evidence": row.get("evidence_texts"),
                          "perturbed_evidence": [{"turn_ref": turn.get("turn_ref"), "speaker": turn.get("speaker"), "text": turn.get("text"), "origin": turn.get("origin")}
                              for turn in perturbed if turn.get("turn_ref") in set(row.get("evidence_turn_refs") or []) or turn.get("turn_ref") in set(added_refs)]}
                records.append(record)
                accepted = generation_audit["valid"] and semantic_audit["valid"] and preservation["valid"]
                if accepted:
                    evaluation_cases.append((row, perturbed, perturbation_type))
        if evaluation_cases:
            solver, solver_error = maybe_solver(run_root)
            if solver is not None:
                for row, perturbed, perturbation_type in evaluation_cases:
                    solver_rows.extend(_evaluate_solver(solver, row, turns, perturbed, perturbation_type))
        elif live:
            solver_error = "not_loaded_no_valid_perturbations"
    finally:
        if solver is not None:
            solver.close()
    metrics = _metrics(records, solver_rows, client.calls if client else 0, solver_error)
    write_jsonl(run_root / "inputs.jsonl", [{"source": data["inputs"], "qa_ids": [row["qa_id"] for row in selected], "legacy_p2_preserved": True}])
    write_jsonl(run_root / "generations.jsonl", records)
    write_jsonl(run_root / "audits.jsonl", [{"qa_id": row["qa_id"], "perturbation_type": row["perturbation_type"], "generation_audit": row["generation_audit"], "semantic_audit": row["semantic_audit"], "original_preservation": row["original_preservation"]} for row in records])
    write_jsonl(run_root / "solver_rollouts.jsonl", solver_rows)
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {"probe": "H2/P2", "probe_version": PROBE_VERSION, "legacy_p2_preserved": True,
        "legacy_probe_path": "p2_source_paraphrase", "api_calls": client.calls if client else 0, "writer_calls": 0, "memorax_add_calls": 0,
        "outputs": ["inputs.jsonl", "generations.jsonl", "audits.jsonl", "solver_rollouts.jsonl", "metrics.json", "report.md", "manifest.json"]})
    (run_root / "report.md").write_text(_report(metrics, records), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="independent P2 evidence perturbation probe")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-items", type=int, default=3)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"difficulty_probe_suite_v1_p2_evidence_perturbation_{stamp}").resolve()
    metrics = run_probe(run_root=run_root, config=args.config, live=args.live, max_items=args.max_items)
    print(json.dumps({"run_root": str(run_root), "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
