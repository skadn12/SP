"""Live isolated candidate-selection and composition probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty, write_json,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import load_config
from scripts.current_method.restart_n5_spell.simple_append_v1.live_runtime import load_json_object_strict
from scripts.current_method.restart_n5_spell.h4_relation_guided_composition_probe_v1.gate import load_atomic_ir

from .candidate_audit import audit_selector_output
from .composer_contract import assemble_canonical_response
from .composer_prompt import build_request as build_composer_request
from .composite_audit import audit_candidate_composite
from .selector_prompt import build_selector_request
from .solver_eval import evaluate_candidate


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dialogue_rows(expanded: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("turn_id") and value.get("text") is not None:
                rows.append({"turn_id": str(value["turn_id"]), "speaker": value.get("speaker"), "text": value.get("text")})
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(expanded)
    seen: set[str] = set()
    return [row for row in rows if not (row["turn_id"] in seen or seen.add(row["turn_id"]))]


def _call_json(*, prompt: str, request_id: str, config_path: Path,
               run_root: Path, stem: str, max_tokens: int = 4096) -> dict[str, Any]:
    api = load_config(config_path)
    payload = {
        "model": api["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        str(api["base_url"]).rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": "Bearer " + str(api["api_key"]),
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        envelope = json.loads(response.read().decode())
    choice = (envelope.get("choices") or [{}])[0]
    raw = str((choice.get("message") or {}).get("content") or "")
    parsed = load_json_object_strict(raw)
    directory = run_root / "api"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}_raw_response.txt").write_text(raw, encoding="utf-8")
    write_json(directory / f"{stem}_response_envelope.json", parsed)
    write_json(directory / f"{stem}_response_meta.json", {
        "request_id": request_id, "model": api["model"],
        "usage": envelope.get("usage"), "finish_reason": choice.get("finish_reason"),
        "api_calls": 1,
    })
    return parsed


def run(*, atomic_normalized: Path, expanded_dialogue: Path,
        config: Path, tokenizer: Path, solver: Path, output: Path) -> dict[str, Any]:
    prepare_empty(output)
    atomic = load_atomic_ir(atomic_normalized)
    expanded = _load(expanded_dialogue)
    dialogue_rows = _dialogue_rows(expanded)
    by_id = {qa.qa_id: qa for qa in atomic}
    write_json(output / "run_freeze_manifest.json", {
        "freeze_version": "h4_relation_aware_composition_probe_v1",
        "atomic_normalized": {"path": str(atomic_normalized), "sha256": _sha256(atomic_normalized)},
        "expanded_dialogue": {"path": str(expanded_dialogue), "sha256": _sha256(expanded_dialogue)},
        "eligible_atomic_count": len(atomic), "source_artifacts_mutated": False,
        "selector_q17_q21_hint": False, "planner_calls": 0, "realizer_calls": 0,
        "qa_writer_calls": 0, "memorax_calls": 0,
    })
    write_json(output / "atomic_pool.json", {"eligible_atomic_count": len(atomic), "items": [
        {"qa_id": qa.qa_id, "question": qa.question, "gold_text": qa.gold_text,
         "source_scope": qa.source_scope, "question_type": qa.question_type,
         "evidence_turn_refs": qa.evidence_turn_refs, "evidence_texts": qa.evidence_texts}
        for qa in atomic
    ]})
    selector_request = build_selector_request(atomic)
    write_json(output / "selector_prompt.json", selector_request)
    selector = _call_json(
        prompt=selector_request["prompt"], request_id=selector_request["request_id"],
        config_path=config, run_root=output, stem="selector", max_tokens=4096)
    write_json(output / "selector_response.json", selector)
    selector_audit = audit_selector_output(
        selector, atomic_by_id=by_id, dialogue_rows=dialogue_rows)
    write_json(output / "selector_audit.json", selector_audit)
    valid_candidates = [row for row in selector_audit["candidates"] if row["static_valid"]]
    candidate_results: list[dict[str, Any]] = []
    composer_calls = 0
    solver_calls = 0
    for candidate in valid_candidates[:5]:
        candidate_dir = output / "candidates" / candidate["candidate_id"]
        candidate_dir.mkdir(parents=True, exist_ok=False)
        request = build_composer_request(candidate=candidate, atomic_by_id=by_id)
        write_json(candidate_dir / "composer_prompt.json", request)
        response = _call_json(
            prompt=request["prompt"], request_id=request["request_id"],
            config_path=config, run_root=candidate_dir, stem="composer", max_tokens=3072)
        composer_calls += 1
        write_json(candidate_dir / "composer_response.json", response)
        canonical, adapter_audit = assemble_canonical_response(response, candidate=candidate)
        write_json(candidate_dir / "composer_adapter_audit.json", adapter_audit)
        write_json(candidate_dir / "canonical_composite.json", canonical)
        static = audit_candidate_composite(
            canonical, candidate=candidate, atomic_by_id=by_id)
        write_json(candidate_dir / "static_audit.json", static)
        solver_eval = None
        if static["valid"]:
            solver_eval = evaluate_candidate(
                composite=canonical, candidate=candidate, expanded=expanded,
                run_root=candidate_dir, tokenizer_path=tokenizer, solver_model=solver,
                freeze_manifest=output / "run_freeze_manifest.json")
            solver_calls += 4 + 5 * (1 + len(candidate["source_qids"]))
        result = {
            "candidate_id": candidate["candidate_id"],
            "source_qids": candidate["source_qids"],
            "target": candidate["target"],
            "static_valid": static["valid"],
            "static_errors": static["errors"],
            "full_context_accuracy": solver_eval.get("full_context_accuracy") if solver_eval else None,
            "no_context_accuracy": solver_eval.get("no_context_accuracy") if solver_eval else None,
            "source_ablation_accuracy": {
                key: value.get("accuracy") for key, value in (solver_eval or {}).get("ablations", {}).items()
            },
            "necessity_pass": solver_eval.get("necessity_pass") if solver_eval else False,
            "strict_admitted": bool(static["valid"] and solver_eval and solver_eval.get("necessity_pass")),
        }
        write_json(candidate_dir / "result.json", result)
        candidate_results.append(result)
    hidden_q17 = [row for row in valid_candidates if set(row["source_qids"]) == {
        "cc-train-episode-178392-simple-append-mcq-17",
        "cc-train-episode-178392-simple-append-mcq-21",
    }]
    admitted = sum(bool(row["strict_admitted"]) for row in candidate_results)
    metrics = {
        "probe_version": "h4_relation_aware_composition_probe_v1",
        "status": "completed", "analysis_execution_api_calls": composer_calls + 1,
        "selector_api_calls": 1, "composer_api_calls": composer_calls,
        "solver_local_generations": solver_calls, "source_artifacts_mutated": False,
        "atomic_qa_count": len(atomic), "candidate_proposed": len(selector_audit["candidates"]),
        "candidate_static_valid": len(valid_candidates),
        "composite_qa_generated": sum(row["static_valid"] for row in candidate_results),
        "full_context_5_of_5": sum(row["full_context_accuracy"] == 1.0 for row in candidate_results),
        "no_context_not_all_correct": sum(
            row["no_context_accuracy"] is None or row["no_context_accuracy"] < 1.0 for row in candidate_results),
        "all_source_necessity_passed": sum(bool(row["necessity_pass"]) for row in candidate_results),
        "strict_admitted": admitted,
        "q17_q21_auto_discovered": bool(hidden_q17),
        "q17_q21_candidate_ids": [row["candidate_id"] for row in hidden_q17],
        "candidate_results": candidate_results,
        "selector_errors": selector_audit["errors"],
        "conclusion": (
            "STRICT_COMPOSITE_FOUND"
            if admitted else
            "CANDIDATES_OR_COMPOSITES_FOUND_BUT_STRICT_NECESSITY_FAILED"
            if candidate_results else
            "NO_STATIC_VALID_CANDIDATE"
        ),
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "request_ledger.json", {
        "probe_version": "h4_relation_aware_composition_probe_v1",
        "selector_requests": 1, "composer_requests": composer_calls,
        "solver_local_generations": solver_calls,
    })
    write_json(output / "POST_RUN_INTEGRITY.json", {
        "status": "complete", "source_artifacts_mutated": False,
        "selector_api_calls": 1, "composer_api_calls": composer_calls,
        "strict_admitted": admitted,
    })
    report = [
        "# H4 Relation-aware Composition Probe v1", "",
        f"Conclusion: **{metrics['conclusion']}**", "",
        "The Candidate Selector was not told about Q17+Q21; that pair is only "
        "checked after selection as a hidden sanity candidate.", "",
        f"- Atomic QA count (eligible): `{metrics['atomic_qa_count']}`",
        f"- Candidate proposed: `{metrics['candidate_proposed']}`",
        f"- Candidate static valid: `{metrics['candidate_static_valid']}`",
        f"- Composite QA generated: `{metrics['composite_qa_generated']}`",
        f"- Full-context 5/5: `{metrics['full_context_5_of_5']}`",
        f"- No-context not all-correct: `{metrics['no_context_not_all_correct']}`",
        f"- All-source necessity passed: `{metrics['all_source_necessity_passed']}`",
        f"- Strict admitted: `{metrics['strict_admitted']}`",
        f"- Q17+Q21 auto-discovered: `{metrics['q17_q21_auto_discovered']}`", "",
        "## Candidate results", "",
    ]
    for row in candidate_results:
        report.extend([
            f"- `{row['candidate_id']}` sources=`{row['source_qids']}` target=`{row['target']}`",
            f"  static=`{row['static_valid']}` errors=`{row['static_errors']}`",
            f"  full=`{row['full_context_accuracy']}` no-context=`{row['no_context_accuracy']}`",
            f"  source ablations=`{row['source_ablation_accuracy']}` necessity=`{row['necessity_pass']}` admitted=`{row['strict_admitted']}`",
        ])
    report.extend(["", "No Planner, Realizer, atomic QA Writer, production QA, or Memorax call was made.", ""])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic-normalized", type=Path, required=True)
    parser.add_argument("--expanded-dialogue", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        atomic_normalized=args.atomic_normalized.resolve(),
        expanded_dialogue=args.expanded_dialogue.resolve(), config=args.config.resolve(),
        tokenizer=args.tokenizer.resolve(), solver=args.solver.resolve(),
        output=args.output.resolve(),
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
