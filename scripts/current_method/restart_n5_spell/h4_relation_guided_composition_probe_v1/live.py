"""H4 relation-guided composition gate isolation probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.request

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import load_config
from scripts.current_method.restart_n5_spell.simple_append_v1.live_runtime import load_json_object_strict
from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.static_audit import audit_composite

from .composer_prompt import build_request
from .gate import build_current_summary, build_relation_guided_candidates, load_atomic_ir
from .solver_eval import evaluate_composite


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call_composer(*, request: dict[str, Any], config_path: Path, run_root: Path) -> dict[str, Any]:
    api = load_config(config_path)
    payload = {
        "model": api["model"],
        "messages": [{"role": "user", "content": request["prompt"]}],
        "temperature": 0.2,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    http_request = urllib.request.Request(
        str(api["base_url"]).rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": "Bearer " + str(api["api_key"]),
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(http_request, timeout=600) as response:
        envelope = json.loads(response.read().decode())
    choice = (envelope.get("choices") or [{}])[0]
    raw = str((choice.get("message") or {}).get("content") or "")
    parsed = load_json_object_strict(raw)
    write_json(run_root / "composer_response_envelope.json", parsed)
    (run_root / "composer_raw_response.txt").write_text(raw, encoding="utf-8")
    write_json(run_root / "composer_response_meta.json", {
        "model": api["model"], "usage": envelope.get("usage"),
        "finish_reason": choice.get("finish_reason"), "api_calls": 1,
    })
    return parsed


def run(*, atomic_normalized: Path, current_composition: Path,
        expanded_dialogue: Path, relation_contract: Path, relation_audit: Path,
        config: Path, tokenizer: Path, solver: Path, run_root: Path) -> dict[str, Any]:
    prepare_empty(run_root)
    atomic = load_atomic_ir(atomic_normalized)
    contract = _load(relation_contract)
    audit = _load(relation_audit)
    expanded = _load(expanded_dialogue)
    write_json(run_root / "run_freeze_manifest.json", {
        "freeze_version": "h4_relation_guided_composition_probe_v1",
        "atomic_normalized": {"path": str(atomic_normalized), "sha256": _sha256(atomic_normalized)},
        "current_composition": str(current_composition),
        "expanded_dialogue": {"path": str(expanded_dialogue), "sha256": _sha256(expanded_dialogue)},
        "relation_contract": {"path": str(relation_contract), "sha256": _sha256(relation_contract)},
        "relation_audit": {"path": str(relation_audit), "sha256": _sha256(relation_audit)},
        "condition_a_composer_calls": 0,
        "condition_b_composer_calls": 1,
        "planner_calls": 0, "realizer_calls": 0, "qa_writer_calls": 0,
        "memorax_calls": 0,
    })
    write_json(run_root / "run_status.json", {"status": "initialized", "api_calls": 0})
    current = build_current_summary(current_composition)
    guided = build_relation_guided_candidates(
        atomic, relation_audit=audit, relation_contract=contract)
    write_json(run_root / "relation_guided_candidates.json", guided)
    if not guided:
        result = {"status": "no_relation_guided_candidate", "current": current,
                  "relation_guided_candidates": 0, "relation_guided_composer_ready": 0}
        write_json(run_root / "metrics.json", result)
        return result
    by_id = {qa.qa_id: qa for qa in atomic}
    pair = guided[0]
    left, right = (by_id[pair["qa_ids"][0]], by_id[pair["qa_ids"][1]])
    request = build_request(pair=pair, left=left, right=right,
                            relation_contract=contract, relation_audit=audit)
    write_json(run_root / "relation_guided_composer_prompt.json", request)
    response = _call_composer(request=request, config_path=config, run_root=run_root)
    static = audit_composite(response, atomic_by_id=by_id)
    write_json(run_root / "composer_static_audit.json", static)
    solver_eval = None
    if static["valid"]:
        solver_eval = evaluate_composite(
            composite=response, expanded=expanded,
            relation_source_turns=audit.get("relation_source_turns") or {},
            run_root=run_root, tokenizer_path=tokenizer, solver_model=solver)
    else:
        write_json(run_root / "solver_evaluation.json", {
            "status": "not_run_static_audit_failed", "necessity_pass": False})
    response_binding = response.get("relation_binding") or {}
    metrics = {
        "probe_version": "h4_relation_guided_composition_probe_v1",
        "status": "completed",
        "analysis_execution_api_calls": 0,
        "composer_api_calls": 1,
        "current_pair_candidates": current["current_pair_candidates"],
        "relation_guided_candidates": len(guided),
        "current_composer_ready": current["current_composer_ready"],
        "relation_guided_composer_ready": sum(bool(row.get("composer_ready")) for row in guided),
        "composite_generated": int(static["valid"]),
        "static_audit": static,
        "relation_used": bool(static["valid"]),
        "fact_A_used": response_binding.get("used_fact_A") is True,
        "trigger_used": response_binding.get("used_trigger") is True,
        "fact_B_used": response_binding.get("used_fact_B") is True,
        "full_context_accuracy": solver_eval.get("full_context_accuracy") if solver_eval else None,
        "remove_A_accuracy": solver_eval.get("remove_A_accuracy") if solver_eval else None,
        "remove_trigger_accuracy": solver_eval.get("remove_trigger_accuracy") if solver_eval else None,
        "remove_B_accuracy": solver_eval.get("remove_B_accuracy") if solver_eval else None,
        "necessity_pass": solver_eval.get("necessity_pass") if solver_eval else False,
        "admitted": bool(static["valid"] and solver_eval and solver_eval.get("necessity_pass")),
        "current": current,
        "guided_pair": pair,
    }
    write_json(run_root / "metrics.json", metrics)
    conclusion = (
        "COMPOSITION_GATE_BOTTLENECK_CONFIRMED" if metrics["admitted"] and current["current_composer_ready"] == 0 else
        "RELATION_GUIDED_COMPOSITE_GENERATED_BUT_NECESSITY_FAILED" if static["valid"] and not metrics["admitted"] else
        "RELATION_GUIDED_COMPOSER_RESPONSE_INVALID")
    report = [
        "# H4 Relation-guided Composition Gate Isolation Probe v1", "",
        f"Conclusion: **{conclusion}**", "",
        "Condition A is the frozen current pair compiler result. Condition B "
        "uses the same atomic QA and adds only a relation-guided pair/composer wrapper.", "",
        "## Composition", "",
        f"- Current pair candidates: `{current['current_pair_candidates']}`",
        f"- Relation-guided candidates: `{len(guided)}`",
        f"- Current Composer-ready: `{current['current_composer_ready']}`",
        f"- Relation-guided Composer-ready: `{metrics['relation_guided_composer_ready']}`",
        f"- Composite generated: `{metrics['composite_generated']}`",
        "", "## Relation binding", "",
        f"- fact_A used: `{metrics['fact_A_used']}`",
        f"- trigger used: `{metrics['trigger_used']}`",
        f"- fact_B used: `{metrics['fact_B_used']}`", "",
        "## Difficulty", "",
        f"- Full context accuracy: `{metrics['full_context_accuracy']}`",
        f"- Remove A accuracy: `{metrics['remove_A_accuracy']}`",
        f"- Remove trigger accuracy: `{metrics['remove_trigger_accuracy']}`",
        f"- Remove B accuracy: `{metrics['remove_B_accuracy']}`",
        f"- Necessity pass: `{metrics['necessity_pass']}`",
        f"- Admitted: `{metrics['admitted']}`", "",
        "No Planner, Realizer, QA Writer, or Memorax call was made. The Solver "
        "was used only as the unchanged local K=5/ablation evaluator.", "",
    ]
    (run_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="H4 relation-guided composition gate probe")
    parser.add_argument("--atomic-normalized", type=Path, required=True)
    parser.add_argument("--current-composition", type=Path, required=True)
    parser.add_argument("--expanded-dialogue", type=Path, required=True)
    parser.add_argument("--relation-contract", type=Path, required=True)
    parser.add_argument("--relation-audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    result = run(
        atomic_normalized=args.atomic_normalized.resolve(),
        current_composition=args.current_composition.resolve(),
        expanded_dialogue=args.expanded_dialogue.resolve(),
        relation_contract=args.relation_contract.resolve(),
        relation_audit=args.relation_audit.resolve(), config=args.config.resolve(),
        tokenizer=args.tokenizer.resolve(), solver=args.solver.resolve(),
        run_root=args.run_root.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
