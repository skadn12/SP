"""Run the QA-only targeted distractor probe on frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import load_config
from scripts.current_method.restart_n5_spell.simple_append_v1.live_runtime import load_json_object_strict
from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.static_audit import audit_composite
from scripts.current_method.restart_n5_spell.h4_relation_guided_composition_probe_v1.gate import (
    build_relation_guided_candidates, load_atomic_ir,
)
from scripts.current_method.restart_n5_spell.h4_relation_guided_composition_probe_v1.solver_eval import evaluate_composite

from .prompt import build_request
from .slot_audit import audit_targeted
from .admission import strict_necessity_admission


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
        "temperature": 0.0,
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


def run(*, atomic_normalized: Path, expanded_dialogue: Path,
        relation_contract: Path, relation_audit: Path, config: Path,
        tokenizer: Path, solver: Path, output: Path) -> dict[str, Any]:
    prepare_empty(output)
    atomic = load_atomic_ir(atomic_normalized)
    contract = _load(relation_contract)
    audit = _load(relation_audit)
    expanded = _load(expanded_dialogue)
    guided = build_relation_guided_candidates(
        atomic, relation_audit=audit, relation_contract=contract)
    if not guided:
        result = {"status": "no_relation_guided_candidate", "composer_api_calls": 0,
                  "targeted_composer_ready": 0, "admitted": False}
        write_json(output / "metrics.json", result)
        return result
    pair = guided[0]
    by_id = {qa.qa_id: qa for qa in atomic}
    left, right = (by_id[pair["qa_ids"][0]], by_id[pair["qa_ids"][1]])
    request = build_request(
        pair=pair, left=left, right=right,
        relation_contract=contract, relation_audit=audit)
    write_json(output / "targeted_composer_prompt.json", request)
    response = _call_composer(request=request, config_path=config, run_root=output)
    static = audit_composite(response, atomic_by_id=by_id)
    targeted = audit_targeted(
        response, atomic_by_id=by_id, relation_contract=contract)
    write_json(output / "composer_static_audit.json", static)
    write_json(output / "targeted_slot_audit.json", targeted)
    solver_eval = None
    if targeted["valid"]:
        solver_eval = evaluate_composite(
            composite=response, expanded=expanded,
            relation_source_turns=audit.get("relation_source_turns") or {},
            run_root=output, tokenizer_path=tokenizer, solver_model=solver)
    else:
        write_json(output / "solver_evaluation.json", {
            "status": "not_run_targeted_static_audit_failed", "necessity_pass": False,
        })
    binding = response.get("relation_binding") or {}
    necessity_pass, admission = strict_necessity_admission(solver_eval or {})
    result = {
        "probe_version": "h4_qa_targeted_distractor_necessity_probe_v1",
        "status": "completed", "composer_api_calls": 1,
        "source_artifacts_mutated": False,
        "relation_guided_candidates": len(guided),
        "targeted_composer_ready": int(targeted["valid"]),
        "composite_generated": int(static["valid"]),
        "targeted_audit_valid": targeted["valid"],
        "relation_binding": {
            "used_fact_A": binding.get("used_fact_A") is True,
            "used_trigger": binding.get("used_trigger") is True,
            "used_fact_B": binding.get("used_fact_B") is True,
        },
        "full_context_accuracy": solver_eval.get("full_context_accuracy") if solver_eval else None,
        "no_context_accuracy": (solver_eval or {}).get("grounding", {}).get("semantic_correct_rate") if solver_eval else None,
        "remove_A_accuracy": solver_eval.get("remove_A_accuracy") if solver_eval else None,
        "remove_trigger_accuracy": solver_eval.get("remove_trigger_accuracy") if solver_eval else None,
        "remove_B_accuracy": solver_eval.get("remove_B_accuracy") if solver_eval else None,
        "necessity_pass": necessity_pass,
        "admission": admission,
        "admitted": bool(targeted["valid"] and solver_eval and necessity_pass),
        "guided_pair": pair,
    }
    write_json(output / "metrics.json", result)
    write_json(output / "run_freeze_manifest.json", {
        "freeze_version": "h4_qa_targeted_distractor_necessity_probe_v1",
        "atomic_normalized": {"path": str(atomic_normalized), "sha256": _sha256(atomic_normalized)},
        "expanded_dialogue": {"path": str(expanded_dialogue), "sha256": _sha256(expanded_dialogue)},
        "relation_contract": {"path": str(relation_contract), "sha256": _sha256(relation_contract)},
        "relation_audit": {"path": str(relation_audit), "sha256": _sha256(relation_audit)},
        "planner_calls": 0, "realizer_calls": 0, "qa_writer_calls": 0,
        "memorax_calls": 0, "composer_calls": 1,
    })
    write_json(output / "POST_RUN_INTEGRITY.json", {
        "status": "complete", "probe_version": "h4_qa_targeted_distractor_necessity_probe_v1",
        "source_artifacts_mutated": False, "composer_api_calls": 1,
        "admitted": result["admitted"],
    })
    report = [
        "# H4 QA Targeted Distractor Necessity Probe v1", "",
        f"- Targeted static audit: `{targeted['valid']}`",
        f"- Composite generated: `{static['valid']}`",
        f"- Full-context accuracy: `{result['full_context_accuracy']}`",
        f"- No-context accuracy: `{result['no_context_accuracy']}`",
        f"- Remove fact_A: `{result['remove_A_accuracy']}`",
        f"- Remove trigger: `{result['remove_trigger_accuracy']}`",
        f"- Remove fact_B: `{result['remove_B_accuracy']}`",
        f"- Necessity pass: `{result['necessity_pass']}`",
        f"- Admitted: `{result['admitted']}`", "",
        "This probe used one isolated Composer call and unchanged local Solver. "
        "Planner, Realizer, production QA Writer, expansion runs, and old results were not modified.", "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic-normalized", type=Path, required=True)
    parser.add_argument("--expanded-dialogue", type=Path, required=True)
    parser.add_argument("--relation-contract", type=Path, required=True)
    parser.add_argument("--relation-audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        atomic_normalized=args.atomic_normalized.resolve(),
        expanded_dialogue=args.expanded_dialogue.resolve(),
        relation_contract=args.relation_contract.resolve(),
        relation_audit=args.relation_audit.resolve(), config=args.config.resolve(),
        tokenizer=args.tokenizer.resolve(), solver=args.solver.resolve(),
        output=args.output.resolve(),
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
