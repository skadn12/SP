"""Live, isolated H4 relation-contract refinement probe."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import LiveJSONClient
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.batch_probe_v1 import _adapt_plan, _find_source
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.scene_release_probe_v2 import _renumber
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.scene_release_probe_v3 import _safe_extract
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import normalize_realizer_output

from .audit import audit_relation
from .contracts import build_relation_contract
from .prompt import build_relation_scene_prompt


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "h4_relation_contract_refinement_probe_v1"
MAX_TOKENS = 8192


def run(*, run_root: Path, config: Path, case_id: str = "POB03") -> dict[str, Any]:
    prepare_empty(run_root)
    source, provenance = _find_source(case_id)
    parent, plan = provenance["parent"], _adapt_plan(source)
    contract = build_relation_contract(plan)
    write_json(run_root / "source_record.json", source)
    write_json(run_root / "adapted_plan.json", plan)
    write_json(run_root / "relation_contract.json", contract)
    client = LiveJSONClient(config, max_tokens=MAX_TOKENS, temperature=0.0)
    prior: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    transport_error = None
    for index, current in enumerate(plan["scenes"], 1):
        scene_root = run_root / f"scene_{index:02d}"
        scene_root.mkdir(parents=True, exist_ok=True)
        prompt = build_relation_scene_prompt(
            parent=parent, prior=prior, current=current, index=index,
            contract=contract)
        write_json(scene_root / "prompt.json", prompt)
        raw, meta, parsed, error = "", {}, {}, None
        calls_before = client.calls
        try:
            raw, meta = client.complete(prompt["prompt"])
            (scene_root / "raw_response.txt").write_text(raw, encoding="utf-8")
            parsed, mode = _safe_extract(raw)
            write_json(scene_root / "parsed_response.json", parsed)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            transport_error = error
            (scene_root / "raw_response.txt").write_text(raw, encoding="utf-8")
            mode = None
        mechanical = (
            normalize_realizer_output(
                parsed, plan={"scenes": [current]}, parent=parent, round_index=1)
            if error is None else {"valid": False, "errors": [error]})
        if mechanical.get("valid") and mechanical.get("child", {}).get("scenes"):
            scene = _renumber(
                mechanical["child"]["scenes"][0], index,
                str(parent["root_id"]), str(current.get("memory_event_id")))
            prior.append(scene)
            write_json(scene_root / "child_scene.json", scene)
        record = {
            "scene_index": index,
            "request_attempts": 1,
            "realizer_calls_completed": client.calls - calls_before,
            "response_meta": meta,
            "prompt_chars": len(prompt["prompt"]),
            "max_tokens": MAX_TOKENS,
            "transport_error": error,
            "mechanical_contract_valid": bool(mechanical.get("valid")),
            "envelope_mode": mode,
            "relation_contract_injected": True,
        }
        write_json(scene_root / "mechanical_audit.json", mechanical)
        write_json(scene_root / "metrics.json", record)
        records.append(record)
        if error or not mechanical.get("valid"):
            break
    child = {
        "schema_version": "memory_event_continuation_child_v1",
        "root_id": parent["root_id"], "round_index": 1, "scenes": prior,
    }
    write_json(run_root / "child_dialogue.json", child)
    relation_audit = audit_relation(child, contract)
    write_json(run_root / "relation_audit.json", relation_audit)
    if len(prior) == 3 and relation_audit["relation_audit_pass"]:
        conclusion = "RELATION_CONTRACT_PASS"
    elif transport_error:
        conclusion = "RELATION_CONTRACT_UNDECIDED_TRANSPORT"
    else:
        conclusion = "RELATION_CONTRACT_FAIL"
    metrics = {
        "probe_version": PROBE_VERSION,
        "hypothesis": "H4",
        "execution_variant": "relation_contract_refinement",
        "case_id": case_id,
        "root_id": parent["root_id"],
        "relation_type": contract["relation_type"],
        "scene_records": records,
        "scenes_completed": len(prior),
        "realizer_calls_completed": client.calls,
        "transport_error": transport_error,
        "relation_present": relation_audit["relation_present"],
        "relation_audit_pass": relation_audit["relation_audit_pass"],
        "trigger_supported": relation_audit["trigger_supported"],
        "state_chain_valid": relation_audit["state_chain_valid"],
        "temporal_leakage_free": relation_audit["temporal_leakage_free"],
        "early_relation_leakage": relation_audit["early_relation_leakage"],
        "relation_source_turns": relation_audit["relation_source_turns"],
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "atomic_generated": 0,
        "pair_candidate": 0,
        "composer_ready": 0,
        "admitted": 0,
        "ablation_status": "pending_qa_pipeline",
        "conclusion": conclusion,
        "analysis_execution_api_calls": 0,
        "planner_calls": 0,
        "auditor_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
    }
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION,
        "hypothesis": "H4",
        "isolated": True,
        "refinement_of": ["H4A", "H4B"],
        "mainline_modified": False,
        "production_writer_modified": False,
        "production_solver_modified": False,
        "memorax_modified": False,
        "one_request_per_scene": True,
        "future_plans_in_any_scene_prompt": False,
        "retry_budget": 0,
        "api_role_allowlist": ["realizer"],
        "max_tokens": MAX_TOKENS,
        "qa_stage": "not_run_by_generation_probe",
    })
    report = [
        "# H4 Relation-Contract Refinement Probe v1", "",
        f"Verdict: **{conclusion}**", "",
        f"- Root: `{parent['root_id']}`", f"- Relation type: `{contract['relation_type']}`",
        f"- Scenes completed: `{len(prior)}/3`", f"- Realizer calls: `{client.calls}`",
        f"- relation_present: `{relation_audit['relation_present']}`",
        f"- relation_audit_pass: `{relation_audit['relation_audit_pass']}`",
        f"- trigger_supported: `{relation_audit['trigger_supported']}`",
        f"- state_chain_valid: `{relation_audit['state_chain_valid']}`", "",
        f"- temporal_leakage_free: `{relation_audit['temporal_leakage_free']}`",
        f"- early_relation_leakage: `{relation_audit['early_relation_leakage']}`", "",
        "This is an H4 refinement probe, not a replacement for H4A/H4B. "
        "It changes only the isolated Realizer contract and leaves the production "
        "Planner, QA Writer/Solver, Memorax, and historical runs untouched.", "",
        "The existing QA pipeline must be run on `child_dialogue.json` before "
        "QA-utilization and single-fact ablation can be scored.", "",
    ]
    (run_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="isolated H4 relation-contract refinement probe")
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--case-id", default="POB03")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    result = run(run_root=run_root, config=args.config, case_id=args.case_id)
    print(json.dumps({"run_root": str(run_root), "metrics": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
