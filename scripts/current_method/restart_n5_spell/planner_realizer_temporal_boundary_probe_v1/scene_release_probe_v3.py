from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import LiveJSONClient
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import normalize_realizer_output
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.batch_probe_v1 import _adapt_plan, _audit, _find_source
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.scene_release_probe_v2 import _renumber, build_single_scene_prompt_v2

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "planner_realizer_temporal_scene_release_probe_v3"


def _safe_extract(raw: str) -> tuple[Any, str]:
    value = json.loads(raw)
    if isinstance(value, Mapping) and set(value) == {"scenes"}:
        return value, "direct_object"
    if isinstance(value, Mapping):
        for key in value:
            if isinstance(key, str) and key.lstrip().startswith("{\"scenes\""):
                nested = json.loads(key)
                if isinstance(nested, Mapping) and set(nested) == {"scenes"}:
                    return nested, "nested_json_key"
    raise ValueError("unsupported_realizer_envelope")


def run(*, run_root: Path, config: Path, case_id: str = "POB03") -> dict[str, Any]:
    prepare_empty(run_root)
    source, provenance = _find_source(case_id)
    parent, plan = provenance["parent"], _adapt_plan(source)
    write_json(run_root / "source_record.json", source)
    write_json(run_root / "adapted_plan.json", plan)
    client = LiveJSONClient(config, max_tokens=8192, temperature=0.0)
    prior: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    transport_error = None
    envelope_modes: list[str] = []
    for index, current in enumerate(plan["scenes"], 1):
        root = run_root / f"scene_{index:02d}"
        root.mkdir(parents=True, exist_ok=True)
        prompt = build_single_scene_prompt_v2(parent=parent, prior=prior, current=current, index=index)
        write_json(root / "prompt.json", prompt)
        raw, meta, parsed, error = "", {}, {}, None
        try:
            raw, meta = client.complete(prompt["prompt"])
            (root / "raw_response.txt").write_text(raw, encoding="utf-8")
            parsed, mode = _safe_extract(raw)
            envelope_modes.append(mode)
            write_json(root / "parsed_response.json", parsed)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            transport_error = error
            (root / "raw_response.txt").write_text(raw, encoding="utf-8")
        mechanical = normalize_realizer_output(parsed, plan={"scenes": [current]}, parent=parent, round_index=1) if error is None else {"valid": False, "errors": [error]}
        if mechanical.get("valid") and mechanical.get("child", {}).get("scenes"):
            scene = _renumber(mechanical["child"]["scenes"][0], index, str(parent["root_id"]), str(current.get("memory_event_id")))
            prior.append(scene)
            write_json(root / "child_scene.json", scene)
        record = {"scene_index": index, "request_attempts": 1, "realizer_calls_completed": client.calls, "response_meta": meta, "transport_error": error, "mechanical_contract_valid": bool(mechanical.get("valid")), "envelope_mode": envelope_modes[-1] if envelope_modes else None}
        write_json(root / "mechanical_audit.json", mechanical)
        write_json(root / "metrics.json", record)
        records.append(record)
        if error or not mechanical.get("valid"):
            break
    child = {"schema_version": "memory_event_continuation_child_v1", "root_id": parent["root_id"], "round_index": 1, "scenes": prior}
    write_json(run_root / "child_dialogue.json", child)
    audit = _audit(child, case_id) if len(prior) == 3 else {"overall_pass": False, "checks": {"three_scenes": False}}
    write_json(run_root / "final_action_audit.json", audit)
    metrics = {"probe_version": PROBE_VERSION, "case_id": case_id, "root_id": parent["root_id"], "scene_records": records, "envelope_modes": envelope_modes, "scenes_completed": len(prior), "realizer_calls_completed": client.calls, "transport_error": transport_error, "final_action_audit_pass": bool(audit.get("overall_pass")), "conclusion": "SCENE_RELEASE_V3_PASS" if len(prior) == 3 and audit.get("overall_pass") else ("SCENE_RELEASE_V3_UNDECIDED_TIMEOUT" if transport_error and "Timeout" in transport_error else "SCENE_RELEASE_V3_FAIL"), "analysis_execution_api_calls": 0, "planner_calls": 0, "qa_writer_calls": 0, "solver_calls": 0, "writer_calls": 0, "auditor_calls": 0, "memorax_search_calls": 0, "memorax_add_calls": 0}
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {"probe_version": PROBE_VERSION, "isolated": True, "one_request_per_scene": True, "future_plans_in_any_scene_prompt": False, "retry_budget": 0, "api_role_allowlist": ["realizer"], "parser_mode_allowlist": ["direct_object", "nested_json_key"]})
    (run_root / "report.md").write_text("# Realizer Temporal Scene Release Probe v3\n\n" + f"Verdict: **{metrics['conclusion']}**\n\nScenes completed: {len(prior)}/3; Realizer calls: {client.calls}; envelope modes: {envelope_modes}; final-action audit: {audit.get('overall_pass')}\nTransport error: {transport_error or 'none'}\n\nThis probe accepts only a direct scenes object or an explicitly nested JSON-key envelope. No QA, Planner, Solver, Writer, Auditor, or Memorax calls were made.\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--case-id", default="POB03")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(run_root=run_root, config=args.config, case_id=args.case_id)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
