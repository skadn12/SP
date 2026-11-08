from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    sha256_file,
    write_json,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import (
    LiveJSONClient,
    extract_json,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    normalize_realizer_output,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import (
    build_realizer_prompt,
)
from scripts.current_method.restart_n5_spell.planner_realizer_adherence_canary_v1.canary import (
    _load_source,
    adapt_plan,
    semantic_audit,
)
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.probe import (
    SOURCE_A_RUN,
    _load_a,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "planner_realizer_temporal_boundary_probe_v2"


def _compact_boundary_block(plan: Mapping[str, Any]) -> str:
    scenes = list(plan.get("scenes") or [])
    return "\n".join(
        [
            "TEMPORAL BOUNDARY CONTROL (strict)",
            "Realize only the current scene's transition and immediate consequence.",
            "Keep every non-final scene open/provisional; never call it final, complete, or ready.",
            "Scene 1: establish the ingredient decision only; do not apply, brush, coat, or cook.",
            "Scene 2: first observe that the glaze is too thick, then add water and thin/revise it; do not apply, brush, coat, baste, use the oven, or set a timer.",
            "Scene 3: only here may the glaze be applied and the ham cooked; only here may the timer/final result appear.",
            "A future action may be mentioned only as an explicitly future plan, never as already done.",
            f"This plan has {len(scenes)} scenes; preserve the plan's scene order exactly.",
        ]
    )


def build_compact_temporal_prompt(*, parent: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    base = build_realizer_prompt(parent=parent, plan=plan, include_full_parent=True)
    boundary = _compact_boundary_block(plan)
    return {
        **base,
        "prompt_version": "planner_realizer_temporal_boundary_prompt_v2_compact",
        "prompt": str(base["prompt"]) + "\n\n" + boundary + "\nReturn only the same JSON object with exactly the `scenes` key.",
        "temporal_boundary_control": boundary,
    }


def run(*, run_root: Path, config: Path, source_a_run: Path = SOURCE_A_RUN) -> dict[str, Any]:
    prepare_empty(run_root)
    source, parent, provenance = _load_source()
    plan = adapt_plan(source)
    condition_a, child_a = _load_a(source_a_run)
    prompt_b = build_compact_temporal_prompt(parent=parent, plan=plan)
    write_json(run_root / "source_record.json", source)
    write_json(run_root / "adapted_plan.json", plan)
    write_json(run_root / "condition_a.json", condition_a)
    write_json(run_root / "condition_a_child.json", child_a)
    write_json(run_root / "condition_b_prompt.json", prompt_b)

    client = LiveJSONClient(config, max_tokens=8192, temperature=0.0)
    request_attempts = 1
    raw = ""
    meta: dict[str, Any] = {}
    parsed: Any = {}
    error = None
    try:
        raw, meta = client.complete(prompt_b["prompt"])
        (run_root / "condition_b_raw_response.txt").write_text(raw, encoding="utf-8")
        parsed = extract_json(raw)
        write_json(run_root / "condition_b_parsed_response.json", parsed)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        (run_root / "condition_b_raw_response.txt").write_text(raw, encoding="utf-8")

    mechanical_b = normalize_realizer_output(parsed, plan=plan, parent=parent, round_index=1) if error is None else {"valid": False, "errors": [error]}
    semantic_b = semantic_audit(mechanical_b.get("child") or {}) if mechanical_b.get("valid") else {"overall_pass": False, "scene_audits": [], "stage_boundary_checks": {}}
    if mechanical_b.get("child"):
        write_json(run_root / "condition_b_child.json", mechanical_b["child"])
    write_json(run_root / "condition_b_mechanical_audit.json", mechanical_b)
    write_json(run_root / "condition_b_semantic_audit.json", semantic_b)

    a_pass = bool(condition_a["mechanical_contract_valid"] and condition_a["semantic_audit"].get("overall_pass"))
    b_pass = bool(mechanical_b.get("valid") and semantic_b.get("overall_pass"))
    completed = client.calls
    if error and completed == 0:
        conclusion = "TEMPORAL_BOUNDARY_CONTROL_UNDECIDED_TIMEOUT" if "Timeout" in error else "TEMPORAL_BOUNDARY_CONTROL_UNDECIDED_TRANSPORT"
    elif b_pass and not a_pass:
        conclusion = "TEMPORAL_BOUNDARY_CONTROL_RECOVERED"
    elif not b_pass:
        conclusion = "TEMPORAL_BOUNDARY_CONTROL_NOT_RECOVERED"
    else:
        conclusion = "BASELINE_ALREADY_PASS"
    metrics = {
        "probe_version": PROBE_VERSION,
        "source": provenance,
        "source_condition_a_run": str(source_a_run.resolve()),
        "source_condition_a_child_sha256": condition_a["child_sha256"],
        "condition_a": {"mechanical_contract_valid": condition_a["mechanical_contract_valid"], "temporal_audit_pass": a_pass, "realizer_calls": condition_a["realizer_calls"]},
        "condition_b": {"request_attempts": request_attempts, "mechanical_contract_valid": bool(mechanical_b.get("valid")), "temporal_audit_pass": b_pass, "realizer_calls_completed": completed, "response_meta": meta, "transport_error": error},
        "comparison": {"temporal_boundary_recovered": b_pass and not a_pass, "A_pass": a_pass, "B_pass": b_pass},
        "conclusion": conclusion,
        "analysis_execution_api_calls": 0,
        "planner_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "auditor_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
    }
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION, "isolated": True,
        "source_runs_read_only": [str(source_a_run.resolve()), provenance["source_records_path"]],
        "api_role_allowlist": ["realizer"], "condition_a_realizer_calls": condition_a["realizer_calls"],
        "condition_b_request_attempts": request_attempts, "condition_b_realizer_calls_completed": completed, "condition_b_retry_budget": 0,
        "analysis_execution_api_calls": 0, "planner_calls": 0, "qa_writer_calls": 0,
        "solver_calls": 0, "writer_calls": 0, "auditor_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
    })
    (run_root / "report.md").write_text(
        "# Planner Realizer Temporal Boundary Probe v2 (compact)\n\n"
        f"Verdict: **{conclusion}**\n\n"
        f"- Condition A original prompt: pass={a_pass}\n"
        f"- Condition B compact boundary prompt: pass={b_pass}\n"
        f"- B request attempts: `{request_attempts}`; completed responses: `{completed}`\n"
        f"- Transport error: `{error or 'none'}`\n\n"
        "Condition A is the frozen original canary. Condition B is one fresh Realizer request with no repair or retry.\n\n"
        "No Planner, QA, Solver, Writer, Auditor, or Memorax calls were made.\n",
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="compact A/B Realizer temporal-boundary probe")
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--source-a-run", type=Path, default=SOURCE_A_RUN)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_realizer_temporal_boundary_probe_v2_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(run_root=run_root, config=args.config, source_a_run=args.source_a_run)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
