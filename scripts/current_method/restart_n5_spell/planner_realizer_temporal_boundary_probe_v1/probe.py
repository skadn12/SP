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


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
SOURCE_A_RUN = RUN_BASE / "planner_realizer_adherence_canary_v1_20260822T160000Z"
PROBE_VERSION = "planner_realizer_temporal_boundary_probe_v1"


def _future_boundary_block(plan: Mapping[str, Any]) -> str:
    scenes = list(plan.get("scenes") or [])
    lines = [
        "TEMPORAL BOUNDARY CONTROL (hard contract)",
        "Each scene may realize only its own required transition and immediate consequence.",
        "Do not perform, announce, or imply a future scene's action or final state early.",
    ]
    for index, scene in enumerate(scenes, 1):
        status = str(scene.get("after_state") or "")
        lines.append(f"SCENE {index} LOCAL END STATE: {status}")
        if index < len(scenes):
            future = scenes[index:]
            future_actions: list[str] = []
            for future_scene in future:
                future_actions.extend(str(value) for value in future_scene.get("interaction_beats") or [])
                future_actions.append(str(future_scene.get("required_transition") or ""))
                future_actions.append(str(future_scene.get("after_state") or ""))
            lines.append(
                f"SCENE {index} FORBIDDEN FUTURE CONTENT: {json.dumps(future_actions, ensure_ascii=False)}")
        if "resolution=final" not in status.casefold() and index < len(scenes):
            lines.append(
                f"SCENE {index} MUST REMAIN OPEN/PROVISIONAL: do not say final, complete, ready to proceed, or equivalent.")
    lines.extend([
        "For this plan specifically: do not apply/brush/coat/baste the glaze, place the ham in the oven, or set a timer before Scene 3.",
        "Scene 1 must not declare the recipe final; Scene 2 must first contain the too-thick observation, then water/thinning, and remain provisional.",
        "If a future action is mentioned as a plan, keep it explicitly future and do not perform it in the current scene.",
    ])
    return "\n".join(lines)


def build_temporal_prompt(*, parent: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    base = build_realizer_prompt(parent=parent, plan=plan, include_full_parent=True)
    prompt = str(base["prompt"]) + "\n\n" + _future_boundary_block(plan)
    prompt += "\nReturn only the same JSON object with exactly the `scenes` key."
    return {
        **base,
        "prompt_version": "planner_realizer_temporal_boundary_prompt_v1",
        "prompt": prompt,
        "temporal_boundary_control": _future_boundary_block(plan),
    }


def _load_a(source_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    child_path = source_run / "child_dialogue.json"
    metrics_path = source_run / "metrics.json"
    audit_path = source_run / "semantic_adherence_audit.json"
    if not child_path.exists() or not metrics_path.exists() or not audit_path.exists():
        raise FileNotFoundError("condition_a_artifacts_missing")
    child = json.loads(child_path.read_text())
    strict_audit = semantic_audit(child)
    return {
        "condition": "A_original_prompt",
        "source_run": str(source_run.resolve()),
        "child_sha256": sha256_file(child_path),
        "metrics_sha256": sha256_file(metrics_path),
        "mechanical_contract_valid": bool(json.loads((source_run / "mechanical_audit.json").read_text()).get("valid")),
        "original_audit_sha256": sha256_file(audit_path),
        "semantic_audit": strict_audit,
        "realizer_calls": json.loads(metrics_path.read_text()).get("realizer_calls"),
    }, child


def run(*, run_root: Path, config: Path, source_a_run: Path = SOURCE_A_RUN) -> dict[str, Any]:
    prepare_empty(run_root)
    source, parent, provenance = _load_source()
    plan = adapt_plan(source)
    condition_a, child_a = _load_a(source_a_run)
    prompt_b = build_temporal_prompt(parent=parent, plan=plan)
    write_json(run_root / "source_record.json", source)
    write_json(run_root / "adapted_plan.json", plan)
    write_json(run_root / "condition_a.json", condition_a)
    write_json(run_root / "condition_a_child.json", child_a)
    write_json(run_root / "condition_b_prompt.json", prompt_b)
    client = LiveJSONClient(config, max_tokens=8192, temperature=0.0)
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
    metrics = {
        "probe_version": PROBE_VERSION,
        "source": provenance,
        "source_condition_a_run": str(source_a_run.resolve()),
        "source_condition_a_child_sha256": condition_a["child_sha256"],
        "condition_a": {"mechanical_contract_valid": condition_a["mechanical_contract_valid"], "temporal_audit_pass": a_pass, "realizer_calls": condition_a["realizer_calls"]},
        "condition_b": {"mechanical_contract_valid": bool(mechanical_b.get("valid")), "temporal_audit_pass": b_pass, "realizer_calls": client.calls, "response_meta": meta, "transport_error": error},
        "comparison": {"temporal_boundary_recovered": b_pass and not a_pass, "A_pass": a_pass, "B_pass": b_pass},
        "conclusion": (
            "TEMPORAL_BOUNDARY_CONTROL_RECOVERED" if b_pass and not a_pass else
            ("TEMPORAL_BOUNDARY_CONTROL_UNDECIDED_TIMEOUT" if error else
             ("TEMPORAL_BOUNDARY_CONTROL_NOT_RECOVERED" if not b_pass else "BASELINE_ALREADY_PASS"))
        ),
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
        "condition_b_realizer_calls": client.calls, "condition_b_retry_budget": 0,
        "analysis_execution_api_calls": 0, "planner_calls": 0, "qa_writer_calls": 0,
        "solver_calls": 0, "writer_calls": 0, "auditor_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
    })
    lines = [
        "# Planner Realizer Temporal Boundary Probe v1", "",
        f"Verdict: **{metrics['conclusion']}**", "",
        f"- Condition A original prompt: pass={a_pass}",
        f"- Condition B temporal-boundary prompt: pass={b_pass}",
        f"- B Realizer calls: `{client.calls}`", "",
        "Condition A is the previously frozen original canary; it is not rerun.",
        "Condition B is one fresh Realizer request with no repair or retry.", "",
        "No Planner, QA, Solver, Writer, Auditor, or Memorax calls were made in this probe.", "",
        "The strict temporal audit rejects premature finality, application, basting, or timer setup before Scene 3.", "",
    ]
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B Realizer temporal-boundary probe")
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--source-a-run", type=Path, default=SOURCE_A_RUN)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_realizer_temporal_boundary_probe_v1_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(
        run_root=run_root, config=args.config, source_a_run=args.source_a_run)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
