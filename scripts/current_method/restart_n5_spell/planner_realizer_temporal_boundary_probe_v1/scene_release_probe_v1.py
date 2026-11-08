from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import LiveJSONClient, extract_json
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import normalize_realizer_output
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.batch_probe_v1 import _adapt_plan, _audit, _find_source

RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "planner_realizer_temporal_scene_release_probe_v1"


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _turn_text(scene: Mapping[str, Any]) -> str:
    return "\n".join(f"{t.get('speaker')} | {t.get('text')}" for t in scene.get("dialogue_turns") or [])


def build_single_scene_prompt(*, parent: Mapping[str, Any], prior: list[Mapping[str, Any]], current: Mapping[str, Any], index: int) -> dict[str, Any]:
    parent_text = "\n".join(f"{t.get('speaker')} | {t.get('text')}" for t in parent.get("dialogue", {}).get("turns") or [])
    prior_text = "\n\n".join(f"CHILD SCENE {i}\n{_turn_text(s)}" for i, s in enumerate(prior, 1)) or "No prior Child scene."
    prompt = (
        "You are the Natural Realizer for exactly one current Child scene. Future scene plans are withheld. "
        "Do not infer or introduce any later scene trigger, action, state, or resolution. The Parent and prior Child scenes are authoritative history. "
        "The current plan is the only plan you may realize. CURRENT STATE is already true. REQUIRED TRANSITION must happen now. REQUIRED END STATE must be true at scene end. "
        "Do not replace the transition with discussion, planning, or intention. Use exact Parent speaker names and never use user or assistant as a character. "
        "Return only JSON with exactly one scene and 9 to 48 dialogue turns.\n\nPARENT\n" + parent_text
        + "\n\nPRIOR CHILD\n" + prior_text
        + f"\n\nCURRENT SCENE PLAN {index}\n" + _dump(current)
        + "\n\nOutput shape: {\"scenes\": [{\"dialogue_turns\": [{\"speaker\": \"...\", \"text\": \"...\"}]}]}"
    )
    return {"prompt_version": PROBE_VERSION + "_prompt", "prompt": prompt, "scene_index": index, "future_plans_included": False, "prior_child_scene_count": len(prior)}


def _renumber(scene: Mapping[str, Any], index: int, root_id: str, event_id: str) -> dict[str, Any]:
    out = copy.deepcopy(dict(scene))
    ref = f"R01:S{index:02d}"
    out["scene_ref"] = ref
    out["session_ref"] = f"{root_id}:r01:s{index:02d}"
    out["memory_event_id"] = event_id
    out["dialogue_turns"] = [{"turn_ref": f"{ref}:T{j:03d}", "speaker": str(t.get("speaker") or ""), "text": str(t.get("text") or "")} for j, t in enumerate(out.get("dialogue_turns") or [], 1)]
    return out


def run(*, run_root: Path, config: Path, case_id: str = "POS07") -> dict[str, Any]:
    prepare_empty(run_root)
    source, provenance = _find_source(case_id)
    parent, plan = provenance["parent"], _adapt_plan(source)
    write_json(run_root / "source_record.json", source)
    write_json(run_root / "adapted_plan.json", plan)
    client = LiveJSONClient(config, max_tokens=8192, temperature=0.0)
    prior: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, current in enumerate(plan["scenes"], 1):
        root = run_root / f"scene_{index:02d}"
        root.mkdir(parents=True, exist_ok=True)
        prompt = build_single_scene_prompt(parent=parent, prior=prior, current=current, index=index)
        write_json(root / "prompt.json", prompt)
        raw, meta, parsed, error = "", {}, {}, None
        try:
            raw, meta = client.complete(prompt["prompt"])
            (root / "raw_response.txt").write_text(raw, encoding="utf-8")
            parsed = extract_json(raw)
            write_json(root / "parsed_response.json", parsed)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            (root / "raw_response.txt").write_text(raw, encoding="utf-8")
        mechanical = normalize_realizer_output(parsed, plan={"scenes": [current]}, parent=parent, round_index=1) if error is None else {"valid": False, "errors": [error]}
        if mechanical.get("valid") and mechanical.get("child", {}).get("scenes"):
            scene = _renumber(mechanical["child"]["scenes"][0], index, str(parent["root_id"]), str(current.get("memory_event_id")))
            prior.append(scene)
            write_json(root / "child_scene.json", scene)
        record = {"scene_index": index, "request_attempts": 1, "realizer_calls_completed": client.calls, "response_meta": meta, "transport_error": error, "mechanical_contract_valid": bool(mechanical.get("valid"))}
        write_json(root / "mechanical_audit.json", mechanical)
        write_json(root / "metrics.json", record)
        records.append(record)
        if error or not mechanical.get("valid"):
            break
    child = {"schema_version": "memory_event_continuation_child_v1", "root_id": parent["root_id"], "round_index": 1, "scenes": prior}
    write_json(run_root / "child_dialogue.json", child)
    audit = _audit(child, case_id) if len(prior) == 3 else {"overall_pass": False, "checks": {"three_scenes": False}}
    write_json(run_root / "final_audit.json", audit)
    metrics = {"probe_version": PROBE_VERSION, "case_id": case_id, "root_id": parent["root_id"], "scene_records": records, "scenes_completed": len(prior), "realizer_calls_completed": client.calls, "final_action_audit_pass": bool(audit.get("overall_pass")), "conclusion": "SCENE_RELEASE_PASS" if len(prior) == 3 and audit.get("overall_pass") else "SCENE_RELEASE_FAIL", "analysis_execution_api_calls": 0, "planner_calls": 0, "qa_writer_calls": 0, "solver_calls": 0, "writer_calls": 0, "auditor_calls": 0, "memorax_search_calls": 0, "memorax_add_calls": 0}
    if transport_error and client.calls == 0 and "Timeout" in transport_error:
        metrics["conclusion"] = "SCENE_RELEASE_UNDECIDED_TIMEOUT"
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {"probe_version": PROBE_VERSION, "isolated": True, "one_request_per_scene": True, "future_plans_in_any_scene_prompt": False, "retry_budget": 0, "api_role_allowlist": ["realizer"]})
    (run_root / "report.md").write_text("# Realizer Temporal Scene Release Probe v1\n\n" + f"Verdict: **{metrics['conclusion']}**\n\nScenes completed: {len(prior)}/3; Realizer calls: {client.calls}; final-action audit: {audit.get('overall_pass')}\n\nFuture plans were withheld from every scene request. No QA, Planner, Solver, Writer, Auditor, or Memorax calls were made.\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--case-id", default="POS07")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(run_root=run_root, config=args.config, case_id=args.case_id)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
