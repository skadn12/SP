from __future__ import annotations

import argparse
import hashlib
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
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.batch_probe_v1 import (
    BOUNDARY_PROFILES,
    _adapt_plan,
    _audit,
    _find_source,
    _generic_boundary_block,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "planner_realizer_temporal_boundary_integration_preflight_v1"
CANARY_CASE_ID = "POB04"
PRODUCTION_PROMPTS = Path(
    "scripts/current_method/restart_n5_spell/"
    "simple_append_memorax_v1_memory_event_continuation_probe_v1/prompts.py"
)


def build_integrated_temporal_prompt(
    *, parent: Mapping[str, Any], plan: Mapping[str, Any],
    case_id: str = CANARY_CASE_ID,
) -> dict[str, Any]:
    """Build an experiment-only wrapper around the unchanged production builder."""
    base = build_realizer_prompt(parent=parent, plan=plan, include_full_parent=True)
    boundary = _generic_boundary_block(plan)
    profile = BOUNDARY_PROFILES[case_id]
    profile_block = (
        "FINAL-SCENE-ONLY ACTIONS (hard): the following actions/states may occur only in Scene 3; "
        "do not perform or assert them in Scenes 1-2: "
        + "; ".join(profile["forbidden_before_final"])
    )
    prompt = (
        str(base["prompt"])
        + "\n\n"
        + boundary
        + "\n"
        + profile_block
        + "\nReturn only the same JSON object with exactly the scenes key."
    )
    return {
        **base,
        "prompt_version": f"{PROBE_VERSION}_wrapper",
        "base_prompt_version": base.get("prompt_version"),
        "prompt": prompt,
        "temporal_boundary_control": boundary,
        "profile_block": profile_block,
        "integration_wrapper": {
            "production_builder": "build_realizer_prompt",
            "production_prompt_file": str(PRODUCTION_PROMPTS),
            "production_prompt_modified": False,
            "append_only": True,
        },
    }


def preflight(*, run_root: Path, case_id: str = CANARY_CASE_ID) -> dict[str, Any]:
    prepare_empty(run_root)
    source, provenance = _find_source(case_id)
    parent = provenance["parent"]
    plan = _adapt_plan(source)
    prompt = build_integrated_temporal_prompt(parent=parent, plan=plan, case_id=case_id)
    prompt_text = str(prompt["prompt"])
    scenes = list(plan.get("scenes") or [])
    canonical_speakers = sorted({
        str(turn.get("speaker") or "")
        for turn in parent.get("dialogue", {}).get("turns", [])
        if str(turn.get("speaker") or "")
    })
    plan_speakers = sorted({
        str(value)
        for scene in scenes
        for value in scene.get("required_participants") or []
    })
    checks = {
        "source_is_accepted_o1": bool(
            source.get("contract", {}).get("valid")
            and source.get("analysis", {}).get("complex_chain")
        ),
        "production_builder_version_preserved": (
            prompt.get("base_prompt_version")
            == "memory_event_continuation_realizer_prompt_v6"
        ),
        "full_parent_included": (
            prompt.get("full_parent_included") is True
            and len(parent.get("dialogue", {}).get("turns", [])) > 0
        ),
        "three_scene_plan": len(scenes) == 3,
        "canonical_speaker_coverage": bool(canonical_speakers)
        and set(plan_speakers).issubset(set(canonical_speakers)),
        "exact_output_contract_retained": (
            "exactly the scenes key" in prompt_text
            and '"scenes": [' in prompt_text
        ),
        "boundary_block_appended_once": (
            prompt_text.count("TEMPORAL BOUNDARY CONTROL (strict)") == 1
        ),
        "profile_block_appended_once": (
            prompt_text.count("FINAL-SCENE-ONLY ACTIONS (hard)") == 1
        ),
        "production_prompt_hash_available": PRODUCTION_PROMPTS.exists(),
    }
    result = {
        "probe_version": PROBE_VERSION,
        "phase": "static_preflight",
        "case_id": case_id,
        "root_id": source["root_id"],
        "source": {
            **provenance,
            "source_case_id": source.get("case_id"),
            "source_objective": source.get("objective"),
        },
        "production_prompts_sha256": sha256_file(PRODUCTION_PROMPTS),
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "prompt_characters": len(prompt_text),
        "prompt_estimated_tokens": (len(prompt_text) + 3) // 4,
        "parent_turn_count": len(parent.get("dialogue", {}).get("turns", [])),
        "scene_count": len(scenes),
        "canonical_speakers": canonical_speakers,
        "plan_speakers": plan_speakers,
        "checks": checks,
        "preflight_pass": bool(all(checks.values())),
        "api_calls": 0,
    }
    write_json(run_root / "source_record.json", source)
    write_json(run_root / "adapted_plan.json", plan)
    write_json(run_root / "integrated_prompt.json", prompt)
    write_json(run_root / "preflight.json", result)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION,
        "isolated": True,
        "phase": "static_preflight",
        "source_runs_read_only": [
            provenance["source_records_path"],
            provenance["dialogue_path"],
        ],
        "production_prompt_modified": False,
        "api_calls": 0,
    })
    (run_root / "preflight_report.md").write_text(
        "# Realizer Temporal Boundary Integration Preflight v1\n\n"
        f"Verdict: **{'PASS' if result['preflight_pass'] else 'FAIL'}**\n\n"
        f"- Case: {case_id} / root {source['root_id']}\n"
        f"- Base builder prompt: {prompt['base_prompt_version']}\n"
        f"- Parent turns: {result['parent_turn_count']}; plan scenes: {result['scene_count']}\n"
        f"- Prompt characters: {result['prompt_characters']}; estimated tokens: {result['prompt_estimated_tokens']}\n"
        "- API calls: 0\n\n"
        "The production Realizer prompt file is read-only; this branch appends "
        "the boundary wrapper only in the isolated probe.\n",
        encoding="utf-8",
    )
    return result


def live_canary(
    *, run_root: Path, config: Path, preflight_result: Mapping[str, Any],
    case_id: str = CANARY_CASE_ID,
) -> dict[str, Any]:
    if not preflight_result.get("preflight_pass"):
        raise ValueError("static_preflight_failed")
    source, provenance = _find_source(case_id)
    parent = provenance["parent"]
    plan = _adapt_plan(source)
    prompt = build_integrated_temporal_prompt(parent=parent, plan=plan, case_id=case_id)
    write_json(run_root / "live_prompt.json", prompt)
    client = LiveJSONClient(config, max_tokens=8192, temperature=0.0)
    raw = ""
    meta: dict[str, Any] = {}
    parsed: Any = {}
    error = None
    try:
        raw, meta = client.complete(prompt["prompt"])
        (run_root / "live_raw_response.txt").write_text(raw, encoding="utf-8")
        parsed = extract_json(raw)
        write_json(run_root / "live_parsed_response.json", parsed)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        (run_root / "live_raw_response.txt").write_text(raw, encoding="utf-8")
    mechanical = (
        normalize_realizer_output(parsed, plan=plan, parent=parent, round_index=1)
        if error is None else {"valid": False, "errors": [error]}
    )
    semantic = (
        _audit(mechanical.get("child") or {}, case_id)
        if mechanical.get("valid") else {"overall_pass": False, "checks": {}}
    )
    if mechanical.get("child"):
        write_json(run_root / "live_child_dialogue.json", mechanical["child"])
    write_json(run_root / "live_mechanical_audit.json", mechanical)
    write_json(run_root / "live_semantic_audit.json", semantic)
    result = {
        "probe_version": PROBE_VERSION,
        "phase": "live_canary",
        "case_id": case_id,
        "root_id": source["root_id"],
        "request_attempts": 1,
        "realizer_calls_completed": client.calls,
        "response_meta": meta,
        "transport_error": error,
        "mechanical_contract_valid": bool(mechanical.get("valid")),
        "temporal_audit_pass": bool(semantic.get("overall_pass")),
        "conclusion": (
            "INTEGRATED_TEMPORAL_CANARY_PASS"
            if mechanical.get("valid") and semantic.get("overall_pass")
            else (
                "INTEGRATED_TEMPORAL_CANARY_UNDECIDED_TIMEOUT"
                if error and client.calls == 0 and "Timeout" in error
                else "INTEGRATED_TEMPORAL_CANARY_FAIL"
            )
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
    write_json(run_root / "live_metrics.json", result)
    write_json(run_root / "live_manifest.json", {
        "probe_version": PROBE_VERSION,
        "isolated": True,
        "phase": "live_canary",
        "api_role_allowlist": ["realizer"],
        "request_attempts": 1,
        "retry_budget": 0,
        "production_prompt_modified": False,
        "analysis_execution_api_calls": 0,
        "planner_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "auditor_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
    })
    (run_root / "live_report.md").write_text(
        "# Realizer Temporal Boundary Integration Live Canary v1\n\n"
        f"Verdict: **{result['conclusion']}**\n\n"
        f"- Case: {case_id} / root {source['root_id']}\n"
        f"- Request attempts: {result['request_attempts']}; completed: {result['realizer_calls_completed']}\n"
        f"- Mechanical: {result['mechanical_contract_valid']}; temporal audit: {result['temporal_audit_pass']}\n"
        f"- Transport error: {error or 'none'}\n\n"
        "This canary used the experiment-only wrapper around the unchanged "
        "production builder. No QA, Solver, Writer, Planner, Auditor, or "
        "Memorax calls were made.\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Realizer temporal-boundary integration preflight and canary"
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("/root/autodl-tmp/gxy-selfplay/config.json"),
    )
    parser.add_argument("--case-id", default=CANARY_CASE_ID)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (
        args.run_root
        or RUN_BASE / f"{PROBE_VERSION}_{stamp}"
    ).resolve()
    result = preflight(run_root=run_root, case_id=args.case_id)
    if args.live:
        live_canary(
            run_root=run_root,
            config=args.config,
            preflight_result=result,
            case_id=args.case_id,
        )
    print(json.dumps({"run_root": str(run_root), "preflight": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
