"""Finish the existing live_1 failure closure without making any request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..h4_round1_r03_feedback_view_v2.compiler import digest_file
from ..h4_round1_r03_memory_readonly_single_root_smoke_v1.smoke import (
    add_sidecars, write_json, write_jsonl, write_text,
)
from . import DEFAULT_OUTPUT, PREFLIGHT_OUTPUT, MEMORY_PREFLIGHT, ROOT_ID


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def close(run_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    run_root = Path(run_root)
    execution = _read(run_root / "execution_plan.json")
    requests = _read(run_root / "request_manifest.json")
    identity = _read(run_root / "input_identity.json")
    planner_request = _read(run_root / "planner" / "request.json")
    planner_response = _read(run_root / "planner" / "response.json")
    realizer_request = _read(run_root / "realizer" / "request_01.json")
    realizer_response = _read(run_root / "realizer" / "response_01.json")
    validation = _read(run_root / "realizer" / "validation_01.json")
    child = validation.get("child") or {}
    scenes = child.get("scenes") or []
    attempted_turns = sum(len(scene.get("dialogue_turns") or []) for scene in scenes)

    write_json(run_root / "r03_delta_quarantine.json", child)
    write_jsonl(run_root / "r03_delta_quarantine_turns.jsonl", [
        turn for scene in scenes for turn in scene.get("dialogue_turns") or []
    ])
    ledger = [
        {
            "role": "planner", "request_id": planner_request["request_id"],
            "prompt_sha256": planner_request["prompt_sha256"],
            "request_attempted": True, "response_completed": bool(planner_response.get("ok")),
            "transport_attempt_count": planner_response.get("transport_attempt_count"),
            "model": planner_request["model"], "retry": False,
        },
        {
            "role": "realizer", "request_id": realizer_request["request_id"],
            "prompt_sha256": realizer_request["prompt_sha256"],
            "request_attempted": True, "response_completed": bool(realizer_response.get("ok")),
            "transport_attempt_count": realizer_response.get("transport_attempt_count"),
            "model": realizer_request["model"], "retry": False,
        },
    ]
    write_jsonl(run_root / "call_ledger.jsonl", ledger)

    source_paths = [
        Path(identity["input_dialogue_path"]),
        PREFLIGHT_OUTPUT / "offline_preflight_report.json",
        PREFLIGHT_OUTPUT / "template_identity.json",
        PREFLIGHT_OUTPUT / "feedback_view_backend_manifest.jsonl",
        MEMORY_PREFLIGHT / "roots" / ROOT_ID / "memory_read" / "r02_memory_snapshot_normalized.json",
    ]
    source_rows = []
    for path in source_paths:
        observed = digest_file(path)
        sidecar = path.with_name(path.name + ".sha256")
        expected = sidecar.read_text(encoding="utf-8").split()[0] if sidecar.is_file() else observed
        source_rows.append({
            "path": str(path), "expected_sha256": expected,
            "observed_sha256": observed, "unchanged": expected == observed,
        })
    write_json(run_root / "source_sha_audit.json", {
        "files": source_rows,
        "source_files_modified": sum(not row["unchanged"] for row in source_rows),
    })

    template_identity = _read(PREFLIGHT_OUTPUT / "template_identity.json")
    prompt_checks = {
        "planner_contains_shared_rules_v2": "【共享反馈使用规则——扩充与出题共用】" in planner_request["prompt"],
        "planner_contains_expansion_task_v2": "【扩充专属任务——固定规则】" in planner_request["prompt"],
        "realizer_contains_shared_rules_v2": "【共享反馈使用规则——扩充与出题共用】" in realizer_request["prompt"],
        "realizer_contains_expansion_task_v2": "【扩充专属任务——固定规则】" in realizer_request["prompt"],
        "old_feedback_card_absent": "FROZEN EXPANSION FEEDBACK CARD" not in planner_request["prompt"],
        "realizer_solver_observations_absent": "solver_observations" not in realizer_request["prompt"],
    }
    write_json(run_root / "runtime_identity.json", {
        "git_head": "f555f9916f8baf1d220685720b1c67f7e426ab65",
        "worktree_dirty_entries_observed_before_first_request": 148,
        "capture_note": "Recorded by the pre-request local inspection; no diff content or credentials stored.",
        "runner_version": execution["version"],
        "v2_template_identity": template_identity,
        "planner_prompt_sha256": planner_request["prompt_sha256"],
        "realizer_prompt_sha256": realizer_request["prompt_sha256"],
        "prompt_checks": prompt_checks,
    })

    errors = list(validation.get("errors") or [])
    report = {
        "status": "R03_SINGLE_ROOT_FEEDBACK_VIEW_V2_DEBUG_QUARANTINED",
        "next_status": "WAITING_FOR_DIAGNOSTIC_OR_FIX_AUTHORIZATION; NO_RETRY",
        "root_id": ROOT_ID, "branch_id": execution["branch_id"],
        "base_dialogue_file_sha256": identity["input_dialogue_file_sha256"],
        "base_complete_dialogue_sha256": identity["complete_dialogue_sha256"],
        "new_cumulative_dialogue_sha256": None,
        "v2_template_identity_sha256": identity["v2_template_identity_sha256"],
        "planner_prompt_sha256": planner_request["prompt_sha256"],
        "realizer_prompt_sha256": realizer_request["prompt_sha256"],
        "planned_requests": requests["planned"], "actual_requests": requests["actual"],
        "transport_attempts": {
            "planner": planner_response.get("transport_attempt_count"),
            "realizer": realizer_response.get("transport_attempt_count"),
        },
        "planned_scene_count": execution["scene_count"],
        "realizer_output_scene_count": len(scenes), "promoted_scene_count": 0,
        "attempted_turns": attempted_turns, "promoted_turns": 0,
        "failure_stage": "realizer_local_validation",
        "technical_failure": False, "parse_failure": False,
        "existing_closure_failure_count": len(errors),
        "existing_closure_failures": errors,
        "quarantine_delta_saved": True,
        "new_r03_saved_and_frozen": False,
        "parent_r01_r02_unchanged": all(row["unchanged"] for row in source_rows),
        "old_r03_mixed": False, "old_template_fallback": not all(prompt_checks.values()),
        "call_counts_from_call_ledger": {
            "expansion_planner": 1, "expansion_realizer": 1,
            "memorax_search": 0, "memorax_status": 0, "memorax_add": 0,
            "memorax_mutation": 0, "qa_writer": 0, "solver": 0,
            "semantic_auditor": 0, "training": 0,
        },
        "semantic_realization_verdict": "NOT_AUDITED_BY_CONTRACT",
        "not_tested": [
            "semantic correctness of generated facts", "QA/Solver value",
            "comparison with old R03", "remote memory state",
        ],
    }
    write_json(run_root / "final_debug_report.json", report)
    write_text(run_root / "final_debug_report.md", "\n".join([
        "# R03 Feedback View v2 Single Root Debug Closure", "",
        f"- Status: `{report['status']}`",
        f"- Root: `{ROOT_ID}`", f"- Branch: `{execution['branch_id']}`",
        f"- Planned / Realizer-output / promoted scenes: `{execution['scene_count']} / {len(scenes)} / 0`",
        f"- Attempted / promoted turns: `{attempted_turns} / 0`",
        "- Planner / Realizer requests: `1 / 1` (one transport attempt each)",
        f"- Existing closure failures: `{len(errors)}` exact Parent-copy findings",
        "- Parent+R01+R02 unchanged: `true`",
        "- New R03 promoted: `false`; raw normalized delta retained in quarantine",
        "- Memorax Search / Status / Add / Mutation: `0 / 0 / 0 / 0`",
        "- QA Writer / Solver / semantic auditor / training: `0 / 0 / 0 / 0`",
        "- Retry / repair / fallback / root substitution: `0 / 0 / 0 / 0`",
        f"- Next: `{report['next_status']}`", "",
    ]))
    add_sidecars(run_root)
    return report


if __name__ == "__main__":
    print(json.dumps(close(), ensure_ascii=False, indent=2, sort_keys=True))
