"""Run exactly one frozen Realizer request for each A/B condition.

The runner never renders a prompt, retries a request, reruns Planner, or creates
any non-Realizer client. Both frozen inputs are verified before the first call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..h4_round1_r03_feedback_view_v2_single_root_debug.run import (
    local_continuation_gate,
    planner_parent,
)
from ..h4_round1_r03_realizer_ab_single_generation_v1 import (
    CONFIG_PATH,
    DEFAULT_OUTPUT,
    PREPARATION_RUN,
    ROOT_ID,
    SOURCE_RUN,
    TIMEOUT_SECONDS,
    VERSION,
)
from ..live_backends import load_challenger_config
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    normalize_realizer_output,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.live_runtime import (
    load_json_object_strict,
)
from ..simple_append_v1.assembly import append_expansion_round, validate_appended_round


CONDITIONS = (
    ("A_FULL_PARENT", "a_full_parent"),
    ("B_TEXT_FREE", "b_text_free"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def add_sidecars(run_root: Path) -> None:
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name.endswith(".sha256"):
            continue
        if path.suffix not in {".json", ".jsonl", ".md", ".txt"}:
            continue
        path.with_name(path.name + ".sha256").write_text(
            f"{digest_file(path)}  {path.name}\n", encoding="utf-8")


def _source_round(turn_ref: str) -> str:
    for part in str(turn_ref).split(":"):
        if part.startswith("P") and part[1:].isdigit():
            return "Parent"
        if part in {"R01", "R02"}:
            return part
    return "UNKNOWN"


def _flatten_raw_scenes(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenes = value.get("scenes") if isinstance(value, Mapping) else None
    if not isinstance(scenes, list):
        return rows
    for scene_index, scene in enumerate(scenes, 1):
        if not isinstance(scene, Mapping):
            continue
        turns = scene.get("dialogue_turns")
        if not isinstance(turns, list):
            continue
        for turn_index, turn in enumerate(turns, 1):
            if not isinstance(turn, Mapping):
                continue
            rows.append({
                "scene_index": scene_index,
                "turn_index": turn_index,
                "output_turn_ref": f"R03:S{scene_index:02d}:T{turn_index:03d}",
                "speaker": str(turn.get("speaker") or ""),
                "text": str(turn.get("text") or ""),
            })
    return rows


def _exact_copy_analysis(raw_child: Mapping[str, Any],
                         dialogue: Mapping[str, Any]) -> dict[str, Any]:
    history_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dialogue.get("turns") or []:
        text = str(row.get("text") or "").strip()
        history_by_text[text].append({
            "turn_ref": str(row.get("turn_ref") or ""),
            "speaker": str(row.get("speaker") or ""),
            "text": str(row.get("text") or ""),
            "source_round": _source_round(str(row.get("turn_ref") or "")),
        })

    output_rows = _flatten_raw_scenes(raw_child)
    matches: list[dict[str, Any]] = []
    matched_positions: dict[int, set[int]] = defaultdict(set)
    for output in output_rows:
        for historical in history_by_text.get(output["text"].strip(), []):
            matched_positions[output["scene_index"]].add(output["turn_index"])
            matches.append({
                **output,
                "historical_turn_ref": historical["turn_ref"],
                "historical_speaker": historical["speaker"],
                "historical_text": historical["text"],
                "source_round": historical["source_round"],
                "speaker_equal": output["speaker"] == historical["speaker"],
                "comparison": "stripped exact text equality; speaker ignored",
            })

    segments: list[dict[str, Any]] = []
    for scene_index, positions in sorted(matched_positions.items()):
        current: list[int] = []
        for position in sorted(positions):
            if not current or position == current[-1] + 1:
                current.append(position)
            else:
                segments.append({"scene_index": scene_index,
                                 "output_turn_indices": current,
                                 "length": len(current)})
                current = [position]
        if current:
            segments.append({"scene_index": scene_index,
                             "output_turn_indices": current,
                             "length": len(current)})

    unique_output = {
        (row["scene_index"], row["turn_index"]) for row in matches
    }
    source_counts = {
        name: len({(row["scene_index"], row["turn_index"])
                   for row in matches if row["source_round"] == name})
        for name in ("Parent", "R01", "R02", "UNKNOWN")
    }
    return {
        "validator_comparator": "str(output.text).strip() in historical text set; speaker ignored",
        "output_turn_count": len(output_rows),
        "match_pair_count": len(matches),
        "exact_copy_output_turn_count": len(unique_output),
        "source_round_unique_output_turn_counts": source_counts,
        "longest_contiguous_output_copy_length": max(
            (row["length"] for row in segments), default=0),
        "segments": segments,
        "matches": matches,
    }


def _verify_frozen_inputs() -> dict[str, Any]:
    report_path = PREPARATION_RUN / "ab_input_preparation_report.json"
    source_identity_path = PREPARATION_RUN / "source_identity.json"
    report = read_json(report_path)
    identity = read_json(source_identity_path)
    if report.get("status") != "OFFLINE_AB_INPUT_PREPARATION_PASS":
        raise RuntimeError("frozen_preparation_not_passed")
    if identity.get("root_id") != ROOT_ID:
        raise RuntimeError("frozen_root_id_mismatch")

    dialogue_path = Path(identity["dialogue_path"])
    plan_path = Path(identity["plan_path"])
    validator_path = Path(identity["validator_path"])
    source_checks = {
        "dialogue_file_sha256": digest_file(dialogue_path)
        == identity["dialogue_sha256"],
        "plan_file_sha256": digest_file(plan_path) == identity["plan_sha256"],
        "validator_file_sha256": digest_file(validator_path)
        == identity["validator_sha256"],
        "source_run_unchanged": str(SOURCE_RUN) == identity["source_run"],
    }
    dialogue = read_json(dialogue_path)
    plan = read_json(plan_path)
    source_checks.update({
        "dialogue_root_id": dialogue.get("root_id") == ROOT_ID,
        "dialogue_turn_count": len(dialogue.get("turns") or [])
        == identity["dialogue_turn_count"],
        "dialogue_complete_sha256": dialogue.get("complete_dialogue_sha256")
        == identity["dialogue_complete_sha256"],
        "plan_scene_count": len(plan.get("scenes") or []) == 3,
    })

    modes: dict[str, Any] = {}
    messages_values: list[Any] = []
    for mode, directory in CONDITIONS:
        mode_root = PREPARATION_RUN / directory
        messages_path = mode_root / "messages.json"
        request_path = mode_root / "request.json"
        metadata_path = mode_root / "prompt_metadata.json"
        request = read_json(request_path)
        messages = read_json(messages_path)
        metadata = read_json(metadata_path)
        messages_values.append(messages)
        messages_file_sha = (mode_root / "messages.json.sha256").read_text(
            encoding="utf-8").split()[0]
        request_file_sha = (mode_root / "request.json.sha256").read_text(
            encoding="utf-8").split()[0]
        checks = {
            "mode": request.get("mode") == mode == metadata.get("mode"),
            "messages_file_sidecar": digest_file(messages_path) == messages_file_sha,
            "request_file_sidecar": digest_file(request_path) == request_file_sha,
            "request_messages_equal_frozen_messages": request.get("messages") == messages,
            "one_user_message": isinstance(messages, list) and len(messages) == 1
            and messages[0].get("role") == "user",
            "prompt_sha256": sha_bytes(messages[0]["content"].encode("utf-8"))
            == request.get("prompt_sha256") == metadata.get("prompt_sha256"),
            "model": request.get("model") == "Qwen/Qwen3-14B",
            "temperature": request.get("temperature") == 0.25,
            "max_tokens": request.get("max_tokens") == 8192,
            "response_format": request.get("response_format")
            == {"type": "json_object"},
            "endpoint": request.get("endpoint") == "https://api.siliconflow.cn/v1",
            "retry_disabled": request.get("automatic_retry_allowed") is False,
            "fallback_disabled": request.get("provider_fallback_allowed") is False,
            "planner_rerun_disabled": request.get("planner_rerun_allowed") is False,
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"frozen_{mode}_preflight_failed:" + ",".join(
                    key for key, passed in checks.items() if not passed))
        modes[mode] = {
            "directory": directory,
            "messages_path": str(messages_path),
            "messages_file_sha256": digest_file(messages_path),
            "request_path": str(request_path),
            "request_file_sha256": digest_file(request_path),
            "prompt_sha256": request["prompt_sha256"],
            "prompt_tokens_local": request["prompt_tokens_local"],
            "checks": checks,
            "request": request,
            "messages": messages,
        }
    if modes["A_FULL_PARENT"]["request"]["model"] != modes["B_TEXT_FREE"]["request"]["model"]:
        raise RuntimeError("ab_model_mismatch")
    for field in ("temperature", "max_tokens", "response_format", "endpoint", "role"):
        if (modes["A_FULL_PARENT"]["request"].get(field)
                != modes["B_TEXT_FREE"]["request"].get(field)):
            raise RuntimeError(f"ab_request_parameter_mismatch:{field}")
    if report.get("same_frozen_inputs", {}).get("same_plan") is not True:
        raise RuntimeError("preparation_report_same_plan_not_true")

    protected = [
        report_path,
        source_identity_path,
        dialogue_path,
        plan_path,
        validator_path,
    ]
    for _, directory in CONDITIONS:
        protected.extend([
            PREPARATION_RUN / directory / "messages.json",
            PREPARATION_RUN / directory / "request.json",
            PREPARATION_RUN / directory / "prompt_metadata.json",
        ])
    return {
        "report": report,
        "identity": identity,
        "dialogue": dialogue,
        "plan": plan,
        "modes": modes,
        "protected_paths": protected,
        "protected_before": {str(path): digest_file(path) for path in protected},
        "source_checks": source_checks,
    }


def _request_once(*, endpoint: str, api_key: str, payload: Mapping[str, Any],
                  request_id: str) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        },
        method="POST",
    )
    started_at = now()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body_bytes = response.read()
            http_status = int(response.status)
        completed_at = now()
        body = json.loads(body_bytes.decode("utf-8"))
        choice = body["choices"][0]
        message = choice["message"]
        return {
            "ok": True,
            "request_id": request_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "http_status": http_status,
            "transport_attempt_count": 1,
            "provider_response": body,
            "raw_response": str(message.get("content") or ""),
            "model_identifier": body.get("model") or payload["model"],
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage") if isinstance(body.get("usage"), Mapping) else None,
        }
    except urllib.error.HTTPError as error:
        return {
            "ok": False,
            "request_id": request_id,
            "started_at": started_at,
            "completed_at": now(),
            "error_type": "HTTPError",
            "http_status": int(error.code),
            "error_body": error.read().decode("utf-8", errors="replace"),
            "transport_attempt_count": 1,
            "response_status": "KNOWN_HTTP_FAILURE",
        }
    except Exception as error:
        return {
            "ok": False,
            "request_id": request_id,
            "started_at": started_at,
            "completed_at": now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "transport_attempt_count": 1,
            "response_status": "UNKNOWN_OR_TRANSPORT_FAILURE_NO_RESEND",
        }


def _run_condition(*, run_root: Path, mode: str, frozen: Mapping[str, Any],
                   dialogue: Mapping[str, Any], plan: Mapping[str, Any],
                   api_key: str) -> dict[str, Any]:
    condition_root = run_root / frozen["directory"]
    request = frozen["request"]
    messages = frozen["messages"]
    request_id = f"r03-realizer-ab-122711-live-1-{mode.lower()}-01"
    payload: dict[str, Any] = {
        "model": request["model"],
        "messages": messages,
        "temperature": request["temperature"],
        "max_tokens": request["max_tokens"],
        "response_format": request["response_format"],
    }
    actual_request = {
        "request_id": request_id,
        "role": request["role"],
        "mode": mode,
        "endpoint": request["endpoint"],
        "timeout_seconds": TIMEOUT_SECONDS,
        "payload": payload,
        "api_key_recorded": False,
        "automatic_retry_allowed": False,
        "repair_allowed": False,
        "provider_fallback_allowed": False,
        "planner_rerun_allowed": False,
        "frozen_messages_path": frozen["messages_path"],
        "frozen_messages_file_sha256": frozen["messages_file_sha256"],
        "actual_messages_sha256": sha_bytes(canonical(messages).encode("utf-8")),
        "prompt_sha256": request["prompt_sha256"],
    }
    write_json(condition_root / "actual_messages.json", messages)
    write_json(condition_root / "actual_request.json", actual_request)
    write_json(condition_root / "call_ledger.json", {
        "request_id": request_id,
        "mode": mode,
        "request_attempted": False,
        "response_completed": False,
        "transport_attempt_count": 0,
        "retry": False,
        "prompt_sha256": request["prompt_sha256"],
    })

    response = _request_once(
        endpoint=request["endpoint"], api_key=api_key,
        payload=payload, request_id=request_id)
    provider_response = response.pop("provider_response", None)
    raw_response = response.pop("raw_response", None)
    if provider_response is not None:
        write_json(condition_root / "provider_response.json", provider_response)
    if raw_response is not None:
        write_text(condition_root / "raw_response.txt", raw_response)
    write_json(condition_root / "response_metadata.json", {
        **response,
        "raw_response_sha256": (
            sha_bytes(raw_response.encode("utf-8"))
            if raw_response is not None else None),
    })
    write_json(condition_root / "call_ledger.json", {
        "request_id": request_id,
        "mode": mode,
        "request_attempted": True,
        "response_completed": bool(response.get("ok")),
        "transport_attempt_count": 1,
        "retry": False,
        "prompt_sha256": request["prompt_sha256"],
        "response_status": response.get("response_status") or (
            "COMPLETED" if response.get("ok") else "FAILED"),
    })

    result: dict[str, Any] = {
        "mode": mode,
        "request_id": request_id,
        "request_completed": bool(response.get("ok")),
        "technical_failure": None,
        "parse_failure": None,
        "scene_count": 0,
        "attempted_turns": 0,
        "promoted_turns": 0,
        "exact_copy_output_turn_count": 0,
        "copy_sources": {},
        "longest_contiguous_output_copy_length": 0,
        "closure_passed": False,
        "closure_stage": "transport",
    }
    if not response.get("ok"):
        result["technical_failure"] = {
            key: response.get(key) for key in (
                "error_type", "http_status", "error", "error_body",
                "response_status") if response.get(key) is not None
        }
        write_json(condition_root / "condition_result.json", result)
        return result

    try:
        raw_child = load_json_object_strict(raw_response)
    except Exception as error:
        result["parse_failure"] = f"{type(error).__name__}:{error}"
        result["closure_stage"] = "parse"
        write_json(condition_root / "condition_result.json", result)
        return result

    write_json(condition_root / "parsed_response.json", raw_child)
    raw_rows = _flatten_raw_scenes(raw_child)
    result["scene_count"] = len(raw_child.get("scenes") or []) if isinstance(
        raw_child.get("scenes"), list) else 0
    result["attempted_turns"] = len(raw_rows)
    copy_analysis = _exact_copy_analysis(raw_child, dialogue)
    write_json(condition_root / "exact_copy_analysis.json", copy_analysis)
    write_jsonl(condition_root / "exact_copy_matches.jsonl", copy_analysis["matches"])
    result["exact_copy_output_turn_count"] = copy_analysis[
        "exact_copy_output_turn_count"]
    result["copy_sources"] = copy_analysis[
        "source_round_unique_output_turn_counts"]
    result["longest_contiguous_output_copy_length"] = copy_analysis[
        "longest_contiguous_output_copy_length"]

    parent = planner_parent(dialogue)
    validation = normalize_realizer_output(
        raw_child, plan=plan, parent=parent, round_index=3)
    write_json(condition_root / "realizer_validation.json", validation)
    if not validation.get("valid"):
        write_json(condition_root / "r03_delta_quarantine.json",
                   validation.get("child") or raw_child)
        result["closure_stage"] = "realizer_validation"
        result["validation_errors"] = validation.get("errors") or []
        write_json(condition_root / "condition_result.json", result)
        return result

    child = validation["child"]
    cumulative = append_expansion_round(prior_expanded=dialogue, child=child)
    cumulative_validation = validate_appended_round(
        cumulative, prior_expanded=dialogue)
    continuation_gate = local_continuation_gate(
        prior=dialogue, child=child, cumulative=cumulative,
        cumulative_validation=cumulative_validation,
        realizer_validation=validation, plan=plan)
    write_json(condition_root / "cumulative_validation.json", cumulative_validation)
    write_json(condition_root / "local_continuation_gate_report.json",
               continuation_gate)
    if not continuation_gate.get("valid"):
        write_json(condition_root / "r03_delta_quarantine.json", child)
        write_json(condition_root / "candidate_cumulative_quarantine.json", cumulative)
        result["closure_stage"] = "local_continuation_gate"
        result["validation_errors"] = continuation_gate.get("failure_codes") or []
        write_json(condition_root / "condition_result.json", result)
        return result

    r03_turns = cumulative["turns"][len(dialogue.get("turns") or []):]
    write_json(condition_root / "r03_delta.json", child)
    write_jsonl(condition_root / "r03_delta_turns.jsonl", r03_turns)
    write_json(condition_root / "parent_r01_r02_r03_cumulative_dialogue.json",
               cumulative)
    write_json(condition_root / "lineage_manifest.json", {
        "experiment": VERSION,
        "condition": mode,
        "root_id": ROOT_ID,
        "source_prefix": "Parent+R01+R02-feedback",
        "source_dialogue_file_sha256": digest_file(
            Path(read_json(PREPARATION_RUN / "source_identity.json")["dialogue_path"])),
        "source_complete_dialogue_sha256": dialogue.get("complete_dialogue_sha256"),
        "planner_plan_sha256": digest_file(
            Path(read_json(PREPARATION_RUN / "source_identity.json")["plan_path"])),
        "frozen_messages_file_sha256": frozen["messages_file_sha256"],
        "prompt_sha256": request["prompt_sha256"],
        "r03_turn_count": len(r03_turns),
        "cumulative_dialogue_sha256": cumulative.get("complete_dialogue_sha256"),
        "append_only": True,
        "remote_memory_writes": 0,
    })
    result["closure_passed"] = True
    result["closure_stage"] = "passed"
    result["promoted_turns"] = len(r03_turns)
    write_json(condition_root / "condition_result.json", result)
    return result


def execute(run_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    run_root = Path(run_root)
    if run_root.exists():
        raise RuntimeError(f"ab_live_output_root_exists:{run_root}")

    frozen = _verify_frozen_inputs()
    if not all(frozen["source_checks"].values()):
        raise RuntimeError(
            "source_identity_preflight_failed:" + ",".join(
                key for key, passed in frozen["source_checks"].items()
                if not passed))
    config = load_challenger_config(CONFIG_PATH)
    expected_request = frozen["modes"]["A_FULL_PARENT"]["request"]
    config_checks = {
        "endpoint": config.base_url == expected_request["endpoint"],
        "configured_model": config.model == expected_request["model"],
        "api_key_present": bool(config.api_key),
        "enable_thinking_not_injected": config.enable_thinking is None,
    }
    if not all(config_checks.values()):
        raise RuntimeError(
            "live_config_preflight_failed:" + ",".join(
                key for key, passed in config_checks.items() if not passed))

    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "execution_identity.json", {
        "version": VERSION,
        "created_at": now(),
        "root_id": ROOT_ID,
        "preparation_run": str(PREPARATION_RUN),
        "source_run": str(SOURCE_RUN),
        "planner_rerun": False,
        "maximum_realizer_requests": 2,
        "conditions": [mode for mode, _ in CONDITIONS],
        "config_checks": config_checks,
        "source_checks": frozen["source_checks"],
        "same_frozen_plan": True,
        "same_frozen_dialogue_snapshot": True,
        "same_validator": True,
        "retry": False,
        "repair": False,
        "fallback": False,
    })
    write_json(run_root / "frozen_input_identity.json", frozen["identity"])
    write_json(run_root / "call_ledger.json", {
        "planned": {"realizer": 2, "planner": 0},
        "actual": {"realizer": 0, "planner": 0},
        "request_ids": [],
        "other_external_calls": 0,
    })

    results: list[dict[str, Any]] = []
    request_ids: list[str] = []
    for mode, _ in CONDITIONS:
        result = _run_condition(
            run_root=run_root, mode=mode, frozen=frozen["modes"][mode],
            dialogue=frozen["dialogue"], plan=frozen["plan"],
            api_key=config.api_key)
        results.append(result)
        request_ids.append(result["request_id"])
        write_json(run_root / "call_ledger.json", {
            "planned": {"realizer": 2, "planner": 0},
            "actual": {"realizer": len(results), "planner": 0},
            "request_ids": request_ids,
            "other_external_calls": 0,
            "retry": 0,
            "repair": 0,
            "fallback": 0,
        })

    protected_rows = []
    for path in frozen["protected_paths"]:
        after = digest_file(path)
        before = frozen["protected_before"][str(path)]
        protected_rows.append({
            "path": str(path), "before_sha256": before,
            "after_sha256": after, "unchanged": before == after,
        })
    source_audit = {
        "all_unchanged": all(row["unchanged"] for row in protected_rows),
        "files": protected_rows,
    }
    write_json(run_root / "source_sha_audit.json", source_audit)

    report = {
        "status": "R03_REALIZER_AB_SINGLE_GENERATION_COMPLETE",
        "next_status": "AB_REAL_RESULTS_SAVED_WAITING_FOR_COMPARATIVE_ANALYSIS_AND_DECISION",
        "root_id": ROOT_ID,
        "results": results,
        "calls": {
            "realizer": 2,
            "planner": 0,
            "qa_writer": 0,
            "solver": 0,
            "semantic_auditor": 0,
            "training": 0,
            "memorax_search": 0,
            "memorax_status": 0,
            "memorax_add": 0,
            "memorax_mutation": 0,
            "retry": 0,
            "repair": 0,
            "fallback": 0,
        },
        "source_files_unchanged": source_audit["all_unchanged"],
        "semantic_content_analysis": "PENDING_OFFLINE_COMPARISON",
        "default_route_selected": False,
    }
    write_json(run_root / "generation_report.json", report)
    write_text(run_root / "generation_report.md", "\n".join([
        "# R03 Realizer A/B Single Generation", "",
        f"- Status: `{report['status']}`",
        f"- Root: `{ROOT_ID}`",
        "- Planner / Realizer calls: `0 / 2`",
        "- Retry / repair / fallback: `0 / 0 / 0`",
        "- Memorax / QA / Solver / auditor / training calls: `0`", "",
        "| Condition | Completed | Scenes | Attempted | Exact copy | Longest run | Closure | Promoted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['mode']} | {row['request_completed']} | {row['scene_count']} | "
            f"{row['attempted_turns']} | {row['exact_copy_output_turn_count']} | "
            f"{row['longest_contiguous_output_copy_length']} | {row['closure_passed']} | "
            f"{row['promoted_turns']} |"
            for row in results
        ], "",
    ]))
    add_sidecars(run_root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = execute(args.run_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

