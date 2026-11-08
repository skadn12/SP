"""One-shot live LLM probe for the frozen Realizer Adherence Auditor."""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, sha256_bytes, sha256_file
from ..backends import BackendResponse
from ..ledger import RequestLedger
from ..live_backends import ChallengerAPIConfig, load_challenger_config
from .realizer_adherence import (
    build_adherence_auditor_input,
    build_adherence_auditor_prompt,
    load_json,
    validate_and_materialize_adherence_audit,
)


RUNNER_VERSION = "realizer_adherence_live_probe_v1"
DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_34406_memory_event_continuation_"
    "contract_v2_live_1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_34406_realizer_adherence_live_probe_1"
)
DEFAULT_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_REFERENCE = Path(__file__).resolve().parent / "fixtures" / (
    "episode_34406_realizer_adherence_semantic_judgment.json"
)
CONTEXT_LIMIT = 40_960
AUDITOR_MAX_TOKENS = 4_096


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _directory_manifest(root: Path) -> str:
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        rows.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    body = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _module_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    rows = [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for path in sorted(root.glob("*.py")) if path.is_file()]
    body = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return {"files": rows, "manifest_sha256": hashlib.sha256(body).hexdigest()}


def _token_counter(tokenizer_path: Path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return lambda value: len(tokenizer.encode(value, add_special_tokens=False))


def _load_inputs(source_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = source_run / "planner" / "materialized_plan.json"
    child_path = source_run / "child_dialogue.json"
    if not plan_path.is_file() or not child_path.is_file():
        raise FileNotFoundError("frozen_adherence_inputs_missing")
    plan = load_json(plan_path)
    child = load_json(child_path)
    if not isinstance(plan, dict) or not isinstance(child, dict):
        raise ValueError("frozen_adherence_inputs_must_be_objects")
    return plan, child


def preflight(*, source_run: Path, output_root: Path, config_path: Path,
              tokenizer_path: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "source_run_exists": source_run.is_dir(),
        "output_root_absent": not output_root.exists(),
        "plan_exists": (source_run / "planner" / "materialized_plan.json").is_file(),
        "child_exists": (source_run / "child_dialogue.json").is_file(),
        "config_exists": config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "api_calls_zero": True,
        "planner_calls_zero": True,
        "gate_calls_zero": True,
        "realizer_calls_zero": True,
        "memorax_calls_zero": True,
        "solver_calls_zero": True,
        "retry_budget_zero": True,
    }
    details: dict[str, Any] = {}
    plan: dict[str, Any] = {}
    child: dict[str, Any] = {}
    auditor_input: dict[str, Any] | None = None
    prompt_data: dict[str, Any] | None = None
    try:
        plan, child = _load_inputs(source_run)
        auditor_input = build_adherence_auditor_input(plan=plan, child=child)
        prompt_data = build_adherence_auditor_prompt(auditor_input=auditor_input)
        checks["input_contract_valid"] = True
        details["scene_count"] = len(auditor_input.get("scenes", []))
        details["source_manifest_sha256"] = _directory_manifest(source_run)
        details["plan_sha256"] = sha256_file(
            source_run / "planner" / "materialized_plan.json")
        details["child_sha256"] = sha256_file(source_run / "child_dialogue.json")
    except Exception as error:
        checks["input_contract_valid"] = False
        details["input_error"] = f"{type(error).__name__}:{error}"
    try:
        config = load_challenger_config(config_path)
        checks["model_configured"] = bool(config.model)
        checks["base_url_configured"] = bool(config.base_url)
        checks["timeout_configured"] = config.timeout_seconds > 0
        details["api_configuration"] = {
            "model": config.model,
            "base_url": config.base_url,
            "temperature": config.temperature,
            "timeout_seconds": config.timeout_seconds,
            "api_key_recorded": False,
        }
    except Exception as error:
        checks["model_configured"] = False
        checks["base_url_configured"] = False
        checks["timeout_configured"] = False
        details["config_error"] = f"{type(error).__name__}:{error}"
    try:
        if prompt_data is not None:
            counter = _token_counter(tokenizer_path)
            prompt_tokens = int(counter(str(prompt_data["prompt"])))
            checks["context_fits"] = prompt_tokens + AUDITOR_MAX_TOKENS <= CONTEXT_LIMIT
            details["prompt_tokens_local"] = prompt_tokens
            details["auditor_max_tokens"] = AUDITOR_MAX_TOKENS
            details["context_limit"] = CONTEXT_LIMIT
    except Exception as error:
        checks["context_fits"] = False
        details["tokenizer_error"] = f"{type(error).__name__}:{error}"
    checks["reference_deferred_until_after_api"] = True
    report = {
        "preflight_version": RUNNER_VERSION,
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "details": details,
        "source_run": str(source_run.resolve()),
        "output_root": str(output_root.resolve()),
        "config_path": str(config_path.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "api_budget": {"adherence_auditor": 1, "total": 1, "retry": 0,
                        "planner": 0, "gate": 0, "realizer": 0,
                        "memorax": 0, "solver": 0},
        "api_calls": 0,
        "prompt_sha256": (sha256_bytes(str(prompt_data["prompt"]).encode())
                          if prompt_data else None),
        "module_manifest": _module_manifest(),
    }
    return report


def _complete_once(config: ChallengerAPIConfig, *, prompt: str,
                   max_tokens: int) -> BackendResponse:
    """Make exactly one HTTP request; this function intentionally has no retry."""
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    if config.enable_thinking is not None:
        payload["enable_thinking"] = config.enable_thinking
    request = urllib.request.Request(
        config.base_url + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.api_key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = body["choices"][0]
    return BackendResponse(
        str(choice["message"]["content"]), config.model,
        {"finish_reason": choice.get("finish_reason"),
         "usage": body.get("usage") if isinstance(body.get("usage"), dict) else None,
         "transport_attempt_count": 1,
         "transport_retry_count": 0,
         "rate_limit_retry_count": 0},
    )


def _compare(audit: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    refs = reference.get("scene_audits") or []
    rows = []
    participant_matches = development_matches = outcome_matches = exact = 0
    for index, scene in enumerate(audit.get("scenes") or [], 1):
        human = refs[index - 1] if index <= len(refs) and isinstance(refs[index - 1], Mapping) else {}
        fields = {
            "participant_adherence": scene.get("participant_adherence") == human.get("participant_adherence"),
            "development_status": scene.get("development_status") == human.get("development_status"),
            "durable_outcome_status": scene.get("durable_outcome_status") == human.get("durable_outcome_status"),
        }
        participant_matches += int(fields["participant_adherence"])
        development_matches += int(fields["development_status"])
        outcome_matches += int(fields["durable_outcome_status"])
        exact += int(all(fields.values()))
        rows.append({
            "scene_ref": scene.get("scene_ref"),
            "human_reference": {key: human.get(key) for key in fields},
            "llm_verdict": {key: scene.get(key) for key in fields},
            "evidence_turn_refs": {key: scene.get(key) for key in (
                "participant_evidence_turn_refs", "development_evidence_turn_refs",
                "durable_outcome_evidence_turn_refs")},
            "matches": fields,
            "exact_scene_match": all(fields.values()),
        })
    return {
        "scenes": rows,
        "participant_match_count": participant_matches,
        "development_match_count": development_matches,
        "durable_outcome_match_count": outcome_matches,
        "exact_scene_match_count": exact,
        "scene_count": len(audit.get("scenes") or []),
    }


def _report(*, source_run: Path, output_root: Path, preflight_report: Mapping[str, Any],
            raw_meta: Mapping[str, Any], validation: Mapping[str, Any],
            audit: Mapping[str, Any] | None, comparison: Mapping[str, Any] | None,
            error: str | None, reference_comparison_enabled: bool) -> str:
    lines = ["# Realizer Adherence Live Probe Report", "",
             f"- Source run (read-only): `{source_run}`",
             f"- Probe run: `{output_root}`",
             "- Auditor API calls: `1`", "- Auditor retries: `0`",
             "- Planner/Gate/Realizer/Memorax/Solver calls: `0`", "",
             "## Transport and Parse", "",
             f"- Finish reason: `{raw_meta.get('finish_reason')}`",
             f"- Prompt tokens: `{raw_meta.get('prompt_tokens')}`",
             f"- Completion tokens: `{raw_meta.get('completion_tokens')}`",
             f"- Total tokens: `{raw_meta.get('total_tokens')}`",
             f"- JSON parse success: `{raw_meta.get('json_parse_success')}`",
             f"- Contract valid: `{validation.get('valid')}`"]
    if error:
        lines.extend([f"- Runtime/validation error: `{error}`", ""])
    if audit is not None and comparison is not None:
        lines.extend(["", "## Scene Comparison", ""])
        for row in (comparison or {}).get("scenes", []):
            lines.extend([
                f"### {row['scene_ref']}", "",
                f"- Human reference: `{json.dumps(row['human_reference'], ensure_ascii=False, sort_keys=True)}`",
                f"- LLM verdict: `{json.dumps(row['llm_verdict'], ensure_ascii=False, sort_keys=True)}`",
                f"- Matches: `{json.dumps(row['matches'], sort_keys=True)}`",
                f"- Exact scene match: `{row['exact_scene_match']}`",
                "- Evidence refs:",
                f"  - participant: `{', '.join(row['evidence_turn_refs']['participant_evidence_turn_refs'])}`",
                f"  - development: `{', '.join(row['evidence_turn_refs']['development_evidence_turn_refs'])}`",
                f"  - durable outcome: `{', '.join(row['evidence_turn_refs']['durable_outcome_evidence_turn_refs'])}`",
                "- Evidence support: manually inspect the cited Child turns; no extra LLM call was made.", "",
            ])
        lines.extend([
            "## Match Totals", "",
            f"- Participant: `{comparison['participant_match_count']}/{comparison['scene_count']}`",
            f"- Development: `{comparison['development_match_count']}/{comparison['scene_count']}`",
            f"- Durable outcome: `{comparison['durable_outcome_match_count']}/{comparison['scene_count']}`",
            f"- Exact scene: `{comparison['exact_scene_match_count']}/{comparison['scene_count']}`", "",
        ])
    elif audit is not None:
        lines.extend(["", "## Scene Audit", ""])
        for scene in audit.get("scenes") or []:
            lines.extend([
                f"### {scene.get('scene_ref')}", "",
                f"- Participant adherence: `{scene.get('participant_adherence')}`",
                f"- Development: `{scene.get('development_status')}`",
                f"- Durable outcome: `{scene.get('durable_outcome_status')}`",
                f"- Scene verdict: `{scene.get('scene_verdict')}`",
                f"- Reason: {scene.get('reason')}", "",
            ])
    lines.extend(["## Frozen Checks", "",
                  f"- Preflight decision: `{preflight_report.get('decision')}`",
                  ("- Reference verdicts were loaded only after the API response."
                   if reference_comparison_enabled else
                   "- Reference comparison: `disabled` (no matching frozen reference)."),
                  "- Source run was not modified.", ""])
    return "\n".join(lines)


def run(*, source_run: Path, output_root: Path, config_path: Path,
        tokenizer_path: Path,
        reference_comparison_enabled: bool = True) -> dict[str, Any]:
    pre = preflight(source_run=source_run, output_root=output_root,
                    config_path=config_path, tokenizer_path=tokenizer_path)
    if pre["decision"] != "pass":
        raise RuntimeError("REALIZER_ADHERENCE_PREFLIGHT_FAILED:" + ",".join(pre["failed_checks"]))
    if output_root.exists():
        raise FileExistsError(f"adherence_live_probe_exists:{output_root}")
    source_manifest_before = _directory_manifest(source_run)
    plan, child = _load_inputs(source_run)
    auditor_input = build_adherence_auditor_input(plan=plan, child=child)
    prompt_data = build_adherence_auditor_prompt(auditor_input=auditor_input)
    output_root.mkdir(parents=True, exist_ok=False)
    for part in ("realizer_adherence", "attempts", "raw/prompts", "raw/backend_responses", "reports"):
        (output_root / part).mkdir(parents=True, exist_ok=True)
    freeze = {**pre, "freeze_version": RUNNER_VERSION, "frozen_at": _now(),
              "source_manifest_before_api": source_manifest_before,
              "first_api_request": None, "no_retry": True,
              "reference_fixture_sha256": None}
    atomic_json(output_root / "run_freeze_manifest.json", freeze)
    atomic_json(output_root / "run_status.json", {"status": "initialized", "api_calls": 0})
    atomic_json(output_root / "realizer_adherence" / "input.json", auditor_input)
    atomic_text(output_root / "realizer_adherence" / "prompt.txt", prompt_data["prompt"])
    atomic_json(output_root / "realizer_adherence" / "prompt_manifest.json", {
        "prompt_version": prompt_data["prompt_version"],
        "prompt_sha256": sha256_bytes(prompt_data["prompt"].encode()),
        "prompt_tokens_local": pre["details"].get("prompt_tokens_local"),
    })
    config = load_challenger_config(config_path)
    counter = _token_counter(tokenizer_path)
    ledger = RequestLedger(output_root, pre["module_manifest"]["manifest_sha256"], "live")
    request_id = "adherence_auditor_attempt_01"
    raw_meta: dict[str, Any] = {
        "finish_reason": None, "prompt_tokens": counter(prompt_data["prompt"]),
        "completion_tokens": 0, "total_tokens": counter(prompt_data["prompt"]),
        "json_parse_success": False,
    }
    error: str | None = None
    result: dict[str, Any] | None = None
    try:
        freeze["first_api_request"] = _now()
        atomic_json(output_root / "run_freeze_manifest.json", freeze)
        response = ledger.invoke(
            request_id=request_id, intent="realizer_adherence_auditor", attempt=1,
            prompt_sha256=sha256_bytes(prompt_data["prompt"].encode()),
            call=lambda: _complete_once(config, prompt=prompt_data["prompt"],
                                        max_tokens=AUDITOR_MAX_TOKENS),
            budget_class="realizer_adherence_auditor", budget_limit=1,
            total_budget_limit=1,
            request_metadata={"logical_role": "realizer_adherence_auditor",
                              "role_version": RUNNER_VERSION,
                              "prompt_version": prompt_data["prompt_version"],
                              "root_id": child.get("root_id"),
                              "cache_allowed": False,
                              "imported_response_allowed": False,
                              "orchestrator_retry_allowed": False},
            backend_identity={"transport": "single_http_no_retry",
                              "model": config.model, "base_url": config.base_url,
                              "temperature": config.temperature},
        )
        raw = response.raw_text
        usage = response.metadata.get("usage") if isinstance(response.metadata, Mapping) else None
        usage = usage if isinstance(usage, Mapping) else {}
        raw_meta.update({
            "finish_reason": response.metadata.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens") or counter(raw),
            "total_tokens": usage.get("total_tokens") or counter(prompt_data["prompt"]) + counter(raw),
            "provider_prompt_tokens": usage.get("prompt_tokens"),
            "provider_completion_tokens": usage.get("completion_tokens"),
            "provider_total_tokens": usage.get("total_tokens"),
            "model": response.model_identifier,
            "json_parse_success": False,
        })
        atomic_json(output_root / "realizer_adherence" / "raw_output.json", {
            "request_id": request_id, "raw_text": raw,
            "metadata": response.metadata,
        })
        parsed = json.loads(raw.strip())
        raw_meta["json_parse_success"] = True
        result = validate_and_materialize_adherence_audit(parsed, plan=plan, child=child)
        if not result["valid"]:
            error = "adherence_contract_invalid:" + ",".join(result["errors"])
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        if not (output_root / "realizer_adherence" / "raw_output.json").exists():
            atomic_json(output_root / "realizer_adherence" / "raw_output.json", {
                "request_id": request_id, "error": error,
            })
    audit = result.get("audit") if result and result.get("valid") else None
    validation = {
        "validation_version": "realizer_adherence_live_validation_v1",
        "valid": bool(result and result.get("valid") and not error),
        "errors": list(result.get("errors", [])) if result else ([error] if error else []),
        "api_calls": 1 if ledger.read(request_id) is not None else 0,
        "retries": 0,
        "planner_calls": 0, "semantic_gate_calls": 0, "realizer_calls": 0,
        "memorax_calls": 0, "solver_calls": 0,
        "source_manifest_before_api": source_manifest_before,
        "source_manifest_after_api": _directory_manifest(source_run),
        "source_run_unchanged": source_manifest_before == _directory_manifest(source_run),
    }
    atomic_json(output_root / "realizer_adherence" / "audit.json", audit or {})
    atomic_json(output_root / "realizer_adherence" / "validation.json", validation)
    reference = (load_json(DEFAULT_REFERENCE)
                 if reference_comparison_enabled else None)
    comparison = (_compare(audit, reference)
                  if audit and reference is not None else None)
    freeze["reference_fixture_sha256"] = (
        sha256_file(DEFAULT_REFERENCE) if reference_comparison_enabled else None)
    freeze["reference_comparison_enabled"] = reference_comparison_enabled
    freeze["source_manifest_after_api"] = validation["source_manifest_after_api"]
    freeze["api_call_count"] = validation["api_calls"]
    atomic_json(output_root / "run_freeze_manifest.json", freeze)
    atomic_json(output_root / "run_status.json", {
        "status": "completed" if audit else "failed",
        "failure_stage": "adherence_validation" if error else None,
        "api_calls": validation["api_calls"], "retries": 0,
    })
    report_text = _report(source_run=source_run, output_root=output_root,
                          preflight_report=pre, raw_meta=raw_meta,
                          validation=validation, audit=audit,
                          comparison=comparison, error=error,
                          reference_comparison_enabled=reference_comparison_enabled)
    atomic_text(output_root / "reports" / "REALIZER_ADHERENCE_LIVE_PROBE_REPORT.md", report_text)
    return {"run_root": str(output_root), "error": error,
            "raw_meta": raw_meta, "audit": audit,
            "comparison": comparison, "validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--skip-reference-comparison", action="store_true")
    args = parser.parse_args(argv)
    report = preflight(source_run=args.source_run, output_root=args.output_root,
                       config_path=args.config, tokenizer_path=args.tokenizer)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.preflight_only or not args.live:
        return 0 if report["decision"] == "pass" else 2
    result = run(source_run=args.source_run, output_root=args.output_root,
                 config_path=args.config, tokenizer_path=args.tokenizer,
                 reference_comparison_enabled=not args.skip_reference_comparison)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
