"""Safety-gated live executor for the isolated 30-root H4 batch probe.

The executor only orchestrates the already frozen expansion and QA CLIs.  It
does not import or alter their Planner, Realizer, Writer, Solver, or Memory
logic.  Expansion outputs become QA inputs only after a completed report,
generation-ready flag, root/input identity checks, and dialogue validation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, sha256_file
from .plan import (
    QA_MODULE,
    _namespace,
    select_successful_expansion_source,
)


EXECUTOR_VERSION = "h4_batch30_memory_event_k3_executor_v1"
EXPANSION_REPORT = Path("reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json")
PROMOTED_DIALOGUE = Path("promotion/promotable_expanded_dialogue.json")
RAW_DIALOGUE = Path("expanded_dialogue.json")
ALIAS_RE = re.compile(r"(?:^|\s)(?:Classmates|Neighbors|Co-workers)\s+[AB](?:$|\W)")
SPEAKER_ALIAS_RE = re.compile(r"(?:^|\s)[AB]$")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not text:
        raise ValueError("safe_name_empty")
    return text


def _json_line(**payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _replace_cli_value(command: Sequence[str], flag: str, value: str) -> list[str]:
    result = list(command)
    try:
        index = result.index(flag)
    except ValueError as error:
        raise ValueError(f"command_flag_missing:{flag}") from error
    if index + 1 >= len(result):
        raise ValueError(f"command_flag_value_missing:{flag}")
    result[index + 1] = value
    return result


def _expanded_validation(path: Path, *, root_id: str) -> dict[str, Any]:
    errors: list[str] = []
    try:
        value = _load_object(path)
    except Exception as error:
        return {
            "valid": False,
            "errors": [f"{type(error).__name__}:{error}"],
            "path": str(path),
        }
    if str(value.get("root_id") or "") != root_id:
        errors.append("expanded_root_id_mismatch")
    turns = value.get("turns")
    if not isinstance(turns, list) or not turns:
        errors.append("expanded_turns_missing")
        turns = []
    parent_count = value.get("parent_turn_count")
    child_count = value.get("child_turn_count")
    if not isinstance(parent_count, int) or parent_count <= 0:
        errors.append("parent_turn_count_invalid")
    if not isinstance(child_count, int) or child_count <= 0:
        errors.append("child_turn_count_invalid")
    if isinstance(parent_count, int) and isinstance(child_count, int):
        if parent_count + child_count != len(turns):
            errors.append("expanded_turn_count_mismatch")
    alias_hits: list[str] = []
    for row in turns:
        if not isinstance(row, Mapping):
            errors.append("expanded_turn_not_object")
            continue
        speaker = str(row.get("speaker") or "")
        text = str(row.get("text") or "")
        if SPEAKER_ALIAS_RE.search(speaker) or ALIAS_RE.search(speaker):
            alias_hits.append(f"speaker:{speaker}")
        if ALIAS_RE.search(text):
            alias_hits.append(f"text:{text[:100]}")
    if alias_hits:
        errors.append("speaker_alias_residue")
    return {
        "valid": not errors,
        "errors": errors,
        "path": str(path),
        "sha256": sha256_file(path),
        "root_id": value.get("root_id"),
        "turn_count": len(turns),
        "parent_turn_count": parent_count,
        "child_turn_count": child_count,
        "alias_hits": alias_hits,
    }


def validate_expansion_attempt(
    attempt_root: Path,
    *,
    root_id: str,
    input_sha256: str,
    parent_dialogue_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a fail-closed eligibility record for one expansion attempt."""
    report_path = attempt_root / EXPANSION_REPORT
    report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            report = _load_object(report_path)
        except Exception:
            report = {}
    selection = select_successful_expansion_source(
        attempt_root,
        expected_root_id=root_id,
        expected_input_sha256=input_sha256,
        expected_parent_dialogue_sha256=parent_dialogue_sha256,
    )
    errors = list(selection.get("reason_codes") or [])
    # These route checks are executor policy in addition to the generic source
    # selector.  They prevent an accidental reuse of a successful artifact
    # produced by a different expansion mode.
    if report and report.get("child_add_verified") is not True:
        errors.append("child_add_not_verified")
    if report and report.get("event_bucket_only") is not True:
        errors.append("event_bucket_route_mismatch")
    eligible = bool(selection.get("eligible")) and not errors
    selected = Path(str(selection["selected_path"])) if eligible else None
    selected_kind = selection.get("source_kind") if eligible else None
    dialogue_validation: dict[str, Any] | None = None
    if selected is not None:
        dialogue_validation = _expanded_validation(selected, root_id=root_id)
        if not dialogue_validation["valid"]:
            errors.extend(
                f"dialogue_{error}"
                for error in dialogue_validation.get("errors") or [])
            eligible = False
        else:
            dialogue_validation["complete_dialogue_sha256"] = selection.get(
                "complete_dialogue_sha256")
            dialogue_validation["selector_checks"] = selection.get("checks") or {}
    return {
        "executor_version": EXECUTOR_VERSION,
        "root_id": root_id,
        "attempt_root": str(attempt_root),
        "eligible": eligible and selected is not None,
        "errors": errors,
        "report_path": str(report_path),
        "report_status": report.get("status"),
        "generation_ready": report.get("generation_ready"),
        "failure_stage": report.get("failure_stage"),
        "runtime_error": report.get("runtime_error"),
        "event_bucket_only": report.get("event_bucket_only"),
        "child_add_verified": report.get("child_add_verified"),
        "input_sha256": report.get("input_sha256"),
        "selected_source": str(selected) if selected else None,
        "selected_source_kind": selected_kind,
        "selected_source_sha256": (
            dialogue_validation.get("sha256") if dialogue_validation else None
        ),
        "source_selector": selection,
        "dialogue_validation": dialogue_validation,
        "request_accounting": {
            "expansion_calls": report.get("expansion_calls"),
            "memorax_calls": report.get("memorax_calls"),
            "llm_usage": report.get("llm_usage"),
        },
    }


def _attempt_root(base: Path, attempt_index: int) -> Path:
    if attempt_index == 1:
        return base
    return base.with_name(f"{base.name}__attempt_{attempt_index:02d}")


def _attempt_command(
    root_plan: Mapping[str, Any], *, attempt_index: int,
    namespace_prefix: str,
) -> tuple[list[str], Path, str]:
    root_id = str(root_plan["root_id"])
    base = Path(str(root_plan["expansion"]["run_root"]))
    run_root = _attempt_root(base, attempt_index)
    namespace = _namespace(namespace_prefix, root_id, attempt_index=attempt_index)
    command = _replace_cli_value(
        root_plan["expansion"]["command"], "--run-root", str(run_root)
    )
    command = _replace_cli_value(
        command, "--memorax-user-namespace", namespace
    )
    return command, run_root, namespace


def _run_command(
    command: Sequence[str], *, cwd: Path, log_path: Path, timeout_seconds: int,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(cwd) if not current_pythonpath else f"{cwd}:{current_pythonpath}"
    )
    try:
        with log_path.open("wb") as handle:
            completed = subprocess.run(
                list(command), cwd=cwd, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, timeout=timeout_seconds, check=False,
            )
        return {
            "returncode": completed.returncode,
            "timed_out": False,
            "log_path": str(log_path),
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": None,
            "timed_out": True,
            "log_path": str(log_path),
        }


def _result_path(shard: Mapping[str, Any], root_id: str) -> Path:
    return Path(str(shard["run_root"])) / "expansion_results" / f"{_safe_name(root_id)}.json"


def _existing_ready(
    path: Path,
    *,
    root_id: str | None = None,
    input_sha256: str | None = None,
    parent_dialogue_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = _load_object(path)
    except Exception:
        return None
    selected = Path(str(value.get("selected_source") or ""))
    expected_hash = str(value.get("selected_source_sha256") or "")
    if not (
        value.get("status") == "EXPANSION_READY"
        and selected.is_file()
        and expected_hash
        and sha256_file(selected) == expected_hash
    ):
        return None
    if root_id is not None and input_sha256 is not None:
        attempt_root_raw = value.get("successful_attempt_root")
        if not attempt_root_raw:
            return None
        selection = select_successful_expansion_source(
            Path(str(attempt_root_raw)),
            expected_root_id=str(root_id),
            expected_input_sha256=str(input_sha256),
            expected_parent_dialogue_sha256=parent_dialogue_sha256,
        )
        if not selection.get("eligible"):
            return None
        if (str(selection.get("selected_path")) != str(selected)
                or selection.get("selected_sha256") != expected_hash):
            return None
    return value


def execute_expansion_root(
    root_plan: Mapping[str, Any], *, shard: Mapping[str, Any], repo_root: Path,
    namespace_prefix: str, max_attempts: int, timeout_seconds: int,
    imported_attempt_root: Path | None = None,
) -> dict[str, Any]:
    root_id = str(root_plan["root_id"])
    result_path = _result_path(shard, root_id)
    existing = _existing_ready(
        result_path,
        root_id=root_id,
        input_sha256=str(root_plan["input_sha256"]),
        parent_dialogue_sha256=str(
            root_plan.get("parent_dialogue_sha256") or "") or None,
    )
    if existing is not None:
        _json_line(event="expansion_reused", root_id=root_id,
                   selected_source=existing["selected_source"])
        return existing

    input_sha = str(root_plan["input_sha256"])
    attempts: list[dict[str, Any]] = []
    if imported_attempt_root is not None:
        audit = validate_expansion_attempt(
            imported_attempt_root, root_id=root_id, input_sha256=input_sha,
            parent_dialogue_sha256=str(
                root_plan.get("parent_dialogue_sha256") or "") or None,
        )
        attempts.append({"attempt_index": 0, "imported": True, "audit": audit})
        if audit["eligible"]:
            result = {
                "executor_version": EXECUTOR_VERSION,
                "root_id": root_id,
                "status": "EXPANSION_READY",
                "imported_verified_attempt": True,
                "attempts": attempts,
                "selected_source": audit["selected_source"],
                "selected_source_kind": audit["selected_source_kind"],
                "selected_source_sha256": audit["selected_source_sha256"],
                "successful_attempt_root": str(imported_attempt_root),
            }
            atomic_json(result_path, result)
            _json_line(event="expansion_import_verified", root_id=root_id,
                       selected_source=result["selected_source"])
            return result
        _json_line(event="expansion_import_rejected", root_id=root_id,
                   errors=audit["errors"])

    for attempt_index in range(1, max_attempts + 1):
        command, attempt_root, namespace = _attempt_command(
            root_plan, attempt_index=attempt_index,
            namespace_prefix=namespace_prefix,
        )
        if attempt_root.exists():
            audit = validate_expansion_attempt(
                attempt_root, root_id=root_id, input_sha256=input_sha,
                parent_dialogue_sha256=str(
                    root_plan.get("parent_dialogue_sha256") or "") or None,
            )
            attempts.append({
                "attempt_index": attempt_index,
                "namespace": namespace,
                "command": command,
                "existing_attempt": True,
                "audit": audit,
            })
            if audit["eligible"]:
                execution = {"returncode": 0, "timed_out": False,
                             "log_path": None, "existing_attempt": True}
            else:
                _json_line(event="expansion_existing_quarantined",
                           root_id=root_id, attempt_index=attempt_index,
                           errors=audit["errors"])
                continue
        else:
            log_path = (
                Path(str(shard["run_root"])) / "logs" /
                f"{_safe_name(root_id)}__attempt_{attempt_index:02d}.log"
            )
            _json_line(event="expansion_started", root_id=root_id,
                       attempt_index=attempt_index, namespace=namespace,
                       run_root=str(attempt_root))
            execution = _run_command(
                command, cwd=repo_root, log_path=log_path,
                timeout_seconds=timeout_seconds,
            )
            audit = validate_expansion_attempt(
                attempt_root, root_id=root_id, input_sha256=input_sha,
                parent_dialogue_sha256=str(
                    root_plan.get("parent_dialogue_sha256") or "") or None,
            )
            attempts.append({
                "attempt_index": attempt_index,
                "namespace": namespace,
                "command": command,
                "execution": execution,
                "audit": audit,
            })
        if audit["eligible"]:
            result = {
                "executor_version": EXECUTOR_VERSION,
                "root_id": root_id,
                "status": "EXPANSION_READY",
                "imported_verified_attempt": False,
                "attempts": attempts,
                "selected_source": audit["selected_source"],
                "selected_source_kind": audit["selected_source_kind"],
                "selected_source_sha256": audit["selected_source_sha256"],
                "successful_attempt_root": str(attempt_root),
            }
            atomic_json(result_path, result)
            _json_line(event="expansion_ready", root_id=root_id,
                       attempt_index=attempt_index,
                       selected_source=result["selected_source"])
            return result
        _json_line(event="expansion_attempt_failed", root_id=root_id,
                   attempt_index=attempt_index,
                   returncode=execution.get("returncode"),
                   timed_out=execution.get("timed_out"),
                   errors=audit["errors"],
                   failure_stage=audit.get("failure_stage"))

    result = {
        "executor_version": EXECUTOR_VERSION,
        "root_id": root_id,
        "status": "EXPANSION_QUARANTINED",
        "attempts": attempts,
        "selected_source": None,
        "selected_source_sha256": None,
    }
    atomic_json(result_path, result)
    return result


def _select_shards(plan: Mapping[str, Any], names: Sequence[str]) -> list[dict[str, Any]]:
    shards = [dict(row) for row in plan.get("shards") or []]
    if not names or "all" in names:
        return shards
    wanted = set(names)
    selected = [row for row in shards if str(row.get("shard")) in wanted]
    missing = sorted(wanted - {str(row.get("shard")) for row in selected})
    if missing:
        raise ValueError("unknown_shards:" + ",".join(missing))
    return selected


def execute_expansion(
    plan: Mapping[str, Any], *, shards: Sequence[Mapping[str, Any]],
    repo_root: Path, namespace_prefix: str, max_attempts: int,
    timeout_seconds: int, imports: Mapping[str, Path], max_roots: int | None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    scheduled = 0
    for shard in shards:
        for root_plan in shard.get("roots") or []:
            if max_roots is not None and scheduled >= max_roots:
                break
            scheduled += 1
            root_id = str(root_plan["root_id"])
            results.append(execute_expansion_root(
                root_plan, shard=shard, repo_root=repo_root,
                namespace_prefix=namespace_prefix, max_attempts=max_attempts,
                timeout_seconds=timeout_seconds,
                imported_attempt_root=imports.get(root_id),
            ))
        if max_roots is not None and scheduled >= max_roots:
            break
    ready = [row for row in results if row.get("status") == "EXPANSION_READY"]
    quarantined = [row for row in results if row.get("status") != "EXPANSION_READY"]
    summary = {
        "executor_version": EXECUTOR_VERSION,
        "status": "completed" if not quarantined else "partial",
        "scheduled_root_count": scheduled,
        "ready_root_count": len(ready),
        "quarantined_root_count": len(quarantined),
        "ready_root_ids": [row["root_id"] for row in ready],
        "quarantined_root_ids": [row["root_id"] for row in quarantined],
        "results": results,
    }
    atomic_json(Path(str(plan["run_root"])) / "expansion_execution_report.json", summary)
    _json_line(event="expansion_phase_completed", **{
        key: summary[key] for key in (
            "status", "scheduled_root_count", "ready_root_count",
            "quarantined_root_count", "quarantined_root_ids",
        )
    })
    return summary


def _qa_sources(shard: Mapping[str, Any]) -> tuple[list[Path], list[str]]:
    sources: list[Path] = []
    missing: list[str] = []
    for root_plan in shard.get("roots") or []:
        root_id = str(root_plan["root_id"])
        result = _existing_ready(
            _result_path(shard, root_id),
            root_id=root_id,
            input_sha256=str(root_plan.get("input_sha256") or ""),
            parent_dialogue_sha256=str(
                root_plan.get("parent_dialogue_sha256") or "") or None,
        )
        if result is None:
            missing.append(root_id)
            continue
        sources.append(Path(str(result["selected_source"])))
    return sources, missing


def execute_qa_shard(
    shard: Mapping[str, Any], *, repo_root: Path, config: Path,
    tokenizer: Path, solver_model: Path, timeout_seconds: int,
    allow_partial: bool,
) -> dict[str, Any]:
    sources, missing = _qa_sources(shard)
    expected = int(shard["root_count"])
    if missing and not allow_partial:
        result = {
            "executor_version": EXECUTOR_VERSION,
            "shard": shard["shard"],
            "status": "blocked_incomplete_expansion",
            "expected_root_count": expected,
            "selected_root_count": len(sources),
            "missing_root_ids": missing,
        }
        atomic_json(Path(str(shard["run_root"])) / "qa_execution_result.json", result)
        _json_line(event="qa_blocked", shard=shard["shard"],
                   missing_root_ids=missing)
        return result
    if not sources:
        raise RuntimeError(f"qa_sources_empty:{shard['shard']}")
    qa_root = Path(str(shard["qa_root"]))
    stages = [
        (
            "inventory",
            [sys.executable, "-m", QA_MODULE, "--phase", "inventory",
             "--run-root", str(qa_root), "--max-roots", str(len(sources))]
            + [part for source in sources for part in ("--root-source", str(source))],
        ),
        (
            "smoke",
            [sys.executable, "-m", QA_MODULE, "--phase", "smoke", "--live",
             "--run-root", str(qa_root), "--api-config", str(config),
             "--tokenizer", str(tokenizer), "--solver-model", str(solver_model),
             "--smoke-candidate-limit", "3"],
        ),
        (
            "batch",
            [sys.executable, "-m", QA_MODULE, "--phase", "batch", "--live",
             "--run-root", str(qa_root), "--api-config", str(config),
             "--tokenizer", str(tokenizer), "--solver-model", str(solver_model)],
        ),
    ]
    records: list[dict[str, Any]] = []
    for stage, command in stages:
        if stage == "inventory" and qa_root.exists():
            manifest = qa_root / "manifest.json"
            if manifest.is_file():
                records.append({"stage": stage, "status": "existing"})
                continue
            raise RuntimeError(f"qa_root_exists_without_manifest:{qa_root}")
        if stage == "smoke":
            report = qa_root / "smoke_report.json"
            if report.is_file() and _load_object(report).get("status") == "pass":
                records.append({"stage": stage, "status": "existing_pass"})
                continue
        if stage == "batch":
            report = qa_root / "batch_report.json"
            if report.is_file() and _load_object(report).get("status") in {
                "completed", "completed_with_generation_gaps"
            }:
                records.append({"stage": stage, "status": "existing_completed"})
                continue
        log_path = Path(str(shard["run_root"])) / "logs" / f"qa_{stage}.log"
        _json_line(event="qa_stage_started", shard=shard["shard"], stage=stage,
                   root_count=len(sources))
        execution = _run_command(
            command, cwd=repo_root, log_path=log_path,
            timeout_seconds=timeout_seconds,
        )
        record = {"stage": stage, "command": command, **execution}
        records.append(record)
        if execution["returncode"] != 0 or execution["timed_out"]:
            result = {
                "executor_version": EXECUTOR_VERSION,
                "shard": shard["shard"], "status": f"qa_{stage}_failed",
                "expected_root_count": expected,
                "selected_root_count": len(sources),
                "missing_root_ids": missing,
                "stages": records,
            }
            atomic_json(Path(str(shard["run_root"])) / "qa_execution_result.json", result)
            _json_line(event="qa_stage_failed", shard=shard["shard"], stage=stage,
                       returncode=execution["returncode"],
                       timed_out=execution["timed_out"])
            return result
        _json_line(event="qa_stage_completed", shard=shard["shard"], stage=stage)
    batch_report = _load_object(qa_root / "batch_report.json")
    result = {
        "executor_version": EXECUTOR_VERSION,
        "shard": shard["shard"],
        "status": "completed" if batch_report.get("status") in {
            "completed", "completed_with_generation_gaps"
        } else "partial",
        "expected_root_count": expected,
        "selected_root_count": len(sources),
        "missing_root_ids": missing,
        "qa_run_root": str(qa_root),
        "qa_batch_status": batch_report.get("status"),
        "stages": records,
    }
    atomic_json(Path(str(shard["run_root"])) / "qa_execution_result.json", result)
    return result


def _parse_imports(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        root_id, separator, raw_path = value.partition("=")
        if not separator or not root_id or not raw_path:
            raise ValueError("verified_expansion_format_requires_ROOT_ID=ATTEMPT_ROOT")
        if root_id in result:
            raise ValueError(f"duplicate_verified_expansion:{root_id}")
        result[root_id] = Path(raw_path).expanduser().resolve()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("expansion", "qa", "all"), required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--shard", action="append", default=[])
    parser.add_argument("--namespace-prefix", required=True)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-roots", type=int)
    parser.add_argument("--expansion-timeout-seconds", type=int, default=3600)
    parser.add_argument("--qa-timeout-seconds", type=int, default=43200)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--verified-expansion", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_attempts <= 0:
        raise ValueError("max_attempts_must_be_positive")
    if args.max_roots is not None and args.max_roots <= 0:
        raise ValueError("max_roots_must_be_positive")
    plan_path = args.plan_root.resolve() / "batch_plan.json"
    plan = _load_object(plan_path)
    if plan.get("offline_only") is not True:
        raise ValueError("batch_plan_offline_contract_missing")
    shards = _select_shards(plan, args.shard)
    imports = _parse_imports(args.verified_expansion)
    result: dict[str, Any] = {
        "executor_version": EXECUTOR_VERSION,
        "plan_root": str(args.plan_root.resolve()),
        "phase": args.phase,
        "selected_shards": [row["shard"] for row in shards],
    }
    if args.phase in {"expansion", "all"}:
        result["expansion"] = execute_expansion(
            plan, shards=shards, repo_root=args.repo_root.resolve(),
            namespace_prefix=args.namespace_prefix,
            max_attempts=args.max_attempts,
            timeout_seconds=args.expansion_timeout_seconds,
            imports=imports, max_roots=args.max_roots,
        )
    if args.phase in {"qa", "all"}:
        if args.max_roots is not None:
            raise ValueError("max_roots_is_not_supported_for_qa")
        config = Path(str(plan["runtime_assets"]["challenger_config"]))
        tokenizer = Path(str(plan["runtime_assets"]["tokenizer"]))
        solver_model = Path(str(plan["runtime_assets"]["solver_model"]))
        qa_results = [execute_qa_shard(
            shard, repo_root=args.repo_root.resolve(), config=config,
            tokenizer=tokenizer, solver_model=solver_model,
            timeout_seconds=args.qa_timeout_seconds,
            allow_partial=args.allow_partial,
        ) for shard in shards]
        result["qa"] = qa_results
    atomic_json(args.plan_root.resolve() / "execution_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    failed = False
    if (result.get("expansion") or {}).get("status") == "partial":
        failed = True
    if any(row.get("status") != "completed" for row in result.get("qa") or []):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
