"""Gold-blind semantic recheck for frozen H4 QA candidates.

This is an independent diagnostic stage.  The live auditor never receives a
gold label, submitted evidence refs, candidate/source metadata, or prior audit
judgments.  The local runner compares its reconstructed answer to the frozen
answer key only after the response has been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_bytes, sha256_file
from ..ledger import AmbiguousRequestError, RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..h4_qa_multiroot_k3_t08_semantic_audit_v1.runner import (
    _frozen_turns_for_root,
    _root_dir,
)
from .protocol import PROTOCOL_VERSION, PROMPT_VERSION, build_prompt, validate_output


PROBE_VERSION = "h4_qa_multiroot_k3_t08_blind_semantic_recheck_v1"
DEFAULT_INPUT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_qa_multiroot_k3_t08_semantic_audit_v1_live_3"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_qa_multiroot_k3_t08_blind_semantic_recheck_v1_live_1"
)
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
MAX_TOKENS = 768
TEMPERATURE = 0.0
MAX_ATTEMPTS = 2


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _load_candidates(input_run: Path) -> list[dict[str, Any]]:
    """Load hard candidates from the completed visible-audit run.

    The nested candidate is retained only in the local frozen inventory.  The
    protocol layer strips its gold/evidence/source fields before building a
    model-visible prompt.
    """
    payload = _load(input_run / "audit_results.json")
    rows: list[dict[str, Any]] = []
    for result in payload.get("results") or []:
        if result.get("source_difficulty_bucket") != "hard_candidate":
            continue
        source = result.get("candidate")
        if not isinstance(source, dict):
            raise ValueError("visible_audit_candidate_missing")
        row = dict(source)
        row["source_difficulty_bucket"] = "hard_candidate"
        row["visible_audit_status"] = result.get("semantic_status")
        row["visible_audit_answer_valid"] = result.get("answer_semantically_valid")
        rows.append(row)
    seen: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen:
            raise ValueError(f"duplicate_or_missing_candidate_id:{candidate_id}")
        seen.add(candidate_id)
    return rows


def _input_hashes(input_run: Path) -> dict[str, Any]:
    return {
        "audit_results": sha256_file(input_run / "audit_results.json"),
        "dialogues": {
            str(root_id): sha256_file(
                _root_dir(input_run, str(root_id)) / "frozen_dialogue.json"
            )
            for root_id in sorted({str(row.get("root_id")) for row in _load_candidates(input_run)})
        },
        "report": sha256_file(input_run / "report.json"),
    }


def _inventory(input_run: Path) -> dict[str, Any]:
    candidates = _load_candidates(input_run)
    root_ids = sorted({str(row.get("root_id") or "") for row in candidates})
    if "" in root_ids:
        raise ValueError("candidate_root_id_missing")
    source_hashes: dict[str, str] = {}
    for root_id in root_ids:
        frozen = _root_dir(input_run, root_id) / "frozen_dialogue.json"
        if not frozen.is_file():
            raise FileNotFoundError(frozen)
        turns = _frozen_turns_for_root(input_run, root_id)
        if not turns:
            raise ValueError(f"empty_frozen_dialogue:{root_id}")
        source_hashes[root_id] = sha256_file(frozen)
    return {
        "probe_version": PROBE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "created_at": _now(),
        "input_run": str(input_run),
        "input_audit_results_sha256": sha256_file(input_run / "audit_results.json"),
        "input_report_sha256": sha256_file(input_run / "report.json"),
        "input_dialogue_sha256": source_hashes,
        "candidate_count": len(candidates),
        "root_ids": root_ids,
        "scope": {
            "hard_candidate": sum(row["source_difficulty_bucket"] == "hard_candidate" for row in candidates),
            "invalid_or_ambiguous": sum(row["source_difficulty_bucket"] == "invalid_or_ambiguous" for row in candidates),
        },
        # This is local-only frozen metadata.  It is never included in a
        # blind prompt or request metadata.
        "candidates": candidates,
    }


def inventory_phase(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_root.exists():
        raise RuntimeError(f"output_run_already_exists:{args.run_root}")
    if not args.input_run.is_dir():
        raise FileNotFoundError(args.input_run)
    inventory = _inventory(args.input_run)
    args.run_root.mkdir(parents=True, exist_ok=False)
    atomic_json(args.run_root / "manifest.json", inventory)
    (args.run_root / "frozen").mkdir(parents=True, exist_ok=True)
    # Keep an opaque local linkage table separate from prompt payloads.
    atomic_json(args.run_root / "frozen/candidate_keys.json", [
        {"audit_id": _safe_id(f"{PROBE_VERSION}:{row['candidate_id']}"),
         "candidate_id": row["candidate_id"],
         "root_id": row["root_id"],
         "source_difficulty_bucket": row["source_difficulty_bucket"]}
        for row in inventory["candidates"]
    ])
    for root_id in inventory["root_ids"]:
        root_out = _root_dir(args.run_root, root_id)
        root_out.mkdir(parents=True, exist_ok=True)
        source = _root_dir(args.input_run, root_id) / "frozen_dialogue.json"
        atomic_json(root_out / "frozen_dialogue.json", _load(source))
    report = {
        "status": "completed", "phase": "inventory", "offline_only": True,
        "candidate_count": inventory["candidate_count"],
        "root_count": len(inventory["root_ids"]), "scope": inventory["scope"],
        "challenger_api_calls": 0, "solver_calls": 0,
        "run_root": str(args.run_root),
    }
    atomic_json(args.run_root / "inventory_report.json", report)
    atomic_text(args.run_root / "summary.md", "# H4 blind semantic recheck inventory\n\n" +
                json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def _manifest(run_root: Path) -> dict[str, Any]:
    manifest = _load(run_root / "manifest.json")
    if manifest.get("probe_version") != PROBE_VERSION:
        raise ValueError("blind_recheck_manifest_version_mismatch")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("blind_recheck_protocol_version_mismatch")
    return manifest


def _verify_input_unchanged(manifest: Mapping[str, Any]) -> dict[str, Any]:
    input_run = Path(str(manifest["input_run"]))
    actual_audit = sha256_file(input_run / "audit_results.json")
    actual_report = sha256_file(input_run / "report.json")
    actual_dialogues = {
        str(root_id): sha256_file(
            _root_dir(input_run, str(root_id)) / "frozen_dialogue.json"
        )
        for root_id in manifest.get("root_ids") or []
    }
    mismatches: dict[str, Any] = {}
    if manifest.get("input_audit_results_sha256") != actual_audit:
        mismatches["audit_results"] = [manifest.get("input_audit_results_sha256"), actual_audit]
    if manifest.get("input_report_sha256") != actual_report:
        mismatches["report"] = [manifest.get("input_report_sha256"), actual_report]
    if manifest.get("input_dialogue_sha256") != actual_dialogues:
        mismatches["dialogues"] = {
            "expected": manifest.get("input_dialogue_sha256"), "actual": actual_dialogues,
        }
    if mismatches:
        raise RuntimeError(f"blind_input_integrity_changed:{json.dumps(mismatches, sort_keys=True)}")
    return {"unchanged": True, "audit_results": actual_audit,
            "report": actual_report, "dialogues": actual_dialogues}


def _candidate_result_path(run_root: Path, audit_id: str) -> Path:
    return run_root / "results" / f"{audit_id}.json"


def _backend(args: argparse.Namespace) -> OpenAICompatibleChallengerClient:
    from dataclasses import replace
    config = load_challenger_config(args.api_config)
    return OpenAICompatibleChallengerClient(replace(
        config, timeout_seconds=max(180, int(config.timeout_seconds)),
        questioner_max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        enable_thinking=False))


def _labels(candidate: Mapping[str, Any]) -> list[str]:
    if str(candidate.get("question_type")) == "ordering":
        return [str(x) for x in (candidate.get("items") or {}).keys()]
    return [str(x) for x in (candidate.get("options") or {}).keys()]


def _local_gold(candidate: Mapping[str, Any]) -> list[str]:
    if str(candidate.get("question_type")) == "single_choice":
        return [str(candidate.get("correct_option"))]
    return [str(x) for x in candidate.get("gold") or []]


def _compare(candidate: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    qtype = str(candidate.get("question_type") or "")
    status = str(audit.get("answer_status") or "")
    predicted = [str(x) for x in audit.get("predicted_answer") or []]
    gold = _local_gold(candidate)
    if status != "determinate":
        match = "not_attempted"
        blind_status = "ambiguous" if status == "ambiguous" else "unsupported"
    elif qtype == "multi_select":
        match = "exact" if set(predicted) == set(gold) else "disputed"
        blind_status = "confirmed" if match == "exact" else "disputed"
    else:
        match = "exact" if predicted == gold else "disputed"
        blind_status = "confirmed" if match == "exact" else "disputed"
    quality_ok = (audit.get("question_answerable") is True and
                  audit.get("option_quality") == "unique" and
                  audit.get("question_answer_leakage") == "clean")
    if blind_status == "confirmed" and not quality_ok:
        blind_status = "confirmed_with_quality_issue"
    return {
        "predicted_answer": predicted,
        "gold_match": match,
        "blind_status": blind_status,
        "gold_answer_local": gold,
        "quality_ok": quality_ok,
    }


def _audit_candidate(*, candidate: Mapping[str, Any], turns: Sequence[Mapping[str, Any]],
                     audit_id: str, run_root: Path, ledger: RequestLedger,
                     client: OpenAICompatibleChallengerClient,
                     total_budget: int) -> dict[str, Any]:
    result_path = _candidate_result_path(run_root, audit_id)
    if result_path.is_file():
        existing = _load(result_path)
        if (existing.get("protocol_version") != PROTOCOL_VERSION or
                existing.get("audit_id") != audit_id):
            raise RuntimeError(f"stale_blind_result:{audit_id}")
        return existing
    prompt = build_prompt(candidate=candidate, turns=turns)
    logical_id = f"blind-recheck-{audit_id}"
    refs = [str(turn.get("turn_ref")) for turn in turns]
    attempts: list[dict[str, Any]] = []
    final_validation: dict[str, Any] | None = None
    final_raw: Any = None
    for attempt in range(MAX_ATTEMPTS):
        retry = attempt > 0
        rendered = prompt
        if retry:
            rendered += (
                "\n\nYour previous response did not match the JSON shape. Return the same independent "
                "answer reconstruction with exactly the required fields; do not add an answer key."
            )
        request_id = f"{logical_id}-a{attempt:02d}"
        prompt_hash = sha256_bytes(rendered.encode("utf-8"))
        prompt_path = run_root / "raw" / "prompts" / f"{request_id}.json"
        atomic_json(prompt_path, {
            "request_id": request_id, "audit_id": audit_id, "attempt": attempt,
            "protocol_version": PROTOCOL_VERSION, "prompt_sha256": prompt_hash,
            "prompt": rendered,
        })
        try:
            response = ledger.invoke(
                request_id=request_id, intent="blind_semantic_recheck", attempt=attempt + 1,
                prompt_sha256=prompt_hash,
                call=lambda rendered=rendered: client.complete(
                    prompt=rendered, max_tokens=MAX_TOKENS, json_object=True),
                budget_class="blind_semantic_recheck", budget_limit=total_budget,
                total_budget_limit=total_budget,
                request_metadata={"audit_id": audit_id, "attempt": attempt,
                                  "retry": retry, "protocol_version": PROTOCOL_VERSION},
                backend_identity={"backend": "openai_compatible_blind_auditor",
                                  "model_identifier": client.config.model,
                                  "temperature": client.config.temperature,
                                  "max_tokens": MAX_TOKENS, "response_format": "json_object"},
            )
        except (AmbiguousRequestError, Exception) as error:
            # Keep technical failures distinct from a model's disputed answer.
            result = {"audit_id": audit_id, "candidate_id": candidate.get("candidate_id"),
                      "root_id": candidate.get("root_id"),
                      "status": "technical_error", "technical_error": str(error),
                      "attempts": attempts, "protocol_version": PROTOCOL_VERSION,
                      "candidate": dict(candidate)}
            atomic_json(result_path, result)
            return result
        state = ledger.read(request_id) or {}
        raw_text = response.raw_text
        try:
            parsed = json.loads(raw_text)
            validation = validate_output(
                parsed, question_type=str(candidate.get("question_type") or ""),
                valid_labels=_labels(candidate), valid_turn_refs=refs)
            parse_status = "json_parsed"
        except json.JSONDecodeError as error:
            parsed = None
            validation = {"decision": "protocol_invalid", "errors": ["json_invalid", str(error)],
                          "audit": None}
            parse_status = "json_invalid"
        should_retry = attempt == 0 and validation.get("decision") == "protocol_invalid"
        ledger.record_decision(request_id, "retry_format_only" if should_retry else
                               str(validation.get("decision")))
        raw_record = {
            "request_id": request_id, "audit_id": audit_id, "attempt": attempt,
            "parse_status": parse_status, "validation": validation,
            "raw_response": raw_text, "response_metadata": response.metadata,
            "raw_response_sha256": state.get("raw_response_sha256"),
        }
        raw_path = run_root / "raw" / "auditor" / f"{request_id}.json"
        atomic_json(raw_path, raw_record)
        attempts.append({k: v for k, v in raw_record.items() if k != "raw_response"} |
                        {"raw_record": str(raw_path.relative_to(run_root))})
        final_validation, final_raw = validation, parsed
        if not should_retry:
            break
    protocol_valid = bool(final_validation and final_validation.get("decision") == "protocol_valid")
    result: dict[str, Any] = {
        "audit_id": audit_id,
        "candidate_id": candidate.get("candidate_id"),
        "root_id": candidate.get("root_id"),
        "source_difficulty_bucket": candidate.get("source_difficulty_bucket"),
        "question_type": candidate.get("question_type"),
        "status": "completed" if protocol_valid else "protocol_error",
        "protocol_validation": final_validation or {"decision": "protocol_invalid"},
        "blind_audit": final_raw if protocol_valid else None,
        "attempts": attempts,
        "candidate": dict(candidate),
        "protocol_version": PROTOCOL_VERSION,
    }
    if protocol_valid:
        result.update(_compare(candidate, final_raw))
    else:
        result.update({"predicted_answer": [], "gold_match": "not_attempted",
                       "blind_status": "protocol_error", "quality_ok": False})
    atomic_json(result_path, result)
    return result


def _all_candidates(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in manifest.get("candidates") or []]


def _stats(run_root: Path) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    ledger_dir = run_root / "ledger" / "requests"
    for path in ledger_dir.glob("*.json") if ledger_dir.is_dir() else []:
        value = _load(path)
        if value.get("status") == "completed":
            states.append(value)
    usage = [((row.get("response_metadata") or {}).get("usage") or {}) for row in states]
    return {
        "completed_requests": len(states),
        "format_retry_requests": sum(
            int((row.get("request_metadata") or {}).get("attempt") or 0) > 0
            for row in states),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in usage),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in usage),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in usage),
        "ambiguous_requests": sum(
            _load(path).get("status") != "completed"
            for path in ledger_dir.glob("*.json")) if ledger_dir.is_dir() else 0,
    }


def _summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [row for row in results if row.get("status") == "completed"]
    statuses = Counter(str(row.get("blind_status") or "protocol_error") for row in results)
    summary: dict[str, Any] = {
        "candidate_count": len(results),
        "completed_count": len(complete),
        "protocol_or_technical_error_count": len(results) - len(complete),
        "blind_status_counts": dict(sorted(statuses.items())),
        "exact_gold_match_count": sum(row.get("gold_match") == "exact" for row in complete),
        "confirmed_count": sum(row.get("blind_status") == "confirmed" for row in complete),
        "confirmed_with_quality_issue_count": sum(
            row.get("blind_status") == "confirmed_with_quality_issue" for row in complete),
        "disputed_count": sum(row.get("blind_status") == "disputed" for row in complete),
        "ambiguous_count": sum(row.get("blind_status") in {"ambiguous", "unsupported"}
                               for row in complete),
    }
    for field in ("source_difficulty_bucket", "question_type"):
        grouped: dict[str, Any] = {}
        values = sorted({str(row.get(field) or "unknown") for row in results})
        for value in values:
            selected = [row for row in results if str(row.get(field) or "unknown") == value]
            grouped[value] = {
                "candidate_count": len(selected),
                "completed_count": sum(row.get("status") == "completed" for row in selected),
                "confirmed_count": sum(row.get("blind_status") == "confirmed" for row in selected),
                "confirmed_with_quality_issue_count": sum(
                    row.get("blind_status") == "confirmed_with_quality_issue" for row in selected),
                "disputed_count": sum(row.get("blind_status") == "disputed" for row in selected),
                "ambiguous_count": sum(row.get("blind_status") in {"ambiguous", "unsupported"}
                                       for row in selected),
            }
        summary["by_" + field] = grouped
    return summary


def _visible_comparison(run_root: Path, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the first audit only after blind judgments are complete."""
    input_run = Path(str(_manifest(run_root)["input_run"]))
    visible_path = input_run / "audit_results.json"
    if not visible_path.is_file():
        return {"available": False}
    visible = {str(row.get("candidate_id")): row
               for row in (_load(visible_path).get("results") or [])}
    matrix: Counter[tuple[str, str]] = Counter()
    for row in results:
        old = visible.get(str(row.get("candidate_id")))
        if not old:
            continue
        old_class = "valid" if old.get("answer_semantically_valid") else str(old.get("semantic_status") or "unknown")
        new_class = str(row.get("blind_status") or "protocol_error")
        matrix[(old_class, new_class)] += 1
    return {"available": True, "matrix": {
        f"{old}__{new}": count for (old, new), count in sorted(matrix.items())
    }}


def _run_candidates(args: argparse.Namespace, *, phase: str) -> dict[str, Any]:
    manifest = _manifest(args.run_root)
    input_integrity = _verify_input_unchanged(manifest)
    candidates = _all_candidates(manifest)
    # Smoke covers all three structured types when present.
    if phase == "smoke":
        selected: list[dict[str, Any]] = []
        for qtype in ("single_choice", "multi_select", "ordering"):
            row = next((x for x in candidates if x.get("question_type") == qtype), None)
            if row is not None:
                selected.append(row)
        candidates = selected
    ledger = RequestLedger(args.run_root, sha256_file(args.run_root / "manifest.json"), "live")
    client = _backend(args)
    turns_by_root = {
        root_id: _frozen_turns_for_root(args.run_root, root_id)
        for root_id in manifest.get("root_ids") or []
    }
    results = []
    for candidate in candidates:
        audit_id = _safe_id(f"{PROBE_VERSION}:{candidate['candidate_id']}")
        results.append(_audit_candidate(
            candidate=candidate, turns=turns_by_root[str(candidate["root_id"])],
            audit_id=audit_id, run_root=args.run_root, ledger=ledger,
            client=client, total_budget=2 * max(1, len(_all_candidates(manifest)))))
    summary = _summarize(results)
    passed = summary["protocol_or_technical_error_count"] == 0
    report = {
        "probe_version": PROBE_VERSION, "protocol_version": PROTOCOL_VERSION,
        "phase": phase, "status": "pass" if phase == "smoke" and passed else
        ("completed" if passed else "completed_with_errors"),
        "input_run": manifest["input_run"], "run_root": str(args.run_root),
        "summary": summary, "api_statistics": _stats(args.run_root),
        "solver_calls": 0, "input_run_unchanged": input_integrity["unchanged"],
        "production_mainline_modified": False,
    }
    if phase == "batch":
        report["visible_audit_comparison"] = _visible_comparison(args.run_root, results)
        report["readiness"] = {
            "blind_recheck_exported": passed,
            "h4_admission_performed": False,
            "next_required_step": "manual_review_of_discordant_or_ambiguous_candidates",
        }
        atomic_json(args.run_root / "blind_results.json", {
            "probe_version": PROBE_VERSION, "protocol_version": PROTOCOL_VERSION,
            "status": report["status"], "results": results,
        })
        atomic_json(args.run_root / "report.json", report)
        atomic_text(args.run_root / "summary.md", "# H4 Gold-Blind Semantic Recheck\n\n" +
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    else:
        atomic_json(args.run_root / "smoke_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("inventory", "smoke", "batch"), required=True)
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "inventory":
        result = inventory_phase(args)
    else:
        result = _run_candidates(args, phase=args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
