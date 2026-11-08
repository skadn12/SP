"""Fresh semantic audit runner for the frozen H4 K=3 review queues.

This package owns a new output run and never writes to the K=3 probe input,
production QA, or historical runs.  It supports an inventory, a two-item smoke,
and a resumable batch.  The only live calls are independent semantic-auditor
requests; no QA generation or Solver request is made here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_bytes, sha256_file
from ..ledger import AmbiguousRequestError, RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from .protocol import PROTOCOL_VERSION, PROMPT_VERSION, build_prompt, validate_output


PROBE_VERSION = "h4_qa_multiroot_k3_t08_semantic_audit_v1"
DEFAULT_INPUT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_qa_multiroot_k3_t08_probe_v1_live_1"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_qa_multiroot_k3_t08_semantic_audit_v1_live_3"
)
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
AUDIT_MAX_TOKENS = 1536
AUDIT_TEMPERATURE = 0.0
MAX_ATTEMPTS = 2
INPUT_CANDIDATE_FILES = ("hard_candidates.json", "invalid_or_ambiguous_candidates.json")
ALIAS_RESIDUE_PATTERN = re.compile(
    r"\b(?:Neighbor|Person|Speaker|Character)\s+[AB]\b|"
    r"(?:^|[\s,])(?:A|B)(?:[,.!?]|$)"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _root_dir(run_root: Path, root_id: str) -> Path:
    return run_root / "roots" / root_id


def _input_candidate_hashes(input_run: Path) -> dict[str, str]:
    return {
        filename: sha256_file(input_run / filename)
        for filename in INPUT_CANDIDATE_FILES
    }


def _verify_input_unchanged(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if any frozen-source input changed after inventory."""
    input_run = Path(str(manifest["input_run"]))
    expected_candidates = manifest.get("input_candidate_sha256") or {}
    actual_candidates = _input_candidate_hashes(input_run)
    expected_dialogues = manifest.get("source_dialogue_sha256") or {}
    actual_dialogues = {
        str(root_id): sha256_file(
            _root_dir(input_run, str(root_id)) / "expanded_dialogue_clean.json"
        )
        for root_id in manifest.get("root_ids") or []
    }
    actual_report = sha256_file(input_run / "report.json")
    actual_final_audit = sha256_file(input_run / "final_audit.json")
    mismatches: dict[str, Any] = {}
    if expected_candidates != actual_candidates:
        mismatches["candidate_files"] = {
            "expected": expected_candidates, "actual": actual_candidates,
        }
    if expected_dialogues != actual_dialogues:
        mismatches["dialogues"] = {
            "expected": expected_dialogues, "actual": actual_dialogues,
        }
    if manifest.get("input_run_report_sha256") != actual_report:
        mismatches["report"] = {
            "expected": manifest.get("input_run_report_sha256"),
            "actual": actual_report,
        }
    if manifest.get("input_run_final_audit_sha256") != actual_final_audit:
        mismatches["final_audit"] = {
            "expected": manifest.get("input_run_final_audit_sha256"),
            "actual": actual_final_audit,
        }
    if mismatches:
        raise RuntimeError(f"input_run_integrity_changed:{json.dumps(mismatches, sort_keys=True)}")
    return {
        "unchanged": True,
        "candidate_files": actual_candidates,
        "dialogues": actual_dialogues,
        "report": actual_report,
        "final_audit": actual_final_audit,
    }


def _input_candidates(input_run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename, source_bucket in (
        ("hard_candidates.json", "hard_candidate"),
        ("invalid_or_ambiguous_candidates.json", "invalid_or_ambiguous"),
    ):
        payload = _load(input_run / filename)
        for row in payload.get("candidates") or []:
            item = dict(row)
            item["source_difficulty_bucket"] = source_bucket
            rows.append(item)
    seen: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen:
            raise ValueError(f"duplicate_or_missing_candidate_id:{candidate_id}")
        seen.add(candidate_id)
    return rows


def _source_turns(input_run: Path, root_id: str) -> list[dict[str, Any]]:
    path = _root_dir(input_run, root_id) / "expanded_dialogue_clean.json"
    payload = _load(path)
    turns = payload.get("turns") or []
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"empty_dialogue:{root_id}")
    return [dict(row) for row in turns]


def _inventory(input_run: Path) -> dict[str, Any]:
    candidates = _input_candidates(input_run)
    by_root: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    for candidate in candidates:
        root_id = str(candidate.get("root_id") or "")
        if not root_id:
            raise ValueError("candidate_root_id_missing")
        root_source = _root_dir(input_run, root_id) / "expanded_dialogue_clean.json"
        if not root_source.is_file():
            raise FileNotFoundError(root_source)
        turns = _source_turns(input_run, root_id)
        valid_refs = {str(turn.get("turn_ref")) for turn in turns}
        refs = [str(ref) for ref in candidate.get("evidence_refs") or []]
        if not refs or any(ref not in valid_refs for ref in refs):
            raise ValueError(f"evidence_ref_inventory_failure:{candidate.get('candidate_id')}")
        by_root.setdefault(root_id, []).append(candidate)
        source_hashes[root_id] = sha256_file(root_source)
    return {
        "probe_version": PROBE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "created_at": _now(),
        "input_run": str(input_run),
        "input_run_report_sha256": sha256_file(input_run / "report.json"),
        "input_run_final_audit_sha256": sha256_file(input_run / "final_audit.json"),
        "input_candidate_sha256": _input_candidate_hashes(input_run),
        "candidate_count": len(candidates),
        "root_ids": sorted(by_root),
        "candidate_ids": [str(row["candidate_id"]) for row in candidates],
        "source_dialogue_sha256": source_hashes,
        "scope": {"hard_candidate": sum(row["source_difficulty_bucket"] == "hard_candidate" for row in candidates),
                  "invalid_or_ambiguous": sum(row["source_difficulty_bucket"] == "invalid_or_ambiguous" for row in candidates)},
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
    atomic_json(args.run_root / "frozen/candidates.json", inventory["candidates"])
    for root_id in inventory["root_ids"]:
        root_out = _root_dir(args.run_root, root_id)
        root_out.mkdir(parents=True, exist_ok=True)
        turns = _source_turns(args.input_run, root_id)
        atomic_json(root_out / "frozen_dialogue.json", {
            "root_id": root_id,
            "source_sha256": inventory["source_dialogue_sha256"][root_id],
            "turns": turns,
        })
        atomic_json(root_out / "candidates.json", {
            "root_id": root_id,
            "candidates": [row for row in inventory["candidates"]
                           if str(row.get("root_id")) == root_id],
        })
    report = {
        "status": "completed", "phase": "inventory", "offline_only": True,
        "challenger_api_calls": 0, "solver_calls": 0,
        "candidate_count": inventory["candidate_count"],
        "root_count": len(inventory["root_ids"]), "scope": inventory["scope"],
        "run_root": str(args.run_root),
    }
    atomic_json(args.run_root / "inventory_report.json", report)
    atomic_text(args.run_root / "summary.md", "# H4 semantic audit inventory\n\n" +
                json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def _load_manifest(run_root: Path) -> dict[str, Any]:
    manifest = _load(run_root / "manifest.json")
    if manifest.get("probe_version") != PROBE_VERSION:
        raise ValueError("semantic_audit_manifest_version_mismatch")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("semantic_audit_protocol_version_mismatch")
    return manifest


def _candidate_result_path(run_root: Path, candidate_id: str) -> Path:
    root_id = candidate_id.split(":", 1)[0]
    return _root_dir(run_root, root_id) / "results" / f"{_safe_id(candidate_id)}.json"


def _reconcile_existing_result(existing: Mapping[str, Any], *, candidate: Mapping[str, Any],
                               refs: Sequence[str], static_review: Mapping[str, Any],
                               run_root: Path) -> dict[str, Any] | None:
    """Re-validate a stored raw response after a validator-only fix.

    This path never calls the API.  It lets a protocol-error result recover
    when the raw auditor object already contains all semantic fields but only
    violated a non-semantic formatting rule.
    """
    if existing.get("status") != "protocol_error":
        return None
    attempts = existing.get("attempts") or []
    for attempt in reversed(attempts):
        raw_record_ref = attempt.get("raw_record")
        if not raw_record_ref:
            continue
        raw_record_path = run_root / str(raw_record_ref)
        if not raw_record_path.is_file():
            continue
        raw_record = _load(raw_record_path)
        raw_text = raw_record.get("raw_response")
        if not isinstance(raw_text, str):
            continue
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            continue
        validation = validate_output(parsed, expected_evidence_refs=refs)
        if validation.get("decision") != "protocol_valid":
            continue
        existing = dict(existing)
        existing.update({
            "status": "completed",
            "protocol_validation": validation,
            "audit": validation.get("audit"),
            "semantic_status": validation.get("semantic_status"),
            "answer_supported": bool(validation.get("answer_supported")),
            "answer_semantically_valid": bool(validation.get("answer_semantically_valid")),
            "training_ready": bool(validation.get("training_ready")) and
                not bool(static_review.get("training_blockers")),
            "static_review": dict(static_review),
            "protocol_reconciled_without_api": True,
        })
        atomic_json(_candidate_result_path(run_root, str(candidate["candidate_id"])), existing)
        return existing
    return None


def _evidence_turns(candidate: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_ref = {str(turn.get("turn_ref")): dict(turn) for turn in turns}
    return [by_ref[str(ref)] for ref in candidate.get("evidence_refs") or [] if str(ref) in by_ref]


def _static_review_flags(candidate: Mapping[str, Any],
                         turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Record deterministic hygiene risks without making semantic claims."""
    flat_refs = [str(ref) for ref in candidate.get("evidence_refs") or []]
    flat_set = set(flat_refs)
    groups = candidate.get("evidence_groups") or {}
    if not isinstance(groups, Mapping):
        groups = {}
    group_refs = [
        str(ref) for values in groups.values() if isinstance(values, list)
        for ref in values
    ]
    group_set = set(group_refs)
    by_ref = {str(turn.get("turn_ref")): dict(turn) for turn in turns}
    evidence = [by_ref[ref] for ref in flat_refs if ref in by_ref]
    evidence_alias_refs = [
        str(turn.get("turn_ref")) for turn in evidence
        if ALIAS_RESIDUE_PATTERN.search(str(turn.get("text") or ""))
    ]
    all_alias_refs = [
        str(turn.get("turn_ref")) for turn in turns
        if ALIAS_RESIDUE_PATTERN.search(str(turn.get("text") or ""))
    ]
    flags: list[str] = []
    if group_set != flat_set and groups:
        flags.append("evidence_group_flat_mismatch")
    if len(flat_refs) != len(flat_set):
        flags.append("duplicate_evidence_ref")
    if evidence_alias_refs:
        flags.append("alias_residue_in_evidence")
    elif all_alias_refs:
        flags.append("alias_residue_in_context")
    # These are review blockers for a training export, not automatic semantic
    # rejection. An auditor may still find the question correct in context.
    blockers = [flag for flag in flags if flag in {
        "evidence_group_flat_mismatch", "duplicate_evidence_ref",
        "alias_residue_in_evidence",
    }]
    return {
        "flags": flags,
        "training_blockers": blockers,
        "flat_evidence_refs": flat_refs,
        "group_evidence_refs_union": sorted(group_set),
        "group_only_refs": sorted(group_set - flat_set),
        "flat_only_refs": sorted(flat_set - group_set),
        "alias_residue_evidence_refs": evidence_alias_refs,
        "alias_residue_context_refs": all_alias_refs,
    }


def _backend(args: argparse.Namespace) -> OpenAICompatibleChallengerClient:
    config = load_challenger_config(args.api_config)
    from dataclasses import replace
    return OpenAICompatibleChallengerClient(replace(
        config, timeout_seconds=max(180, int(config.timeout_seconds)),
        questioner_max_tokens=AUDIT_MAX_TOKENS,
        temperature=AUDIT_TEMPERATURE, enable_thinking=False))


def _audit_candidate(*, candidate: Mapping[str, Any], turns: Sequence[Mapping[str, Any]],
                     run_root: Path, ledger: RequestLedger,
                     client: OpenAICompatibleChallengerClient,
                     total_budget: int) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    result_path = _candidate_result_path(run_root, candidate_id)
    if result_path.is_file():
        existing = _load(result_path)
        # A result from another protocol/manifest must never be silently
        # reused after a prompt or frozen input changes.
        if (existing.get("protocol_version") != PROTOCOL_VERSION or
                existing.get("candidate", {}).get("candidate_id") != candidate_id):
            raise RuntimeError(f"stale_semantic_audit_result:{candidate_id}")
        static_review = _static_review_flags(candidate, turns)
        reconciled = _reconcile_existing_result(
            existing, candidate=candidate,
            refs=[str(ref) for ref in candidate.get("evidence_refs") or []],
            static_review=static_review, run_root=run_root,
        )
        if reconciled is not None:
            return reconciled
        if "static_review" not in existing:
            existing["static_review"] = static_review
            if static_review["training_blockers"]:
                existing["training_ready"] = False
            atomic_json(result_path, existing)
        return existing
    refs = [str(ref) for ref in candidate.get("evidence_refs") or []]
    static_review = _static_review_flags(candidate, turns)
    prompt_candidate = {**dict(candidate), "audit_metadata": static_review}
    prompt = build_prompt(candidate=prompt_candidate, turns=turns,
                          evidence_turns=_evidence_turns(candidate, turns))
    logical_id = f"semantic-audit-{_safe_id(PROTOCOL_VERSION + ':' + candidate_id)}"
    attempts: list[dict[str, Any]] = []
    final_validation: dict[str, Any] | None = None
    final_raw: Any = None
    for attempt in range(MAX_ATTEMPTS):
        retry = attempt > 0
        rendered_prompt = prompt
        if retry:
            rendered_prompt += (
                "\n\nYour previous response failed the JSON contract. Regenerate the same audit "
                "with exactly the required fields; do not change the semantic judgment."
            )
        request_id = f"{logical_id}-a{attempt:02d}"
        prompt_hash = sha256_bytes(rendered_prompt.encode("utf-8"))
        prompt_path = run_root / "raw" / "prompts" / f"{request_id}.json"
        atomic_json(prompt_path, {
            "request_id": request_id, "candidate_id": candidate_id,
            "attempt": attempt, "retry": retry,
            "protocol_version": PROTOCOL_VERSION,
            "prompt_sha256": prompt_hash, "prompt": rendered_prompt,
        })
        try:
            response = ledger.invoke(
                request_id=request_id, intent="semantic_audit", attempt=attempt + 1,
                prompt_sha256=prompt_hash,
                call=lambda rendered_prompt=rendered_prompt: client.complete(
                    prompt=rendered_prompt, max_tokens=AUDIT_MAX_TOKENS,
                    json_object=True),
                budget_class="semantic_audit", budget_limit=total_budget,
                total_budget_limit=total_budget,
                request_metadata={
                    "candidate_id": candidate_id, "attempt": attempt,
                    "retry": retry, "protocol_version": PROTOCOL_VERSION,
                },
                backend_identity={
                    "backend": "openai_compatible_semantic_auditor",
                    "model_identifier": client.config.model,
                    "temperature": client.config.temperature,
                    "max_tokens": AUDIT_MAX_TOKENS,
                    "response_format": "json_object",
                },
            )
        except AmbiguousRequestError as error:
            return {
                "candidate_id": candidate_id, "root_id": candidate.get("root_id"),
                "status": "technical_error", "technical_error": str(error),
                "attempts": attempts,
            }
        except Exception as error:
            return {
                "candidate_id": candidate_id, "root_id": candidate.get("root_id"),
                "status": "technical_error",
                "technical_error": f"{type(error).__name__}:{error}",
                "attempts": attempts,
            }
        state = ledger.read(request_id) or {}
        raw_text = response.raw_text
        try:
            parsed = json.loads(raw_text)
            validation = validate_output(parsed, expected_evidence_refs=refs)
            parse_status = "json_parsed"
        except json.JSONDecodeError as error:
            parsed = None
            validation = {"decision": "protocol_invalid", "errors": ["json_invalid",
                            str(error)], "audit": None}
            parse_status = "json_invalid"
        should_retry = attempt == 0 and validation.get("decision") == "protocol_invalid"
        ledger.record_decision(request_id, "retry_format_only" if should_retry else
                               str(validation.get("decision")))
        raw_record = {
            "request_id": request_id, "candidate_id": candidate_id,
            "attempt": attempt, "parse_status": parse_status,
            "validation": validation, "raw_response": raw_text,
            "response_metadata": response.metadata,
            "raw_response_sha256": state.get("raw_response_sha256"),
        }
        raw_path = run_root / "raw" / "auditor" / f"{request_id}.json"
        atomic_json(raw_path, raw_record)
        attempts.append({key: value for key, value in raw_record.items()
                         if key != "raw_response"} | {"raw_record": str(raw_path.relative_to(run_root))})
        final_validation = validation
        final_raw = parsed
        if not should_retry:
            break
    protocol_valid = bool(final_validation and
                          final_validation.get("decision") == "protocol_valid")
    semantic_status = (final_validation or {}).get("semantic_status")
    answer_supported = bool((final_validation or {}).get("answer_supported"))
    answer_valid = bool((final_validation or {}).get("answer_semantically_valid"))
    training_ready = bool((final_validation or {}).get("training_ready"))
    if static_review["training_blockers"]:
        training_ready = False
    result = {
        "candidate_id": candidate_id,
        "root_id": candidate.get("root_id"),
        "intent_id": candidate.get("intent_id"),
        "question_type": candidate.get("question_type"),
        "generation_source": candidate.get("generation_source"),
        "source_difficulty_bucket": candidate.get("source_difficulty_bucket"),
        "status": "completed" if protocol_valid else "protocol_error",
        "protocol_validation": final_validation or {"decision": "protocol_invalid"},
        "audit": final_raw if final_validation and
        final_validation.get("decision") == "protocol_valid" else None,
        "semantic_status": semantic_status,
        "answer_supported": answer_supported,
        "answer_semantically_valid": answer_valid,
        "training_ready": training_ready,
        "static_review": static_review,
        "attempts": attempts,
        "candidate": dict(candidate),
        "protocol_version": PROTOCOL_VERSION,
    }
    atomic_json(result_path, result)
    return result


def _all_candidates(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in manifest.get("candidates") or []]


def _turns_for_root(input_run: Path, root_id: str) -> list[dict[str, Any]]:
    return _source_turns(input_run, root_id)


def _frozen_turns_for_root(run_root: Path, root_id: str) -> list[dict[str, Any]]:
    payload = _load(_root_dir(run_root, root_id) / "frozen_dialogue.json")
    turns = payload.get("turns") or []
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"empty_frozen_dialogue:{root_id}")
    return [dict(row) for row in turns]


def _stats(run_root: Path) -> dict[str, Any]:
    states = []
    ledger_dir = run_root / "ledger" / "requests"
    if ledger_dir.is_dir():
        for path in ledger_dir.glob("*.json"):
            value = _load(path)
            if value.get("status") == "completed":
                states.append(value)
    usage = [((row.get("response_metadata") or {}).get("usage") or {}) for row in states]
    return {
        "completed_requests": len(states),
        "format_retry_requests": sum(
            1 for row in states
            if int((row.get("request_metadata") or {}).get("attempt") or 0) > 0
        ),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in usage),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in usage),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in usage),
        "ambiguous_requests": sum(1 for path in ledger_dir.glob("*.json")
                                   if _load(path).get("status") != "completed")
        if ledger_dir.is_dir() else 0,
    }


def _summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in results if row.get("status") == "completed"]
    counts = Counter(str(row.get("semantic_status") or "protocol_error") for row in results)
    answer_supported = [row for row in completed if row.get("answer_supported")]
    answer_valid = [row for row in completed if row.get("answer_semantically_valid")]
    training_ready = [row for row in completed if row.get("training_ready")]
    static_flag_counts = Counter(
        str(flag)
        for row in results
        for flag in ((row.get("static_review") or {}).get("flags") or [])
    )
    training_blocked = [
        row for row in completed
        if row.get("answer_semantically_valid") and not row.get("training_ready")
    ]
    by_source: dict[str, Any] = {}
    by_type: dict[str, Any] = {}
    by_bucket: dict[str, Any] = {}
    for field, target in (("generation_source", by_source),
                          ("question_type", by_type),
                          ("source_difficulty_bucket", by_bucket)):
        for value in sorted({str(row.get(field) or "unknown") for row in results}):
            selected = [row for row in results if str(row.get(field) or "unknown") == value]
            supported = sum(bool(row.get("answer_supported")) for row in selected)
            valid = sum(bool(row.get("answer_semantically_valid")) for row in selected)
            ready = sum(bool(row.get("training_ready")) for row in selected)
            target[value] = {
                "candidate_count": len(selected),
                "answer_supported_count": supported,
                "answer_semantically_valid_count": valid,
                "training_ready_count": ready,
                "semantic_valid_rate": valid / len(selected) if selected else None,
                "training_ready_rate": ready / len(selected) if selected else None,
                "semantic_status_counts": dict(sorted(Counter(
                    str(row.get("semantic_status") or "protocol_error") for row in selected).items())),
            }
    hard_results = [row for row in results if row.get("source_difficulty_bucket") == "hard_candidate"]
    invalid_results = [row for row in results if row.get("source_difficulty_bucket") == "invalid_or_ambiguous"]
    return {
        "candidate_count": len(results),
        "completed_count": len(completed),
        "protocol_or_technical_error_count": len(results) - len(completed),
        "answer_supported_count": len(answer_supported),
        "answer_semantically_valid_count": len(answer_valid),
        "training_ready_count": len(training_ready),
        "semantically_valid_but_training_blocked_count": len(training_blocked),
        "static_review_flag_counts": dict(sorted(static_flag_counts.items())),
        "semantic_status_counts": dict(sorted(counts.items())),
        "hard_candidate_semantically_valid_count": sum(bool(row.get("answer_semantically_valid")) for row in hard_results),
        "hard_candidate_training_ready_count": sum(bool(row.get("training_ready")) for row in hard_results),
        "full_context_zero_answer_valid_count": sum(
            bool(row.get("answer_semantically_valid")) for row in invalid_results),
        "by_generation_source": by_source,
        "by_question_type": by_type,
        "by_source_difficulty_bucket": by_bucket,
        "interpretation": (
            "Answer support, valid QA semantics, and training readiness are separate. "
            "Semantic validity requires a supported gold plus determinate, clean, unique QA wording. "
            "Training-ready additionally requires sufficient submitted evidence and no static blockers."
        ),
    }


def _write_exports(run_root: Path, results: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    valid_hard = [row for row in results if row.get("source_difficulty_bucket") == "hard_candidate"
                  and row.get("answer_semantically_valid")]
    ready_hard = [row for row in valid_hard if row.get("training_ready")]
    invalid = [row for row in results if row.get("source_difficulty_bucket") == "invalid_or_ambiguous"]
    rejected_hard = [row for row in results if row.get("source_difficulty_bucket") == "hard_candidate"
                     and not row.get("answer_semantically_valid")]
    atomic_json(run_root / "semantically_valid_hard_candidates.json", {
        "candidate_count": len(valid_hard), "candidates": valid_hard,
    })
    atomic_json(run_root / "training_ready_hard_candidates.json", {
        "candidate_count": len(ready_hard), "candidates": ready_hard,
    })
    atomic_json(run_root / "rejected_or_uncertain_hard_candidates.json", {
        "candidate_count": len(rejected_hard), "candidates": rejected_hard,
    })
    atomic_json(run_root / "full_context_zero_audit.json", {
        "candidate_count": len(invalid), "candidates": invalid,
    })
    lines = [
        "# H4 K=3 Semantic Audit",
        "",
        f"Candidates audited: **{summary['candidate_count']}**",
        f"Gold answers supported: **{summary['answer_supported_count']}**",
        f"Semantically valid: **{summary['answer_semantically_valid_count']}**",
        f"Training-ready: **{summary['training_ready_count']}**",
        f"Hard candidates semantically valid: **{summary['hard_candidate_semantically_valid_count']}**",
        f"Hard candidates training-ready: **{summary['hard_candidate_training_ready_count']}**",
        f"Full-context-zero candidates judged answer-valid: **{summary['full_context_zero_answer_valid_count']}**",
        f"Static review flags: `{json.dumps(summary['static_review_flag_counts'], sort_keys=True)}`",
        "",
        "This audit validates semantics against the complete frozen dialogue. It does not alter the input run or promote data into the production pipeline.",
    ]
    atomic_text(run_root / "summary.md", "\n".join(lines) + "\n")


def smoke_phase(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.run_root)
    input_integrity = _verify_input_unchanged(manifest)
    candidates = _all_candidates(manifest)
    # Keep one item from each queue; this tests both a presumed-hard and a
    # full-context-zero item before spending the batch budget.
    selected = [next((row for row in candidates
                      if row.get("source_difficulty_bucket") == bucket), None)
                for bucket in ("hard_candidate", "invalid_or_ambiguous")]
    selected = [row for row in selected if row is not None]
    if not selected:
        raise RuntimeError("semantic_audit_smoke_selection_empty")
    ledger = RequestLedger(args.run_root, sha256_file(args.run_root / "manifest.json"), "live")
    client = _backend(args)
    results = []
    for candidate in selected:
        turns = _frozen_turns_for_root(args.run_root, str(candidate["root_id"]))
        results.append(_audit_candidate(candidate=candidate, turns=turns,
                                        run_root=args.run_root, ledger=ledger,
                                        client=client, total_budget=2 * len(candidates)))
    passed = all(row.get("status") == "completed" for row in results)
    report = {
        "status": "pass" if passed else "fail", "phase": "smoke",
        "candidate_ids": [row["candidate_id"] for row in selected],
        "results": [{"candidate_id": row.get("candidate_id"),
                     "status": row.get("status"),
                     "semantic_status": row.get("semantic_status"),
                     "protocol_decision": (row.get("protocol_validation") or {}).get("decision")}
                    for row in results],
        "challenger_api_calls": _stats(args.run_root)["completed_requests"],
        "solver_calls": 0,
        "input_run_unchanged": input_integrity["unchanged"],
    }
    atomic_json(args.run_root / "smoke_report.json", report)
    return report


def batch_phase(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.run_root)
    input_integrity = _verify_input_unchanged(manifest)
    smoke = _load(args.run_root / "smoke_report.json")
    if smoke.get("status") != "pass":
        raise RuntimeError("semantic_audit_smoke_gate_failed")
    candidates = _all_candidates(manifest)
    ledger = RequestLedger(args.run_root, sha256_file(args.run_root / "manifest.json"), "live")
    client = _backend(args)
    turns_by_root = {root_id: _frozen_turns_for_root(args.run_root, root_id)
                     for root_id in manifest.get("root_ids") or []}
    results = []
    for candidate in candidates:
        results.append(_audit_candidate(candidate=candidate,
                                        turns=turns_by_root[str(candidate["root_id"])],
                                        run_root=args.run_root, ledger=ledger,
                                        client=client, total_budget=2 * len(candidates)))
    input_integrity = _verify_input_unchanged(manifest)
    summary = _summarize(results)
    _write_exports(args.run_root, results, summary)
    stats = _stats(args.run_root)
    batch_complete = summary["protocol_or_technical_error_count"] == 0
    report = {
        "probe_version": PROBE_VERSION, "protocol_version": PROTOCOL_VERSION,
        "status": "completed" if batch_complete else "completed_with_errors",
        "input_run": manifest["input_run"],
        "run_root": str(args.run_root), "summary": summary,
        "api_statistics": stats, "solver_calls": 0,
        "semantic_grounding_reviewed": batch_complete,
        "production_mainline_modified": False,
        "input_run_modified": not input_integrity["unchanged"],
        "input_run_unchanged": input_integrity["unchanged"],
        "readiness": {
            "training_ready_candidates_exported": batch_complete,
            "production_admission_performed": False,
            "next_required_step": (
                "manual_review_or_independent_recheck_of_semantically_valid_hard_candidates"
                if batch_complete else "resolve_protocol_or_technical_errors_then_resume"
            ),
        },
    }
    atomic_json(args.run_root / "audit_results.json", {
        "probe_version": PROBE_VERSION, "protocol_version": PROTOCOL_VERSION,
        "status": report["status"], "results": results,
    })
    atomic_json(args.run_root / "report.json", report)
    _write_exports(args.run_root, results, summary)
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
    elif args.phase == "smoke":
        result = smoke_phase(args)
    else:
        result = batch_phase(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
