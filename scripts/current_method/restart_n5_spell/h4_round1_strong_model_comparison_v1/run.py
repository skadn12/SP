"""Offline preflight and blind mirrored strong-model evaluation.

The preflight is intentionally independent of QA, Solver, and semantic-auditor
code. Historical expansion artifacts are read only. API credentials are read
from a local mode-600 env file and are never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core import canonical_json, sha256_bytes, sha256_file
from . import PROMPT_VERSION, SEED_SCHEDULE, VERSION


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
R01_RUN = RUN_BASE / "h4_unexpanded122_memory_event_k3_v1"
R02_PREFIX = "h4_round1_cumulative_continuation_fresh_mining_v1_live_"
AUTHORITY_MANIFEST = RUN_BASE / f"{R02_PREFIX}47" / "round1_root_manifest.jsonl"
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round1_strong_model_comparison_v1_live_1"
DEFAULT_CONFIG = Path("/root/.config/gxy-selfplay/round1_strong_judge.env")
EXPECTED_CONFIG = {
    "provider": "TJU AI",
    "model": "tju-llm",
    "base_url": "https://ai.tju.edu.cn/api/v3",
    "api_key_env": "ROUND1_STRONG_JUDGE_API_KEY",
}
VALID_COMPLETED_STATES = {
    "COMPLETED", "COMPLETED_REUSED_PRIOR", "COMPLETED_REUSED_SMOKE",
}
CRITERIA = (
    "factual_consistency",
    "state_transition_quality",
    "incremental_novelty",
    "redundancy_control",
    "participant_coherence",
    "qa_readiness",
)
EXPECTED_ROOT_COUNT = 70
TEMPERATURE = 0.0
MAX_TOKENS = 2048
TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (5, 20)
CONFIDENCE_THRESHOLD = 0.70
_ROUND_REF = re.compile(r"^R(\d+):S\d+:T\d+$")
_LOCAL_REF = re.compile(r"^[AB]:T\d{3}$")


class PreflightError(RuntimeError):
    """Raised when an offline invariant prevents model requests."""


class ResponseValidationError(ValueError):
    """Raised when a response is not strict evaluator JSON."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        sha256_file(path) + "\n", encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2,
                                 sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_text(path, "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows))


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise PreflightError(f"file_or_sidecar_missing:{path}")
    words = sidecar.read_text(encoding="utf-8").strip().split()
    if not words or words[0] != sha256_file(path):
        raise PreflightError(f"sidecar_mismatch:{path}")


def _canonical_dialogue_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("complete_dialogue_sha256", None)
    return sha256_bytes(canonical_json(body).encode("utf-8"))


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise PreflightError(f"strong_model_config_missing:{path}")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PreflightError(f"strong_model_config_permissions_not_private:{oct(mode)}")
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise PreflightError(f"strong_model_config_invalid_line:{number}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if ((value.startswith("'") and value.endswith("'")) or
                (value.startswith('"') and value.endswith('"'))):
            value = value[1:-1]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise PreflightError(f"strong_model_config_invalid_key:{number}")
        result[key] = value
    for key, expected in EXPECTED_CONFIG.items():
        if result.get(key) != expected:
            raise PreflightError(
                f"strong_model_config_mismatch:{key}:expected={expected}:actual={result.get(key)}")
    key_name = result["api_key_env"]
    api_key = os.environ.get(key_name) or result.get(key_name, "")
    placeholders = {"", "密码", "<YOUR_API_KEY>", "YOUR_API_KEY", "你的完整Key"}
    if api_key in placeholders:
        raise PreflightError(f"strong_model_api_key_missing:{key_name}")
    result["_api_key"] = api_key
    result["_credential_source"] = (
        "environment" if os.environ.get(key_name) else "private_env_file")
    return result


def _identity(config: Mapping[str, str], config_path: Path) -> dict[str, Any]:
    return {
        "identity_version": VERSION + "_identity_v1",
        "provider": config["provider"],
        "model": config["model"],
        "base_url": config["base_url"],
        "chat_completions_url": config["base_url"].rstrip("/") + "/chat/completions",
        "api_key_env": config["api_key_env"],
        "api_key_present": True,
        "credential_source": config["_credential_source"],
        "credential_file": str(config_path.resolve()),
        "credential_serialized": False,
        "independence": {
            "original_expansion_provider": "SiliconFlow",
            "original_expansion_model": "Qwen/Qwen3-14B",
            "same_provider": False,
            "same_model": False,
        },
        "generation": {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_attempts_total": MAX_ATTEMPTS,
            "retry_backoff_seconds": list(RETRY_BACKOFF_SECONDS),
            "seed_required": True,
        },
        "prompt_version": PROMPT_VERSION,
        "seed_schedule": SEED_SCHEDULE,
    }


def _contract() -> dict[str, Any]:
    return {
        "contract_version": VERSION + "_contract_v1",
        "prompt_version": PROMPT_VERSION,
        "seed_schedule": {
            "version": SEED_SCHEDULE,
            "formula": "first_31_bits(SHA256(prompt_version + r01_dialogue_sha256 + r02_dialogue_sha256))",
            "encoding": "UTF-8",
            "same_seed_for_mirrors": True,
            "provider_seed_support_required": True,
        },
        "comparison_interpretation": {
            "candidate_1": "R01 continuation quality relative to Parent",
            "candidate_2": "R02 continuation quality relative to Parent+R01",
            "same_start_causal_ab_experiment": False,
        },
        "scores": {
            "criteria": list(CRITERIA),
            "range": {"minimum": 0, "maximum": 5, "integer_only": True},
            "anchors": {
                "5": "clearly excellent; no substantive issue",
                "4": "good; minor issues only",
                "3": "usable with clear limitations",
                "2": "serious degradation",
                "1": "mostly unusable",
                "0": "confirmed history violation, structural collapse, or impossible to evaluate",
            },
            "total_score": "programmatic sum of six criteria; range 0-30",
            "mean_score": "programmatic total_score / 6",
            "judge_weighting_allowed": False,
        },
        "confidence": {
            "range": [0.0, 1.0],
            "meaning": "textual support for the pairwise judgment, not text quality",
            "minimum_for_stable": CONFIDENCE_THRESHOLD,
        },
        "uncertain_if_any": [
            "either mirror explicitly returns UNCERTAIN",
            "either mirror confidence is below 0.70",
            "remapped mirror pairwise preferences disagree",
            "either mirror remains invalid after all retries",
            "critical dialogue content or temporal/participant relation is indeterminate",
        ],
        "score_instability_if_any": [
            "any remapped criterion differs by more than 1",
            "either remapped candidate total differs by more than 6",
        ],
        "blinding": {
            "forbidden": [
                "root_id", "split", "R01/R02 labels", "file paths", "SHA values",
                "generation source", "Memorax namespace/session_id",
                "old auditor/gate labels", "Solver labels", "QA Gold",
                "training labels", "manual blacklist",
            ],
            "visible": [
                "speaker", "ordered dialogue text", "preceding context",
                "new continuation", "local T001-style references",
            ],
            "timestamps": "only canonical dialogue facts, if present",
        },
        "mirroring": {
            "initial_assignment": "balanced 35/35 after stable hash ordering",
            "second_request": "strict A/B swap",
            "ordinary_numeric_difference_causes_uncertain": False,
        },
        "response": {
            "strict_json_only": True,
            "manual_repair": False,
            "string_extraction": False,
            "retry_request_body_identical": True,
        },
        "prohibited_actions": [
            "QA", "Solver", "semantic auditor", "training", "data modification",
            "dialogue rewrite", "training manifest construction", "Dev/Test access",
            "automatic deletion or blacklist",
        ],
        "recommendation": {
            "READY": "program preflight passes, 70 roots have stable conclusions, and no blocking issue exists",
            "READY_WITH_WARNINGS": "contract passes with a small number of low scores or uncertain roots",
            "REVIEW_REQUIRED": "factual consistency degrades materially, uncertainty exceeds 10%, or lineage/SHA/Memory problems exist",
        },
    }


def paired_seed(r01_sha: str, r02_sha: str) -> int:
    digest = hashlib.sha256(
        (PROMPT_VERSION + r01_sha + r02_sha).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") >> 1


def _state_candidates(authority_ids: set[str]) -> dict[str, list[tuple[int, Path, dict[str, Any]]]]:
    result: dict[str, list[tuple[int, Path, dict[str, Any]]]] = defaultdict(list)
    for live_index in range(2, 48):
        live_root = RUN_BASE / f"{R02_PREFIX}{live_index}"
        if not live_root.is_dir():
            raise PreflightError(f"r02_live_directory_missing:{live_index}")
        for state_path in sorted(live_root.glob("shards/*/roots/*/root_state.json")):
            state = _load(state_path)
            root_id = str(state.get("root_id") or "")
            if root_id and root_id not in authority_ids:
                raise PreflightError(f"r02_state_root_outside_authority:{root_id}:{state_path}")
            if state_path.parent.name != root_id:
                raise PreflightError(f"r02_state_path_root_mismatch:{state_path}")
            result[root_id].append((live_index, state_path, state))
    return result


def _assert_turns(turns: Any, root_id: str, label: str) -> list[dict[str, Any]]:
    if not isinstance(turns, list) or not turns:
        raise PreflightError(f"{label}_turns_missing:{root_id}")
    refs = []
    normalized = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise PreflightError(f"{label}_turn_not_object:{root_id}")
        ref = str(turn.get("turn_ref") or "")
        speaker = str(turn.get("speaker") or "").strip()
        text = str(turn.get("text") or "").strip()
        if not ref or not speaker or not text:
            raise PreflightError(f"{label}_turn_invalid:{root_id}")
        refs.append(ref)
        normalized.append(dict(turn))
    if len(refs) != len(set(refs)):
        raise PreflightError(f"{label}_turn_ref_duplicate:{root_id}")
    return normalized


def _validate_root_candidate(
    authority: Mapping[str, Any], live_index: int, state_path: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    root_id = str(authority["root_id"])
    _verify_sidecar(state_path)
    if state.get("status") not in VALID_COMPLETED_STATES:
        raise PreflightError(f"state_not_completed:{root_id}:{state.get('status')}")
    if int(state.get("round_index") or 0) != 2:
        raise PreflightError(f"state_round_index_not_2:{root_id}")
    if state.get("root_id") != root_id:
        raise PreflightError(f"state_root_mismatch:{root_id}")
    if state.get("memory_write_verified") is not True:
        raise PreflightError(f"state_memory_write_not_verified:{root_id}")
    if state.get("memory_visibility_verified") is not True:
        raise PreflightError(f"state_memory_visibility_not_verified:{root_id}")
    if int(state.get("qa_calls") or 0) != 0 or int(state.get("solver_calls") or 0) != 0:
        raise PreflightError(f"state_qa_or_solver_call_nonzero:{root_id}")
    expected_session = f"{root_id}:memorax:R02"
    if state.get("r02_session_id") != expected_session:
        raise PreflightError(f"state_r02_session_mismatch:{root_id}")

    r01_path = Path(str(authority.get("canonical_dialogue_path") or ""))
    if not r01_path.is_file():
        raise PreflightError(f"r01_dialogue_missing:{root_id}")
    r01_file_sha = sha256_file(r01_path)
    if r01_file_sha != authority.get("canonical_dialogue_sha256"):
        raise PreflightError(f"r01_dialogue_file_sha_drift:{root_id}")
    if (Path(str(state.get("input_path") or "")).resolve() != r01_path.resolve() or
            state.get("input_sha256") != r01_file_sha):
        raise PreflightError(f"r02_input_sha_or_path_drift:{root_id}")
    r01 = _load(r01_path)
    if r01.get("root_id") != root_id:
        raise PreflightError(f"r01_root_mismatch:{root_id}")
    if r01.get("append_mode") != "parent_tail_only":
        raise PreflightError(f"r01_not_cumulative_parent_tail:{root_id}")
    if _canonical_dialogue_hash(r01) != r01.get("complete_dialogue_sha256"):
        raise PreflightError(f"r01_internal_sha_invalid:{root_id}")
    if r01.get("complete_dialogue_sha256") != authority.get("complete_dialogue_sha256"):
        raise PreflightError(f"r01_internal_sha_drift:{root_id}")
    r01_turns = _assert_turns(r01.get("turns"), root_id, "r01")
    parent_count = int(r01.get("parent_turn_count") or 0)
    if not 0 < parent_count < len(r01_turns):
        raise PreflightError(f"r01_parent_boundary_invalid:{root_id}")
    parent_turns = r01_turns[:parent_count]
    round1_turns = r01_turns[parent_count:]
    if (len(r01_turns) != int(r01.get("turn_count") or -1) or
            len(round1_turns) != int(r01.get("child_turn_count") or -1) or
            any(not str(t["turn_ref"]).startswith("R01:") for t in round1_turns)):
        raise PreflightError(f"r01_cumulative_counts_or_refs_invalid:{root_id}")

    attempt_root = Path(str(state.get("attempt_root") or ""))
    if not attempt_root.is_dir():
        raise PreflightError(f"r02_attempt_root_missing:{root_id}")
    expected = {
        "report": attempt_root / "reports" / "MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json",
        "intent": attempt_root / "memory" / "child_add_intent.json",
        "completion": attempt_root / "memory" / "child_add_completion.json",
        "dialogue": attempt_root / "promotion" / "promotable_expanded_dialogue.json",
        "validation": attempt_root / "promotion" / "promotable_expanded_dialogue_validation.json",
    }
    for label, path in expected.items():
        if not path.is_file():
            raise PreflightError(f"r02_{label}_missing:{root_id}:{path}")
    if Path(str(state.get("report_path") or "")).resolve() != expected["report"].resolve():
        raise PreflightError(f"r02_report_path_mismatch:{root_id}")

    report = _load(expected["report"])
    report_checks = {
        "status": report.get("status") == "completed",
        "root_id": report.get("root_id") == root_id,
        "round_2": report.get("round_2") is True,
        "parent_add_skipped": report.get("parent_add_skipped") is True,
        "existing_namespace_reused": report.get("existing_namespace_reused") is True,
        "child_add_calls": int(report.get("child_add_calls") or 0) == 1,
        "memory_write_verified": report.get("memory_write_verified") is True,
        "memory_visibility_verified": report.get("memory_visibility_verified") is True,
        "qa_calls": int(report.get("qa_calls") or 0) == 0,
        "solver_calls": int(report.get("solver_calls") or 0) == 0,
    }
    failed = [name for name, value in report_checks.items() if not value]
    if failed:
        raise PreflightError(f"r02_report_checks_failed:{root_id}:{','.join(failed)}")

    prior_user_id = str(authority.get("prior_user_id") or "")
    if (not prior_user_id or state.get("prior_user_id") != prior_user_id or
            report.get("user_id") != prior_user_id):
        raise PreflightError(f"r02_user_id_reuse_mismatch:{root_id}")
    intent = _load(expected["intent"])
    completion = _load(expected["completion"])
    r02 = _load(expected["dialogue"])
    validation = _load(expected["validation"])
    if (intent.get("root_id") != root_id or intent.get("round_index") != 2 or
            intent.get("user_id") != prior_user_id or
            intent.get("session_id") != expected_session):
        raise PreflightError(f"r02_child_intent_identity_invalid:{root_id}")
    if (completion.get("final_status") != "success" or
            completion.get("verified_complete") is not True or
            completion.get("write_state", {}).get("memory_write_verified") is not True):
        raise PreflightError(f"r02_child_completion_invalid:{root_id}")
    if validation.get("valid") is not True or validation.get("errors") not in ([], None):
        raise PreflightError(f"r02_promotion_validation_invalid:{root_id}")
    if (r02.get("root_id") != root_id or
            r02.get("append_mode") != "multi_round_parent_tail_only" or
            int(r02.get("round_count") or 0) != 2 or
            r02.get("latest_round_boundary", {}).get("round_index") != 2 or
            r02.get("parent_dialogue_sha256") != r01.get("parent_dialogue_sha256") or
            int(r02.get("parent_turn_count") or -1) != parent_count):
        raise PreflightError(f"r02_cumulative_metadata_invalid:{root_id}")
    r02_turns_all = _assert_turns(r02.get("turns"), root_id, "r02")
    if r02_turns_all[:len(r01_turns)] != r01_turns:
        raise PreflightError(f"r02_does_not_preserve_complete_r01_prefix:{root_id}")
    round2_turns = r02_turns_all[len(r01_turns):]
    if not round2_turns or any(
            not str(t["turn_ref"]).startswith("R02:") for t in round2_turns):
        raise PreflightError(f"r02_delta_refs_invalid:{root_id}")
    if (int(r02.get("turn_count") or -1) != len(r02_turns_all) or
            int(r02.get("latest_round_child_turn_count") or -1) != len(round2_turns) or
            int(validation.get("prior_turn_count") or -1) != len(r01_turns) or
            int(validation.get("new_turn_count") or -1) != len(round2_turns) or
            int(validation.get("turn_count") or -1) != len(r02_turns_all)):
        raise PreflightError(f"r02_cumulative_counts_invalid:{root_id}")
    r02_complete_sha = _canonical_dialogue_hash(r02)
    if r02.get("complete_dialogue_sha256") != r02_complete_sha:
        raise PreflightError(f"r02_internal_sha_invalid:{root_id}")
    if intent.get("source_dialogue_sha256") != r02_complete_sha:
        raise PreflightError(f"r02_memory_source_sha_invalid:{root_id}")
    intent_refs = [str(value) for value in intent.get("turn_refs") or []]
    round2_refs = [str(turn["turn_ref"]) for turn in round2_turns]
    if intent_refs != round2_refs:
        raise PreflightError(f"r02_memory_delta_refs_invalid:{root_id}")
    messages = intent.get("payload", {}).get("messages") or []
    expected_messages = [{
        "role": "user", "timestamp": -1,
        "content": f"[Speaker: {turn['speaker']}] {turn['text']}",
    } for turn in round2_turns]
    if messages != expected_messages:
        raise PreflightError(f"r02_memory_delta_content_invalid:{root_id}")

    return {
        "root_id": root_id,
        "candidate_ids": [f"{root_id}:R01", f"{root_id}:R02"],
        "split": "train",
        "logical_shard": authority.get("logical_shard"),
        "selected_live_index": live_index,
        "selected_state_status": state.get("status"),
        "root_state_path": str(state_path.resolve()),
        "root_state_sha256": sha256_file(state_path),
        "attempt_root": str(attempt_root.resolve()),
        "report_path": str(expected["report"].resolve()),
        "report_sha256": sha256_file(expected["report"]),
        "r01_dialogue_path": str(r01_path.resolve()),
        "r01_dialogue_file_sha256": r01_file_sha,
        "r01_dialogue_sha256": str(r01["complete_dialogue_sha256"]),
        "r02_dialogue_path": str(expected["dialogue"].resolve()),
        "r02_dialogue_file_sha256": sha256_file(expected["dialogue"]),
        "r02_dialogue_sha256": r02_complete_sha,
        "r02_validation_path": str(expected["validation"].resolve()),
        "r02_validation_sha256": sha256_file(expected["validation"]),
        "child_add_intent_path": str(expected["intent"].resolve()),
        "child_add_intent_sha256": sha256_file(expected["intent"]),
        "child_add_completion_path": str(expected["completion"].resolve()),
        "child_add_completion_sha256": sha256_file(expected["completion"]),
        "prior_user_id": prior_user_id,
        "r02_session_id": expected_session,
        "parent_turn_count": len(parent_turns),
        "r01_new_turn_count": len(round1_turns),
        "r01_cumulative_turn_count": len(r01_turns),
        "r02_new_turn_count": len(round2_turns),
        "r02_cumulative_turn_count": len(r02_turns_all),
        "generation_source": str(report.get("summary_version") or "unknown"),
        "question_type": "not_applicable_root_dialogue_comparison",
        "position_mechanism": "cumulative_parent_tail_only",
        "seed": paired_seed(str(r01["complete_dialogue_sha256"]), r02_complete_sha),
    }


def _check_manifest_drift(authority: Sequence[Mapping[str, Any]]) -> int:
    fields = (
        "canonical_dialogue_sha256", "complete_dialogue_sha256",
        "prior_user_id", "prior_session_id", "r02_session_id", "split",
    )
    expected = {str(row["root_id"]): row for row in authority}
    checked = 0
    for live_index in range(2, 48):
        path = RUN_BASE / f"{R02_PREFIX}{live_index}" / "round1_root_manifest.jsonl"
        _verify_sidecar(path)
        rows = _load_jsonl(path)
        actual = {str(row.get("root_id") or ""): row for row in rows}
        if len(rows) != EXPECTED_ROOT_COUNT or set(actual) != set(expected):
            raise PreflightError(f"root_manifest_drift:{live_index}:root_set")
        for root_id, expected_row in expected.items():
            for field in fields:
                if actual[root_id].get(field) != expected_row.get(field):
                    raise PreflightError(
                        f"root_manifest_drift:{live_index}:{root_id}:{field}")
        checked += 1
    return checked


def _active_expansion_processes() -> list[dict[str, Any]]:
    patterns = (
        "h4_round1_cumulative_continuation_fresh_mining_v1.batch",
        "simple_append_memorax_v1_memory_event_continuation_probe_v1.live",
    )
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and any(pattern in command for pattern in patterns):
            matches.append({"pid": int(entry.name), "command": command})
    return sorted(matches, key=lambda row: row["pid"])


def _balanced_assignments(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ranked = sorted(
        (sha256_bytes((PROMPT_VERSION + str(row["root_id"])).encode("utf-8")),
         str(row["root_id"])) for row in rows)
    midpoint = len(ranked) // 2
    return {root_id: ("R01" if index < midpoint else "R02")
            for index, (_, root_id) in enumerate(ranked)}


def build_preflight(*, run_root: Path, config_path: Path) -> dict[str, Any]:
    if run_root.exists():
        raise PreflightError(f"run_root_already_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    for dirname in ("request_ledger", "raw_prompts", "raw_responses"):
        (run_root / dirname).mkdir()
    started_at = _now()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    errors: list[str] = []
    config: dict[str, str] | None = None
    try:
        config = _read_env_file(config_path)
        checks["strong_model_identity_explicit"] = True
        checks["strong_model_api_key_present"] = True
        checks["strong_model_config_private"] = True
        _write_json(run_root / "strong_evaluator_identity.json",
                    _identity(config, config_path))
        _write_json(run_root / "strong_evaluator_contract.json", _contract())

        _verify_sidecar(AUTHORITY_MANIFEST)
        authority = _load_jsonl(AUTHORITY_MANIFEST)
        authority_ids = [str(row.get("root_id") or "") for row in authority]
        checks["authority_root_count_70"] = len(authority) == EXPECTED_ROOT_COUNT
        checks["authority_root_ids_unique_complete"] = (
            len(set(authority_ids)) == EXPECTED_ROOT_COUNT and "" not in authority_ids)
        checks["train_only_no_dev_test"] = all(
            row.get("split") == "train" and root_id.startswith("cc-train-")
            for row, root_id in zip(authority, authority_ids))
        if not all(checks.values()):
            raise PreflightError("authority_manifest_contract_failed")
        details["manifest_copies_sha_checked"] = _check_manifest_drift(authority)
        checks["r01_manifest_sha_no_drift_live_2_to_47"] = True

        candidates = _state_candidates(set(authority_ids))
        checks["all_authority_roots_have_state_candidates"] = (
            set(candidates) == set(authority_ids))
        if not checks["all_authority_roots_have_state_candidates"]:
            missing = sorted(set(authority_ids) - set(candidates))
            raise PreflightError("r02_state_candidate_missing:" + ",".join(missing))
        selected = []
        rejected = []
        authority_by_id = {str(row["root_id"]): row for row in authority}
        for root_id in sorted(authority_ids):
            valid_row = None
            local_errors = []
            ordered = sorted(candidates[root_id], key=lambda value: value[0], reverse=True)
            for live_index, state_path, state in ordered:
                try:
                    valid_row = _validate_root_candidate(
                        authority_by_id[root_id], live_index, state_path, state)
                    break
                except Exception as error:
                    local_errors.append({
                        "live_index": live_index,
                        "state_path": str(state_path.resolve()),
                        "status": state.get("status"),
                        "reason": f"{type(error).__name__}:{error}",
                    })
            rejected.extend({"root_id": root_id, **item} for item in local_errors)
            if valid_row is None:
                raise PreflightError(f"no_valid_completed_r02_state:{root_id}")
            valid_row["state_candidate_count"] = len(ordered)
            valid_row["newer_or_equal_rejected_state_count"] = len(local_errors)
            selected.append(valid_row)

        assignments = _balanced_assignments(selected)
        for row in selected:
            row["initial_a_assignment"] = assignments[row["root_id"]]
            row["mirror_1_order"] = [assignments[row["root_id"]],
                                      "R02" if assignments[row["root_id"]] == "R01" else "R01"]
            row["mirror_2_order"] = list(reversed(row["mirror_1_order"]))
        candidate_ids = [candidate_id for row in selected
                         for candidate_id in row["candidate_ids"]]
        active = _active_expansion_processes()
        checks.update({
            "r02_success_root_count_70": len(selected) == EXPECTED_ROOT_COUNT,
            "selected_root_ids_unique_complete": (
                len({row["root_id"] for row in selected}) == EXPECTED_ROOT_COUNT and
                {row["root_id"] for row in selected} == set(authority_ids)),
            "evaluation_candidate_ids_unique_complete": (
                len(candidate_ids) == 2 * EXPECTED_ROOT_COUNT and
                len(set(candidate_ids)) == 2 * EXPECTED_ROOT_COUNT),
            "r02_round_index_2_all": True,
            "r02_child_write_verified_all": True,
            "r02_memory_visibility_verified_all": True,
            "r02_parent_add_skipped_all": True,
            "r02_existing_namespace_reused_all": True,
            "r02_user_id_reused_all": True,
            "r02_session_id_pattern_all": True,
            "r01_r02_input_sha_no_drift": True,
            "qa_calls_zero": True,
            "solver_calls_zero": True,
            "semantic_auditor_calls_zero_for_comparison": True,
            "parent_r01_r02_complete_cumulative_dialogues": True,
            "active_expansion_processes_zero": not active,
            "blind_assignment_balanced_35_35": (
                Counter(assignments.values()) == {"R01": 35, "R02": 35}),
            "paired_seed_present_all": all(
                isinstance(row["seed"], int) and 0 <= row["seed"] < 2**31
                for row in selected),
            "api_calls_zero_during_offline_preflight": True,
        })
        details.update({
            "state_files_scanned": sum(len(value) for value in candidates.values()),
            "completed_roots_selected": len(selected),
            "rejected_state_candidates_before_selection": len(rejected),
            "selected_live_indices": dict(sorted(Counter(
                str(row["selected_live_index"]) for row in selected).items())),
            "selected_state_statuses": dict(sorted(Counter(
                str(row["selected_state_status"]) for row in selected).items())),
            "initial_a_assignment": dict(sorted(Counter(assignments.values()).items())),
            "active_expansion_processes": active,
            "qa_calls": 0,
            "solver_calls": 0,
            "semantic_auditor_calls": 0,
            "api_calls": 0,
        })
        if rejected:
            details["rejected_state_candidates"] = rejected
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise PreflightError("preflight_checks_failed:" + ",".join(failed))
        _write_jsonl(run_root / "root_evaluation_manifest.jsonl", selected)
    except Exception as error:
        errors.append(f"{type(error).__name__}:{error}")

    report = {
        "preflight_version": VERSION + "_preflight_v1",
        "started_at": started_at,
        "finished_at": _now(),
        "status": "PASS" if not errors and checks and all(checks.values()) else "BLOCKED",
        "run_root": str(run_root.resolve()),
        "authority_manifest": str(AUTHORITY_MANIFEST.resolve()),
        "r01_source_run": str(R01_RUN.resolve()),
        "r02_source_live_range": [2, 47],
        "source_runs_read_only": True,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "errors": errors,
        "details": details,
        "api_calls": 0,
        "qa_calls": 0,
        "solver_calls": 0,
        "semantic_auditor_calls": 0,
        "next_step": (
            "one_request_protocol_smoke" if not errors and all(checks.values())
            else "stop_without_model_call"),
    }
    _write_json(run_root / "preflight_report.json", report)
    return report


def _load_frozen_run(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for name in (
        "preflight_report.json", "strong_evaluator_identity.json",
        "strong_evaluator_contract.json", "root_evaluation_manifest.jsonl",
    ):
        _verify_sidecar(run_root / name)
    report = _load(run_root / "preflight_report.json")
    if report.get("status") != "PASS" or report.get("api_calls") != 0:
        raise PreflightError("offline_preflight_not_passed")
    rows = _load_jsonl(run_root / "root_evaluation_manifest.jsonl")
    if len(rows) != EXPECTED_ROOT_COUNT or len({row["root_id"] for row in rows}) != EXPECTED_ROOT_COUNT:
        raise PreflightError("root_evaluation_manifest_not_closed")
    for row in rows:
        for path_key, sha_key in (
            ("root_state_path", "root_state_sha256"),
            ("report_path", "report_sha256"),
            ("r01_dialogue_path", "r01_dialogue_file_sha256"),
            ("r02_dialogue_path", "r02_dialogue_file_sha256"),
            ("r02_validation_path", "r02_validation_sha256"),
            ("child_add_intent_path", "child_add_intent_sha256"),
            ("child_add_completion_path", "child_add_completion_sha256"),
        ):
            path = Path(str(row[path_key]))
            if not path.is_file() or sha256_file(path) != row[sha_key]:
                raise PreflightError(f"frozen_input_drift:{row['root_id']}:{path_key}")
    if _active_expansion_processes():
        raise PreflightError("active_expansion_process_detected_after_preflight")
    return report, rows


def _turn_lines(turns: Sequence[Mapping[str, Any]], candidate: str,
                start: int = 1) -> tuple[list[str], dict[str, str]]:
    lines = []
    refs = {}
    for index, turn in enumerate(turns, start):
        local = f"{candidate}:T{index:03d}"
        lines.append(f"[{local}] {turn['speaker']}: {turn['text']}")
        refs[local] = str(turn["turn_ref"])
    return lines, refs


def _candidate_block(label: str, preceding: Sequence[Mapping[str, Any]],
                     continuation: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, str]]:
    preceding_lines, preceding_refs = _turn_lines(preceding, label)
    continuation_lines, continuation_refs = _turn_lines(
        continuation, label, len(preceding) + 1)
    text = (
        f"CANDIDATE {label}\n"
        "PRECEDING CONTEXT\n" + "\n".join(preceding_lines) + "\n"
        "NEW CONTINUATION\n" + "\n".join(continuation_lines)
    )
    return text, {**preceding_refs, **continuation_refs}


def _dialogue_parts(row: Mapping[str, Any]) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    r01 = _load(Path(str(row["r01_dialogue_path"])))
    r02 = _load(Path(str(row["r02_dialogue_path"])))
    r01_turns = list(r01["turns"])
    r02_turns = list(r02["turns"])
    parent_count = int(row["parent_turn_count"])
    return {
        "R01": (r01_turns[:parent_count], r01_turns[parent_count:]),
        "R02": (r01_turns, r02_turns[len(r01_turns):]),
    }


def build_prompt(row: Mapping[str, Any], mirror_index: int) -> tuple[str, dict[str, str]]:
    if mirror_index not in (1, 2):
        raise ValueError("mirror_index_must_be_1_or_2")
    order = list(row[f"mirror_{mirror_index}_order"])
    parts = _dialogue_parts(row)
    block_a, refs_a = _candidate_block("A", *parts[order[0]])
    block_b, refs_b = _candidate_block("B", *parts[order[1]])
    prompt = f"""You are an independent dialogue-continuation quality judge.

Compare two continuation cases. They are not two alternatives from the same starting point. Each candidate has its own complete preceding context and its own new continuation. Judge how well each new continuation extends its displayed context.

For each candidate, assign integer scores from 0 to 5 for exactly these criteria:
- factual_consistency: consistency with facts in the preceding context
- state_transition_quality: natural, coherent, completable state transitions
- incremental_novelty: useful new information rather than unsupported invention
- redundancy_control: avoidance of repeating already completed content
- participant_coherence: correct people, speaker bindings, and temporal relations
- qa_readiness: clarity and usefulness for later question generation

Score anchors: 5 clearly excellent with no substantive issue; 4 good with minor issues only; 3 usable with clear limitations; 2 serious degradation; 1 mostly unusable; 0 confirmed history violation, structural collapse, or impossible to evaluate. Do not output total or mean scores; the caller computes them. confidence is 0.0-1.0 and means confidence that the dialogue text supports the pairwise judgment, not text quality.

Use only the text below. Return pairwise_preference as A, B, TIE, or UNCERTAIN. Use UNCERTAIN when critical content is insufficient or temporal/participant relations cannot be determined. Evidence references must use the local A:T001 or B:T001 form. Issues must identify candidate A or B.

Return exactly one JSON object, with no markdown and no additional keys, matching this shape:
{{"candidate_a":{{"criteria":{{"factual_consistency":0,"state_transition_quality":0,"incremental_novelty":0,"redundancy_control":0,"participant_coherence":0,"qa_readiness":0}}}},"candidate_b":{{"criteria":{{"factual_consistency":0,"state_transition_quality":0,"incremental_novelty":0,"redundancy_control":0,"participant_coherence":0,"qa_readiness":0}}}},"pairwise_preference":"A|B|TIE|UNCERTAIN","confidence":0.0,"blocking_issues":[{{"candidate":"A|B","issue":"..."}}],"minor_issues":[{{"candidate":"A|B","issue":"..."}}],"evidence_turn_refs":["A:T001"],"reason":"..."}}

{block_a}

{block_b}
"""
    forbidden_values = (
        str(row["root_id"]), str(row["r01_dialogue_path"]),
        str(row["r02_dialogue_path"]), str(row["r01_dialogue_file_sha256"]),
        str(row["r02_dialogue_file_sha256"]), str(row["prior_user_id"]),
        str(row["r02_session_id"]),
    )
    if any(value and value in prompt for value in forbidden_values):
        raise PreflightError(f"blind_prompt_metadata_leak:{row['root_id']}")
    return prompt, {**refs_a, **refs_b}


def _validate_issue_list(value: Any, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ResponseValidationError(f"{field}_not_list")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"candidate", "issue"}:
            raise ResponseValidationError(f"{field}_item_schema")
        if item["candidate"] not in {"A", "B"} or not isinstance(item["issue"], str) or not item["issue"].strip():
            raise ResponseValidationError(f"{field}_item_value")
        result.append({"candidate": item["candidate"], "issue": item["issue"].strip()})
    return result


def validate_response(content: str, valid_refs: set[str]) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ResponseValidationError(f"invalid_json:{error.msg}") from error
    expected_keys = {
        "candidate_a", "candidate_b", "pairwise_preference", "confidence",
        "blocking_issues", "minor_issues", "evidence_turn_refs", "reason",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ResponseValidationError("top_level_schema")
    for field in ("candidate_a", "candidate_b"):
        candidate = value[field]
        if not isinstance(candidate, dict) or set(candidate) != {"criteria"}:
            raise ResponseValidationError(f"{field}_schema")
        criteria = candidate["criteria"]
        if not isinstance(criteria, dict) or set(criteria) != set(CRITERIA):
            raise ResponseValidationError(f"{field}_criteria_schema")
        if any(isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5
               for score in criteria.values()):
            raise ResponseValidationError(f"{field}_criteria_range")
    if value["pairwise_preference"] not in {"A", "B", "TIE", "UNCERTAIN"}:
        raise ResponseValidationError("pairwise_preference")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ResponseValidationError("confidence")
    value["confidence"] = float(confidence)
    value["blocking_issues"] = _validate_issue_list(
        value["blocking_issues"], "blocking_issues")
    value["minor_issues"] = _validate_issue_list(value["minor_issues"], "minor_issues")
    refs = value["evidence_turn_refs"]
    if (not isinstance(refs, list) or
            any(not isinstance(ref, str) or not _LOCAL_REF.fullmatch(ref) or
                ref not in valid_refs for ref in refs)):
        raise ResponseValidationError("evidence_turn_refs")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ResponseValidationError("reason")
    value["reason"] = value["reason"].strip()
    return value


def _request_body(config: Mapping[str, str], prompt: str, seed: int) -> dict[str, Any]:
    return {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": (
                "Follow the evaluation contract exactly. Output strict JSON only. ")},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "stream": False,
    }


def _post_chat(config: Mapping[str, str], body_bytes: bytes) -> tuple[int, bytes]:
    endpoint = config["base_url"].rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint, data=body_bytes, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config["_api_key"],
        })
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()


def _extract_content(response_bytes: bytes) -> tuple[str, dict[str, Any]]:
    try:
        envelope = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseValidationError(f"invalid_api_envelope:{error}") from error
    choices = envelope.get("choices") if isinstance(envelope, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ResponseValidationError("api_choices_missing")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ResponseValidationError("api_message_content_missing")
    metadata = {
        "response_id": envelope.get("id"),
        "response_model": envelope.get("model"),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": envelope.get("usage"),
    }
    return content, metadata


def run_one_request_smoke(*, run_root: Path, config_path: Path,
                          root_id: str | None = None) -> dict[str, Any]:
    _, rows = _load_frozen_run(run_root)
    config = _read_env_file(config_path)
    if root_id is None:
        row = min(rows, key=lambda item: (
            int(item["r01_cumulative_turn_count"]) +
            int(item["r02_cumulative_turn_count"]), item["root_id"]))
    else:
        matches = [item for item in rows if item["root_id"] == root_id]
        if len(matches) != 1:
            raise PreflightError(f"unknown_smoke_root_id:{root_id}")
        row = matches[0]
    request_id = f"{row['root_id']}__mirror-01"
    for directory in ("request_ledger", "raw_prompts", "raw_responses"):
        if any((run_root / directory).glob(request_id + "__attempt-*")):
            raise PreflightError(f"smoke_request_already_exists:{request_id}")
    prompt, ref_map = build_prompt(row, 1)
    body = _request_body(config, prompt, int(row["seed"]))
    body_bytes = canonical_json(body).encode("utf-8")
    prompt_path = run_root / "raw_prompts" / f"{request_id}.json"
    _write_json(prompt_path, body)
    prompt_sha = sha256_file(prompt_path)
    valid: dict[str, Any] | None = None
    terminal_reason = None
    api_metadata: dict[str, Any] = {}
    attempts = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response_path = (run_root / "raw_responses" /
                         f"{request_id}__attempt-{attempt:02d}.json")
        ledger_path = (run_root / "request_ledger" /
                       f"{request_id}__attempt-{attempt:02d}.json")
        started_at = _now()
        http_status = None
        validation_error = None
        response_bytes = b""
        try:
            http_status, response_bytes = _post_chat(config, body_bytes)
            _write_text(response_path, response_bytes.decode("utf-8", errors="replace"))
            if http_status != 200:
                raise ResponseValidationError(f"http_status:{http_status}")
            content, api_metadata = _extract_content(response_bytes)
            valid = validate_response(content, set(ref_map))
        except Exception as error:
            validation_error = f"{type(error).__name__}:{error}"
            terminal_reason = validation_error
            if not response_path.exists():
                _write_text(response_path, response_bytes.decode("utf-8", errors="replace"))
        ledger = {
            "ledger_version": VERSION + "_request_ledger_v1",
            "request_id": request_id,
            "attempt": attempt,
            "root_id": row["root_id"],
            "mirror_index": 1,
            "a_assignment": row["mirror_1_order"][0],
            "b_assignment": row["mirror_1_order"][1],
            "provider": config["provider"],
            "model": config["model"],
            "base_url": config["base_url"],
            "api_key_env": config["api_key_env"],
            "authorization_header_recorded": False,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "seed": row["seed"],
            "prompt_version": PROMPT_VERSION,
            "request_body_sha256": sha256_bytes(body_bytes),
            "raw_prompt_path": str(prompt_path.resolve()),
            "raw_prompt_file_sha256": prompt_sha,
            "raw_response_path": str(response_path.resolve()),
            "raw_response_sha256": sha256_file(response_path),
            "started_at": started_at,
            "finished_at": _now(),
            "http_status": http_status,
            "schema_valid": valid is not None,
            "validation_error": validation_error,
            "terminal_invalid_response": attempt == MAX_ATTEMPTS and valid is None,
            "api_metadata": api_metadata if valid is not None else {},
        }
        _write_json(ledger_path, ledger)
        attempts.append(ledger)
        if valid is not None:
            break
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS[attempt - 1])

    smoke_report: dict[str, Any] = {
        "smoke_version": VERSION + "_one_request_smoke_v1",
        "created_at": _now(),
        "status": "PASS" if valid is not None else "BLOCKED",
        "purpose": "single request protocol and seed-parameter acceptance test",
        "root_id": row["root_id"],
        "mirror_index": 1,
        "final_pairwise_conclusion": False,
        "second_mirror_required": True,
        "attempt_count": len(attempts),
        "http_status": attempts[-1]["http_status"],
        "seed_sent": row["seed"],
        "seed_parameter_accepted": (
            valid is not None and attempts[-1]["http_status"] == 200),
        "strict_json_schema_valid": valid is not None,
        "terminal_reason": None if valid is not None else terminal_reason,
        "request_body_identical_across_attempts": len({
            item["request_body_sha256"] for item in attempts}) == 1,
        "api_metadata": api_metadata if valid is not None else {},
        "api_calls": len(attempts),
        "qa_calls": 0,
        "solver_calls": 0,
        "semantic_auditor_calls": 0,
    }
    if valid is not None:
        a_name, b_name = row["mirror_1_order"]
        scores = {}
        for candidate_key, source_name in (("candidate_a", a_name),
                                           ("candidate_b", b_name)):
            criteria = valid[candidate_key]["criteria"]
            total = sum(criteria.values())
            scores[source_name.lower()] = {
                "criteria": criteria,
                "total_score": total,
                "mean_score": total / len(CRITERIA),
            }
        preference = valid["pairwise_preference"]
        remapped = ({"A": a_name, "B": b_name}.get(preference, preference))
        smoke_report["preliminary_result"] = {
            **scores,
            "pairwise_preference": remapped,
            "confidence": valid["confidence"],
            "blocking_issue_count": len(valid["blocking_issues"]),
            "minor_issue_count": len(valid["minor_issues"]),
            "reason": valid["reason"],
        }
    _write_json(run_root / "api_smoke_report.json", smoke_report)
    return smoke_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke-one"), required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root-id")
    args = parser.parse_args(argv)
    if args.phase == "preflight":
        report = build_preflight(run_root=args.run_root, config_path=args.config)
    else:
        report = run_one_request_smoke(
            run_root=args.run_root, config_path=args.config, root_id=args.root_id)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
