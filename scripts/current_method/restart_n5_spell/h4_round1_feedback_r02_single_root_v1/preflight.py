"""Offline-only preflight for one feedback-controlled R02 continuation.

No model, API, Solver, QA, auditor, or Memorax call is made here. The output
is a new, versioned run directory and never mutates the R01 or historical R02
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import VERSION
from ..live_backends import CHALLENGER_DEFAULT_TEMPERATURE


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PARITY_ROOT = RUN_BASE / "h4_round1_feedback_memory_semantic_parity_v1_live_7"
RECON_PREFLIGHT_ROOT = RUN_BASE / "h4_round1_feedback_memory_reconstruction_smoke_v1_live_2"
RECON_ROOT = RUN_BASE / "h4_round1_feedback_memory_reconstruction_smoke_v1_live_3"
COMPILER_ROOT = RUN_BASE / "h4_round1_feedback_compiler_v1_live_1"
TOKEN_BUDGET = COMPILER_ROOT / "feedback_prompt_token_budget.json"
CONFIG_PATH = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER_PATH = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_OUTPUT = RUN_BASE / "h4_round1_feedback_r02_single_root_v1_live_1"
ROOT_ID = "cc-train-episode-254852"

FORBIDDEN_CARD_KEYS = {
    "question", "question_text", "options", "items", "gold", "answer",
    "solver_output", "raw_rollouts", "rollouts", "evidence_only",
}
FORBIDDEN_CARD_TEXT = (
    "<box>", "raw k3", "solver output", "auditor explanation",
    "a/b/c/d", "i1/i2", "i3/i4",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(_sha(path) + "\n",
                                                       encoding="utf-8")


def _load(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"missing_file:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing_file:{path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"jsonl_row_not_object:{path}")
            rows.append(row)
    return rows


def _sidecar_ok(path: Path) -> bool:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False
    value = sidecar.read_text(encoding="utf-8").strip().split()
    return bool(value) and value[0] == _sha(path)


def _redacted_config(path: Path) -> dict[str, Any]:
    value = _load(path)
    api = value.get("api") if isinstance(value, Mapping) else {}
    generation = value.get("generation") if isinstance(value, Mapping) else {}
    if not isinstance(api, Mapping):
        api = {}
    if not isinstance(generation, Mapping):
        generation = {}
    return {
        "config_path": str(path.resolve()),
        "config_sha256": _sha(path),
        "api": {
            "base_url": api.get("base_url"),
            "model": api.get("model"),
            "temperature_declared": api.get("temperature"),
            "temperature_effective": float(
                api.get("temperature")
                if api.get("temperature") is not None
                else CHALLENGER_DEFAULT_TEMPERATURE
            ),
            "temperature_default": CHALLENGER_DEFAULT_TEMPERATURE,
            "api_key_recorded": bool(api.get("api_key")),
        },
        "generation": {
            key: generation.get(key)
            for key in ("augmentation_rounds", "window_size", "min_added_turns",
                        "max_added_turns", "temperature", "candidate_count")
        },
    }


def _dialogue_facts(path: Path, expected_complete_sha: str) -> dict[str, Any]:
    value = _load(path)
    if str(value.get("root_id") or "") != ROOT_ID:
        raise RuntimeError("dialogue_root_mismatch")
    turns = value.get("turns")
    if not isinstance(turns, list) or not turns:
        raise RuntimeError("dialogue_turns_missing")
    refs = [str(row.get("turn_ref") or "") for row in turns]
    if any(not ref for ref in refs) or len(refs) != len(set(refs)):
        raise RuntimeError("dialogue_turn_refs_invalid")
    if any(re.search(r"R0?2|R02-feedback", ref, re.I) for ref in refs):
        raise RuntimeError("r02_ref_in_r01_input")
    sessions = {str(row.get("session_ref") or "") for row in turns}
    if not sessions or any("R02" in session.upper() for session in sessions):
        raise RuntimeError("r02_session_in_r01_input")
    complete_sha = str(value.get("complete_dialogue_sha256") or "")
    if complete_sha != expected_complete_sha:
        raise RuntimeError("complete_dialogue_sha_mismatch")
    round_refs = sorted({
        "parent" if ref.startswith(ROOT_ID + ":P") else ref.split(":", 1)[0]
        for ref in refs
    })
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha(path),
        "complete_dialogue_sha256": complete_sha,
        "root_id": ROOT_ID,
        "turn_count": len(turns),
        "turn_refs": refs,
        "round_refs": round_refs,
        "r02_content_present": False,
    }


def _card_facts(path: Path, dialogue: Mapping[str, Any]) -> dict[str, Any]:
    rows = _load_jsonl(path)
    selected = [row for row in rows if row.get("root_id") == ROOT_ID]
    if len(selected) != 1:
        raise RuntimeError(f"feedback_card_not_unique:{len(selected)}")
    card = selected[0]
    target = card.get("target")
    if not isinstance(target, Mapping):
        raise RuntimeError("feedback_card_target_missing")
    forbidden_keys = sorted(FORBIDDEN_CARD_KEYS & set(card))
    forbidden_keys += sorted(FORBIDDEN_CARD_KEYS & set(target))
    card_text = json.dumps(card, ensure_ascii=False, sort_keys=True).casefold()
    forbidden_text = [token for token in FORBIDDEN_CARD_TEXT
                      if token.casefold() in card_text]
    turn_refs = {str(row.get("turn_ref")) for row in dialogue.get("turns", [])}
    anchors = [str(ref) for ref in target.get("anchor_refs") or []]
    if not anchors or any(ref not in turn_refs for ref in anchors):
        raise RuntimeError("feedback_anchor_missing_from_dialogue")
    if forbidden_keys or forbidden_text:
        raise RuntimeError("feedback_card_contains_forbidden_content")
    allowed_target = {"anchor_refs", "constraints", "desired_event_operator",
                      "frontier_status", "gap_type", "source_candidate_sha256"}
    unexpected = sorted(set(target) - allowed_target)
    if unexpected:
        raise RuntimeError("feedback_card_unexpected_target_fields:" + ",".join(unexpected))
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha(path),
        "root_id": ROOT_ID,
        "feedback_version": card.get("feedback_version"),
        "target": dict(target),
        "anchor_refs": anchors,
        "forbidden_keys": forbidden_keys,
        "forbidden_text": forbidden_text,
        "contains_gold_or_solver_output": False,
    }


def _branch_facts(snapshot_path: Path, intent_path: Path,
                 completion_path: Path, parent_intent_path: Path,
                 parent_completion_path: Path, expected_user: str,
                 expected_source_sha: str) -> dict[str, Any]:
    for path in (snapshot_path, intent_path, completion_path,
                 parent_intent_path, parent_completion_path):
        if not _sidecar_ok(path):
            raise RuntimeError(f"artifact_sidecar_invalid:{path}")
    snapshot = _load(snapshot_path)
    intent = _load(intent_path)
    completion = _load(completion_path)
    parent_intent = _load(parent_intent_path)
    parent_completion = _load(parent_completion_path)
    if str(snapshot.get("user_id") or "") != expected_user:
        raise RuntimeError("snapshot_user_id_mismatch")
    if str(intent.get("user_id") or "") != expected_user:
        raise RuntimeError("r01_intent_user_id_mismatch")
    if str(intent.get("session_id") or "") != f"{ROOT_ID}:memorax:R01":
        raise RuntimeError("r01_session_mismatch")
    if str(intent.get("source_dialogue_sha256") or "") != expected_source_sha:
        raise RuntimeError("r01_source_dialogue_sha_mismatch")
    if completion.get("final_status") != "success" or completion.get("verified_complete") is not True:
        raise RuntimeError("r01_memory_write_not_complete")
    if str(parent_intent.get("user_id") or "") != expected_user:
        raise RuntimeError("parent_intent_user_id_mismatch")
    if str(parent_intent.get("session_id") or "") != f"{ROOT_ID}:memorax:R00":
        raise RuntimeError("parent_session_mismatch")
    if parent_completion.get("final_status") != "success" or parent_completion.get("verified_complete") is not True:
        raise RuntimeError("parent_memory_write_not_complete")
    buckets = snapshot.get("buckets")
    if not isinstance(buckets, Mapping):
        raise RuntimeError("snapshot_buckets_missing")
    counts = {str(key): len(value) for key, value in buckets.items()
              if isinstance(value, Mapping)}
    return {
        "user_id": expected_user,
        "r01_session_id": intent.get("session_id"),
        "parent_intent_path": str(parent_intent_path.resolve()),
        "parent_intent_sha256": _sha(parent_intent_path),
        "parent_completion_path": str(parent_completion_path.resolve()),
        "parent_completion_sha256": _sha(parent_completion_path),
        "intent_path": str(intent_path.resolve()),
        "intent_sha256": _sha(intent_path),
        "completion_path": str(completion_path.resolve()),
        "completion_sha256": _sha(completion_path),
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_sha256": _sha(snapshot_path),
        "bucket_counts": counts,
        "memory_write_verified": True,
        "parent_memory_write_verified": True,
        "r02_content_present": False,
    }


def _runtime_contract(config: Mapping[str, Any], token_budget: Mapping[str, Any]) -> dict[str, Any]:
    api = config["api"]
    generation = config["generation"]
    summary = {
        "target_api_model": api.get("model"),
        "api_temperature": api.get("temperature_effective"),
        "generation_temperature": generation.get("temperature"),
        "min_added_turns": generation.get("min_added_turns"),
        "max_added_turns": generation.get("max_added_turns"),
        "candidate_count": generation.get("candidate_count"),
        "thinking": False,
        "input": "complete canonical Parent + R01 dialogue",
        "memory_retrieval": "feedback R01-only namespace snapshot",
        "feedback_card": "one frozen abstract card; no Gold/options/Solver text",
        "feedback_persistence": {
            "messages": "R02-feedback dialogue only",
            "window_context": "R02-feedback dialogue only",
            "card_in_payload": False,
            "card_in_payload_metadata": False,
            "provenance_storage": "local reports and request ledger only",
        },
        "output": "R02 delta only",
        "memorax": {
            "namespace": "gxy-selfplay:round1-paired-feedback:2338d0e1c349",
            "user_id": f"gxy-selfplay:round1-paired-feedback:2338d0e1c349:{ROOT_ID}",
            "session_id": f"{ROOT_ID}:memorax:R02-feedback",
            "messages": "R02-feedback only",
            "window_context": "R02-feedback only",
            "parent_or_r01_rewrite": False,
        },
        "request_budget_reference": {
            "prompt_budget_path": str(TOKEN_BUDGET.resolve()),
            "prompt_budget_sha256": _sha(TOKEN_BUDGET),
            "root_row_present": bool(token_budget),
            "expansion_prompt_tokens": token_budget.get("expansion_prompt_tokens"),
            "expansion_output_reserve_tokens": token_budget.get("expansion_output_reserve_tokens"),
            "expansion_total_tokens": token_budget.get("expansion_total_tokens"),
            "endpoint_context_limit": token_budget.get("endpoint_context_limit"),
            "future_solver_total_tokens": token_budget.get("future_solver_total_tokens"),
            "future_sft_total_tokens": token_budget.get("future_sft_total_tokens"),
        },
        "prohibited": ["QA", "Solver", "semantic auditor", "Dev", "Test"],
    }
    return summary


def run(output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output_not_empty:{output_root}")
    parity = _load(PARITY_ROOT / "memory_semantic_parity_report.json")
    if parity.get("status") != "SEMANTIC_MEMORY_PARITY_PASS":
        raise RuntimeError("semantic_memory_parity_not_pass")
    preflight = _load(RECON_PREFLIGHT_ROOT / "preflight_report.json")
    smoke = _load(RECON_ROOT / "final_feedback_memory_smoke_report.json")
    mapping = _load(RECON_PREFLIGHT_ROOT / "namespace_mapping.json")
    if preflight.get("root_id") != ROOT_ID or smoke.get("root_id") != ROOT_ID:
        raise RuntimeError("reconstruction_root_mismatch")
    branch = mapping.get("feedback") or {}
    expected_user = str(branch.get("user_id") or "")
    expected_namespace = "gxy-selfplay:round1-paired-feedback:2338d0e1c349"
    if expected_user != f"{expected_namespace}:{ROOT_ID}":
        raise RuntimeError("feedback_namespace_mapping_mismatch")
    if branch.get("r02_session_id") != f"{ROOT_ID}:memorax:R02-feedback":
        raise RuntimeError("feedback_r02_session_mapping_mismatch")
    if branch.get("feedback_card_exposed") is not True:
        raise RuntimeError("feedback_card_not_exposed_in_mapping")
    dialogue_path = Path(str(parity["source"]["dialogue_path"]))
    dialogue = _load(dialogue_path)
    dialogue_info = _dialogue_facts(dialogue_path,
                                    str(parity["source"]["complete_dialogue_sha256"]))
    card_info = _card_facts(COMPILER_ROOT / "root_feedback_cards.jsonl", dialogue)
    snapshot_path = Path(str(PARITY_ROOT.parent / "h4_round1_feedback_memory_semantic_parity_v1_live_3"))
    # Keep the explicit path above tied to the captured R01 snapshot run.
    snapshot_path = RUN_BASE / "h4_round1_feedback_memory_semantic_parity_v1_live_3" / "feedback_r01_bucket_snapshot.json"
    feedback_dir = RECON_ROOT / "feedback_reconstruction"
    expected_source_sha = str(parity["source"]["complete_dialogue_sha256"])
    branch_info = _branch_facts(
        snapshot_path,
        feedback_dir / "r01_child_add_intent.json",
        feedback_dir / "r01_child_add_completion.json",
        feedback_dir / "parent_add_intent.json",
        feedback_dir / "parent_add_completion.json",
        expected_user,
        expected_source_sha,
    )
    token_budget_all = _load(TOKEN_BUDGET)
    token_rows = [row for row in token_budget_all.get("per_root", [])
                  if row.get("root_id") == ROOT_ID]
    if len(token_rows) != 1:
        raise RuntimeError(f"token_budget_root_not_unique:{len(token_rows)}")
    token_row = token_rows[0]
    if token_row.get("parent_plus_r01_truncated") is not False:
        raise RuntimeError("parent_plus_r01_budget_truncated")
    if token_row.get("card_within_1200") is not True or token_row.get("expansion_within_endpoint") is not True:
        raise RuntimeError("feedback_expansion_token_budget_invalid")
    if token_row.get("future_solver_within_16384") is not True:
        raise RuntimeError("future_solver_budget_invalid")
    config = _redacted_config(CONFIG_PATH)
    contract = _runtime_contract(config, token_row)
    checks = {
        "semantic_parity_pass": True,
        "r01_only_dialogue": set(dialogue_info["round_refs"]) == {"parent", "R01"},
        "r02_not_in_dialogue": dialogue_info["r02_content_present"] is False,
        "feedback_card_unique_and_abstract": not card_info["forbidden_keys"] and not card_info["forbidden_text"],
        "feedback_anchors_present": len(card_info["anchor_refs"]) >= 1,
        "feedback_r01_write_verified": branch_info["memory_write_verified"],
        "feedback_parent_write_verified": branch_info["parent_memory_write_verified"],
        "feedback_r01_snapshot_present": Path(branch_info["snapshot_path"]).is_file(),
        "feedback_namespace_frozen": expected_namespace == contract["memorax"]["namespace"],
        "r02_feedback_session_frozen": contract["memorax"]["session_id"] == f"{ROOT_ID}:memorax:R02-feedback",
        "r02_delta_only": contract["memorax"]["messages"] == "R02-feedback only" and contract["memorax"]["window_context"] == "R02-feedback only",
        "parent_r01_rewrite_forbidden": contract["memorax"]["parent_or_r01_rewrite"] is False,
        "token_budget_fits": True,
        "api_calls_zero": True,
        "memorax_calls_zero": True,
        "r02_generation_zero": True,
        "r02_add_zero": True,
        "qa_calls_zero": True,
        "solver_calls_zero": True,
        "semantic_auditor_calls_zero": True,
        "dev_test_access_zero": True,
        "old_r02_control_not_used": True,
    }
    adapter_blockers = {
        "existing_runner_card_injection": False,
        "existing_runner_session_override": False,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    source = {
        "root_id": ROOT_ID,
        "parity_report_path": str((PARITY_ROOT / "memory_semantic_parity_report.json").resolve()),
        "parity_report_sha256": _sha(PARITY_ROOT / "memory_semantic_parity_report.json"),
        "reconstruction_preflight_path": str((RECON_PREFLIGHT_ROOT / "preflight_report.json").resolve()),
        "reconstruction_preflight_sha256": _sha(RECON_PREFLIGHT_ROOT / "preflight_report.json"),
        "dialogue": dialogue_info,
        "feedback_card": card_info,
        "feedback_r01_memory": branch_info,
        "config": config,
        "token_budget_path": str(TOKEN_BUDGET.resolve()),
        "token_budget_sha256": _sha(TOKEN_BUDGET),
    }
    _write(output_root / "source_manifest.json", source)
    runner_files = [
        Path(__file__).resolve().parents[1]
        / "simple_append_memorax_v1_memory_event_continuation_probe_v1/live.py",
        Path(__file__).resolve().parents[1]
        / "simple_append_memorax_v1_memory_event_continuation_probe_v1/payloads.py",
    ]
    runner_audit = {
        "audit_version": VERSION + "_runner_compatibility_v1",
        "files": [{"path": str(path.resolve()), "sha256": _sha(path)}
                  for path in runner_files],
        "feedback_card_injection": {
            "supported": False,
            "evidence": "no feedback_card parameter or card prompt injection in existing live runner",
        },
        "r02_feedback_session_override": {
            "supported": False,
            "evidence": "payloads.build_add_intent derives session_id as <root>:memorax:R<round:02d> and accepts no session override",
        },
        "safe_to_launch_for_feedback": False,
        "required_adapter": [
            "planner card injection",
            "R02-feedback session override",
            "R02-only persistence scope",
        ],
    }
    _write(output_root / "runner_compatibility_audit.json", runner_audit)
    _write(output_root / "r02_feedback_contract.json", {
        "contract_version": VERSION + "_contract_v1",
        "status": "FROZEN",
        "source_manifest_sha256": _sha(output_root / "source_manifest.json"),
        "semantic_parity_report_sha256": _sha(PARITY_ROOT / "memory_semantic_parity_report.json"),
        "dialogue_input": "complete canonical Parent + R01 only",
        "memory_input": "R01-only feedback namespace retrieval",
        "feedback_card": "frozen abstract card; no old question/options/Gold/Solver output",
        "generation_output": "R02-feedback delta only",
        "memorax": contract["memorax"],
        "model": {
            "model": config["api"]["model"],
            "api_temperature": config["api"]["temperature_effective"],
            "generation_temperature": config["generation"]["temperature"],
            "thinking": False,
        },
        "prohibited": contract["prohibited"],
    })
    _write(output_root / "memory_input_manifest.json", {
        "namespace": expected_namespace,
        "user_id": expected_user,
        "r01_session_id": f"{ROOT_ID}:memorax:R01",
        "snapshot_path": branch_info["snapshot_path"],
        "snapshot_sha256": branch_info["snapshot_sha256"],
        "retrieval_scope": "R01-only snapshot; no old control R02",
        "write_scope": "R02-feedback only",
    })
    _write(output_root / "prompt_budget.json", {
        "root_id": ROOT_ID,
        "tokenizer_identity": token_budget_all.get("tokenizer_identity"),
        "row": token_row,
        "checks": {"card_le_1200": token_row.get("card_within_1200"), "expansion_fits_endpoint": token_row.get("expansion_within_endpoint"), "future_solver_fits_16384": token_row.get("future_solver_within_16384"), "parent_plus_r01_truncated": token_row.get("parent_plus_r01_truncated")},
    })
    plan = {
        "plan_version": VERSION + "_generation_plan_v1",
        "status": "READY_PENDING_RUNNER_ADAPTER",
        "root_id": ROOT_ID,
        "workflow": [
            "load canonical Parent + R01 without R02",
            "retrieve feedback namespace R01-only Memory",
            "inject exactly one frozen feedback card into planner input",
            "generate and gate an append-only R02-feedback continuation",
            "write only R02-feedback messages/window_context to feedback namespace",
            "verify Add completion and post-write visibility",
        ],
        "expected_namespace": expected_namespace,
        "expected_user_id": expected_user,
        "expected_session_id": f"{ROOT_ID}:memorax:R02-feedback",
        "expected_api_calls": "existing workflow budget; no calls in preflight",
        "old_control_r02": "read-only and excluded from generation input",
        "runner_adapter_required": True,
        "adapter_requirements": [
            "append feedback card to planner prompt without exposing Gold/options/Solver output",
            "force session_id to <root_id>:memorax:R02-feedback",
            "preserve R02-only messages and window_context",
            "retain request ledger and atomic response envelopes",
        ],
        "checks": checks,
        "adapter_blockers": adapter_blockers,
    }
    _write(output_root / "r02_feedback_generation_plan.json", plan)
    report = {
        "preflight_version": VERSION + "_preflight_v1",
        "created_at": _now(),
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "decision": "READY_FOR_R02_FEEDBACK_SINGLE_ROOT_AFTER_RUNNER_ADAPTER" if all(checks.values()) else "BLOCK_R02_FEEDBACK",
        "root_id": ROOT_ID,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "adapter_blockers": adapter_blockers,
        "source_manifest_sha256": _sha(output_root / "source_manifest.json"),
        "r02_feedback_contract_sha256": _sha(output_root / "r02_feedback_contract.json"),
        "r02_generation_plan_sha256": _sha(output_root / "r02_feedback_generation_plan.json"),
        "api_calls": 0,
        "memorax_calls": 0,
        "r02_generation": 0,
        "r02_add_calls": 0,
        "qa_calls": 0,
        "solver_calls": 0,
        "semantic_auditor_calls": 0,
        "r02_content_access": 0,
    }
    _write(output_root / "preflight_report.json", report)
    md = (
        "# R02 Feedback Single-Root Preflight\n\n"
        f"- Status: `{report['status']}`\n"
        f"- Decision: `{report['decision']}`\n"
        f"- Root: `{ROOT_ID}`\n"
        "- Input: `complete canonical Parent + R01`\n"
        "- Memory write: `R02-feedback delta only`\n"
        "- API/Memorax calls made: `0`\n"
        "- Existing runner launch: `blocked until card/session adapter is frozen`\n"
    )
    (output_root / "final_r02_feedback_preflight_report.md").write_text(md, encoding="utf-8")
    (output_root / "final_r02_feedback_preflight_report.md.sha256").write_text(
        _sha(output_root / "final_r02_feedback_preflight_report.md") + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run(args.output_root.resolve())
    print(json.dumps({"status": report["status"], "decision": report["decision"], "failed_checks": report["failed_checks"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
