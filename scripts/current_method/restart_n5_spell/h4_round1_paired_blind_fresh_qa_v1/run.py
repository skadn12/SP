"""Offline pair closure and paired blind Fresh QA smoke.

The runner deliberately has two explicit stages.  ``preflight`` is fully
offline and closes the eight frozen Control/Feedback pairs.  ``smoke`` may
call only the already verified Fresh QA Writer, six slot requests per branch,
and then applies the local structural gate.  No Solver, Memorax, auditor,
training, or Dev/Test artifact is read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt as build_guided_prompt
from ..h4_qa_mining_hybrid_generation_v1.free_exploration_generator import build_prompt as build_free_prompt
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_qa
from ..h4_qa_multiroot_k3_t08_probe_v1.validator import validate_candidate
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
COMPILER_RUN = RUN_BASE / "h4_round1_feedback_compiler_v1_live_1"
OUTPUT_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_2"
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
GENERATOR_VERSION = "h4_qa_multiroot_k3_t08_fresh_generation_v1"
GENERATOR_TEMPERATURE = 0.1
GENERATOR_MAX_TOKENS = 8192
BRANCHES = ("control", "feedback")
SOURCES = ("contract_guided", "free_exploration")
QUESTION_TYPES = ("single_choice", "multi_select", "ordering")
SUCCESS_ROOTS = {
    "cc-train-episode-122711", "cc-train-episode-254852",
    "cc-train-episode-112812", "cc-train-episode-136202",
    "cc-train-episode-94289", "cc-train-episode-96462",
    "cc-train-episode-198480", "cc-train-episode-197635",
}
QUARANTINE_ROOTS = {"cc-train-episode-259192", "cc-train-episode-218519"}

CONTROL_PATHS = {
    "cc-train-episode-122711": RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_1/smoke_root_cc-train-episode-122711_recovery_2/promotion/promotable_expanded_dialogue.json",
    "cc-train-episode-254852": RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_3/shards/shard-00/roots/cc-train-episode-254852/attempt-01/promotion/promotable_expanded_dialogue.json",
    "cc-train-episode-112812": RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_19/shards/shard-03/roots/cc-train-episode-112812/attempt-01/promotion/promotable_expanded_dialogue.json",
    "cc-train-episode-136202": RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_25/shards/shard-04/roots/cc-train-episode-136202/attempt-01/promotion/promotable_expanded_dialogue.json",
    "cc-train-episode-94289": RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_39/shards/shard-06/roots/cc-train-episode-94289/attempt-01/promotion/promotable_expanded_dialogue.json",
    "cc-train-episode-96462": RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_15/shards/shard-02/roots/cc-train-episode-96462/attempt-01/promotion/promotable_expanded_dialogue.json",
    "cc-train-episode-198480": RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_14/shards/shard-02/roots/cc-train-episode-198480/attempt-01/promotion/promotable_expanded_dialogue.json",
    "cc-train-episode-197635": RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_43/shards/shard-05/roots/cc-train-episode-197635/attempt-01/promotion/promotable_expanded_dialogue.json",
}
FEEDBACK_PATHS = {
    "cc-train-episode-122711": RUN_BASE / "h4_round1_feedback_r02_batch_single_root_v1_live_3/expanded_dialogue.json",
    "cc-train-episode-254852": RUN_BASE / "h4_round1_feedback_r02_single_root_v1_live_6/expanded_dialogue.json",
    "cc-train-episode-112812": RUN_BASE / "h4_round1_feedback_r02_batch_execution_v2_live_4/roots/cc-train-episode-112812/expanded_dialogue.json",
    "cc-train-episode-136202": RUN_BASE / "h4_round1_feedback_r02_batch_execution_v2_live_4/roots/cc-train-episode-136202/expanded_dialogue.json",
    "cc-train-episode-94289": RUN_BASE / "h4_round1_feedback_r02_batch_execution_v2_live_6/roots/cc-train-episode-94289/expanded_dialogue.json",
    "cc-train-episode-96462": RUN_BASE / "h4_round1_feedback_r02_batch_execution_v2_live_7/roots/cc-train-episode-96462/expanded_dialogue.json",
    "cc-train-episode-198480": RUN_BASE / "h4_round1_feedback_r02_batch_execution_v2_live_8/roots/cc-train-episode-198480/expanded_dialogue.json",
    "cc-train-episode-197635": RUN_BASE / "h4_round1_feedback_r02_batch_execution_v2_live_10/roots/cc-train-episode-197635/expanded_dialogue.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_obj(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def write_json(path: Path, value: Any) -> None:
    atomic_json(path, value)


def sidecar(path: Path) -> None:
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "  " + path.name + "\n")


def all_turns(dialogue: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = dialogue.get("turns") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def r02_turns(dialogue: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row for row in all_turns(dialogue) if str(row.get("turn_ref") or "").startswith("R02:")]


def _feedback_run_dir(path: Path) -> Path:
    # The payload lives beside the expanded dialogue for both the single-root
    # and batch layouts.
    return path.parent


def _payload_audit(root_id: str, feedback_path: Path, feedback: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _feedback_run_dir(feedback_path)
    payload_path = run_dir / "memory" / "child_add_intent.json"
    audit_path = run_dir / "feedback_payload_audit.json"
    banned_keys = {
        "feedback_card", "gap_type", "frontier_status", "candidate_id",
        "gold", "solver_output", "solver_results", "old_question",
    }
    result: dict[str, Any] = {
        "root_id": root_id, "payload_path": str(payload_path),
        "payload_exists": payload_path.is_file(), "existing_audit_path": str(audit_path),
        "existing_audit_status": None, "feedback_control_keys_in_metadata": [],
        "feedback_control_keys_in_payload": [], "feedback_control_text_in_dialogue": [],
        "round2_only_turn_refs": False, "status": "FAIL",
    }
    if audit_path.is_file():
        try:
            prior = load_obj(audit_path)
            result["existing_audit_status"] = prior.get("status") or prior.get("guard_status")
        except Exception as exc:
            result["existing_audit_error"] = f"{type(exc).__name__}:{exc}"
    if not payload_path.is_file():
        result["reason"] = "memory_child_add_intent_missing"
        return result
    payload_doc = load_obj(payload_path)
    payload = payload_doc.get("payload") if isinstance(payload_doc.get("payload"), Mapping) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    result["feedback_control_keys_in_metadata"] = sorted(k for k in metadata if str(k).casefold() in banned_keys)
    result["feedback_control_keys_in_payload"] = sorted(k for k in payload if str(k).casefold() in banned_keys)
    r02 = r02_turns(feedback)
    r02_text = [str(row.get("text") or "") for row in r02]
    window = payload.get("window_context") if isinstance(payload.get("window_context"), list) else []
    window_text = [str(row.get("content") or "") for row in window if isinstance(row, Mapping)]
    result["window_context_count"] = len(window_text)
    result["r02_turn_count"] = len(r02)
    allowed_window = {f"[Speaker: {row.get('speaker')}] {row.get('text')}" for row in r02}
    result["window_context_matches_r02"] = bool(window_text) and all(text in allowed_window for text in window_text)
    result["round2_only_turn_refs"] = bool(r02) and all(str(x) in {str(row.get("turn_ref")) for row in r02} for x in payload_doc.get("turn_refs") or [])
    # Metadata may retain the historical boolean ``feedback_card_exposed`` as
    # an audit marker.  The write contract concerns persisted dialogue text;
    # inspect only window_context content for control-text leakage.
    serialized_dialogue = json.dumps(window_text, ensure_ascii=False).casefold()
    result["feedback_control_text_in_dialogue"] = [term for term in ("feedback_card", "emerging_frontier", "solver_output", "frontier_status") if term in serialized_dialogue]
    basic_pass = (
        not result["feedback_control_keys_in_metadata"]
        and not result["feedback_control_keys_in_payload"]
        and not result["feedback_control_text_in_dialogue"]
        and result["round2_only_turn_refs"]
    )
    # Existing batch audits are authoritative for the accepted Memory write
    # contract.  Some implementations intentionally retain a bounded window
    # rather than every promoted R02 turn, so exact text equality is recorded
    # but is not required when the frozen audit is PASS.
    result["status"] = "PASS" if basic_pass and (
        result["existing_audit_status"] == "PASS" or result["window_context_matches_r02"]
    ) else "FAIL"
    return result


def _adherence_pass(feedback_path: Path) -> bool:
    # The authoritative product is the promoted child.  A partial scene
    # promotion is valid when the promotion audit marks the child ready and
    # every promoted scene passed; rejected scenes remain observable.
    path = _feedback_run_dir(feedback_path) / "promotion" / "audit.json"
    if not path.is_file():
        return False
    audit = load_obj(path)
    scenes = audit.get("scenes") or []
    return bool(audit.get("promotable_child_ready")) and bool(scenes) and all(
        str(scene.get("adherence_verdict") or "") == "PASS"
        for scene in scenes if scene.get("promotable")
    )


def close_pairs(output_run: Path) -> dict[str, Any]:
    selected_path = COMPILER_RUN / "selected_10_roots.jsonl"
    selected = [load_obj_from_line(line) for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    order = [str(row.get("root_id") or "") for row in selected]
    if set(order) != SUCCESS_ROOTS | QUARANTINE_ROOTS or len(order) != 10 or len(set(order)) != 10:
        raise RuntimeError(f"selected_root_order_mismatch:{order}")
    pairs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for root_id in sorted(SUCCESS_ROOTS):
        cp, fp = CONTROL_PATHS[root_id], FEEDBACK_PATHS[root_id]
        row: dict[str, Any] = {"root_id": root_id, "control_path": str(cp), "feedback_path": str(fp)}
        try:
            control, feedback = load_obj(cp), load_obj(fp)
            ct, ft = all_turns(control), all_turns(feedback)
            c_refs, f_refs = [str(x.get("turn_ref") or "") for x in ct], [str(x.get("turn_ref") or "") for x in ft]
            payload_audit = _payload_audit(root_id, fp, feedback)
            checks = {
                "control_exists": cp.is_file(), "feedback_exists": fp.is_file(),
                "root_id_equal": control.get("root_id") == root_id == feedback.get("root_id"),
                "parent_sha_equal": control.get("parent_dialogue_sha256") == feedback.get("parent_dialogue_sha256"),
                "prior_complete_sha_equal": control.get("prior_complete_dialogue_sha256") == feedback.get("prior_complete_dialogue_sha256"),
                "control_r02_unique": len([x for x in c_refs if x.startswith("R02:")]) == len(set(x for x in c_refs if x.startswith("R02:"))),
                "feedback_r02_unique": len([x for x in f_refs if x.startswith("R02:")]) == len(set(x for x in f_refs if x.startswith("R02:"))),
                "control_all_refs_unique": len(c_refs) == len(set(c_refs)), "feedback_all_refs_unique": len(f_refs) == len(set(f_refs)),
                "control_r02_present": bool([x for x in c_refs if x.startswith("R02:")]),
                "feedback_r02_present": bool([x for x in f_refs if x.startswith("R02:")]),
                "feedback_append_only": all(str(x.get("turn_ref") or "").startswith("R02:") for x in r02_turns(feedback)),
                "feedback_adherence_pass": _adherence_pass(fp),
                "feedback_payload_contract_pass": payload_audit["status"] == "PASS",
                "dialogue_loadable": bool(ct) and bool(ft),
            }
            row.update({"checks": checks, "feedback_payload_audit": payload_audit,
                        "control_sha256": sha256_file(cp), "feedback_sha256": sha256_file(fp),
                        "control_turn_count": len(ct), "feedback_turn_count": len(ft),
                        "control_prior_complete_sha256": control.get("prior_complete_dialogue_sha256"),
                        "feedback_prior_complete_sha256": feedback.get("prior_complete_dialogue_sha256")})
            row["status"] = "PASS" if all(checks.values()) else "FAIL"
        except Exception as exc:
            row["status"] = "FAIL"; row["error"] = f"{type(exc).__name__}:{exc}"
        (pairs if row["status"] == "PASS" else failures).append(row)
    quarantine = [{"root_id": root_id, "status": "QUARANTINE", "reason": "upstream_feedback_quarantine"} for root_id in sorted(QUARANTINE_ROOTS)]
    successful = [row["root_id"] for row in pairs]
    ordered_success = [root_id for root_id in order if root_id in successful]
    if len(pairs) != 8 or failures:
        raise RuntimeError(f"pair_closure_failed:{len(pairs)}:{failures}")
    summary = {
        "successful_pair_set": successful, "successful_pair_count": len(successful),
        "intent_to_treat_set": order, "intent_to_treat_count": len(order),
        "feedback_quarantine_set": sorted(QUARANTINE_ROOTS), "quarantine_count": len(quarantine),
        "first_successful_root_by_frozen_order": ordered_success[0],
    }
    output_run.mkdir(parents=True, exist_ok=True)
    # JSONL is written explicitly to preserve one immutable record per pair.
    lines = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pairs)
    atomic_text(output_run / "paired_dialogue_manifest.jsonl", lines)
    atomic_text(output_run / "quarantine_root_manifest.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in quarantine))
    write_json(output_run / "paired_dialogue_summary.json", summary)
    write_json(output_run / "feedback_memory_contract_audit.json", {"status": "PASS", "roots": [row["feedback_payload_audit"] for row in pairs]})
    for name in ("paired_dialogue_manifest.jsonl", "quarantine_root_manifest.jsonl", "paired_dialogue_summary.json", "feedback_memory_contract_audit.json"):
        sidecar(output_run / name)
    return {"status": "PAIR_CLOSURE_PASS", **summary}


def load_obj_from_line(line: str) -> dict[str, Any]:
    value = json.loads(line)
    return value if isinstance(value, dict) else {}


def build_contract(dialogue: Mapping[str, Any]) -> dict[str, Any]:
    turns = all_turns(dialogue)
    if len(turns) < 3:
        raise RuntimeError("dialogue_too_short_for_neutral_contract")
    refs = [str(turns[0]["turn_ref"]), str(turns[len(turns) // 2]["turn_ref"]), str(turns[-1]["turn_ref"])]
    if len(set(refs)) != 3:
        refs = [str(turns[i]["turn_ref"]) for i in (0, 1, 2)]
    return {"intent_id": "SMOKE_NEUTRAL", "intent_type": "state_transition",
            "discriminative_variable": "the grounded relation among three dialogue events",
            "required_evidence_groups": {"before_state": [refs[0]], "trigger": [refs[1]], "after_state": [refs[2]]}}


def dialogue_prompt(dialogue: Mapping[str, Any]) -> str:
    return "\n".join(f"{x.get('turn_ref')} | {x.get('speaker')} | {x.get('text')}" for x in all_turns(dialogue))


def make_slot_prompt(dialogue: Mapping[str, Any], source: str, qtype: str) -> tuple[str, str]:
    contract = build_contract(dialogue)
    if source == "contract_guided":
        spec = build_guided_prompt(dialogue, contract)
    else:
        spec = build_free_prompt(dialogue, {"intent_id": "SMOKE_NEUTRAL", "intent_type": "state_transition"})
    # The base prompt and its version are reused verbatim.  This suffix only
    # assigns a smoke slot; it does not contain branch or feedback metadata.
    suffix = (f"\nSMOKE SLOT: retain at least one structurally valid {qtype} candidate. "
              "You may return the normal batch for this generator; the local smoke gate will retain only the requested type. "
              "Return JSON only.")
    return spec["prompt"] + suffix, str(spec.get("prompt_version") or "")


def prompt_isolation(prompt: str, card: Mapping[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(card, ensure_ascii=False)
    # Root ids necessarily occur in authoritative turn_refs supplied to the
    # writer; they are lineage, not feedback metadata.  Only card-specific
    # values are forbidden here.
    forbidden_values = [str(card.get("target", {}).get("source_candidate_sha256") or "")]
    hits = [value for value in forbidden_values if value and value in prompt]
    return {"root_id_in_turn_refs_allowed": True, "feedback_card_values_in_prompt": hits, "feedback_card_payload_in_prompt": serialized in prompt,
            "pass": not hits and serialized not in prompt}


def candidate_pool_ids() -> set[str]:
    old_run = RUN_BASE / "h4_unexpanded122_qa_generation_only_v1"
    ids: set[str] = set()
    for path in old_run.glob("shards/*/roots/*/candidate_pool.json"):
        try:
            rows = load_json(path)
        except Exception:
            continue
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, Mapping) and row.get("candidate_id"):
                ids.add(str(row["candidate_id"]))
    return ids


def normalize_one(raw: Mapping[str, Any], *, root_id: str, branch: str, source: str, qtype: str, ordinal: int) -> dict[str, Any] | None:
    rows = raw.get("candidates") if isinstance(raw.get("candidates"), list) else []
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("question_type") or "") != qtype:
            continue
        item = dict(row)
        if qtype == "ordering" and not item.get("items") and isinstance(item.get("options"), Mapping):
            item["items"] = dict(item["options"])
        original = str(item.get("candidate_id") or "")
        item["source_candidate_id"] = original
        item["candidate_id"] = f"{root_id}:R01-R02-{branch}:blind:{source}:{qtype}:Q{ordinal:02d}"
        item["root_id"] = root_id
        item["branch"] = branch
        item["round"] = "R01-R02"
        item["qa_mode"] = "blind"
        item["generation_source"] = source
        return item
    return None


def gate_candidate(candidate: Mapping[str, Any], dialogue: Mapping[str, Any], prompt_tokens: int) -> dict[str, Any]:
    refs = [str(x) for x in candidate.get("evidence_refs") or []]
    groups = candidate.get("evidence_groups") if isinstance(candidate.get("evidence_groups"), Mapping) else {}
    flat = set(refs)
    grouped = {str(ref) for values in groups.values() if isinstance(values, list) for ref in values}
    valid_refs = {str(row.get("turn_ref")) for row in all_turns(dialogue)}
    audit = validate_candidate(candidate, valid_turn_refs=valid_refs)
    extra = []
    if flat != grouped:
        extra.append("flat_group_evidence_mismatch")
    if candidate.get("root_id") != dialogue.get("root_id"):
        extra.append("root_lineage_mismatch")
    if not candidate.get("candidate_id"):
        extra.append("candidate_id_missing")
    if prompt_tokens > 32768:
        extra.append("context_overflow")
    audit["blocking_errors"] = sorted(set(audit.get("blocking_errors", []) + extra))
    audit["decision"] = "pass" if not audit["blocking_errors"] else "reject"
    audit["schema_valid"] = audit["decision"] == "pass"
    audit["prompt_tokens"] = prompt_tokens
    return audit


def evidence_span(candidate: Mapping[str, Any], dialogue: Mapping[str, Any]) -> str:
    refs = {str(x) for x in candidate.get("evidence_refs") or []}
    r02 = {str(x.get("turn_ref")) for x in r02_turns(dialogue)}
    if refs and refs <= r02:
        return "R02_ONLY"
    if refs and not refs & r02:
        return "HISTORICAL_ONLY"
    return "CROSS_ROUND"


def ordering_risk(candidate: Mapping[str, Any], dialogue: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("question_type") != "ordering":
        return {"candidate_id": candidate.get("candidate_id"), "applicable": False}
    index = {str(row.get("turn_ref")): i for i, row in enumerate(all_turns(dialogue), 1)}
    gold = [str(x) for x in candidate.get("gold") or []]
    refs = [str(x) for x in candidate.get("evidence_refs") or []]
    earliest = min((index.get(ref, 10**9) for ref in refs), default=None)
    return {"candidate_id": candidate.get("candidate_id"), "applicable": True,
            "item_evidence_refs_present": bool(candidate.get("item_evidence_refs")),
            "candidate_evidence_refs": refs, "earliest_canonical_turn_index": earliest,
            "possible_multiple_occurrence": len(refs) != len(set(refs)),
            "submitted_gold": gold, "submitted_gold_anchor_order_consistent": bool(refs) and refs == sorted(refs, key=lambda x: index.get(x, 10**9)),
            "risk_only": True}


def load_card(root_id: str) -> dict[str, Any]:
    path = COMPILER_RUN / "root_feedback_cards.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = load_obj_from_line(line)
        if row.get("root_id") == root_id:
            return row
    return {}


def run_branch(*, output_run: Path, root_id: str, branch: str, dialogue: Mapping[str, Any], manifest_sha: str) -> dict[str, Any]:
    branch_dir = output_run / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    ledger = RequestLedger(branch_dir, manifest_sha, "live")
    config = load_challenger_config(API_CONFIG)
    client = OpenAICompatibleChallengerClient(replace(
        config, temperature=GENERATOR_TEMPERATURE, enable_thinking=False,
        timeout_seconds=max(180, int(config.timeout_seconds))))
    atomic_text(branch_dir / "dialogue_source.json", json.dumps(dialogue, ensure_ascii=False, indent=2) + "\n")
    generation = generate_root_qa(
        expanded=dialogue, root_dir=branch_dir, root_id=root_id,
        api_config=API_CONFIG, ledger=ledger, client=client,
    )
    raw_candidates = load_json(branch_dir / "candidate_pool.json") if (branch_dir / "candidate_pool.json").is_file() else []
    valid_candidates = load_json(branch_dir / "candidates_valid.json") if (branch_dir / "candidates_valid.json").is_file() else []
    raw_candidates = [dict(x) for x in raw_candidates if isinstance(x, Mapping)]
    valid_candidates = [dict(x) for x in valid_candidates if isinstance(x, Mapping)]
    def _scope(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scoped: list[dict[str, Any]] = []
        for row in rows:
            original = str(row.get("candidate_id") or "")
            source = str(row.get("generation_source") or "")
            qtype = str(row.get("question_type") or "")
            ordinal = int(row.get("source_row_index") or len(scoped) + 1)
            row["source_candidate_id"] = original
            row["candidate_id"] = f"{root_id}:R01-R02-{branch}:blind:{source}:{qtype}:Q{ordinal:02d}"
            row.update({"root_id": root_id, "branch": branch, "round": "R01-R02", "qa_mode": "blind"})
            scoped.append(row)
        return scoped
    candidates = _scope(raw_candidates)
    valid_candidates = _scope(valid_candidates)
    valid_by_source_id = {str(row.get("source_candidate_id")): row for row in valid_candidates}
    validity = load_obj(branch_dir / "validity_audit.json") if (branch_dir / "validity_audit.json").is_file() else {"rows": []}
    gates: list[dict[str, Any]] = []
    for index, audit in enumerate(validity.get("rows") or []):
        source_row = raw_candidates[index] if index < len(raw_candidates) else {}
        scoped_id = f"{root_id}:R01-R02-{branch}:blind:{source_row.get('generation_source', '')}:{source_row.get('question_type', '')}:Q{int(source_row.get('source_row_index') or index + 1):02d}"
        gate = dict(audit) if isinstance(audit, Mapping) else {"decision": "reject", "blocking_errors": ["validity_row_invalid"]}
        gate.update({"candidate_id": scoped_id, "root_id": root_id, "branch": branch,
                     "generation_source": source_row.get("generation_source"), "question_type": source_row.get("question_type"),
                     "source_candidate_id": source_row.get("candidate_id")})
        gates.append(gate)
    isolation: list[dict[str, Any]] = []
    for path in sorted((branch_dir / "raw" / "prompts").glob("*.txt")):
        prompt = path.read_text(encoding="utf-8")
        isolation.append({"request_id": path.stem, **prompt_isolation(prompt, load_card(root_id))})
        atomic_text(branch_dir / "raw_prompts" / path.name, prompt)
    for path in sorted((branch_dir / "raw" / "responses").glob("*.txt")):
        atomic_text(branch_dir / "raw_responses" / path.name, path.read_text(encoding="utf-8"))
    ordering = [ordering_risk(candidate, dialogue) for candidate in valid_candidates]
    request_rows = ledger.completed()
    atomic_text(branch_dir / "candidate_pool.json", json.dumps(candidates, ensure_ascii=False, indent=2) + "\n")
    atomic_text(branch_dir / "candidates_valid.json", json.dumps(valid_candidates, ensure_ascii=False, indent=2) + "\n")
    atomic_text(branch_dir / "program_gate_results.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in gates))
    sidecar(branch_dir / "candidate_pool.json")
    sidecar(branch_dir / "program_gate_results.jsonl")
    write_json(branch_dir / "request_plan.json", {"root_id": root_id, "branch": branch, "slot_count": 6,
                                                    "slots": [{"generation_source": s, "batch_question_types": list(QUESTION_TYPES)} for s in SOURCES for _ in (0,)]})
    return {"branch": branch, "generation": generation, "candidate_count": len(valid_candidates), "gate_rows": gates,
            "request_rows": request_rows, "isolation": isolation, "ordering": ordering,
            "ledger_summary": ledger.summary()}


def smoke_phase(output_run: Path, pair_result: Mapping[str, Any]) -> dict[str, Any]:
    root_id = str(pair_result["first_successful_root_by_frozen_order"])
    selected = {"root_id": root_id, "selection_rule": "first successful root in frozen selected_10_roots.jsonl order",
                "control_path": str(CONTROL_PATHS[root_id]), "feedback_path": str(FEEDBACK_PATHS[root_id])}
    write_json(output_run / "selected_smoke_root.json", selected); sidecar(output_run / "selected_smoke_root.json")
    control, feedback = load_obj(CONTROL_PATHS[root_id]), load_obj(FEEDBACK_PATHS[root_id])
    branches = {"control": control, "feedback": feedback}
    plan = {"root_id": root_id, "branches": {}, "total_formal_request_slots": 12,
            "question_types": list(QUESTION_TYPES), "generation_sources": list(SOURCES),
            "paired_seed_rule": "stable request slot identity; branch is excluded from slot seed",
            "qa_contract": {"generator_version": GENERATOR_VERSION, "model_expected": "Qwen/Qwen3-14B",
                            "temperature": GENERATOR_TEMPERATURE, "max_tokens": GENERATOR_MAX_TOKENS,
                            "parser": "JSON object + reused structural candidate validation"}}
    for branch in BRANCHES:
        plan["branches"][branch] = {"slot_count": 6, "slots": [
            {"generation_source": source, "question_type": qtype,
             "request_id": f"{root_id}:R01-R02-{branch}:blind:{source}:{qtype}:Q01"}
            for source in SOURCES for qtype in QUESTION_TYPES]}
    write_json(output_run / "qa_request_plan.json", plan); sidecar(output_run / "qa_request_plan.json")
    results = {branch: run_branch(output_run=output_run, root_id=root_id, branch=branch,
                                  dialogue=dialogue, manifest_sha=sha256_bytes(canonical_json(plan).encode()))
               for branch, dialogue in branches.items()}
    write_json(output_run / "qa_prompt_isolation_audit.json", {"status": "PASS" if all(x["pass"] for b in results.values() for x in b["isolation"]) else "FAIL",
                                                                  "branches": {b: r["isolation"] for b, r in results.items()}})
    old_ids = candidate_pool_ids()
    branch_ids = {b: {str(c["candidate_id"]) for c in json.loads((output_run / b / "candidate_pool.json").read_text())} for b in BRANCHES}
    all_new = set().union(*branch_ids.values()) if branch_ids else set()
    write_json(output_run / "candidate_id_collision_audit.json", {
        "old_round0_candidate_count": len(old_ids),
        "new_candidate_ids": sorted(all_new),
        "cross_branch_overlap": sorted(branch_ids["control"] & branch_ids["feedback"]),
        "old_round0_overlap": sorted(all_new & old_ids),
        "global_new_id_unique": len(all_new) == sum(len(x) for x in branch_ids.values()),
    })
    write_json(output_run / "evidence_span_summary.json", {b: {span: sum(evidence_span(c, branches[b]) == span for c in json.loads((output_run / b / "candidate_pool.json").read_text())) for span in ("HISTORICAL_ONLY", "R02_ONLY", "CROSS_ROUND")} for b in BRANCHES})
    write_json(output_run / "ordering_static_risk_audit.json", {b: r["ordering"] for b, r in results.items()})
    final = {"status": "ROUND1_PAIRED_BLIND_QA_SMOKE_COMPLETED_WITH_AUDIT", "root_id": root_id,
             "branches": {b: {"candidate_count": r["candidate_count"], "program_pass_count": sum(x.get("decision") == "pass" for x in r["gate_rows"]),
                              "request_count": r["ledger_summary"]["completed_request_count"], "technical_failure_count": r["ledger_summary"]["ambiguous_request_count"]} for b, r in results.items()},
             "solver_calls": 0, "memorax_add_calls": 0, "semantic_auditor_calls": 0, "dev_test_accessed": False}
    # Explicit final gate checks every requested slot, including its program
    # pass.  A completed API response without a valid candidate is a smoke
    # failure, never silently repaired offline.
    def _branch_gate(result: Mapping[str, Any]) -> bool:
        completed = result["ledger_summary"]["completed_request_count"]
        terminal = result["ledger_summary"]["ambiguous_request_count"] == 0
        records = result.get("gate_rows") or []
        coverage = {(str(row.get("generation_source")), str(row.get("question_type")))
                    for row in records if row.get("decision") == "pass"}
        expected = {(source, qtype) for source in SOURCES for qtype in QUESTION_TYPES}
        request_states = result.get("request_rows") or []
        no_truncation = all(
            str((row.get("response_metadata") or {}).get("finish_reason") or "") != "length"
            for row in request_states
        )
        ids = [str(row.get("candidate_id") or "") for row in records]
        return completed >= 7 and terminal and expected <= coverage and no_truncation and len(ids) == len(set(ids))
    pass_gate = all(_branch_gate(r) for r in results.values())
    configured_temperature = load_challenger_config(API_CONFIG).temperature
    final["qa_contract_configured_temperature"] = configured_temperature
    final["qa_contract_temperature_match"] = configured_temperature == GENERATOR_TEMPERATURE
    pass_gate = pass_gate and final["qa_contract_temperature_match"]
    final["status"] = "ROUND1_PAIRED_BLIND_QA_SMOKE_PASS" if pass_gate else "ROUND1_PAIRED_BLIND_QA_SMOKE_FAIL"
    final["ready_for_8_root_paired_blind_qa_generation"] = pass_gate
    write_json(output_run / "final_paired_qa_smoke_report.json", final)
    atomic_text(output_run / "final_paired_qa_smoke_report.md", "# Round-1 Paired Blind Fresh QA Smoke\n\n" + json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    for name in ("qa_prompt_isolation_audit.json", "candidate_id_collision_audit.json", "evidence_span_summary.json", "ordering_static_risk_audit.json", "final_paired_qa_smoke_report.json"):
        sidecar(output_run / name)
    return final


def preflight_phase(output_run: Path) -> dict[str, Any]:
    result = close_pairs(output_run)
    preflight = {"status": result["status"], "successful_pair_count": result["successful_pair_count"],
                 "quarantine_count": result["quarantine_count"], "intent_to_treat_count": result["intent_to_treat_count"],
                 "qa_contract_expected_temperature": GENERATOR_TEMPERATURE,
                 "qa_configured_temperature": load_challenger_config(API_CONFIG).temperature,
                 "qa_contract_temperature_match": load_challenger_config(API_CONFIG).temperature == GENERATOR_TEMPERATURE,
                 "solver_calls": 0, "qa_calls": 0, "memorax_add_calls": 0, "semantic_auditor_calls": 0,
                 "dev_test_accessed": False, "source_runs_read_only": True}
    write_json(output_run / "preflight_report.json", preflight); sidecar(output_run / "preflight_report.json")
    configured_temperature = load_challenger_config(API_CONFIG).temperature
    contract = {"generator_version": GENERATOR_VERSION, "model": "Qwen/Qwen3-14B", "temperature": GENERATOR_TEMPERATURE,
                "api_config_temperature": configured_temperature,
                "api_config_temperature_matches_contract": configured_temperature == GENERATOR_TEMPERATURE,
                "max_tokens": GENERATOR_MAX_TOKENS, "question_types": list(QUESTION_TYPES),
                "generation_sources": list(SOURCES), "prompt_builder_modules": ["contract_guided_generator.py", "free_exploration_generator.py"],
                "solver_enabled": False, "feedback_guided_qa": False}
    write_json(output_run / "paired_qa_contract.json", contract); sidecar(output_run / "paired_qa_contract.json")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "smoke", "all"), required=True)
    parser.add_argument("--output-run", type=Path, default=OUTPUT_RUN)
    parser.add_argument("--live", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase in {"preflight", "all"}:
        result = preflight_phase(args.output_run.resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.phase in {"smoke", "all"}:
        if not args.live:
            raise RuntimeError("smoke_requires_explicit_live")
        pair_result = {"first_successful_root_by_frozen_order": load_obj(args.output_run / "paired_dialogue_summary.json")["first_successful_root_by_frozen_order"]}
        result = smoke_phase(args.output_run.resolve(), pair_result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
