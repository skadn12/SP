"""Build and audit immutable 16-row smoke manifests for the two B0 policies."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_round0_b0_matched_training_v1 import builder as matched
from . import (
    BASE_MODEL,
    DATA_RUN,
    EXPECTED_OPTIMIZER_STEPS,
    LOSS_MASK_AUDIT_VERSION,
    RECIPE_VERSION,
    REQUIRED_MAX_LENGTH,
    REPLACEMENT_COUNT,
    SHARED_COUNT,
    SMOKE_COUNT,
    SMOKE_MANIFEST_VERSION,
    TRAINER,
)


TRAIN_FILES = {
    "M_clean_policy": "m_clean_policy_train.jsonl",
    "M_inclusive_policy": "m_inclusive_policy_train.jsonl",
}
MANIFEST_FILES = {
    "M_clean_policy": "m_clean_policy_training_manifest.jsonl",
    "M_inclusive_policy": "m_inclusive_policy_training_manifest.jsonl",
}
SOURCE_FILES = tuple(sorted(set(TRAIN_FILES.values()) | set(MANIFEST_FILES.values()) | {
    "training_interface_contract.json", "training_distribution_comparison.json",
    "matched_clean_risk_pairs.jsonl",
    "dev_full_eval_manifest.jsonl", "dev_m0_hard_eval_manifest.jsonl",
    "test_full_eval_manifest.jsonl", "test_m0_hard_eval_manifest.jsonl",
}))


def stable_int(*parts: Any) -> int:
    return int.from_bytes(bytes.fromhex(sha256_bytes(canonical_json([str(x) for x in parts]).encode()))[:8], "big")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    atomic_json(path, value)
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def read_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"source_sha_sidecar_missing:{path}")
    actual = sha256_file(path)
    if sidecar.read_text(encoding="utf-8").strip() != actual:
        raise RuntimeError(f"source_sha_sidecar_mismatch:{path}")
    return actual


def _id_map(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        cid = str(row.get("candidate_id") or "")
        if not cid or cid in out:
            raise RuntimeError(f"{label}_duplicate_or_missing_candidate:{index}:{cid}")
        out[cid] = dict(row)
    return out


def _verify_display_rebinding(row: Mapping[str, Any]) -> None:
    mapping = row.get("display_to_canonical")
    target = row.get("displayed_target")
    canonical = row.get("canonical_gold")
    if not isinstance(mapping, Mapping) or not isinstance(target, list) or not isinstance(canonical, list):
        raise RuntimeError(f"display_rebinding_fields_missing:{row.get('candidate_id')}")
    rebound = [str(mapping.get(label) or "") for label in target]
    if row["question_type"] == "multi_select":
        ok = set(rebound) == set(str(x) for x in canonical) and len(rebound) == len(set(rebound))
    else:
        ok = rebound == [str(x) for x in canonical]
    if not ok:
        raise RuntimeError(f"display_rebinding_failed:{row.get('candidate_id')}:{target}:{rebound}:{canonical}")


def _pick_pair_slots(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_type[str(row["stratum"]["question_type"])].append(row)
    chosen: list[dict[str, Any]] = []
    for qtype in ("single_choice", "multi_select", "ordering"):
        candidates = by_type.get(qtype, [])
        if not candidates:
            raise RuntimeError(f"matched_pairs_missing_question_type:{qtype}")
        chosen.append(min(candidates, key=lambda row: (
            float(row["prompt_token_relative_delta"]),
            float(row["context_token_relative_delta"]),
            int(row["matching_cost"]), row["pair_id"],
        )))
    used = {row["pair_id"] for row in chosen}
    remaining = [row for row in pair_rows if row["pair_id"] not in used]
    if not remaining:
        raise RuntimeError("matched_pairs_have_no_fourth_slot")
    chosen.append(min(remaining, key=lambda row: (
        float(row["prompt_token_relative_delta"]),
        float(row["context_token_relative_delta"]),
        int(row["matching_cost"]), row["pair_id"],
    )))
    return sorted(chosen, key=lambda row: row["pair_id"])


def _pick_shared_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < SHARED_COUNT:
        raise RuntimeError(f"shared_rows_below_smoke_count:{len(rows)}")
    selected: list[dict[str, Any]] = []
    long_rows = [row for row in rows if int(row["input_token_count"]) > 4096]
    if not long_rows:
        raise RuntimeError("no_shared_sample_above_4096_tokens")
    selected.append(max(long_rows, key=lambda row: (int(row["input_token_count"]), row["candidate_id"])))
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["candidate_id"] != selected[0]["candidate_id"]:
            by_type[str(row["question_type"])].append(row)
    target_counts = {"single_choice": 4, "multi_select": 4, "ordering": 4}
    target_counts[selected[0]["question_type"]] -= 1
    for qtype in ("single_choice", "multi_select", "ordering"):
        choices = sorted(by_type[qtype], key=lambda row: (stable_int(SMOKE_MANIFEST_VERSION, row["candidate_id"]), row["candidate_id"]))
        selected.extend(choices[: target_counts[qtype]])
    if len(selected) != SHARED_COUNT:
        raise RuntimeError(f"shared_selection_count_failed:{len(selected)}")
    return sorted(selected, key=lambda row: (stable_int(SMOKE_MANIFEST_VERSION, "shared", row["candidate_id"]), row["candidate_id"]))


def _source_snapshot(data_run: Path) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = data_run / name
        if not path.is_file():
            raise RuntimeError(f"matched_source_missing:{path}")
        hashes[name] = read_sidecar(path)
    return {"data_run": str(data_run), "files": hashes}


def _manifest_row(
    row: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    policy: str,
    smoke_role: str,
    smoke_slot_id: str,
    source_manifest_sha: str,
    source_train_sha: str,
    tokenizer: Any,
) -> dict[str, Any]:
    messages = training.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise RuntimeError(f"training_messages_invalid:{row.get('candidate_id')}")
    _verify_display_rebinding(row)
    if sha256_bytes(canonical_json(messages).encode()) != row["messages_sha256"]:
        raise RuntimeError(f"source_messages_sha_mismatch:{row.get('candidate_id')}")
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
        full_ids.append(tokenizer.eos_token_id)
    if len(full_ids) > REQUIRED_MAX_LENGTH:
        raise RuntimeError(f"smoke_sample_would_truncate:{row['candidate_id']}:{len(full_ids)}")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if len(prompt_ids) > len(full_ids) or not any(label != -100 for label in labels):
        raise RuntimeError(f"smoke_supervision_missing:{row['candidate_id']}")
    supervised_ids = [label for label in labels if label != -100]
    supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    target_text = str(messages[-1].get("content") or "")
    if not supervised_text.startswith(target_text):
        raise RuntimeError(f"smoke_supervised_decode_mismatch:{row['candidate_id']}:{supervised_text!r}:{target_text!r}")
    return {
        "smoke_manifest_version": SMOKE_MANIFEST_VERSION,
        "policy": policy,
        "smoke_slot_id": smoke_slot_id,
        "smoke_role": smoke_role,
        "candidate_id": row["candidate_id"],
        "root_id": row["root_id"],
        "split": row["split"],
        "question_type": row["question_type"],
        "generation_source": row["generation_source"],
        "evidence_status": row["evidence_status"],
        "position_mechanism": row["position_mechanism"],
        "pair_id": row.get("pair_id"),
        "displayed_target": row["displayed_target"],
        "canonical_gold": row["canonical_gold"],
        "display_to_canonical": row["display_to_canonical"],
        "target_text": target_text,
        "input_token_count": len(full_ids),
        "prompt_token_count": len(prompt_ids),
        "supervised_token_count": len(supervised_ids),
        "supervised_decoded_text": supervised_text,
        "truncated": False,
        "max_length": REQUIRED_MAX_LENGTH,
        "messages_sha256": sha256_bytes(canonical_json(messages).encode()),
        "source_train_jsonl": str(DATA_RUN / TRAIN_FILES[policy]),
        "source_train_jsonl_sha256": source_train_sha,
        "source_training_manifest": str(DATA_RUN / MANIFEST_FILES[policy]),
        "source_training_manifest_sha256": source_manifest_sha,
        "candidate_pool_path": row["candidate_pool_path"],
        "candidate_pool_sha256": row["candidate_pool_sha256"],
        "dialogue_path": row["dialogue_path"],
        "dialogue_sha256": row["dialogue_sha256"],
        "messages": messages,
        "metadata": training.get("metadata"),
    }


def build_smoke_manifests(*, data_run: Path = DATA_RUN, run_root: Path) -> dict[str, Any]:
    data_run = data_run.resolve()
    run_root = run_root.resolve()
    if run_root.exists():
        raise RuntimeError(f"smoke_output_run_already_exists:{run_root}")
    source_snapshot = _source_snapshot(data_run)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    all_rows: dict[str, dict[str, dict[str, Any]]] = {}
    all_training: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in TRAIN_FILES:
        manifest_rows = _id_map(load_jsonl(data_run / MANIFEST_FILES[policy]), f"{policy}_manifest")
        training_rows = load_jsonl(data_run / TRAIN_FILES[policy])
        training_map = {str(row["metadata"]["slot_id"]): row for row in training_rows}
        if len(manifest_rows) != 386:
            raise RuntimeError(f"unexpected_training_manifest_shape:{policy}:{len(manifest_rows)}")
        all_rows[policy] = manifest_rows
        all_training[policy] = training_map
    if set(all_rows["M_clean_policy"]) & set(all_rows["M_inclusive_policy"]):
        # Candidate overlap is expected, but slot identity must be the same and is checked per policy.
        pass
    clean_rows = list(all_rows["M_clean_policy"].values())
    inclusive_rows = list(all_rows["M_inclusive_policy"].values())
    clean_by_slot = {str(row["slot_id"]): row for row in clean_rows}
    inclusive_by_slot = {str(row["slot_id"]): row for row in inclusive_rows}
    if set(clean_by_slot) != set(inclusive_by_slot):
        raise RuntimeError("smoke_policy_slot_sets_differ")
    shared_slots = [slot for slot, row in clean_by_slot.items() if row["selection_role"] == "shared_clean"]
    pair_rows = load_jsonl(data_run / "matched_clean_risk_pairs.jsonl")
    selected_pairs = _pick_pair_slots(pair_rows)
    pair_slots = {f"replacement:{row['pair_id']}": row for row in selected_pairs}
    selected_shared = _pick_shared_rows([clean_by_slot[slot] for slot in shared_slots])
    selected_slots = [f"shared:{row['candidate_id']}" for row in selected_shared] + sorted(pair_slots)
    if len(selected_slots) != SMOKE_COUNT or len(selected_shared) != SHARED_COUNT or len(pair_slots) != REPLACEMENT_COUNT:
        raise RuntimeError(f"smoke_slot_count_failed:{len(selected_slots)}:{len(selected_shared)}:{len(pair_slots)}")
    clean_smoke: list[dict[str, Any]] = []
    inclusive_smoke: list[dict[str, Any]] = []
    for slot in selected_slots:
        clean_row = clean_by_slot[slot]
        inclusive_row = inclusive_by_slot[slot]
        if slot.startswith("shared:"):
            if clean_row["candidate_id"] != inclusive_row["candidate_id"]:
                raise RuntimeError(f"shared_candidate_mismatch:{slot}")
            roles = ((clean_row, "shared_clean"), (inclusive_row, "shared_clean"))
        else:
            pair = pair_slots[slot]
            if clean_row["candidate_id"] != pair["clean_candidate_id"] or inclusive_row["candidate_id"] != pair["risk_candidate_id"]:
                raise RuntimeError(f"pair_slot_candidate_mismatch:{slot}")
            roles = ((clean_row, "clean_control"), (inclusive_row, "risk_replacement"))
        for policy, row, role, output in (
            ("M_clean_policy", roles[0][0], roles[0][1], clean_smoke),
            ("M_inclusive_policy", roles[1][0], roles[1][1], inclusive_smoke),
        ):
            training = all_training[policy].get(str(row["slot_id"]))
            if training is None:
                raise RuntimeError(f"training_record_missing:{policy}:{row['slot_id']}")
            output.append(_manifest_row(
                row, training, policy=policy, smoke_role=role, smoke_slot_id=slot,
                source_manifest_sha=source_snapshot["files"][MANIFEST_FILES[policy]],
                source_train_sha=source_snapshot["files"][TRAIN_FILES[policy]],
                tokenizer=tokenizer,
            ))
    clean_smoke.sort(key=lambda row: row["smoke_slot_id"])
    inclusive_smoke.sort(key=lambda row: row["smoke_slot_id"])
    if [row["smoke_slot_id"] for row in clean_smoke] != [row["smoke_slot_id"] for row in inclusive_smoke]:
        raise RuntimeError("smoke_slot_order_mismatch")
    for clean, inclusive in zip(clean_smoke, inclusive_smoke):
        if clean["displayed_target"] != inclusive["displayed_target"]:
            raise RuntimeError(f"smoke_target_balance_mismatch:{clean['smoke_slot_id']}")
        if clean["question_type"] != inclusive["question_type"]:
            raise RuntimeError(f"smoke_qtype_balance_mismatch:{clean['smoke_slot_id']}")
    for smoke in (clean_smoke, inclusive_smoke):
        if not any(row["input_token_count"] > 4096 for row in smoke):
            raise RuntimeError("smoke_long_sample_missing")
    train_candidates = {row["candidate_id"] for row in clean_rows + inclusive_rows}
    train_roots = {row["root_id"] for row in clean_rows + inclusive_rows}
    eval_candidates: set[str] = set()
    eval_roots: set[str] = set()
    for name in ("dev_full_eval_manifest.jsonl", "test_full_eval_manifest.jsonl"):
        eval_rows = load_jsonl(data_run / name)
        eval_candidates.update(str(row["candidate_id"]) for row in eval_rows)
        eval_roots.update(str(row["root_id"]) for row in eval_rows)
    if train_candidates & eval_candidates or train_roots & eval_roots:
        raise RuntimeError("smoke_train_eval_leakage")
    run_root.mkdir(parents=True, exist_ok=False)
    write_jsonl(run_root / "clean_smoke_manifest.jsonl", clean_smoke)
    write_jsonl(run_root / "inclusive_smoke_manifest.jsonl", inclusive_smoke)
    selected_pair_report = {
        "smoke_manifest_version": SMOKE_MANIFEST_VERSION,
        "shared_count": len(selected_shared),
        "replacement_count": len(selected_pairs),
        "selected_pair_ids": [row["pair_id"] for row in selected_pairs],
        "selected_shared_candidate_ids": [row["candidate_id"] for row in selected_shared],
    }
    return {
        "source_snapshot": source_snapshot,
        "clean_smoke": clean_smoke,
        "inclusive_smoke": inclusive_smoke,
        "selected_pair_report": selected_pair_report,
        "train_eval_candidate_overlap": 0,
        "train_eval_root_overlap": 0,
        "candidate_count": SMOKE_COUNT,
        "shared_count": SHARED_COUNT,
        "replacement_count": REPLACEMENT_COUNT,
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
    }


def audit_loss_mask(rows: list[dict[str, Any]], *, tokenizer: Any, policy: str) -> dict[str, Any]:
    import train_qwen3_lora_sft as trainer

    details: list[dict[str, Any]] = []
    for row in rows:
        messages = row["messages"]
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
            full_ids.append(tokenizer.eos_token_id)
        encoded = trainer.tokenize_record(tokenizer, {"messages": messages}, REQUIRED_MAX_LENGTH)
        labels = encoded["labels"]
        if encoded["input_ids"] != full_ids or full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"trainer_tokenization_contract_mismatch:{row['candidate_id']}")
        supervised_ids = [x for x in labels if x != -100]
        details.append({
            "candidate_id": row["candidate_id"], "root_id": row["root_id"],
            "question_type": row["question_type"], "smoke_role": row["smoke_role"],
            "input_token_count": len(full_ids), "supervised_token_count": len(supervised_ids),
            "supervised_decoded_text": tokenizer.decode(supervised_ids, skip_special_tokens=False),
            "truncated": len(full_ids) > REQUIRED_MAX_LENGTH,
            "prompt_tokens_all_masked": all(x == -100 for x in labels[:len(prompt_ids)]),
            "assistant_tokens_unmasked": all(x != -100 for x in labels[len(prompt_ids):]),
            "target_text": row["target_text"],
        })
    missing = sum(x["supervised_token_count"] == 0 for x in details)
    truncated = sum(x["truncated"] for x in details)
    prompt_bad = sum(not x["prompt_tokens_all_masked"] for x in details)
    assistant_bad = sum(not x["assistant_tokens_unmasked"] for x in details)
    return {
        "loss_mask_audit_version": LOSS_MASK_AUDIT_VERSION,
        "policy": policy, "sample_count": len(details), "max_length": REQUIRED_MAX_LENGTH,
        "truncated_sample_count": truncated, "assistant_supervision_missing": missing,
        "prompt_tokens_not_masked_count": prompt_bad, "assistant_tokens_masked_count": assistant_bad,
        "status": "PASS" if not any((truncated, missing, prompt_bad, assistant_bad)) else "FAIL",
        "samples": details,
    }


__all__ = ["audit_loss_mask", "build_smoke_manifests", "load_json", "load_jsonl", "write_json"]
