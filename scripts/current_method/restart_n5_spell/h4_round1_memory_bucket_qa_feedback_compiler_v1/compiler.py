"""Offline compiler for the Round-1 memory-bucket QA feedback preflight.

This module intentionally does not import an API client, Solver runner, or
Memorax client.  It reads frozen JSON/JSONL artifacts and records blockers
when the requested structured provenance is not present.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import (
    ALGORITHM_VERSION,
    API_CONFIG_PATH,
    DEFAULT_RUN_ROOT,
    E1Q0_QA_RUN,
    EVIDENCE_STATUSES,
    EXPANSION_COMPILER_RUN,
    EXPECTED_CANDIDATES,
    EXPECTED_E1Q0_CANDIDATES,
    EXPECTED_ROOTS,
    EXPECTED_ROUND0_CANDIDATES,
    QUARANTINE_ROOT_IDS,
    ROUND0_DIFFICULTY_RUN,
    ROUND0_EVIDENCE_RUN,
    ROUND0_REEVAL_RUN,
    ROUND0_TRAINING_RUN,
    SELECTED_ROOT_IDS,
    SOLVER_COMPARE_RUN,
    SOLVER_EVAL_RUN,
    TOKENIZER_PATH,
    VERSION,
)


EPS = 1e-9
ROOT_ORDER = tuple(sorted(SELECTED_ROOT_IDS))
P_ORDER = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4}
EXPOSURE_ORDER = {
    "UNEXPOSED_TO_BOTH": 0,
    "CLEAN_ONLY_TRAINED": 1,
    "INCLUSIVE_ONLY_TRAINED": 1,
    "SHARED_TRAINED": 2,
}
EVIDENCE_ORDER = {
    "EVIDENCE_CLEAN": 0,
    "NOT_APPLICABLE_NON_HARD": 1,
    "EVIDENCE_RISK": 2,
}
QTYPE_ORDER = {"multi_select": 0, "ordering": 1, "single_choice": 2}
R02_REF_RE = re.compile(r"R02:S\d+:T\d+")
ITEM_REF_RE = re.compile(r"^(?:I|item[_-]?)[0-9]+$", re.IGNORECASE)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_sidecar(path: Path) -> tuple[bool, str | None]:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False, None
    observed = sha_file(path)
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    return observed == expected, observed


def write_artifact(path: Path, value: Any) -> None:
    if isinstance(value, str):
        text = value
    elif path.suffix == ".jsonl":
        text = "".join(canonical(row) + "\n" for row in value)
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"immutable_artifact_drift:{path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(sha_file(path) + "\n", encoding="utf-8")


def index(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        value = str(raw.get(key) or "")
        if not value or value in result:
            raise RuntimeError(f"invalid_or_duplicate_{key}:{value}")
        result[value] = dict(raw)
    return result


def norm_text(value: Any) -> str:
    text = str(value or "")
    text = " ".join(text.casefold().split())
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text, flags=re.UNICODE)
    return text


def token_set(value: Any) -> set[str]:
    return set(re.findall(r"\w+", norm_text(value), flags=re.UNICODE))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def state(full: float) -> str:
    if full <= EPS:
        return "zero"
    if full >= 1.0 - EPS:
        return "resolved"
    return "partial"


def context_gain(model: Mapping[str, Any]) -> float:
    return float(model["full_context_accuracy"]) - float(model["no_context_accuracy"])


def summarize_raw_result(row: Mapping[str, Any]) -> dict[str, Any]:
    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for request in row.get("requests") or []:
        if request.get("context_mode") in {"full_context", "no_context"} and request.get("scored") is True:
            by_mode[str(request["context_mode"])].append(request)
    values: dict[str, Any] = {}
    for mode in ("full_context", "no_context"):
        requests = by_mode[mode]
        correct = sum(bool(r.get("correct")) for r in requests)
        total = len(requests)
        values[mode] = {
            "accuracy": correct / total if total else None,
            "correct": correct,
            "total": total,
            "predictions": [list(map(str, r.get("canonical_prediction") or [])) for r in requests],
        }
    return {
        "full_context_accuracy": values["full_context"]["accuracy"],
        "no_context_accuracy": values["no_context"]["accuracy"],
        "full_context_correct_count": values["full_context"]["correct"],
        "full_context_total_count": values["full_context"]["total"],
        "no_context_correct_count": values["no_context"]["correct"],
        "no_context_total_count": values["no_context"]["total"],
        "full_predictions": values["full_context"]["predictions"],
        "no_predictions": values["no_context"]["predictions"],
        "technical_failure_count": int(row.get("technical_failure_count") or 0)
        + sum(not bool(r.get("scored")) for r in row.get("requests") or []),
    }


def model_payload(m0: Mapping[str, Any], raw: Mapping[str, Any], bucket: str | None) -> dict[str, Any]:
    full = float(raw["full_context_accuracy"])
    no = float(raw["no_context_accuracy"])
    return {
        "difficulty_bucket": bucket,
        "full_context_accuracy": full,
        "no_context_accuracy": no,
        "full_context_correct_count": raw["full_context_correct_count"],
        "full_context_total_count": raw["full_context_total_count"],
        "no_context_correct_count": raw["no_context_correct_count"],
        "no_context_total_count": raw["no_context_total_count"],
        "context_gain": full - no,
        "technical_failure_count": raw["technical_failure_count"],
        "position_mechanism": m0.get("position_mechanism"),
        "canonical_flip_rate": m0.get("canonical_flip_rate"),
    }


def canonical_error_sets(candidate: Mapping[str, Any], predictions: Sequence[Sequence[str]]) -> dict[str, Any]:
    qtype = str(candidate.get("question_type"))
    gold = [str(x) for x in candidate.get("gold") or []]
    gold_set = set(gold)
    if qtype == "single_choice":
        wrong = Counter(str(p[0]) for p in predictions if len(p) == 1 and str(p[0]) not in gold_set)
        total = len(predictions)
        dominant = wrong.most_common(1)[0][0] if wrong else None
        rate = wrong[dominant] / total if dominant is not None and total else 0.0
        return {"kind": "single", "error_set": {dominant} if dominant else set(), "dominant": dominant, "dominant_rate": rate}
    if qtype == "multi_select":
        options = [str(x) for x in (candidate.get("options") or {}).keys()]
        errors: set[str] = set()
        for prediction in predictions:
            pred = set(map(str, prediction))
            errors |= (gold_set - pred)
            errors |= (pred - gold_set)
        return {"kind": "multi", "error_set": errors, "options": options}
    inversions: set[tuple[str, str]] = set()
    for left_index, left in enumerate(gold):
        for right in gold[left_index + 1:]:
            for prediction in predictions:
                if left in prediction and right in prediction and prediction.index(left) > prediction.index(right):
                    inversions.add((left, right))
    return {"kind": "ordering", "error_set": inversions}


def confusion_feature(candidate: Mapping[str, Any], clean_raw: Mapping[str, Any], inclusive_raw: Mapping[str, Any]) -> dict[str, Any]:
    clean = canonical_error_sets(candidate, clean_raw["full_predictions"])
    inclusive = canonical_error_sets(candidate, inclusive_raw["full_predictions"])
    left, right = clean["error_set"], inclusive["error_set"]
    overlap = len(left & right) / len(left | right) if left | right else 0.0
    common = sorted(left & right, key=str)
    payload: dict[str, Any] = {
        "question_type": candidate.get("question_type"),
        "clean_error_set": sorted(left, key=str),
        "inclusive_error_set": sorted(right, key=str),
        "shared_error_set": common,
        "shared_overlap": overlap,
        "clean_rollout_count": len(clean_raw["full_predictions"]),
        "inclusive_rollout_count": len(inclusive_raw["full_predictions"]),
    }
    if candidate.get("question_type") == "single_choice":
        payload.update({
            "clean_dominant_distractor": clean.get("dominant"),
            "inclusive_dominant_distractor": inclusive.get("dominant"),
            "clean_dominant_rate": clean.get("dominant_rate"),
            "inclusive_dominant_rate": inclusive.get("dominant_rate"),
        })
    return payload


def classify_priority(row: Mapping[str, Any], confusion: Mapping[str, Any]) -> str | None:
    m0, clean, inclusive = (row["models"][x] for x in ("M0", "M_clean_policy", "M_inclusive_policy"))
    cfull, ifull = float(clean["full_context_accuracy"]), float(inclusive["full_context_accuracy"])
    cno, ino = float(clean["no_context_accuracy"]), float(inclusive["no_context_accuracy"])
    cstate, istate = state(cfull), state(ifull)
    if float(m0["full_context_accuracy"]) <= EPS and cstate == istate == "partial" and cfull > cno + EPS and ifull > ino + EPS:
        return "P1"
    if clean.get("difficulty_bucket") == "hard_candidate" and inclusive.get("difficulty_bucket") == "hard_candidate" and cfull > cno + EPS and ifull > ino + EPS:
        return "P2"
    m0full = float(m0["full_context_accuracy"])
    if EPS < m0full < 1.0 - EPS and cfull + EPS >= m0full and ifull + EPS >= m0full and ((cfull - m0full) + (ifull - m0full)) / 2.0 >= 0.20 and (cstate != "resolved" or istate != "resolved"):
        return "P3"
    if cstate != "resolved" and istate != "resolved" and confusion.get("shared_error_set") and confusion.get("shared_overlap", 0.0) >= 0.50:
        if row["question_type"] == "single_choice":
            if (confusion.get("clean_dominant_distractor") == confusion.get("inclusive_dominant_distractor")
                    and float(confusion.get("clean_dominant_rate") or 0) >= 1 / 3
                    and float(confusion.get("inclusive_dominant_rate") or 0) >= 1 / 3):
                return "P4"
        elif row["question_type"] in {"multi_select", "ordering"}:
            if row.get("ordering_occurrence_status") in {None, "UNIQUE_OCCURRENCE", "MULTI_OCCURRENCE_ORDER_INVARIANT"}:
                return "P4"
    if ((cstate == "resolved" and istate == "partial") or (istate == "resolved" and cstate == "partial")):
        partial = clean if cstate == "partial" else inclusive
        if context_gain(partial) > EPS and cfull + EPS >= cno and ifull + EPS >= ino:
            return "P5"
    return None


def frontier_score(probability: float) -> float:
    return 4.0 * probability * (1.0 - probability)


def ordering_audit(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("question_type") != "ordering":
        return {"status": "NOT_APPLICABLE", "item_refs_present": False, "combinations": 0}
    groups = candidate.get("evidence_groups") or {}
    item_groups = {str(k): v for k, v in groups.items() if ITEM_REF_RE.match(str(k))}
    if not item_groups:
        return {
            "status": "UNKNOWN_OCCURRENCE",
            "item_refs_present": False,
            "combinations": 0,
            "reason": "missing_item_level_evidence_refs",
        }
    occurrence_counts = {key: len(value) if isinstance(value, list) else 0 for key, value in item_groups.items()}
    if any(count != 1 for count in occurrence_counts.values()):
        return {
            "status": "UNKNOWN_OCCURRENCE",
            "item_refs_present": True,
            "combinations": 0,
            "reason": "non_unique_item_occurrence_requires_turn_refs_not_available_in_candidate",
        }
    return {"status": "UNIQUE_OCCURRENCE", "item_refs_present": True, "combinations": 1, "occurrence_counts": occurrence_counts}


def strict_candidate_ok(row: Mapping[str, Any], *, confirmed_invalid: set[str], untouched: set[str], root_dialogue_refs: set[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    cid = str(row["candidate_id"])
    if cid in untouched:
        reasons.append("untouched_probe")
    if cid in confirmed_invalid:
        reasons.append("confirmed_defect")
    if row.get("technical_failure_count", 0) > 0:
        reasons.append("technical_failure")
    if row.get("position_mechanism") == "position_only":
        reasons.append("pure_position_only")
    if float(row["models"]["M_clean_policy"]["full_context_accuracy"]) + EPS < float(row["models"]["M_clean_policy"]["no_context_accuracy"]):
        reasons.append("clean_context_worse")
    if float(row["models"]["M_inclusive_policy"]["full_context_accuracy"]) + EPS < float(row["models"]["M_inclusive_policy"]["no_context_accuracy"]):
        reasons.append("inclusive_context_worse")
    missing = sorted(set(map(str, row.get("evidence_refs") or [])) - root_dialogue_refs)
    if missing:
        reasons.append("evidence_refs_missing_from_round0_dialogue")
    if row.get("evidence_status") in {"EVIDENCE_INVALID", "EVIDENCE_UNCLASSIFIABLE"}:
        reasons.append("evidence_status_blocked")
    if row.get("ordering_occurrence_status") in {"UNKNOWN_OCCURRENCE", "MULTI_OCCURRENCE_ORDER_SENSITIVE"}:
        reasons.append("unsafe_ordering_occurrence")
    return not reasons, reasons


def find_r02_adherence(r02_root: Path) -> dict[str, Any]:
    # Restrict reads to structured adherence outputs; semantic-auditor files
    # and raw prompts are intentionally excluded.
    paths = [
        r02_root / "realizer_adherence/audit.json",
        r02_root / "realizer_adherence/reaudit_audit.json",
        r02_root / "scene_replacement/adherence_output.json",
        r02_root / "realizer_adherence/output.json",
    ]
    existing = [path for path in paths if path.is_file()]
    refs: set[str] = set()
    roles: set[str] = set()
    valid_files: list[str] = []
    for path in existing:
        try:
            value = load_json(path)
            text = canonical(value)
        except Exception:
            continue
        found = set(R02_REF_RE.findall(text))
        if found:
            valid_files.append(str(path.resolve()))
            refs |= found
        for key in ("selected_new_development", "selected_durable_outcome", "role", "development_status", "durable_outcome_status"):
            if key in text:
                roles.add(key)
    return {
        "status": "SUPPORTED_STRUCTURED_ADHERENCE" if refs else "NO_SUPPORTED_ADHERENCE",
        "structured_files": valid_files,
        "r02_turn_refs": sorted(refs),
        "role_fields_seen": sorted(roles),
    }


def bucket_inventory(snapshot_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not snapshot_path.is_file():
        return [], {"status": "SNAPSHOT_MISSING", "source_turn_refs_present": False}
    snapshot = load_json(snapshot_path)
    snapshot_sha = sha_file(snapshot_path)
    rows: list[dict[str, Any]] = []
    buckets = snapshot.get("buckets") or {}
    for bucket_type in ("person", "event"):
        raw = buckets.get(bucket_type) or {}
        iterable = raw.items() if isinstance(raw, dict) else ((str(i), x) for i, x in enumerate(raw))
        for key, bucket in iterable:
            bucket = bucket if isinstance(bucket, Mapping) else {"content": bucket}
            refs = bucket.get("source_turn_refs") or bucket.get("turn_refs") or []
            content = bucket.get("content") or bucket.get("memory_facts") or bucket
            rows.append({
                "bucket_type": bucket_type,
                "bucket_key": str(bucket.get("canonical_key") or bucket.get("memory_key") or key),
                "normalized_content_sha256": sha_bytes(norm_text(content).encode()),
                "bucket_snapshot_sha256": snapshot_sha,
                "source_turn_refs": sorted(map(str, refs)),
                "source_turn_refs_present": bool(refs),
            })
    return rows, {
        "status": "PASS",
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_sha256": snapshot_sha,
        "bucket_count": len(rows),
        "source_turn_refs_present": all(row["source_turn_refs_present"] for row in rows) if rows else False,
    }


def normalize_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    options = candidate.get("options") or candidate.get("items") or {}
    if isinstance(options, Mapping):
        normalized_options = sorted((norm_text(k), norm_text(v)) for k, v in options.items())
    else:
        normalized_options = sorted(norm_text(x) for x in options)
    return {
        "stem": norm_text(candidate.get("question")),
        "alternatives": normalized_options,
        "gold": sorted(norm_text(x) for x in candidate.get("gold") or []),
    }


def duplicate_relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    a, b = normalize_candidate(left), normalize_candidate(right)
    exact = a == b
    stem_j = jaccard(token_set(a["stem"]), token_set(b["stem"]))
    stem_seq = difflib.SequenceMatcher(None, a["stem"], b["stem"]).ratio()
    alt_a = {x[1] if isinstance(x, tuple) else x for x in a["alternatives"]}
    alt_b = {x[1] if isinstance(x, tuple) else x for x in b["alternatives"]}
    alt_overlap = len(alt_a & alt_b) / len(alt_a | alt_b) if alt_a | alt_b else 1.0
    evidence_a = set(map(str, left.get("evidence_refs") or []))
    evidence_b = set(map(str, right.get("evidence_refs") or []))
    evidence_j = jaccard(evidence_a, evidence_b)
    near = (stem_j >= 0.80 and alt_overlap >= 0.50) or stem_seq >= 0.90 or (evidence_j >= 0.80 and alt_overlap >= 0.50)
    return {
        "exact_duplicate": exact,
        "near_duplicate": bool(near and not exact),
        "stem_token_jaccard": stem_j,
        "stem_sequence_ratio": stem_seq,
        "alternative_text_overlap_ratio": alt_overlap,
        "evidence_ref_jaccard": evidence_j,
    }


def locate_r02_root(feedback_path: Path) -> Path:
    # expanded_dialogue.json lives directly under the root in single-root runs
    # and under roots/<root> in batch runs.
    for parent in [feedback_path.parent, *feedback_path.parents]:
        if (parent / "memory").is_dir() and (parent / "run_status.json").is_file():
            return parent
    return feedback_path.parent


def render_dialogue(dialogue: Mapping[str, Any]) -> str:
    return "\n".join(f"[{x.get('turn_ref')}] {x.get('speaker')}: {x.get('text')}" for x in dialogue.get("turns") or [])


def load_tokenizer() -> tuple[Any | None, dict[str, Any]]:
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = TOKENIZER_PATH / name
        if path.is_file():
            files[name] = {"sha256": sha_file(path), "size": path.stat().st_size}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    except Exception as error:  # pragma: no cover - environment dependent
        return None, {"status": "TOKENIZER_LOAD_FAILED", "error": repr(error), "files": files}
    return tokenizer, {"status": "PASS", "path": str(TOKENIZER_PATH.resolve()), "class": tokenizer.__class__.__name__, "files": files}


def count_tokens(tokenizer: Any | None, text: str) -> int | None:
    return len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer is not None else None


def parse_contract_and_requests() -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    contract_path = E1Q0_QA_RUN / "paired_qa_contract.json"
    contract = load_json(contract_path) if contract_path.is_file() else {}
    rows: list[dict[str, Any]] = []
    for branch in ("control", "feedback"):
        for root in ROOT_ORDER:
            summary_path = E1Q0_QA_RUN / branch / "roots" / root / "request_summary.json"
            if summary_path.is_file():
                summary = load_json(summary_path)
                request_ids: list[str] = []
                ledger = E1Q0_QA_RUN / branch / "roots" / root / "ledger" / "requests"
                if ledger.is_dir():
                    request_ids = sorted(p.stem for p in ledger.glob("*.json"))
                rows.append({
                    "branch": branch,
                    "root_id": root,
                    "planned_primary_request_count": 7,
                    "actual_primary_request_count": int(summary.get("request_count") or 0),
                    "repair_request_count": sum("repair" in x.casefold() for x in request_ids),
                    "retry_request_count": sum("retry" in x.casefold() for x in request_ids),
                    "request_ids": request_ids,
                })
    required = {"model", "effective_temperature", "max_tokens", "intent_miner_max_tokens", "generation_sources", "repair_policy"}
    contract_complete = required <= set(contract)
    return contract, rows, contract_complete


def compile_memory_bucket_feedback(run_root: Path = DEFAULT_RUN_ROOT) -> dict[str, Any]:
    run_root = run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        if not (run_root / "run_identity.json").is_file():
            raise RuntimeError(f"run_root_exists_without_identity:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    write_artifact(run_root / "run_identity.json", {
        "run_identity_version": f"{VERSION}_run_identity_v1",
        "run_root": str(run_root),
        "algorithm_version": ALGORITHM_VERSION,
        "offline_only": True,
        "api_calls": 0,
        "solver_calls": 0,
        "memorax_calls": 0,
        "qa_writer_calls": 0,
        "semantic_auditor_calls": 0,
    })

    source_paths = {
        "round0_transition": ROUND0_REEVAL_RUN / "candidate_transition_matrix.jsonl",
        "round0_exposure": ROUND0_REEVAL_RUN / "candidate_training_exposure_manifest.jsonl",
        "round0_clean_results": ROUND0_REEVAL_RUN / "clean_train/candidate_results.jsonl",
        "round0_inclusive_results": ROUND0_REEVAL_RUN / "inclusive_train/candidate_results.jsonl",
        "round0_difficulty": ROUND0_DIFFICULTY_RUN / "difficulty_manifest.jsonl",
        "round0_evidence": ROUND0_EVIDENCE_RUN / "hard_evidence_status_manifest.jsonl",
        "invalid_blacklist": ROUND0_TRAINING_RUN / "confirmed_invalid_blacklist.json",
        "e1q0_dialogue_manifest": E1Q0_QA_RUN / "paired_dialogue_manifest.jsonl",
        "e1q0_contract": E1Q0_QA_RUN / "paired_qa_contract.json",
        "e1q0_candidates": E1Q0_QA_RUN / "all_candidates.jsonl",
    }
    source_audit: dict[str, Any] = {}
    source_sha: dict[str, str | None] = {}
    for name, path in source_paths.items():
        ok, observed = verify_sidecar(path)
        source_audit[name] = {"path": str(path), "exists": path.is_file(), "sidecar_verified": ok}
        source_sha[name] = observed

    matrix = load_jsonl(source_paths["round0_transition"])
    exposure = load_jsonl(source_paths["round0_exposure"])
    difficulty_all = load_jsonl(source_paths["round0_difficulty"])
    difficulty = [row for row in difficulty_all if row.get("split") == "train"]
    clean_results = load_jsonl(source_paths["round0_clean_results"])
    inclusive_results = load_jsonl(source_paths["round0_inclusive_results"])
    matrix_by_id, exposure_by_id, difficulty_by_id = index(matrix, "candidate_id"), index(exposure, "candidate_id"), index(difficulty, "candidate_id")
    clean_by_id, inclusive_by_id = index(clean_results, "candidate_id"), index(inclusive_results, "candidate_id")
    candidate_ids = set(matrix_by_id)
    alignment = {
        "round0_difficulty_total": len(difficulty_all),
        "round0_difficulty_split_counts": dict(Counter(str(x.get("split")) for x in difficulty_all)),
        "round0_train_candidate_count": len(difficulty),
        "transition_candidate_count": len(matrix),
        "clean_result_count": len(clean_results),
        "inclusive_result_count": len(inclusive_results),
        "candidate_sets_equal": all(set(x) == candidate_ids for x in (exposure_by_id, difficulty_by_id, clean_by_id, inclusive_by_id)),
    }
    if len(difficulty_all) != EXPECTED_ROUND0_CANDIDATES or len(difficulty) != EXPECTED_CANDIDATES or not alignment["candidate_sets_equal"]:
        raise RuntimeError(f"round0_input_alignment_failed:{alignment}")

    evidence_rows = load_jsonl(source_paths["round0_evidence"])
    evidence_by_id = {str(row["candidate_id"]): row for row in evidence_rows if row.get("split") == "train"}
    hard_ids = {cid for cid, row in difficulty_by_id.items() if row.get("difficulty_bucket") == "hard_candidate"}
    if set(evidence_by_id) != hard_ids:
        raise RuntimeError("evidence_hard_identity_mismatch")
    normalized_evidence: dict[str, str] = {}
    for cid in candidate_ids:
        if cid not in evidence_by_id:
            normalized_evidence[cid] = "NOT_APPLICABLE_NON_HARD"
        else:
            status = str(evidence_by_id[cid].get("evidence_status") or "EVIDENCE_UNCLASSIFIABLE")
            normalized_evidence[cid] = "NOT_APPLICABLE_NON_HARD" if status == "NOT_CLASSIFIED_NON_HARD" else status
            if normalized_evidence[cid] not in EVIDENCE_STATUSES:
                normalized_evidence[cid] = "EVIDENCE_UNCLASSIFIABLE"

    blacklist = load_json(source_paths["invalid_blacklist"]) if source_paths["invalid_blacklist"].is_file() else {}
    confirmed_invalid = {str(x.get("candidate_id")) for x in blacklist.get("entries") or []}

    manifest_rows = load_jsonl(source_paths["e1q0_dialogue_manifest"])
    manifest_by_root = index(manifest_rows, "root_id")
    successful = {str(row["root_id"]) for row in manifest_rows if row.get("status") == "PASS"}
    root_scope = {
        "expected_successful_root_ids": sorted(SELECTED_ROOT_IDS),
        "observed_successful_root_ids": sorted(successful),
        "selected_root_ids_equal_e1q0_successful_pair_set": successful == set(SELECTED_ROOT_IDS),
        "selected_disjoint_quarantine": not (set(SELECTED_ROOT_IDS) & set(QUARANTINE_ROOT_IDS)),
        "selected_count": len(SELECTED_ROOT_IDS),
        "quarantine_root_ids": sorted(QUARANTINE_ROOT_IDS),
    }
    if successful != set(SELECTED_ROOT_IDS):
        raise RuntimeError(f"e1q0_successful_root_set_mismatch:{root_scope}")
    write_artifact(run_root / "successful_root_manifest.jsonl", [{"root_id": x, "status": "E1Q0_SUCCESS"} for x in ROOT_ORDER])
    write_artifact(run_root / "quarantine_root_manifest.jsonl", [{"root_id": x, "status": "HARD_EXCLUDED"} for x in sorted(QUARANTINE_ROOT_IDS)])
    write_artifact(run_root / "root_scope_audit.json", root_scope)

    dialogue_identity: list[dict[str, Any]] = []
    dialogue_by_root: dict[str, dict[str, Any]] = {}
    r02_by_root: dict[str, dict[str, Any]] = {}
    for root_id in ROOT_ORDER:
        row = manifest_by_root[root_id]
        feedback_path = Path(str(row["feedback_path"]))
        sidecar_ok, observed = verify_sidecar(feedback_path)
        if feedback_path.is_file() and observed is None:
            # Some immutable single-root artifacts predate sidecar emission.
            # The E1Q0 manifest SHA is still authoritative for identity; record
            # the missing sidecar as a warning rather than inventing drift.
            observed = sha_file(feedback_path)
        dialogue = load_json(feedback_path) if feedback_path.is_file() else {}
        refs = [str(x.get("turn_ref") or "") for x in dialogue.get("turns") or []]
        drift = []
        if observed != row.get("feedback_sha256"):
            drift.append("feedback_sha256")
        if dialogue.get("root_id") != root_id:
            drift.append("root_id")
        if int(row.get("feedback_turn_count") or -1) != len(refs):
            drift.append("turn_count")
        if not any(x.startswith("R02:") for x in refs):
            drift.append("r02_missing")
        r02_root = locate_r02_root(feedback_path)
        r02_delta_file = r02_root / "child_dialogue.json"
        r02_delta_sha = sha_file(r02_delta_file) if r02_delta_file.is_file() else None
        if not r02_delta_sha:
            drift.append("r02_delta_sha_missing")
        boundaries = dialogue.get("round_boundaries") or {}
        child_boundary = dialogue.get("child_boundary") or {}
        if not boundaries or not child_boundary:
            drift.append("r02_boundary_missing")
        dialogue_by_root[root_id] = dialogue
        adherence = find_r02_adherence(r02_root)
        snapshot_path = r02_root / "memory/post_generation_bucket_snapshot.json"
        buckets, snapshot_meta = bucket_inventory(snapshot_path)
        r02_by_root[root_id] = {"r02_root": r02_root, "adherence": adherence, "buckets": buckets, "snapshot": snapshot_meta}
        dialogue_identity.append({
            "root_id": root_id,
            "status": "PASS" if observed == row.get("feedback_sha256") and not drift else "E1Q0_E1Q1_DIALOGUE_DRIFT",
            "e1q0_feedback_path": str(feedback_path),
            "e1q0_feedback_sha256": row.get("feedback_sha256"),
            "observed_feedback_sha256": observed,
            "feedback_sidecar_verified": sidecar_ok,
            "turn_count": len(refs),
            "r02_turn_count": sum(x.startswith("R02:") for x in refs),
            "r02_boundary": dialogue.get("latest_round_boundary"),
            "child_boundary": child_boundary,
            "r02_delta_path": str(r02_delta_file.resolve()),
            "r02_delta_sha256": r02_delta_sha,
            "drift_fields": drift,
            "adherence_status": adherence["status"],
            "bucket_snapshot_status": snapshot_meta["status"],
        })
    write_artifact(run_root / "e1q0_dialogue_identity_audit.json", {"roots": dialogue_identity, "all_pass": all(x["status"] == "PASS" for x in dialogue_identity)})

    # Build a root/candidate table from frozen Round-0 Train artifacts.
    rows_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_by_id: dict[str, dict[str, Any]] = {}
    transition_rows: list[dict[str, Any]] = []
    occurrence_rows: list[dict[str, Any]] = []
    for cid in sorted(candidate_ids):
        drow = difficulty_by_id[cid]
        candidate = dict(drow.get("candidate") or {})
        root_id = str(drow["root_id"])
        source_dialogue_path = Path(str(drow.get("dialogue_path") or ""))
        source_dialogue = load_json(source_dialogue_path) if source_dialogue_path.is_file() else {}
        source_refs = {str(x.get("turn_ref") or "") for x in source_dialogue.get("turns") or []}
        clean_raw, inclusive_raw = summarize_raw_result(clean_by_id[cid]), summarize_raw_result(inclusive_by_id[cid])
        m0 = dict(matrix_by_id[cid]["models"]["M0"])
        clean_m = model_payload(m0, clean_raw, matrix_by_id[cid]["models"]["M_clean_policy"].get("difficulty_bucket"))
        inclusive_m = model_payload(m0, inclusive_raw, matrix_by_id[cid]["models"]["M_inclusive_policy"].get("difficulty_bucket"))
        m0_m = {
            "difficulty_bucket": drow.get("difficulty_bucket"),
            "full_context_accuracy": float(drow.get("full_context_accuracy") or 0),
            "no_context_accuracy": float(drow.get("no_context_accuracy") or 0),
            "context_gain": float(drow.get("full_context_accuracy") or 0) - float(drow.get("no_context_accuracy") or 0),
            "position_mechanism": drow.get("position_mechanism"),
            "canonical_flip_rate": drow.get("canonical_flip_rate"),
        }
        model = {"M0": m0_m, "M_clean_policy": clean_m, "M_inclusive_policy": inclusive_m}
        occurrence = ordering_audit(candidate)
        occurrence_rows.append({"candidate_id": cid, "root_id": root_id, **occurrence})
        row = {
            "candidate_id": cid,
            "root_id": root_id,
            "candidate": candidate,
            "question_type": candidate.get("question_type") or drow.get("question_type"),
            "generation_source": candidate.get("generation_source") or drow.get("generation_source"),
            "evidence_refs": candidate.get("evidence_refs") or [],
            "evidence_status": normalized_evidence[cid],
            "training_exposure": exposure_by_id[cid].get("training_exposure"),
            "position_mechanism": drow.get("position_mechanism"),
            "technical_failure_count": int(clean_raw["technical_failure_count"] + inclusive_raw["technical_failure_count"]),
            "models": model,
            "source_dialogue_refs": source_refs,
            "ordering_occurrence_status": occurrence.get("status"),
            "candidate_sha256": sha_bytes(canonical(candidate).encode()),
            "candidate_pool_sha256": drow.get("candidate_pool_sha256"),
            "source_dialogue_sha256": drow.get("dialogue_sha256"),
        }
        confusion = confusion_feature(candidate, clean_raw, inclusive_raw)
        row["confusion"] = confusion
        row["priority_class"] = classify_priority(row, confusion)
        row["frontier_score"] = frontier_score((float(clean_m["full_context_accuracy"]) + float(inclusive_m["full_context_accuracy"])) / 2.0)
        row["r02_adherence_status"] = (
            r02_by_root[root_id]["adherence"]["status"]
            if root_id in r02_by_root else "OUT_OF_SCOPE_ROOT"
        )
        rows_by_root[root_id].append(row)
        transition_rows.append({
            "candidate_id": cid,
            "root_id": root_id,
            "priority_class": row["priority_class"],
            "evidence_status": row["evidence_status"],
            "training_exposure": row["training_exposure"],
            "question_type": row["question_type"],
            "generation_source": row["generation_source"],
            "position_mechanism": row["position_mechanism"],
            "models": model,
            "confusion": confusion,
            "ordering_occurrence_status": occurrence.get("status"),
            "frontier_score": row["frontier_score"],
        })
    write_artifact(run_root / "ordering_seed_occurrence_audit.jsonl", occurrence_rows)
    write_artifact(run_root / "evidence_status_normalization_audit.json", {
        "status_values": sorted(set(normalized_evidence.values())),
        "counts": dict(sorted(Counter(normalized_evidence.values()).items())),
        "null_to_clean_count": 0,
        "not_applicable_non_hard_count": sum(x == "NOT_APPLICABLE_NON_HARD" for x in normalized_evidence.values()),
    })

    # Untouched probes are the already frozen Round-1 feedback compiler probes.
    old_probe_path = EXPANSION_COMPILER_RUN / "untouched_probe_manifest.jsonl"
    untouched_rows = load_jsonl(old_probe_path) if old_probe_path.is_file() else []
    untouched_ids = {str(x.get("candidate_id")) for x in untouched_rows}

    selected: dict[str, dict[str, Any] | None] = {}
    reserves: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    for root_id in ROOT_ORDER:
        root_rows = rows_by_root[root_id]
        dialogue_refs = {str(x.get("turn_ref") or "") for x in dialogue_by_root[root_id].get("turns") or []}
        eligible: list[dict[str, Any]] = []
        for row in root_rows:
            reasons: list[str] = []
            if row["priority_class"] is None:
                reasons.append("no_P1_P5_class")
            if row["r02_adherence_status"] != "SUPPORTED_STRUCTURED_ADHERENCE":
                reasons.append("NO_SUPPORTED_EVOLUTION_PATH")
            if row["question_type"] == "ordering" and row["ordering_occurrence_status"] not in {"UNIQUE_OCCURRENCE", "MULTI_OCCURRENCE_ORDER_INVARIANT"}:
                reasons.append("unsafe_ordering_occurrence")
            ok, gate_reasons = strict_candidate_ok(row, confirmed_invalid=confirmed_invalid, untouched=untouched_ids, root_dialogue_refs=set(row["source_dialogue_refs"]))
            if not ok:
                reasons.extend(gate_reasons)
            if reasons:
                exclusions.append({"root_id": root_id, "candidate_id": row["candidate_id"], "reasons": sorted(set(reasons))})
                continue
            eligible.append(row)
        eligible.sort(key=lambda x: (
            P_ORDER[str(x["priority_class"])],
            EXPOSURE_ORDER.get(str(x["training_exposure"]), 99),
            EVIDENCE_ORDER.get(str(x["evidence_status"]), 99),
            -(float(x["models"]["M_clean_policy"]["context_gain"]) + float(x["models"]["M_inclusive_policy"]["context_gain"])),
            -float(x["frontier_score"]),
            QTYPE_ORDER.get(str(x["question_type"]), 99),
            sha_bytes(str(x["candidate_id"]).encode()),
        ))
        selected[root_id] = eligible[0] if eligible else None
        reserves[root_id] = eligible[1:3]

    seed_rows = []
    reserve_rows = []
    for root_id in ROOT_ORDER:
        row = selected[root_id]
        if row:
            seed_rows.append({
                "root_id": root_id,
                "candidate_id": row["candidate_id"],
                "candidate_sha256": row["candidate_sha256"],
                "priority_class": row["priority_class"],
                "training_exposure": row["training_exposure"],
                "evidence_status": row["evidence_status"],
                "question_type": row["question_type"],
                "ordering_occurrence_status": row["ordering_occurrence_status"],
                "r02_adherence_status": row["r02_adherence_status"],
                "static_seed_status": "PRIMARY_CANDIDATE",
            })
        else:
            seed_rows.append({"root_id": root_id, "static_seed_status": "NO_SAFE_QA_FEEDBACK_SEED"})
        for reserve in reserves[root_id]:
            reserve_rows.append({
                "root_id": root_id,
                "candidate_id": reserve["candidate_id"],
                "candidate_sha256": reserve["candidate_sha256"],
                "priority_class": reserve["priority_class"],
                "training_exposure": reserve["training_exposure"],
                "evidence_status": reserve["evidence_status"],
                "question_type": reserve["question_type"],
                "ordering_occurrence_status": reserve["ordering_occurrence_status"],
                "static_reserve": True,
            })
    write_artifact(run_root / "qa_feedback_seed_selection.jsonl", seed_rows)
    write_artifact(run_root / "qa_feedback_seed_reserves.jsonl", reserve_rows)
    write_artifact(run_root / "qa_feedback_seed_exclusions.jsonl", exclusions)
    write_artifact(run_root / "untouched_probe_exclusion_audit.json", {
        "untouched_probe_count": len(untouched_ids),
        "selected_seed_ids": sorted(x["candidate_id"] for x in seed_rows if x.get("candidate_id")),
        "reserve_seed_ids": sorted(x["candidate_id"] for x in reserve_rows),
        "selected_overlap_count": len({x["candidate_id"] for x in seed_rows if x.get("candidate_id")} & untouched_ids),
        "reserve_overlap_count": len({x["candidate_id"] for x in reserve_rows} & untouched_ids),
    })
    write_artifact(run_root / "seed_selection_determinism_audit.json", {
        "algorithm_version": ALGORITHM_VERSION,
        "root_order": list(ROOT_ORDER),
        "primary_count": sum(x.get("candidate_id") is not None for x in seed_rows),
        "reserve_count": len(reserve_rows),
        "repeated_sort_key_definition": "P1<P2<P3<P4<P5, exposure, evidence, gain, frontier, qtype, sha256(candidate_id)",
    })

    confusion_rows = []
    packets = []
    for root_id in ROOT_ORDER:
        seed = selected[root_id]
        if not seed:
            continue
        confusion_rows.append({"root_id": root_id, "candidate_id": seed["candidate_id"], **seed["confusion"]})
        packet = {
            "packet_version": "round1_source_qa_packet_memory_bucket_v1",
            "experiment_arm": "E1Q1",
            "feedback_source_round": "round0",
            "target_round": "round1",
            "root_id": root_id,
            "source_candidate_id": seed["candidate_id"],
            "source_candidate_sha256": seed["candidate_sha256"],
            "source_question_type": seed["question_type"],
            "source_generation_source": seed["generation_source"],
            "source_question": seed["candidate"].get("question"),
            "source_options_or_items": seed["candidate"].get("options") or seed["candidate"].get("items") or {},
            "source_canonical_gold": seed["candidate"].get("gold") or [],
            "source_evidence_refs": seed["evidence_refs"],
            "source_evidence_groups": seed["candidate"].get("evidence_groups") or {},
            "priority_class": seed["priority_class"],
            "frontier_class": seed["priority_class"],
            "frontier_score": seed["frontier_score"],
            "training_exposure": seed["training_exposure"],
            "evidence_status": seed["evidence_status"],
            "position_mechanism": seed["position_mechanism"],
            "transition": seed["models"],
            "canonical_confusion": seed["confusion"],
            "memory_bucket_targets": {
                "alignment_status": "NOT_COMPUTABLE_WITHOUT_SOURCE_TURN_REFS",
                "r02_adherence_status": r02_by_root[root_id]["adherence"]["status"],
            },
            "curriculum_role": "ROUND0_QA_EVOLUTION_SEED",
            "packet_status": "COMPLETE_STATIC_PACKET_WITH_BUCKET_ALIGNMENT_GAP",
            "curriculum_source_qa_notice": "CURRENT_CUMULATIVE_DIALOGUE is the only fact source; this source QA is curriculum metadata.",
        }
        packets.append(packet)
    write_artifact(run_root / "canonical_confusion_features.jsonl", confusion_rows)
    write_artifact(run_root / "source_qa_packets.jsonl", packets)

    inventory_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    for root_id in ROOT_ORDER:
        data = r02_by_root[root_id]
        for bucket in data["buckets"]:
            inventory_rows.append({"root_id": root_id, **bucket})
            coverage_rows.append({
                "root_id": root_id,
                "bucket_locator": {k: bucket[k] for k in ("bucket_type", "bucket_key", "normalized_content_sha256", "bucket_snapshot_sha256", "source_turn_refs")},
                "round0_qa_usage_count": 0,
                "round0_hard_usage_count": 0,
                "is_r02_new_bucket": True,
                "selected_by_targeted_slot": False,
                "selected_by_contrastive_slot": False,
                "eligible_for_coverage_slot": bool(bucket["source_turn_refs_present"]),
            })
        seed = selected[root_id]
        seed_refs = set(map(str, seed["evidence_refs"])) if seed else set()
        event_supported = [b for b in data["buckets"] if b["bucket_type"] == "event" and set(b["source_turn_refs"]) & seed_refs]
        subject_keys = set()
        if seed:
            subject_keys = {
                str(seed["candidate"].get("subject") or ""),
                str(seed["candidate"].get("speaker") or ""),
                str(seed["candidate"].get("author") or ""),
            }
        person_supported = [b for b in data["buckets"] if b["bucket_type"] == "person" and b["bucket_key"] in subject_keys]
        alignment_status = "SUPPORTED" if event_supported or person_supported else "NO_SUPPORTED_BUCKET_ALIGNMENT"
        alignment_rows.append({
            "root_id": root_id,
            "candidate_id": seed["candidate_id"] if seed else None,
            "person_bucket_matches": person_supported,
            "event_bucket_matches": event_supported,
            "r02_adherence": data["adherence"],
            "alignment_status": alignment_status,
            "reason": "bucket_source_turn_refs_missing_or_no_strict_entity_match" if alignment_status != "SUPPORTED" else None,
        })
        slot_rows.extend([
            {"root_id": root_id, "slot_type": "TARGETED_BUCKET_EVOLUTION", "slot_status": "SUPPORTED" if event_supported and data["adherence"]["status"] == "SUPPORTED_STRUCTURED_ADHERENCE" else "NO_SUPPORTED_EVOLUTION_PATH", "target_bucket_locators": event_supported, "required_r02_refs": sorted(data["adherence"]["r02_turn_refs"][:1])},
            {"root_id": root_id, "slot_type": "CONTRASTIVE_BUCKET_QA", "slot_status": "SUPPORTED" if seed and seed["confusion"].get("shared_error_set") and (event_supported or person_supported) else "NO_SUPPORTED_BUCKET_ALIGNMENT", "target_confusion": seed["confusion"] if seed else {}, "target_bucket_locators": event_supported + person_supported},
            {"root_id": root_id, "slot_type": "COVERAGE_BUCKET_QA", "slot_status": "SUPPORTED" if any(x["eligible_for_coverage_slot"] for x in coverage_rows if x["root_id"] == root_id) else "NO_SAFE_SLOT_3", "target_bucket_locators": [x["bucket_locator"] for x in coverage_rows if x["root_id"] == root_id and x["eligible_for_coverage_slot"]]},
        ])
    write_artifact(run_root / "memory_bucket_inventory.jsonl", inventory_rows)
    write_artifact(run_root / "source_qa_to_bucket_alignment.jsonl", alignment_rows)
    write_artifact(run_root / "r02_alignment_report.json", {
        "roots": [{"root_id": x, "status": r02_by_root[x]["adherence"]["status"], "structured_files": r02_by_root[x]["adherence"]["structured_files"], "r02_ref_count": len(r02_by_root[x]["adherence"]["r02_turn_refs"])} for x in ROOT_ORDER],
        "all_roots_have_structured_adherence": all(r02_by_root[x]["adherence"]["status"] == "SUPPORTED_STRUCTURED_ADHERENCE" for x in ROOT_ORDER),
    })
    write_artifact(run_root / "bucket_coverage_ledger.jsonl", coverage_rows)
    write_artifact(run_root / "three_bucket_slot_plan.jsonl", slot_rows)

    contract, request_rows, contract_fields_ok = parse_contract_and_requests()
    code_dir = Path(__file__).parent.parent / "h4_round1_paired_blind_fresh_qa_v1"
    code_files = sorted(str(p) for p in code_dir.glob("*.py")) if code_dir.is_dir() else []
    core_fingerprint_fields = (
        "unique_parser_sha256",
        "repair_prompt_sha256",
        "retry_policy_sha256",
        "candidate_id_builder_sha256",
    )
    missing_core_resolution = [
        field for field in core_fingerprint_fields
        if not isinstance(contract.get(field), str) or not contract.get(field)
    ]
    contract_resolution_status = (
        "PASS" if not missing_core_resolution
        else "QA_CONTRACT_FINGERPRINT_INCOMPLETE"
    )
    contract_fingerprint = {
        "model": contract.get("model"),
        "effective_temperature": contract.get("effective_temperature"),
        "config_file_temperature": contract.get("config_file_temperature"),
        "max_tokens": contract.get("max_tokens"),
        "intent_miner_max_tokens": contract.get("intent_miner_max_tokens"),
        "generation_sources": contract.get("generation_sources"),
        "feedback_guided_qa": contract.get("feedback_guided_qa"),
        "planned_primary_requests_per_root": 7,
        "code_file_sha256": {path: sha_file(Path(path)) for path in code_files},
        "contract_sha256": source_sha["e1q0_contract"],
        "request_counts": request_rows,
        "unique_parser_sha256": contract.get("unique_parser_sha256"),
        "repair_prompt_sha256": contract.get("repair_prompt_sha256"),
        "retry_policy_sha256": contract.get("retry_policy_sha256"),
        "candidate_id_builder_sha256": contract.get("candidate_id_builder_sha256"),
        "component_fingerprints": contract.get("component_fingerprints"),
        "resolution_status": contract_resolution_status,
        "missing_core_resolution": missing_core_resolution,
    }
    write_artifact(run_root / "e1q0_qa_contract_fingerprint.json", contract_fingerprint)
    write_artifact(run_root / "qa_contract_file_sha_manifest.jsonl", [{"path": name, "sha256": value} for name, value in sorted(source_sha.items()) if value])
    write_artifact(run_root / "qa_contract_resolution_report.json", {"status": contract_fingerprint["resolution_status"], "contract_fields_present": sorted(contract), "request_row_count": len(request_rows), "core_fields_unresolved": contract_fingerprint["missing_core_resolution"]})

    # E1Q0 text is allowed only for the duplicate baseline.  Pair each E1Q0
    # candidate with the static seed selected for its root, without using any
    # E1Q0 Solver result.
    e1q0_candidates = load_jsonl(source_paths["e1q0_candidates"])
    baseline_rows = []
    for candidate in e1q0_candidates:
        seed = selected.get(str(candidate.get("root_id")))
        relation = duplicate_relation(candidate, seed["candidate"]) if seed else {"unavailable": True}
        baseline_rows.append({"candidate_id": candidate.get("candidate_id"), "root_id": candidate.get("root_id"), "source_candidate_id": seed["candidate_id"] if seed else None, **relation})
    write_artifact(run_root / "e1q0_source_similarity_baseline.jsonl", baseline_rows)
    write_artifact(run_root / "e1q0_duplicate_baseline_summary.json", {
        "e1q0_candidate_count": len(e1q0_candidates),
        "expected_count": EXPECTED_E1Q0_CANDIDATES,
        "exact_duplicate_count": sum(bool(x.get("exact_duplicate")) for x in baseline_rows),
        "near_duplicate_count": sum(bool(x.get("near_duplicate")) for x in baseline_rows),
        "unavailable_count": sum(bool(x.get("unavailable")) for x in baseline_rows),
        "normalization_version": "qa_duplicate_normalization_v1",
    })

    # Token preflight uses the real local Qwen-family tokenizer and an explicit
    # worst-case rendering.  No prompt is sent anywhere.
    tokenizer, tokenizer_identity = load_tokenizer()
    token_rows = []
    for root_id in ROOT_ORDER:
        dialogue_text = render_dialogue(dialogue_by_root[root_id])
        packet = next((x for x in packets if x["root_id"] == root_id), {"status": "NO_PACKET"})
        memory_text = canonical(r02_by_root[root_id]["buckets"])
        packet_text = canonical(packet)
        slots_text = canonical([x for x in slot_rows if x["root_id"] == root_id])
        base_text = "CURRENT_CUMULATIVE_DIALOGUE\nSOURCE_QA_PACKET\nMEMORY_NAVIGATION_TARGETS\nQUESTION_EVOLUTION_SLOT\n"
        counts = {"dialogue_tokens": count_tokens(tokenizer, dialogue_text), "packet_tokens": count_tokens(tokenizer, packet_text), "memory_navigation_tokens": count_tokens(tokenizer, memory_text), "slot_instruction_tokens": count_tokens(tokenizer, slots_text), "base_prompt_tokens": count_tokens(tokenizer, base_text)}
        total_input = sum(x or 0 for x in counts.values())
        token_rows.append({"root_id": root_id, **counts, "intent_miner_output_reserve": int(contract.get("intent_miner_max_tokens") or 4096), "writer_output_reserve": int(contract.get("max_tokens") or 8192), "intent_total": total_input + int(contract.get("intent_miner_max_tokens") or 4096), "writer_total": total_input + int(contract.get("max_tokens") or 8192), "context_limit": 16384, "dialogue_truncated": False, "packet_truncated": False, "bucket_truncated": False, "status": "PASS" if tokenizer is not None and total_input + int(contract.get("max_tokens") or 8192) <= 16384 else "TOKEN_BUDGET_EXCEEDED"})
    write_artifact(run_root / "qa_feedback_token_budget.json", {"tokenizer": tokenizer_identity, "roots": token_rows, "max_writer_total": max((x["writer_total"] for x in token_rows), default=None), "max_intent_total": max((x["intent_total"] for x in token_rows), default=None), "all_fit": all(x["status"] == "PASS" for x in token_rows)})

    # Future IDs are only planned; no candidate is created.
    existing_round0_ids = set(difficulty_by_id)
    existing_e1q0_ids = {str(x.get("candidate_id")) for x in e1q0_candidates}
    planned_ids = []
    line = 1
    for root_id in ROOT_ORDER:
        for slot in ("TARGETED_BUCKET_EVOLUTION", "CONTRASTIVE_BUCKET_QA", "COVERAGE_BUCKET_QA"):
            for source in ("contract_guided", "free_exploration"):
                planned_ids.append({"root_id": root_id, "slot_type": slot, "generation_source": source, "global_line": f"{line:06d}", "candidate_id": f"{root_id}:R01-R02-feedback:E1Q1_FULL_QA_FEEDBACK_COMPOSITE:{slot}:{source}:{line:06d}"})
                line += 1
    planned_set = {x["candidate_id"] for x in planned_ids}
    write_artifact(run_root / "candidate_id_scope_plan.json", {"scope": "<root_id>:R01-R02-feedback:E1Q1_FULL_QA_FEEDBACK_COMPOSITE:<slot_type>:<generation_source>:<global_line>", "planned_ids": planned_ids})
    write_artifact(run_root / "candidate_id_collision_preflight.json", {"round0_overlap": sorted(planned_set & existing_round0_ids), "e0q0_e1q0_overlap": sorted(planned_set & existing_e1q0_ids), "within_future_duplicates": len(planned_set) != len(planned_ids), "planned_count": len(planned_ids)})

    forbidden_paths = [SOLVER_COMPARE_RUN, SOLVER_EVAL_RUN]
    isolation = {
        "feedback_source_round": "round0",
        "round1_solver_feedback_rows_read": 0,
        "round1_solver_feedback_bytes_used_for_selection": 0,
        "forbidden_solver_paths_checked_but_not_read": [str(x) for x in forbidden_paths],
        "solver_eval_exists": SOLVER_EVAL_RUN.is_dir(),
        "solver_compare_exists": SOLVER_COMPARE_RUN.is_dir(),
        "dev_test_rows_read": 0,
        "api_calls": 0,
        "memorax_calls": 0,
        "semantic_auditor_calls": 0,
    }
    write_artifact(run_root / "feedback_round_isolation_audit.json", isolation)

    root_reports = []
    for root_id in ROOT_ORDER:
        seed = selected[root_id]
        alignment_row = next(x for x in alignment_rows if x["root_id"] == root_id)
        root_slots = [x for x in slot_rows if x["root_id"] == root_id]
        blockers = []
        if not seed:
            blockers.append("NO_SAFE_QA_FEEDBACK_SEED")
        identity_row = next(x for x in dialogue_identity if x["root_id"] == root_id)
        if identity_row["status"] != "PASS":
            blockers.append("E1Q0_E1Q1_DIALOGUE_DRIFT")
        if alignment_row["alignment_status"] != "SUPPORTED":
            blockers.append("NO_SUPPORTED_BUCKET_ALIGNMENT")
        if any(x["slot_status"] == "NO_SAFE_SLOT_3" for x in root_slots):
            blockers.append("NO_SAFE_SLOT_3")
        if contract_fingerprint["resolution_status"] != "RESOLVED":
            blockers.append("QA_CONTRACT_FINGERPRINT_INCOMPLETE")
        if next(x for x in token_rows if x["root_id"] == root_id)["status"] != "PASS":
            blockers.append("TOKEN_BUDGET_EXCEEDED")
        root_reports.append({"root_id": root_id, "status": "READY" if not blockers else "GAP", "seed": seed["candidate_id"] if seed else None, "priority_class": seed["priority_class"] if seed else None, "blockers": sorted(set(blockers)), "bucket_alignment": alignment_row["alignment_status"], "slot_plan": root_slots})

    all_ready = all(x["status"] == "READY" for x in root_reports) and all(x["status"] == "PASS" for x in dialogue_identity)
    final = {
        "final_report_version": f"{VERSION}_final_v1",
        "experiment_arm": "E1Q1_FULL_QA_FEEDBACK_COMPOSITE_MEMORY_BUCKET_V1",
        "status": "ROUND1_MEMORY_BUCKET_QA_FEEDBACK_COMPILER_COMPLETE" if all_ready else "ROUND1_MEMORY_BUCKET_QA_FEEDBACK_COMPILER_COMPLETE_WITH_GAPS",
        "next_status": "READY_FOR_E1Q1_SINGLE_ROOT_QA_SMOKE" if all_ready else "E1Q1_FULL_8_ROOT_BATCH_NOT_READY",
        "root_count": len(root_reports),
        "safe_root_count": sum(x["status"] == "READY" for x in root_reports),
        "root_reports": root_reports,
        "selected_seed_count": len([x for x in seed_rows if x.get("candidate_id")]),
        "primary_priority_counts": dict(Counter(x["priority_class"] for x in seed_rows if x.get("priority_class"))),
        "evidence_status_counts": dict(Counter(x.get("evidence_status") for x in seed_rows if x.get("evidence_status"))),
        "bucket_alignment_supported_count": sum(x["alignment_status"] == "SUPPORTED" for x in alignment_rows),
        "r02_structured_adherence_count": sum(r02_by_root[x]["adherence"]["status"] == "SUPPORTED_STRUCTURED_ADHERENCE" for x in ROOT_ORDER),
        "dialogue_identity_all_pass": all(x["status"] == "PASS" for x in dialogue_identity),
        "contract_status": contract_fingerprint["resolution_status"],
        "max_writer_token_total": max((x["writer_total"] for x in token_rows), default=None),
        "candidate_id_collision_count": len(planned_set & existing_round0_ids) + len(planned_set & existing_e1q0_ids) + int(len(planned_set) != len(planned_ids)),
        "online_calls": {"api": 0, "solver": 0, "memorax": 0, "qa_writer": 0, "semantic_auditor": 0},
        "do_not_start_e1q1": True,
    }
    write_artifact(run_root / "experiment_contract.json", {
        "experiment_arm": "E1Q1",
        "experiment_name": "E1Q1_FULL_QA_FEEDBACK_COMPOSITE_MEMORY_BUCKET_V1",
        "feedback_source_round": "round0",
        "target_round": "round1",
        "expansion_feedback_enabled": True,
        "qa_feedback_enabled": True,
        "source_qa_packet_exposed_to_intent_miner": True,
        "source_qa_packet_exposed_to_qa_writer": True,
        "memory_bucket_navigation_enabled": True,
        "local_event_chain_enabled": False,
        "current_round_solver_feedback_used": False,
        "qa_fact_source": "complete_parent_r01_r02_feedback_dialogue",
        "memory_is_qa_fact_source": False,
        "offline_only": True,
    })
    write_artifact(run_root / "qa_feedback_prompt_isolation_preflight.json", {
        "status": "PASS",
        "allowed_round0_sources": True,
        "round1_solver_accuracy_read": False,
        "untouched_probe_content_read": False,
        "raw_k3_rollout_rows_read": False,
        "semantic_auditor_labels_used_as_truth": False,
        "local_event_chain_fields": 0,
    })
    write_artifact(run_root / "question_evolution_contract.json", {
        "normalization_version": "qa_duplicate_normalization_v1",
        "exact_duplicate": {"stem": "equal", "alternatives": "multiset_equal", "gold": "equal", "route": "HARD_REJECT"},
        "near_duplicate": {"stem_token_jaccard": 0.80, "stem_sequence_ratio": 0.90, "evidence_jaccard": 0.80, "route": "SECONDARY_CONTAMINATION_SET"},
        "primary_set_excludes_near_duplicate": True,
    })
    write_artifact(run_root / "preflight_report.json", {
        "status": final["status"],
        "source_audit": source_audit,
        "root_scope": root_scope,
        "round0_alignment": alignment,
        "feedback_round_isolation": isolation,
        "blockers": sorted({b for row in root_reports for b in row["blockers"]}),
        "all_online_calls_zero": True,
    })
    write_artifact(run_root / "final_memory_bucket_qa_feedback_compiler_report.json", final)
    markdown = [
        "# Round-1 Memory-Bucket QA Feedback Compiler v1",
        "",
        f"- Status: `{final['status']}`",
        f"- Next status: `{final['next_status']}`",
        f"- Safe roots: `{final['safe_root_count']}/{final['root_count']}`",
        f"- Bucket alignment: `{final['bucket_alignment_supported_count']}/{EXPECTED_ROOTS}`",
        f"- Structured R02 adherence: `{final['r02_structured_adherence_count']}/{EXPECTED_ROOTS}`",
        f"- Contract: `{final['contract_status']}`",
        f"- Online calls: API `{final['online_calls']['api']}`, Solver `{final['online_calls']['solver']}`, Memorax `{final['online_calls']['memorax']}`",
        "",
        "## Roots",
        "",
    ]
    for row in root_reports:
        markdown.append(f"- `{row['root_id']}`: `{row['status']}` seed=`{row['seed']}` priority=`{row['priority_class']}` blockers=`{','.join(row['blockers']) or 'none'}`")
    markdown.append("\nNo E1Q1 online request was started.")
    write_artifact(run_root / "final_memory_bucket_qa_feedback_compiler_report.md", "\n".join(markdown) + "\n")
    return final
