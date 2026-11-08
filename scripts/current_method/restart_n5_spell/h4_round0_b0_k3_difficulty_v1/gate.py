"""Pure-program candidate gate and frozen Solver contract/plan builders."""

from __future__ import annotations

import importlib.metadata
import inspect
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from ..core import normalize_option, sha256_bytes, sha256_file
from ..prompts import (
    POSITION_NEUTRAL_RESPONDER_PROMPT_VERSION,
    SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT,
    SOLVER_TRAILING_FORMAT_INSTRUCTION,
    SOLVER_TRAILING_FORMAT_VERSION,
)
from ..structured_qa_box_protocol import structured_box_choices
from . import (
    BASE_SOLVER_MODEL,
    BASE_TOKENIZER,
    FORBIDDEN_SOLVER_MODEL,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    PARSER_SINGLE,
    PARSER_STRUCTURED,
    QUESTION_KEYS,
    ROLLOUTS,
    ROOT_SPLIT_NAMESPACE,
    SEED_SCHEDULE_VERSION,
    SINGLE_TEMPERATURE,
    STRUCTURED_OUTPUT_VERSION,
    STRUCTURED_TEMPERATURE,
    TOP_P,
    question_keys,
)
from .contract import (
    CANONICAL_SCORING_VERSION,
    PERMUTATION_CONTRACT_VERSION,
    PROMPT_CONTRACT_VERSION,
    STRUCTURED_SYSTEM_PROMPT,
    audit_rebinding,
    build_prompt,
    dialogue_rows,
    position_variants,
)
from .io_utils import load_json, stable_hash_int


PROGRAM_PASS = "PROGRAM_PASS"
PROGRAM_REJECT = "PROGRAM_REJECT"
CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"


def _mapping(candidate: Mapping[str, Any], field: str) -> dict[str, str]:
    value = candidate.get(field)
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(text) for key, text in value.items()}


def validate_candidate_structure(
    candidate: Mapping[str, Any], *, root_id: str, dialogue_root_id: str,
) -> list[str]:
    """Apply only mechanical checks; semantic/evidence fields are ignored."""
    errors: list[str] = []
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        errors.append("candidate_id_missing")
    elif not candidate_id.startswith(root_id + ":"):
        errors.append("candidate_id_root_prefix_mismatch")
    if str(candidate.get("root_id") or "") != root_id:
        errors.append("candidate_root_mismatch")
    if dialogue_root_id != root_id:
        errors.append("dialogue_root_mismatch")
    if not isinstance(candidate.get("question"), str) or not candidate.get("question", "").strip():
        errors.append("question_missing")
    qtype = str(candidate.get("question_type") or "")
    if qtype not in QUESTION_KEYS:
        errors.append("unsupported_question_type")
        return errors
    field = "items" if qtype == "ordering" else "options"
    other_field = "options" if qtype == "ordering" else "items"
    values = _mapping(candidate, field)
    other_values = candidate.get(other_field)
    canonical = question_keys(qtype, len(values))
    expected = set(canonical)
    if not canonical or tuple(values) != canonical:
        errors.append(f"{field}_keys")
    normalized = [normalize_option(value) for value in values.values()]
    if len(normalized) != len(expected) or any(not value for value in normalized):
        errors.append(f"{field}_empty")
    if len(normalized) != len(set(normalized)):
        errors.append(f"{field}_duplicate_text")
    if not isinstance(other_values, Mapping) or other_values:
        errors.append(f"{other_field}_must_be_empty_mapping")
    raw_gold = candidate.get("gold")
    gold = [str(value) for value in raw_gold] if isinstance(raw_gold, list) else []
    if qtype == "single_choice":
        correct = str(candidate.get("correct_option") or "")
        if correct not in expected:
            errors.append("single_correct_option")
        if len(gold) != 1 or gold != [correct]:
            errors.append("single_gold_exactly_one_and_bound")
    elif qtype == "multi_select":
        if candidate.get("correct_option") not in {None, ""}:
            errors.append("multi_correct_option_must_be_null")
        if not 2 <= len(gold) <= len(expected) or len(gold) != len(set(gold)) or any(value not in expected for value in gold):
            errors.append("multi_gold_nonempty_unique_labels")
    else:
        if candidate.get("correct_option") not in {None, ""}:
            errors.append("ordering_correct_option_must_be_null")
        if len(gold) != len(expected) or set(gold) != expected:
            errors.append("ordering_gold_complete_permutation")
    return errors


def gate_candidate(
    candidate: Mapping[str, Any],
    expanded: Mapping[str, Any],
    tokenizer: Any,
    *,
    root_id: str,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "")
    qtype = str(candidate.get("question_type") or "")
    errors = validate_candidate_structure(
        candidate,
        root_id=root_id,
        dialogue_root_id=str(expanded.get("root_id") or ""),
    )
    base: dict[str, Any] = {
        "candidate_id": candidate_id,
        "root_id": root_id,
        "question_type": qtype,
        "gate_status": PROGRAM_REJECT if errors else None,
        "program_errors": errors,
        "semantic_judgment_applied": False,
        "evidence_gate_applied": False,
        "training_admission_applied": False,
    }
    if errors:
        return base
    try:
        variants = position_variants(candidate)
        rebinding = audit_rebinding(candidate, variants)
    except (KeyError, TypeError, ValueError) as error:
        base.update({
            "gate_status": PROGRAM_REJECT,
            "program_errors": [f"permutation_contract:{type(error).__name__}:{error}"],
        })
        return base
    if not rebinding["pass"]:
        base.update({
            "gate_status": PROGRAM_REJECT,
            "program_errors": list(rebinding["errors"]),
            "gold_rebinding_audit": rebinding,
        })
        return base
    dialogue = dialogue_rows(expanded)
    prompt_rows: list[dict[str, Any]] = []
    try:
        for variant in variants:
            for context_mode in ("no_context", "full_context"):
                prompt = build_prompt(
                    tokenizer, candidate, variant, dialogue, context_mode=context_mode,
                )
                count = len(tokenizer.encode(prompt, add_special_tokens=False))
                prompt_rows.append({
                    "variant_id": variant["variant_id"],
                    "context_mode": context_mode,
                    "prompt_token_count": count,
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "fits_context": count + MAX_TOKENS <= MAX_MODEL_LEN,
                })
    except Exception as error:
        base.update({
            "gate_status": PROGRAM_REJECT,
            "program_errors": [f"prompt_contract:{type(error).__name__}:{error}"],
            "gold_rebinding_audit": rebinding,
        })
        return base
    overflow = [row for row in prompt_rows if not row["fits_context"]]
    base.update({
        "gate_status": CONTEXT_OVERFLOW if overflow else PROGRAM_PASS,
        "program_errors": ["full_prompt_context_overflow"] if overflow else [],
        "gold_rebinding_audit": rebinding,
        "position_count": len(variants),
        "prompt_count": len(prompt_rows),
        "min_prompt_token_count": min(row["prompt_token_count"] for row in prompt_rows),
        "max_prompt_token_count": max(row["prompt_token_count"] for row in prompt_rows),
        "output_reserve": MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "prompt_audit": prompt_rows,
    })
    return base


def run_program_gate(inventory: Mapping[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for source in inventory.get("roots") or []:
        if int(source.get("candidate_count", 0)) <= 0:
            continue
        pool_path = Path(str(source["candidate_pool_path"]))
        dialogue_path = Path(str(source["dialogue_path"]))
        if sha256_file(pool_path) != source["candidate_pool_sha256"]:
            raise RuntimeError(f"source_candidate_pool_drift:{source['root_id']}")
        if sha256_file(dialogue_path) != source["dialogue_sha256"]:
            raise RuntimeError(f"source_dialogue_drift:{source['root_id']}")
        candidates = load_json(pool_path)
        expanded = load_json(dialogue_path)
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            if candidate_id in seen:
                duplicates.append(candidate_id)
            seen.add(candidate_id)
            row = gate_candidate(
                candidate, expanded, tokenizer, root_id=str(source["root_id"]),
            )
            row.update({
                "candidate_pool_path": str(pool_path),
                "candidate_pool_sha256": source["candidate_pool_sha256"],
                "dialogue_path": str(dialogue_path),
                "dialogue_sha256": source["dialogue_sha256"],
            })
            rows.append(row)
    if duplicates:
        raise RuntimeError(f"duplicate_candidate_ids_during_program_gate:{duplicates[:10]}")
    gate_counts = Counter(row["gate_status"] for row in rows)
    by_type = Counter(
        row["question_type"] for row in rows if row["gate_status"] == PROGRAM_PASS
    )
    report = {
        "program_gate_version": "h4_round0_program_gate_v1",
        "candidate_count": len(rows),
        "gate_counts": {
            status: gate_counts[status]
            for status in (PROGRAM_PASS, PROGRAM_REJECT, CONTEXT_OVERFLOW)
        },
        "program_pass_by_question_type": dict(sorted(by_type.items())),
        "max_observed_prompt_tokens": max(
            (int(row.get("max_prompt_token_count", 0)) for row in rows), default=0,
        ),
        "semantic_judgment_applied": False,
        "evidence_gate_applied": False,
        "training_admission_applied": False,
        "solver_calls": 0,
    }
    return rows, report


def assert_solver_paths(model_path: Path, tokenizer_path: Path) -> tuple[Path, Path]:
    model = model_path.resolve()
    tokenizer = tokenizer_path.resolve()
    if model == FORBIDDEN_SOLVER_MODEL.resolve():
        raise RuntimeError(f"forbidden_solver_model:{model}")
    if model != BASE_SOLVER_MODEL.resolve():
        raise RuntimeError(f"solver_model_must_be_base_qwen3_4b:{model}")
    if tokenizer != BASE_TOKENIZER.resolve():
        raise RuntimeError(f"tokenizer_must_be_base_qwen3_4b:{tokenizer}")
    if not model.is_dir() or not tokenizer.is_dir():
        raise FileNotFoundError("base_solver_model_or_tokenizer_missing")
    return model, tokenizer


def directory_identity(path: Path) -> dict[str, Any]:
    files = []
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        relative_name = child.relative_to(path).as_posix()
        files.append({
            "name": relative_name,
            "size_bytes": child.stat().st_size,
            "sha256": sha256_file(child),
        })
    return {"path": str(path), "files": files}


def _contract_source_identity() -> list[dict[str, Any]]:
    package = Path(__file__).resolve().parent
    parent = package.parent
    package_sources = sorted(
        path for path in package.glob("*.py") if not path.name.startswith("test_")
    )
    shared_sources = [
        parent / "backends.py",
        parent / "core.py",
        parent / "prompts.py",
        parent / "structured_qa_box_protocol.py",
        parent / "vllm_backends.py",
    ]
    return [
        {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
        for path in package_sources + shared_sources
    ]


def output_vocabulary_budget(tokenizer: Any) -> dict[str, Any]:
    vocabularies = {
        "single_choice": [f"<box>{key}</box>" for key in QUESTION_KEYS["single_choice"]],
        "multi_select": structured_box_choices(
            "multi_select", allowed_item_ids=question_keys("multi_select", 6),
        ),
        "ordering": structured_box_choices(
            "ordering", allowed_item_ids=question_keys("ordering", 6),
        ),
    }
    rows: dict[str, Any] = {}
    for question_type, choices in vocabularies.items():
        lengths = [
            len(tokenizer.encode(choice, add_special_tokens=False)) for choice in choices
        ]
        if not lengths:
            raise RuntimeError(f"legal_output_vocabulary_empty:{question_type}")
        maximum = max(lengths)
        if maximum >= MAX_TOKENS:
            raise RuntimeError(
                f"legal_output_may_hit_generation_limit:{question_type}:"
                f"{maximum}:{MAX_TOKENS}"
            )
        rows[question_type] = {
            "legal_output_count": len(choices),
            "max_legal_output_tokens": maximum,
            "max_tokens": MAX_TOKENS,
            "minimum_token_margin": MAX_TOKENS - maximum,
        }
    return rows


@lru_cache(maxsize=1)
def detect_structured_output_api() -> tuple[str, str]:
    """Return the finite-choice API exposed by the installed vLLM."""
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("vllm_package_missing") from error
    from vllm import SamplingParams, sampling_params

    parameters = inspect.signature(SamplingParams).parameters
    if (
        hasattr(sampling_params, "StructuredOutputsParams")
        and "structured_outputs" in parameters
    ):
        return "structured_outputs=StructuredOutputsParams(choice)", version
    if (
        hasattr(sampling_params, "GuidedDecodingParams")
        and "guided_decoding" in parameters
    ):
        return "guided_decoding=GuidedDecodingParams(choice)", version
    raise RuntimeError(f"vllm_structured_choice_api_missing:{version}")


def build_solver_contract(
    tokenizer: Any,
    *,
    model_path: Path = BASE_SOLVER_MODEL,
    tokenizer_path: Path = BASE_TOKENIZER,
) -> dict[str, Any]:
    model, tokenizer_dir = assert_solver_paths(model_path, tokenizer_path)
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    if not chat_template:
        raise RuntimeError("tokenizer_chat_template_missing")
    structured_api, vllm_version = detect_structured_output_api()
    checkpoint_identity = directory_identity(model)
    tokenizer_identity = (
        checkpoint_identity if tokenizer_dir == model else directory_identity(tokenizer_dir)
    )
    return {
        "solver_contract_version": "h4_round0_b0_k3_solver_contract_v1",
        "solver_checkpoint": str(model),
        "tokenizer": str(tokenizer_dir),
        "forbidden_solver_model": str(FORBIDDEN_SOLVER_MODEL.resolve()),
        "forbidden_solver_model_references_allowed": 0,
        "checkpoint_identity": checkpoint_identity,
        "tokenizer_identity": tokenizer_identity,
        "contract_source_files": _contract_source_identity(),
        "output_vocabulary_budget": output_vocabulary_budget(tokenizer),
        "max_model_len": MAX_MODEL_LEN,
        "chat_template": chat_template,
        "chat_template_sha256": sha256_bytes(chat_template.encode("utf-8")),
        "enable_thinking": False,
        "add_generation_prompt": True,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "permutation_contract_version": PERMUTATION_CONTRACT_VERSION,
        "canonical_scoring_version": CANONICAL_SCORING_VERSION,
        "single_system_prompt_version": POSITION_NEUTRAL_RESPONDER_PROMPT_VERSION,
        "single_system_prompt": SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT,
        "single_user_format_version": SOLVER_TRAILING_FORMAT_VERSION,
        "single_user_format": SOLVER_TRAILING_FORMAT_INSTRUCTION,
        "structured_system_prompt": STRUCTURED_SYSTEM_PROMPT,
        "parsers": {
            "single_choice": PARSER_SINGLE,
            "multi_select": PARSER_STRUCTURED,
            "ordering": PARSER_STRUCTURED,
        },
        "temperatures": {
            "single_choice": SINGLE_TEMPERATURE,
            "multi_select": STRUCTURED_TEMPERATURE,
            "ordering": STRUCTURED_TEMPERATURE,
        },
        "generation_config": {
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "single_choice_structured_output": False,
            "structured_output_api": structured_api,
            "structured_output_version": STRUCTURED_OUTPUT_VERSION,
            "vllm_version": vllm_version,
            "structured_stop": ["</box>"],
            "structured_include_stop_str_in_output": True,
            "vllm_use_flashinfer_sampler": "0",
        },
        "seed_schedule": {
            "version": SEED_SCHEDULE_VERSION,
            "inputs": ["root_id", "candidate_id", "slot_id"],
            "excludes": ["context_mode", "position"],
        },
        "rollouts_per_position": ROLLOUTS,
        "context_modes": ["no_context", "full_context"],
        "position_counts": {
            "single_choice": {"fixed": 4},
            "multi_select": {"min": 4, "max": 6, "source": "candidate"},
            "ordering": {"min": 4, "max": 6, "source": "candidate"},
        },
        "output_mode": "B0_label",
    }


def build_solver_plan(
    gate_rows: list[Mapping[str, Any]], split_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    split_by_root = {str(row["root_id"]): str(row["split"]) for row in split_rows}
    passed = [row for row in gate_rows if row.get("gate_status") == PROGRAM_PASS]

    def request_count(row: Mapping[str, Any]) -> int:
        # Historical gate manifests predate position_count. New v1.1 gate rows
        # always carry it, so only immutable B0 rows use this compatibility value.
        positions = int(row.get("position_count") or len(QUESTION_KEYS[str(row["question_type"])]))
        return positions * 2 * ROLLOUTS

    by_type = Counter(str(row["question_type"]) for row in passed)
    by_split = Counter(split_by_root[str(row["root_id"])] for row in passed)
    request_by_type = {
        qtype: sum(request_count(row)
                   for row in passed if row["question_type"] == qtype)
        for qtype in QUESTION_KEYS
    }
    roots_by_type: dict[str, set[str]] = {qtype: set() for qtype in QUESTION_KEYS}
    for row in passed:
        roots_by_type[str(row["question_type"])].add(str(row["root_id"]))
    smoke_eligible = set.intersection(*(values for values in roots_by_type.values()))
    smoke_roots = sorted(
        (root for root in smoke_eligible if split_by_root[root] == "train"),
        key=lambda root: (
            stable_hash_int(ROOT_SPLIT_NAMESPACE, "smoke", root), root,
        ),
    )[:2]
    if len(smoke_roots) != 2:
        raise RuntimeError("two_type_complete_train_smoke_roots_required")
    smoke_candidates: list[dict[str, str]] = []
    for root_id in smoke_roots:
        for qtype in QUESTION_KEYS:
            choices = [
                row for row in passed
                if row["root_id"] == root_id and row["question_type"] == qtype
            ]
            selected = min(
                choices,
                key=lambda row: (
                    stable_hash_int(ROOT_SPLIT_NAMESPACE, "smoke", row["candidate_id"]),
                    row["candidate_id"],
                ),
            )
            smoke_candidates.append({
                "root_id": root_id,
                "candidate_id": str(selected["candidate_id"]),
                "question_type": qtype,
                "position_count": int(selected.get("position_count") or len(QUESTION_KEYS[qtype])),
            })
    smoke_requests = sum(
        int(row["position_count"]) * 2 * ROLLOUTS for row in smoke_candidates
    )
    return {
        "solver_plan_version": "h4_round0_b0_k3_solver_plan_v1",
        "program_pass_candidate_count": len(passed),
        "program_pass_by_question_type": dict(sorted(by_type.items())),
        "program_pass_by_split": dict(sorted(by_split.items())),
        "requests_per_candidate": "candidate_position_count * 2_context_modes * 3_rollouts",
        "request_count_by_question_type": request_by_type,
        "planned_request_count": sum(request_by_type.values()),
        "planned_unique_prompt_count": sum(request_by_type.values()) // ROLLOUTS,
        "smoke_root_ids": smoke_roots,
        "smoke_candidates": smoke_candidates,
        "smoke_candidate_count": len(smoke_candidates),
        "smoke_request_count": smoke_requests,
        "smoke_requests_reused_by_batch": True,
        "only_b0": True,
        "semantic_audit": False,
        "strict_broad_classification": False,
        "training": False,
    }


__all__ = [
    "CONTEXT_OVERFLOW",
    "PROGRAM_PASS",
    "PROGRAM_REJECT",
    "assert_solver_paths",
    "build_solver_contract",
    "build_solver_plan",
    "detect_structured_output_api",
    "directory_identity",
    "gate_candidate",
    "output_vocabulary_budget",
    "run_program_gate",
    "validate_candidate_structure",
]
