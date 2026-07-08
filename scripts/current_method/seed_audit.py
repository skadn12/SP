from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ALLOWED_SAMPLE_NAMES = {f"sample{i}" for i in range(6)}
SEED_SCOPE = "sample0_5_train_only"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_dialogue_path(row: dict[str, Any]) -> str:
    for value in (
        row.get("source_dialogue_path"),
        row.get("metadata", {}).get("source_dialogue_path") if isinstance(row.get("metadata"), dict) else None,
        row.get("state_chain", {}).get("source_dialogue_path") if isinstance(row.get("state_chain"), dict) else None,
    ):
        if value:
            return str(value)
    return ""


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata") if isinstance(row.get("metadata"), dict) else {}


def _seed_source(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}


def _qa(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("qa") if isinstance(row.get("qa"), dict) else {}


def _validation(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("validation") if isinstance(row.get("validation"), dict) else {}


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _normalize_split(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    lower = text.lower()
    if lower in {"train", "train_candidate"}:
        return "train"
    if lower in {"dev", "dev_sanity_eval_only"}:
        return "dev"
    if lower in {"heldout", "heldout_eval_only", "test"}:
        return "heldout"
    return lower


def _normalize_sample_name(value: Any) -> str | None:
    text = _normalize_text(value)
    return text.lower() if text else None


def _normalize_training_use(value: Any) -> str | None:
    text = _normalize_text(value)
    return text.lower() if text else None


def _path_source_metadata(row: dict[str, Any]) -> dict[str, str]:
    source_path = source_dialogue_path(row)
    source_split = "unknown"
    sample_name = "unknown"
    match = re.search(r"/(train|dev|test|heldout)/(sample\d+)/", source_path)
    if match:
        source_split = _normalize_split(match.group(1)) or "unknown"
        sample_name = _normalize_sample_name(match.group(2)) or "unknown"
    else:
        sample_match = re.search(r"/(sample\d+)/", source_path)
        if sample_match:
            sample_name = _normalize_sample_name(sample_match.group(1)) or "unknown"
    return {
        "source_dialogue_path": source_path,
        "source_split": source_split,
        "sample_name": sample_name,
    }


def _pick_first(
    candidates: Iterable[tuple[str, Any]],
    *,
    normalizer,
    unknown_value: Any,
) -> tuple[Any, str]:
    for source_name, value in candidates:
        normalized = normalizer(value)
        if normalized is not None:
            return normalized, source_name
    return unknown_value, "unknown"


def _extract_fields(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(row)
    seed_source = _seed_source(row)
    qa = _qa(row)
    path_meta = _path_source_metadata(row)

    split, split_source = _pick_first(
        (
            ("row.split", row.get("split")),
            ("metadata.split", metadata.get("split")),
            ("seed_source.split", seed_source.get("split")),
            ("frontier_from_split", row.get("frontier_from_split")),
            ("row.source_split", row.get("source_split")),
            ("metadata.source_split", metadata.get("source_split")),
            ("qa.split", qa.get("split")),
            ("legacy.source_dialogue_path.split", path_meta.get("source_split")),
        ),
        normalizer=_normalize_split,
        unknown_value="unknown",
    )
    training_use, training_use_source = _pick_first(
        (
            ("row.training_use", row.get("training_use")),
            ("metadata.training_use", metadata.get("training_use")),
            ("seed_source.raw_training_use", seed_source.get("raw_training_use")),
            ("seed_source.training_use", seed_source.get("training_use")),
        ),
        normalizer=_normalize_training_use,
        unknown_value="unknown",
    )
    source_split, source_split_source = _pick_first(
        (
            ("row.source_split", row.get("source_split")),
            ("metadata.source_split", metadata.get("source_split")),
            ("seed_source.source_split", seed_source.get("source_split")),
            ("frontier_from_split", row.get("frontier_from_split")),
            ("legacy.source_dialogue_path.split", path_meta.get("source_split")),
        ),
        normalizer=_normalize_split,
        unknown_value="unknown",
    )
    sample_name, sample_name_source = _pick_first(
        (
            ("row.sample_name", row.get("sample_name")),
            ("metadata.sample_name", metadata.get("sample_name")),
            ("seed_source.sample_name", seed_source.get("sample_name")),
            ("frontier_from_sample_name", row.get("frontier_from_sample_name")),
            ("row.seed_sample_name", row.get("seed_sample_name")),
            ("metadata.seed_sample_name", metadata.get("seed_sample_name")),
            ("row.parent_sample_name", row.get("parent_sample_name")),
            ("metadata.parent_sample_name", metadata.get("parent_sample_name")),
            ("legacy.source_dialogue_path.sample_name", path_meta.get("sample_name")),
        ),
        normalizer=_normalize_sample_name,
        unknown_value="unknown",
    )
    heldout_split, heldout_split_source = _pick_first(
        (
            ("row.heldout_split", row.get("heldout_split")),
            ("metadata.heldout_split", metadata.get("heldout_split")),
            ("seed_source.heldout_split", seed_source.get("heldout_split")),
        ),
        normalizer=_normalize_split,
        unknown_value=None,
    )
    return {
        "split": split,
        "split_source": split_source,
        "raw_training_use": training_use,
        "training_use_source": training_use_source,
        "heldout_split": heldout_split,
        "heldout_split_source": heldout_split_source,
        "source_split": source_split,
        "source_split_source": source_split_source,
        "sample_name": sample_name,
        "sample_name_source": sample_name_source,
        "source_dialogue_path": path_meta["source_dialogue_path"],
    }


def _explicit_bool(mapping: dict[str, Any], key: str) -> bool | None:
    if key not in mapping:
        return None
    value = mapping.get(key)
    if value is True:
        return True
    if value is False:
        return False
    return None


def _append_forbidden_if_present(
    reasons: list[str],
    mapping: dict[str, Any],
    *,
    key: str,
    expected: bool,
    reason: str,
) -> None:
    actual = _explicit_bool(mapping, key)
    if actual is expected:
        reasons.append(reason)


def extract_seed_source(
    row: dict[str, Any],
    *,
    allowed_sample_names: Iterable[str] = DEFAULT_ALLOWED_SAMPLE_NAMES,
) -> dict[str, Any]:
    allowed = set(allowed_sample_names)
    extracted = _extract_fields(row)
    seed_source = _seed_source(row)
    validation = _validation(row)
    forbidden_reasons: list[str] = []
    if extracted["split"] != "train":
        forbidden_reasons.append(f"split={extracted['split']}")
    if extracted["source_split"] != "train":
        forbidden_reasons.append(f"source_split={extracted['source_split']}")
    if extracted["heldout_split"] is not None:
        forbidden_reasons.append(f"heldout_split={extracted['heldout_split']}")
    if extracted["sample_name"] not in allowed:
        forbidden_reasons.append(f"sample_name={extracted['sample_name']}")

    if _explicit_bool(seed_source, "allowed_train_seed") is False:
        forbidden_reasons.append("seed_source.allowed_train_seed=false")
    seed_source_forbidden_reasons = seed_source.get("forbidden_reasons")
    if isinstance(seed_source_forbidden_reasons, list) and seed_source_forbidden_reasons:
        forbidden_reasons.append("seed_source.forbidden_reasons_present")

    _append_forbidden_if_present(
        forbidden_reasons, row, key="heldout_contamination", expected=True, reason="row.heldout_contamination=true"
    )
    _append_forbidden_if_present(
        forbidden_reasons,
        validation,
        key="heldout_contamination",
        expected=True,
        reason="validation.heldout_contamination=true",
    )
    _append_forbidden_if_present(
        forbidden_reasons, row, key="unconstrained_llm_gold", expected=True, reason="row.unconstrained_llm_gold=true"
    )
    _append_forbidden_if_present(
        forbidden_reasons,
        validation,
        key="unconstrained_llm_gold",
        expected=True,
        reason="validation.unconstrained_llm_gold=true",
    )
    _append_forbidden_if_present(
        forbidden_reasons,
        row,
        key="gold_derivable_from_graph_or_state_chain",
        expected=False,
        reason="row.gold_derivable_from_graph_or_state_chain=false",
    )
    _append_forbidden_if_present(
        forbidden_reasons,
        validation,
        key="gold_derivable_from_graph_or_state_chain",
        expected=False,
        reason="validation.gold_derivable_from_graph_or_state_chain=false",
    )
    _append_forbidden_if_present(
        forbidden_reasons,
        row,
        key="option_validation_pass",
        expected=False,
        reason="row.option_validation_pass=false",
    )
    _append_forbidden_if_present(
        forbidden_reasons,
        validation,
        key="option_validation_pass",
        expected=False,
        reason="validation.option_validation_pass=false",
    )
    _append_forbidden_if_present(
        forbidden_reasons, row, key="leakage_free", expected=False, reason="row.leakage_free=false"
    )
    _append_forbidden_if_present(
        forbidden_reasons,
        validation,
        key="leakage_free",
        expected=False,
        reason="validation.leakage_free=false",
    )
    _append_forbidden_if_present(
        forbidden_reasons,
        row,
        key="graph_dialogue_alignment",
        expected=False,
        reason="row.graph_dialogue_alignment=false",
    )
    _append_forbidden_if_present(
        forbidden_reasons,
        validation,
        key="graph_dialogue_alignment",
        expected=False,
        reason="validation.graph_dialogue_alignment=false",
    )
    return {
        **extracted,
        "allowed_train_seed": not forbidden_reasons,
        "forbidden_reasons": forbidden_reasons,
    }


def build_seed_source_audit(
    rows: list[dict[str, Any]],
    *,
    seed_source: str | Path,
    run_name: str,
    round_id: str,
    parent_round_id: str | None,
    allowed_sample_names: Iterable[str] = DEFAULT_ALLOWED_SAMPLE_NAMES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed = set(allowed_sample_names)
    metas = [extract_seed_source(row, allowed_sample_names=allowed) for row in rows]
    raw_split_counts = Counter(meta["split"] for meta in metas)
    raw_training_use_counts = Counter(meta["raw_training_use"] for meta in metas)
    raw_sample_name_counts = Counter(meta["sample_name"] for meta in metas)
    raw_source_split_counts = Counter(meta["source_split"] for meta in metas)
    extracted_field_sources = {
        "split": dict(sorted(Counter(meta["split_source"] for meta in metas).items())),
        "training_use": dict(sorted(Counter(meta["training_use_source"] for meta in metas).items())),
        "source_split": dict(sorted(Counter(meta["source_split_source"] for meta in metas).items())),
        "sample_name": dict(sorted(Counter(meta["sample_name_source"] for meta in metas).items())),
        "heldout_split": dict(sorted(Counter(meta["heldout_split_source"] for meta in metas).items())),
    }
    schema_compatibility_mode = (
        "current_method_generation"
        if any(
            isinstance(row.get("seed_source"), dict)
            or row.get("frontier_from_split") is not None
            or row.get("frontier_from_sample_name") is not None
            for row in rows
        )
        else "legacy_source_dialogue_path"
    )
    if parent_round_id and str(parent_round_id).startswith("current_method_r"):
        seed_parent_round_accepted = schema_compatibility_mode == "current_method_generation"
        if seed_parent_round_accepted:
            seed_parent_round_reason = (
                f"accepted parent round {parent_round_id} using current_method generation schema compatibility"
            )
        else:
            seed_parent_round_reason = (
                f"parent round {parent_round_id} expects current_method generation schema, but it was not detected"
            )
    else:
        seed_parent_round_accepted = True
        seed_parent_round_reason = (
            f"accepted parent round {parent_round_id}" if parent_round_id else "accepted without explicit parent round"
        )

    forbidden = [meta for meta in metas if meta["allowed_train_seed"] is not True]
    allowed_rows = [row for row, meta in zip(rows, metas) if meta["allowed_train_seed"] is True]
    sample_name_counts = Counter(meta["sample_name"] for meta in metas if meta["allowed_train_seed"] is True)
    train_seed_count = sum(1 for meta in metas if meta["allowed_train_seed"] is True)
    dev_seed_count = sum(1 for meta in metas if meta["split"] == "dev")
    heldout_seed_count = sum(
        1 for meta in metas if meta["split"] == "heldout" or meta["source_split"] == "heldout" or meta["heldout_split"] is not None
    )
    unknown_split_count = raw_split_counts.get("unknown", 0)
    unknown_sample_name_count = raw_sample_name_counts.get("unknown", 0)

    audit = {
        "method": "current_method",
        "run_name": run_name,
        "round": round_id,
        "parent_round": parent_round_id,
        "seed_source": str(seed_source),
        "seed_scope": SEED_SCOPE,
        "raw_seed_count": len(rows),
        "raw_split_counts": dict(sorted(raw_split_counts.items())),
        "raw_training_use_counts": dict(sorted(raw_training_use_counts.items())),
        "raw_source_split_counts": dict(sorted(raw_source_split_counts.items())),
        "raw_sample_name_counts": dict(sorted(raw_sample_name_counts.items())),
        "train_seed_count": train_seed_count,
        "dev_seed_count": dev_seed_count,
        "heldout_seed_count": heldout_seed_count,
        "sample_name_counts": dict(sorted(sample_name_counts.items())),
        "allowed_sample_names": sorted(allowed),
        "forbidden_seed_count": len(forbidden),
        "forbidden_reason_counts": dict(
            sorted(Counter(reason for meta in forbidden for reason in meta["forbidden_reasons"]).items())
        ),
        "schema_compatibility_mode": schema_compatibility_mode,
        "seed_parent_round_accepted": seed_parent_round_accepted,
        "seed_parent_round_reason": seed_parent_round_reason,
        "extracted_field_sources": extracted_field_sources,
        "unknown_split_count": unknown_split_count,
        "unknown_sample_name_count": unknown_sample_name_count,
        "pass": (
            train_seed_count > 0
            and dev_seed_count == 0
            and heldout_seed_count == 0
            and len(forbidden) == 0
        ),
    }
    if not audit["pass"]:
        allowed_rows = []
    return audit, allowed_rows


def audit_seed_file(
    seed_path: str | Path,
    *,
    report_path: str | Path,
    run_name: str,
    round_id: str,
    parent_round_id: str | None,
    allowed_sample_names: Iterable[str] = DEFAULT_ALLOWED_SAMPLE_NAMES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(seed_path)
    audit, allowed_rows = build_seed_source_audit(
        rows,
        seed_source=seed_path,
        run_name=run_name,
        round_id=round_id,
        parent_round_id=parent_round_id,
        allowed_sample_names=allowed_sample_names,
    )
    write_json(report_path, audit)
    if audit.get("forbidden_seed_count") != 0:
        raise RuntimeError(f"Seed source audit failed: {report_path}")
    if audit.get("pass") is not True:
        raise RuntimeError(f"Seed source audit did not pass: {report_path}")
    return audit, allowed_rows
