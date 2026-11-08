#!/usr/bin/env python3
"""Replay one accepted Stage3 Claim-first sample without any network access.

This adapter exists for a deliberately narrow migration step: a dialogue that
was already generated and aligned is kept byte-for-byte/semantically frozen,
while the local Claim-first compiler is replayed against its immutable Stage64
target.  Sibling QAs are checked independently and may be dropped; the source
dialogue and its graph are never rewritten.

The program installs a process-local socket guard before compilation.  A
network attempt is therefore both counted and rejected, and a successful
report always records zero attempted API/network calls.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import socket
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import (  # noqa: E402
    recalibrate_stage64_shuffled_frontier as stage64_recalibrate,
)
from scripts.current_method import stage3_claim_proof_compiler  # noqa: E402
from scripts.current_method import stage3_claim_semantic_validator  # noqa: E402
from scripts.current_method import stage3_claim_validator  # noqa: E402


SCHEMA_VERSION = "stage3_claim_first_target_adapter_replay_v1"
DEFAULT_SOURCE_RUN = (
    ROOT
    / "base_restart_current_method_validation"
    / "224_stage3_claim_first_smoke_n1"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "base_restart_current_method_validation"
    / "225_stage3_claim_first_target_adapter_no_api"
)


class OfflineNetworkAttempt(RuntimeError):
    """Raised if replay code attempts to leave the local process."""


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path) -> dict[str, str]:
    """Hash every source file so replay cannot silently mutate run224."""

    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@contextlib.contextmanager
def offline_network_guard() -> Iterator[list[dict[str, Any]]]:
    """Reject and record socket-based network attempts during replay."""

    attempts: list[dict[str, Any]] = []

    def blocked(*args: Any, **kwargs: Any) -> Any:
        destination = None
        if len(args) >= 2:
            destination = repr(args[1])
        elif args:
            destination = repr(args[0])
        attempts.append({"destination": destination, "kwargs": sorted(kwargs)})
        raise OfflineNetworkAttempt(
            "network access is forbidden during Stage3 offline replay"
        )

    with (
        mock.patch.object(socket.socket, "connect", blocked),
        mock.patch.object(socket.socket, "connect_ex", blocked),
        mock.patch("socket.create_connection", blocked),
    ):
        yield attempts


def _load_source(source_run: Path) -> dict[str, Any]:
    generation = source_run / "generation"
    validated_path = generation / "graph_conditioned_api_dialogue_validated.jsonl"
    clean_path = generation / "clean_samples.jsonl"
    if not validated_path.is_file() or not clean_path.is_file():
        raise ValueError("run224 source is missing validated or clean samples")
    validated_rows = _read_jsonl(validated_path)
    clean_rows = _read_jsonl(clean_path)
    if len(validated_rows) != 1 or len(clean_rows) != 1:
        raise ValueError(
            "offline n=1 replay requires exactly one validated and one clean row; "
            f"found validated={len(validated_rows)}, clean={len(clean_rows)}"
        )
    validated = validated_rows[0]
    source_clean = clean_rows[0]
    embedded_clean = validated.get("clean_sample")
    if not isinstance(embedded_clean, dict):
        raise ValueError("validated row has no embedded clean_sample")
    if embedded_clean != source_clean:
        raise ValueError("validated embedded clean_sample differs from clean_samples.jsonl")
    plan = validated.get("current_event_plan")
    alignment = validated.get("current_alignment")
    validated_dialogue = validated.get("clean_dialogue")
    if not isinstance(plan, dict) or not isinstance(alignment, dict):
        raise ValueError("validated row lacks current_event_plan/current_alignment")
    if not isinstance(validated_dialogue, (dict, list)):
        raise ValueError("validated row lacks clean_dialogue")
    dialogue = source_clean.get("dialogue")
    if not isinstance(dialogue, (dict, list)):
        raise ValueError("clean sample lacks dialogue")
    # The validated envelope adds only ``frozen_dialogue=True``.  Compare the
    # durable payload rather than requiring those two envelopes to have the
    # same bookkeeping keys.
    if isinstance(dialogue, dict) and isinstance(validated_dialogue, dict):
        if dialogue.get("turns") != validated_dialogue.get("turns") or dialogue.get(
            "dialogue_hash"
        ) != validated_dialogue.get("dialogue_hash"):
            raise ValueError(
                "validated clean_dialogue payload differs from clean sample dialogue"
            )
    elif dialogue != validated_dialogue:
        raise ValueError(
            "validated clean_dialogue payload differs from clean sample dialogue"
        )
    source_bundle = source_clean.get("claim_first_multiqa_bundle")
    if not isinstance(source_bundle, dict):
        raise ValueError("source clean sample has no Claim-first bundle")
    claim_ir = source_bundle.get("claim_ir")
    if not isinstance(claim_ir, dict):
        raise ValueError("source Claim-first bundle has no claim_ir")
    target = source_clean.get("stage64_target")
    if not isinstance(target, dict):
        raise ValueError("source clean sample has no immutable Stage64 target")
    return {
        "validated_path": validated_path,
        "clean_path": clean_path,
        "validated": validated,
        "source_clean": source_clean,
        "source_bundle": source_bundle,
        "claim_ir": claim_ir,
        "target": target,
        "plan": plan,
        "alignment": alignment,
        "dialogue": dialogue,
    }


def _claim_validator_report(
    claim_ir: dict[str, Any], qa: dict[str, Any], dialogue: dict[str, Any] | list[Any]
) -> dict[str, Any]:
    return stage3_claim_validator.validation_report(
        {
            "claims": copy.deepcopy(claim_ir.get("claims") or []),
            "proof": copy.deepcopy(qa.get("proof") or {}),
            "support_ledger": copy.deepcopy(claim_ir.get("support_ledger") or {}),
            "dialogue": copy.deepcopy(dialogue),
        }
    )


def independently_filter_qas(
    *,
    claim_ir: dict[str, Any],
    qas: list[dict[str, Any]],
    dialogue: dict[str, Any] | list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate each QA in isolation; a failed sibling cannot delete others."""

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw_qa in qas:
        qa = copy.deepcopy(raw_qa)
        proof_report = _claim_validator_report(claim_ir, qa, dialogue)
        semantic_report = stage3_claim_semantic_validator.validate_qa_semantics(
            claim_ir, qa, dialogue
        )
        qa["independent_validation_report"] = copy.deepcopy(proof_report)
        qa["independent_semantic_validation_report"] = copy.deepcopy(
            semantic_report
        )
        if proof_report.get("pass") is True and semantic_report.get("pass") is True:
            accepted.append(qa)
        else:
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "qa_id": qa.get("qa_id"),
                    "rejection_stage": "independent_sibling_semantic_gate",
                    "proof_validator_report": proof_report,
                    "semantic_validator_report": semantic_report,
                    "qa": qa,
                    "dialogue_retained": True,
                    "sibling_qas_retained": True,
                    "api_calls_made": 0,
                }
            )
    return accepted, rejected


def _event_ids(values: Any) -> list[str]:
    return list(dict.fromkeys(_normalize(item) for item in (values or []) if _normalize(item)))


def render_legacy_primary_qa(primary: dict[str, Any]) -> dict[str, Any]:
    """Render a target-compatible compiler QA into the old Stage64 envelope."""

    qa = copy.deepcopy(primary)
    question = _normalize(qa.get("question") or qa.get("question_text_draft"))
    if not question:
        raise ValueError("target primary has no question surface")
    qa["question"] = question
    qa["question_text"] = question
    qa["question_type"] = "single_choice"
    qa["answer_format"] = "single_choice_mcq"
    qa["reasoning_subtype"] = _normalize(
        qa.get("reasoning_subtype") or qa.get("qa_subtype")
    )
    query_plan = qa.get("query_plan")
    if not isinstance(query_plan, dict):
        raise ValueError("target primary compiler did not emit query_plan")
    qa["query_plan"] = copy.deepcopy(query_plan)
    options: list[dict[str, Any]] = []
    gold_labels = {_normalize(item) for item in qa.get("gold") or []}
    evidence_turn_ids: list[str] = []
    ledger_by_claim: dict[str, list[str]] = {}
    for entry in qa.get("evidence_ledger") or []:
        if not isinstance(entry, dict):
            continue
        claim_id = _normalize(entry.get("claim_id"))
        turn_id = _normalize(entry.get("turn_id"))
        if claim_id and turn_id:
            ledger_by_claim.setdefault(claim_id, []).append(turn_id)
            if turn_id not in evidence_turn_ids:
                evidence_turn_ids.append(turn_id)
    for raw_option in qa.get("options") or []:
        if not isinstance(raw_option, dict):
            raise ValueError("target primary contains a non-object option")
        option = copy.deepcopy(raw_option)
        option_id = _normalize(option.get("id") or option.get("option_id"))
        claim_id = _normalize(option.get("source_claim_id"))
        event_id = _normalize(
            option.get("source_event_instance_id")
            or (qa.get("claim_event_map") or {}).get(claim_id)
        )
        if not option_id or not event_id:
            raise ValueError(
                f"target primary option lacks lossless graph identity: {option_id!r}"
            )
        option.update(
            {
                "id": option_id,
                "option_id": option_id,
                "source_event_instance_id": event_id,
                "anchor_ref": event_id,
                "anchor_kind": "event",
                "is_gold": option_id in gold_labels,
                "truth_value": option_id in gold_labels,
                "answer_type": qa.get("answer_type"),
                "turn_ids": list(dict.fromkeys(ledger_by_claim.get(claim_id, []))),
            }
        )
        options.append(option)
    qa["options"] = options
    qa["evidence_turn_ids"] = evidence_turn_ids
    qa["gold_source"] = "claim_first_local_immutable_target_compiler"
    qa["question_surface_source"] = "local_frozen_skeleton"
    qa["generated_by_api_stage64_schema"] = False
    qa["graph_validation_pass"] = True
    qa["semantic_contract_frozen"] = True
    qa["anchor_refs"] = {
        "support": _event_ids(
            query_plan.get("resolved_supporting_event_instance_ids")
            or query_plan.get("supporting_anchor_refs")
        ),
        "gold": _event_ids(
            query_plan.get("resolved_gold_event_instance_ids")
            or query_plan.get("gold_anchor_refs")
        ),
        "distractors": _event_ids(
            query_plan.get("resolved_distractor_event_instance_ids")
            or query_plan.get("distractor_anchor_refs")
        ),
    }
    return qa


def _with_primary_sample(
    *,
    source_clean: dict[str, Any],
    compiled: dict[str, Any],
    accepted_qas: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    primary: dict[str, Any],
) -> dict[str, Any]:
    sample = copy.deepcopy(source_clean)
    rendered = render_legacy_primary_qa(primary)
    bundle = copy.deepcopy(source_clean["claim_first_multiqa_bundle"])
    bundle.update(
        {
            "qas": copy.deepcopy(accepted_qas),
            "qa_count": len(accepted_qas),
            "validator_pass_qa_count": len(accepted_qas),
            "validator_reject_qa_count": len(rejects),
            "qa_reject_count": len(rejects),
            "qa_rejects": copy.deepcopy(rejects),
            "proof_ready": len(accepted_qas) >= 3,
            "target_fulfillment_qa_id": compiled.get("target_fulfillment_qa_id"),
            "target_fulfillment_status": compiled.get("target_fulfillment_status"),
            "immutable_stage64_target": copy.deepcopy(
                compiled.get("immutable_stage64_target")
            ),
            "offline_target_adapter_replay": True,
            "api_calls_made": 0,
            "api_qa_calls_made": 0,
        }
    )
    sample["claim_first_multiqa_bundle"] = bundle
    sample["claim_first_qa_count"] = len(accepted_qas)
    sample["claim_first_primary_qa_id"] = primary.get("qa_id")
    sample["qa"] = rendered
    # Preserve the legacy duplicated QA fields used by Stage64 mining.
    for field in (
        "question_type",
        "question",
        "question_text",
        "options",
        "gold",
        "gold_answer_text",
        "normalized_gold_answer",
        "qa_operation",
        "qa_subtype",
        "reasoning_subtype",
        "reasoning_template",
        "hop_count",
        "answer_type",
        "answer_format",
        "hard_negative_strategy",
        "forbid_max_time_shortcut",
        "query_plan",
        "anchor_refs",
        "evidence_turn_ids",
    ):
        if field in rendered:
            sample[field] = copy.deepcopy(rendered[field])
    sample["gold_answer"] = rendered.get("gold_answer_text")
    sample["gold_answers"] = [rendered.get("gold_answer_text")]
    sample["gold_options"] = copy.deepcopy(rendered.get("gold") or [])
    sample["gold_option"] = (rendered.get("gold") or [None])[0]
    return sample


def _target_mismatches(primary: dict[str, Any], target: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    fields = (
        "qa_operation",
        "reasoning_subtype",
        "reasoning_template",
        "hop_count",
        "hard_negative_strategy",
    )
    for field in fields:
        if primary.get(field) != target.get(field):
            mismatches.append(field)
    preferred = target.get("preferred_answer_type")
    if preferred and primary.get("answer_type") != preferred:
        mismatches.append("preferred_answer_type")
    if target.get("forbid_max_time_shortcut") is True and primary.get(
        "forbid_max_time_shortcut"
    ) is not True:
        mismatches.append("forbid_max_time_shortcut")
    return mismatches


def build_replay(source_run: Path = DEFAULT_SOURCE_RUN, *, min_qas: int = 3) -> dict[str, Any]:
    """Build the run224 target-adapter replay entirely in memory."""

    source_run = source_run.resolve()
    manifest_before = source_manifest(source_run)
    source = _load_source(source_run)
    invariant_before = {
        "current_event_plan": _json_sha256(source["plan"]),
        "graph_update": _json_sha256(source["source_clean"].get("graph_update")),
        "dialogue": _json_sha256(source["dialogue"]),
        "alignment": _json_sha256(source["alignment"]),
        "validated_hashes": _json_sha256(source["validated"].get("hashes")),
        "claim_ir": _json_sha256(source["claim_ir"]),
        "claim_bundle_hash": _normalize(source["claim_ir"].get("claim_bundle_hash")),
    }
    claim_ir_input_hash = _json_sha256(source["claim_ir"])
    with offline_network_guard() as network_attempts:
        compiled = stage3_claim_proof_compiler.compile_claim_first_qas(
            source["claim_ir"],
            seed_sample_id=_normalize(source["source_clean"].get("sample_id")),
            lineage_group_id=_normalize(
                source["source_clean"].get("lineage_group")
                or source["source_clean"].get("parent_sample_id")
            ),
            min_qas=min_qas,
            max_qas=6,
            require_support=True,
            immutable_stage64_target=copy.deepcopy(source["target"]),
        )
        if _json_sha256(source["claim_ir"]) != claim_ir_input_hash:
            raise RuntimeError("target compiler mutated the frozen Claim IR")
        accepted_qas, rejected_qas = independently_filter_qas(
            claim_ir=source["claim_ir"],
            qas=[copy.deepcopy(item) for item in compiled.get("qas") or []],
            dialogue=source["dialogue"],
        )
        primary_id = _normalize(compiled.get("target_fulfillment_qa_id"))
        primary = next(
            (
                item
                for item in accepted_qas
                if _normalize(item.get("qa_id")) == primary_id
            ),
            None,
        )
        if not primary_id or primary is None:
            raise RuntimeError(
                "immutable target primary was not retained by independent semantic gates"
            )
        if _normalize(primary.get("qa_id")) != primary_id:
            raise RuntimeError("target_fulfillment_qa_id did not select the primary QA")
        mismatches = _target_mismatches(primary, source["target"])
        if mismatches:
            raise RuntimeError(
                f"compiled primary does not fulfill immutable target fields: {mismatches}"
            )
        if len(accepted_qas) < min_qas:
            raise RuntimeError(
                f"only {len(accepted_qas)} semantic-clean sibling QAs; need {min_qas}"
            )
        clean_sample = _with_primary_sample(
            source_clean=source["source_clean"],
            compiled=compiled,
            accepted_qas=accepted_qas,
            rejects=rejected_qas,
            primary=primary,
        )
        stage64_clean, stage64_rejects, stage64_summary = (
            stage64_recalibrate.quality_filter_samples([clean_sample])
        )
        if len(stage64_clean) != 1 or stage64_rejects:
            issues = (
                stage64_rejects[0].get("stage64_quality_rejection", {})
                if stage64_rejects
                else {}
            )
            raise RuntimeError(f"primary failed legacy Stage64 quality: {issues}")
        clean_sample = stage64_clean[0]

    if network_attempts:
        raise RuntimeError(f"offline replay attempted network access: {network_attempts}")
    invariant_after = {
        "current_event_plan": _json_sha256(source["plan"]),
        "graph_update": _json_sha256(clean_sample.get("graph_update")),
        "dialogue": _json_sha256(clean_sample.get("dialogue")),
        "alignment": _json_sha256(source["alignment"]),
        "validated_hashes": _json_sha256(source["validated"].get("hashes")),
        "claim_ir": _json_sha256(
            clean_sample.get("claim_first_multiqa_bundle", {}).get("claim_ir")
        ),
        "claim_bundle_hash": _normalize(
            clean_sample.get("claim_first_multiqa_bundle", {})
            .get("claim_ir", {})
            .get("claim_bundle_hash")
        ),
    }
    if invariant_before != invariant_after:
        raise RuntimeError(
            "graph/dialogue/alignment/frozen-hash invariant changed during replay"
        )
    manifest_after = source_manifest(source_run)
    if manifest_before != manifest_after:
        raise RuntimeError("source run files changed during offline replay")
    source_bundle_qa_ids = {
        _normalize(item.get("qa_id"))
        for item in source["source_bundle"].get("qas") or []
        if isinstance(item, dict)
    }
    accepted_ids = [_normalize(item.get("qa_id")) for item in accepted_qas]
    rejected_ids = [_normalize(item.get("qa_id")) for item in rejected_qas]
    semantic_issue_counts: Counter[str] = Counter()
    for row in rejected_qas:
        report = row.get("semantic_validator_report") or {}
        semantic_issue_counts.update(report.get("issue_counts_by_code") or {})
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_run": str(source_run),
        "offline_only": True,
        "api_calls_made": 0,
        "network_attempt_count": len(network_attempts),
        "source_run_immutable": True,
        "source_manifest_sha256_before": _json_sha256(manifest_before),
        "source_manifest_sha256_after": _json_sha256(manifest_after),
        "source_file_count": len(manifest_before),
        "graph_dialogue_alignment_hashes_before": invariant_before,
        "graph_dialogue_alignment_hashes_after": invariant_after,
        "graph_dialogue_alignment_hashes_unchanged": True,
        "claim_bundle_hash_stable": (
            invariant_before["claim_bundle_hash"]
            == invariant_after["claim_bundle_hash"]
        ),
        "compiled_qa_count": len(compiled.get("qas") or []),
        "semantic_clean_qa_count": len(accepted_qas),
        "semantic_rejected_qa_count": len(rejected_qas),
        "semantic_clean_qa_ids": accepted_ids,
        "semantic_rejected_qa_ids": rejected_ids,
        "semantic_issue_counts_by_code": dict(sorted(semantic_issue_counts.items())),
        "dialogue_retained_after_sibling_rejection": True,
        "source_bundle_qa_ids": sorted(source_bundle_qa_ids),
        "target_fulfillment_qa_id": primary_id,
        "primary_qa_id": primary.get("qa_id"),
        "primary_matches_immutable_target": True,
        "primary_legacy_stage64_quality_pass": True,
        "stage64_quality_summary": stage64_summary,
        "minimum_semantic_clean_qas": min_qas,
        "all_acceptance_criteria_pass": True,
    }
    return {
        "clean_samples": [clean_sample],
        "sibling_qas_clean": [
            {
                "schema_version": SCHEMA_VERSION,
                "source_sample_id": source["source_clean"].get("sample_id"),
                "dialogue_family_id": compiled.get("dialogue_family_id"),
                "qa": copy.deepcopy(qa),
                "api_calls_made": 0,
            }
            for qa in accepted_qas
        ],
        "qa_rejects": rejected_qas,
        "report": report,
    }


def markdown_report(result: dict[str, Any], output_dir: Path) -> str:
    report = result["report"]
    return "\n".join(
        [
            "# Stage3 Claim-first run224 target-adapter offline replay",
            "",
            "## Result",
            "",
            f"- immutable target primary: `{report['primary_qa_id']}`",
            f"- old Stage64 quality pass: `{report['primary_legacy_stage64_quality_pass']}`",
            f"- semantic-clean sibling QAs: `{report['semantic_clean_qa_count']}`",
            f"- independently rejected sibling QAs: `{report['semantic_rejected_qa_count']}`",
            f"- graph/dialogue/alignment/hash unchanged: `{report['graph_dialogue_alignment_hashes_unchanged']}`",
            f"- source run immutable: `{report['source_run_immutable']}`",
            "- API/network calls: `0`",
            "",
            "A rejected sibling is written to `qa_rejects.jsonl`; it does not delete",
            "the frozen dialogue or any independently valid sibling QA.",
            "",
            "## Outputs",
            "",
            f"- `{output_dir / 'clean_samples.jsonl'}`",
            f"- `{output_dir / 'sibling_qas_clean.jsonl'}`",
            f"- `{output_dir / 'qa_rejects.jsonl'}`",
            f"- `{output_dir / 'report.json'}`",
        ]
    ) + "\n"


def write_result(
    result: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR
) -> None:
    output_dir = output_dir.resolve()
    source_run = Path(result["report"]["source_run"]).resolve()
    if output_dir == source_run or source_run in output_dir.parents:
        raise ValueError("output directory must be outside immutable source run")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "clean_samples.jsonl", result["clean_samples"])
    _write_jsonl(
        output_dir / "sibling_qas_clean.jsonl", result["sibling_qas_clean"]
    )
    _write_jsonl(output_dir / "qa_rejects.jsonl", result["qa_rejects"])
    _write_json(output_dir / "report.json", result["report"])
    (output_dir / "report.md").write_text(
        markdown_report(result, output_dir), encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-qas", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_replay(args.source_run, min_qas=args.min_qas)
    write_result(result, args.output_dir)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
