#!/usr/bin/env python3
"""Build the Stage3 Claim-first no-API replay corpus from a frozen run.

The source run is treated as immutable evidence.  For every record that
reached the frozen final-QA stage this program:

1. reconstructs and validates the saved graph/dialogue/alignment triple;
2. compiles graph-only claims and 3--6 frozen QA proofs before support binding;
3. recompiles the same claims against the saved dialogue and requires the
   semantic claim hash to remain identical;
4. binds exact support quotes; and
5. independently validates every QA proof, without allowing one failed QA to
   delete its siblings or the dialogue.

No model endpoint is called.  The output is an audit/replay dataset, not a
replacement for the source run and not a final naturalized training set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import audit_stage3_fastlane_salvage as salvage  # noqa: E402
from scripts.current_method import stage3_claim_ir as claim_ir  # noqa: E402
from scripts.current_method import stage3_claim_proof_compiler as proof_compiler  # noqa: E402
from scripts.current_method import stage3_claim_validator as claim_validator  # noqa: E402


SCHEMA_VERSION = "stage3_claim_first_no_api_replay_v1"
DEFAULT_RUN_DIR = (
    ROOT
    / "base_restart_current_method_validation"
    / "175_stage3_backfilled_smoke_n10"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "base_restart_current_method_validation"
    / "223_stage3_claim_ir_no_api"
)


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_id(raw_line_number: int, raw_record: dict[str, Any]) -> str:
    return "claim_replay_" + _json_hash(
        {
            "raw_line_number": raw_line_number,
            "attempt": raw_record.get("attempt"),
            "generation_slot_id": raw_record.get("generation_slot_id"),
            "seed_sample_id": raw_record.get("seed_sample_id"),
        }
    )[:20]


def _qa_instance_id(candidate_id: str, dialogue_hash: Any, qa_id: Any) -> str:
    """Address one semantic QA against one concrete dialogue realization."""

    return "qa_instance_" + _json_hash(
        {
            "candidate_id": candidate_id,
            "dialogue_hash": _normalize(dialogue_hash),
            "qa_id": _normalize(qa_id),
        }
    )[:24]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return salvage.read_jsonl(path)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_tree_manifest(run_dir: Path) -> dict[str, str]:
    """Hash every regular source-run file, using stable relative paths."""

    result: dict[str, str] = {}
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        result[path.relative_to(run_dir).as_posix()] = salvage.file_sha256(path)
    return result


def source_tree_hash(manifest: dict[str, str]) -> str:
    return _json_hash(manifest)


def frozen_candidate_rows(
    raw_rows: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    return [
        (line_number, row)
        for line_number, row in enumerate(raw_rows, start=1)
        if isinstance(row.get("api_stage64_final_qa_attempts"), list)
        and bool(row.get("api_stage64_final_qa_attempts"))
    ]


def _base_metadata(
    *,
    run_dir: Path,
    raw_line_number: int,
    raw_record: dict[str, Any],
) -> dict[str, Any]:
    target = (
        raw_record.get("stage64_target")
        if isinstance(raw_record.get("stage64_target"), dict)
        else {}
    )
    return {
        "source_run_dir": str(run_dir),
        "source_raw_line_number": raw_line_number,
        "candidate_id": _candidate_id(raw_line_number, raw_record),
        "attempt": raw_record.get("attempt"),
        "generation_slot_id": raw_record.get("generation_slot_id"),
        "target_index": raw_record.get("target_index"),
        "seed_sample_id": raw_record.get("seed_sample_id"),
        "source_sample_id": raw_record.get("sample_id"),
        "source_qa_operation": target.get("qa_operation"),
        "source_reasoning_template": target.get("reasoning_template"),
    }


def _validator_input(
    post_bundle: dict[str, Any],
    qa: dict[str, Any],
    final_dialogue: dict[str, Any],
) -> dict[str, Any]:
    """Construct the deliberately small independent-validator contract."""

    return {
        "claims": copy.deepcopy(post_bundle.get("claims") or []),
        "proof": copy.deepcopy(qa.get("proof") or {}),
        "support_ledger": copy.deepcopy(post_bundle.get("support_ledger") or {}),
        "dialogue": copy.deepcopy(final_dialogue),
    }


def _reject_row(
    metadata: dict[str, Any],
    *,
    stage: str,
    reason: str,
    qa: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    details: Any = None,
) -> dict[str, Any]:
    row = {
        "schema_version": SCHEMA_VERSION,
        **copy.deepcopy(metadata),
        "rejection_stage": stage,
        "rejection_reason": reason,
        "qa_id": _normalize((qa or {}).get("qa_id")) or None,
        "dialogue_family_id": _normalize((qa or {}).get("dialogue_family_id"))
        or None,
        "dialogue_family_split": _normalize(
            (qa or {}).get("dialogue_family_split")
        )
        or None,
        "validator_report": copy.deepcopy(validation),
        "details": copy.deepcopy(details),
        "qa": copy.deepcopy(qa),
        "api_calls_made": 0,
    }
    return row


def build_replay(
    run_dir: Path,
    *,
    expected_candidate_count: int | None = 10,
    min_qas: int = 3,
    max_qas: int = 6,
    eval_per_mille: int = 100,
) -> dict[str, Any]:
    """Replay a frozen Stage3 run and return all no-API output tables."""

    if min_qas < 1 or max_qas < min_qas:
        raise ValueError("invalid min_qas/max_qas")
    run_dir = run_dir.resolve()
    paths = salvage.source_paths(run_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"source run is incomplete; missing={missing}")

    source_manifest_before = source_tree_manifest(run_dir)
    raw_rows = _read_jsonl(paths["raw"])
    parents_by_id = salvage.parent_index(_read_jsonl(paths["parents"]))
    candidates = frozen_candidate_rows(raw_rows)
    if expected_candidate_count is not None and len(candidates) != expected_candidate_count:
        raise ValueError(
            f"expected {expected_candidate_count} frozen final-QA candidates, "
            f"found {len(candidates)}"
        )

    claim_bundle_rows: list[dict[str, Any]] = []
    proof_ready_qas: list[dict[str, Any]] = []
    qa_rejects: list[dict[str, Any]] = []
    dialogue_rows: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()

    for raw_line_number, raw_record in candidates:
        metadata = _base_metadata(
            run_dir=run_dir,
            raw_line_number=raw_line_number,
            raw_record=raw_record,
        )
        dialogue_row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            **copy.deepcopy(metadata),
            "replay_pass": False,
            "claim_ir_compiled": False,
            "claim_bundle_hash_stable": False,
            "precompiled_qa_count": 0,
            "support_bound_qa_count": 0,
            "validator_pass_qa_count": 0,
            "validator_reject_qa_count": 0,
            "proof_ready": False,
            "proof_ready_qa_ids": [],
            "dialogue_family_id": None,
            "dialogue_family_split": None,
            "api_calls_made": 0,
        }

        try:
            replay = salvage.replay_candidate(raw_record, run_dir, parents_by_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            dialogue_row["failure_stage"] = "frozen_replay"
            dialogue_row["failure_reason"] = error
            qa_rejects.append(
                _reject_row(
                    metadata,
                    stage="frozen_replay",
                    reason="frozen_candidate_replay_failed",
                    details=error,
                )
            )
            dialogue_rows.append(dialogue_row)
            continue
        dialogue_row["replay_pass"] = bool(replay.get("pass"))
        dialogue_row["frozen_hashes"] = copy.deepcopy(replay.get("actual_hashes"))
        dialogue_row["expected_frozen_hashes"] = copy.deepcopy(
            replay.get("expected_hashes")
        )
        dialogue_row["artifact_id"] = replay.get("artifact_id")
        dialogue_row["artifact_path"] = replay.get("artifact_path")
        dialogue_row["artifact_sha256"] = replay.get("artifact_sha256")
        dialogue_row["reconstruction"] = copy.deepcopy(replay.get("reconstruction"))
        if not replay.get("pass"):
            dialogue_row["failure_stage"] = "frozen_replay"
            dialogue_row["failure_reason"] = "saved graph/dialogue/alignment gate failed"
            qa_rejects.append(
                _reject_row(
                    metadata,
                    stage="frozen_replay",
                    reason="frozen_candidate_replay_failed",
                    details={
                        "validator_pass": replay.get("validator_pass"),
                        "validator_reason": replay.get("validator_reason"),
                        "hash_match": replay.get("hash_match"),
                    },
                )
            )
            dialogue_rows.append(dialogue_row)
            continue

        try:
            pre_bundle = claim_ir.compile_claim_ir(replay["current_event_plan"])
            pre_qas = proof_compiler.compile_claim_first_qas(
                pre_bundle,
                require_support=False,
                min_qas=min_qas,
                max_qas=max_qas,
                seed_sample_id=_normalize(raw_record.get("seed_sample_id")),
                lineage_group_id=_normalize(raw_record.get("seed_sample_id")),
                eval_per_mille=eval_per_mille,
            )
            if int(pre_qas.get("api_calls_made") or 0) != 0:
                raise RuntimeError("proof compiler reported a nonzero API call count")
            post_bundle = claim_ir.compile_claim_ir(
                replay["current_event_plan"],
                current_alignment=replay["current_alignment"],
                dialogue_turns=replay["final_dialogue"],
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            dialogue_row["failure_stage"] = "claim_or_proof_compile"
            dialogue_row["failure_reason"] = error
            qa_rejects.append(
                _reject_row(
                    metadata,
                    stage="claim_or_proof_compile",
                    reason="pre_dialogue_claim_or_proof_compile_failed",
                    details=error,
                )
            )
            dialogue_rows.append(dialogue_row)
            continue

        pre_hash = _normalize(pre_bundle.get("claim_bundle_hash"))
        post_hash = _normalize(post_bundle.get("claim_bundle_hash"))
        hash_stable = bool(pre_hash) and pre_hash == post_hash
        if not hash_stable:
            # This is a semantic freeze violation, not an ordinary bad QA.  A
            # partial output would be unsafe because every proof was compiled
            # against a graph identity that changed after naturalization.
            raise RuntimeError(
                "pre/post claim_bundle_hash changed for "
                f"{metadata['candidate_id']}: pre={pre_hash!r}, post={post_hash!r}"
            )

        claim_bundle_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                **copy.deepcopy(metadata),
                "claim_bundle_hash": pre_hash,
                "claim_bundle_hash_stable": True,
                "pre_claim_bundle": copy.deepcopy(pre_bundle),
                "post_claim_bundle": copy.deepcopy(post_bundle),
                "api_calls_made": 0,
            }
        )
        dialogue_row["claim_ir_compiled"] = True
        dialogue_row["claim_bundle_hash_stable"] = True
        dialogue_row["claim_bundle_hash"] = pre_hash
        dialogue_row["precompiled_qa_count"] = int(pre_qas.get("qa_count") or 0)
        dialogue_row["dialogue_family_id"] = pre_qas.get("dialogue_family_id")
        dialogue_row["dialogue_family_split"] = pre_qas.get(
            "dialogue_family_split"
        )
        dialogue_row["final_dialogue"] = copy.deepcopy(replay["final_dialogue"])
        dialogue_row["current_alignment"] = copy.deepcopy(
            replay["current_alignment"]
        )

        try:
            bound = proof_compiler.bind_compiled_qas_to_support(
                pre_qas, post_bundle
            )
            if int(bound.get("api_calls_made") or 0) != 0:
                raise RuntimeError("support binder reported a nonzero API call count")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            dialogue_row["failure_stage"] = "support_binding"
            dialogue_row["failure_reason"] = error
            for qa in pre_qas.get("qas") or []:
                qa_rejects.append(
                    _reject_row(
                        metadata,
                        stage="support_binding",
                        reason="dialogue_support_binding_failed",
                        qa=qa,
                        details=error,
                    )
                )
            dialogue_rows.append(dialogue_row)
            continue

        dialogue_row["support_bound_qa_count"] = int(bound.get("qa_count") or 0)
        for item in bound.get("qa_support_rejects") or []:
            qa_id = _normalize(item.get("qa_id"))
            source_qa = next(
                (
                    qa
                    for qa in pre_qas.get("qas") or []
                    if _normalize(qa.get("qa_id")) == qa_id
                ),
                None,
            )
            qa_rejects.append(
                _reject_row(
                    metadata,
                    stage="support_binding",
                    reason=_normalize(item.get("reason"))
                    or "required_claim_support_missing",
                    qa=source_qa,
                    details=item,
                )
            )

        validator_pass_rows: list[dict[str, Any]] = []
        validator_fail_count = 0
        for qa in bound.get("qas") or []:
            validation = claim_validator.validation_report(
                _validator_input(post_bundle, qa, replay["final_dialogue"])
            )
            issue_counts.update(validation.get("issue_counts_by_code") or {})
            qa_output = {
                "schema_version": SCHEMA_VERSION,
                **copy.deepcopy(metadata),
                "claim_bundle_hash": pre_hash,
                "dialogue_hash": (replay.get("actual_hashes") or {}).get(
                    "dialogue_hash"
                ),
                "alignment_hash": (replay.get("actual_hashes") or {}).get(
                    "alignment_hash"
                ),
                "qa_id": qa.get("qa_id"),
                "qa_instance_id": _qa_instance_id(
                    metadata["candidate_id"],
                    (replay.get("actual_hashes") or {}).get("dialogue_hash"),
                    qa.get("qa_id"),
                ),
                "dialogue_family_id": qa.get("dialogue_family_id"),
                "dialogue_family_split": qa.get("dialogue_family_split"),
                "validator_report": copy.deepcopy(validation),
                "qa": copy.deepcopy(qa),
                "api_calls_made": 0,
            }
            if validation.get("pass") is True:
                validator_pass_rows.append(qa_output)
            else:
                validator_fail_count += 1
                qa_rejects.append(
                    _reject_row(
                        metadata,
                        stage="independent_claim_validation",
                        reason="claim_validator_rejected_qa",
                        qa=qa,
                        validation=validation,
                    )
                )

        dialogue_row["validator_pass_qa_count"] = len(validator_pass_rows)
        dialogue_row["validator_reject_qa_count"] = validator_fail_count
        if len(validator_pass_rows) >= min_qas:
            dialogue_row["proof_ready"] = True
            dialogue_row["proof_ready_qa_ids"] = [
                item["qa_id"] for item in validator_pass_rows
            ]
            proof_ready_qas.extend(validator_pass_rows)
        else:
            dialogue_row["failure_stage"] = "dialogue_proof_ready_gate"
            dialogue_row["failure_reason"] = (
                f"only {len(validator_pass_rows)} validator-pass QAs; need {min_qas}"
            )
            # These QAs are individually valid and remain auditable, but are
            # not labeled proof-ready because the dialogue-level 3-QA contract
            # did not pass.  Failed siblings have already been recorded above.
            for item in validator_pass_rows:
                qa_rejects.append(
                    _reject_row(
                        metadata,
                        stage="dialogue_proof_ready_gate",
                        reason="dialogue_has_fewer_than_minimum_valid_qas",
                        qa=item["qa"],
                        validation=item["validator_report"],
                        details={
                            "validator_pass_qa_count": len(validator_pass_rows),
                            "minimum_required": min_qas,
                        },
                    )
                )
        dialogue_rows.append(dialogue_row)

    family_splits: dict[str, set[str]] = defaultdict(set)
    for row in dialogue_rows:
        family_id = _normalize(row.get("dialogue_family_id"))
        split = _normalize(row.get("dialogue_family_split"))
        if family_id and split:
            family_splits[family_id].add(split)
    family_split_leaks = {
        family_id: sorted(splits)
        for family_id, splits in family_splits.items()
        if len(splits) > 1
    }
    if family_split_leaks:
        raise RuntimeError(f"dialogue family split leakage: {family_split_leaks}")

    source_manifest_after = source_tree_manifest(run_dir)
    source_immutable = source_manifest_before == source_manifest_after
    if not source_immutable:
        raise RuntimeError("source run changed during no-API replay")

    report = {
        "schema_version": SCHEMA_VERSION,
        "offline_only": True,
        "api_calls_made": 0,
        "source_run_dir": str(run_dir),
        "source_file_sha256_before": source_manifest_before,
        "source_file_sha256_after": source_manifest_after,
        "source_tree_sha256_before": source_tree_hash(source_manifest_before),
        "source_tree_sha256_after": source_tree_hash(source_manifest_after),
        "source_run_immutable": source_immutable,
        "source_raw_row_count": len(raw_rows),
        "frozen_final_qa_candidate_count": len(candidates),
        "replay_pass_count": sum(bool(row.get("replay_pass")) for row in dialogue_rows),
        "claim_ir_compiled_dialogue_count": sum(
            bool(row.get("claim_ir_compiled")) for row in dialogue_rows
        ),
        "claim_bundle_hash_stable_count": sum(
            bool(row.get("claim_bundle_hash_stable")) for row in dialogue_rows
        ),
        "precompiled_qa_count": sum(
            int(row.get("precompiled_qa_count") or 0) for row in dialogue_rows
        ),
        "support_bound_qa_count": sum(
            int(row.get("support_bound_qa_count") or 0) for row in dialogue_rows
        ),
        "validator_pass_qa_count": sum(
            int(row.get("validator_pass_qa_count") or 0) for row in dialogue_rows
        ),
        "validator_reject_qa_count": sum(
            int(row.get("validator_reject_qa_count") or 0) for row in dialogue_rows
        ),
        "proof_ready_dialogue_count": sum(
            bool(row.get("proof_ready")) for row in dialogue_rows
        ),
        "proof_ready_qa_count": len(proof_ready_qas),
        "distinct_proof_ready_qa_id_count": len(
            {_normalize(row.get("qa_id")) for row in proof_ready_qas}
        ),
        "distinct_proof_ready_qa_instance_count": len(
            {_normalize(row.get("qa_instance_id")) for row in proof_ready_qas}
        ),
        "qa_reject_row_count": len(qa_rejects),
        "claim_bundle_row_count": len(claim_bundle_rows),
        "dialogue_row_count": len(dialogue_rows),
        "minimum_validator_pass_qas_per_dialogue": min_qas,
        "maximum_compiled_qas_per_dialogue": max_qas,
        "validator_issue_counts_by_code": dict(sorted(issue_counts.items())),
        "dialogue_family_count": len(family_splits),
        "dialogue_family_split_counts": dict(
            sorted(
                Counter(next(iter(splits)) for splits in family_splits.values()).items()
            )
        ),
        "family_split_leak_count": len(family_split_leaks),
        "family_split_leaks": family_split_leaks,
        "single_qa_failure_deletes_siblings": False,
        "semantic_contract_frozen_before_dialogue": True,
        "proof_ready_policy": (
            "Each QA is validated independently; a dialogue is proof-ready only "
            f"when at least {min_qas} sibling QAs pass."
        ),
    }
    return {
        "claim_bundles": claim_bundle_rows,
        "proof_ready_qas": proof_ready_qas,
        "qa_rejects": qa_rejects,
        "dialogue_rows": dialogue_rows,
        "report": report,
    }


def markdown_report(result: dict[str, Any], output_dir: Path) -> str:
    report = result["report"]
    lines = [
        "# Stage3 Claim-first：run175 无 API replay",
        "",
        "## 结论",
        "",
        f"- 冻结 final-QA 候选：{report['frozen_final_qa_candidate_count']}",
        f"- graph/dialogue/alignment 精确重放通过：{report['replay_pass_count']}",
        f"- pre/post Claim IR 哈希一致：{report['claim_bundle_hash_stable_count']}",
        f"- 本地预编译 QA：{report['precompiled_qa_count']}",
        f"- 独立 validator 通过 QA：{report['validator_pass_qa_count']}",
        f"- proof-ready dialogue：{report['proof_ready_dialogue_count']}",
        f"- proof-ready QA：{report['proof_ready_qa_count']}",
        f"- family split 泄漏：{report['family_split_leak_count']}",
        "- API 调用：0",
        f"- run175 源目录保持不变：{report['source_run_immutable']}",
        "",
        "单条 QA 的失败只进入 qa_rejects；不会删除 sibling QA 或原 dialogue。",
        "未达到每 dialogue 至少 3 条 validator-pass QA 的 dialogue 不标记 proof-ready。",
        "",
        "## Dialogue 结果",
        "",
        "| raw 行 | attempt | source operation | pre QA | validator pass | proof-ready | family split |",
        "|---:|---:|---|---:|---:|---|---|",
    ]
    for row in result["dialogue_rows"]:
        lines.append(
            f"| {row['source_raw_line_number']} | {row.get('attempt')} | "
            f"{row.get('source_qa_operation')} | {row.get('precompiled_qa_count', 0)} | "
            f"{row.get('validator_pass_qa_count', 0)} | {row.get('proof_ready')} | "
            f"{row.get('dialogue_family_split') or '-'} |"
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            f"- `{output_dir / 'claim_bundles.jsonl'}`",
            f"- `{output_dir / 'proof_ready_qas.jsonl'}`",
            f"- `{output_dir / 'qa_rejects.jsonl'}`",
            f"- `{output_dir / 'dialogue_rows.jsonl'}`",
            f"- `{output_dir / 'report.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_result(
    result: dict[str, Any], output_dir: Path, *, overwrite: bool = False
) -> None:
    output_dir = output_dir.resolve()
    source_dir = Path(result["report"]["source_run_dir"]).resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("output directory must be outside the immutable source run")
    if output_dir.exists():
        if not overwrite:
            raise ValueError(f"output directory already exists: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(
                "--overwrite only permits an existing empty directory; refusing "
                "to replace prior replay artifacts"
            )
    else:
        output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "claim_bundles.jsonl", result["claim_bundles"])
    _write_jsonl(output_dir / "proof_ready_qas.jsonl", result["proof_ready_qas"])
    _write_jsonl(output_dir / "qa_rejects.jsonl", result["qa_rejects"])
    _write_jsonl(output_dir / "dialogue_rows.jsonl", result["dialogue_rows"])
    _write_json(output_dir / "report.json", result["report"])
    (output_dir / "report.md").write_text(
        markdown_report(result, output_dir), encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-candidates", type=int, default=10)
    parser.add_argument("--min-qas", type=int, default=3)
    parser.add_argument("--max-qas", type=int, default=6)
    parser.add_argument("--eval-per-mille", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_replay(
        args.run_dir,
        expected_candidate_count=args.expected_candidates,
        min_qas=args.min_qas,
        max_qas=args.max_qas,
        eval_per_mille=args.eval_per_mille,
    )
    write_result(result, args.output_dir, overwrite=args.overwrite)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
