"""Finalize the full manual review of ordering examples used for training."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round0_b0_matched_training_v1_live_2"
)
OUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round0_train_ordering_full_manual_review_v1_live_1"
)
DECISIONS = Path(__file__).with_name("train_manual_adjudications.tsv")
WORKLIST = OUT / "train_ordering_review_worklist.tsv"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    gold = Counter(row["gold_status"] for row in values)
    evidence = Counter(row["evidence_status"] for row in values)
    primary = Counter(row["primary_status"] for row in values)
    errors = sum(gold[value] for value in ("invalid_gold", "ambiguous", "unsupported"))
    return {
        "total": len(values),
        "training_error_count": errors,
        "training_error_rate": errors / len(values) if values else None,
        "valid_gold_count": gold["valid"],
        "valid_gold_rate": gold["valid"] / len(values) if values else None,
        "gold_status_counts": dict(sorted(gold.items())),
        "evidence_status_counts": dict(sorted(evidence.items())),
        "primary_status_counts": dict(sorted(primary.items())),
    }


def grouped_counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: status_counts(values) for name, values in sorted(groups.items())}


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    work = {row["candidate_id"]: row for row in load_tsv(WORKLIST)}
    decisions_list = load_tsv(DECISIONS)
    decisions = {row["candidate_id"]: row for row in decisions_list}
    if len(decisions_list) != len(decisions):
        raise RuntimeError("duplicate candidate_id in manual adjudications")
    if set(work) != set(decisions):
        raise RuntimeError(
            f"review closure mismatch missing={sorted(set(work) - set(decisions))} "
            f"extra={sorted(set(decisions) - set(work))}"
        )

    allowed_gold = {"valid", "invalid_gold", "ambiguous", "unsupported"}
    merged: dict[str, dict[str, Any]] = {}
    for cid in sorted(work):
        source = work[cid]
        decision = decisions[cid]
        gold_status = decision["gold_status"]
        evidence_status = decision["evidence_status"]
        if gold_status not in allowed_gold or evidence_status not in {"clean", "mismatch"}:
            raise RuntimeError(f"invalid status for {cid}")
        expected_primary = (
            "valid" if gold_status == "valid" and evidence_status == "clean"
            else "evidence_mismatch" if gold_status == "valid"
            else gold_status
        )
        if decision["primary_status"] != expected_primary:
            raise RuntimeError(f"primary status mismatch for {cid}")
        gold = source["canonical_gold"]
        corrected = decision["corrected_order"]
        if gold_status == "valid" and corrected != gold:
            raise RuntimeError(f"valid row correction differs from gold for {cid}")
        if gold_status == "invalid_gold" and (not corrected or corrected == gold):
            raise RuntimeError(f"invalid gold row lacks a distinct correction for {cid}")
        if gold_status in {"ambiguous", "unsupported"} and corrected:
            raise RuntimeError(f"unresolvable row has corrected order for {cid}")
        if corrected and sorted(corrected.split(",")) != ["I1", "I2", "I3", "I4"]:
            raise RuntimeError(f"invalid corrected permutation for {cid}")
        merged[cid] = {
            **source,
            **decision,
            "training_error": gold_status != "valid",
        }

    policy_rows: dict[str, list[dict[str, Any]]] = {}
    membership: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    source_manifests: dict[str, dict[str, Any]] = {}
    train_record_failures: list[str] = []
    for policy in ("clean", "inclusive"):
        manifest_path = RUN / f"m_{policy}_policy_training_manifest.jsonl"
        train_path = RUN / f"m_{policy}_policy_train.jsonl"
        manifest_all = load_jsonl(manifest_path)
        train_all = load_jsonl(train_path)
        if len(manifest_all) != len(train_all):
            train_record_failures.append(f"row_count:{policy}")
        for index, (manifest, train) in enumerate(zip(manifest_all, train_all)):
            metadata = train.get("metadata") or {}
            assistant = (train.get("messages") or [{}])[-1]
            if metadata.get("candidate_id") != manifest.get("candidate_id"):
                train_record_failures.append(f"candidate_id:{policy}:{index}")
            if assistant.get("role") != "assistant" or assistant.get("content") != manifest.get("target_text"):
                train_record_failures.append(f"target_text:{policy}:{index}")
        rows: list[dict[str, Any]] = []
        for manifest in manifest_all:
            if manifest.get("question_type") != "ordering":
                continue
            cid = str(manifest["candidate_id"])
            row = {
                **merged[cid],
                "policy": policy,
                "selection_role": manifest["selection_role"],
                "manifest_evidence_status": manifest["evidence_status"],
                "full_context_accuracy": manifest["full_context_accuracy"],
                "no_context_accuracy": manifest["no_context_accuracy"],
                "target_text": manifest["target_text"],
                "display_to_canonical": manifest["display_to_canonical"],
            }
            if row["gold_status"] == "invalid_gold":
                reverse = {
                    canonical: displayed
                    for displayed, canonical in manifest["display_to_canonical"].items()
                }
                canonical_order = row["corrected_order"].split(",")
                row["corrected_target_text"] = f"<box>{','.join(reverse[item] for item in canonical_order)}</box>"
            else:
                row["corrected_target_text"] = ""
            rows.append(row)
            membership[cid][policy] = manifest
        policy_rows[policy] = rows
        source_manifests[policy] = {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "train_path": str(train_path),
            "train_sha256": sha256(train_path),
        }

    if train_record_failures:
        raise RuntimeError("actual train record verification failed: " + ", ".join(train_record_failures))

    union_rows = list(merged.values())
    shared_ids = {cid for cid, policies in membership.items() if len(policies) == 2}
    clean_only_ids = {cid for cid, policies in membership.items() if set(policies) == {"clean"}}
    inclusive_only_ids = {cid for cid, policies in membership.items() if set(policies) == {"inclusive"}}

    structural = Counter()
    for row in union_rows:
        items = json.loads(row["items"])
        refs = json.loads(row["evidence_refs"])
        if len(refs) != len(items):
            structural["evidence_ref_count_mismatch"] += 1
        if len(set(refs)) != len(refs):
            structural["duplicate_evidence_ref"] += 1
        if sorted(row["canonical_gold"].split(",")) != ["I1", "I2", "I3", "I4"]:
            structural["gold_not_permutation"] += 1

    final_fields = [
        "review_index", "candidate_id", "root_id", "policies", "clean_role",
        "inclusive_role", "manifest_evidence_strata", "generation_source", "question",
        "canonical_gold", "items", "evidence_refs", "evidence_text", "evidence_ref_order",
        "gold_status", "evidence_status", "primary_status", "training_error",
        "corrected_order", "note", "review_basis", "dialogue_path", "candidate_pool_path",
    ]
    final_rows = [merged[cid] for cid in sorted(merged)]
    final_path = OUT / "final_train_ordering_manual_review.tsv"
    write_tsv(final_path, final_rows, final_fields)

    policy_error_files: dict[str, str] = {}
    policy_fields = [
        "candidate_id", "root_id", "policy", "selection_role", "manifest_evidence_status",
        "gold_status", "evidence_status", "primary_status", "canonical_gold", "corrected_order",
        "target_text", "corrected_target_text", "question", "items", "note", "dialogue_path",
        "candidate_pool_path",
    ]
    for policy, rows in policy_rows.items():
        path = OUT / f"{policy}_ordering_training_errors.tsv"
        write_tsv(path, [row for row in rows if row["training_error"]], policy_fields)
        policy_error_files[policy] = str(path)

    policy_summary = {}
    for policy, rows in policy_rows.items():
        policy_summary[policy] = {
            **status_counts(rows),
            "by_selection_role": grouped_counts(rows, "selection_role"),
            "by_manifest_evidence_status": grouped_counts(rows, "manifest_evidence_status"),
            "by_generation_source": grouped_counts(rows, "generation_source"),
            "error_root_count": len({row["root_id"] for row in rows if row["training_error"]}),
            "root_count": len({row["root_id"] for row in rows}),
            "error_file": policy_error_files[policy],
        }

    report = {
        "audit_version": "h4_round0_train_ordering_full_manual_review_v1",
        "status": "COMPLETE_FULL_DIALOGUE_MANUAL_REVIEW",
        "audit_scope": "all ordering examples actually present in Clean and Inclusive training files",
        "training_error_definition": ["invalid_gold", "ambiguous", "unsupported"],
        "evidence_mismatch_definition": "gold remains usable but one or more supplied evidence refs do not directly support the corresponding item",
        "offline_only": True,
        "solver_calls": 0,
        "external_api_calls": 0,
        "memorax_calls": 0,
        "qa_writer_calls": 0,
        "semantic_auditor_calls": 0,
        "source_run": str(RUN),
        "source_manifests": source_manifests,
        "actual_train_record_verification_failures": train_record_failures,
        "unique_candidates": status_counts(union_rows),
        "policy_counts": policy_summary,
        "membership_counts": {
            "shared": status_counts(merged[cid] for cid in sorted(shared_ids)),
            "clean_only": status_counts(merged[cid] for cid in sorted(clean_only_ids)),
            "inclusive_only": status_counts(merged[cid] for cid in sorted(inclusive_only_ids)),
        },
        "unique_root_count": len({row["root_id"] for row in union_rows}),
        "structural_issue_counts": dict(sorted(structural.items())),
        "manual_decisions_source": str(DECISIONS),
        "manual_decisions_sha256": sha256(DECISIONS),
        "final_review": str(final_path),
        "final_review_sha256": sha256(final_path),
        "policy_error_files": policy_error_files,
    }
    report_path = OUT / "final_train_ordering_manual_review_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    clean = report["policy_counts"]["clean"]
    inclusive = report["policy_counts"]["inclusive"]
    unique = report["unique_candidates"]
    md = [
        "# Train Ordering Full Manual Review",
        "",
        "This is a read-only, full-dialogue manual audit of every ordering example actually used by the two policy training files.",
        "",
        "| Scope | Total | Training errors | Error rate | Valid gold | Evidence mismatch |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Unique questions | {unique['total']} | {unique['training_error_count']} | {unique['training_error_rate']:.2%} | {unique['valid_gold_count']} | {unique['evidence_status_counts'].get('mismatch', 0)} |",
        f"| Clean actual samples | {clean['total']} | {clean['training_error_count']} | {clean['training_error_rate']:.2%} | {clean['valid_gold_count']} | {clean['evidence_status_counts'].get('mismatch', 0)} |",
        f"| Inclusive actual samples | {inclusive['total']} | {inclusive['training_error_count']} | {inclusive['training_error_rate']:.2%} | {inclusive['valid_gold_count']} | {inclusive['evidence_status_counts'].get('mismatch', 0)} |",
        "",
        "Training errors include invalid gold, ambiguous ordering, and unsupported items. Evidence mismatch alone is not counted as a bad training target because evidence refs are not part of the training prompt.",
        "",
        f"Final row-level review: `{final_path}`",
        f"Clean errors: `{policy_error_files['clean']}`",
        f"Inclusive errors: `{policy_error_files['inclusive']}`",
    ]
    md_path = OUT / "final_train_ordering_manual_review_report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    report["report_markdown"] = str(md_path)
    report["report_markdown_sha256"] = sha256(md_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
