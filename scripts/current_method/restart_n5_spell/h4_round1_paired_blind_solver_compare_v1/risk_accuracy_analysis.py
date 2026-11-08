"""Offline cross-tab of QA static-risk flags against existing Solver results."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


MODELS = ("M_clean_policy", "M_inclusive_policy")
BRANCHES = ("control", "feedback")
CONTEXTS = ("full_context", "no_context")
VERSION = "h4_round1_risk_accuracy_analysis_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    observed = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    token = sidecar.read_text(encoding="utf-8").strip().split()
    if not token or token[0] != observed:
        raise ValueError(f"sha_mismatch:{path}")
    return observed


def group_for(candidate: dict[str, Any], risk_ids: set[str]) -> str:
    cid = str(candidate["candidate_id"])
    if str(candidate.get("question_type")) == "ordering":
        return "ordering_static_risk" if cid in risk_ids else "ordering_no_static_risk"
    return "non_ordering"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"candidate_count": len(rows)}
    for context in CONTEXTS:
        total = sum(int(row.get(f"{context}_total_count") or 0) for row in rows)
        correct = sum(int(row.get(f"{context}_correct_count") or 0) for row in rows)
        values = [float(row[f"{context}_accuracy"]) for row in rows if row.get(f"{context}_accuracy") is not None]
        out[context] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else None,
            "mean_candidate_accuracy": sum(values) / len(values) if values else None,
            "candidate_metric_count": len(values),
        }
    full, no = out["full_context"]["accuracy"], out["no_context"]["accuracy"]
    out["context_gain"] = full - no if full is not None and no is not None else None
    return out


def standardized_by_group(groups: Mapping[str, Any]) -> dict[str, Any]:
    names = sorted(set(groups["control"]) & set(groups["feedback"]))
    pooled_total = sum(groups["control"][name]["full_context"]["total"] + groups["feedback"][name]["full_context"]["total"] for name in names)
    weights = {
        name: (groups["control"][name]["full_context"]["total"] + groups["feedback"][name]["full_context"]["total"]) / pooled_total
        for name in names
    }
    result: dict[str, Any] = {"groups": names, "pooled_request_weights": weights, "branches": {}}
    for branch in BRANCHES:
        full = sum(weights[name] * groups[branch][name]["full_context"]["accuracy"] for name in names)
        no = sum(weights[name] * groups[branch][name]["no_context"]["accuracy"] for name in names)
        result["branches"][branch] = {"full_context_accuracy": full, "no_context_accuracy": no, "context_gain": full - no}
    result["feedback_minus_control"] = {
        metric: result["branches"]["feedback"][metric] - result["branches"]["control"][metric]
        for metric in ("full_context_accuracy", "no_context_accuracy", "context_gain")
    }
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_with_sidecar(path: Path, value: Any) -> None:
    write_json(path, value)
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--solver-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    qa = args.qa_root.resolve()
    solver = args.solver_root.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output_not_empty:{output}")
    output.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "qa_candidates": qa / "all_candidates.jsonl",
        "ordering_audit": qa / "ordering_static_risk_audit.json",
        "qa_report": qa / "final_paired_qa_batch_report.json",
        "solver_candidate_manifest": solver / "candidate_manifest.jsonl",
    }
    source_hashes = {name: verify_sidecar(path) for name, path in source_paths.items()}
    candidates = load_jsonl(source_paths["qa_candidates"])
    risk_ids = {
        str(row["candidate_id"])
        for row in load_json(source_paths["ordering_audit"]).get("feedback", []) + load_json(source_paths["ordering_audit"]).get("control", [])
        if row.get("applicable") and row.get("submitted_gold_anchor_order_consistent") is False
    }
    risk_by_branch = {
        branch: {
            str(row["candidate_id"])
            for row in load_json(source_paths["ordering_audit"])[branch]
            if row.get("applicable") and row.get("submitted_gold_anchor_order_consistent") is False
        }
        for branch in BRANCHES
    }

    candidate_index = {str(row["candidate_id"]): row for row in candidates}
    groups: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    missing: list[str] = []
    for model in MODELS:
        for branch in BRANCHES:
            difficulty_rows = load_jsonl(solver / model / branch / "difficulty_manifest.jsonl")
            by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for difficulty in difficulty_rows:
                cid = str(difficulty["candidate_id"])
                candidate = candidate_index.get(cid)
                if candidate is None:
                    missing.append(cid)
                    continue
                group = group_for(candidate, risk_by_branch[branch])
                by_group[group].append(difficulty)
            for group, rows in by_group.items():
                groups[model][branch][group] = summarize(rows)
    if missing:
        raise ValueError(f"solver_candidate_missing:{missing[:3]}")

    final = {
        "status": "ROUND1_RISK_ACCURACY_ANALYSIS_COMPLETE",
        "version": VERSION,
        "qa_root": str(qa),
        "solver_root": str(solver),
        "output_root": str(output),
        "source_hashes": source_hashes,
        "risk_definition": "ordering candidate with submitted_gold_anchor_order_consistent=false; source audit marks this risk_only and non-blocking",
        "risk_counts": {branch: len(ids) for branch, ids in risk_by_branch.items()},
        "groups": groups,
        "pooled_request_weight_standardization": {
            model: standardized_by_group(groups[model]) for model in MODELS
        },
        "prohibited_calls": {"solver_calls": 0, "external_api_calls": 0, "memorax_calls": 0, "qa_writer_calls": 0, "semantic_auditor_calls": 0, "training_runs": 0},
    }
    write_with_sidecar(output / "risk_stratified_solver_accuracy.json", final)
    lines = [
        "# Static-risk versus Solver accuracy cross-tab",
        "",
        "- Status: `ROUND1_RISK_ACCURACY_ANALYSIS_COMPLETE`",
        "- Static-risk definition: ordering candidate with evidence references not in canonical dialogue order",
        "- The source QA audit marks this `risk_only`; it is not a semantic error label.",
        "",
        "| Model | Branch | Group | Candidates | Full accuracy | No-context accuracy | Context gain |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for branch in BRANCHES:
            for group in ("ordering_static_risk", "ordering_no_static_risk", "non_ordering"):
                row = groups[model][branch].get(group)
                if not row:
                    continue
                lines.append(f"| {model} | {branch} | {group} | {row['candidate_count']} | {row['full_context']['accuracy']:.4f} | {row['no_context']['accuracy']:.4f} | {row['context_gain']:.4f} |")
    lines += [
        "",
        "## Pooled risk-group standardization",
        "",
        "The same pooled request weights are applied to both branches to remove the effect of different proportions of risk, non-risk ordering, and non-ordering candidates.",
        "",
        "| Model | Control Full | Feedback Full | Feedback-Control | Control No-context | Feedback No-context | Feedback-Control |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = final["pooled_request_weight_standardization"][model]
        lines.append(
            f"| {model} | {row['branches']['control']['full_context_accuracy']:.4f} | "
            f"{row['branches']['feedback']['full_context_accuracy']:.4f} | "
            f"{row['feedback_minus_control']['full_context_accuracy']:.4f} | "
            f"{row['branches']['control']['no_context_accuracy']:.4f} | "
            f"{row['branches']['feedback']['no_context_accuracy']:.4f} | "
            f"{row['feedback_minus_control']['no_context_accuracy']:.4f} |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        "Existing QA and Solver artifacts were read offline and sidecars verified. No new Solver or model call was made.",
    ]
    md = output / "risk_stratified_solver_accuracy.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    md.with_name(md.name + ".sha256").write_text(sha256_file(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
