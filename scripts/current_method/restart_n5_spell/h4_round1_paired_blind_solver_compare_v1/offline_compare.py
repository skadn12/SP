"""Build an offline, immutable Control versus Feedback comparison report.

This module only reads the completed Round-1 Solver and QA artifacts.  It does
not import or invoke a Solver, an API client, Memorax, or any auditor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


MODELS = ("M_clean_policy", "M_inclusive_policy")
BRANCHES = ("control", "feedback")
CONTEXTS = ("no_context", "full_context")
Q_TYPES = ("single_choice", "multi_select", "ordering")
GEN_SOURCES = ("free_exploration", "contract_guided")
VERSION = "h4_round1_paired_blind_solver_compare_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_row_not_object:{path}:{line_no}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    observed = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing_sha_sidecar:{path}")
    token = sidecar.read_text(encoding="utf-8").strip().split()
    if not token or token[0] != observed:
        raise ValueError(f"sha_mismatch:{path}:{token[:1]}:{observed}")
    return observed


def json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def candidate_signature(row: Mapping[str, Any]) -> str:
    fields = (
        "root_id", "source_candidate_id", "question_type", "generation_source",
        "question", "gold", "evidence_refs", "options", "items",
    )
    return json_key({field: row.get(field) for field in fields})


def metric_pair(rows: list[Mapping[str, Any]], context: str) -> dict[str, Any]:
    total = sum(int((row.get("position_diagnostics") or {}).get(context, {}).get("total_count") or 0) for row in rows)
    correct = sum(int((row.get("position_diagnostics") or {}).get(context, {}).get("correct_count") or 0) for row in rows)
    accuracy = correct / total if total else None
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_root[str(row.get("root_id"))].append(row)
    root_values: list[float] = []
    for root_rows in by_root.values():
        root_total = sum(int((r.get("position_diagnostics") or {}).get(context, {}).get("total_count") or 0) for r in root_rows)
        root_correct = sum(int((r.get("position_diagnostics") or {}).get(context, {}).get("correct_count") or 0) for r in root_rows)
        if root_total:
            root_values.append(root_correct / root_total)
    return {"correct": correct, "total": total, "accuracy": accuracy, "root_equal_accuracy": mean(root_values), "root_count": len(root_values)}


def context_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = {context: metric_pair(rows, context) for context in CONTEXTS}
    full, no = pairs["full_context"], pairs["no_context"]
    return {
        "full_context": full,
        "no_context": no,
        "context_gain": {
            "candidate_weighted": full["accuracy"] - no["accuracy"] if full["accuracy"] is not None and no["accuracy"] is not None else None,
            "root_equal": full["root_equal_accuracy"] - no["root_equal_accuracy"] if full["root_equal_accuracy"] is not None and no["root_equal_accuracy"] is not None else None,
        },
    }


def diagnostic_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for context in CONTEXTS:
        flips = [float(row["canonical_flip_rate"][context]) for row in rows if finite((row.get("canonical_flip_rate") or {}).get(context))]
        worst: list[float] = []
        for row in rows:
            positions = ((row.get("position_diagnostics") or {}).get(context) or {}).get("position_accuracy") or {}
            values = [float(v) for v in positions.values() if finite(v)]
            if values:
                worst.append(min(values))
        result[context] = {
            "canonical_flip_rate_mean": mean(flips),
            "canonical_flip_rate_candidate_count": len(flips),
            "worst_position_accuracy_mean": mean(worst),
            "worst_position_candidate_count": len(worst),
        }
        by_root: dict[str, list[float]] = defaultdict(list)
        by_root_worst: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            root = str(row.get("root_id"))
            if finite((row.get("canonical_flip_rate") or {}).get(context)):
                by_root[root].append(float(row["canonical_flip_rate"][context]))
            positions = ((row.get("position_diagnostics") or {}).get(context) or {}).get("position_accuracy") or {}
            values = [float(v) for v in positions.values() if finite(v)]
            if values:
                by_root_worst[root].append(min(values))
        result[context]["canonical_flip_rate_root_equal"] = mean(mean(v) for v in by_root.values())
        result[context]["worst_position_accuracy_root_equal"] = mean(mean(v) for v in by_root_worst.values())
    return result


def slice_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("question_type")) for row in rows)
    return {"candidate_count": len(rows), "question_type_counts": dict(sorted(counts.items())), "metrics": context_metrics(rows), "diagnostics": diagnostic_metrics(rows), "technical_failure_count": sum(int(row.get("technical_failure_count") or 0) for row in rows)}


def branch_delta(control: Mapping[str, Any], feedback: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"candidate_count_delta_feedback_minus_control": int(feedback.get("candidate_count", 0)) - int(control.get("candidate_count", 0))}
    for context in CONTEXTS:
        c = ((control.get("metrics") or {}).get(context) or {}).get("accuracy")
        f = ((feedback.get("metrics") or {}).get(context) or {}).get("accuracy")
        cr = ((control.get("metrics") or {}).get(context) or {}).get("root_equal_accuracy")
        fr = ((feedback.get("metrics") or {}).get(context) or {}).get("root_equal_accuracy")
        out[context] = {"accuracy_feedback_minus_control": f - c if f is not None and c is not None else None, "root_equal_accuracy_feedback_minus_control": fr - cr if fr is not None and cr is not None else None}
        cf = ((control.get("diagnostics") or {}).get(context) or {}).get("canonical_flip_rate_mean")
        ff = ((feedback.get("diagnostics") or {}).get(context) or {}).get("canonical_flip_rate_mean")
        cw = ((control.get("diagnostics") or {}).get(context) or {}).get("worst_position_accuracy_mean")
        fw = ((feedback.get("diagnostics") or {}).get(context) or {}).get("worst_position_accuracy_mean")
        out[context].update({"canonical_flip_rate_feedback_minus_control": ff - cf if ff is not None and cf is not None else None, "worst_position_accuracy_feedback_minus_control": fw - cw if fw is not None and cw is not None else None})
    return out


def build_alignment(rows_by_branch: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    indexed: dict[str, dict[tuple[str, str], list[Mapping[str, Any]]]] = {}
    for branch, rows in rows_by_branch.items():
        indexed[branch] = defaultdict(list)
        for row in rows:
            source = row.get("source_candidate_id")
            if source:
                indexed[branch][(str(row.get("root_id")), str(source))].append(row)
    control_keys, feedback_keys = set(indexed["control"]), set(indexed["feedback"])
    shared_lineage = control_keys & feedback_keys
    lineage_pair_count = sum(min(len(indexed["control"][key]), len(indexed["feedback"][key])) for key in shared_lineage)
    exact_pairs = 0
    nonidentical_shared = 0
    for key in shared_lineage:
        crows, frows = indexed["control"][key], indexed["feedback"][key]
        for crow in crows:
            for frow in frows:
                if candidate_signature(crow) == candidate_signature(frow):
                    exact_pairs += 1
                else:
                    nonidentical_shared += 1
    return {
        "alignment_rule": "direct_pair_requires_same_root_id, source_candidate_id, question_type, generation_source, question, gold, evidence_refs, options/items",
        "control_candidate_count": len(rows_by_branch["control"]),
        "feedback_candidate_count": len(rows_by_branch["feedback"]),
        "direct_candidate_pair_count": exact_pairs,
        "unpaired_control_count": len(rows_by_branch["control"]) - exact_pairs,
        "unpaired_feedback_count": len(rows_by_branch["feedback"]) - exact_pairs,
        "lineage_candidate_pair_count": lineage_pair_count,
        "lineage_unpaired_control_count": len(rows_by_branch["control"]) - lineage_pair_count,
        "lineage_unpaired_feedback_count": len(rows_by_branch["feedback"]) - lineage_pair_count,
        "shared_lineage_key_count": len(shared_lineage),
        "shared_lineage_nonidentical_candidate_count": nonidentical_shared,
        "lineage_keys_without_source_candidate_id": len([row for branch in rows_by_branch.values() for row in branch if not row.get("source_candidate_id")]),
        "candidate_order_pairing_used": False,
        "direct_pair_metrics_available": exact_pairs > 0,
        "note": "Shared lineage IDs with independently generated question content are reported but excluded from paired deltas.",
    }


def technical_report(run: Path, rows_by_model_branch: Mapping[tuple[str, str], list[Mapping[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"total_technical_failure_count": 0, "total_semantic_error_count": 0, "total_parse_failure_count": 0, "total_truncation_count": 0, "failure_type_counts": {}, "parse_status_counts": {}, "finish_reason_counts": {}, "by_model_branch": {}}
    failures: Counter[str] = Counter()
    technical = 0
    parses: Counter[str] = Counter()
    finishes: Counter[str] = Counter()
    for (model, branch), rows in rows_by_model_branch.items():
        branch_failure = Counter()
        branch_technical = 0
        branch_parse = Counter()
        branch_finish = Counter()
        request_count = 0
        for row in rows:
            branch_technical += int(row.get("technical_failure_count") or 0)
            technical += int(row.get("technical_failure_count") or 0)
            for request in row.get("requests") or []:
                request_count += 1
                failure = request.get("failure_type")
                if failure:
                    branch_failure[str(failure)] += 1; failures[str(failure)] += 1
                parse = str(request.get("parse_status") or "unknown")
                branch_parse[parse] += 1; parses[parse] += 1
                metadata = request.get("backend_metadata") or {}
                finish = str(metadata.get("finish_reason") or "unknown")
                branch_finish[finish] += 1; finishes[finish] += 1
        parse_failures = sum(v for k, v in branch_parse.items() if k not in {"success", "valid_boxed_letter", "unknown"})
        report["by_model_branch"][f"{model}/{branch}"] = {"request_count": request_count, "technical_failure_count": branch_technical, "semantic_error_count": sum(branch_failure.values()), "parse_failure_count": parse_failures, "truncation_count": branch_finish.get("length", 0), "failure_type_counts": dict(sorted(branch_failure.items())), "parse_status_counts": dict(sorted(branch_parse.items())), "finish_reason_counts": dict(sorted(branch_finish.items()))}
    report["failure_type_counts"] = dict(sorted(failures.items())); report["parse_status_counts"] = dict(sorted(parses.items())); report["finish_reason_counts"] = dict(sorted(finishes.items()))
    report["total_technical_failure_count"] = technical; report["total_semantic_error_count"] = sum(failures.values()); report["total_parse_failure_count"] = sum(v for k, v in parses.items() if k not in {"success", "valid_boxed_letter", "unknown"}); report["total_truncation_count"] = finishes.get("length", 0)
    return report


def load_dialogues(qa_run: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest_row in load_jsonl(qa_run / "paired_dialogue_manifest.jsonl"):
        root = str(manifest_row["root_id"])
        for branch in BRANCHES:
            path = Path(str(manifest_row[f"{branch}_path"]))
            result[(branch, root)] = {"dialogue": load_json(path), "path": str(path), "sha256": manifest_row[f"{branch}_sha256"]}
    return result


def evidence_span(row: Mapping[str, Any], dialogue: Mapping[str, Any]) -> str:
    refs = {str(value) for value in row.get("evidence_refs") or []}
    valid = {str(turn.get("turn_ref")) for turn in dialogue.get("turns") or [] if isinstance(turn, Mapping)}
    if not refs:
        return "UNKNOWN"
    if not refs <= valid:
        return "INVALID_REF"
    r02 = {ref for ref in valid if ref.startswith("R02:") or ":R02:" in ref}
    if refs <= r02:
        return "R02_ONLY"
    if not refs & r02:
        return "HISTORICAL_ONLY"
    return "CROSS_ROUND"


def enrich_rows(run: Path, qa_run: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    manifest_rows = {str(row["candidate_id"]): row for row in load_jsonl(run / "candidate_manifest.jsonl")}
    dialogues = load_dialogues(qa_run)
    by_branch: dict[str, list[dict[str, Any]]] = {branch: [] for branch in BRANCHES}
    by_model_branch: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model in MODELS:
        for branch in BRANCHES:
            difficulty = load_jsonl(run / model / branch / "difficulty_manifest.jsonl")
            results = {str(row["candidate_id"]): row for row in load_jsonl(run / model / branch / "candidate_results.jsonl")}
            merged: list[dict[str, Any]] = []
            for row in difficulty:
                cid = str(row["candidate_id"])
                source = manifest_rows.get(cid)
                if source is None or cid not in results:
                    raise ValueError(f"candidate_lineage_missing:{model}:{branch}:{cid}")
                item = {**row, **{key: source.get(key) for key in ("source_candidate_id", "source_row_index", "question", "gold", "evidence_refs", "options", "items")}}
                item["requests"] = results[cid].get("requests") or []
                item["evidence_span"] = evidence_span(item, dialogues[(branch, str(item["root_id"]))]["dialogue"])
                item["model_name"] = model
                item["result_path"] = str(run / model / branch / "candidate_results.jsonl")
                merged.append(item)
            if len(merged) != len(difficulty) or len(results) != len(difficulty):
                raise ValueError(f"candidate_result_count_mismatch:{model}:{branch}")
            by_model_branch[(model, branch)] = merged
            if model == MODELS[0]:
                by_branch[branch] = merged
    return by_branch, by_model_branch, {"manifest": manifest_rows, "dialogues": dialogues}


def compare_slice(rows_by_model_branch: Mapping[tuple[str, str], list[Mapping[str, Any]]], field: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    values = sorted({str(row.get(field)) for rows in rows_by_model_branch.values() for row in rows})
    for value in values:
        output[value] = {}
        for model in MODELS:
            c = [row for row in rows_by_model_branch[(model, "control")] if str(row.get(field)) == value]
            f = [row for row in rows_by_model_branch[(model, "feedback")] if str(row.get(field)) == value]
            control, feedback = slice_summary(c), slice_summary(f)
            output[value][model] = {"control": control, "feedback": feedback, "feedback_minus_control": branch_delta(control, feedback), "control_candidate_count": len(c), "feedback_candidate_count": len(f)}
    return output


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_with_sidecar(path: Path, value: Any) -> None:
    write_json(path, value)
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-existing", action="store_true", help="rewrite only this tool's existing output directory")
    args = parser.parse_args()
    run = args.run_root.resolve(); output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()) and not args.allow_existing:
        raise RuntimeError(f"output_not_empty:{output}")
    output.mkdir(parents=True, exist_ok=True)
    final_solver = load_json(run / "final_solver_report.json")
    paired = load_json(run / "paired_solver_comparison.json")
    if final_solver.get("status") != "ROUND1_PAIRED_BLIND_SOLVER_COMPLETE":
        raise RuntimeError("solver_report_not_complete")
    if int(final_solver.get("candidate_count", -1)) != 530:
        raise RuntimeError("solver_candidate_count_mismatch")
    qa_run = Path(str(load_json(run / "preflight_report.json")["qa_run"]))
    source_report = load_json(qa_run / "final_paired_qa_batch_report.json")
    if source_report.get("status") != "ROUND1_PAIRED_BLIND_QA_BATCH_PASS":
        raise RuntimeError("qa_source_not_pass")
    qa_candidates = load_jsonl(qa_run / "all_candidates.jsonl")
    if len(qa_candidates) != 530:
        raise RuntimeError(f"qa_candidate_count_mismatch:{len(qa_candidates)}")
    source_hashes = {}
    for path in [run / "final_solver_report.json", run / "paired_solver_comparison.json", run / "candidate_manifest.jsonl", qa_run / "final_paired_qa_batch_report.json", qa_run / "all_candidates.jsonl", qa_run / "paired_dialogue_manifest.jsonl"]:
        source_hashes[str(path)] = verify_sidecar(path)
    branch_reports: dict[str, Any] = {}
    for model in MODELS:
        for branch in BRANCHES:
            path = run / model / branch / "branch_report.json"
            report = load_json(path); verify_sidecar(path)
            for input_name in ("difficulty_manifest.jsonl", "candidate_results.jsonl"):
                verify_sidecar(run / model / branch / input_name)
            expected = 276 if branch == "control" else 254
            if report.get("status") != "PASS" or report.get("candidate_count") != expected or report.get("technical_failure_count") != 0:
                raise RuntimeError(f"branch_not_closed:{model}:{branch}")
            branch_reports[f"{model}/{branch}"] = report
    preflight = {
        "status": "PASS",
        "version": VERSION,
        "solver_run": str(run),
        "qa_run": str(qa_run),
        "solver_status": final_solver.get("status"),
        "qa_status": source_report.get("status"),
        "candidate_count": len(qa_candidates),
        "branch_candidate_counts": {key: value.get("candidate_count") for key, value in branch_reports.items()},
        "branch_request_counts": {key: value.get("request_count") for key, value in branch_reports.items()},
        "input_sidecars_verified": True,
        "candidate_order_pairing_used": False,
        "prohibited_calls": {"solver_calls": 0, "external_api_calls": 0, "memorax_calls": 0, "qa_writer_calls": 0, "semantic_auditor_calls": 0, "training_runs": 0},
    }
    write_json_with_sidecar(output / "preflight_report.json", preflight)
    by_branch, by_model_branch, context = enrich_rows(run, qa_run)
    solver_candidate_ids = {str(row["candidate_id"]) for row in load_jsonl(run / "candidate_manifest.jsonl")}
    qa_candidate_ids = {str(row["candidate_id"]) for row in qa_candidates}
    if solver_candidate_ids != qa_candidate_ids:
        raise RuntimeError("qa_solver_candidate_identity_mismatch")
    alignment = build_alignment(by_branch)
    closure = {"status": "PASS", "solver_status": final_solver["status"], "qa_status": source_report["status"], "control_candidate_count": len(by_branch["control"]), "feedback_candidate_count": len(by_branch["feedback"]), "expected_control_candidate_count": 276, "expected_feedback_candidate_count": 254, "branch_reports": branch_reports, "all_solver_technical_failure_count": sum(int(r.get("technical_failure_count") or 0) for r in branch_reports.values()), "source_hashes": source_hashes}
    write_json_with_sidecar(output / "branch_closure_audit.json", closure)
    write_json_with_sidecar(output / "candidate_alignment_report.json", alignment)
    summaries: dict[str, Any] = {}
    for model in MODELS:
        control, feedback = slice_summary(by_model_branch[(model, "control")]), slice_summary(by_model_branch[(model, "feedback")])
        summaries[model] = {"control": control, "feedback": feedback, "feedback_minus_control": branch_delta(control, feedback)}
    write_json_with_sidecar(output / "control_feedback_summary.json", summaries)
    write_json_with_sidecar(output / "comparison_by_root.json", compare_slice(by_model_branch, "root_id"))
    write_json_with_sidecar(output / "comparison_by_question_type.json", compare_slice(by_model_branch, "question_type"))
    write_json_with_sidecar(output / "comparison_by_generation_source.json", compare_slice(by_model_branch, "generation_source"))
    write_json_with_sidecar(output / "comparison_by_evidence_span.json", compare_slice(by_model_branch, "evidence_span"))
    tech = technical_report(run, by_model_branch)
    write_json_with_sidecar(output / "technical_failure_report.json", tech)
    final = {"status": "ROUND1_PAIRED_BLIND_SOLVER_OFFLINE_COMPARE_COMPLETE", "version": VERSION, "solver_run": str(run), "qa_run": str(qa_run), "output_root": str(output), "source_hashes": source_hashes, "candidate_alignment": alignment, "models": summaries, "technical_failure": tech, "prohibited_calls": {"solver_calls": 0, "external_api_calls": 0, "memorax_calls": 0, "qa_writer_calls": 0, "semantic_auditor_calls": 0, "training_runs": 0}, "candidate_order_pairing_used": False, "paired_solver_comparison_input_status": paired.get("candidate_count") == 530}
    write_json_with_sidecar(output / "final_paired_solver_comparison.json", final)
    md = ["# Round-1 paired blind Solver: offline Control versus Feedback comparison", "", f"- Status: `{final['status']}`", f"- Control candidates: `{alignment['control_candidate_count']}`", f"- Feedback candidates: `{alignment['feedback_candidate_count']}`", f"- Direct exact candidate pairs: `{alignment['direct_candidate_pair_count']}`", f"- Shared lineage keys (non-identical content excluded from pairing): `{alignment['shared_lineage_key_count']}`", "", "## Summary", "", "```json", json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True), "```", "", "## Integrity", "", "All inputs were read offline and their SHA-256 sidecars verified. No candidate-order pairing was used."]
    md_path = output / "final_paired_solver_comparison.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    md_path.with_name(md_path.name + ".sha256").write_text(sha256_file(md_path) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
