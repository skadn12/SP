"""Purely offline comparison of E1Q1 and frozen E1Q0 Feedback Solver data.

The two arms contain independently generated questions.  This report therefore
uses composition-aware summaries and within-root descriptive differences; it
never pairs candidates by order or treats shared lineage as identical content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import BASELINE_RUN, E1Q1_GENERATION_RUN, E1Q1_RUN, MODELS, OLD_QA_RUN, OUTPUT_RUN, VERSION


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    if not path.is_file():
        raise ValueError(f"missing_source:{path}")
    observed = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing_sha_sidecar:{path}")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if not tokens or tokens[0] != observed:
        raise ValueError(f"sha_mismatch:{path}:{tokens[:1]}:{observed}")
    return observed


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def candidate_signature(row: Mapping[str, Any]) -> str:
    fields = (
        "root_id", "source_candidate_id", "question_type", "generation_source",
        "question", "gold", "evidence_refs", "options", "items",
    )
    return json_key({field: row.get(field) for field in fields})


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


def _dialogue_index() -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    # E1Q1's frozen dialogue source is also the E1Q0 Feedback dialogue for the
    # seven common roots.  The old paired manifest supplies the excluded root.
    for root_dir in sorted(E1Q1_GENERATION_RUN.glob("cc-train-episode-*/")):
        path = root_dir / "dialogue_source.json"
        verify_sidecar(path)
        paths[root_dir.name] = path
        expected[root_dir.name] = sha256_file(path)
    manifest = OLD_QA_RUN / "paired_dialogue_manifest.jsonl"
    verify_sidecar(manifest)
    for row in load_jsonl(manifest):
        root = str(row["root_id"])
        path = Path(str(row["feedback_path"]))
        observed = sha256_file(path)
        if observed != str(row["feedback_sha256"]):
            raise ValueError(f"old_feedback_dialogue_sha_mismatch:{root}")
        if root in expected and expected[root] != observed:
            raise ValueError(f"common_feedback_dialogue_sha_mismatch:{root}:{expected[root]}:{observed}")
        paths.setdefault(root, path)
        expected.setdefault(root, observed)
    return paths, expected


def _load_arm(run: Path, *, baseline: bool, dialogues: Mapping[str, Path]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest_path = run / "candidate_manifest.jsonl"
    manifest_sha = verify_sidecar(manifest_path)
    manifest = load_jsonl(manifest_path)
    if baseline:
        manifest = [row for row in manifest if row.get("branch") == "feedback"]
    ids = [str(row.get("candidate_id")) for row in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate_candidate_ids:{run}")
    by_model: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {"candidate_manifest": manifest_sha}
    for model in MODELS:
        scope = run / model / "feedback"
        difficulty_path = scope / "difficulty_manifest.jsonl"
        result_path = scope / "candidate_results.jsonl"
        branch_report = scope / "branch_report.json"
        source_hashes[f"{model}_difficulty_manifest"] = verify_sidecar(difficulty_path)
        source_hashes[f"{model}_candidate_results"] = verify_sidecar(result_path)
        verify_sidecar(branch_report)
        difficulties = {str(row["candidate_id"]): dict(row) for row in load_jsonl(difficulty_path)}
        results = {str(row["candidate_id"]): dict(row) for row in load_jsonl(result_path)}
        if set(difficulties) != set(ids) or set(results) != set(ids):
            raise ValueError(f"candidate_set_mismatch:{run}:{model}")
        merged = []
        for candidate in manifest:
            cid = str(candidate["candidate_id"])
            row = {**candidate, **difficulties[cid]}
            if str(row.get("root_id")) != str(candidate.get("root_id")):
                raise ValueError(f"root_mismatch:{run}:{model}:{cid}")
            row["result"] = results[cid]
            dialogue = load_json(dialogues[str(row["root_id"])])
            row["evidence_span"] = evidence_span(candidate, dialogue)
            row["arm"] = "E1Q0_feedback" if baseline else "E1Q1_feedback"
            merged.append(row)
        by_model[model] = merged
    return by_model, {"source_hashes": source_hashes, "candidate_count": len(manifest), "roots": sorted({str(row["root_id"]) for row in manifest})}


def _counts(row: Mapping[str, Any], context: str) -> tuple[int, int]:
    diagnostic = (row.get("position_diagnostics") or {}).get(context) or {}
    total = int(diagnostic.get("total_count") or row.get(f"{context}_total_count") or 0)
    correct = int(diagnostic.get("correct_count") or row.get(f"{context}_correct_count") or 0)
    return total, correct


def arm_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def pooled(context: str) -> dict[str, Any]:
        total = sum(_counts(row, context)[0] for row in rows)
        correct = sum(_counts(row, context)[1] for row in rows)
        return {"correct": correct, "total": total, "accuracy": correct / total if total else None}

    def pooled_for(subrows: list[Mapping[str, Any]], context: str) -> dict[str, Any]:
        total = sum(_counts(row, context)[0] for row in subrows)
        correct = sum(_counts(row, context)[1] for row in subrows)
        return {"correct": correct, "total": total, "accuracy": correct / total if total else None}

    pairs = {context: pooled(context) for context in ("full_context", "no_context")}
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_root[str(row["root_id"])].append(row)
    root_metrics = {}
    for root, root_rows in sorted(by_root.items()):
        full, no = pooled_for(root_rows, "full_context"), pooled_for(root_rows, "no_context")
        root_metrics[root] = {"candidate_count": len(root_rows), "full_context_accuracy": full["accuracy"], "no_context_accuracy": no["accuracy"], "context_gain": full["accuracy"] - no["accuracy"] if full["accuracy"] is not None and no["accuracy"] is not None else None}
    root_equal = {}
    for key in ("full_context_accuracy", "no_context_accuracy", "context_gain"):
        values = [item[key] for item in root_metrics.values() if finite(item.get(key))]
        root_equal[key] = mean(values)
    return {"candidate_count": len(rows), "candidate_weighted": {"full_context": pairs["full_context"], "no_context": pairs["no_context"], "context_gain": pairs["full_context"]["accuracy"] - pairs["no_context"]["accuracy"] if pairs["full_context"]["accuracy"] is not None and pairs["no_context"]["accuracy"] is not None else None}, "root_equal": root_equal, "root_metrics": root_metrics, "question_type_counts": dict(sorted(Counter(str(row.get("question_type")) for row in rows).items())), "generation_source_counts": dict(sorted(Counter(str(row.get("generation_source")) for row in rows).items())), "evidence_span_counts": dict(sorted(Counter(str(row.get("evidence_span")) for row in rows).items())), "technical_failure_count": sum(int(row.get("technical_failure_count") or 0) for row in rows)}


def grouped_metrics(rows: list[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "UNKNOWN")].append(row)
    return {name: arm_metrics(group) for name, group in sorted(groups.items())}


def delta(e1q1: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for weighting in ("candidate_weighted", "root_equal"):
        result[weighting] = {}
        for metric in ("full_context_accuracy", "no_context_accuracy", "context_gain"):
            if weighting == "candidate_weighted":
                context = "full_context" if metric == "full_context_accuracy" else "no_context"
                e = e1q1[weighting][metric] if metric == "context_gain" else e1q1[weighting][context]["accuracy"]
                b = baseline[weighting][metric] if metric == "context_gain" else baseline[weighting][context]["accuracy"]
            else:
                e, b = e1q1[weighting][metric], baseline[weighting][metric]
            result[weighting][metric] = e - b if finite(e) and finite(b) else None
    return result


def alignment(e1q1_rows: list[Mapping[str, Any]], baseline_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    e1_ids = {str(row["candidate_id"]) for row in e1q1_rows}
    b_ids = {str(row["candidate_id"]) for row in baseline_rows}
    e1_sources = {str(row.get("source_candidate_id")) for row in e1q1_rows if row.get("source_candidate_id")}
    b_sources = {str(row.get("source_candidate_id")) for row in baseline_rows if row.get("source_candidate_id")}
    e1_signatures = {candidate_signature(row) for row in e1q1_rows}
    b_signatures = {candidate_signature(row) for row in baseline_rows}
    return {"e1q1_candidate_count": len(e1_ids), "e1q0_feedback_candidate_count": len(b_ids), "shared_candidate_id_count": len(e1_ids & b_ids), "shared_source_candidate_id_count": len(e1_sources & b_sources), "strict_identical_candidate_signature_count": len(e1_signatures & b_signatures), "candidate_order_pairing_used": False, "note": "E1Q1 and E1Q0 questions are independently generated; shared lineage IDs are not treated as identical candidates."}


def common_root_rows(rows: list[Mapping[str, Any]], roots: set[str]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row["root_id"]) in roots]


def write_json_with_sidecar(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def make_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# E1Q1 QA-feedback Solver offline comparison",
        "",
        "- Status: `E1Q1_QA_FEEDBACK_OFFLINE_COMPARE_COMPLETE`",
        "- Solver/API/Memorax/QA Writer/auditor/training calls: `0`",
        "- Candidate-order pairing: `false`",
        "- Main comparison roots: `" + str(report["common_root_count"]) + "`",
        "",
        "E1Q1 is compared with the frozen E1Q0 Feedback baseline. The questions are independently generated, so the differences are descriptive composition shifts, not candidate-level causal estimates.",
        "",
        "| Model | Weighting | Full-context E1Q0 | Full-context E1Q1 | E1Q1-E1Q0 | No-context E1Q0 | No-context E1Q1 | Context-gain delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        item = report["models"][model]["common_root_comparison"]
        for weighting in ("candidate_weighted", "root_equal"):
            e = item["e1q1"][weighting]
            b = item["e1q0"][weighting]
            d = item["delta"][weighting]
            def fmt(value):
                return "NA" if value is None else f"{value:.4f}"
            lines.append(f"| {model} | {weighting} | {fmt(b['full_context_accuracy'] if weighting == 'root_equal' else b['full_context']['accuracy'])} | {fmt(e['full_context_accuracy'] if weighting == 'root_equal' else e['full_context']['accuracy'])} | {fmt(d['full_context_accuracy'])} | {fmt(b['no_context_accuracy'] if weighting == 'root_equal' else b['no_context']['accuracy'])} | {fmt(e['no_context_accuracy'] if weighting == 'root_equal' else e['no_context']['accuracy'])} | {fmt(d['context_gain'])} |")
    lines += [
        "",
        "The report also contains root-level differences and strata by question type, generation source, and evidence span. Root `96462` is excluded from the common-root main comparison because E1Q1 has no safe QA-feedback seed for it; its baseline-only count is reported separately.",
        "",
    ]
    return "\n".join(lines)


def run(output: Path = OUTPUT_RUN) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dialogue_paths, dialogue_hashes = _dialogue_index()
    baseline, baseline_meta = _load_arm(BASELINE_RUN, baseline=True, dialogues=dialogue_paths)
    e1q1, e1q1_meta = _load_arm(E1Q1_RUN, baseline=False, dialogues=dialogue_paths)
    common_roots = set(baseline_meta["roots"]) & set(e1q1_meta["roots"])
    baseline_only_roots = sorted(set(baseline_meta["roots"]) - common_roots)
    e1q1_only_roots = sorted(set(e1q1_meta["roots"]) - common_roots)
    models: dict[str, Any] = {}
    for model in MODELS:
        b_all, e_all = baseline[model], e1q1[model]
        b_common, e_common = common_root_rows(b_all, common_roots), common_root_rows(e_all, common_roots)
        b_main, e_main = arm_metrics(b_common), arm_metrics(e_common)
        models[model] = {
            "all_available": {"e1q0_feedback": arm_metrics(b_all), "e1q1_feedback": arm_metrics(e_all)},
            "common_root_comparison": {"e1q0": b_main, "e1q1": e_main, "delta": delta(e_main, b_main), "by_question_type": {"e1q0": grouped_metrics(b_common, "question_type"), "e1q1": grouped_metrics(e_common, "question_type")}, "by_generation_source": {"e1q0": grouped_metrics(b_common, "generation_source"), "e1q1": grouped_metrics(e_common, "generation_source")}, "by_evidence_span": {"e1q0": grouped_metrics(b_common, "evidence_span"), "e1q1": grouped_metrics(e_common, "evidence_span")}},
            "root_differences": {root: {"e1q0": arm_metrics([row for row in b_common if str(row["root_id"]) == root]), "e1q1": arm_metrics([row for row in e_common if str(row["root_id"]) == root]), "delta": delta(arm_metrics([row for row in e_common if str(row["root_id"]) == root]), arm_metrics([row for row in b_common if str(row["root_id"]) == root]))} for root in sorted(common_roots)},
        }
    report = {
        "status": "E1Q1_QA_FEEDBACK_OFFLINE_COMPARE_COMPLETE",
        "version": VERSION,
        "baseline_run": str(BASELINE_RUN),
        "e1q1_run": str(E1Q1_RUN),
        "common_root_count": len(common_roots),
        "common_roots": sorted(common_roots),
        "baseline_only_roots": baseline_only_roots,
        "e1q1_only_roots": e1q1_only_roots,
        "baseline_only_candidate_counts": {root: sum(str(row["root_id"]) == root for row in baseline[MODELS[0]]) for root in baseline_only_roots},
        "dialogue_sha256": dialogue_hashes,
        "input_metadata": {"e1q0": baseline_meta, "e1q1": e1q1_meta},
        "alignment": alignment(e1q1[MODELS[0]], baseline[MODELS[0]]),
        "models": models,
        "solver_calls": 0,
        "api_calls": 0,
        "memorax_calls": 0,
        "qa_writer_calls": 0,
        "semantic_auditor_calls": 0,
        "training_calls": 0,
    }
    write_json_with_sidecar(output / "final_qa_feedback_solver_compare_report.json", report)
    markdown = make_markdown(report) + "\n"
    (output / "final_qa_feedback_solver_compare_report.md").write_text(markdown, encoding="utf-8")
    (output / "final_qa_feedback_solver_compare_report.md.sha256").write_text(sha256_file(output / "final_qa_feedback_solver_compare_report.md") + "\n", encoding="utf-8")
    write_json_with_sidecar(output / "input_alignment.json", report["alignment"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_RUN)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
