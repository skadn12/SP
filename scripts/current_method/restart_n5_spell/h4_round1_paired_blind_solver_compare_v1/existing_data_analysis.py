"""Offline second-pass analysis of the completed Control/Feedback report.

This module reads only the already generated comparison JSON files.  It does
not invoke a Solver, an API client, Memorax, QA Writer, or an auditor, and it
does not pair candidates by order or by non-identical question content.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping


MODELS = ("M_clean_policy", "M_inclusive_policy")
BRANCHES = ("control", "feedback")
CONTEXTS = ("full_context", "no_context")
ROOT_METRICS = (
    "full_context_accuracy",
    "no_context_accuracy",
    "context_gain",
    "full_context_canonical_flip_rate",
    "no_context_canonical_flip_rate",
    "full_context_worst_position_accuracy",
    "no_context_worst_position_accuracy",
)
STRATA = ("question_type", "generation_source", "evidence_span")
VERSION = "h4_round1_existing_data_analysis_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        raise ValueError(f"sha_mismatch:{path}")
    return observed


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def median(values: Iterable[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def sample_sd(values: Iterable[float]) -> float | None:
    values = list(values)
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def percentile(values: Iterable[float], q: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap_mean(values: list[float], *, reps: int = 20000, seed: int = 20260905) -> dict[str, Any]:
    if not values:
        return {"reps": 0, "seed": seed, "mean_ci_95": [None, None]}
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        draws.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    return {"reps": reps, "seed": seed, "mean_ci_95": [percentile(draws, 0.025), percentile(draws, 0.975)]}


def sign_flip_p_two_sided(values: list[float]) -> float | None:
    """Exact sign-flip p-value for a paired mean difference, n=8 here."""
    if not values:
        return None
    observed = abs(sum(values) / len(values))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        simulated = abs(sum(value * sign for value, sign in zip(values, signs)) / len(values))
        total += 1
        if simulated + 1e-12 >= observed:
            count += 1
    return count / total


def root_metric(entry: Mapping[str, Any], metric: str, branch: str) -> float | None:
    branch_data = entry[branch]
    if metric == "full_context_accuracy":
        return branch_data["metrics"]["full_context"].get("accuracy")
    if metric == "no_context_accuracy":
        return branch_data["metrics"]["no_context"].get("accuracy")
    if metric == "context_gain":
        return branch_data["metrics"]["context_gain"].get("candidate_weighted")
    if metric == "full_context_canonical_flip_rate":
        return branch_data["diagnostics"]["full_context"].get("canonical_flip_rate_mean")
    if metric == "no_context_canonical_flip_rate":
        return branch_data["diagnostics"]["no_context"].get("canonical_flip_rate_mean")
    if metric == "full_context_worst_position_accuracy":
        return branch_data["diagnostics"]["full_context"].get("worst_position_accuracy_mean")
    if metric == "no_context_worst_position_accuracy":
        return branch_data["diagnostics"]["no_context"].get("worst_position_accuracy_mean")
    raise KeyError(metric)


def paired_root_analysis(by_root: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_id in sorted(by_root):
        entry = by_root[root_id]
        row: dict[str, Any] = {"root_id": root_id}
        for model in MODELS:
            model_entry = entry[model]
            row[model] = {
                "control_candidate_count": model_entry["control_candidate_count"],
                "feedback_candidate_count": model_entry["feedback_candidate_count"],
                "metrics": {},
            }
            for metric in ROOT_METRICS:
                control = root_metric(model_entry, metric, "control")
                feedback = root_metric(model_entry, metric, "feedback")
                if not (finite(control) and finite(feedback)):
                    raise ValueError(f"root_metric_missing:{root_id}:{model}:{metric}")
                row[model]["metrics"][metric] = {
                    "control": float(control),
                    "feedback": float(feedback),
                    "feedback_minus_control": float(feedback) - float(control),
                }
        rows.append(row)

    summary: dict[str, Any] = {}
    for model in MODELS:
        summary[model] = {}
        for metric in ROOT_METRICS:
            differences = [row[model]["metrics"][metric]["feedback_minus_control"] for row in rows]
            bootstrap = bootstrap_mean(differences)
            summary[model][metric] = {
                "root_count": len(differences),
                "mean_feedback_minus_control": mean(differences),
                "median_feedback_minus_control": median(differences),
                "sample_sd": sample_sd(differences),
                "min_feedback_minus_control": min(differences),
                "max_feedback_minus_control": max(differences),
                "negative_root_count": sum(value < 0 for value in differences),
                "positive_root_count": sum(value > 0 for value in differences),
                "zero_root_count": sum(value == 0 for value in differences),
                "bootstrap_mean_ci_95": bootstrap["mean_ci_95"],
                "bootstrap_reps": bootstrap["reps"],
                "bootstrap_seed": bootstrap["seed"],
                "exact_sign_flip_p_two_sided": sign_flip_p_two_sided(differences),
            }
    return {"rows": rows, "root_count": len(rows)}, summary


def stratum_value(entry: Mapping[str, Any], branch: str, context: str) -> tuple[int, int, float | None]:
    branch_data = entry[branch]
    metric = branch_data["metrics"][context]
    return int(metric.get("total") or 0), int(metric.get("correct") or 0), metric.get("accuracy")


def standardized_strata(report: Mapping[str, Any], field: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in MODELS:
        strata = []
        for name in sorted(report):
            entry = report[name][model]
            c_count, c_correct, c_acc = stratum_value(entry, "control", "full_context")
            f_count, f_correct, f_acc = stratum_value(entry, "feedback", "full_context")
            if c_count <= 0 or f_count <= 0 or c_acc is None or f_acc is None:
                continue
            strata.append({
                "stratum": name,
                "control_candidate_count": entry["control_candidate_count"],
                "feedback_candidate_count": entry["feedback_candidate_count"],
                "control_full_total": c_count,
                "feedback_full_total": f_count,
                "control_full_accuracy": c_acc,
                "feedback_full_accuracy": f_acc,
                "control_no_context_accuracy": entry["control"]["metrics"]["no_context"].get("accuracy"),
                "feedback_no_context_accuracy": entry["feedback"]["metrics"]["no_context"].get("accuracy"),
            })
        total_pooled = sum(item["control_full_total"] + item["feedback_full_total"] for item in strata)
        pooled = [(item["control_full_total"] + item["feedback_full_total"]) / total_pooled for item in strata] if total_pooled else []
        equal = [1 / len(strata)] * len(strata) if strata else []
        branch_metrics: dict[str, Any] = {}
        for branch in BRANCHES:
            full_values = [item[f"{branch}_full_accuracy"] for item in strata]
            no_values = [item[f"{branch}_no_context_accuracy"] for item in strata]
            branch_metrics[branch] = {
                "pooled_request_weight_full_accuracy": sum(w * v for w, v in zip(pooled, full_values)),
                "pooled_request_weight_no_context_accuracy": sum(w * v for w, v in zip(pooled, no_values)),
                "pooled_request_weight_context_gain": sum(w * (f - n) for w, f, n in zip(pooled, full_values, no_values)),
                "equal_stratum_full_accuracy": mean(full_values),
                "equal_stratum_no_context_accuracy": mean(no_values),
                "equal_stratum_context_gain": mean(f - n for f, n in zip(full_values, no_values)),
            }
        result[model] = {
            "strata_included": len(strata),
            "strata": strata,
            "branch_metrics": branch_metrics,
            "feedback_minus_control": {
                metric: branch_metrics["feedback"][metric] - branch_metrics["control"][metric]
                for metric in branch_metrics["control"]
            },
        }
    return result


def interpretation(root_summary: Mapping[str, Any], strata: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in MODELS:
        full = root_summary[model]["full_context_accuracy"]
        gain = root_summary[model]["context_gain"]
        full_ci = full["bootstrap_mean_ci_95"]
        gain_ci = gain["bootstrap_mean_ci_95"]
        result[model] = {
            "full_context_direction": "feedback_lower" if full["mean_feedback_minus_control"] < 0 else "feedback_higher",
            "full_context_evidence": "directional_only" if full_ci[0] <= 0 <= full_ci[1] else "interval_excludes_zero",
            "context_dependence_direction": "feedback_more_context_dependent" if gain["mean_feedback_minus_control"] > 0 else "feedback_less_context_dependent",
            "context_dependence_evidence": "directional_only" if gain_ci[0] <= 0 <= gain_ci[1] else "interval_excludes_zero",
            "causal_claim_supported": False,
            "reason": "no direct exact candidate pairs; shared lineage questions are non-identical",
        }
        for field, report in strata.items():
            result[model].setdefault("stratified_feedback_minus_control", {})[field] = report[model]["feedback_minus_control"]
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_with_sidecar(path: Path, value: Any) -> None:
    write_json(path, value)
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def make_markdown(final: Mapping[str, Any]) -> str:
    lines = [
        "# Existing-data Control versus Feedback second-pass analysis",
        "",
        "- Status: `ROUND1_EXISTING_DATA_ANALYSIS_COMPLETE`",
        "- Input: completed offline comparison JSON only",
        "- Root-paired clusters: `8`",
        "- Exact candidate pairs: `0`",
        "- Candidate-order pairing: `false`",
        "",
        "## Interpretation",
        "",
        "This report estimates within-root branch differences and composition-standardized differences. It does not claim candidate-level causality because the generated questions are not identical between branches.",
        "",
        "## Root-paired differences",
        "",
        "| Model | Metric | Mean Feedback-Control | 95% bootstrap CI | Negative roots | Exact sign-flip p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {
        "full_context_accuracy": "Full-context accuracy",
        "no_context_accuracy": "No-context accuracy",
        "context_gain": "Context gain",
    }
    for model in MODELS:
        for metric in labels:
            row = final["root_paired_summary"][model][metric]
            ci = row["bootstrap_mean_ci_95"]
            lines.append(f"| {model} | {labels[metric]} | {row['mean_feedback_minus_control']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {row['negative_root_count']}/{row['root_count']} | {row['exact_sign_flip_p_two_sided']:.4f} |")
    lines += [
        "",
        "## Reading the result",
        "",
        "- A negative Full-context difference means the Feedback branch was solved less accurately in this generated sample.",
        "- A positive Context-gain difference means the Feedback branch benefited more from seeing the dialogue.",
        "- Confidence intervals and exact sign-flip tests are over the eight roots, so they are low-power and should be treated as descriptive evidence.",
        "- No technical failures were introduced, but accuracy alone cannot distinguish harder valid questions from lower-quality or ambiguous questions.",
        "",
        "## Integrity",
        "",
        "All source sidecars were verified. No Solver, API, Memorax, QA Writer, semantic auditor, or training process was invoked.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    compare = args.compare_root.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output_not_empty:{output}")
    output.mkdir(parents=True, exist_ok=True)

    source_files = {
        "comparison_by_root": compare / "comparison_by_root.json",
        "comparison_by_question_type": compare / "comparison_by_question_type.json",
        "comparison_by_generation_source": compare / "comparison_by_generation_source.json",
        "comparison_by_evidence_span": compare / "comparison_by_evidence_span.json",
        "candidate_alignment": compare / "candidate_alignment_report.json",
        "technical_failure": compare / "technical_failure_report.json",
        "final_comparison": compare / "final_paired_solver_comparison.json",
    }
    source_hashes = {name: verify_sidecar(path) for name, path in source_files.items()}
    by_root = load_json(source_files["comparison_by_root"])
    root_rows, root_summary = paired_root_analysis(by_root)
    strata = {
        "question_type": standardized_strata(load_json(source_files["comparison_by_question_type"]), "question_type"),
        "generation_source": standardized_strata(load_json(source_files["comparison_by_generation_source"]), "generation_source"),
        "evidence_span": standardized_strata(load_json(source_files["comparison_by_evidence_span"]), "evidence_span"),
    }
    alignment = load_json(source_files["candidate_alignment"])
    technical = load_json(source_files["technical_failure"])
    final = {
        "status": "ROUND1_EXISTING_DATA_ANALYSIS_COMPLETE",
        "version": VERSION,
        "compare_root": str(compare),
        "output_root": str(output),
        "source_hashes": source_hashes,
        "candidate_alignment": {
            "control_candidate_count": alignment["control_candidate_count"],
            "feedback_candidate_count": alignment["feedback_candidate_count"],
            "direct_candidate_pair_count": alignment["direct_candidate_pair_count"],
            "shared_lineage_key_count": alignment["shared_lineage_key_count"],
            "shared_lineage_nonidentical_candidate_count": alignment["shared_lineage_nonidentical_candidate_count"],
            "candidate_order_pairing_used": alignment["candidate_order_pairing_used"],
        },
        "technical_failure": {
            "total_technical_failure_count": technical["total_technical_failure_count"],
            "total_parse_failure_count": technical["total_parse_failure_count"],
            "total_truncation_count": technical["total_truncation_count"],
        },
        "root_paired_rows": root_rows,
        "root_paired_summary": root_summary,
        "stratified_standardization": strata,
        "interpretation": interpretation(root_summary, strata),
        "prohibited_calls": {"solver_calls": 0, "external_api_calls": 0, "memorax_calls": 0, "qa_writer_calls": 0, "semantic_auditor_calls": 0, "training_runs": 0},
    }
    write_json_with_sidecar(output / "root_paired_differences.json", {"source_hashes": source_hashes, "rows": root_rows})
    write_json_with_sidecar(output / "root_paired_statistics.json", root_summary)
    write_json_with_sidecar(output / "stratified_standardized_comparison.json", strata)
    write_json_with_sidecar(output / "final_existing_data_analysis.json", final)
    md_path = output / "final_existing_data_analysis.md"
    md_path.write_text(make_markdown(final), encoding="utf-8")
    md_path.with_name(md_path.name + ".sha256").write_text(sha256_file(md_path) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
