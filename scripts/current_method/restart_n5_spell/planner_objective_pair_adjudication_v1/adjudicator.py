from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from scripts.current_method.restart_n5_spell.scene_complexity_probe_v1.analyzer import (
    analyze_chain,
    summarize,
)
from scripts.current_method.restart_n5_spell.planner_objective_probe_v1.contracts import (
    OBJECTIVES,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_SOURCE = RUN_BASE / "planner_objective_probe_v1_batch_20260822T110000Z"
PROBE_VERSION = "planner_objective_pair_adjudication_v2"
ADJUDICATION_RULE = "surface_continuity_waiver_v2"

VALUE_DISCONTINUITY = re.compile(r"^scene_(\d+)_to_(\d+)_value_discontinuity$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
INTENTION_RE = re.compile(
    r"\b(?:consider(?:s|ed|ing)?|decid(?:e|es|ed|ing)|intend(?:s|ed|ing)?|"
    r"may|might|plan(?:s|ned|ning)?|propos(?:e|es|ed|ing)|tentativ(?:e|ely)|will)\b"
)
NEGATION_RE = re.compile(r"\b(?:cannot|can't|didn't|doesn't|don't|never|no|not|won't)\b")
NUMBER_RE = re.compile(r"\b\d+(?::\d+)?\b")

# These tokens do not identify a state value by themselves. Removing them makes
# the waiver tolerant to tense, articles, and light connective rewrites while
# keeping content-bearing nouns and role markers in the comparison.
SURFACE_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "the", "their", "to",
    "was", "were", "with",
}


def _surface_tokens(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return [token for token in TOKEN_RE.findall(text) if token not in SURFACE_STOPWORDS]


def surface_continuity(left: Any, right: Any) -> dict[str, Any]:
    """Return evidence for a conservative lexical-only continuity waiver.

    This is not semantic paraphrase and never changes either value. It only
    recognizes a near-identical adjacent value whose surface wording differs.
    """
    left_text = " ".join(str(left or "").split())
    right_text = " ".join(str(right or "").split())
    left_tokens = _surface_tokens(left_text)
    right_tokens = _surface_tokens(right_text)
    left_set, right_set = set(left_tokens), set(right_tokens)
    union = left_set | right_set
    overlap = left_set & right_set
    jaccard = len(overlap) / len(union) if union else 0.0
    sequence_ratio = SequenceMatcher(
        None,
        " ".join(left_tokens),
        " ".join(right_tokens),
    ).ratio()
    content_overlap = sorted(token for token in overlap if len(token) >= 4)
    left_normalized = unicodedata.normalize("NFKC", left_text).casefold()
    right_normalized = unicodedata.normalize("NFKC", right_text).casefold()
    intention_profile = {
        "left": sorted(set(INTENTION_RE.findall(left_normalized))),
        "right": sorted(set(INTENTION_RE.findall(right_normalized))),
    }
    negation_profile = {
        "left": bool(NEGATION_RE.search(left_normalized)),
        "right": bool(NEGATION_RE.search(right_normalized)),
    }
    number_profile = {
        "left": sorted(set(NUMBER_RE.findall(left_normalized))),
        "right": sorted(set(NUMBER_RE.findall(right_normalized))),
    }
    semantic_guards_passed = bool(
        bool(intention_profile["left"]) == bool(intention_profile["right"])
        and negation_profile["left"] == negation_profile["right"]
        and number_profile["left"] == number_profile["right"]
    )
    accepted = bool(
        left_tokens
        and right_tokens
        and jaccard >= 0.55
        and sequence_ratio >= 0.78
        and len(content_overlap) >= 3
        and semantic_guards_passed
    )
    return {
        "accepted": accepted,
        "left": left_text,
        "right": right_text,
        "left_tokens": left_tokens,
        "right_tokens": right_tokens,
        "content_overlap": content_overlap,
        "intention_profile": intention_profile,
        "negation_profile": negation_profile,
        "number_profile": number_profile,
        "semantic_guards_passed": semantic_guards_passed,
        "jaccard": round(jaccard, 6),
        "sequence_ratio": round(sequence_ratio, 6),
    }


def _value_discontinuities(errors: list[str]) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []
    for error in errors:
        match = VALUE_DISCONTINUITY.match(error)
        if match:
            boundaries.append((int(match.group(1)), int(match.group(2))))
    return boundaries


def adjudicate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Adjudicate one raw Planner record without modifying its plan."""
    raw_contract = dict(record.get("contract") or {})
    raw_errors = [str(error) for error in raw_contract.get("errors") or []]
    plan = record.get("plan") if isinstance(record.get("plan"), Mapping) else {}
    scenes = plan.get("scenes") if isinstance(plan.get("scenes"), list) else []
    boundaries = _value_discontinuities(raw_errors)
    evidence: list[dict[str, Any]] = []
    for left_index, right_index in boundaries:
        if not (1 <= left_index <= len(scenes) and 1 <= right_index <= len(scenes)):
            return {
                "status": "rejected",
                "effective_valid": False,
                "reason": "boundary_out_of_range",
                "raw_errors": raw_errors,
                "waived_errors": [],
                "evidence": [],
            }
        left = scenes[left_index - 1]
        right = scenes[right_index - 1]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return {
                "status": "rejected",
                "effective_valid": False,
                "reason": "boundary_scene_not_object",
                "raw_errors": raw_errors,
                "waived_errors": [],
                "evidence": [],
            }
        evidence.append(surface_continuity(left.get("after_value"), right.get("before_value")))

    only_surface_errors = bool(raw_errors) and len(boundaries) == len(raw_errors)
    if bool(raw_contract.get("valid")):
        return {
            "status": "original_valid",
            "effective_valid": True,
            "reason": "raw_contract_valid",
            "raw_errors": raw_errors,
            "waived_errors": [],
            "evidence": [],
        }
    if only_surface_errors and evidence and all(row["accepted"] for row in evidence):
        return {
            "status": "adjudicated_surface_continuity",
            "effective_valid": True,
            "reason": "only_adjacent_value_surface_variation",
            "raw_errors": raw_errors,
            "waived_errors": raw_errors,
            "evidence": evidence,
        }
    return {
        "status": "rejected",
        "effective_valid": False,
        "reason": "not_a_surface_only_continuity_difference",
        "raw_errors": raw_errors,
        "waived_errors": [],
        "evidence": evidence,
    }


def _effective_rows(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        adjudication = adjudicate_record(record)
        effective_contract = {
            **dict(record.get("contract") or {}),
            "valid": bool(adjudication["effective_valid"]),
            "adjudication_status": adjudication["status"],
        }
        analysis = analyze_chain(record.get("plan") or {}) if effective_contract["valid"] else {}
        rows.append({
            **dict(record),
            "raw_contract": dict(record.get("contract") or {}),
            "contract": effective_contract,
            "analysis": analysis,
            "adjudication": adjudication,
        })
    return rows


def paired_comparison(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in records:
        by_case.setdefault(str(row.get("case_id") or ""), {})[str(row.get("objective") or "")] = row
    complete = [case_id for case_id, rows in sorted(by_case.items())
                if all(rows.get(key, {}).get("contract", {}).get("valid") for key in OBJECTIVES)]
    outcomes = []
    for case_id in complete:
        o0 = bool(by_case[case_id]["O0"].get("analysis", {}).get("complex_chain"))
        o1 = bool(by_case[case_id]["O1"].get("analysis", {}).get("complex_chain"))
        outcomes.append({
            "case_id": case_id,
            "root_id": by_case[case_id]["O0"].get("root_id"),
            "O0_complex": o0,
            "O1_complex": o1,
            "objective_gain": int(o1) - int(o0),
            "O0_adjudication": by_case[case_id]["O0"].get("adjudication", {}).get("status"),
            "O1_adjudication": by_case[case_id]["O1"].get("adjudication", {}).get("status"),
        })
    positive = sum(row["objective_gain"] > 0 for row in outcomes)
    negative = sum(row["objective_gain"] < 0 for row in outcomes)
    return {
        "paired_complete_cases": len(complete),
        "paired_case_ids": complete,
        "excluded_case_ids": sorted(set(by_case) - set(complete)),
        "pair_outcomes": outcomes,
        "positive_pairs": positive,
        "negative_pairs": negative,
        "unchanged_pairs": len(outcomes) - positive - negative,
        "O0_complex_chain_rate": (
            sum(row["O0_complex"] for row in outcomes) / len(outcomes) if outcomes else None
        ),
        "O1_complex_chain_rate": (
            sum(row["O1_complex"] for row in outcomes) / len(outcomes) if outcomes else None
        ),
        "mean_objective_gain": (
            sum(row["objective_gain"] for row in outcomes) / len(outcomes) if outcomes else None
        ),
    }


def conclusion(paired: Mapping[str, Any]) -> dict[str, Any]:
    count = int(paired.get("paired_complete_cases") or 0)
    positive = int(paired.get("positive_pairs") or 0)
    negative = int(paired.get("negative_pairs") or 0)
    unchanged = int(paired.get("unchanged_pairs") or 0)
    if count < 3:
        return {
            "label": "INSUFFICIENT_COMPLETE_PAIRS_AFTER_ADJUDICATION",
            "reason": f"Only {count} complete pair(s) remain after conservative adjudication; at least 3 are required.",
        }
    if positive >= 2 and negative == 0:
        return {
            "label": "OBJECTIVE_CONTROL_BATCH_SIGNAL_AFTER_ADJUDICATION",
            "reason": f"O1 improved {positive}/{count} complete pairs, with {unchanged} unchanged and no negative pair.",
        }
    if positive > negative:
        return {
            "label": "OBJECTIVE_CONTROL_MIXED_POSITIVE_AFTER_ADJUDICATION",
            "reason": f"O1 improved {positive} pairs, reduced {negative}, and left {unchanged} unchanged.",
        }
    if positive == negative:
        return {
            "label": "NO_STABLE_OBJECTIVE_GAIN_AFTER_ADJUDICATION",
            "reason": f"Positive and negative effects tie at {positive}; {unchanged} pairs are unchanged.",
        }
    return {
        "label": "OBJECTIVE_CONTROL_UNSTABLE_OR_HARMFUL_AFTER_ADJUDICATION",
        "reason": f"Negative effects ({negative}) exceed positive effects ({positive}).",
    }


def _report(metrics: Mapping[str, Any]) -> str:
    paired = metrics["paired_comparison"]
    lines = [
        "# Planner Objective Pair Adjudication v1",
        "",
        "Read-only, zero-API adjudication of the frozen five-root Planner objective batch.",
        "The raw Planner plans are preserved byte-for-byte; only a narrow adjacent-value surface-continuity waiver is evaluated.",
        "",
        "## Source",
        "",
        f"- Source run: `{metrics['source']['run_root']}`",
        f"- Source planner_cases SHA256: `{metrics['source']['planner_cases_sha256']}`",
        f"- Source metrics SHA256: `{metrics['source']['metrics_sha256']}`",
        f"- API calls in this adjudication: `{metrics['api_calls']}`",
        "",
        "## Adjudication rule",
        "",
        "Only errors consisting exclusively of adjacent `*_value_discontinuity` are eligible.",
        "Each boundary must retain at least 3 content-bearing shared tokens, Jaccard >= 0.55, and sequence ratio >= 0.78.",
        "Intention-vs-realization, negation, and numeric-value profiles must also match.",
        "Canonical-speaker errors, stage discontinuities, enum errors, and unrelated value changes remain rejected.",
        "",
        "## Strict paired endpoint after adjudication",
        "",
        f"- Complete pairs: `{paired['paired_complete_cases']}`",
        f"- O0 complex-chain rate: `{paired['O0_complex_chain_rate']}`",
        f"- O1 complex-chain rate: `{paired['O1_complex_chain_rate']}`",
        f"- Mean objective gain: `{paired['mean_objective_gain']}`",
        f"- Positive / unchanged / negative: `{paired['positive_pairs']} / {paired['unchanged_pairs']} / {paired['negative_pairs']}`",
        "",
        "## Pair outcomes",
        "",
    ]
    for row in paired["pair_outcomes"]:
        lines.append(
            f"- `{row['case_id']}` / `{row['root_id']}`: "
            f"O0={row['O0_complex']}, O1={row['O1_complex']}, gain={row['objective_gain']:+d}; "
            f"adjudication=({row['O0_adjudication']}, {row['O1_adjudication']})"
        )
    if not paired["pair_outcomes"]:
        lines.append("- None.")
    lines.extend([
        "",
        "## Conclusion",
        "",
        f"Verdict: **{metrics['conclusion']['label']}**",
        "",
        metrics["conclusion"]["reason"],
        "",
        "This report does not create a new Planner output and does not authorize a mainline change.",
    ])
    return "\n".join(lines) + "\n"


def run_adjudication(*, source_run: Path, run_root: Path) -> dict[str, Any]:
    prepare_empty(run_root)
    planner_path = source_run / "planner_cases.jsonl"
    metrics_path = source_run / "metrics.json"
    if not planner_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"adjudication_source_missing:{source_run}")
    raw_records = read_jsonl(planner_path)
    source_metrics = read_json(metrics_path)
    records = _effective_rows(raw_records)
    by_objective = {
        key: summarize([row for row in records if row.get("objective") == key])
        for key in OBJECTIVES
    }
    paired = paired_comparison(records)
    metrics = {
        "probe_version": PROBE_VERSION,
        "adjudication_rule": ADJUDICATION_RULE,
        "source": {
            "run_root": str(source_run.resolve()),
            "planner_cases_sha256": sha256_file(planner_path),
            "metrics_sha256": sha256_file(metrics_path),
            "source_probe_version": source_metrics.get("probe_version"),
            "source_api_calls": source_metrics.get("api_calls"),
        },
        "dataset": {
            "number_of_roots": source_metrics.get("dataset", {}).get("number_of_roots"),
            "number_of_events": source_metrics.get("dataset", {}).get("number_of_events"),
            "number_of_planner_cases": len(records),
        },
        "by_objective": by_objective,
        "paired_comparison": paired,
        "conclusion": conclusion(paired),
        "api_calls": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
        "limitations": [
            "This is lexical continuity adjudication, not semantic paraphrase or model review.",
            "The raw source batch remains the primary result; waived records are explicitly marked.",
            "No Realizer adherence, QA, Solver, or Memorax behavior is tested.",
        ],
    }
    write_jsonl(run_root / "effective_records.jsonl", records)
    write_jsonl(run_root / "audits.jsonl", [{
        "case_id": row.get("case_id"),
        "root_id": row.get("root_id"),
        "objective": row.get("objective"),
        "raw_contract": row.get("raw_contract"),
        "effective_contract": row.get("contract"),
        "adjudication": row.get("adjudication"),
        "analysis": row.get("analysis"),
    } for row in records])
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION,
        "isolated": True,
        "read_only_source_run": str(source_run.resolve()),
        "api_calls": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
        "outputs": ["effective_records.jsonl", "audits.jsonl", "metrics.json", "report.md", "manifest.json"],
    })
    (run_root / "report.md").write_text(_report(metrics), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="read-only Planner objective pair adjudication")
    parser.add_argument("--replay-from", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_objective_pair_adjudication_v1_{stamp}").resolve()
    metrics = run_adjudication(source_run=args.replay_from.resolve(), run_root=run_root)
    print(json.dumps({"run_root": str(run_root), "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
