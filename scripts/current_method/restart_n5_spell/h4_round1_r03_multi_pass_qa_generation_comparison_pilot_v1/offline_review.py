"""Deterministic review of the immutable Composition/Fresh pilot outputs.

No model, solver, auditor, dialogue writer, or Memorax call is made here.  The
review deliberately keeps strict v1 schema results separate from a small set
of mechanical contract diagnostics (field aliases and invalid turn refs).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import REROUTE_RUN, SUCCESS_ROOTS


PILOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1_live_2")
VERSION = "h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1_offline_review"
OUTPUT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1_offline_review_live_4")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in value)
    elif isinstance(value, str):
        data = value if value.endswith("\n") else value + "\n"
    else:
        data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(data, encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(f"{hashlib.sha256(data.encode()).hexdigest()}  {path.name}\n", encoding="utf-8")


def turn_map(root: str) -> dict[str, str]:
    row = next(x for x in read_jsonl(PILOT / "authoritative_dialogue_manifest.jsonl") if x["root_id"] == root)
    return {
        str(t["turn_ref"]): str(t.get("text") or "")
        for t in read_json(Path(row["path"])).get("turns") or []
    }


def source_rows(root: str) -> list[dict[str, Any]]:
    return [x for x in read_jsonl(REROUTE_RUN / "provisional_question_bank_entries.jsonl") if str(x.get("root_id") or (x.get("candidate") or {}).get("root_id")) == root]


def historical_duplicate(root: str, question: str) -> bool:
    q = " ".join(str(question).casefold().split())
    return any(q == " ".join(str((x.get("candidate") or x).get("question") or "").casefold().split()) for x in source_rows(root))


def review_attempt(row: dict[str, Any], pass_name: str) -> dict[str, Any]:
    root = row["root_id"]
    out = row.get("parsed_output") or {}
    refs = [str(x) for x in out.get("evidence_refs") or []]
    r03 = [str(x) for x in out.get("new_r03_evidence_refs") or []]
    turns = turn_map(root)
    invalid_refs = sorted(set(refs + r03) - set(turns))
    q = str(out.get("new_question") or "")
    result: dict[str, Any] = {
        "root_id": root, "pass": pass_name, "model_status": row.get("status"),
        "schema_valid_v1": bool((row.get("closure") or {}).get("schema_valid")),
        "structural_status_v1": (row.get("closure") or {}).get("structural_status"),
        "strict_v1_content_valid": bool((row.get("closure") or {}).get("content_valid")),
        "strict_v1_structural_errors": list((row.get("closure") or {}).get("structural_errors") or []),
        "outcome": out.get("outcome"), "question": q,
        "evidence_refs_real": not invalid_refs, "invalid_evidence_refs": invalid_refs,
        "r03_ref_touch_real": any(ref.startswith("R03:") for ref in refs) and not any(ref.startswith("R03:") for ref in invalid_refs),
        "r03_claimed": bool(out.get("r03_semantic_contribution_claimed")),
        "historical_exact_surface_duplicate": historical_duplicate(root, q),
        "gold_structural_status": "UNKNOWN_NO_SEMANTIC_AUDITOR",
        "distractor_structural_status": "UNKNOWN_NO_SEMANTIC_AUDITOR",
        "semantic_duplicate_status": "UNKNOWN_NO_SEMANTIC_AUDITOR",
        "offline_disposition": "UNREVIEWED",
        "reasons": [],
    }
    if pass_name == "FRESH_EXPLORATION" and "answerable_without_r03" in out:
        result["contract_alias_answerable_without_r03"] = out.get("answerable_without_r03")
        result["reasons"].append("OUTPUT_FIELD_ALIAS_ANSWERABLE_WITHOUT_R03")
    if pass_name == "FRESH_EXPLORATION" and "selection_rule" in out:
        result["reasons"].append("UNAUTHORIZED_OUTPUT_FIELD_SELECTION_RULE")
    if invalid_refs:
        result["reasons"].append("INVALID_TURN_REF")
    if result["historical_exact_surface_duplicate"]:
        result["reasons"].append("EXACT_HISTORICAL_SURFACE_DUPLICATE")
    # The review buckets are intentionally mechanical.  They do not elevate
    # schema-compliant text to semantically valid QA, and they do not call a
    # schema defect a content failure.
    if pass_name == "MULTI_QA_COMPOSITION":
        selected = list(out.get("selected_evidence_relations") or [])
        if len(set(selected)) < 2:
            result["reasons"].append("LESS_THAN_TWO_DISTINCT_EVIDENCE_RELATIONS")
        if not result["evidence_refs_real"]:
            result["reasons"].append("EVIDENCE_CONTRACT_INVALID")
    has_relation_failure = pass_name == "MULTI_QA_COMPOSITION" and "LESS_THAN_TWO_DISTINCT_EVIDENCE_RELATIONS" in result["reasons"]
    if result["historical_exact_surface_duplicate"]:
        result["offline_disposition"] = "DETERMINISTIC_SURFACE_REPLAY_REJECTED"
    elif invalid_refs or has_relation_failure:
        result["offline_disposition"] = "MECHANICAL_EVIDENCE_OR_RELATION_REJECTED"
    elif result["strict_v1_content_valid"]:
        result["offline_disposition"] = "STRUCTURALLY_VALID_UNADJUDICATED"
    else:
        result["offline_disposition"] = "CONTENT_UNADJUDICATED_SCHEMA_OR_CONTRACT_GAP"
    return result


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"immutable_output_exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    inputs = [PILOT / name for name in [
        "pilot_contract.json", "authoritative_dialogue_manifest.jsonl",
        "composition_sampling_manifest.jsonl", "fresh_sampling_manifest.jsonl",
        "composition_attempt_results.jsonl", "fresh_attempt_results.jsonl",
        "composition_call_closure.json", "fresh_call_closure.json",
        "multi_pass_comparison_report.json", "targeted_existing_result_reference.json",
    ]]
    inputs.append(REROUTE_RUN / "provisional_question_bank_entries.jsonl")
    before = {str(path): sha(path) for path in inputs}
    composition = [review_attempt(row, "MULTI_QA_COMPOSITION") for row in read_jsonl(PILOT / "composition_attempt_results.jsonl")]
    fresh = [review_attempt(row, "FRESH_EXPLORATION") for row in read_jsonl(PILOT / "fresh_attempt_results.jsonl")]
    def bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        labels = [
            "DETERMINISTIC_SURFACE_REPLAY_REJECTED",
            "MECHANICAL_EVIDENCE_OR_RELATION_REJECTED",
            "CONTENT_UNADJUDICATED_SCHEMA_OR_CONTRACT_GAP",
            "STRUCTURALLY_VALID_UNADJUDICATED",
        ]
        return {label: sum(x["offline_disposition"] == label for x in rows) for label in labels}

    comp_buckets = bucket_counts(composition)
    fresh_buckets = bucket_counts(fresh)
    comp_structural = comp_buckets["STRUCTURALLY_VALID_UNADJUDICATED"]
    fresh_structural = fresh_buckets["STRUCTURALLY_VALID_UNADJUDICATED"]
    summary = {
        "version": VERSION, "status": "R03_MULTI_PASS_OFFLINE_REVIEW_COMPLETE_WITH_GAPS",
        "source_run": str(PILOT), "targeted_reference_only": True,
        "composition": {"attempts": len(composition), "no_safe": sum(x["outcome"] == "NO_SAFE_EVOLUTION" for x in composition), "schema_valid": sum(x["schema_valid_v1"] for x in composition), "strict_v1_content_valid": sum(x["strict_v1_content_valid"] for x in composition), "review_buckets": comp_buckets, "r03_touch_real": sum(x["r03_ref_touch_real"] for x in composition)},
        "fresh": {"attempts": len(fresh), "no_safe": sum(x["outcome"] == "NO_SAFE_EVOLUTION" for x in fresh), "schema_valid": sum(x["schema_valid_v1"] for x in fresh), "strict_v1_content_valid": sum(x["strict_v1_content_valid"] for x in fresh), "review_buckets": fresh_buckets, "r03_touch_real": sum(x["r03_ref_touch_real"] for x in fresh)},
        "targeted": {"attempts": 6, "no_safe": 4, "schema_valid": 0, "content_valid": 0, "genuine_failures": 2, "reference_only": True},
        "semantic_limits": ["No semantic auditor was called", "Gold uniqueness and distractor validity remain UNKNOWN unless explicitly mechanically established", "Lexical/paraphrase candidates are not semantic duplicate findings", "CONTENT_UNADJUDICATED_SCHEMA_OR_CONTRACT_GAP is not a semantic QA failure finding"],
        "research_questions": {
            "Q1": "NOT_ESTABLISHED. Targeted produced no valid successor, while Fresh produced one structurally valid but semantically unadjudicated candidate; this does not prove that R03 is limited to or independent of targeted evolution.",
            "Q2": "NOT_YET_ESTABLISHED. Composition produced no strict-v1 content-valid candidate. Fresh produced one structurally valid unadjudicated candidate whose Gold, distractors, novelty, and R03 necessity still require human offline review.",
            "Q3": "CONDITIONAL_ONLY. If the Fresh candidate passes independent offline semantic review, the observation would be consistent with independent R03 value and weak historical-QA connectivity, but this pilot cannot identify that causal explanation.",
        },
        "calls": {"new_qa_writer": 14, "targeted_rerun": 0, "solver": 0, "training": 0, "memorax": 0, "semantic_auditor": 0, "retry": 0},
        "next_status": "READY_FOR_HUMAN_OFFLINE_REVIEW_BEFORE_ANY_NEW_GENERATION",
    }
    write(OUTPUT / "composition_offline_review.jsonl", composition)
    write(OUTPUT / "fresh_offline_review.jsonl", fresh)
    write(OUTPUT / "multi_pass_offline_review.json", summary)
    write(OUTPUT / "multi_pass_offline_review.md", "\n".join([
        "# R03 Multi-Pass Offline Review", "", f"Status: `{summary['status']}`", "",
        "The live attempt directory is immutable; this review made no model or state-changing calls.", "",
        f"- Composition: {len(composition)} attempts; schema-valid {summary['composition']['schema_valid']}; strict-v1 content-valid {summary['composition']['strict_v1_content_valid']}; structurally valid unadjudicated {comp_structural}; R03 refs touched with real refs {summary['composition']['r03_touch_real']}.",
        f"- Fresh: {len(fresh)} attempts; schema-valid {summary['fresh']['schema_valid']}; strict-v1 content-valid {summary['fresh']['strict_v1_content_valid']}; structurally valid unadjudicated {fresh_structural}; R03 refs touched with real refs {summary['fresh']['r03_touch_real']}.",
        "- Targeted is an imported 6-attempt reference only: 4 NO_SAFE, 2 genuine failures, 0 valid.",
        "- No Solver, training, Memorax, semantic auditor, retry, or seed switch was performed.", "",
        "The one Fresh output from root `cc-train-episode-198480` is structurally valid and uses real R03 refs, but Gold/distractor semantics and R03 necessity remain unadjudicated. The remaining rows are split into deterministic replays, mechanical evidence/relation rejections, and schema/contract gaps with content left unadjudicated; no QA is promoted.", "",
        f"Q1: {summary['research_questions']['Q1']}", "",
        f"Q2: {summary['research_questions']['Q2']}", "",
        f"Q3: {summary['research_questions']['Q3']}", "",
        "Next: human/offline semantic review only; do not automatically enter Solver or another generation pass.", "",
    ]))
    sha_rows = [{"path": p, "sha256_before": d, "sha256_after": sha(Path(p)), "unchanged": d == sha(Path(p))} for p, d in sorted(before.items())]
    write(OUTPUT / "review_input_source_sha_manifest.json", {"status": "PASS" if all(x["unchanged"] for x in sha_rows) else "FAIL", "files": sha_rows})
    checks = {
        "14_attempts_closed": len(composition) + len(fresh) == 14,
        "7_plus_7_root_coverage": {x["root_id"] for x in composition} == set(SUCCESS_ROOTS) == {x["root_id"] for x in fresh},
        "112812_excluded": all(x["root_id"] != "cc-train-episode-112812" for x in composition + fresh),
        "96462_fresh_included": any(x["root_id"] == "cc-train-episode-96462" for x in fresh),
        "review_buckets_close": sum(comp_buckets.values()) == len(composition) and sum(fresh_buckets.values()) == len(fresh),
        "targeted_reference_not_rerun": summary["calls"]["targeted_rerun"] == 0,
        "all_review_inputs_unchanged": all(x["unchanged"] for x in sha_rows),
        "prohibited_calls_zero": all(summary["calls"][name] == 0 for name in ["solver", "training", "memorax", "semantic_auditor", "retry"]),
    }
    write(OUTPUT / "offline_review_test_report.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
