"""Offline quality audit for the new generation-only B sample.

This report separates deterministic structural checks from heuristics.  It
does not claim semantic entailment from lexical overlap alone.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text
from ..h4_qa_alignment_audit_v1.audit import audit_candidate


STOP = set(
    "the a an and or of to in on for with from by is are was were be been being has have had do did does how what which "
    "following their they them he she it this that these those says said mentions mentioned talks talk discussed discusses "
    "asks asked suggests suggested encourages encouraged tells told gives gave expresses expressed feels felt about after "
    "before during as at her his its into over under through based sequence order occurred dialogue events statements "
    "made can could would should will"
    .split()
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9']+", str(value or "").casefold())
        if len(token) > 2 and token not in STOP
    }


def _candidate_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (run_root / "sample_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = _load_line(line)
        root = Path(str(sample["lineage"]["source_root_dir"]))
        pool = _load(root / "candidate_pool.json")
        matches = [row for row in pool if str(row.get("candidate_id")) == str(sample["candidate_id"])]
        if len(matches) != 1:
            raise ValueError(f"candidate_lineage_not_unique:{sample['candidate_id']}:{len(matches)}")
        candidate = dict(matches[0])
        dialogue = _load(root / "expanded_dialogue_clean.json")
        turns = {str(row.get("turn_ref")): str(row.get("text") or "") for row in dialogue.get("turns") or []}
        rows.append({"sample": sample, "candidate": candidate, "turns": turns, "root": root})
    return rows


def _load_line(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("sample_manifest_row_not_object")
    return value


def _structural(row: dict[str, Any]) -> dict[str, Any]:
    c = row["candidate"]
    qtype = str(c.get("question_type") or "")
    container = c.get("items") if qtype == "ordering" else c.get("options")
    keys = {"I1", "I2", "I3", "I4"} if qtype == "ordering" else {"A", "B", "C", "D", "E"} if qtype == "multi_select" else {"A", "B", "C", "D"}
    errors: list[str] = []
    if not c.get("schema_valid") or c.get("schema_errors"):
        errors.append("schema_invalid")
    if not isinstance(container, dict) or set(container) != keys:
        errors.append("option_item_keys_invalid")
    texts = [str(container.get(key) or "") for key in sorted(keys)] if isinstance(container, dict) else []
    if any(not text.strip() for text in texts):
        errors.append("empty_option_item")
    normalized = [" ".join(text.casefold().split()) for text in texts]
    if len(normalized) != len(set(normalized)):
        errors.append("duplicate_option_item")
    gold = [str(c.get("correct_option"))] if qtype == "single_choice" else [str(x) for x in c.get("gold") or []]
    if qtype == "single_choice" and gold[0] not in keys:
        errors.append("single_gold_invalid")
    if qtype == "multi_select" and (not gold or len(set(gold)) != len(gold) or any(x not in keys for x in gold)):
        errors.append("multi_gold_invalid")
    if qtype == "ordering" and set(gold) != keys:
        errors.append("ordering_gold_invalid")
    refs = [str(x) for x in c.get("evidence_refs") or []]
    missing_refs = [ref for ref in refs if ref not in row["turns"]]
    if not refs:
        errors.append("evidence_refs_missing")
    if missing_refs:
        errors.append("evidence_ref_unknown")
    return {"errors": sorted(set(errors)), "evidence_count": len(refs), "missing_evidence_refs": missing_refs}


def _lexical_gold_support(row: dict[str, Any]) -> list[float]:
    c = row["candidate"]
    evidence = _tokens(" ".join(row["turns"].get(ref, "") for ref in c.get("evidence_refs") or []))
    container = c.get("items") if c.get("question_type") == "ordering" else c.get("options") or {}
    gold = [str(c.get("correct_option"))] if c.get("question_type") == "single_choice" else [str(x) for x in c.get("gold") or []]
    return [len(_tokens(container.get(key, "")) & evidence) / max(1, len(_tokens(container.get(key, "")))) for key in gold]


def _load_b0(run_root: Path) -> dict[str, dict[str, list[int]]]:
    rows = _load(run_root / "b0_results.json").get("rows") or []
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["candidate_id"])][str(row["context_mode"])].append(int(bool(row.get("correct"))))
    return grouped


def _old_type_context(old_run: Path) -> dict[str, dict[str, dict[str, float | int]]]:
    paths = {
        "B0": (old_run, "label"),
        "B1": (old_run.parent / "h4_position_permutation_ab_probe_v1_b1_repair_live_1", "content_visible"),
        "B2": (old_run.parent / "h4_position_permutation_ab_probe_v1_b2_repair_live_1", "content_hidden"),
    }
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for stage, (path, mode) in paths.items():
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for result_path in (path / "results").glob("*.json"):
            row = _load(result_path)
            if row.get("output_mode") == mode and row.get("is_cyclic") is True:
                grouped[(str(row.get("question_type")), str(row.get("context_mode")))].append(int(bool(row.get("correct"))))
        out[stage] = {
            f"{qtype}/{context}": {"count": len(values), "accuracy": sum(values) / len(values) if values else None}
            for (qtype, context), values in sorted(grouped.items())
        }
    return out


def audit(run_root: Path, old_run: Path | None = None) -> dict[str, Any]:
    rows = _candidate_rows(run_root.resolve())
    b0 = _load_b0(run_root.resolve())
    structural = [_structural(row) for row in rows]
    source_alignment: dict[str, Counter[str]] = defaultdict(Counter)
    contract_rows: list[dict[str, Any]] = []
    for row in rows:
        sample, candidate, turns = row["sample"], row["candidate"], row["turns"]
        contracts = _load(row["root"] / "reasoning_contracts.json")
        contract = next((item for item in contracts if str(item.get("intent_id")) == str(candidate.get("intent_id"))), None)
        result = audit_candidate(candidate, contract=contract, turn_text=turns, difficulty=None, static=None)
        if sample.get("generation_source") == "contract_guided":
            source_alignment["contract_guided"][str(result["alignment_bucket"])] += 1
            contract_rows.append({"candidate_id": sample["candidate_id"], "alignment_bucket": result["alignment_bucket"], "issues": result["issues"]})

    easy_ids: list[str] = []
    b0_no_context_high_ids: list[str] = []
    for row in rows:
        cid = str(row["candidate"]["candidate_id"])
        values = b0.get(cid) or {}
        means = {mode: sum(values.get(mode, [])) / len(values.get(mode, [])) for mode in ("no_context", "full_context") if values.get(mode)}
        if means and min(means.values()) >= 0.75:
            easy_ids.append(cid)
        if means.get("no_context", 0.0) >= 0.75:
            b0_no_context_high_ids.append(cid)

    missing_metadata = [str(row["candidate"]["candidate_id"]) for row in rows if not str(row["candidate"].get("distractor_strategy") or "").strip()]
    narrow = [str(row["candidate"]["candidate_id"]) for row, check in zip(rows, structural) if check["evidence_count"] < 2]
    lexical_zero_single_non_not = []
    for row in rows:
        c = row["candidate"]
        # Structured answers are usually paraphrases of several turns, so a
        # lexical zero is too noisy to flag there.  Restrict this triage check
        # to single-choice answers and exclude explicit NOT questions.
        if c.get("question_type") != "single_choice" or re.search(r"\bnot\b", str(c.get("question") or "").casefold()):
            continue
        supports = _lexical_gold_support(row)
        if supports and min(supports) == 0.0:
            lexical_zero_single_non_not.append(str(c["candidate_id"]))

    report: dict[str, Any] = {
        "audit_version": "h4_position_ab_new_generation_data_audit_v1",
        "run_root": str(run_root.resolve()),
        "offline_only": True,
        "candidate_count": len(rows),
        "structural": {
            "error_count": sum(bool(check["errors"]) for check in structural),
            "errors_by_type": dict(Counter(error for check in structural for error in check["errors"])),
            "missing_evidence_ref_count": sum(len(check["missing_evidence_refs"]) for check in structural),
            "evidence_count_distribution": dict(Counter(check["evidence_count"] for check in structural)),
        },
        "contract_guided_alignment": dict(source_alignment.get("contract_guided", {})),
        "quality_flags": {
            "narrow_evidence_scope_ids": narrow,
            "missing_distractor_strategy_ids": missing_metadata,
            "lexical_zero_single_choice_support_non_not_ids": lexical_zero_single_non_not,
            "manual_review_note": "lexical support is only a triage heuristic; it is not semantic entailment",
        },
        "difficulty_signal_from_completed_B0": {
            "candidate_ids_correct_in_both_contexts_at_least_75pct": easy_ids,
            "count": len(easy_ids),
            "no_context_at_least_75pct_count": len(b0_no_context_high_ids),
        },
        "conclusion": "Structurally valid does not imply semantically admitted. The new 60-item run is usable as a generation-only diagnostic, but should not be treated as an apples-to-apples replacement for the old hard-screened 30-item run without semantic/difficulty review.",
    }
    if old_run is not None:
        report["old_30_type_context_reference"] = _old_type_context(old_run.resolve())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--old-run", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.run_root, args.old_run)
    atomic_json(args.run_root.resolve() / "data_quality_audit.json", report)
    atomic_text(args.run_root.resolve() / "data_quality_audit.md", "# H4 new generation-only data quality audit\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": "written",
        "report": str(args.run_root.resolve() / "data_quality_audit.json"),
        "structural_error_count": report["structural"]["error_count"],
        "contract_guided_alignment": report["contract_guided_alignment"],
        "narrow_evidence_scope_count": len(report["quality_flags"]["narrow_evidence_scope_ids"]),
        "lexical_zero_single_choice_support_non_not_ids": report["quality_flags"]["lexical_zero_single_choice_support_non_not_ids"],
        "b0_easy_candidate_count": report["difficulty_signal_from_completed_B0"]["count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
