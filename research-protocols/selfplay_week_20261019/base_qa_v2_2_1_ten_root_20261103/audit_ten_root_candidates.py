"""Audit the ten-root candidate preparation without upgrading historical PASS.

This is deliberately a read-only audit of the copied candidate artifacts.  It
does not create source approval, run a model, call an API, or run Solver.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOTS = HERE / "roots"
MANIFEST = HERE / "TEN_ROOT_MANIFEST.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scene_of(ref: str) -> str:
    # Current extension turn IDs use R03:A:S01:T001.  Keep unknown formats
    # explicit rather than inventing scene identities.
    m = re.match(r"^(R[^:]+:[^:]+:[^:]+):T", ref or "")
    return m.group(1) if m else "UNKNOWN_SCENE"


def audit_one(root_dir: Path, manifest_row: dict) -> dict:
    source_path = root_dir / "FULL_SOURCE.json"
    candidate_path = root_dir / "HISTORICAL_CANDIDATE.json"
    scope_path = root_dir / "QUESTION_SCOPE_AUDIT.json"
    source = load(source_path)
    candidate = load(candidate_path)
    scope = load(scope_path)
    by_ref = {row["turn_ref"]: row for row in source["turns"]}
    quote_errors = []
    refs = []
    for option_id, row in candidate.get("item_analysis", {}).items():
        for ev in row.get("evidence", []):
            ref = ev.get("turn_ref")
            refs.append(ref)
            turn = by_ref.get(ref)
            if not turn:
                quote_errors.append(f"{option_id}:UNKNOWN_TURN:{ref}")
                continue
            if ev.get("speaker") != turn.get("speaker"):
                quote_errors.append(f"{option_id}:SPEAKER:{ref}")
            if not isinstance(ev.get("quote"), str) or ev["quote"] not in turn.get("text", ""):
                quote_errors.append(f"{option_id}:QUOTE:{ref}")
    unique_refs = sorted(set(refs))
    scenes = sorted({scene_of(ref) for ref in unique_refs})
    origins = Counter(by_ref[ref].get("origin", "UNKNOWN") for ref in unique_refs if ref in by_ref)
    gold = candidate.get("gold", [])
    options = candidate.get("options", {})
    gold_positions = Counter(gold)
    gold_lengths = [len(str(options[x]).split()) for x in gold if x in options]
    non_gold_lengths = [len(str(v).split()) for k, v in options.items() if k not in gold]
    source_binding = candidate.get("source_binding", {})
    hash_ok = source_binding.get("sha256") == sha256(source_path)
    scope_binding = scope.get("source_binding", {})
    scope_hash_ok = scope_binding.get("sha256") == sha256(source_path)
    current_pending = (
        scope.get("current_v221_formal_admission") == "NOT_REPLAYED"
        and scope.get("current_formal_qa_allowed") is False
    )
    # This package contains legacy candidate records, not typed V2.2.1 graph,
    # target freeze, construction audit, or a new blind-review receipt.
    reasons = [
        "CURRENT_V221_ADMISSION_NOT_REPLAYED",
        "CITED_TURNS_ARE_CODEX_SYNTHETIC_EXTENSION",
        "NO_CURRENT_INDEPENDENT_BLIND_REVIEW",
        "HISTORICAL_CANDIDATE_REUSE_NOT_FRESH_V221_GENERATION",
        "NO_TYPED_RELATION_GRAPH_OR_TARGET_FREEZE_IN_THIS_PACKAGE",
    ]
    if len(scenes) <= 1:
        reasons.append("EVIDENCE_SINGLE_SCENE_OR_UNRESOLVED_SCENE_IDENTITY")
    return {
        "root_id": candidate.get("root_id"),
        "candidate_id": candidate.get("candidate_id"),
        "source_file": str(source_path),
        "source_sha256": sha256(source_path),
        "source_binding_sha256": source_binding.get("sha256"),
        "source_hash_match": hash_ok,
        "scope_binding_hash_match": scope_hash_ok,
        "source_turn_count": len(source.get("turns", [])),
        "evidence_reference_count": len(refs),
        "evidence_unique_turn_count": len(unique_refs),
        "evidence_scene_ids": scenes,
        "evidence_origins": dict(origins),
        "quote_and_speaker_errors": quote_errors,
        "quote_and_speaker_check": "PASS" if not quote_errors else "FAIL",
        "historical_review_decision": candidate.get("historical_review_decision"),
        "current_v221_formal_admission": scope.get("current_v221_formal_admission"),
        "current_formal_qa_allowed": scope.get("current_formal_qa_allowed"),
        "current_status": "RESEARCH_CANDIDATE_HOLD",
        "hold_reasons": reasons,
        "question_type": candidate.get("question_type"),
        "task_operation": candidate.get("task_operation"),
        "option_count": len(options),
        "gold": gold,
        "gold_position_counts": dict(gold_positions),
        "gold_option_word_counts": gold_lengths,
        "non_gold_option_word_counts": non_gold_lengths,
        "gold_mean_words": round(sum(gold_lengths) / len(gold_lengths), 2) if gold_lengths else None,
        "non_gold_mean_words": round(sum(non_gold_lengths) / len(non_gold_lengths), 2) if non_gold_lengths else None,
        "manifest_status": manifest_row.get("status"),
    }


def main() -> None:
    manifest = load(MANIFEST)
    rows = []
    for row in manifest["candidates"]:
        result = audit_one(HERE / row["root_output"], row)
        rows.append(result)
    rows.sort(key=lambda x: x["root_id"])
    summary = {
        "schema_version": "base_qa_v2_2_1_ten_root_current_audit_v1",
        "experiment": "base_qa_v2_2_1_ten_root_20261103",
        "read_only": True,
        "candidate_count": len(rows),
        "formal_pass_count": sum(r["current_formal_qa_allowed"] is True for r in rows),
        "quote_check_pass_count": sum(r["quote_and_speaker_check"] == "PASS" for r in rows),
        "source_hash_match_count": sum(r["source_hash_match"] for r in rows),
        "all_current_admission_replayed": all(r["current_v221_formal_admission"] != "NOT_REPLAYED" for r in rows),
        "question_type_counts": dict(Counter(r["question_type"] for r in rows)),
        "task_operation_counts": dict(Counter(r["task_operation"] for r in rows)),
        "gold_position_counts": dict(Counter(
            ",".join(r["gold"]) for r in rows
        )),
        "all_referenced_origins": dict(Counter(origin for r in rows for origin in r["evidence_origins"] for _ in range(r["evidence_origins"][origin]))),
        "rows": rows,
        "interpretation": (
            "The ten records are historical candidate reuse. Exact quote/speaker and "
            "hash checks pass, but current V2.2.1 admission, typed discovery/target "
            "validation, and independent blind review were not replayed. No row is "
            "eligible for formal QA or Solver."
        ),
    }
    out = HERE / "CURRENT_TEN_ROOT_AUDIT.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = HERE / "TEN_ROOT_EXPANSION_REPORT.md"
    lines = [
        "# V2.2.1 ten-root expansion audit",
        "",
        "This report audits the existing ten candidate records without upgrading historical review to current formal PASS.",
        "",
        "## Result",
        "",
        f"- Candidate records: **{len(rows)}**.",
        f"- Exact source-hash matches: **{summary['source_hash_match_count']}/{len(rows)}**.",
        f"- Exact quote and speaker checks: **{summary['quote_check_pass_count']}/{len(rows)}**.",
        f"- Current formal V2.2.1 PASS: **{summary['formal_pass_count']}/{len(rows)}**.",
        "- Current admission replay: **0/10**; every record remains `RESEARCH_CANDIDATE_HOLD`.",
        "",
        "## What was actually produced",
        "",
        "The directory contains one historical candidate per selected root. They are all six-option A–F `multi_select` questions with `MC_CONSTRAINT_CONJUNCTION`; no ordering candidate was produced. The records are useful for audit and comparison, but they are not a fresh V2.2.1 discovery/construction run.",
        "",
        "They also do not satisfy the current V2.2.1 candidate contract: they have legacy `gold`/label fields but no typed semantic answer, discovery/target hashes, dependency graph, full-source counterpath audit, or target freeze. Renaming fields would not constitute a valid conversion; each question needs a fresh typed discovery and construction pass.",
        "",
        "All cited turns have exact text and speaker matches and all cited origins are `codex_synthetic_extension`. That proves local provenance integrity only. It does not establish permission to use synthetic material as formal evidence.",
        "",
        "## Why formal use is held",
        "",
        "1. The per-question records explicitly say `current_v221_formal_admission=NOT_REPLAYED` and `current_formal_qa_allowed=false`.",
        "2. The package has no current typed relation graph, target freeze, construction audit, or current independent blind-review receipt.",
        "3. Historical PASS belongs to the older review run and cannot be silently promoted.",
        "4. The ten questions are structurally repetitive. Gold positions are concentrated, and Gold options are longer on average than distractors; this requires a new shuffled-label blind review before any Solver use.",
        "5. The cited evidence is synthetic extension material. Any formal admission requires an explicit current authorization/review record; this audit does not create one.",
        "",
        "Later source-review history is mixed: 109164, 136202, 179186 and 246750 have later PASS records; 156136 reached PASS after revision; 194901 is FAIL; and 238359, 24719, 25652 and 45849 remain REVISE. These dispositions are recorded in `SOURCE_AUDIT_SLICE.md` and do not replace current item-level admission.",
        "",
        "## Per-root records",
        "",
        "| root | candidate | type | evidence turns | origins | quote check | current status |",
        "|---|---|---|---:|---|---|---|",
    ]
    for r in rows:
        origins = ", ".join(f"{k}:{v}" for k, v in r["evidence_origins"].items())
        lines.append(f"| {r['root_id']} | {r['candidate_id']} | {r['question_type']} | {r['evidence_unique_turn_count']} | {origins} | {r['quote_and_speaker_check']} | {r['current_status']} |")
    lines += [
        "",
        "Complete question text, options, Gold, and evidence remain in each root's `HISTORICAL_CANDIDATE.json`. The machine-readable details are in `CURRENT_TEN_ROOT_AUDIT.json`.",
        "",
        "## Next admissible step",
        "",
        "For any root that receives explicit current authorization, rerun discovery from its frozen full source, record every rejected relation, freeze a typed target before construction, shuffle stable option IDs, and obtain a new isolated same-model blind review. Only then can a question be considered for Solver evaluation.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(out), "report": str(report), "candidate_count": len(rows), "formal_pass_count": summary["formal_pass_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
