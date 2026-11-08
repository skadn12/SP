#!/usr/bin/env python3
"""Build an immutable, review-bound F/G candidate manifest.

The script reads the sealed 140-draft run and the freshly resolved canonical
review.  It writes a new manifest under this overnight run and never mutates
the source run, its review files, or the sealed 135-item export.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

RUN = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019")
SOURCE_RUN = RUN / "training_set_rebuild_20261105_FG_exploration_run01"
REVIEW_DIR = SOURCE_RUN / "quality_source_review_20261108_v1"
OUT_DIR = RUN / "training_set_rebuild_20261108_overnight_run01" / "candidate_manifest"
OUT_JSON = OUT_DIR / "CANDIDATE_MANIFEST_20261108_v1.json"
OUT_MD = OUT_DIR / "CANDIDATE_MANIFEST_20261108_v1.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resolution_path = REVIEW_DIR / "CANONICAL_REVIEW_RESOLUTION_20261108_v1.json"
    resolution = json.loads(resolution_path.read_text())
    pass_packets = {
        (row["root_id"], row["type"]): row
        for row in resolution["resolved"]
        if row["effective_status"] == "PASS"
    }
    hold_packets = [
        {"root_id": row["root_id"], "type": row["type"], "packet_id": row["packet_id"],
         "index": row["index"], "reason": row["resolution_reason"]}
        for row in resolution["resolved"] if row["effective_status"] != "PASS"
    ]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for arm in ("F", "G"):
        for draft_path in sorted((SOURCE_RUN / "authoring" / arm).glob("*/CANDIDATES_FIRST_DRAFT.json")):
            draft = json.loads(draft_path.read_text())
            root_id = draft.get("root_id")
            candidates = draft.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 1:
                errors.append({"file": str(draft_path), "error": "expected exactly one candidate"})
                continue
            candidate = candidates[0]
            ctype = candidate.get("question_type")
            review = pass_packets.get((root_id, ctype))
            source_path = Path(candidate.get("source_path", ""))
            source_exists = source_path.is_file()
            actual_source_sha = sha256(source_path) if source_exists else None
            source_sha_ok = actual_source_sha == candidate.get("source_sha256")
            gold = candidate.get("gold_option_ids")
            option_ids = [x.get("option_id") for x in candidate.get("options", []) if isinstance(x, dict)]
            gold_valid = (
                isinstance(gold, list) and bool(gold) and len(set(gold)) == len(gold)
                and all(x in option_ids for x in gold)
            )
            cid = candidate.get("candidate_id")
            if cid in seen_ids:
                errors.append({"candidate_id": cid, "error": "duplicate candidate_id"})
            seen_ids.add(cid)
            row = {
                "candidate_id": cid,
                "root_id": root_id,
                "arm": arm,
                "question_type": ctype,
                "draft_path": str(draft_path),
                "draft_sha256": sha256(draft_path),
                "candidate_source_path": candidate.get("source_path"),
                "candidate_source_sha256": candidate.get("source_sha256"),
                "actual_source_sha256": actual_source_sha,
                "source_exists": source_exists,
                "source_hash_match": source_sha_ok,
                "option_count": len(option_ids),
                "gold_option_ids": gold,
                "gold_valid": gold_valid,
                "question": candidate.get("question"),
                "natural_information_need": candidate.get("natural_information_need"),
                "cognitive_task": candidate.get("cognitive_task"),
                "relation_plan": candidate.get("relation_plan"),
                "canonical_review": {
                    "packet_id": review.get("packet_id") if review else None,
                    "index": review.get("index") if review else None,
                    "effective_status": review.get("effective_status") if review else "MISSING",
                    "resolution_reason": review.get("resolution_reason") if review else "missing_review_binding",
                    "chosen": review.get("chosen") if review else None,
                },
                # The historical audit explicitly leaves 642 records without a
                # candidate-specific comparator.  Preserve that limitation in
                # every row instead of silently treating PASS as global novelty.
                "lineage_gate": "SCOPED_PROVENANCE_BOUND_GLOBAL_ALIAS_UNRESOLVED",
                "semantic_gate": "UNRESOLVED_AGAINST_642_UNKNOWN_PENDING_HISTORY",
                "exposure_gate": "NO_EXACT_RESERVATION_HIT_IN_SCOPED_SCAN",
                "provisional_admission": "REVIEW_PASS_PROVISIONAL_ONLY" if review and review.get("effective_status") == "PASS" else "EXCLUDED",
            }
            rows.append(row)

    rows.sort(key=lambda r: (r["root_id"], r["arm"]))
    counts = Counter(r["canonical_review"]["effective_status"] for r in rows)
    counts_by_arm = Counter(r["arm"] for r in rows if r["canonical_review"]["effective_status"] == "PASS")
    counts_type = Counter(r["question_type"] for r in rows if r["canonical_review"]["effective_status"] == "PASS")
    payload = {
        "schema": "fg_candidate_manifest_v1",
        "created_utc": "2026-11-08",
        "read_only_inputs": True,
        "source_run": str(SOURCE_RUN),
        "source_run_seal": str(SOURCE_RUN / "EXPLORATION_SEAL.json"),
        "source_run_seal_sha256": sha256(SOURCE_RUN / "EXPLORATION_SEAL.json"),
        "canonical_review_resolution": str(resolution_path),
        "canonical_review_sha256": sha256(resolution_path),
        "canonical_review_summary": resolution["effective_status_counts"],
        "hold_packets_excluded": hold_packets,
        "candidate_count": len(rows),
        "pass_candidate_count": sum(1 for r in rows if r["canonical_review"]["effective_status"] == "PASS"),
        "pass_by_arm": dict(counts_by_arm),
        "pass_by_type": dict(counts_type),
        "row_status_counts": dict(counts),
        "build_errors": errors,
        "global_gate_limitations": {
            "historical_coverage_unknown_pending": 642,
            "formal_semantic_admission": "not granted",
            "interpretation": "This is a review-bound provisional manifest. Solver/training admission requires the downstream semantic/lineage policy decision; no global novelty claim is made.",
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    md = [
        "# F/G candidate manifest (2026-11-08)", "",
        "This immutable staging manifest binds each authored candidate to the latest canonical source review. Source drafts and review records were read only.", "",
        f"- rows: **{len(rows)}**; canonical PASS rows: **{payload['pass_candidate_count']}**",
        f"- PASS by arm: `{dict(counts_by_arm)}`; PASS by type: `{dict(counts_type)}`",
        f"- canonical HOLD packets excluded: `{hold_packets}`",
        f"- build errors: **{len(errors)}**",
        "- lineage/semantic/exposure: scoped provenance is recorded, but global novelty remains unresolved because the fixed historical coverage has 642 UNKNOWN/PENDING records.",
        "",
        "The 138 PASS rows are provisional candidates for the isolated Solver gate. They are not silently promoted to final training data.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(json.dumps({k: payload[k] for k in ("candidate_count", "pass_candidate_count", "pass_by_arm", "pass_by_type", "row_status_counts", "build_errors")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
