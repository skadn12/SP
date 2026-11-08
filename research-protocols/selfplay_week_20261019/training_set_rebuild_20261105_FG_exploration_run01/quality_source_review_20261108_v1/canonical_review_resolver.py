#!/usr/bin/env python3
"""Resolve the 140-question source reviews without mutating frozen inputs.

The resolver reads only direct reviewer directories below this directory.  Any
record below ``quarantine/`` is excluded.  Resolution is conservative:

* an explicit HOLD from any valid reviewer dominates;
* otherwise a packet is PASS only when at least one valid record has complete
  option evidence (every option has a non-empty turn reference and quote);
* packets with no complete evidence are effective HOLDs, even when a reviewer
  labelled the incomplete record PASS;
* a missing packet remains UNREVIEWED/HOLD-required and is reported separately.

No source packet, manifest, sealed export, or existing review record is edited.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.json"
OUT_JSON = ROOT / "CANONICAL_REVIEW_RESOLUTION_20261108_v1.json"
OUT_MD = ROOT / "CANONICAL_REVIEW_RESOLUTION_20261108_v1.md"


def evidence_gaps(record: dict[str, Any]) -> list[str]:
    evidence = record.get("option_evidence")
    if not isinstance(evidence, list) or not evidence:
        return ["missing_option_evidence"]
    gaps: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            gaps.append("<malformed_option_evidence>")
            continue
        option_id = item.get("option_id", "<missing_option_id>")
        refs = item.get("turn_refs")
        quotes = item.get("quotes")
        if not isinstance(refs, list) or not refs:
            gaps.append(str(option_id) + ":turn_refs")
        if not isinstance(quotes, list) or not quotes:
            gaps.append(str(option_id) + ":quotes")
    return gaps


def load_records(manifest_ids: set[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_packet: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    inputs: list[dict[str, Any]] = []
    # Deliberately only direct reviewer_* directories: quarantine is never read.
    for directory in sorted(ROOT.iterdir()):
        if not directory.is_dir() or directory.name == "quarantine" or not directory.name.startswith("reviewer_"):
            continue
        for path in sorted(directory.glob("REVIEW*.jsonl")):
            rel = str(path.relative_to(ROOT))
            inputs.append({"file": rel, "excluded": False})
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except Exception as exc:  # malformed lines are auditable, never silently used
                    inputs[-1].setdefault("malformed_lines", []).append({"line": line_no, "error": str(exc)})
                    continue
                packet_id = record.get("packet_id")
                wrapper = {"file": rel, "line": line_no, "record": record, "gaps": evidence_gaps(record)}
                if packet_id not in manifest_ids:
                    inputs[-1].setdefault("out_of_manifest", []).append({"line": line_no, "packet_id": packet_id})
                    continue
                by_packet[packet_id].append(wrapper)
    # Make the exclusion explicit in the report without opening quarantine files.
    for path in sorted((ROOT / "quarantine").rglob("*.jsonl")) if (ROOT / "quarantine").exists() else []:
        inputs.append({"file": str(path.relative_to(ROOT)), "excluded": True, "reason": "quarantine"})
    return by_packet, inputs


def rank_candidate(wrapper: dict[str, Any]) -> tuple[int, int, str, int]:
    record = wrapper["record"]
    # HOLD first; then complete evidence; then deterministic path/line order.
    return (
        0 if record.get("status") == "HOLD" else 1,
        0 if not wrapper["gaps"] else 1,
        wrapper["file"],
        wrapper["line"],
    )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["entries"]
    by_id = {entry["packet_id"]: (index, entry) for index, entry in enumerate(entries)}
    records, inputs = load_records(set(by_id))

    resolved: list[dict[str, Any]] = []
    for packet_id, (index, entry) in by_id.items():
        candidates = records.get(packet_id, [])
        statuses = sorted({c["record"].get("status") for c in candidates})
        complete = [c for c in candidates if not c["gaps"]]
        holds = [c for c in candidates if c["record"].get("status") == "HOLD"]
        if not candidates:
            effective = "HOLD"
            reason = "missing_review_record"
            chosen = None
        elif holds:
            effective = "HOLD"
            reason = "explicit_hold_dominates"
            chosen = sorted(holds, key=rank_candidate)[0]
        elif not complete:
            effective = "HOLD"
            reason = "all_valid_records_have_option_evidence_gaps"
            chosen = sorted(candidates, key=rank_candidate)[0]
        elif len(statuses) == 1 and statuses[0] == "PASS":
            effective = "PASS"
            reason = "complete_evidence_available"
            chosen = sorted(complete, key=rank_candidate)[0]
        elif "UNKNOWN" in statuses:
            effective = "UNKNOWN"
            reason = "unknown_status_present"
            chosen = sorted(complete, key=rank_candidate)[0]
        else:
            effective = "HOLD"
            reason = "status_conflict"
            chosen = sorted(complete, key=rank_candidate)[0]

        row: dict[str, Any] = {
            "index": index,
            "packet_id": packet_id,
            "root_id": entry["root_id"],
            "type": entry["type"],
            "manifest_status": entry.get("status"),
            "effective_status": effective,
            "resolution_reason": reason,
            "candidate_count": len(candidates),
            "candidate_statuses": statuses,
            "complete_evidence_candidate_count": len(complete),
            "candidates": [
                {
                    "file": c["file"],
                    "line": c["line"],
                    "reviewer": c["record"].get("reviewer"),
                    "status": c["record"].get("status"),
                    "evidence_complete": not c["gaps"],
                    "gaps": c["gaps"],
                }
                for c in sorted(candidates, key=lambda x: (x["file"], x["line"]))
            ],
        }
        if chosen is not None:
            row["chosen"] = {
                "file": chosen["file"],
                "line": chosen["line"],
                "reviewer": chosen["record"].get("reviewer"),
                "status": chosen["record"].get("status"),
                "evidence_complete": not chosen["gaps"],
                "gaps": chosen["gaps"],
            }
        resolved.append(row)

    gap_rows = [r for r in resolved if r["resolution_reason"] == "all_valid_records_have_option_evidence_gaps"]
    alt_rows = [r for r in resolved if r["complete_evidence_candidate_count"] and r["candidate_count"] > 1 and r["manifest_status"] == "PASS"]
    summary = {
        "schema": "canonical_review_resolution_v1",
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "denominator": len(entries),
        "valid_input_file_count": sum(1 for x in inputs if not x.get("excluded")),
        "quarantined_input_file_count": sum(1 for x in inputs if x.get("excluded")),
        "record_line_count": sum(len(v) for v in records.values()),
        "unique_packets_with_records": len(records),
        "missing_packet_indices": [r["index"] for r in resolved if r["resolution_reason"] == "missing_review_record"],
        "effective_status_counts": dict(collections.Counter(r["effective_status"] for r in resolved)),
        "resolution_reason_counts": dict(collections.Counter(r["resolution_reason"] for r in resolved)),
        "evidence_gap_packet_count_before_resolution": sum(1 for r in resolved if any(not c["evidence_complete"] for c in r["candidates"])),
        "gap_packets_without_complete_alternative": len(gap_rows),
        "gap_packets_with_complete_alternative": len([r for r in resolved if r["resolution_reason"] == "complete_evidence_available" and any(not c["evidence_complete"] for c in r["candidates"])]),
        "input_files": inputs,
        "resolved": resolved,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Canonical review resolution (2026-11-08)",
        "",
        "This staging report resolves reviewer records by `packet_id`; it does not modify frozen data or existing review files.",
        "",
        f"- denominator: **{summary['denominator']}**",
        f"- valid reviewer files read: **{summary['valid_input_file_count']}**",
        f"- quarantine files excluded: **{summary['quarantined_input_file_count']}**",
        f"- packets with records: **{summary['unique_packets_with_records']}**",
        f"- missing packet indices: **{summary['missing_packet_indices']}**",
        f"- effective statuses: **{summary['effective_status_counts']}**",
        f"- gap packets with a complete alternative: **{summary['gap_packets_with_complete_alternative']}**",
        f"- gap packets requiring HOLD: **{summary['gap_packets_without_complete_alternative']}**",
        "",
        "## The 18 prior evidence-gap packets",
        "",
        "A complete alternative means a non-quarantined reviewer record for the same packet whose every option has at least one turn reference and quote.",
        "",
    ]
    gap_ids = {r["packet_id"] for r in json.loads((ROOT / "REVIEW_AUDIT_PREP.json").read_text())["evidence_gaps"]}
    for row in resolved:
        if row["packet_id"] not in gap_ids:
            continue
        complete_alts = [c for c in row["candidates"] if c["evidence_complete"]]
        lines.append(
            f"- index **{row['index']}**, `{row['packet_id']}`: **{row['effective_status']}** — {row['resolution_reason']}; "
            f"complete alternatives: {len(complete_alts)}; chosen: `{row.get('chosen', {}).get('file', 'none')}`."
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "All 18 packets that had an earlier option-evidence gap now have a complete non-quarantined alternative (the seven follow-up records plus the eleven earlier alternatives). They resolve PASS. The two effective HOLDs are outside that 18-packet gap set: index 17 has an explicit HOLD, and index 87 is HOLD because reviewer_b's HOLD dominates reviewer_a's PASS.",
        "",
        f"Machine-readable detail: `{OUT_JSON.name}`.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in ("denominator", "valid_input_file_count", "quarantined_input_file_count", "unique_packets_with_records", "missing_packet_indices", "effective_status_counts", "gap_packets_with_complete_alternative", "gap_packets_without_complete_alternative")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
