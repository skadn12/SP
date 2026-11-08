#!/usr/bin/env python3
"""Read-only postflight, to be run only after the selector has terminated.

This does not establish complete contract compliance or filesystem isolation.
"""
import argparse
import json
import re
from pathlib import Path

from check import OLD, digest, resolve, strict_json, turns


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-terminated", action="store_true",
                        help="Controller confirms the one selector session has ended; not proof of its trace.")
    args = parser.parse_args()
    if not args.selector_terminated:
        parser.error("do not read the result until the selector session has terminated")
    receipt = strict_json(OLD / "continuation_opp13_selector_controller_v2/CONTROLLER_RECEIPT.json")
    packet_path = Path(receipt["selector_packet_path"])
    packet = strict_json(packet_path)
    errors = []
    if digest(packet_path) != receipt["selector_packet_sha256"]:
        errors.append("selector packet hash changed")
    for label, binding in packet["inputs"].items():
        if binding != receipt["copy_bindings"].get(label) or digest(binding["path"]) != binding["sha256"]:
            errors.append(f"input binding or hash changed: {label}")
    result_path = Path(packet["output_path"])
    if result_path.is_symlink() or not result_path.is_file():
        raise ValueError("assigned selector result is missing or is a symlink")
    result = strict_json(result_path)
    if not isinstance(result, dict):
        raise ValueError("selector result is not a JSON object")
    slots = strict_json(packet["inputs"]["slots"]["path"])["slots"]
    slot_ids = [s["slot_id"] for s in slots]
    status, selected = result.get("status"), result.get("selected_slot_id")
    if not isinstance(status, str) or not status.strip():
        errors.append("status missing or invalid")
    if selected is not None and (not isinstance(selected, str) or selected not in slot_ids):
        errors.append("selected slot is not one existing frozen slot")
    if status == "SELECTED" and selected is None:
        errors.append("SELECTED is missing a selected slot")
    if status == "NO_SAFE_QA" and selected is not None:
        errors.append("NO_SAFE_QA must not select a slot")
    source = strict_json(packet["inputs"]["source"]["path"])
    rows = turns(source)
    citation_count = 0
    citation_review = []
    for record in walk(result):
        if "turn_ref" not in record or "quote" not in record:
            continue
        citation_count += 1
        ref, quote = record["turn_ref"], record["quote"]
        try:
            candidates = resolve(ref, rows)
            hits = [turn for turn in candidates.values() if isinstance(quote, str) and quote and quote in turn["text"]]
            if not hits:
                errors.append("structured source quote does not match its reference")
            elif len(hits) > 1:
                citation_review.append("source quote matches multiple turns within its reference")
        except ValueError:
            citation_review.append("structured reference requires manual interpretation")
    result_text = result_path.read_text(encoding="utf-8")
    if re.search(r"cc-train-episode-\d+|OPP27_\d+|(?:OLD|NEW)_[FG]\b", result_text, re.I):
        errors.append("explicit source/opportunity/arm identity appears in result")
    allowed = {str(packet_path), str(result_path)} | {binding["path"] for binding in packet["inputs"].values()}
    claimed = []
    for record in walk(result):
        path = record.get("path")
        if isinstance(path, str) and path.startswith("/"):
            claimed.append(path)
            if path not in allowed:
                errors.append("result declares a path outside its allowlist")
            if path in allowed and "sha256" in record and digest(path) != record["sha256"]:
                errors.append("result declares a mismatching file hash")
    report = {
        "stage": "MECHANICAL_SELECTOR_RESULT_ONLY",
        "mechanical_pass": not errors,
        "errors": errors,
        "result_sha256": digest(result_path),
        "status": status,
        "selected_slot_id": selected,
        "structured_citations_checked": citation_count,
        "citation_manual_review": citation_review,
        "declared_absolute_paths_checked": len(claimed),
        "manual_and_trace_review_required": [
            "Every frozen slot has source feasibility, natural need, scope and historical comparison reasoning as required by the delivered contract and continuation rules.",
            "Historical UNKNOWN and empty comparisons are not described as confirmed uncovered or confirmed nonduplicate; frequency and tie decisions follow the delivered rule.",
            "All source quotations and turn references, including any prose or nonstandard structured citations, match the source and the intended session.",
            "The selector actually read all required inputs, supplied and matched their hashes, and recorded its actual visible reading ranges without out-of-bound access.",
            "The runtime trace establishes one attempt and exclusive-create result writing; file existence, output assertions and this script cannot establish O_EXCL.",
            "No F/G identity, other role content or private path was exposed in the actual input, runtime trace or selector output; regex checks are not exhaustive.",
            "The submitted decision follows the originally delivered contract and disclosed rules; do not infer an unannounced exact output schema or change a decision after the fact."
        ],
        "handoff_authorized_by_this_report": False,
        "resampling_or_retry": False
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
