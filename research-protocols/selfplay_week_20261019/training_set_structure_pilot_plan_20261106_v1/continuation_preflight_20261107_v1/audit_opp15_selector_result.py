#!/usr/bin/env python3
"""Read-only OPP15 postflight. Run only after the selector session terminates.

Mechanical checks are not a complete contract, session, or isolation audit.
"""
import argparse
import json
import re
from pathlib import Path

from audit_opp13_selector_result import walk
from check import OLD, digest, resolve, strict_json, turns

RECEIPT = OLD / "continuation_opp15_selector_controller_v1/CONTROLLER_RECEIPT.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-terminated", action="store_true",
                        help="Controller confirms termination; this flag does not prove it.")
    args = parser.parse_args()
    if not args.selector_terminated:
        parser.error("do not read the result before selector termination")

    receipt = strict_json(RECEIPT)
    packet_path = Path(receipt["selector_packet_path"])
    packet = strict_json(packet_path)
    errors = []
    review = []
    if digest(packet_path) != receipt["selector_packet_sha256"]:
        errors.append("selector packet hash changed")
    for label, binding in packet["inputs"].items():
        if binding != receipt["copy_bindings"].get(label) or digest(binding["path"]) != binding["sha256"]:
            errors.append(f"input binding or hash changed: {label}")
    for prefix in ("original_slots", "slots_freeze"):
        if digest(receipt[prefix + "_path"]) != receipt[prefix + "_sha256"]:
            errors.append(f"{prefix} hash changed")

    original = strict_json(receipt["original_slots_path"])
    expected_projection = {k: v for k, v in original.items()
                           if k not in ("opportunity_id", "root_id", "source_sha256")}
    projected = strict_json(packet["inputs"]["slots"]["path"])
    if projected != expected_projection:
        errors.append("frozen slot projection changed")
    slots = projected["slots"]
    slot_ids = [slot.get("slot_id") for slot in slots]
    if any(not isinstance(s, str) or not s for s in slot_ids) or len(set(slot_ids)) != len(slot_ids):
        errors.append("invalid frozen slot IDs; quarantine without repair or retry")

    result_path = Path(packet["output_path"])
    if result_path.is_symlink() or not result_path.is_file():
        raise ValueError("assigned selector result is missing or is a symlink")
    result = strict_json(result_path)
    if not isinstance(result, dict):
        raise ValueError("selector result is not a JSON object")
    status, selected = result.get("status"), result.get("selected_slot_id")
    if not isinstance(status, str) or not status.strip():
        errors.append("status missing or invalid")
    if selected is not None and (not isinstance(selected, str) or selected not in slot_ids):
        errors.append("selected slot is not one existing frozen slot")
    if status == "SELECTED" and selected is None:
        errors.append("SELECTED is missing a selected slot")
    if status == "NO_SAFE_QA" and selected is not None:
        errors.append("NO_SAFE_QA must not select a slot")

    rows = turns(strict_json(packet["inputs"]["source"]["path"]))
    checked = 0
    for record in walk(result):
        if "turn_ref" not in record or "quote" not in record:
            continue
        checked += 1
        try:
            candidates = resolve(record["turn_ref"], rows)
            quote = record["quote"]
            hits = [turn for turn in candidates.values() if isinstance(quote, str) and quote and quote in turn["text"]]
            if not hits:
                errors.append("structured source quote does not match its reference")
            elif len(hits) > 1:
                review.append("source quote matches multiple turns within its reference")
        except ValueError:
            review.append("structured reference requires manual interpretation")
    if re.search(r"cc-train-episode-\d+|OPP27_\d+|(?:OLD|NEW)_[FG]\b", result_path.read_text(encoding="utf-8"), re.I):
        errors.append("explicit source/opportunity/arm identity appears in result")
    allowed = {str(packet_path), str(result_path)} | {binding["path"] for binding in packet["inputs"].values()}
    for record in walk(result):
        path = record.get("path")
        if isinstance(path, str) and path.startswith("/"):
            if path not in allowed:
                errors.append("result declares a path outside its allowlist")
            elif "sha256" in record and digest(path) != record["sha256"]:
                errors.append("result declares a mismatching file hash")

    report = {
        "stage": "MECHANICAL_SELECTOR_RESULT_ONLY",
        "mechanical_pass": not errors,
        "errors": errors,
        "result_sha256": digest(result_path),
        "status": status,
        "selected_slot_id": selected,
        "structured_citations_checked": checked,
        "citation_manual_review": review,
        "manual_and_trace_review_required": [
            "Confirm session terminated, one attempt, delivered role instruction and exact input access, including explicit permission for the continuation rules.",
            "Confirm complete visible source reads in chunks at most 30 turns and 12000 characters, neutral separately, all slots and every supplied historical record in nontruncated chunks at most 12000 characters; verify actual read ranges and hashes against session trace.",
            "For each frozen slot verify feasibility, natural need, scope, exact evidence quotations, alternative sufficient paths and semantic historical comparisons; inspect nonstandard/prose citations manually.",
            "Verify UNKNOWN and empty comparisons remain uncertain, known duplicates are established by supplied reviewed history, and reliable uncovered slots receive the specified preference.",
            "When no reliably uncovered feasible slot exists, verify whole-tier frequency sorting only if every remaining unknown slot has a unique HIGH-confidence supported operation and evidence family, with both frozen counts present; otherwise use Unicode code-point slot_id order.",
            "Verify rationale, confidence, counts or the reason frequency was skipped, pre-draft limitations, final decision, actual UTC and model/usage/cost or null where unavailable.",
            "Check session trace for exclusive-create output, prohibited role/identity/card/path access and privacy leakage. A shared filesystem provides no verified OS isolation; missing trace is not proof.",
            "Independent review cannot alter the selection or trigger retry. Do not add an unannounced exact output schema or acceptance threshold."
        ],
        "handoff_authorized_by_this_report": False,
        "resampling_or_retry": False
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
