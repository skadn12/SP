#!/usr/bin/env python3
"""Read-only packet integrity review; does not establish OS isolation."""
import json
import re
from pathlib import Path

from check import OLD, digest, evidence, strict_json, turns


def main():
    receipt = strict_json(OLD / "continuation_opp13_selector_controller_v2/CONTROLLER_RECEIPT.json")
    packet_path = Path(receipt["selector_packet_path"])
    packet = strict_json(packet_path)
    errors = []
    if digest(packet_path) != receipt["selector_packet_sha256"]:
        errors.append("packet bytes mismatch")
    if set(packet) != {"schema_version", "inputs", "output_path", "output_requirements", "status"}:
        errors.append("packet fields mismatch")
    if set(packet["inputs"]) != {"source", "neutral", "history", "operation_summary", "contract", "rules", "slots"}:
        errors.append("input allowlist mismatch")
    for label, entry in packet["inputs"].items():
        if set(entry) != {"path", "sha256"} or not Path(entry["path"]).is_absolute():
            errors.append(label + " binding malformed")
            continue
        path = Path(entry["path"])
        if path.is_symlink() or path.parent != packet_path.parent or digest(path) != entry["sha256"]:
            errors.append(label + " copy or location mismatch")
        if entry != receipt["copy_bindings"].get(label):
            errors.append(label + " receipt mismatch")
    output = Path(packet["output_path"])
    if output.parent != packet_path.parent.parent / "out" or output.parent.is_symlink() or output.exists() or any(output.parent.iterdir()):
        errors.append("output is not exclusive and empty")
    raw_files = [Path(packet["inputs"][key]["path"]) for key in ("source", "neutral", "history", "operation_summary", "slots")]
    pattern = re.compile(r"cc-train-episode-\d+|OPP27_\d+|(?:OLD|NEW)_[FG]\b|/root/autodl-tmp/.*/role_packets/", re.I)
    for path in raw_files:
        strict_json(path)
        if pattern.search(path.read_text(encoding="utf-8")):
            errors.append("private identity or path in " + path.name)
    original = strict_json(receipt["original_slots_path"])
    projected = strict_json(packet["inputs"]["slots"]["path"])
    if set(projected) != {"slots"} or projected["slots"] != original["slots"]:
        errors.append("slot values or order changed")
    if digest(receipt["original_slots_path"]) != receipt["original_slots_sha256"]:
        errors.append("original slots changed")
    if digest(receipt["slots_freeze_path"]) != receipt["slots_freeze_sha256"]:
        errors.append("slot freeze changed")
    source = strict_json(packet["inputs"]["source"]["path"])
    refs = evidence(projected, turns(source))
    if refs["issues"] or refs["checked"] != receipt["quote_entries_checked"]:
        errors.append("evidence did not resolve uniquely")
    history = strict_json(packet["inputs"]["history"]["path"])
    if history["record_level_comparisons"]["records"] != [] or "UNKNOWN" not in history["record_level_comparisons"]["status"]:
        errors.append("history uncertainty not preserved")
    result = {"stage": "SELECTOR_PACKET_INTEGRITY_ONLY", "pass": not errors, "issues": errors,
              "slot_count": len(projected["slots"]), "quote_entries": refs["checked"],
              "unknown_history": "RETAINED", "semantic_privacy": "REQUIRES_INDEPENDENT_REVIEW",
              "os_isolation": "NOT_VERIFIED", "rules_access_amendment": "REQUIRES_EXPLICIT_ROLE_INSTRUCTION",
              "selector_started": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
