#!/usr/bin/env python3
"""Read-only selector packet integrity review; never invokes a role."""
import json
import re
from pathlib import Path

from check import OLD, digest, evidence, strict_json, turns

RECEIPT = OLD / "continuation_opp15_selector_controller_v1/CONTROLLER_RECEIPT.json"
STAGE = "OPP15_SELECTOR_PACKET_INTEGRITY_ONLY"


def main():
    receipt = strict_json(RECEIPT)
    packet_path = Path(receipt["selector_packet_path"])
    packet = strict_json(packet_path)
    issues = []
    if digest(packet_path) != receipt["selector_packet_sha256"]:
        issues.append("packet hash mismatch")
    if set(packet) != {"schema_version", "inputs", "output_path", "output_requirements", "status"}:
        issues.append("unexpected packet fields")
    expected_inputs = {"source", "neutral", "history", "operation_summary", "contract", "rules", "slots"}
    if set(packet["inputs"]) != expected_inputs:
        issues.append("selector allowlist mismatch")
    for label, binding in packet["inputs"].items():
        if set(binding) != {"path", "sha256"} or binding != receipt["copy_bindings"].get(label):
            issues.append(label + " binding mismatch")
            continue
        path = Path(binding["path"])
        if not path.is_absolute() or path.is_symlink() or path.parent != packet_path.parent or digest(path) != binding["sha256"]:
            issues.append(label + " path or bytes mismatch")
    output = Path(packet["output_path"])
    if output.parent != packet_path.parent.parent / "out" or output.parent.is_symlink() or output.exists() or any(output.parent.iterdir()):
        issues.append("output directory not empty or unsafe")
    forbidden = re.compile(r"cc-train-episode-\d+|OPP27_\d+|(?:OLD|NEW)_[FG]\b|FC09_MCLEAN|ABSTRACT_FEEDBACK_F|f33ca0b972a7921d178564961c9cf4a9612e334d0c1616dcd5f5690e190745db|/root/autodl-tmp/.*/role_packets/", re.I)
    for label in ("source", "neutral", "history", "operation_summary", "slots"):
        path = Path(packet["inputs"][label]["path"])
        strict_json(path)
        if forbidden.search(path.read_text(encoding="utf-8")):
            issues.append(label + " contains a private identifier, card or path")
    if forbidden.search(packet_path.read_text(encoding="utf-8")):
        issues.append("packet includes private identifier, card or path")
    original_slots = strict_json(receipt["original_slots_path"])
    projected = strict_json(packet["inputs"]["slots"]["path"])
    expected_projection = {k: v for k, v in original_slots.items() if k not in ("opportunity_id", "root_id", "source_sha256")}
    if projected != expected_projection:
        issues.append("slot semantics, values or order changed")
    if digest(receipt["original_slots_path"]) != receipt["original_slots_sha256"] or digest(receipt["slots_freeze_path"]) != receipt["slots_freeze_sha256"]:
        issues.append("slot or freeze bytes changed")
    source = strict_json(packet["inputs"]["source"]["path"])
    refs = evidence(projected, turns(source))
    if refs["issues"] or refs["checked"] != receipt["quote_entries_checked"]:
        issues.append("citation mismatch or ambiguity")
    history = strict_json(packet["inputs"]["history"]["path"])
    if history["record_level_comparisons"]["records"] != [] or "UNKNOWN" not in history["record_level_comparisons"]["status"]:
        issues.append("unknown history not retained")
    report = {"stage": STAGE, "pass": not issues, "issues": issues,
              "slots": len(projected["slots"]), "quotes": refs["checked"], "unknown_history": "RETAINED",
              "feedback_card": "NOT_IN_ALLOWLIST_OR_SCANNED_JSON", "semantic_privacy": "REQUIRES_INDEPENDENT_REVIEW",
              "filesystem_isolation": "NOT_VERIFIED", "rules_access_amendment": "REQUIRES_EXPLICIT_ROLE_INSTRUCTION",
              "selector_started_by_this_tool": False}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
