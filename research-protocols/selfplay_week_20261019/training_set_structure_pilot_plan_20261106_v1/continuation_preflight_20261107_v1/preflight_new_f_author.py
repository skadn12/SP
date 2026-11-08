#!/usr/bin/env python3
"""Read-only NEW_F author-packet check; no role is started."""
import argparse
import json
from pathlib import Path

from check import OLD, chunks, digest, strict_json, turns
from prepare_new_f_author import CARD_SHA, TARGETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("number", type=int, choices=sorted(TARGETS))
    number = parser.parse_args().number
    code, root = TARGETS[number]
    dest = OLD / f"continuation_opp{number}_effective_packet_v1"
    receipt = strict_json(dest / "CONTROLLER_RECEIPT.json")
    packet_path = dest / "INPUT.json"
    current = strict_json(packet_path)
    binding = next(x for x in strict_json(OLD / "PACKET_MANIFEST.json")["packets"] if x["opportunity_id"] == code)
    original = strict_json(binding["input_path"])
    auth_path = OLD / "AUTHOR_LAUNCH_AUTHORIZATION.json"
    auth = strict_json(auth_path)
    issues = []
    if digest(binding["input_path"]) != binding["input_sha256"] or digest(packet_path) != receipt["effective_input_sha256"]:
        issues.append("original or effective packet hash mismatch")
    if digest(auth_path) != receipt["original_authorization_sha256"]:
        issues.append("authorization hash mismatch")
    expected = dict(original)
    expected["contract_path"] = str(OLD / "contracts/ISOLATED_AUTHOR_CONTRACT.txt")
    expected["contract_sha256"] = receipt["full_contract_sha256"]
    expected["generation_authorization"] = auth["generation_authorization"]
    if current != expected:
        issues.append("effective input differs beyond reviewed changes")
    if (current.get("opportunity_id") != code or current.get("root_id") != "cc-train-episode-" + root
            or current.get("workflow") != "NEW" or current.get("group_id") != "NEW_F"
            or current.get("feedback_card_sha256") != CARD_SHA or "workflow_instruction_path" in current):
        issues.append("assignment, workflow or feedback mismatch")
    for label in ("source", "neutral", "contract", "feedback_card"):
        if digest(current[label + "_path"]) != current[label + "_sha256"]:
            issues.append(label + " bytes mismatch")
    if digest(receipt["controller_only_original_short_instruction_path"]) != receipt["controller_only_original_short_instruction_sha256"]:
        issues.append("controller-only short instruction mismatch")
    rows = turns(strict_json(current["source_path"]))
    strict_json(current["neutral_path"])
    strict_json(current["feedback_card_path"])
    if len(rows) != current["source_turn_count"]:
        issues.append("source count mismatch")
    output = Path(current["output_dir"])
    if output.exists() and any(output.iterdir()):
        issues.append("author output already exists")
    if {p.name for p in Path(binding["input_path"]).parent.iterdir()} != {"INPUT.json"}:
        issues.append("original role directory is not unused")
    selector = OLD / "selector" / code
    if selector.exists() and any(selector.iterdir()):
        issues.append("selector output already exists")
    result = {"stage": "NEW_F_AUTHOR_INPUT_ONLY", "opportunity": code, "pass": not issues,
              "issues": issues, "turns": len(rows), "source_chunks": len(chunks(rows)),
              "feedback_card_hash": CARD_SHA, "selector_packet": "NOT_PREPARED_OR_VERIFIED",
              "author_source_visible_read": "NOT_STARTED", "output_exclusive_create": "NOT_YET_OBSERVED",
              "unlogged_dispatch_absence": "NOT_PROVED", "os_isolation": "NOT_ENFORCED",
              "same_prior_model": "NOT_ESTABLISHED", "role_started_by_this_tool": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
