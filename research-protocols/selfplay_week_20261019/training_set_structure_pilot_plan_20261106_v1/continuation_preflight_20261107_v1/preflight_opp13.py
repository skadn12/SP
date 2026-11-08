#!/usr/bin/env python3
"""Read-only OPP13 author-packet check. Not a selector or output check."""
import json
from pathlib import Path

from check import OLD, chunks, digest, strict_json, turns

CODE = "OPP27_13_e5c5de9d35"
DEST = OLD / "continuation_opp13_effective_packet_v1"


def main():
    problems = []
    manifest = strict_json(OLD / "PACKET_MANIFEST.json")
    binding = next(x for x in manifest["packets"] if x["opportunity_id"] == CODE)
    original = strict_json(binding["input_path"])
    current_path = DEST / "INPUT.json"
    current = strict_json(current_path)
    receipt = strict_json(DEST / "CONTROLLER_RECEIPT.json")
    authorization_path = OLD / "AUTHOR_LAUNCH_AUTHORIZATION.json"
    authorization = strict_json(authorization_path)
    if digest(binding["input_path"]) != binding["input_sha256"]:
        problems.append("original input hash mismatch")
    if digest(current_path) != receipt["effective_input_sha256"]:
        problems.append("effective input hash mismatch")
    if digest(authorization_path) != receipt["original_authorization_sha256"]:
        problems.append("authorization hash mismatch")
    expected = dict(original)
    expected["contract_path"] = str(OLD / "contracts/ISOLATED_AUTHOR_CONTRACT.txt")
    expected["contract_sha256"] = receipt["full_contract_sha256"]
    expected["generation_authorization"] = authorization["generation_authorization"]
    if current != expected:
        problems.append("effective packet differs beyond reviewed changes")
    if current.get("workflow") != "NEW" or current.get("group_id") != "NEW_G" or "feedback_card_path" in current:
        problems.append("workflow or feedback mismatch")
    for name in ("source", "neutral", "contract"):
        if digest(current[name + "_path"]) != current[name + "_sha256"]:
            problems.append(name + " bytes mismatch")
    if digest(receipt["controller_only_original_short_instruction_path"]) != receipt["controller_only_original_short_instruction_sha256"]:
        problems.append("short instruction binding mismatch")
    rows = turns(strict_json(current["source_path"]))
    if len(rows) != current["source_turn_count"]:
        problems.append("turn count mismatch")
    grouped = chunks(rows)
    output = Path(current["output_dir"])
    if output.exists() and any(output.iterdir()):
        problems.append("author output exists")
    selector_dir = OLD / "selector" / CODE
    if selector_dir.exists() and any(selector_dir.iterdir()):
        problems.append("selector output exists")
    result = {"stage": "OPP13_AUTHOR_INPUT_ONLY", "pass": not problems, "issues": problems,
              "turns": len(rows), "source_chunks": len(grouped),
              "model_visible_source_read": "NOT_STARTED", "selector_packet": "NOT_CHECKED",
              "selector_history": "NOT_CHECKED", "slots": "NOT_CREATED", "question": "NOT_CREATED",
              "os_isolation": "NOT_ENFORCED", "same_prior_model": "NOT_ESTABLISHED"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
