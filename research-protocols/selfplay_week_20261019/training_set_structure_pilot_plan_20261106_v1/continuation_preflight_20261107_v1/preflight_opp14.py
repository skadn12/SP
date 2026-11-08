#!/usr/bin/env python3
"""Read-only OPP14 author-input integrity check; no role is started."""
import json
from pathlib import Path

from check import OLD, chunks, digest, strict_json, turns

CODE = "OPP27_14_0396c2550a"
DEST = OLD / "continuation_opp14_effective_packet_v1"


def main():
    problems = []
    manifest = strict_json(OLD / "PACKET_MANIFEST.json")
    binding = next(x for x in manifest["packets"] if x["opportunity_id"] == CODE)
    original = strict_json(binding["input_path"])
    current_path = DEST / "INPUT.json"
    current = strict_json(current_path)
    receipt = strict_json(DEST / "CONTROLLER_RECEIPT.json")
    auth_path = OLD / "AUTHOR_LAUNCH_AUTHORIZATION.json"
    auth = strict_json(auth_path)
    if digest(binding["input_path"]) != binding["input_sha256"]:
        problems.append("original input hash mismatch")
    if digest(current_path) != receipt["effective_input_sha256"] or digest(auth_path) != receipt["original_authorization_sha256"]:
        problems.append("effective input or authorization hash mismatch")
    expected = dict(original)
    expected["contract_path"] = str(OLD / "contracts/ISOLATED_AUTHOR_CONTRACT.txt")
    expected["contract_sha256"] = receipt["full_contract_sha256"]
    expected["generation_authorization"] = auth["generation_authorization"]
    if current != expected:
        problems.append("effective input differs beyond reviewed changes")
    if current.get("workflow") != "OLD" or current.get("group_id") != "OLD_G" or "feedback_card_path" in current:
        problems.append("workflow or feedback mismatch")
    if "workflow_instruction_path" in current:
        problems.append("controller-only short instruction path exposed")
    for name in ("source", "neutral", "contract"):
        if digest(current[name + "_path"]) != current[name + "_sha256"]:
            problems.append(name + " bytes mismatch")
    if digest(receipt["controller_only_original_short_instruction_path"]) != receipt["controller_only_original_short_instruction_sha256"]:
        problems.append("controller-only short instruction mismatch")
    rows = turns(strict_json(current["source_path"]))
    strict_json(current["neutral_path"])
    if len(rows) != current["source_turn_count"]:
        problems.append("turn count mismatch")
    output = Path(current["output_dir"])
    if output.exists() and any(output.iterdir()):
        problems.append("author output exists")
    if {p.name for p in Path(binding["input_path"]).parent.iterdir()} != {"INPUT.json"}:
        problems.append("original role directory not unused")
    result = {"stage": "OPP14_AUTHOR_INPUT_ONLY", "pass": not problems, "issues": problems,
              "turns": len(rows), "source_chunks": len(chunks(rows)),
              "author_source_visible_read": "NOT_STARTED", "output_exclusive_create": "NOT_YET_OBSERVED",
              "unlogged_dispatch_absence": "NOT_PROVED", "os_isolation": "NOT_ENFORCED",
              "same_prior_model": "NOT_ESTABLISHED", "author_started_by_this_tool": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
