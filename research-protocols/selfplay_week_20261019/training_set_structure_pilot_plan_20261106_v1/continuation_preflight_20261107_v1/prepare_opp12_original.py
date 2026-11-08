#!/usr/bin/env python3
"""Prepare controller-reviewed OPP12 instructions bound to the original source."""
import json
from pathlib import Path

from check import OLD, chunks, digest, strict_json, turns
from prepare_opp12 import CODE, exclusive, json_bytes

HERE = Path(__file__).resolve().parent
DEST = OLD / "continuation_opp12_author_packet_v1"


def main():
    record = next(x for x in strict_json(OLD / "PACKET_MANIFEST.json")["packets"] if x["opportunity_id"] == CODE)
    old_path = Path(record["input_path"])
    if digest(old_path) != record["input_sha256"]:
        raise ValueError("original input hash mismatch")
    old = strict_json(old_path)
    if old["workflow"] != "OLD" or old["group_id"] != "OLD_G":
        raise ValueError("schedule mismatch")
    for label in ("source", "neutral", "contract"):
        if digest(old[label + "_path"]) != old[label + "_sha256"]:
            raise ValueError(label + " hash mismatch")
    source_rows = turns(strict_json(old["source_path"]))
    strict_json(old["neutral_path"])
    if len(source_rows) != old["source_turn_count"]:
        raise ValueError("source count mismatch")
    output = Path(old["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("output is not empty")
    full = OLD / "contracts/ISOLATED_AUTHOR_CONTRACT.txt"
    full_hash = "8d83f8c0f3321aaccf824eebb517f724954f03c05bf5f8cf74a29de83ade5568"
    if digest(full) != full_hash:
        raise ValueError("full contract hash mismatch")
    if "feedback_card_path" in old:
        raise ValueError("G opportunity must not receive feedback")
    DEST.mkdir(mode=0o700, exist_ok=False)
    projected = dict(old)
    projected["contract_path"] = str(full)
    projected["contract_sha256"] = full_hash
    projected["workflow_instruction_path"] = old["contract_path"]
    projected["workflow_instruction_sha256"] = old["contract_sha256"]
    # This file is a draft. Authority must be assessed against the original
    # authorization and independent continuation audit; it grants none.
    projected["generation_authorization"] = "NONE"
    exclusive(DEST / "INPUT.json", json_bytes(projected))
    receipt = {
        "status": "DRAFT_NOT_AUTHORIZED", "opportunity": CODE,
        "original_input_sha256": record["input_sha256"],
        "draft_input_sha256": digest(DEST / "INPUT.json"),
        "original_source_sha256": old["source_sha256"],
        "original_neutral_sha256": old["neutral_sha256"],
        "full_contract_sha256": full_hash, "old_instruction_sha256": old["contract_sha256"],
        "turn_count": len(source_rows), "source_chunk_count": len(chunks(source_rows)),
        "original_output_dir": str(output), "original_output_empty_or_absent": True,
        "original_authorization_path": str(OLD / "AUTHOR_LAUNCH_AUTHORIZATION.json"),
        "original_authorization_sha256": digest(OLD / "AUTHOR_LAUNCH_AUTHORIZATION.json"),
        "limitations": ["No OS-level isolation", "Actual model equality not established",
                        "Independent packet and launch preflight not yet completed",
                        "Source turn visibility cannot be proven before author reads it"],
        "author_invoked": False
    }
    exclusive(DEST / "CONTROLLER_RECEIPT.json", json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
