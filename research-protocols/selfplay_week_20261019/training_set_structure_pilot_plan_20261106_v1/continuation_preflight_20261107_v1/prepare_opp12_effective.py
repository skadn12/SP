#!/usr/bin/env python3
"""Bind OPP12's existing authorization to a new effective author input."""
import json
from pathlib import Path

from check import OLD, chunks, digest, strict_json, turns
from prepare_opp12 import CODE, exclusive, json_bytes

BASE = OLD / "continuation_opp12_author_packet_v1"
DEST = OLD / "continuation_opp12_effective_packet_v1"


def main():
    auth_path = OLD / "AUTHOR_LAUNCH_AUTHORIZATION.json"
    auth = strict_json(auth_path)
    if (auth.get("generation_authorization") != "ISOLATED_AUTHOR_SMALL_TRIAL_ONLY"
            or auth.get("status") != "AUTHORIZED_ISOLATED_AUTHOR_SMALL_TRIAL_ONLY"
            or auth.get("opportunity_count") != 16 or auth.get("one_attempt_per_opportunity") is not True
            or auth.get("scope", {}).get("authors") is not True):
        raise ValueError("original authorization does not permit the trial")
    if any(auth["scope"].get(x) is not False for x in ("solver", "training", "export", "retry", "fallback")):
        raise ValueError("original authorization scope mismatch")
    for binding in auth["bound_artifacts"].values():
        if digest(binding["path"]) != binding["sha256"]:
            raise ValueError("authorized artifact binding mismatch")
    receipt = strict_json(BASE / "CONTROLLER_RECEIPT.json")
    input_path = BASE / "INPUT.json"
    if digest(input_path) != receipt["draft_input_sha256"]:
        raise ValueError("draft input mismatch")
    packet = strict_json(input_path)
    if packet["opportunity_id"] != CODE or packet["generation_authorization"] != "NONE":
        raise ValueError("unexpected draft input")
    for kind in ("source", "neutral", "contract", "workflow_instruction"):
        if digest(packet[kind + "_path"]) != packet[kind + "_sha256"]:
            raise ValueError(kind + " hash mismatch")
    rows = turns(strict_json(packet["source_path"]))
    strict_json(packet["neutral_path"])
    if len(rows) != packet["source_turn_count"]:
        raise ValueError("turn count mismatch")
    if "feedback_card_path" in packet or packet["workflow"] != "OLD":
        raise ValueError("opportunity is not OLD_G")
    output = Path(packet["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("output is not empty")
    packet["generation_authorization"] = auth["generation_authorization"]
    DEST.mkdir(mode=0o700, exist_ok=False)
    exclusive(DEST / "INPUT.json", json_bytes(packet))
    summary = {
        "status": "PACKET_PREPARED_PENDING_INDEPENDENT_PREFLIGHT", "opportunity": CODE,
        "effective_input_sha256": digest(DEST / "INPUT.json"),
        "prior_draft_input_sha256": digest(input_path),
        "original_authorization_sha256": digest(auth_path),
        "authorization_basis": "Preparation input NONE predates explicit user approval. Existing authorization is limited to the 16 scheduled opportunities and one attempt each; no new approval is represented.",
        "source_sha256": packet["source_sha256"], "neutral_sha256": packet["neutral_sha256"],
        "contract_sha256": packet["contract_sha256"], "old_instruction_sha256": packet["workflow_instruction_sha256"],
        "source_turn_count": len(rows), "source_chunk_count": len(chunks(rows)),
        "output_dir": str(output), "output_empty_or_absent": True,
        "limitations": ["No OS-level isolation", "Same actual model as prior opportunities is not established", "Visible reading must be checked during author execution"],
        "author_invoked": False
    }
    exclusive(DEST / "CONTROLLER_RECEIPT.json", json_bytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
