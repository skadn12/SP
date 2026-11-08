#!/usr/bin/env python3
"""Prepare OPP13 author input from frozen v1 artifacts; never invoke a role."""
import json
from pathlib import Path

from check import OLD, chunks, digest, strict_json, turns
from prepare_opp12 import exclusive, json_bytes

CODE = "OPP27_13_e5c5de9d35"
DEST = OLD / "continuation_opp13_effective_packet_v1"
FULL_CONTRACT_SHA = "8d83f8c0f3321aaccf824eebb517f724954f03c05bf5f8cf74a29de83ade5568"


def main():
    auth_path = OLD / "AUTHOR_LAUNCH_AUTHORIZATION.json"
    auth = strict_json(auth_path)
    if (auth.get("status") != "AUTHORIZED_ISOLATED_AUTHOR_SMALL_TRIAL_ONLY"
            or auth.get("generation_authorization") != "ISOLATED_AUTHOR_SMALL_TRIAL_ONLY"
            or auth.get("opportunity_count") != 16 or auth.get("one_attempt_per_opportunity") is not True):
        raise ValueError("original authorization is not applicable")
    scope = auth.get("scope", {})
    if not all(scope.get(key) is True for key in ("authors", "semantic_slot_submission", "formal_question_implementation_after_NEW_selection")):
        raise ValueError("author or slot scope missing")
    if not all(scope.get(key) is False for key in ("solver", "training", "export", "retry", "fallback")):
        raise ValueError("unexpected scope")
    for binding in auth["bound_artifacts"].values():
        if digest(binding["path"]) != binding["sha256"]:
            raise ValueError("authorization binding mismatch")
    schedule = strict_json(OLD / "SCHEDULE_RESTRICTED.json")
    scheduled = [x for x in schedule["rows"] if x["opportunity_id"] == CODE]
    if len(scheduled) != 1 or scheduled[0]["root_id"] != "53623" or scheduled[0]["group_id"] != "NEW_G":
        raise ValueError("schedule mismatch")
    manifest = strict_json(OLD / "PACKET_MANIFEST.json")
    records = [x for x in manifest["packets"] if x["opportunity_id"] == CODE]
    if len(records) != 1:
        raise ValueError("packet binding missing or duplicated")
    input_path = Path(records[0]["input_path"])
    if digest(input_path) != records[0]["input_sha256"]:
        raise ValueError("original input hash mismatch")
    packet = strict_json(input_path)
    if (packet["opportunity_id"] != CODE or packet["root_id"] != "cc-train-episode-53623"
            or packet["workflow"] != "NEW" or packet["group_id"] != "NEW_G"
            or packet["slots_ceiling"] != 3 or packet["opportunity_ceiling"] != 1
            or packet["generation_authorization"] != "NONE" or "feedback_card_path" in packet):
        raise ValueError("opportunity packet mismatch")
    for label in ("source", "neutral", "contract"):
        if digest(packet[label + "_path"]) != packet[label + "_sha256"]:
            raise ValueError(label + " hash mismatch")
    full_contract = OLD / "contracts/ISOLATED_AUTHOR_CONTRACT.txt"
    if digest(full_contract) != FULL_CONTRACT_SHA:
        raise ValueError("full contract mismatch")
    rows = turns(strict_json(packet["source_path"]))
    strict_json(packet["neutral_path"])
    if len(rows) != packet["source_turn_count"]:
        raise ValueError("source count mismatch")
    output = Path(packet["output_dir"])
    selector_output = OLD / "selector" / CODE
    if output.exists() and any(output.iterdir()):
        raise ValueError("author output is not empty")
    if selector_output.exists() and any(selector_output.iterdir()):
        raise ValueError("selector output already exists")
    reconciliation_path = OLD / "PROGRESS_RECONCILIATION_20261107_v1.json"
    reconciliation = strict_json(reconciliation_path)
    inventory = reconciliation["inventory"]
    if inventory["attempted_opportunities"] != 11 or inventory["unattempted_opportunities"] != 5:
        raise ValueError("unattempted status must be rechecked")
    # Preserve original source/neutral paths and bytes. The short instruction
    # is bound in the controller receipt but is not an author-readable input.
    new_packet = dict(packet)
    new_packet["contract_path"] = str(full_contract)
    new_packet["contract_sha256"] = FULL_CONTRACT_SHA
    new_packet["generation_authorization"] = auth["generation_authorization"]
    DEST.mkdir(mode=0o700, exist_ok=False)
    exclusive(DEST / "INPUT.json", json_bytes(new_packet))
    receipt = {
        "status": "AUTHOR_PACKET_PREPARED_PENDING_INDEPENDENT_PREFLIGHT", "opportunity": CODE,
        "effective_input_sha256": digest(DEST / "INPUT.json"),
        "original_input_sha256": records[0]["input_sha256"],
        "original_authorization_sha256": digest(auth_path),
        "authorization_basis": "Existing explicit user approval permits this 16-opportunity trial; one attempt per opportunity. No new approval is claimed.",
        "schedule_sha256": digest(OLD / "SCHEDULE_RESTRICTED.json"),
        "reconciliation_sha256": digest(reconciliation_path),
        "unattempted_basis": "Prior reconciliation records opportunities 12-16 unstarted; original author and selector output locations are empty or absent. This does not prove absence of an unlogged role dispatch.",
        "source_sha256": packet["source_sha256"], "neutral_sha256": packet["neutral_sha256"],
        "full_contract_sha256": FULL_CONTRACT_SHA,
        "controller_only_original_short_instruction_path": packet["contract_path"],
        "controller_only_original_short_instruction_sha256": packet["contract_sha256"],
        "source_turn_count": len(rows), "source_chunk_count": len(chunks(rows)),
        "output_dir": str(output), "output_empty_or_absent": True,
        "feedback_card_provided": False,
        "selector_packet_preflight": "NOT_DONE", "selector_history_preflight": "NOT_DONE",
        "limitations": ["No OS-level isolation", "Same actual model as earlier opportunities is not established", "Visible source reading must be checked during execution"],
        "author_invoked": False
    }
    exclusive(DEST / "CONTROLLER_RECEIPT.json", json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
