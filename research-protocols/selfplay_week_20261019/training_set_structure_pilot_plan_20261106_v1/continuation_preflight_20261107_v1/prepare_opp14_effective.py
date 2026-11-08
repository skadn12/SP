#!/usr/bin/env python3
"""Prepare OPP14 author input from the original frozen trial; do not launch."""
import json
from pathlib import Path

from check import OLD, chunks, digest, strict_json, turns
from prepare_opp12 import exclusive, json_bytes

CODE = "OPP27_14_0396c2550a"
DEST = OLD / "continuation_opp14_effective_packet_v1"
FULL_CONTRACT_SHA = "8d83f8c0f3321aaccf824eebb517f724954f03c05bf5f8cf74a29de83ade5568"


def main():
    auth_path = OLD / "AUTHOR_LAUNCH_AUTHORIZATION.json"
    auth = strict_json(auth_path)
    if (auth.get("status") != "AUTHORIZED_ISOLATED_AUTHOR_SMALL_TRIAL_ONLY"
            or auth.get("generation_authorization") != "ISOLATED_AUTHOR_SMALL_TRIAL_ONLY"
            or auth.get("opportunity_count") != 16 or auth.get("one_attempt_per_opportunity") is not True):
        raise ValueError("original authorization is not applicable")
    scope = auth.get("scope", {})
    if scope.get("authors") is not True or any(scope.get(x) is not False for x in ("solver", "training", "export", "retry", "fallback")):
        raise ValueError("authorization scope mismatch")
    for binding in auth["bound_artifacts"].values():
        if digest(binding["path"]) != binding["sha256"]:
            raise ValueError("authorization binding mismatch")
    schedule = strict_json(OLD / "SCHEDULE_RESTRICTED.json")
    scheduled = [x for x in schedule["rows"] if x["opportunity_id"] == CODE]
    if len(scheduled) != 1 or scheduled[0]["root_id"] != "53623" or scheduled[0]["group_id"] != "OLD_G":
        raise ValueError("schedule mismatch")
    manifest = strict_json(OLD / "PACKET_MANIFEST.json")
    records = [x for x in manifest["packets"] if x["opportunity_id"] == CODE]
    if len(records) != 1:
        raise ValueError("original packet binding missing or duplicated")
    input_path = Path(records[0]["input_path"])
    if digest(input_path) != records[0]["input_sha256"]:
        raise ValueError("original input hash mismatch")
    packet = strict_json(input_path)
    if (packet["opportunity_id"] != CODE or packet["root_id"] != "cc-train-episode-53623"
            or packet["workflow"] != "OLD" or packet["group_id"] != "OLD_G"
            or packet["slots_ceiling"] is not None or packet["opportunity_ceiling"] != 1
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
    if output.exists() and any(output.iterdir()):
        raise ValueError("author output is not empty")
    selector_output = OLD / "selector" / CODE
    if selector_output.exists() and any(selector_output.iterdir()):
        raise ValueError("unexpected selector output")
    original_role_dir = input_path.parent
    if {p.name for p in original_role_dir.iterdir()} != {"INPUT.json"}:
        raise ValueError("original role directory is not unused")
    reconciliation_path = OLD / "PROGRESS_RECONCILIATION_20261107_v1.json"
    inventory = strict_json(reconciliation_path)["inventory"]
    if inventory["attempted_opportunities"] != 11 or inventory["unattempted_opportunities"] != 5:
        raise ValueError("historical unattempted status requires recheck")
    effective = dict(packet)
    effective["contract_path"] = str(full_contract)
    effective["contract_sha256"] = FULL_CONTRACT_SHA
    effective["generation_authorization"] = auth["generation_authorization"]
    DEST.mkdir(mode=0o700, exist_ok=False)
    exclusive(DEST / "INPUT.json", json_bytes(effective))
    receipt = {
        "status": "AUTHOR_PACKET_PREPARED_PENDING_INDEPENDENT_PREFLIGHT", "opportunity": CODE,
        "effective_input_sha256": digest(DEST / "INPUT.json"),
        "original_input_sha256": records[0]["input_sha256"],
        "original_authorization_sha256": digest(auth_path),
        "authorization_basis": "Existing explicit approval covers the 16-opportunity trial, one attempt per opportunity; no new approval is claimed.",
        "schedule_sha256": digest(OLD / "SCHEDULE_RESTRICTED.json"),
        "reconciliation_sha256": digest(reconciliation_path),
        "unattempted_basis": "Prior reconciliation lists opportunities 12-16 as unstarted; original role directory contains only INPUT.json and selector output is absent or empty. An unlogged dispatch cannot be excluded.",
        "source_sha256": packet["source_sha256"], "neutral_sha256": packet["neutral_sha256"],
        "full_contract_sha256": FULL_CONTRACT_SHA,
        "controller_only_original_short_instruction_path": packet["contract_path"],
        "controller_only_original_short_instruction_sha256": packet["contract_sha256"],
        "source_turn_count": len(rows), "source_chunk_count": len(chunks(rows)),
        "output_dir": str(output), "output_empty_or_absent": True,
        "feedback_card_provided": False,
        "limitations": ["No OS-level isolation", "Same actual model as earlier opportunities is not established", "Visible source reading and exclusive-create writes must be checked during execution"],
        "author_invoked": False
    }
    exclusive(DEST / "CONTROLLER_RECEIPT.json", json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
