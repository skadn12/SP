#!/usr/bin/env python3
"""Prepare one frozen NEW_F author packet without starting any role."""
import argparse
import json
from pathlib import Path

from check import OLD, chunks, digest, strict_json, turns
from prepare_opp12 import exclusive, json_bytes

TARGETS = {15: ("OPP27_15_33eefb12b7", "94052"), 16: ("OPP27_16_9d3abd2fbe", "98366")}
CONTRACT_SHA = "8d83f8c0f3321aaccf824eebb517f724954f03c05bf5f8cf74a29de83ade5568"
CARD_SHA = "f33ca0b972a7921d178564961c9cf4a9612e334d0c1616dcd5f5690e190745db"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("number", type=int, choices=sorted(TARGETS))
    number = parser.parse_args().number
    code, root = TARGETS[number]
    dest = OLD / f"continuation_opp{number}_effective_packet_v1"
    if number == 16 and not (OLD / "continuation_opp15_effective_packet_v1/CONTROLLER_RECEIPT.json").is_file():
        raise ValueError("OPP15 packet must be prepared first")
    auth_path = OLD / "AUTHOR_LAUNCH_AUTHORIZATION.json"
    auth = strict_json(auth_path)
    scope = auth.get("scope", {})
    if (auth.get("status") != "AUTHORIZED_ISOLATED_AUTHOR_SMALL_TRIAL_ONLY"
            or auth.get("generation_authorization") != "ISOLATED_AUTHOR_SMALL_TRIAL_ONLY"
            or auth.get("opportunity_count") != 16 or auth.get("one_attempt_per_opportunity") is not True
            or not all(scope.get(k) is True for k in ("authors", "semantic_slot_submission", "formal_question_implementation_after_NEW_selection"))
            or not all(scope.get(k) is False for k in ("solver", "training", "export", "retry", "fallback"))):
        raise ValueError("original authorization is not applicable")
    for binding in auth["bound_artifacts"].values():
        if digest(binding["path"]) != binding["sha256"]:
            raise ValueError("original authorization binding mismatch")
    scheduled = [x for x in strict_json(OLD / "SCHEDULE_RESTRICTED.json")["rows"] if x["opportunity_id"] == code]
    if (len(scheduled) != 1 or scheduled[0]["root_id"] != root or scheduled[0]["group_id"] != "NEW_F"
            or scheduled[0]["feedback_card"] != "fixed_FC09"):
        raise ValueError("frozen schedule mismatch")
    records = [x for x in strict_json(OLD / "PACKET_MANIFEST.json")["packets"] if x["opportunity_id"] == code]
    if len(records) != 1 or digest(records[0]["input_path"]) != records[0]["input_sha256"]:
        raise ValueError("original input binding mismatch")
    original_path = Path(records[0]["input_path"])
    packet = strict_json(original_path)
    if (packet["opportunity_id"] != code or packet["root_id"] != "cc-train-episode-" + root
            or packet["workflow"] != "NEW" or packet["group_id"] != "NEW_F"
            or packet["slots_ceiling"] != 3 or packet["opportunity_ceiling"] != 1
            or packet["generation_authorization"] != "NONE" or packet["feedback_card_sha256"] != CARD_SHA):
        raise ValueError("original opportunity input mismatch")
    for label in ("source", "neutral", "contract", "feedback_card"):
        if digest(packet[label + "_path"]) != packet[label + "_sha256"]:
            raise ValueError(label + " hash mismatch")
    if Path(packet["feedback_card_path"]) != OLD / "contracts/ABSTRACT_FEEDBACK_F.json":
        raise ValueError("unexpected feedback card path")
    full_contract = OLD / "contracts/ISOLATED_AUTHOR_CONTRACT.txt"
    if digest(full_contract) != CONTRACT_SHA:
        raise ValueError("full author contract mismatch")
    rows = turns(strict_json(packet["source_path"]))
    strict_json(packet["neutral_path"])
    strict_json(packet["feedback_card_path"])
    if len(rows) != packet["source_turn_count"]:
        raise ValueError("source count mismatch")
    output = Path(packet["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("author output is not empty")
    selector_output = OLD / "selector" / code
    if selector_output.exists() and any(selector_output.iterdir()):
        raise ValueError("selector output is not empty")
    if {p.name for p in original_path.parent.iterdir()} != {"INPUT.json"}:
        raise ValueError("original role directory is not unused")
    reconciliation = OLD / "PROGRESS_RECONCILIATION_20261107_v1.json"
    inventory = strict_json(reconciliation)["inventory"]
    if inventory["attempted_opportunities"] != 11 or inventory["unattempted_opportunities"] != 5:
        raise ValueError("original reconciliation requires recheck")
    effective = dict(packet)
    effective["contract_path"] = str(full_contract)
    effective["contract_sha256"] = CONTRACT_SHA
    effective["generation_authorization"] = auth["generation_authorization"]
    dest.mkdir(mode=0o700, exist_ok=False)
    exclusive(dest / "INPUT.json", json_bytes(effective))
    receipt = {
        "status": "AUTHOR_PACKET_PREPARED_PENDING_INDEPENDENT_PREFLIGHT", "opportunity": code,
        "effective_input_sha256": digest(dest / "INPUT.json"), "original_input_sha256": records[0]["input_sha256"],
        "original_authorization_sha256": digest(auth_path),
        "authorization_basis": "Existing explicit approval covers the 16-opportunity trial, one attempt per opportunity; no new approval is claimed.",
        "schedule_sha256": digest(OLD / "SCHEDULE_RESTRICTED.json"), "reconciliation_sha256": digest(reconciliation),
        "unattempted_basis": "Prior reconciliation listed 12-16 unstarted; the original role directory contains only INPUT.json, and selector output is absent or empty. Unlogged dispatch cannot be excluded.",
        "source_sha256": packet["source_sha256"], "neutral_sha256": packet["neutral_sha256"],
        "feedback_card_sha256": packet["feedback_card_sha256"], "full_contract_sha256": CONTRACT_SHA,
        "controller_only_original_short_instruction_path": packet["contract_path"],
        "controller_only_original_short_instruction_sha256": packet["contract_sha256"],
        "source_turn_count": len(rows), "source_chunk_count": len(chunks(rows)),
        "output_dir": str(output), "output_empty_or_absent": True,
        "selector_feedback_boundary": "Author input, group and feedback card are not selector inputs; selector packet remains unprepared.",
        "limitations": ["No OS-level isolation", "Same actual model as earlier opportunities is not established", "Visible source reading and exclusive-create writes must be checked during execution"],
        "author_invoked": False, "selector_invoked": False
    }
    exclusive(dest / "CONTROLLER_RECEIPT.json", json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
