#!/usr/bin/env python3
"""Prepare an anonymous OPP15 selector packet without invoking a selector."""
import json
import secrets
from pathlib import Path

from check import OLD, digest, evidence, strict_json, turns
from prepare_opp12 import exclusive, json_bytes
from prepare_opp13_selector import FIELDS, check_semantic_identity

CODE = "OPP27_15_33eefb12b7"
AUTHOR = OLD / "role_packets" / CODE / "author_output"
HISTORY = OLD / "continuation_new_history_view_v1"
PROJECTION = OLD / "controller_25b2cf7d8309fa1b4f467cc2195ca591/CONTROLLER_PRIVATE_BINDINGS.json"
PRIVATE = OLD / "continuation_opp15_selector_controller_v1"
AUTHOR_PACKET = OLD / "continuation_opp15_effective_packet_v1"


def main():
    if PRIVATE.exists():
        raise ValueError("controller destination already exists")
    if {p.name for p in AUTHOR.iterdir()} != {"NEW_SLOTS.json", "SLOTS_FREEZE.json", "INPUT_READ_LOG.json"}:
        raise ValueError("unexpected author files or incomplete freeze")
    author_input_path = AUTHOR_PACKET / "INPUT.json"
    author_receipt = strict_json(AUTHOR_PACKET / "CONTROLLER_RECEIPT.json")
    if digest(author_input_path) != author_receipt["effective_input_sha256"]:
        raise ValueError("effective author input changed")
    inp = strict_json(author_input_path)
    if inp["opportunity_id"] != CODE:
        raise ValueError("author opportunity mismatch")
    private_projection = strict_json(PROJECTION)
    matches = [x for x in private_projection["bindings"] if x["opportunity_id"] == CODE]
    if len(matches) != 1:
        raise ValueError("source projection binding missing or duplicated")
    binding = matches[0]
    for name in ("source", "neutral"):
        if inp[name + "_path"] != binding[name + "_path"] or digest(inp[name + "_path"]) != inp[name + "_sha256"] or inp[name + "_sha256"] != binding[name + "_sha256"]:
            raise ValueError(name + " source binding mismatch")
        if digest(binding["selector_" + name + "_path"]) != binding["selector_" + name + "_sha256"]:
            raise ValueError(name + " projection hash mismatch")
    source = strict_json(inp["source_path"])
    neutral = strict_json(inp["neutral_path"])
    anon_source = Path(binding["selector_source_path"])
    anon_neutral = Path(binding["selector_neutral_path"])
    if {k: v for k, v in source.items() if k != "dataID"} != strict_json(anon_source):
        raise ValueError("source projection altered nonidentity fields")
    if {k: v for k, v in neutral.items() if k not in ("root_id", "canonical_source_binding")} != strict_json(anon_neutral):
        raise ValueError("neutral projection altered nonidentity fields")
    slots_path = AUTHOR / "NEW_SLOTS.json"
    freeze_path = AUTHOR / "SLOTS_FREEZE.json"
    freeze = strict_json(freeze_path)
    wrapper = strict_json(slots_path)
    frozen_hash = freeze.get("sha256", freeze.get("slots_sha256"))
    if ("sha256" in freeze and "slots_sha256" in freeze and freeze["sha256"] != freeze["slots_sha256"]):
        raise ValueError("conflicting frozen slot hashes")
    if (freeze.get("opportunity_id") != CODE or Path(freeze.get("slots_path", "")) != slots_path
            or digest(slots_path) != frozen_hash or wrapper.get("opportunity_id") != CODE
            or wrapper.get("root_id") != inp["root_id"] or wrapper.get("source_sha256") != inp["source_sha256"]):
        raise ValueError("frozen slot binding mismatch")
    if set(wrapper) not in ({"opportunity_id", "root_id", "source_sha256", "slots"},
                            {"opportunity_id", "root_id", "source_sha256", "source_turn_count", "slots"}):
        raise ValueError("unexpected outer slot fields")
    if "source_turn_count" in wrapper and wrapper["source_turn_count"] != inp["source_turn_count"]:
        raise ValueError("source turn count mismatch")
    slots = wrapper["slots"]
    if not isinstance(slots, list) or not 1 <= len(slots) <= 3:
        raise ValueError("invalid slot count")
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != FIELDS:
            raise ValueError("unexpected slot semantic fields")
        if not all(isinstance(slot[k], str) and slot[k] for k in ("slot_id", "natural_information_need", "precise_query_scope")):
            raise ValueError("invalid semantic text")
        if not isinstance(slot["answer_program"], (str, list)) or not slot["answer_program"]:
            raise ValueError("invalid answer program")
        if not all(isinstance(slot[k], list) for k in ("facts", "required_relations", "alternative_sufficient_paths", "unresolved_relations")):
            raise ValueError("invalid semantic lists")
        for fact in slot["facts"]:
            if not isinstance(fact, dict) or set(fact) != {"fact_id", "proposition", "evidence"} or not isinstance(fact["evidence"], list):
                raise ValueError("invalid fact structure")
            if any(not isinstance(e, dict) or set(e) != {"turn_ref", "quote", "origin"} for e in fact["evidence"]):
                raise ValueError("invalid evidence structure")
    ids = [s["slot_id"] for s in slots]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate slot identifiers")
    check_semantic_identity(slots)
    evidence_result = evidence(slots, turns(source))
    if evidence_result["issues"] or not evidence_result["checked"]:
        raise ValueError("slot evidence does not uniquely resolve")
    freeze_receipt_path = HISTORY / "CONTROLLER_PREPARATION_FREEZE_RECEIPT.json"
    freeze_receipt = strict_json(freeze_receipt_path)
    for artifact in freeze_receipt["artifacts"]:
        if digest(OLD / artifact["path"]) != artifact["sha256"]:
            raise ValueError("prior continuation freeze binding mismatch")
    history = HISTORY / "SELECTOR_HISTORY_VIEW.json"
    history_data = strict_json(history)
    if history_data["record_level_comparisons"]["records"] != [] or "UNKNOWN" not in history_data["record_level_comparisons"]["status"]:
        raise ValueError("history unknown state not preserved")
    summary_freeze_path = OLD.parent / "execution_20261106_v1/manifests/SELECTOR_OPERATION_SUMMARY_FREEZE.json"
    summary_binding = strict_json(summary_freeze_path)
    summary = Path(summary_binding["path"])
    if digest(summary) != summary_binding["sha256"]:
        raise ValueError("operation summary hash mismatch")
    box = OLD / ("selector_box_" + secrets.token_hex(12))
    box.mkdir(mode=0o700, exist_ok=False)
    (box / "in").mkdir(mode=0o700, exist_ok=False)
    (box / "out").mkdir(mode=0o700, exist_ok=False)
    inputs = {"source": (anon_source, "source.json"), "neutral": (anon_neutral, "neutral.json"),
              "history": (history, "history.json"), "operation_summary": (summary, "operations.json"),
              "contract": (OLD / "contracts/ISOLATED_SELECTOR_CONTRACT.txt", "contract.txt"),
              "rules": (HISTORY / "SELECTOR_CONTINUATION_RULES_DRAFT.txt", "rules.txt")}
    copies = {}
    for label, (original, name) in inputs.items():
        target = box / "in" / name
        exclusive(target, original.read_bytes())
        if digest(target) != digest(original):
            raise ValueError("copy hash mismatch")
        copies[label] = {"path": str(target), "sha256": digest(target)}
    projected_path = box / "in/slots.json"
    projection = {k: v for k, v in wrapper.items() if k not in ("opportunity_id", "root_id", "source_sha256")}
    exclusive(projected_path, json_bytes(projection))
    if strict_json(projected_path)["slots"] != slots:
        raise ValueError("slot values changed in projection")
    copies["slots"] = {"path": str(projected_path), "sha256": digest(projected_path)}
    output = box / "out/result.json"
    packet_path = box / "in/packet.json"
    packet = {"schema_version": "isolated_selector_packet_v1", "inputs": copies,
              "output_path": str(output), "output_requirements": "Follow the supplied contract and rules; output once using exclusive create.",
              "status": "PREPARED_PENDING_INDEPENDENT_PREFLIGHT"}
    exclusive(packet_path, json_bytes(packet))
    if output.exists():
        raise ValueError("selector output already exists")
    PRIVATE.mkdir(mode=0o700, exist_ok=False)
    receipt = {"status": "DRAFT_PENDING_INDEPENDENT_PREFLIGHT", "opportunity_id": CODE,
               "author_input_sha256": digest(author_input_path), "projection_private_binding_sha256": digest(PROJECTION),
               "original_slots_path": str(slots_path), "original_slots_sha256": digest(slots_path),
               "slots_freeze_path": str(freeze_path), "slots_freeze_sha256": digest(freeze_path),
               "projection_policy": "Remove outer opportunity_id, root_id and source_sha256; preserve the slots array and all its values and order.",
               "slot_ids": ids, "slot_projection_sha256": digest(projected_path),
               "quote_entries_checked": evidence_result["checked"], "selector_packet_path": str(packet_path),
               "selector_packet_sha256": digest(packet_path), "copy_bindings": copies,
               "preparation_freeze_receipt_sha256": digest(freeze_receipt_path), "operation_summary_freeze_sha256": digest(summary_freeze_path),
               "history_semantic_comparison": "UNKNOWN; an empty comparison list does not imply no history",
               "limitations": ["Shared filesystem; no OS-level isolation verified", "Semantic and privacy audit needs independent review", "Selector has not been invoked"],
               "selector_started": False}
    exclusive(PRIVATE / "CONTROLLER_RECEIPT.json", json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
