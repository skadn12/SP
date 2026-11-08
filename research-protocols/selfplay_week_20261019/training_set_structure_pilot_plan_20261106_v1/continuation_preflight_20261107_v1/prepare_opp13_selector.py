#!/usr/bin/env python3
"""Prepare a draft selector packet; never run the selector."""
import json
import re
import secrets
from pathlib import Path

from check import OLD, digest, evidence, strict_json, turns
from prepare_opp12 import exclusive, json_bytes

CODE = "OPP27_13_e5c5de9d35"
AUTHOR = OLD / "role_packets" / CODE / "author_output"
PROJECTION = OLD / "continuation_selector_projection_20261107_v1"
HISTORY = OLD / "continuation_new_history_view_v1"
PRIVATE = OLD / "continuation_opp13_selector_controller_v2"
FIELDS = {"slot_id", "natural_information_need", "precise_query_scope", "facts", "semantic_answer",
          "answer_program", "required_relations", "alternative_sufficient_paths", "unresolved_relations"}


def check_semantic_identity(obj):
    if isinstance(obj, dict):
        forbidden = {"opportunity_id", "root_id", "group_id", "arm", "source_path", "feedback"}
        if forbidden.intersection(obj):
            raise ValueError("identity field inside semantic projection")
        for v in obj.values():
            check_semantic_identity(v)
    elif isinstance(obj, list):
        for v in obj:
            check_semantic_identity(v)
    elif isinstance(obj, str) and re.search(r"cc-train-episode-\d+|OPP27_\d+|(?:OLD|NEW)_[FG]\b|/root/", obj, re.I):
        raise ValueError("identity or private path inside semantic projection")


def main():
    if PRIVATE.exists():
        raise ValueError("controller receipt destination already exists")
    allowed_author_files = {"NEW_SLOTS.json", "SLOTS_FREEZE.json", "INPUT_READ_LOG.json"}
    if {p.name for p in AUTHOR.iterdir()} != allowed_author_files:
        raise ValueError("unexpected author output or missing frozen artifact")
    source_packet = strict_json(OLD / "continuation_opp13_effective_packet_v1/INPUT.json")
    source = strict_json(source_packet["source_path"])
    neutral = strict_json(source_packet["neutral_path"])
    if digest(source_packet["source_path"]) != source_packet["source_sha256"] or digest(source_packet["neutral_path"]) != source_packet["neutral_sha256"]:
        raise ValueError("source bytes do not match author packet")
    anon_source = PROJECTION / "selector_visible/SOURCE.json"
    anon_neutral = PROJECTION / "selector_visible/NEUTRAL.json"
    private_projection = strict_json(PROJECTION / "CONTROLLER_PRIVATE_BINDINGS.json")
    if digest(anon_source) != private_projection["source_projection_sha256"] or digest(anon_neutral) != private_projection["neutral_projection_sha256"]:
        raise ValueError("source projection digest mismatch")
    if {k: v for k, v in source.items() if k != "dataID"} != strict_json(anon_source):
        raise ValueError("source projection changed more than identity")
    if {k: v for k, v in neutral.items() if k not in ("root_id", "canonical_source_binding")} != strict_json(anon_neutral):
        raise ValueError("neutral projection changed more than identity/binding")
    freeze = strict_json(AUTHOR / "SLOTS_FREEZE.json")
    slots_file = AUTHOR / "NEW_SLOTS.json"
    slots = strict_json(slots_file)
    if (freeze.get("opportunity_id") != CODE or freeze.get("slots_file") != "NEW_SLOTS.json"
            or digest(slots_file) != freeze.get("sha256") or slots.get("opportunity_id") != CODE
            or slots.get("root_id") != source_packet["root_id"] or slots.get("source_sha256") != source_packet["source_sha256"]):
        raise ValueError("slots freeze binding mismatch")
    if set(slots) != {"opportunity_id", "root_id", "source_sha256", "slots"}:
        raise ValueError("unexpected outer slot fields")
    targets = slots["slots"]
    if not isinstance(targets, list) or not 1 <= len(targets) <= 3:
        raise ValueError("invalid slots count")
    if any(not isinstance(s, dict) or set(s) != FIELDS for s in targets):
        raise ValueError("unexpected semantic slot fields")
    for slot in targets:
        if not all(isinstance(slot[k], str) and slot[k] for k in ("slot_id", "natural_information_need", "precise_query_scope", "answer_program")):
            raise ValueError("invalid semantic slot text")
        if not all(isinstance(slot[k], list) for k in ("facts", "required_relations", "alternative_sufficient_paths", "unresolved_relations")):
            raise ValueError("invalid semantic slot list")
        for fact in slot["facts"]:
            if not isinstance(fact, dict) or set(fact) != {"fact_id", "proposition", "evidence"} or not isinstance(fact["evidence"], list):
                raise ValueError("invalid fact fields")
            if any(not isinstance(e, dict) or set(e) != {"turn_ref", "quote", "origin"} for e in fact["evidence"]):
                raise ValueError("invalid evidence fields")
    ids = [s["slot_id"] for s in targets]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid slot identifiers")
    check_semantic_identity(targets)
    checked = evidence(targets, turns(source))
    if checked["issues"] or checked["checked"] == 0:
        raise ValueError(f"source evidence check failed: {checked!r}")
    freeze_receipt = strict_json(HISTORY / "CONTROLLER_PREPARATION_FREEZE_RECEIPT.json")
    for artifact in freeze_receipt["artifacts"]:
        if digest(OLD / artifact["path"]) != artifact["sha256"]:
            raise ValueError("preparation freeze artifact hash mismatch")
    selector_contract = OLD / "contracts/ISOLATED_SELECTOR_CONTRACT.txt"
    rules = HISTORY / "SELECTOR_CONTINUATION_RULES_DRAFT.txt"
    history = HISTORY / "SELECTOR_HISTORY_VIEW.json"
    summary_freeze = strict_json(OLD.parent / "execution_20261106_v1/manifests/SELECTOR_OPERATION_SUMMARY_FREEZE.json")
    operation_summary = Path(summary_freeze["path"])
    if digest(operation_summary) != summary_freeze["sha256"]:
        raise ValueError("frozen operation summary mismatch")
    history_data = strict_json(history)
    if history_data["record_level_comparisons"]["status"] != "UNKNOWN_NOT_INDEPENDENTLY_REVIEWED_FOR_SELECTOR" or history_data["record_level_comparisons"]["records"] != []:
        raise ValueError("history uncertainty differs from frozen view")
    box = OLD / ("selector_box_" + secrets.token_hex(12))
    box.mkdir(mode=0o700, exist_ok=False)
    (box / "in").mkdir(mode=0o700, exist_ok=False)
    (box / "out").mkdir(mode=0o700, exist_ok=False)
    copied = {"source": (anon_source, "source.json"), "neutral": (anon_neutral, "neutral.json"),
              "history": (history, "history.json"), "operation_summary": (operation_summary, "operations.json"),
              "contract": (selector_contract, "contract.txt"),
              "rules": (rules, "rules.txt")}
    bindings = {}
    for label, (original, name) in copied.items():
        target = box / "in" / name
        exclusive(target, original.read_bytes())
        if digest(target) != digest(original):
            raise ValueError("copied bytes mismatch")
        bindings[label] = {"path": str(target), "sha256": digest(target)}
    projected_slots = {"slots": targets}
    slot_projection_path = box / "in/slots.json"
    exclusive(slot_projection_path, json_bytes(projected_slots))
    if strict_json(slot_projection_path)["slots"] != targets:
        raise ValueError("slot projection changed values")
    bindings["slots"] = {"path": str(slot_projection_path), "sha256": digest(slot_projection_path)}
    packet = {"schema_version": "isolated_selector_packet_v1", "inputs": bindings,
              "output_path": str(box / "out/result.json"),
              "output_requirements": "Follow the supplied contract and rules; output once using exclusive create.",
              "status": "PREPARED_PENDING_INDEPENDENT_PREFLIGHT"}
    packet_path = box / "in/packet.json"
    exclusive(packet_path, json_bytes(packet))
    if (box / "out/result.json").exists():
        raise ValueError("output path already exists")
    PRIVATE.mkdir(mode=0o700, exist_ok=False)
    receipt = {"status": "DRAFT_PENDING_INDEPENDENT_PREFLIGHT", "opportunity_id": CODE,
               "author_input_sha256": digest(OLD / "continuation_opp13_effective_packet_v1/INPUT.json"),
               "original_slots_path": str(slots_file), "original_slots_sha256": digest(slots_file),
               "slots_freeze_path": str(AUTHOR / "SLOTS_FREEZE.json"), "slots_freeze_sha256": digest(AUTHOR / "SLOTS_FREEZE.json"),
               "projection_policy": "Outer opportunity_id, root_id and source_sha256 removed; entire slots array preserved by value and order.",
               "slot_projection_sha256": digest(slot_projection_path), "slot_ids": ids,
               "quote_entries_checked": checked["checked"], "selector_packet_path": str(packet_path),
               "selector_packet_sha256": digest(packet_path), "copy_bindings": bindings,
               "source_projection_sha256": digest(anon_source), "neutral_projection_sha256": digest(anon_neutral),
               "preparation_freeze_receipt_sha256": digest(HISTORY / "CONTROLLER_PREPARATION_FREEZE_RECEIPT.json"),
               "operation_summary_freeze_sha256": digest(OLD.parent / "execution_20261106_v1/manifests/SELECTOR_OPERATION_SUMMARY_FREEZE.json"),
               "history_semantic_comparison": "UNKNOWN; empty records do not indicate absence",
               "limitations": ["Shared filesystem; no OS-level isolation verified", "Semantic and privacy audit remains independent and incomplete", "No selector has been started"],
               "selector_started": False}
    exclusive(PRIVATE / "CONTROLLER_RECEIPT.json", json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
