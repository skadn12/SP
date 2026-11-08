#!/usr/bin/env python3
"""Prepare a draft author packet only; never call an author or change old files."""
import json
from pathlib import Path

from check import OLD, chunks, digest, strict_json, turns

HERE = Path(__file__).resolve().parent
DEST = HERE / "opp12_draft"
CODE = "OPP27_12_c52989cb9b"


def exclusive(path, content):
    with path.open("xb") as stream:
        stream.write(content)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main():
    packet = next(x for x in strict_json(OLD / "PACKET_MANIFEST.json")["packets"]
                  if x["opportunity_id"] == CODE)
    old_input_path = Path(packet["input_path"])
    if digest(old_input_path) != packet["input_sha256"]:
        raise ValueError("original input hash mismatch")
    old_input = strict_json(old_input_path)
    if old_input["workflow"] != "OLD" or old_input["group_id"] != "OLD_G":
        raise ValueError("original schedule does not match OPP12")
    for kind in ("source", "neutral", "contract"):
        if digest(old_input[kind + "_path"]) != old_input[kind + "_sha256"]:
            raise ValueError(kind + " hash mismatch")
    old_output = Path(old_input["output_dir"])
    if old_output.exists() and any(old_output.iterdir()):
        raise ValueError("original output already exists")
    source = strict_json(old_input["source_path"])
    source_before = turns(source)
    if len(source_before) != old_input["source_turn_count"]:
        raise ValueError("source turn count mismatch")
    if set(k for k in source if k.lower() == "dataid") != {"dataID"}:
        raise ValueError("unexpected source identity schema")
    source["dataID"] = "SOURCE_12"
    neutral = strict_json(old_input["neutral_path"])
    neutral["root_id"] = "SOURCE_12"
    neutral.pop("canonical_source_binding", None)
    if turns(source) != source_before:
        raise ValueError("source projection altered turns")
    groups = chunks(source_before)
    full_contract = OLD / "contracts/ISOLATED_AUTHOR_CONTRACT.txt"
    short_contract = Path(old_input["contract_path"])
    DEST.mkdir(mode=0o700, exist_ok=False)
    output = DEST / "author_output"
    output.mkdir(mode=0o700, exist_ok=False)
    exclusive(DEST / "SOURCE.json", json_bytes(source))
    exclusive(DEST / "NEUTRAL_SOURCE_INFO.json", json_bytes(neutral))
    exclusive(DEST / "ISOLATED_AUTHOR_CONTRACT.txt", full_contract.read_bytes())
    exclusive(DEST / "OLD_INSTRUCTION.txt", short_contract.read_bytes())
    projected_input = {
        "opportunity_id": CODE, "root_id": "SOURCE_12", "workflow": "OLD",
        "source_path": str(DEST / "SOURCE.json"), "source_sha256": digest(DEST / "SOURCE.json"),
        "source_turn_count": len(source_before),
        "neutral_path": str(DEST / "NEUTRAL_SOURCE_INFO.json"),
        "neutral_sha256": digest(DEST / "NEUTRAL_SOURCE_INFO.json"),
        "contract_path": str(DEST / "ISOLATED_AUTHOR_CONTRACT.txt"),
        "contract_sha256": digest(DEST / "ISOLATED_AUTHOR_CONTRACT.txt"),
        "workflow_instruction_path": str(DEST / "OLD_INSTRUCTION.txt"),
        "workflow_instruction_sha256": digest(DEST / "OLD_INSTRUCTION.txt"),
        "candidate_id": "Q_12", "output_dir": str(output), "opportunity_ceiling": 1,
        "slots_ceiling": None, "generation_authorization": "NONE"
    }
    exclusive(DEST / "INPUT.json", json_bytes(projected_input))
    receipt = {
        "status": "DRAFT_NOT_AUTHORIZED", "source_original_sha256": old_input["source_sha256"],
        "neutral_original_sha256": old_input["neutral_sha256"],
        "original_input_sha256": packet["input_sha256"],
        "source_projection_changes": ["dataID replaced by SOURCE_12"],
        "neutral_projection_changes": ["root_id replaced by SOURCE_12", "canonical_source_binding removed"],
        "full_contract_original_sha256": digest(full_contract),
        "full_contract_copy_sha256": digest(DEST / "ISOLATED_AUTHOR_CONTRACT.txt"),
        "input_sha256": digest(DEST / "INPUT.json"),
        "source_turns_preserved": len(source_before), "source_chunks": len(groups),
        "output_directory_empty": not any(output.iterdir()),
        "limitations": ["No OS-level isolation", "Exact runtime model equality not established",
                        "Role packet and source projection require independent review",
                        "Author must visibly read source chunks; preflight does not establish that"],
        "author_invoked": False
    }
    exclusive(DEST / "CONTROLLER_RECEIPT.json", json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
