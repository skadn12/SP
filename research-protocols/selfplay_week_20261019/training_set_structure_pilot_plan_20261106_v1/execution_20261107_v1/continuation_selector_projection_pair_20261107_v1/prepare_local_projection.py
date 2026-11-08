#!/usr/bin/env python3
"""Create local selector source projections without reading author outputs."""

import hashlib
import json
import os
import re
import secrets
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PREVIOUS = Path(__file__).resolve().parent / "CONTROLLER_PRIVATE_BINDINGS.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    raw = path.read_bytes()

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=unique), raw


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def write_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def hits(value):
    key_pattern = re.compile(r"^(?:root_id|group_id|arm|workflow|feedback|candidate_id|dataid|canonical_source_binding|path|source_path|neutral_path)$", re.I)
    value_pattern = re.compile(r"cc-train-episode-\d+|OPP27_\d+|(?:^|\s)/(?:root|home|tmp)/|\b(?:OLD|NEW)_[FG]\b|\bF/G\b", re.I)
    if isinstance(value, dict):
        return any(key_pattern.search(key) or hits(item) for key, item in value.items())
    if isinstance(value, list):
        return any(hits(item) for item in value)
    return isinstance(value, str) and bool(value_pattern.search(value))


def main():
    prior, _ = read_json(PREVIOUS)
    manifest, manifest_raw = read_json(BASE / "PACKET_MANIFEST.json")
    if sha(manifest_raw) != prior["manifest_sha256"] or len(prior["bindings"]) != 2:
        raise ValueError("manifest mismatch or unexpected binding count")
    prepared = []
    for entry in prior["bindings"]:
        opportunity = entry["opportunity_id"]
        packet_path = BASE / "role_packets" / opportunity / "INPUT.json"
        packet, packet_raw = read_json(packet_path)
        source_path = Path(entry["controller_original_source_path"])
        neutral_path = Path(entry["controller_original_neutral_path"])
        source, source_raw = read_json(source_path)
        neutral, neutral_raw = read_json(neutral_path)
        rows = [row for row in manifest["packets"] if row["opportunity_id"] == opportunity]
        if len(rows) != 1 or rows[0]["input_sha256"] != sha(packet_raw) or sha(packet_raw) != entry["controller_original_packet_sha256"]:
            raise ValueError("packet binding mismatch")
        if packet["opportunity_id"] != opportunity or packet["workflow"] != "NEW":
            raise ValueError("unexpected opportunity or workflow")
        if packet["source_path"] != str(source_path) or packet["neutral_path"] != str(neutral_path):
            raise ValueError("packet path mismatch")
        if sha(source_raw) != packet["source_sha256"] or sha(source_raw) != entry["controller_original_source_sha256"]:
            raise ValueError("source hash mismatch")
        if sha(neutral_raw) != packet["neutral_sha256"] or sha(neutral_raw) != entry["controller_original_neutral_sha256"]:
            raise ValueError("neutral hash mismatch")
        binding = neutral["canonical_source_binding"]
        if binding["path"] != str(source_path) or binding["sha256"] != sha(source_raw) or binding["bytes"] != len(source_raw):
            raise ValueError("canonical binding mismatch")
        if source["dataID"] != entry["source_identity_removed"] or neutral["root_id"] != entry["neutral_identity_removed"]:
            raise ValueError("identity mismatch")
        visible_source = {key: value for key, value in source.items() if key != "dataID"}
        visible_neutral = {key: value for key, value in neutral.items() if key not in {"root_id", "canonical_source_binding"}}
        if hits(visible_source) or hits(visible_neutral):
            raise ValueError("identity pattern found")
        dialogue = [key for key in source if key.endswith("_session_dialogue")]
        turns = sum(len(source[key]) for key in dialogue)
        if turns != packet["source_turn_count"] or turns != neutral["turn_count"]:
            raise ValueError("turn count mismatch")
        if any(not isinstance(source[key], list) or len(source[key]) != len(source.get(key.removesuffix("_dialogue") + "_speakers", [])) for key in dialogue):
            raise ValueError("speaker alignment mismatch")
        source_out, neutral_out = encode(visible_source), encode(visible_neutral)
        if sha(source_out) != entry["selector_source_sha256"] or sha(neutral_out) != entry["selector_neutral_sha256"]:
            raise ValueError("previous projection hash mismatch")
        prepared.append((entry, packet_path, source_out, neutral_out, turns))

    controller = BASE / ("controller_" + secrets.token_hex(16))
    controller.mkdir(mode=0o700, exist_ok=False)
    result = []
    for entry, packet_path, source_out, neutral_out, turns in prepared:
        visible = BASE / ("box_" + secrets.token_hex(16))
        visible.mkdir(mode=0o700, exist_ok=False)
        source_dest, neutral_dest = visible / "source.json", visible / "neutral.json"
        write_new(source_dest, source_out)
        write_new(neutral_dest, neutral_out)
        if sha(source_dest.read_bytes()) != sha(source_out) or sha(neutral_dest.read_bytes()) != sha(neutral_out):
            raise ValueError("written projection hash mismatch")
        visible_source, _ = read_json(source_dest)
        visible_neutral, _ = read_json(neutral_dest)
        original_source, _ = read_json(Path(entry["controller_original_source_path"]))
        original_neutral, _ = read_json(Path(entry["controller_original_neutral_path"]))
        if visible_source != {key: value for key, value in original_source.items() if key != "dataID"} or visible_neutral != {key: value for key, value in original_neutral.items() if key not in {"root_id", "canonical_source_binding"}}:
            raise ValueError("written projection differs from original")
        result.append({"opportunity_id": entry["opportunity_id"], "packet_path": str(packet_path), "packet_sha256": entry["controller_original_packet_sha256"], "source_path": entry["controller_original_source_path"], "source_sha256": entry["controller_original_source_sha256"], "neutral_path": entry["controller_original_neutral_path"], "neutral_sha256": entry["controller_original_neutral_sha256"], "selector_source_path": str(source_dest), "selector_source_sha256": sha(source_out), "selector_neutral_path": str(neutral_dest), "selector_neutral_sha256": sha(neutral_out), "turns": turns, "speaker_alignment": True, "source_removed_keys": ["dataID"], "neutral_removed_keys": ["root_id", "canonical_source_binding"], "other_fields_equal": True, "identity_pattern_hits": []})
    report = {"status": "LOCAL_PROJECTIONS_PREPARED_INDEPENDENT_REVIEW_PENDING", "manifest_sha256": sha(manifest_raw), "bindings": result, "limitations": ["Pattern checks do not prove semantic anonymity.", "Shared filesystem and permissions do not provide OS-level isolation from the same user.", "These projections are not complete role packets or authorization to invoke a selector."], "selector_started": False}
    report_path = controller / "CONTROLLER_PRIVATE_BINDINGS.json"
    write_new(report_path, encode(report))
    print(json.dumps({"controller_bindings": str(report_path), "bindings_sha256": sha(report_path.read_bytes()), "projection_count": len(result)}))


if __name__ == "__main__":
    main()
