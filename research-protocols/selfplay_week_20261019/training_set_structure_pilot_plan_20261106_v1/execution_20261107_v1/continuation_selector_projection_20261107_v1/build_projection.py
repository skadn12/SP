#!/usr/bin/env python3
"""Prepare one anonymous selector source projection; never reads author output."""

import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEST = Path(__file__).resolve().parent
INPUT = BASE / "role_packets/OPP27_13_e5c5de9d35/INPUT.json"
SOURCE = BASE / "inputs/cc-train-episode-53623/SOURCE.json"
NEUTRAL = BASE / "inputs/cc-train-episode-53623/NEUTRAL_SOURCE_INFO.json"
MANIFEST = BASE / "PACKET_MANIFEST.json"
OPPORTUNITY = "OPP27_13_e5c5de9d35"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(path):
    raw = path.read_bytes()

    def unique_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj

    return json.loads(raw, object_pairs_hook=unique_pairs), raw


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_new(path, raw):
    with path.open("xb") as f:
        f.write(raw)


def leak_hits(value, where="$", hits=None):
    if hits is None:
        hits = []
    forbidden_keys = re.compile(r"^(?:root_id|group_id|arm|workflow|feedback|candidate_id|dataid|canonical_source_binding|path|source_path|neutral_path)$", re.I)
    forbidden_values = re.compile(r"cc-train-episode-\d+|OPP27_\d+|(?:^|\s)/(?:root|home|tmp)/|\b(?:OLD|NEW)_[FG]\b|\bF/G\b", re.I)
    if isinstance(value, dict):
        for key, item in value.items():
            if forbidden_keys.search(key):
                hits.append({"location": where, "kind": "identity_key"})
            leak_hits(item, where + "." + key, hits)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leak_hits(item, f"{where}[{index}]", hits)
    elif isinstance(value, str) and forbidden_values.search(value):
        hits.append({"location": where, "kind": "identity_value"})
    return hits


def main():
    packet, packet_raw = strict_json(INPUT)
    source, source_raw = strict_json(SOURCE)
    neutral, neutral_raw = strict_json(NEUTRAL)
    manifest, manifest_raw = strict_json(MANIFEST)
    matches = [row for row in manifest["packets"] if row["opportunity_id"] == OPPORTUNITY]
    if len(matches) != 1 or matches[0]["input_sha256"] != digest(packet_raw):
        raise ValueError("packet manifest binding mismatch")
    if packet["source_path"] != str(SOURCE) or packet["neutral_path"] != str(NEUTRAL):
        raise ValueError("packet input paths differ from fixed assignments")
    if packet["source_sha256"] != digest(source_raw) or packet["neutral_sha256"] != digest(neutral_raw):
        raise ValueError("source or neutral bytes hash mismatch")
    if packet["workflow"] != "NEW" or packet["opportunity_id"] != OPPORTUNITY:
        raise ValueError("wrong opportunity or workflow")
    if source.get("dataID") != "episode-53623" or neutral.get("root_id") != "cc-train-episode-53623":
        raise ValueError("source identity mismatch")
    binding = neutral.get("canonical_source_binding", {})
    if binding.get("sha256") != digest(source_raw) or binding.get("bytes") != len(source_raw):
        raise ValueError("neutral source binding mismatch")

    public_source = {key: value for key, value in source.items() if key != "dataID"}
    public_neutral = {key: value for key, value in neutral.items() if key not in {"root_id", "canonical_source_binding"}}
    if {key: value for key, value in source.items() if key != "dataID"} != public_source:
        raise ValueError("source fields differ")
    if {key: value for key, value in neutral.items() if key not in {"root_id", "canonical_source_binding"}} != public_neutral:
        raise ValueError("neutral fields differ")
    dialogue_keys = [key for key in source if key.endswith("_session_dialogue")]
    turns = sum(len(source[key]) for key in dialogue_keys)
    if turns != packet["source_turn_count"] or turns != neutral["turn_count"]:
        raise ValueError("source turn count mismatch")
    for key in dialogue_keys:
        speakers = key.removesuffix("_dialogue") + "_speakers"
        if not isinstance(source[key], list) or len(source[key]) != len(source.get(speakers, [])):
            raise ValueError("speaker alignment mismatch")
    hits = leak_hits(public_source) + leak_hits(public_neutral)
    if hits:
        raise ValueError(f"identity-pattern check failed: {hits}")

    public_dir = DEST / "selector_visible"
    public_dir.mkdir(exist_ok=False)
    source_projection_raw = encode(public_source)
    neutral_projection_raw = encode(public_neutral)
    report = {
        "status": "DRAFT_MECHANICAL_CHECKS_PASSED_INDEPENDENT_REVIEW_PENDING",
        "strict_json_and_original_bindings": True,
        "source_removed_keys": ["dataID"],
        "neutral_removed_keys": ["root_id", "canonical_source_binding"],
        "all_other_source_fields_equal": True,
        "all_other_neutral_fields_equal": True,
        "dialogue_and_speaker_alignment": True,
        "dialogue_turn_count": turns,
        "time_and_summary_fields_preserved": True,
        "neutral_risk_fields_preserved": True,
        "literal_arm_is_body_part_in_source_text": True,
        "generic_root_word_remains_in_risk_note": True,
        "identity_pattern_hits": hits,
        "privacy_limit": "Pattern checks cannot establish semantic anonymity; independent review and restricted role access are required.",
        "selector_packet_complete": False,
        "frozen_slots_available": False,
        "history_view_and_rule_independently_approved": False,
        "selector_started": False,
    }
    private = {
        "status": "CONTROLLER_ONLY_DRAFT",
        "opportunity_id": OPPORTUNITY,
        "packet_path": str(INPUT),
        "packet_sha256": digest(packet_raw),
        "manifest_path": str(MANIFEST),
        "manifest_sha256": digest(manifest_raw),
        "source_path": str(SOURCE),
        "source_sha256": digest(source_raw),
        "source_projection_sha256": digest(source_projection_raw),
        "neutral_path": str(NEUTRAL),
        "neutral_sha256": digest(neutral_raw),
        "neutral_projection_sha256": digest(neutral_projection_raw),
        "original_identity": {"dataID": source["dataID"], "root_id": neutral["root_id"]},
        "neutral_canonical_binding": binding,
        "access_policy": "Only selector_visible files may be mounted for the selector; never expose this file or its parent directory listing.",
    }
    write_new(public_dir / "SOURCE.json", source_projection_raw)
    write_new(public_dir / "NEUTRAL.json", neutral_projection_raw)
    write_new(DEST / "CONTROLLER_PRIVATE_BINDINGS.json", encode(private))
    write_new(DEST / "PROJECTION_PREFLIGHT.json", encode(report))
    print(json.dumps({"status": report["status"], "source_projection_sha256": private["source_projection_sha256"], "neutral_projection_sha256": private["neutral_projection_sha256"], "turn_count": turns}))


if __name__ == "__main__":
    main()
