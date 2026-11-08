#!/usr/bin/env python3
"""Create two anonymous source projections without opening author outputs."""

import hashlib
import json
import os
import re
import secrets
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEST = Path(__file__).resolve().parent
SPECS = (
    ("OPP27_15_33eefb12b7", "94052"),
    ("OPP27_16_9d3abd2fbe", "98366"),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
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
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def exclusive_write(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)


def scan(value, location="$", hits=None):
    if hits is None:
        hits = []
    identity_keys = re.compile(r"^(?:dataid|root_id|group_id|arm|workflow|feedback|candidate_id|canonical_source_binding|path|source_path|neutral_path)$", re.I)
    identity_values = re.compile(r"cc-train-episode-\d+|OPP27_\d+|(?:^|\s)/(?:root|home|tmp)/|\b(?:OLD|NEW)_[FG]\b|\bF/G\b", re.I)
    if isinstance(value, dict):
        for key, item in value.items():
            if identity_keys.fullmatch(key):
                hits.append({"location": location, "kind": "identity_key"})
            scan(item, location + "." + key, hits)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            scan(item, f"{location}[{index}]", hits)
    elif isinstance(value, str) and identity_values.search(value):
        hits.append({"location": location, "kind": "identity_value"})
    return hits


def main():
    manifest, manifest_raw = load(BASE / "PACKET_MANIFEST.json")
    bindings = []
    for opportunity, source_id in SPECS:
        packet_path = BASE / "role_packets" / opportunity / "INPUT.json"
        packet, packet_raw = load(packet_path)
        rows = [r for r in manifest["packets"] if r["opportunity_id"] == opportunity]
        if len(rows) != 1 or rows[0]["input_sha256"] != sha(packet_raw):
            raise ValueError("packet binding failure")
        source_path = BASE / "inputs" / ("cc-train-episode-" + source_id) / "SOURCE.json"
        neutral_path = source_path.with_name("NEUTRAL_SOURCE_INFO.json")
        source, source_raw = load(source_path)
        neutral, neutral_raw = load(neutral_path)
        if packet["source_path"] != str(source_path) or packet["neutral_path"] != str(neutral_path):
            raise ValueError("source path mismatch")
        if sha(source_raw) != packet["source_sha256"] or sha(neutral_raw) != packet["neutral_sha256"]:
            raise ValueError("source bytes mismatch")
        if source["dataID"] != "episode-" + source_id or neutral["root_id"] != "cc-train-episode-" + source_id:
            raise ValueError("source identity mismatch")
        canonical = neutral["canonical_source_binding"]
        if canonical["path"] != str(source_path) or canonical["sha256"] != sha(source_raw) or canonical["bytes"] != len(source_raw):
            raise ValueError("canonical binding mismatch")
        public_source = {k: v for k, v in source.items() if k != "dataID"}
        public_neutral = {k: v for k, v in neutral.items() if k not in {"root_id", "canonical_source_binding"}}
        keys = [k for k in source if k.endswith("_session_dialogue")]
        count = sum(len(source[k]) for k in keys)
        if count != packet["source_turn_count"] or count != neutral["turn_count"]:
            raise ValueError("turn count mismatch")
        for key in keys:
            if len(source[key]) != len(source[key.removesuffix("_dialogue") + "_speakers"]):
                raise ValueError("speaker alignment mismatch")
        hits = scan(public_source) + scan(public_neutral)
        if hits:
            raise ValueError(f"identity pattern matches: {hits}")
        directory = Path("/tmp") / ("q" + secrets.token_hex(16))
        directory.mkdir(mode=0o700, exist_ok=False)
        src = directory / "source.json"
        neu = directory / "neutral.json"
        exclusive_write(src, encode(public_source))
        exclusive_write(neu, encode(public_neutral))
        actual_source, actual_source_raw = load(src)
        actual_neutral, actual_neutral_raw = load(neu)
        if actual_source != public_source or actual_neutral != public_neutral:
            raise ValueError("written projection not equivalent")
        bindings.append({
            "opportunity_id": opportunity,
            "controller_original_source_path": str(source_path),
            "controller_original_source_sha256": sha(source_raw),
            "controller_original_neutral_path": str(neutral_path),
            "controller_original_neutral_sha256": sha(neutral_raw),
            "controller_original_packet_sha256": sha(packet_raw),
            "selector_source_path": str(src),
            "selector_source_sha256": sha(actual_source_raw),
            "selector_neutral_path": str(neu),
            "selector_neutral_sha256": sha(actual_neutral_raw),
            "source_identity_removed": source["dataID"],
            "neutral_identity_removed": neutral["root_id"],
            "neutral_private_canonical_binding": canonical,
            "source_removed_keys": ["dataID"],
            "neutral_removed_keys": ["root_id", "canonical_source_binding"],
            "all_remaining_source_fields_equal": actual_source == public_source,
            "all_remaining_neutral_fields_equal": actual_neutral == public_neutral,
            "dialogue_turn_count": count,
            "speaker_alignment_pass": True,
            "identity_pattern_hits": hits,
        })
    report = {
        "status": "PROJECTIONS_PREPARED_INDEPENDENT_REVIEW_PENDING",
        "visibility": "CONTROLLER_ONLY",
        "manifest_sha256": sha(manifest_raw),
        "bindings": bindings,
        "scope": "Only original role inputs, original source and neutral, and original packet manifest were read. No author output, feedback card or question was read.",
        "limitations": [
            "Pattern scanning does not prove semantic anonymity.",
            "Temporary paths and permissions do not provide OS isolation from another process under the same user.",
            "The projections are not complete role packets and do not authorize role invocation.",
        ],
        "author_started_by_this_preparation": False,
        "selector_started": False,
    }
    exclusive_write(DEST / "CONTROLLER_PRIVATE_BINDINGS.json", encode(report))
    print(json.dumps({"status": report["status"], "bindings": [{"opportunity_id": b["opportunity_id"], "source_sha256": b["selector_source_sha256"], "neutral_sha256": b["selector_neutral_sha256"], "turn_count": b["dialogue_turn_count"]} for b in bindings]}))


if __name__ == "__main__":
    main()
