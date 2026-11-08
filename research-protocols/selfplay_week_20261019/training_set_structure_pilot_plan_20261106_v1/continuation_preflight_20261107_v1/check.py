#!/usr/bin/env python3
"""Read-only checks for the unstarted v1 opportunities. Does not launch roles."""
import argparse
import hashlib
import json
import re
from pathlib import Path

PLAN = Path(__file__).resolve().parent.parent
OLD = PLAN / "execution_20261107_v1"
EXPECTED = {12, 13, 14, 15, 16}


def strict_json(path):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate key: {key}")
            obj[key] = value
        return obj
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def turns(source):
    """Return session-local and global positions without assuming they coincide."""
    result = []
    for key, dialogue in source.items():
        if not key.endswith("_session_dialogue") or not isinstance(dialogue, list):
            continue
        session = key[:-len("_dialogue")]
        speakers = source.get(session + "_speakers")
        if not isinstance(speakers, list) or len(speakers) != len(dialogue):
            raise ValueError(f"speaker count mismatch: {key}")
        session_num = len({row["session"] for row in result}) + 1
        for local, (text, speaker) in enumerate(zip(dialogue, speakers), 1):
            if not isinstance(text, str) or not isinstance(speaker, str):
                raise ValueError(f"invalid turn: {key}")
            result.append(dict(session=session, session_num=session_num, local=local,
                               global_num=len(result) + 1, speaker=speaker, text=text))
    if not result:
        raise ValueError("source has no turns")
    return result


def resolve(ref, rows):
    """Resolve session-scoped global OR local numbering; a quote must disambiguate."""
    if not isinstance(ref, str):
        raise ValueError("turn reference is not a string")
    match = re.fullmatch(r"([^:]+):T(\d+)(?:-T(\d+))?", ref)
    if not match:
        raise ValueError(f"unsupported reference {ref!r}")
    name, first, last = match.group(1), int(match.group(2)), int(match.group(3) or match.group(2))
    if first < 1 or last < first or last - first > 1000:
        raise ValueError(f"invalid range {ref!r}")
    if re.fullmatch(r"S\d+", name):
        candidates = [r for r in rows if r["session_num"] == int(name[1:])]
    else:
        session = name.removesuffix("_dialogue")
        candidates = [r for r in rows if r["session"] == session]
    local = [r for r in candidates if first <= r["local"] <= last]
    global_ = [r for r in candidates if first <= r["global_num"] <= last]
    length = last - first + 1
    if len(local) != length:
        local = []
    if len(global_) != length:
        global_ = []
    return {r["global_num"]: r for r in local + global_}


def evidence(obj, rows):
    issues, count = [], 0
    def walk(value):
        nonlocal count
        if isinstance(value, dict):
            if "turn_ref" in value and "quote" in value:
                count += 1
                ref, quote = value["turn_ref"], value["quote"]
                try:
                    matches = resolve(ref, rows)
                    if value.get("origin") != "SOURCE.json":
                        issues.append(f"invalid or missing source origin: {ref!r}")
                    hit = [r for r in matches.values() if isinstance(quote, str) and quote and quote in r["text"]]
                    if len(hit) != 1:
                        issues.append(f"quote must identify exactly one referenced turn (found {len(hit)}): {ref!r}")
                except ValueError as exc:
                    issues.append(str(exc))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(obj)
    return {"checked": count, "issues": issues}


def chunks(rows):
    result, group = [], []
    for row in rows:
        display = {"turn_ref": f'S{row["session_num"]:02d}:T{row["global_num"]:03d}',
                   "speaker": row["speaker"], "text": row["text"], "origin": "SOURCE.json"}
        trial = group + [display]
        if len(trial) > 30 or len(json.dumps(trial, ensure_ascii=False)) > 12000:
            if not group:
                raise ValueError("single source turn exceeds chunk limit")
            result.append(group)
            group = [display]
        else:
            group = trial
        if len(json.dumps(group, ensure_ascii=False)) > 12000:
            raise ValueError("single source turn exceeds chunk limit")
    if group:
        result.append(group)
    return result


def check_inputs():
    schedule = strict_json(OLD / "SCHEDULE_RESTRICTED.json")
    manifest = strict_json(OLD / "PACKET_MANIFEST.json")
    by_id = {item["opportunity_id"]: item for item in manifest["packets"]}
    results = []
    for entry in schedule["rows"]:
        number = int(entry["opportunity_id"].split("_")[1])
        if number not in EXPECTED:
            continue
        packet = by_id[entry["opportunity_id"]]
        inp_path = Path(packet["input_path"])
        issues = []
        if digest(inp_path) != packet["input_sha256"]:
            issues.append("input hash mismatch")
        inp = strict_json(inp_path)
        for label in ("source", "neutral", "contract", "feedback_card"):
            if label + "_path" in inp and digest(inp[label + "_path"]) != inp[label + "_sha256"]:
                issues.append(label + " hash mismatch")
        source = strict_json(inp["source_path"])
        strict_json(inp["neutral_path"])
        rows = turns(source)
        if len(rows) != inp["source_turn_count"]:
            issues.append("turn count mismatch")
        groups = chunks(rows)
        output_dir = Path(inp["output_dir"])
        if output_dir.exists() and any(output_dir.iterdir()):
            issues.append("existing author output; do not launch")
        results.append({"opportunity_id": entry["opportunity_id"], "workflow": inp["workflow"],
                        "source_turns": len(rows), "read_chunks": len(groups), "issues": issues})
    if len(results) != 5:
        raise ValueError("expected exactly five scheduled opportunities")
    return {"stage": "SOURCE_BINDINGS_ONLY", "results": results,
            "pass": all(not r["issues"] for r in results),
            "not_checked": ["anonymous role packets", "selector-safe history", "slots", "selection", "candidate", "OS isolation", "model equality"],
            "launch_authorized": False}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inputs")
    review = sub.add_parser("evidence")
    review.add_argument("source", type=Path)
    review.add_argument("artifact", type=Path)
    args = parser.parse_args()
    result = check_inputs() if args.command == "inputs" else evidence(strict_json(args.artifact), turns(strict_json(args.source)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("issues") or result.get("pass") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
