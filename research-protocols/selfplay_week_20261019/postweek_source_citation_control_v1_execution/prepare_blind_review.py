from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "postweek_source_citation_control_v1"
SCORING = HERE / "outputs" / "scoring"
REVIEW_DIR = HERE / "outputs" / "blind_review"


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(row: dict) -> None:
    p = Path(row["path"])
    if not p.is_file() or p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
        raise RuntimeError("HASH_BINDING_MISMATCH:" + str(p))


def bind(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    if REVIEW_DIR.exists():
        raise RuntimeError("BLIND_REVIEW_ALREADY_PREPARED")
    freeze = read(SCORING / "BLIND_SCORE_FREEZE.json")
    for key in ("raw_response_freeze", "score_file", "private_unblind_key", "request_file"):
        check(freeze[key])
    if freeze["status"] != "BLIND_MECHANICAL_SCORES_COMMITTED":
        raise RuntimeError("BLIND_MECHANICAL_COMMIT_REQUIRED")
    requests = read(PARENT / "REQUESTS.json")["requests"]
    by_id = {r["request_id"]: r for r in requests}
    unblind = read(Path(freeze["private_unblind_key"]["path"]))["mapping"]
    raw_freeze = read(Path(freeze["raw_response_freeze"]["path"]))
    raw_by_id = {}
    for row in raw_freeze["files"]:
        check(row)
        raw = read(Path(row["path"]))
        raw_by_id[raw["request_id"]] = raw
    score_by_blind = {r["blind_id"]: r for r in read(Path(freeze["score_file"]["path"]))["rows"]}
    source_map = {}
    for path in (PARENT / "source_review_packets").glob("*.json"):
        packet = read(path)
        root = packet["root_id"]
        source_map[root] = packet["source_turns"]

    grouped = {}
    for blind_id, meta in unblind.items():
        request = by_id[meta["request_id"]]
        raw = raw_by_id[meta["request_id"]]
        obj = json.loads(raw["raw_output"]) if score_by_blind[blind_id]["json_schema_valid"] else None
        source_turns = source_map[request["root_id"]]
        turn_by_ref = {t["turn_ref"]: t for t in source_turns}
        for label in request["labels"]:
            evidence = obj["item_evidence"][label] if obj is not None else {"judgment": "INVALID_OUTPUT", "turn_refs": []}
            cited_refs = evidence["turn_refs"]
            for ref in cited_refs:
                if ref not in turn_by_ref:
                    raise RuntimeError("CITED_SOURCE_REF_NOT_FOUND:" + ref)
            # Deduplicate identical semantic item + judgment + citation bundles.
            # Labels, final answers, request order and condition never enter the review packet.
            dedup_key = json.dumps({"root_id": request["root_id"], "question": request["question"],
                "item_text": request["display"][label], "judgment": evidence["judgment"],
                "turn_refs": sorted(cited_refs)}, sort_keys=True, ensure_ascii=False)
            if dedup_key not in grouped:
                grouped[dedup_key] = {
                    "review_id": secrets.token_hex(12), "root_id": request["root_id"],
                    "question": request["question"], "item_text": request["display"][label],
                    "reported_judgment": evidence["judgment"],
                    "cited_turn_refs": sorted(cited_refs),
                    "cited_source_turns": [turn_by_ref[ref] for ref in sorted(cited_refs)],
                    "members": [],
                }
            grouped[dedup_key]["members"].append({"blind_id": blind_id, "displayed_label": label})
    reviews = []
    private_members = {}
    for row in grouped.values():
        members = row.pop("members")
        private_members[row["review_id"]] = members
        reviews.append(row)
    secrets.SystemRandom().shuffle(reviews)
    if sum(len(v) for v in private_members.values()) != 144:
        raise RuntimeError("BLIND_REVIEW_ITEM_DENOMINATOR_CHANGED")
    REVIEW_DIR.mkdir(parents=True)
    by_root = {}
    for row in reviews:
        by_root.setdefault(row["root_id"], []).append(row)
    for root, rows in by_root.items():
        packet = {
            "schema_version": "source_citation_control_blind_citation_packet_v1",
            "blinded_fields": ["condition", "final_answer", "gold_order", "request_id", "display_mapping", "sampling_seed", "replicate"],
            "review_instruction": "For each review_id, decide whether the cited turns directly and sufficiently support the reported judgment about the exact item. Use full source only to resolve references or state; additional uncited facts cannot by themselves make a bad citation sufficient. Empty citations or INVALID_OUTPUT count as not sufficient. Do not infer or report final ordering. Record sufficient:boolean, semantic_judgment_correct:boolean, error_category, reason. Categories: OK, EMPTY_OR_INVALID_CITATION, SOURCE_LOCALIZATION, STATE_DISTINCTION, AMBIGUOUS_WORDING, OTHER. A correct first sufficient anchor is sufficient if the reported judgment is supported and necessary local reference context is recoverable. Review without opening any scoring keys or condition mappings.",
            "root_id": root, "source_turns": source_map[root], "review_rows": rows,
        }
        save(REVIEW_DIR / (root + ".json"), packet)
    private_path = REVIEW_DIR / "private" / "MEMBERS.json"
    save(private_path, {"review_id_to_blind_items": private_members})
    os.chmod(private_path, 0o600)
    files = [bind(p) for p in sorted(REVIEW_DIR.glob("*.json"))]
    save(REVIEW_DIR / "PACKET_FREEZE.json", {
        "status": "BLIND_REVIEW_PACKETS_COMMITTED", "item_denominator": 144,
        "distinct_review_bundles": len(reviews), "packet_bindings": files,
        "private_member_binding": bind(private_path),
        "blind_score_freeze": bind(SCORING / "BLIND_SCORE_FREEZE.json"),
    })
    print(json.dumps({"status": "BLIND_REVIEW_PACKETS_COMMITTED", "items": 144,
        "distinct_review_bundles": len(reviews), "root_packet_count": len(by_root)}))


if __name__ == "__main__":
    main()
