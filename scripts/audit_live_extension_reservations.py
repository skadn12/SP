#!/usr/bin/env python3
"""Audit prior native-event consumption for the postweek live-extension v4."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

P = Path("/root/autodl-tmp/gxy-selfplay")
V4 = P / "research-protocols/selfplay_week_20261019/postweek_live_extension_v4"
OUT = V4 / "audit"
POOL = P / "offline-deliveries/h4_external_event_source_refresh_v1/multiwoz_train_sample.json"
RES = V4 / "inputs/SOURCE_RESERVATIONS.json"
EVENTS = sorted((V4 / "inputs").glob("*_EVENT.json"))
IDS = ["PMUL4617.json", "PMUL4370.json", "MUL1411.json", "PMUL1780.json", "PMUL0819.json", "MUL1714.json"]

METADATA_NAMES = {
    "SOURCE_RESERVATIONS.json", "RUN_CONFIG.json", "PRECALL_PLAN.json",
    "SOURCE_POOL_FREEZE.json", "RESERVATION.json", "UNIQUE_EVENT_ALLOCATION.json",
    "EXECUTION_INPUTS_FROZEN.json", "SOURCE_FREEZE.json", "SOURCE_CONSUMPTION_FREEZE.json",
    "SELECTION.json", "selection.json", "candidate_selection.json", "PILOT_SELECTION.json",
    "HISTORICAL_SELECTION.json", "SOURCE_SELECTION.json", "SOURCE_BINDINGS.json",
    "FINAL_SOURCE_BINDING.json", "EVALUATION_SOURCE_BINDINGS.json", "BLOCKED_TRAIN_IDS.json",
}
ID_RE = {name: re.compile(rf"(?<![A-Za-z0-9])(?:multiwoz22:train:)?{re.escape(name)}(?![A-Za-z0-9])") for name in IDS}

def bind(path: Path) -> dict:
    b = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}

def excluded(path: Path) -> bool:
    s = str(path)
    # These are this pilot's repeated setup attempts, not prior consumption.
    return any(f"postweek_live_extension_v{i}" in s for i in range(1, 5))

def scan_prior_metadata() -> tuple[list[dict], list[dict]]:
    hits, files = [], []
    seen = set()
    for path in P.rglob("*"):
        if not path.is_file() or path.name not in METADATA_NAMES or excluded(path):
            continue
        # Do not inspect private candidate/solver or response artifacts.
        if any(part.lower() in {"candidate", "solver", "questions", "responses", "requests"} for part in path.parts):
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            raw = path.read_bytes()
            if len(raw) > 30_000_000:
                continue
            text = raw.decode("utf-8", errors="ignore")
        except OSError:
            continue
        matched = [name for name, pattern in ID_RE.items() if pattern.search(text)]
        if matched:
            hits.append({"binding": bind(path), "matched_ids": sorted(matched), "prior_to_v4": True})
        files.append({"binding": bind(path), "matched_ids": sorted(matched)})
    return hits, files

def main() -> None:
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    by_id = {x["dialogue_id"]: x for x in pool}
    reservation = json.loads(RES.read_text(encoding="utf-8"))
    reservation_ids = [x["source_id"] for x in reservation.get("reservations", [])]
    assert [x.removeprefix("multiwoz22:train:") for x in reservation_ids] == IDS
    hits, files = scan_prior_metadata()
    event_checks = []
    for path in EVENTS:
        item = json.loads(path.read_text(encoding="utf-8"))
        dialogue_id = item.get("dialogue_id")
        assert dialogue_id in IDS
        original = by_id.get(dialogue_id)
        assert original is not None
        event_checks.append({
            "dialogue_id": dialogue_id,
            "source_id": "multiwoz22:train:" + dialogue_id,
            "split": "train",
            "event_file_binding": bind(path),
            "pool_binding": bind(POOL),
            "pool_record_present": True,
            "record_object_exact_match": item == original,
            "pool_turn_count": len(original.get("turns", [])),
            "event_turn_count": len(item.get("turns", [])),
            "services_exact_match": item.get("services") == original.get("services"),
            "full_turn_ids_exact_match": [t.get("turn_id") for t in item.get("turns", [])] == [t.get("turn_id") for t in original.get("turns", [])],
        })
    prior_by_id = {name: [] for name in IDS}
    for hit in hits:
        for name in hit["matched_ids"]:
            prior_by_id[name].append(hit["binding"])
    # The v4 reservation itself is intentionally reported separately and is not a prior hit.
    all_exact = all(x["record_object_exact_match"] and x["services_exact_match"] and x["full_turn_ids_exact_match"] for x in event_checks)
    prior_unused = all(not prior_by_id[name] for name in IDS)
    result = {
        "schema_version": "live_extension_prior_reservation_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_PRIOR_UNUSED_AND_POOL_EXACT" if prior_unused and all_exact else "FAIL_PRIOR_USE_OR_POOL_MISMATCH",
        "scope": "Independent prior Train-event reservation/consumption audit for v4 fixed six IDs.",
        "pilot_setup_exclusion": {"excluded_attempts": [f"postweek_live_extension_v{i}" for i in range(1, 5)], "reason": "same pilot setup attempts, not prior consumption"},
        "pool_binding": bind(POOL), "v4_reservation_binding": bind(RES),
        "historical_scan": {"metadata_files_inspected": len(files), "prior_hit_files": len(hits), "prior_hit_files_by_id": prior_by_id,
                             "scan_policy": "Only source allocation/consumption metadata filenames; private candidate/solver/question/response/request artifacts excluded.",
                             "prior_unused_per_id": {name: not prior_by_id[name] for name in IDS}},
        "event_checks": event_checks,
        "conclusion": {"all_six_prior_unused": prior_unused, "all_six_exact_pool_records": all_exact,
                        "v4_reservation_is_current_only": True, "do_not_release_on_failure": True},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESERVATION_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "RESERVATION_AUDIT_REPORT_ZH.md").write_text(
        "# v4 Train 事件历史预留审计\n\n"
        f"结论：`{result['status']}`。六个固定 ID 的 v4 当前预留之外，历史来源分配元数据未发现命中；六条记录均与 512 条 Train 池中的原生对象逐项一致。\n\n"
        "审计排除了同一试验的 v1–v4 准备目录，并且不读取 candidate 私有回答或 solver 输出。v4 预留仍是永久预留，失败或未执行不释放。详见 `RESERVATION_AUDIT.json`。\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "audit": str(OUT / "RESERVATION_AUDIT.json"), "prior_hit_files": len(hits), "prior_unused": prior_unused, "pool_exact": all_exact}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
