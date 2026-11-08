#!/usr/bin/env python3
"""Create discovery-only native event records for the live extension pilot."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

P = Path("/root/autodl-tmp/gxy-selfplay")
W = P / "research-protocols/selfplay_week_20261019/postweek_live_extension_v1"
EVENT_DIR = W / "inputs/events"
POOL = P / "offline-deliveries/h4_external_event_source_refresh_v1/multiwoz_train_sample.json"
PROVENANCE = P / "offline-deliveries/h4_external_event_source_refresh_v1/multiwoz_train_source.json"
BLOCKED = P / "research-protocols/selfplay_week_20261019/day3_targeted_4root_v1/discovery/BLOCKED_TRAIN_IDS.json"
RESERVATION = P / "research-protocols/selfplay_week_20261019/day3_targeted_4root_v1/SOURCE_RESERVATIONS.json"
ROOTS = {
    "cc-train-episode-19708": P / "offline-deliveries/h4_codex_train_pool_v125/sources/cc-train-episode-19708.json",
    "cc-train-episode-20432": P / "offline-deliveries/h4_codex_train_pool_v125/sources/cc-train-episode-20432.json",
}

def bind(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

def pool_records() -> dict[str, tuple[dict, bytes, int, int]]:
    text = POOL.read_bytes()
    decoded = text.decode("utf-8")
    decoder = json.JSONDecoder()
    pos = decoded.index("[") + 1
    out = {}
    while True:
        while pos < len(decoded) and decoded[pos] in " \t\r\n,":
            pos += 1
        if decoded[pos] == "]":
            return out
        start = pos
        obj, pos = decoder.raw_decode(decoded, pos)
        # Re-encode the exact decoded object only for validation; source bytes are
        # taken from the original UTF-8 span below, preserving the pool record.
        raw = decoded[start:pos].encode("utf-8")
        out[obj["dialogue_id"]] = (obj, raw, len(decoded[:start].encode("utf-8")), len(decoded[:pos].encode("utf-8")))

def reservation_ids() -> set[str]:
    ids = set(json.loads(BLOCKED.read_text(encoding="utf-8")))
    # Include the latest day-3 permanent reservations, which predate this pilot
    # but were not present in the older discovery snapshot.
    x = json.loads(RESERVATION.read_text(encoding="utf-8"))
    ids.update(x.get("source_ids", []))
    ids.update(r["source_id"] for r in x.get("reservations", []) if r.get("source_id"))
    return ids

def main() -> None:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    records = pool_records()
    blocked = reservation_ids()
    choices = [
        {
            "root_id": "cc-train-episode-19708", "scene_ref": "LEX1:B:S01", "native_id": "MUL1454.json",
            "material_role": "train schedule with a requested arrival cutoff, alternative departure/arrival offers, and a separate restaurant booking; supports explicit temporal comparison without inferring a same-day itinerary.",
            "preserve_native_facts": ["Train request is London Kings Cross to Cambridge with arrival by 18:30 on Thursday.", "The record contains more than one train suggestion and explicit departure/arrival times.", "Restaurant details and booking state remain separate from the train request."],
            "ambiguities": ["Do not infer that the restaurant and train are the same journey or that an offered train was accepted unless the native turns say so.", "Keep native changes in requested route, date, and time visible."],
        },
        {
            "root_id": "cc-train-episode-19708", "scene_ref": "LEX1:B:S02", "native_id": "MUL1411.json",
            "material_role": "Saturday train-after-18:45 constraint with an offered 19:17 departure, alongside a separate restaurant reservation; supports ordering a constraint, offer, confirmation, and follow-up.",
            "preserve_native_facts": ["The train request specifies Saturday and departure after 18:45.", "The service offers a train leaving at an explicit time and the record contains booking/follow-up state.", "Restaurant booking facts are separate."],
            "ambiguities": ["Do not transfer restaurant date, party size, or booking status to the train unless explicitly stated.", "Do not turn an offer into completed travel."],
        },
        {
            "root_id": "cc-train-episode-19708", "scene_ref": "LEX1:B:S03", "native_id": "PMUL2175.json",
            "material_role": "restaurant information, attraction details, and a taxi arrival deadline; supplies a cross-service procedure with explicit time relation and state changes.",
            "preserve_native_facts": ["Restaurant lookup and attraction lookup are distinct requests.", "Taxi request contains an explicit arrival-time constraint and a booking response.", "Addresses, phone numbers, and booking state remain exactly as supplied."],
            "ambiguities": ["Do not infer travel completion or a relationship between the people in the native service dialogue.", "Do not repair inconsistent service answers by external knowledge."],
        },
        {
            "root_id": "cc-train-episode-20432", "scene_ref": "LEX1:B:S01", "native_id": "PMUL3783.json",
            "material_role": "train planning with explicit departure/arrival information combined with a hotel stay; supports temporal and procedural relations while keeping booking intent distinct from completion.",
            "preserve_native_facts": ["The train request includes route, day/time constraints, and an offered schedule.", "Hotel selection and stay dates are independent fields.", "The native booking state and any missing fields must remain explicit."],
            "ambiguities": ["Do not infer that the train and hotel are one confirmed itinerary.", "Preserve any unavailable or unconfirmed booking fields."],
        },
        {
            "root_id": "cc-train-episode-20432", "scene_ref": "LEX1:B:S02", "native_id": "SNG01208.json",
            "material_role": "arrival-deadline correction followed by a revised train offer and booking reference; supplies a natural correction/procedure sequence and explicit times.",
            "preserve_native_facts": ["The user corrects an arrival-time misunderstanding.", "The service provides a revised arrival time and a booking response with a reference.", "The final native wording about payment/booking must be retained exactly."],
            "ambiguities": ["Do not collapse the initial and revised offers into one time.", "Do not infer a departure time if it is not supplied."],
        },
        {
            "root_id": "cc-train-episode-20432", "scene_ref": "LEX1:B:S03", "native_id": "PMUL4617.json",
            "material_role": "route reversal and explicit train departure/arrival followed by a hotel reservation; supports constraint update, state transition, and cross-domain ordering.",
            "preserve_native_facts": ["The user changes the requested train direction and retains a departure-after constraint.", "The selected train has explicit departure and arrival times and a booking reference.", "Hotel stay details are separate and must not be merged into the train facts."],
            "ambiguities": ["Preserve the original mistaken direction as part of the interaction; do not silently rewrite it.", "Do not infer actual travel or hotel check-in from a booking response."],
        },
    ]
    by_id = {c["native_id"]: c for c in choices}
    assert len(by_id) == 6
    out = []
    for c in choices:
        sid = "multiwoz22:train:" + c["native_id"]
        if sid in blocked:
            raise SystemExit(f"candidate already reserved: {sid}")
        obj, raw, start, end = records[c["native_id"]]
        target = EVENT_DIR / c["native_id"]
        target.write_bytes(raw)
        assert json.loads(raw) == obj
        out.append({**c, "source_id": sid, "dataset": "MultiWOZ_2.2", "split": "train",
                    "status": "DISCOVERY_RECOMMENDATION_ONLY_NOT_RESERVED",
                    "native_record_binding": bind(target),
                    "pool_record_byte_span": {"start_inclusive": start, "end_exclusive": end},
                    "native_turn_ids": [t.get("turn_id") for t in obj.get("turns", [])],
                    "native_services": obj.get("services", []),
                    "complete_native_record_preserved": True, "full_event_utterances_read": True,
                    "no_solver_scores_used": True})
    manifest = {
        "schema_version": "live_extension_native_event_discovery_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SIX_COMPATIBLE_NATIVE_RECORDS_RECOMMENDED_NOT_RESERVED",
        "scope": "Discovery-only native Train event inventory. No source authoring, questions, solver calls, training, or formal reservation.",
        "pool_binding": bind(POOL), "pool_provenance_binding": bind(PROVENANCE),
        "blocked_snapshot_binding": bind(BLOCKED), "latest_reservation_binding": bind(RESERVATION),
        "blocked_source_ids_checked": sorted(blocked),
        "roots_read": [{"root_id": r, "source_binding": bind(p), "full_source_read": True,
                        "speakers": sorted({t.get("speaker") for t in json.loads(p.read_text()).get("turns", [])})}
                       for r, p in ROOTS.items()],
        "allocation_recommendations": out,
        "reservation_policy": "Parent must revalidate all historical reservations and permanently reserve all six exact records before authoring; failed/unexecuted reservations remain blocked.",
        "speaker_policy": "Generated scenes may use only the established speakers in each root; native records are service-event material and do not add speakers to the root.",
        "no_formal_quality_claim": True,
    }
    dest = W / "EVENT_CANDIDATE_INVENTORY.json"
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (W / "EVENT_INVENTORY_REPORT_ZH.md").write_text("""# Live extension Train 事件发现（仅候选）\n\n本清单为两个 Train root 各 3 条的原生事件候选，保留完整 MultiWOZ 记录字节和池切片绑定。候选尚未正式预留；主控必须在冻结前重新检查全局永久预留并写入正式 `SOURCE_RESERVATIONS.json`。\n\n| root | B1 | B2 | B3 |\n|---|---|---|---|\n| 19708 | MUL1454 | MUL1411 | PMUL2175 |\n| 20432 | PMUL3783 | SNG01208 | PMUL4617 |\n\n该阶段不生成对话、题目或 solver 结果。\n""", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "path": str(dest), "events": [x["source_id"] for x in out]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
