"""Create two offline, dated dialogue variants without editing source turns."""
from datetime import date, timedelta
import hashlib
import itertools
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def segment_id(turn):
    if turn.get("scene_ref"):
        return turn["scene_ref"]
    return turn["turn_ref"].split(":")[-2]


def previous_weekend(day):
    monday = day - timedelta(days=day.weekday())
    return monday - timedelta(days=2), monday - timedelta(days=1)


def check_constraint(rule, dates):
    kind = rule["kind"]
    if kind == "min_day_gap":
        return (dates[rule["later"]] - dates[rule["earlier"]]).days >= rule["days"]
    if kind == "same_date":
        return len({dates[key] for key in rule["segments"]}) == 1
    if kind == "same_month":
        return len({(dates[key].year, dates[key].month) for key in rule["segments"]}) == 1
    if kind == "same_previous_weekend":
        return len({previous_weekend(dates[key]) for key in rule["segments"]}) == 1
    if kind == "future_weekday_same_week":
        start, target = dates[rule["segment"]], date.fromisoformat(rule["target_date"])
        return (target > start and target.weekday() == rule["weekday"]
                and start.isocalendar()[:2] == target.isocalendar()[:2])
    raise ValueError(f"Unknown constraint: {kind}")


def main():
    plan = read_json(BASE / "date_plan.json")
    prepared = []
    for spec in plan["roots"]:
        source_path = Path(spec["source_path"])
        assert file_sha(source_path) == spec["source_file_sha256"]
        source = read_json(source_path)
        turns = source["turns"]
        assert source["root_id"] == spec["root_id"]
        assert len(turns) == spec["expected_turns"]
        assert len({turn["turn_ref"] for turn in turns}) == len(turns)
        groups = [(key, list(values)) for key, values in itertools.groupby(turns, key=segment_id)]
        assert [key for key, _ in groups] == [s["segment_id"] for s in spec["segments"]]
        assert len({key for key, _ in groups}) == len(groups)
        dates = {s["segment_id"]: date.fromisoformat(s["date"]) for s in spec["segments"]}
        assert list(dates.values()) == sorted(dates.values())
        by_ref = {turn["turn_ref"]: turn for turn in turns}
        quote_checks = 0
        for note in spec["constraints"] + spec["residual_notes"]:
            for evidence in note["evidence"]:
                assert evidence["quote"] in by_ref[evidence["turn_ref"]]["text"]
                quote_checks += 1
        for rule in spec["constraints"]:
            assert check_constraint(rule, dates), rule
        segments, blocks = [], []
        cursor = 0
        for segment, (key, members) in zip(spec["segments"], groups):
            assert len({(t.get("session_ref"), t.get("scene_ref")) for t in members}) == 1
            segments.append({
                "segment_id": key, "conversation_date": segment["date"],
                "date_origin": plan["date_origin"],
                "first_turn_index_1_based": cursor + 1,
                "last_turn_index_1_based": cursor + len(members),
                "first_turn_ref": members[0]["turn_ref"], "last_turn_ref": members[-1]["turn_ref"],
                "turn_count": len(members),
            })
            header = f'[Conversation date: {segment["date"]} | Session: {key}]'
            lines = [f'[{t["turn_ref"]}] {t["speaker"]}: {t["text"]}' for t in members]
            blocks.append(header + "\n" + "\n".join(lines))
            cursor += len(members)
        text = "\n\n".join(blocks) + "\n"
        assert text.count("[Conversation date:") == len(segments)
        assert [line for block in blocks for line in block.splitlines()[1:]] == [
            f'[{t["turn_ref"]}] {t["speaker"]}: {t["text"]}' for t in turns]
        output = {
            "dated_dialogue_version": plan["version"], "root_id": spec["root_id"],
            "source_identity": {
                "path": str(source_path), "file_sha256": spec["source_file_sha256"],
                "original_expanded_dialogue_version": source["expanded_dialogue_version"],
                "unchanged_turns_canonical_sha256": object_sha(turns),
            },
            "date_origin": plan["date_origin"], "date_scope": plan["date_scope"],
            "turn_count": len(turns), "segment_count": len(segments),
            "segments": segments, "turns": turns,
        }
        prepared.append((spec, output, text, quote_checks))

    results = []
    for spec, output, text, quote_checks in prepared:
        folder = BASE / spec["root_id"]
        folder.mkdir(exist_ok=True)
        json_path, text_path = folder / "dated_dialogue.json", folder / "dated_dialogue.txt"
        write_json(json_path, output)
        text_path.write_text(text, encoding="utf-8")
        assert read_json(json_path)["turns"] == read_json(Path(spec["source_path"]))["turns"]
        assert file_sha(Path(spec["source_path"])) == spec["source_file_sha256"]
        results.append({
            "root_id": spec["root_id"], "turns": output["turn_count"],
            "segments": output["segment_count"], "headers": text.count("[Conversation date:"),
            "source_unchanged": True, "all_turn_objects_unchanged": True,
            "coverage_and_order_preserved": True, "dates_valid_and_nondecreasing": True,
            "documented_calendar_constraints_checked": len(spec["constraints"]),
            "source_quotes_checked": quote_checks,
            "unresolved_note_count": len(spec["residual_notes"]),
            "files": [{"path": str(p.relative_to(BASE)), "sha256": file_sha(p)}
                      for p in (json_path, text_path)],
        })
    validation = {
        "status": "PASS_DATE_ANNOTATION_AND_SOURCE_PRESERVATION",
        "full_semantic_consistency_certified": False,
        "date_origin": plan["date_origin"], "external_api_calls": 0,
        "generation_or_scoring_run": False, "frozen_inputs_replaced": False,
        "roots": results, "total_turns": sum(r["turns"] for r in results),
        "total_headers": sum(r["headers"] for r in results),
        "plan_sha256": file_sha(BASE / "date_plan.json"),
        "script_sha256": file_sha(Path(__file__)),
    }
    write_json(BASE / "validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
