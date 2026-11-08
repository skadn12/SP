"""Mechanical same-root authoring preflight; no semantic repair or certification.

Full anchor text is copied from the selected root's canonical source. This does
not prove the author actually read it. QA wording is never generated or repaired.
The optional sequence runner stops at the first rejected command or missing
expected artifact. Original frozen author/review tools remain authoritative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SCENES = tuple(f"R02:{r}:S{i:02d}" for r in "AB" for i in (1, 2, 3))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_same_root(unit, root):
    unit = Path(unit).resolve()
    suffix = root.removeprefix("cc-train-episode-")
    require(root == "cc-train-episode-" + suffix and suffix.isdigit(), "CANONICAL_ROOT_REQUIRED")
    require(unit.name == "root" + suffix, "UNIT_ROOT_MISMATCH")
    path = unit / "roots" / ("root" + suffix) / "CUMULATIVE_SOURCE.json"
    source = read(path)
    require(source.get("root_id") == root, "SOURCE_ROOT_MISMATCH")
    turns = source.get("turns")
    require(isinstance(turns, list) and turns, "SOURCE_TURNS_REQUIRED")
    by_ref = {t["turn_ref"]: t for t in turns}
    require(len(by_ref) == len(turns), "DUPLICATE_TURN_REFERENCE")
    require(all(not t["turn_ref"].startswith("cc-train-episode-") or t["turn_ref"].startswith(root + ":") for t in turns), "PARENT_TURN_FROM_OTHER_ROOT")
    require(all(isinstance(t.get("speaker"), str) and isinstance(t.get("text"), str) and t["text"].strip() for t in turns), "INVALID_SOURCE_TURN")
    return path, source, by_ref


def check_anchors(plan, root, source_path, turns):
    require(plan.get("root_id", root) == root, "PLAN_ROOT_MISMATCH")
    require(plan.get("source_sha256", sha(source_path)) == sha(source_path), "PLAN_SOURCE_HASH_MISMATCH")
    slots = plan.get("slots")
    require(isinstance(slots, list) and len(slots) == 2 and {str(s["slot_id"]) for s in slots} == {"1", "2"}, "TWO_SLOTS_REQUIRED")
    for slot in slots:
        anchors = slot["anchors"]
        require(len(anchors) == 6 and len({a["anchor_id"] for a in anchors}) == 6 and len({a["turn_ref"] for a in anchors}) == 6, "SIX_DISTINCT_ANCHORS_REQUIRED")
        require(all(a["turn_ref"] in turns for a in anchors), "ANCHOR_OUTSIDE_SAME_ROOT_SOURCE")
        require({turns[a["turn_ref"]].get("scene_ref") for a in anchors} == set(SCENES), "ONE_ANCHOR_PER_SCENE_REQUIRED")
        for anchor in anchors:
            turn = turns[anchor["turn_ref"]]
            require(anchor["speaker"] == turn["speaker"] and anchor["full_turn_quote"] == turn["text"], "ANCHOR_SAME_ROOT_FULL_TURN_MISMATCH")


def build_anchors(selection, root, source_path, turns):
    require(selection.get("root_id") == root, "SELECTION_ROOT_MISMATCH")
    require("slots" in selection, "SELECTION_SLOTS_REQUIRED")
    plan = {"root_id": root, "source_sha256": sha(source_path), "slots": []}
    for slot in selection["slots"]:
        anchors = []
        for selected in slot["anchors"]:
            require(set(selected) == {"anchor_id", "turn_ref"}, "SELECTION_ONLY_IDS_AND_REFS")
            require(selected["turn_ref"] in turns, "SELECTED_REF_OUTSIDE_SAME_ROOT")
            turn = turns[selected["turn_ref"]]
            anchors.append({**selected, "speaker": turn["speaker"], "full_turn_quote": turn["text"]})
        plan["slots"].append({"slot_id": str(slot["slot_id"]), "anchors": anchors})
    check_anchors(plan, root, source_path, turns)
    return plan


def check_qa(draft, plan, root, source_path, turns):
    check_anchors(plan, root, source_path, turns)
    require(draft.get("root_id", root) == root, "QA_ROOT_MISMATCH")
    require(draft.get("source_sha256", sha(source_path)) == sha(source_path), "QA_SOURCE_HASH_MISMATCH")
    facts = draft.get("facts", [])
    require(len(facts) == 6 and len({f["fact_id"] for f in facts}) == 6, "SIX_FACTS_REQUIRED")
    require(all(f.get("source_turn_ref") in turns for f in facts), "FACT_OUTSIDE_SAME_ROOT")
    require({turns[f["source_turn_ref"]].get("scene_ref") for f in facts} == set(SCENES), "ONE_FACT_PER_SCENE_REQUIRED")
    slots = draft.get("ordering", [])
    require(len(slots) == 2 and {str(s["slot_id"]) for s in slots} == {"1", "2"}, "TWO_QA_SLOTS_REQUIRED")
    for slot in slots:
        if slot.get("failure"):
            continue
        anchors = {a["anchor_id"]: a for a in next(s for s in plan["slots"] if str(s["slot_id"]) == str(slot["slot_id"]))["anchors"]}
        items = slot["items"]
        require(len(items) == 6 and {i["item_id"] for i in items} == {f"I{i}" for i in range(1, 7)}, "SIX_ITEMS_REQUIRED")
        require({i["anchor_id"] for i in items} == set(anchors), "ITEM_FROZEN_ANCHOR_COVERAGE")
        for item in items:
            require(isinstance(item.get("support_quote"), str) and item["support_quote"] and item["support_quote"] in anchors[item["anchor_id"]]["full_turn_quote"], "ITEM_QUOTE_OUTSIDE_SAME_ROOT_ANCHOR")


def execute_checked_steps(steps, runner=subprocess.run):
    require(isinstance(steps, list) and steps, "NONEMPTY_STEPS_REQUIRED")
    completed = []
    for step in steps:
        argv = step["argv"]
        require(isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv), "ARGV_ARRAY_REQUIRED_NO_SHELL")
        for item in step.get("requires", []):
            p = Path(item["path"])
            require(p.is_file(), "PREDECESSOR_ARTIFACT_MISSING:" + str(p))
            if item.get("sha256"):
                require(sha(p) == item["sha256"], "PREDECESSOR_HASH_MISMATCH")
        result = runner(argv, cwd=step.get("cwd"), check=True, shell=False)
        require(result.returncode == 0, "PREDECESSOR_COMMAND_FAILED")
        for path in step.get("expects", []):
            require(Path(path).is_file(), "EXPECTED_OUTPUT_MISSING:" + str(path))
        completed.append(step.get("name", str(len(completed) + 1)))
    return completed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["build-anchors", "check-anchors", "check-qa", "run-sequence"])
    p.add_argument("--unit", type=Path)
    p.add_argument("--root")
    p.add_argument("--selection", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--qa", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--sequence", type=Path)
    a = p.parse_args()
    if a.mode == "run-sequence":
        require(a.sequence is not None, "SEQUENCE_FILE_REQUIRED")
        print(json.dumps({"completed": execute_checked_steps(read(a.sequence)["steps"])}))
        return
    require(a.unit is not None and a.root, "UNIT_AND_ROOT_REQUIRED")
    source_path, source, turns = load_same_root(a.unit, a.root)
    if a.mode == "build-anchors":
        require(a.selection is not None and a.out is not None, "SELECTION_AND_OUT_REQUIRED")
        require(not any((source_path.parent / n).exists() for n in ["ORDERING_ANCHOR_PLAN.json", "QA_DRAFT.json", "QA_AUTHOR_PAYLOAD.json", "QA_AUTHOR_FAILURE.json"]), "ANCHOR_OR_QA_ALREADY_ATTEMPTED")
        plan = build_anchors(read(a.selection), a.root, source_path, turns)
        with a.out.open("x", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2); f.write("\n")
    else:
        require(a.plan is not None, "PLAN_REQUIRED")
        plan = read(a.plan)
        check_anchors(plan, a.root, source_path, turns)
        if a.mode == "check-qa":
            require(a.qa is not None, "QA_REQUIRED")
            canonical = source_path.parent / "ORDERING_ANCHOR_PLAN.json"
            receipt = source_path.parent / "ORDERING_ANCHOR_RECEIPT.json"
            require(canonical.exists() and receipt.exists() and a.plan.resolve() == canonical.resolve(), "FROZEN_CANONICAL_PLAN_AND_RECEIPT_REQUIRED")
            require(read(receipt)["plan_binding"] == {"path": str(canonical.resolve()), "sha256": sha(canonical)}, "FROZEN_PLAN_BINDING_MISMATCH")
            check_qa(read(a.qa), plan, a.root, source_path, turns)
    print(json.dumps({"status": "MECHANICAL_SAME_ROOT_PREFLIGHT_PASS", "root_id": a.root, "source_sha256": sha(source_path), "semantic_correctness_certified": False, "actual_full_reading_proven": False}))


if __name__ == "__main__":
    main()
