"""Unchanged native question stages using the declared v120 source adapter."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import scene_controller_v120 as scene

HERE = Path(__file__).resolve().parent
MASTER = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(MASTER / "scripts"))
spec = importlib.util.spec_from_file_location("v120_native_question_controller", MASTER / "scripts/codex_sequential_questions.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
# These three bindings change only how historical source prefixes are resolved.
native.inspect_root = scene.inspect_root
native.config = scene.load_config
native.scene_folder = scene.scene_folder


def run(unit, action):
    unit = Path(unit).resolve()
    master = scene.check_freeze(unit)
    freeze = {x["path"]: x["sha256"] for x in scene.read(master / "MASTER_FREEZE.json")["files"]}
    for path in (Path(__file__), MASTER / "scripts/codex_sequential_questions.py"):
        scene.require(freeze.get(str(path.resolve())) == scene.digest(path), "UNFROZEN_QUESTION_RUNTIME:" + str(path))
    scene.require(Path(__file__).resolve() == master / "tools/question_controller_v120.py", "FOREIGN_QUESTION_CONTROLLER")
    scene.require(Path(scene.__file__).resolve() == master / "tools/scene_controller_v120.py", "FOREIGN_SCENE_CONTROLLER")
    scene.require(Path(native.__file__).resolve() == master / "scripts/codex_sequential_questions.py", "FOREIGN_NATIVE_QUESTION_CONTROLLER")
    for function in (native.publish_assets, native.compose, native.finalize):
        scene.require_loaded_function(function, MASTER / "scripts/codex_sequential_questions.py")
    cfg = scene.load_config(unit)
    roots = [r["root_id"] if isinstance(r, dict) else r for r in cfg["roots"]]
    scene.require(len(roots) == 1, "EXACTLY_ONE_ROOT_PER_UNIT")
    root = roots[0]
    if action in {"compose", "finalize"}:
        gate_module = scene.frozen_tool("review_provenance_gate_v2")
        scene.require_loaded_function(gate_module.check, scene.HERE / "review_provenance_gate_v2.py")
        gate = gate_module.check(master, root, "asset" if action == "compose" else "public")
        scene.require(gate["pass"], "QUESTION_REVIEW_PROVENANCE_BLOCKED:" + json.dumps(gate.get("errors", [])))
    else:
        public_anchor = scene.frozen_tool("public_anchor_v1")
        scene.require_loaded_function(public_anchor.validate_plan, scene.HERE / "public_anchor_v1.py")
        folder = scene.root_folder(unit, root)
        source_path = folder / "CUMULATIVE_SOURCE.json"
        plan_path = folder / "ORDERING_ANCHOR_PLAN.json"
        draft_path = folder / "QA_DRAFT.json"
        plan, draft = scene.read(plan_path), scene.read(draft_path)
        public_anchor.validate_plan(plan, root, scene.read(source_path), scene.digest(source_path))
        role = next(r for r in scene.read(master / "ROLE_ASSIGNMENTS.json")["roots"] if r["root_id"] == root)
        receipt = scene.read(folder / "QA_AUTHOR_RECEIPT.json")
        scene.require(draft.get("author") == receipt.get("author") == receipt.get("executor") == role["author"], "QA_AUTHOR_NOT_ASSIGNED")
        scene.require(receipt.get("ordering_anchor_plan_binding") == scene.binding(plan_path), "QA_RECEIPT_ANCHOR_BINDING")
        scene.require(receipt.get("qa_draft_sha256") == scene.digest(draft_path), "QA_RECEIPT_DRAFT_BINDING")
        scene.require({str(s["slot_id"]) for s in draft["ordering"]} == {"1", "2"}, "QA_TWO_ORDERING_SLOTS")
        for slot in plan["slots"]:
            expected_cid = cfg["batch_id"] + "-root" + root.rsplit("-", 1)[-1] + "-ordering" + slot["slot_id"]
            scene.require(slot["candidate_id"] == expected_cid, "PLAN_CANDIDATE_ID_MISMATCH")
            item = next(s for s in draft["ordering"] if str(s["slot_id"]) == slot["slot_id"])
            if not item.get("failure"):
                expected = {a["item_id"]: a["anchor_id"] for a in slot["anchors"]}
                actual = {a["item_id"]: a["anchor_id"] for a in item["items"]}
                scene.require(len(item["items"]) == 6 and actual == expected, "QA_ITEMS_NOT_FROZEN_MAPPING")
    actions = {"publish": native.publish_assets, "compose": native.compose, "finalize": native.finalize}
    return actions[action](unit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["publish", "compose", "finalize"])
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.batch_dir, args.action)
