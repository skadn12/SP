"""Native scene rules with an explicit, hash-bound historical prefix import.

Imported artifacts retain their original authors, reviewers, and timestamps.
The independent v120 prefix judgment permits reuse, never re-signs history.
No semantic judgment is made by this adapter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

from historical_fixed_import_v122 import expected_author

HERE = Path(__file__).resolve().parent
MASTER = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(MASTER / "scripts"))
spec = importlib.util.spec_from_file_location("v120_native_scene_controller", MASTER / "scripts/codex_sequential_ab.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
_native_inspect = native.inspect_root
SCENES = native.SCENES
CHECKS = native.CHECKS
TERMINAL_RECEIPTS = native.TERMINAL_RECEIPTS
receipts = native.receipts
load_config = native.load_config
root_config = native.root_config
root_folder = native.root_folder
scene_folder = native.scene_folder
load_baseline = native.load_baseline
reservation_for = native.reservation_for
append_receipt = native.append_receipt
digest = native.digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(item):
    path = Path(item["path"])
    require(binding(path) == item, "PREFIX_BOUND_FILE_CHANGED:" + str(path))
    return path


def check_freeze(batch):
    batch = Path(batch).resolve()
    master = Path(batch).resolve().parents[1]
    freeze = read(master / "MASTER_FREEZE.json")
    rows = freeze["files"]
    names = ["scene_controller_v120.py", "question_controller_v120.py", "author_v120.py", "review_v4.py",
             "review_provenance_gate_v2.py", "scene_review_provenance_v1.py", "public_anchor_v1.py", "run_guarded_questions_v2.py"]
    required = [HERE / n for n in names] + [master / "ROLE_ASSIGNMENTS.json"]
    required += [MASTER / "scripts" / n for n in ("codex_sequential_ab.py", "codex_controlled_batch.py", "codex_sequential_questions.py", "codex_scene_payload.py")]
    for path in required:
        matches = [r for r in rows if r.get("path") == str(path.resolve())]
        require(len(matches) == 1 and matches[0]["sha256"] == digest(path), "UNFROZEN_SCENE_RUNTIME:" + str(path))
    for name in names:
        require_tool_origin(Path(name).stem)
    require(Path(native.__file__).resolve() == (MASTER / "scripts/codex_sequential_ab.py").resolve(), "FOREIGN_NATIVE_SCENE_MODULE")
    for function in (_native_inspect, native.load_config, native.expected_source, native.validate_draft):
        require_loaded_function(function, MASTER / "scripts/codex_sequential_ab.py")
    for function in (native.digest, native.read, native.save, native.root_folder):
        require_loaded_function(function, MASTER / "scripts/codex_controlled_batch.py")
    check_execution_freeze(batch, master)
    return master


def require_loaded_function(function, expected):
    code = getattr(function, "__code__", None)
    require(code is not None and Path(code.co_filename).resolve() == Path(expected).resolve(),
            "FOREIGN_LOADED_RUNTIME:" + str(expected))


def frozen_tool(name):
    """Resolve before import so a cached or path-shadowed gate cannot run."""
    expected = require_tool_origin(name)
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve() == expected, "FOREIGN_LOADED_RUNTIME:" + name)
    return module


def require_tool_origin(name):
    expected = (HERE / (name + ".py")).resolve()
    spec = importlib.util.find_spec(name)
    require(spec is not None and spec.origin is not None and Path(spec.origin).resolve() == expected,
            "FOREIGN_LOADED_RUNTIME:" + name)
    return expected


def execution_input_paths(unit):
    unit = Path(unit).resolve()
    fixed = [unit / name for name in ("RUN_CONFIG.json", "SOURCE_RESERVATIONS.json", "selection.json",
                                     "PREFIX_IMPORT.json", "A_PLAN_POLICY.json")]
    fixed += [unit / "inputs" / name for name in ("BASELINE.json", "CONTINUATION_CONSTRAINTS.json", "EVENT_POOL.json",
                                                  "B_S01_EVENT.json", "B_S02_EVENT.json", "B_S03_EVENT.json")]
    fixed += [unit / "evidence/MATCHING" / name for name in ("FULL_INPUT.txt", "READ_CONTEXT.json", "MATCHING_PAYLOAD.json")]
    for path in fixed:
        require(path.is_file(), "EXECUTION_REQUIRED_INPUT_MISSING:" + str(path))
    return sorted(set(fixed + [p for p in (unit / "inputs").glob("*") if p.is_file()]))


def check_execution_freeze(unit, master):
    """Consume the post-matching freeze before any scene/QA read or write."""
    path = Path(unit) / "EXECUTION_INPUTS_FROZEN.json"
    require(path.is_file(), "EXECUTION_INPUTS_NOT_FROZEN")
    document = read(path)
    require(document.get("status") == "READY_FOR_FIXED_REMAINING_SCENES"
            and document.get("generation_started") is False, "EXECUTION_FREEZE_NOT_READY")
    require(document.get("master_freeze_binding") == binding(master / "MASTER_FREEZE.json"), "EXECUTION_MASTER_CHANGED")
    expected = {str(p.resolve()) for p in execution_input_paths(unit)}
    rows = document.get("files")
    require(isinstance(rows, list) and len(rows) == len(expected)
            and {r.get("path") for r in rows} == expected, "EXECUTION_FREEZE_FILE_SET")
    for row in rows:
        require(binding(row["path"]) == row, "EXECUTION_INPUT_CHANGED:" + row["path"])
    root = document.get("root_id")
    cfg = read(Path(unit) / "RUN_CONFIG.json")
    require([r["root_id"] if isinstance(r, dict) else r for r in cfg["roots"]] == [root], "EXECUTION_FREEZE_ROOT")
    roles = [r for r in read(master / "ROLE_ASSIGNMENTS.json")["roots"] if r["root_id"] == root]
    require(len(roles) == 1 and document.get("author") == roles[0]["author"]
            and document.get("reviewer") == roles[0]["reviewer"], "EXECUTION_FREEZE_ROLES")
    receipt = read(Path(unit) / "evidence/MATCHING/SAVE_RECEIPT.json")
    require(receipt.get("root_id") == root and receipt.get("actual_executor") == roles[0]["author"], "EXECUTION_MATCHING_RECEIPT_IDENTITY")
    for key, target in (("execution_freeze_binding", path), ("selection_binding", Path(unit) / "selection.json"),
                        ("reservations_binding", Path(unit) / "SOURCE_RESERVATIONS.json"),
                        ("raw_payload_binding", Path(unit) / "evidence/MATCHING/MATCHING_PAYLOAD.json")):
        require(receipt.get(key) == binding(target), "EXECUTION_MATCHING_RECEIPT_CHANGED:" + key)
    return document


def import_record(unit, root):
    """Check the complete prefix import lineage before trusting any old scene."""
    unit = Path(unit).resolve()
    path = unit / "PREFIX_IMPORT.json"
    require(path.is_file(), "EXPLICIT_PREFIX_IMPORT_REQUIRED")
    doc = read(path)
    require(doc["schema_version"] == "historical_prefix_import_v1" and doc["root_id"] == root, "PREFIX_IMPORT_IDENTITY")
    scenes = doc["retained_scene_refs"]
    require(scenes == list(SCENES[:len(scenes)]) and len(scenes) <= 6, "IMPORT_MUST_BE_DECLARED_CONTIGUOUS_PREFIX")
    master = unit.parents[1]
    freeze = read(master / "MASTER_FREEZE.json")
    matches = [r for r in freeze["files"] if r.get("path") == str(path)]
    require(len(matches) == 1 and matches[0]["sha256"] == digest(path), "PREFIX_IMPORT_NOT_FROZEN")
    require(doc["role_assignments_binding"] == binding(master / "ROLE_ASSIGNMENTS.json"), "PREFIX_IMPORT_ROLES_CHANGED")
    roles = [r for r in read(master / "ROLE_ASSIGNMENTS.json")["roots"] if r["root_id"] == root]
    require(len(roles) == 1, "PREFIX_IMPORT_ROLE_REQUIRED")
    audit_path = verify(doc["prefix_review_binding"])
    audit = read(audit_path)
    require(audit["actual_executor"] == roles[0]["prefix_reviewer"] != roles[0]["author"], "PREFIX_REVIEW_ROLE_MISMATCH")
    require(audit["root_id"] == root and audit["content_reuse_approved"] is True, "PREFIX_NOT_APPROVED")
    require(audit["judgment"]["decision"] == "PASS" and audit["retained_scene_refs"] == scenes, "PREFIX_REVIEW_SCOPE")
    for item in audit["input_bindings"]:
        verify(item)
    for key in ("read_context_binding", "input_render_binding", "raw_judgment_binding", "candidate_prefix_binding"):
        verify(audit[key])
    require(doc["candidate_prefix_binding"] == audit["candidate_prefix_binding"], "IMPORT_PREFIX_REVIEW_DIFFERENT")
    require(doc["prefix_read_receipt_binding"] == binding(audit_path.parent / "SAVE_RECEIPT.json"), "PREFIX_SAVE_RECEIPT_CHANGED")
    receipt = read(verify(doc["prefix_read_receipt_binding"]))
    require(receipt["output_binding"] == binding(audit_path), "PREFIX_SAVE_OUTPUT_CHANGED")
    require(receipt["actual_executor"] == roles[0]["prefix_reviewer"], "PREFIX_SAVE_WRONG_READER")
    require(audit["judgment"] == read(verify(audit["raw_judgment_binding"])), "PREFIX_RAW_JUDGMENT_CHANGED")
    context = read(verify(audit["read_context_binding"]))
    require(context["executor"] == roles[0]["prefix_reviewer"], "PREFIX_CONTEXT_WRONG_READER")
    require(context["input_bindings"] == audit["input_bindings"], "PREFIX_CONTEXT_INPUTS_MISMATCH")
    candidate = read(verify(doc["candidate_prefix_binding"]))
    require(context["turn_refs_read"] == [t["turn_ref"] for t in candidate["turns"]], "PREFIX_READ_COVERAGE")
    baseline = read(unit / "inputs/BASELINE.json")
    previous = baseline["turns"]
    require([r["scene_ref"] for r in doc["scenes"]] == scenes, "IMPORT_SCENE_SET")
    for row in doc["scenes"]:
        folder = scene_folder(unit, root, row["scene_ref"])
        require(set(row["verbatim_artifacts"]) == {"CUMULATIVE_SOURCE.json", "EXPANSION_DRAFT.json", "SOURCE_REVIEW.json"}, "IMPORTED_ARTIFACT_SET_INCOMPLETE")
        for name, item in row["verbatim_artifacts"].items():
            original = verify(item["historical_binding"])
            current = verify(item["imported_binding"])
            require(current == (folder / name).resolve(), "IMPORT_ARTIFACT_PATH")
            require(original.read_bytes() == current.read_bytes(), "IMPORT_NOT_VERBATIM")
        source = read(folder / "CUMULATIVE_SOURCE.json")
        draft = read(folder / "EXPANSION_DRAFT.json")
        historical_review = read(folder / "SOURCE_REVIEW.json")
        require(source["turns"] == previous + draft["turns"], "IMPORTED_SCENE_NOT_EXACT_PREFIX")
        require(all(t["scene_ref"] == row["scene_ref"] for t in draft["turns"]), "IMPORT_WRONG_SCENE")
        require(historical_review["decision"] == "READY", "IMPORT_REQUIRES_HISTORICAL_READY")
        require(historical_review["source_sha256"] == digest(folder / "CUMULATIVE_SOURCE.json"), "IMPORT_HISTORICAL_REVIEW_SOURCE")
        previous = source["turns"]
    require(previous == candidate["turns"], "IMPORT_FINAL_TURNS_DIFFER_FROM_APPROVED_PREFIX")
    return doc


def new_scene_english_guard(draft):
    """Block CJK text in new scenes; natural English still needs human review."""
    turns = draft.get("turns", []) if isinstance(draft, dict) else []
    for turn in turns if isinstance(turns, list) else []:
        value = turn.get("text", "") if isinstance(turn, dict) else ""
        if isinstance(value, str):
            require(not any("\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff" for char in value),
                    "NEW_SCENE_CJK_TEXT_BLOCKED_ENGLISH_PROTOCOL")


def check_new_scene_language(unit, root, scene):
    if root == "cc-train-episode-254852" and scene == "R02:A:S02":
        # Exact fixed candidate is an explicit historical re-review, not a new draft.
        expected_author(Path(unit).parents[1], unit, root, "scene", scene)
        return
    new_scene_english_guard(read(scene_folder(unit, root, scene) / "EXPANSION_DRAFT.json"))


@contextmanager
def imported_prefix_runtime():
    materialization, review = native.validate_materialization, native.validate_review

    def validate_materialization(batch, cfg, root, scene, input_path, baseline_path, baseline):
        doc = import_record(batch, root)
        if scene not in doc["retained_scene_refs"]:
            check_new_scene_language(batch, root, scene)
            return materialization(batch, cfg, root, scene, input_path, baseline_path, baseline)
        path = scene_folder(batch, root, scene) / "CUMULATIVE_SOURCE.json"
        source = read(path)
        draft = read(path.parent / "EXPANSION_DRAFT.json")
        require(source["turns"] == read(input_path)["turns"] + draft["turns"], "IMPORTED_PREDECESSOR_MISMATCH")
        return path, source

    def validate_review(folder, root, scene, source_path, source):
        unit = Path(folder).parents[3]
        doc = import_record(unit, root)
        if scene not in doc["retained_scene_refs"]:
            gate_module = frozen_tool("review_provenance_gate_v2")
            require_loaded_function(gate_module.check_scene, HERE / "review_provenance_gate_v2.py")
            gate = gate_module.check_scene(unit.parents[1], root, scene)
            require(gate["pass"], "NEW_SCENE_REVIEW_PROVENANCE_BLOCKED:" + json.dumps(gate.get("errors", []), ensure_ascii=False))
            return review(folder, root, scene, source_path, source)
        path = Path(folder) / "SOURCE_REVIEW.json"
        return path, read(path), [{"kind": "HISTORICAL_READY_ALREADY_RECORDED_NOT_RESIGNED"}]

    native.validate_materialization, native.validate_review = validate_materialization, validate_review
    try:
        yield
    finally:
        native.validate_materialization, native.validate_review = materialization, review


def inspect_root(batch, cfg, root_id, record_reviews=True):
    check_freeze(batch)
    import_record(batch, root_id)
    with imported_prefix_runtime():
        result = _native_inspect(batch, cfg, root_id, record_reviews=record_reviews)
    result["historical_prefix_explicitly_imported"] = True
    return result


def materialize_scene(batch, root_id, scene_ref):
    master = check_freeze(batch)
    doc = import_record(batch, root_id)
    require(scene_ref not in doc["retained_scene_refs"], "CANNOT_REAUTHOR_IMPORTED_SCENE")
    draft_path = scene_folder(batch, root_id, scene_ref) / "EXPANSION_DRAFT.json"
    if draft_path.exists():
        check_new_scene_language(batch, root_id, scene_ref)
        role = next(r for r in read(master / "ROLE_ASSIGNMENTS.json")["roots"] if r["root_id"] == root_id)
        require(read(draft_path).get("author") == expected_author(master, batch, root_id, "scene", scene_ref), "NEW_DRAFT_AUTHOR_NOT_ASSIGNED_OR_EXACT_IMPORTED_AUTHOR")
    inspector = native.inspect_root
    native.inspect_root = inspect_root
    try:
        return native.materialize_scene(batch, root_id, scene_ref)
    finally:
        native.inspect_root = inspector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["materialize-scene", "progress"])
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--root-id", required=True)
    parser.add_argument("--scene-ref", choices=SCENES)
    args = parser.parse_args()
    if args.action == "materialize-scene":
        require(args.scene_ref, "SCENE_REQUIRED")
        result = materialize_scene(args.batch_dir, args.root_id, args.scene_ref)
    else:
        result = inspect_root(args.batch_dir, load_config(args.batch_dir), args.root_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == "materialize-scene" and result["status"] in {"STRUCTURAL_FAILURE", "ROOT_STOPPED", "PREDECESSOR_NOT_READY"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
