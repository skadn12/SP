"""Immutable author I/O for v120 with explicit historical prefix reuse; never calls a model.

The executor must be the root's assigned author. A save invocation attests that
the executor read the complete archived rendering; this is a recorded assertion,
not a programmatic proof of cognition or semantic correctness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MASTER = HERE.parent
PROJECT = MASTER.parent.parent
sys.path.insert(0, str(MASTER / "scripts"))
import codex_scene_payload as serializer
import scene_controller_v120 as sequential
import public_anchor_v1 as public_anchor
serializer.inspect_root = sequential.inspect_root
serializer.load_config = sequential.load_config
serializer.load_baseline = sequential.load_baseline
serializer.reservation_for = sequential.reservation_for
serializer.scene_folder = sequential.scene_folder


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def write_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def save(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def resolve(unit, path):
    path = Path(path)
    return path if path.is_absolute() else unit / path


def checked_binding(unit, record):
    path = resolve(unit, record["path"]).resolve()
    require(path.is_file() and digest(path) == record["sha256"], "BINDING_CHANGED:" + str(path))
    return binding(path)


def assigned(master, root, executor):
    master = Path(master).resolve()
    public_anchor.validate_runtime(master, __file__, {"scene_controller_v120.py": sequential.check_freeze,
                                                    "public_anchor_v1.py": public_anchor.validate_plan})
    serializer_path = (master / "scripts/codex_scene_payload.py").resolve()
    require(Path(serializer.__file__).resolve() == serializer_path, "FOREIGN_SCENE_SERIALIZER")
    sequential.require_loaded_function(serializer.write_new, serializer_path)
    sequential.require_loaded_function(serializer.build_draft, serializer_path)
    require(re.fullmatch(r"cc-train-episode-\d+", root) is not None, "INVALID_ROOT")
    path = master / "ROLE_ASSIGNMENTS.json"
    rows = read(path)["roots"]
    require(len({r["root_id"] for r in rows}) == len(rows), "DUPLICATE_ASSIGNED_ROOT")
    matches = [r for r in rows if r["root_id"] == root]
    require(len(matches) == 1, "ROOT_NOT_ASSIGNED")
    role = matches[0]
    require(isinstance(executor, str) and executor == role["author"], "EXECUTOR_NOT_ASSIGNED_AUTHOR")
    require(role["author"] != role["reviewer"], "AUTHOR_REVIEWER_MUST_DIFFER")
    unit = master / "units" / ("root" + root.rsplit("-", 1)[1])
    sequential.check_freeze(unit)
    cfg = sequential.load_config(unit)
    sequential.root_config(cfg, root)
    return unit, cfg, role, binding(path)


def constraints_for(unit, root, role):
    path = unit / "inputs/CONTINUATION_CONSTRAINTS.json"
    document = read(path)
    require(document["root_id"] == root, "CONSTRAINT_ROOT_MISMATCH")
    require(document["decision"] in {"PASS", "WARNING"}, "BASELINE_NOT_ELIGIBLE")
    constraints = document["continuation_constraints"]
    require(isinstance(constraints, list) and constraints and
            all(isinstance(c, str) and c.strip() for c in constraints), "ALL_CONTINUATION_CONSTRAINTS_REQUIRED")
    base = checked_binding(unit, document["baseline_binding"])
    require(base == binding(unit / "inputs/BASELINE.json"), "CONSTRAINT_BASELINE_BINDING_MISMATCH")
    reviews = document["review_bindings"]
    require(len(reviews) == 2, "TWO_BASELINE_REVIEWS_REQUIRED")
    actual = []
    for record in reviews:
        checked = checked_binding(unit, record)
        review = read(checked["path"])
        reviewer = record.get("reviewer", review.get("reviewer"))
        require(reviewer and reviewer == review.get("reviewer"), "BASELINE_REVIEW_IDENTITY_MISMATCH")
        actual.append(reviewer)
    require(sorted(actual) == sorted(role["baseline_reviewers"]) and len(set(actual)) == 2,
            "BASELINE_REVIEWERS_DIFFER_FROM_ASSIGNMENT")
    return path, document


def state_for(unit, cfg, root):
    state = sequential.inspect_root(unit, cfg, root, record_reviews=False)
    require(not state["root_stopped"], "ROOT_STOPPED")
    return state


def scene_source(unit, cfg, root, scene):
    require(scene in sequential.SCENES, "INVALID_SCENE")
    state = state_for(unit, cfg, root)
    index = sequential.SCENES.index(scene)
    require(state["scenes"][index]["status"] == "NOT_STARTED", "SCENE_ALREADY_ATTEMPTED_OR_NOT_READY")
    require(not index or state["scenes"][index - 1]["status"] == "READY", "PREDECESSOR_NOT_READY")
    source = sequential.load_baseline(unit, cfg, root)[0] if not index else (
        sequential.scene_folder(unit, root, sequential.SCENES[index - 1]) / "CUMULATIVE_SOURCE.json")
    return Path(source), state


def native_for(unit, cfg, root, scene):
    reservation = sequential.reservation_for(unit, cfg, root, scene)
    if reservation is None:
        return None
    native_path = Path(reservation["source_item_path"])
    selection_path = resolve(unit, reservation.get("selection_path", "selection.json")).resolve()
    require(selection_path.is_file(), "B_SELECTION_REQUIRED")
    if reservation.get("selection_sha256"):
        require(digest(selection_path) == reservation["selection_sha256"], "B_SELECTION_CHANGED")
    selection = read(selection_path)
    require(selection.get("root_id", root) == root, "B_SELECTION_ROOT_MISMATCH")
    matches = [r for r in selection["allocations"] if r["scene_ref"] == scene]
    require(len(matches) == 1 and matches[0]["source_id"] == reservation["source_id"], "B_SELECTION_SLOT_MISMATCH")
    event = read(native_path)
    require(isinstance(event.get("turns"), list) and event["turns"], "B_NATIVE_TURNS_REQUIRED")
    return {"reservation": reservation, "native_event_binding": binding(native_path),
            "selection_binding": binding(selection_path), "event": event, "selection": selection}


def qa_source(unit, cfg, root):
    state = state_for(unit, cfg, root)
    require(state["full_expansion_complete"] is True, "ALL_SIX_SCENES_MUST_BE_READY")
    scene = sequential.scene_folder(unit, root, sequential.SCENES[-1])
    destination = sequential.root_folder(unit, root)
    for name in ("CUMULATIVE_SOURCE.json", "SOURCE_REVIEW.json"):
        source, target = scene / name, destination / name
        if target.exists():
            require(target.read_bytes() == source.read_bytes(), "ROOT_SUMMARY_COPY_DIFFERENT:" + name)
        else:
            write_bytes(target, source.read_bytes())
    receipt = destination / "SOURCE_AGGREGATION_RECEIPT.json"
    value = {"root_id": root, "kind": "VERBATIM_B3_COPY_NO_NEW_SEMANTIC_REVIEW",
             "source_from": binding(scene / "CUMULATIVE_SOURCE.json"),
             "source_to": binding(destination / "CUMULATIVE_SOURCE.json"),
             "review_from": binding(scene / "SOURCE_REVIEW.json"),
             "review_to": binding(destination / "SOURCE_REVIEW.json")}
    if receipt.exists():
        existing = read(receipt)
        require(all(existing[k] == v for k, v in value.items()), "ROOT_SUMMARY_RECEIPT_CHANGED")
    else:
        save(receipt, {**value, "created_at_utc": now()})
    return destination / "CUMULATIVE_SOURCE.json"


def context_paths(unit, stage):
    return (unit / "evidence" / ("AUTHOR_" + stage + "_READ_CONTEXT.json"),
            unit / "evidence" / ("AUTHOR_" + stage + "_FULL_READ.md"))


def make_read(master, root, executor, scene=None):
    unit, cfg, role, roles_binding = assigned(master, root, executor)
    constraint_path, document = constraints_for(unit, root, role)
    selection_path = unit / "selection.json"
    selection = read(selection_path)
    require(selection.get("root_id") == root, "SELECTION_ROOT_MISMATCH")
    require([p["scene_ref"] for p in selection["a_continuations"]] == list(sequential.SCENES[:3]), "A_SELECTION_SEQUENCE")
    if scene:
        source_path, _ = scene_source(unit, cfg, root, scene)
        native = native_for(unit, cfg, root, scene)
        stage = scene.replace(":", "_")
    else:
        source_path, native, stage = qa_source(unit, cfg, root), None, "QA"
        require(not (sequential.root_folder(unit, root) / "QA_DRAFT.json").exists(), "QA_ALREADY_ATTEMPTED")
    source = read(source_path)
    ctx_path, render_path = context_paths(unit, stage)
    require(not ctx_path.exists() and not render_path.exists(), "AUTHOR_READ_ALREADY_ARCHIVED")
    started = now()
    context = {"schema_version": "v120_author_read_v1", "root_id": root, "author": executor,
        "executor": executor, "stage": stage, "scene_ref": scene, "started_at_utc": started,
        "source_binding": binding(source_path), "constraints_binding": binding(constraint_path),
        "selection_binding": binding(selection_path),
        "role_assignments_binding": roles_binding, "run_config_binding": binding(unit / "RUN_CONFIG.json"),
        "turn_refs_read": [t["turn_ref"] for t in source["turns"]],
        "all_continuation_constraints_read": document["continuation_constraints"],
        "native_event_full_read": bool(native), "source_semantic_certification_by_program": False}
    sections = ["# Complete author input", "", "EXECUTOR " + executor,
                "CONSTRAINTS_FULL " + json.dumps(document, ensure_ascii=False), "", "SOURCE_FULL"]
    sections.extend(t["turn_ref"] + " | " + t["speaker"] + ": " + t["text"] for t in source["turns"])
    sections.extend(["", "FROZEN_SELECTION_FULL " + json.dumps(selection, ensure_ascii=False)])
    if native:
        context.update(native_event_binding=native["native_event_binding"], selection_binding=native["selection_binding"],
            source_id=native["reservation"]["source_id"],
            native_turn_ids_read=[t.get("turn_id", str(i)) for i, t in enumerate(native["event"]["turns"])])
        sections.extend(["", "NATIVE_EVENT_FULL " + json.dumps(native["event"], ensure_ascii=False),
                         "SELECTION_FULL " + json.dumps(native["selection"], ensure_ascii=False)])
    rendered = "\n".join(sections) + "\n"
    write_bytes(render_path, rendered.encode("utf-8"))
    context.update(render_binding=binding(render_path), full_input_render_archived_at_utc=now())
    save(ctx_path, context)
    print(rendered, end="")
    print("AUTHOR_READ_CONTEXT", ctx_path)
    return context


def checked_context(master, root, executor, scene=None):
    unit, cfg, role, roles_binding = assigned(master, root, executor)
    constraint_path, document = constraints_for(unit, root, role)
    stage = scene.replace(":", "_") if scene else "QA"
    ctx_path, _ = context_paths(unit, stage)
    context = read(ctx_path)
    require(context.get("selection_binding") == binding(unit / "selection.json"), "SELECTION_CHANGED_SINCE_READ")
    require(context["executor"] == context["author"] == executor and context["root_id"] == root
            and context["scene_ref"] == scene, "AUTHOR_READ_IDENTITY_MISMATCH")
    require(context["role_assignments_binding"] == roles_binding, "ROLE_ASSIGNMENTS_CHANGED_SINCE_READ")
    require(context["constraints_binding"] == binding(constraint_path), "CONSTRAINTS_CHANGED_SINCE_READ")
    require(context["all_continuation_constraints_read"] == document["continuation_constraints"], "CONSTRAINT_COVERAGE_CHANGED")
    require(context["run_config_binding"] == binding(unit / "RUN_CONFIG.json"), "RUN_CONFIG_CHANGED_SINCE_READ")
    source_path = scene_source(unit, cfg, root, scene)[0] if scene else qa_source(unit, cfg, root)
    require(context["source_binding"] == binding(source_path), "SOURCE_CHANGED_SINCE_READ")
    require(context["turn_refs_read"] == [t["turn_ref"] for t in read(source_path)["turns"]], "FULL_SOURCE_COVERAGE_CHANGED")
    checked_binding(unit, context["render_binding"])
    if scene and ":B:" in scene:
        native = native_for(unit, cfg, root, scene)
        require(context["native_event_binding"] == native["native_event_binding"]
                and context["selection_binding"] == native["selection_binding"], "B_INPUT_CHANGED_SINCE_READ")
    return unit, cfg, context, ctx_path, source_path


def receipt_from(context, ctx_path):
    return {"root_id": context["root_id"], "author": context["author"], "executor": context["executor"],
        "scene_ref": context["scene_ref"], "started_at_utc": context["started_at_utc"],
        "full_source_read": True, "full_source_read_attested_at_utc": now(),
        "turn_refs_read": context["turn_refs_read"],
        "all_continuation_constraints_read": context["all_continuation_constraints_read"],
        "constraints_binding": context["constraints_binding"], "input_source_sha256": context["source_binding"]["sha256"],
        "read_context_binding": binding(ctx_path), "full_render_binding": context["render_binding"],
        "role_assignments_binding": context["role_assignments_binding"],
        "full_read_claim_basis": "Executor save invocation attests complete archived input read; not a mechanical proof of cognition.",
        "backend": "CURRENT_CODEX_SESSION", "exact_model_id": None, "usage_tokens": None,
        "billed_cost": None, "sampling_parameters": None, "attempt": 1, "semantic_repairs": 0,
        "source_replacements": 0, "current_solver_results_read": False, "training_started": False,
        "external_semantic_certification": False,
        **{k: context[k] for k in ("native_event_full_read", "native_event_binding", "selection_binding", "source_id", "native_turn_ids_read") if k in context}}


def scene_save(master, root, executor, scene, payload_path):
    unit, cfg, context, ctx_path, source_path = checked_context(master, root, executor, scene)
    folder = sequential.scene_folder(unit, root, scene)
    raw_target = folder / "AUTHOR_PAYLOAD.json"
    prebinding = folder / "PAYLOAD_SERIALIZATION_BINDING.json"
    receipt_path = folder / "AUTHOR_RECEIPT.json"
    require(not any(p.exists() for p in (raw_target, prebinding, receipt_path, folder / "EXPANSION_DRAFT.json")),
            "FIRST_SCENE_PAYLOAD_ALREADY_SAVED")
    payload_path = Path(payload_path).resolve()
    raw = payload_path.read_bytes()
    write_bytes(raw_target, raw)
    serializer_path = Path(serializer.__file__).resolve()
    save(prebinding, {"root_id": root, "scene_ref": scene, "author": executor,
        "created_at_utc": now(), "raw_payload_source_path": str(payload_path),
        "payload_binding": binding(raw_target), "serializer_binding": binding(serializer_path),
        "author_tool_binding": binding(__file__), "input_source_binding": binding(source_path),
        "read_context_binding": binding(ctx_path), "saved_before_draft": True})
    try:
        payload = json.loads(raw)
        require(isinstance(payload, dict), "SCENE_PAYLOAD_MUST_BE_OBJECT")
        require("turns" in payload and "messages" not in payload, "SCENE_PAYLOAD_REQUIRES_TURNS")
        require(set(payload).issubset({"turns", "state_notes", "event_mapping"}), "SCENE_PAYLOAD_FIELDS")
        normalized = {"messages": payload["turns"], **{k: payload[k] for k in ("state_notes", "event_mapping") if k in payload}}
        draft_path = serializer.write_new(unit, root, scene, executor, normalized)
        require(prebinding.stat().st_mtime_ns <= Path(draft_path).stat().st_mtime_ns, "BINDING_NOT_BEFORE_DRAFT")
        result = sequential.materialize_scene(unit, root, scene)
        require(result["status"] == "STRUCTURAL_PASS_ONLY", "MATERIALIZATION_NOT_STRUCTURAL_PASS")
    except Exception as exc:
        failure = {"root_id": root, "scene_ref": scene, "author": executor, "created_at_utc": now(),
                   "status": "FIRST_AUTHOR_ATTEMPT_FAILED", "error": str(exc), "payload_binding": binding(raw_target)}
        save(folder / "AUTHOR_FAILURE.json", failure)
        if not any(r["status"] in sequential.TERMINAL_RECEIPTS for r in sequential.receipts(folder)):
            sequential.append_receipt(folder, "STRUCTURAL_FAILURE", root_id=root, scene_ref=scene, reason=str(exc))
        raise
    receipt = {**receipt_from(context, ctx_path), "finished_at_utc": now(),
        "payload_binding": binding(raw_target), "payload_sha256": digest(raw_target),
        "serializer_binding": binding(serializer_path), "serializer_path": str(serializer_path),
        "serializer_sha256": digest(serializer_path), "payload_serialization_binding": binding(prebinding),
        "draft_sha256": digest(draft_path), "cumulative_source_binding": binding(folder / "CUMULATIVE_SOURCE.json"),
        "timing_scope": "Wall clock from complete input rendering to first immutable payload, serialization and materialization; includes author work and scheduling, not service latency or billed usage."}
    save(receipt_path, receipt)
    print(json.dumps({"root_id": root, "scene_ref": scene, "status": result["status"], "receipt": str(receipt_path)}))


def validate_anchor(plan, source):
    public_anchor.validate_plan(plan, source["root_id"], source, plan["source_sha256"])


def metadata(payload, root, executor, source_path):
    for key, value in (("root_id", root), ("author", executor), ("source_sha256", digest(source_path))):
        require(key not in payload or payload[key] == value, "PAYLOAD_METADATA_MISMATCH:" + key)
    return {**payload, "root_id": root, "author": executor,
            "source_sha256": digest(source_path), "created_at_utc": now()}


def anchor_save(master, root, executor, payload_path):
    unit, cfg, context, ctx_path, source_path = checked_context(master, root, executor)
    folder = sequential.root_folder(unit, root)
    require(not (folder / "QA_DRAFT.json").exists() and not (folder / "QA_AUTHOR_PAYLOAD.json").exists(), "QA_ALREADY_ATTEMPTED")
    target = folder / "ORDERING_ANCHOR_PLAN.json"
    require(not target.exists(), "ANCHORS_ALREADY_FROZEN")
    plan = read(payload_path)
    require(plan.get("source_sha256") == digest(source_path), "ANCHOR_SOURCE_MISMATCH")
    validate_anchor(plan, read(source_path))
    raw_target = folder / "ORDERING_ANCHOR_PAYLOAD.json"
    write_bytes(raw_target, Path(payload_path).read_bytes())
    save(target, plan)
    save(folder / "ORDERING_ANCHOR_RECEIPT.json", {**receipt_from(context, ctx_path),
        "finished_at_utc": now(), "raw_payload_binding": binding(raw_target), "plan_binding": binding(target),
        "saved_before_qa_draft": True, "author_tool_binding": binding(__file__)})
    print(target)


def validate_qa(draft, plan, source):
    turns = {t["turn_ref"]: t for t in source["turns"]}
    require(isinstance(draft.get("stem"), str) and all(s in draft["stem"] for s in sequential.SCENES), "QA_STEM_SCOPE_REQUIRED")
    facts = draft["facts"]
    require(len(facts) == 6 and len({f["fact_id"] for f in facts}) == 6, "SIX_DISTINCT_FACTS_REQUIRED")
    require({turns[f["source_turn_ref"]]["scene_ref"] for f in facts} == set(sequential.SCENES), "ONE_FACT_PER_SCENE_REQUIRED")
    for fact in facts:
        require(isinstance(fact.get("true_text"), str) and fact["true_text"].strip(), "TRUE_FACT_REQUIRED")
        require(fact.get("false_text") is None or isinstance(fact["false_text"], str) and fact["false_text"].strip(), "FALSE_FACT_TEXT_INVALID")
    ordering = draft["ordering"]
    require(len(ordering) == 2 and {str(s["slot_id"]) for s in ordering} == {"1", "2"}, "TWO_ORDERING_SLOTS_REQUIRED")
    for slot in ordering:
        anchors = {a["anchor_id"]: a for a in next(p for p in plan["slots"] if str(p["slot_id"]) == str(slot["slot_id"]))["anchors"]}
        if slot.get("failure"):
            continue
        items = slot["items"]
        require(len(items) == 6 and {i["item_id"] for i in items} == {f"I{i}" for i in range(1, 7)}, "SIX_ORDERING_ITEMS_REQUIRED")
        require({i["anchor_id"] for i in items} == set(anchors), "ORDERING_ANCHOR_COVERAGE")
        for item in items:
            require(isinstance(item.get("text"), str) and item["text"].strip(), "ORDERING_TEXT_REQUIRED")
            require(item.get("support_quote") and item["support_quote"] in anchors[item["anchor_id"]]["full_turn_quote"], "ORDERING_SUPPORT_QUOTE_MISMATCH")
            require(item["item_id"] == anchors[item["anchor_id"]]["item_id"], "ORDERING_ITEM_ID_NOT_FROZEN")


def qa_save(master, root, executor, payload_path):
    unit, cfg, context, ctx_path, source_path = checked_context(master, root, executor)
    folder = sequential.root_folder(unit, root)
    plan_path = folder / "ORDERING_ANCHOR_PLAN.json"
    require(plan_path.exists() and (folder / "ORDERING_ANCHOR_RECEIPT.json").exists(), "ANCHORS_MUST_BE_FROZEN_FIRST")
    plan = read(plan_path)
    require(plan["source_sha256"] == digest(source_path) and read(folder / "ORDERING_ANCHOR_RECEIPT.json")["author"] == executor, "ANCHOR_PLAN_BINDING_MISMATCH")
    require(read(folder / "ORDERING_ANCHOR_RECEIPT.json")["plan_binding"] == binding(plan_path), "ANCHOR_PLAN_CHANGED")
    targets = [folder / x for x in ("QA_DRAFT.json", "QA_AUTHOR_RECEIPT.json", "QA_AUTHOR_PAYLOAD.json", "QA_SERIALIZATION_BINDING.json")]
    require(not any(p.exists() for p in targets), "FIRST_QA_PAYLOAD_ALREADY_SAVED")
    raw = Path(payload_path).read_bytes()
    write_bytes(folder / "QA_AUTHOR_PAYLOAD.json", raw)
    save(folder / "QA_SERIALIZATION_BINDING.json", {"created_at_utc": now(), "root_id": root,
        "author": executor, "payload_binding": binding(folder / "QA_AUTHOR_PAYLOAD.json"),
        "serializer_binding": binding(__file__), "anchor_plan_binding": binding(plan_path), "saved_before_qa_draft": True})
    try:
        draft = metadata(json.loads(raw), root, executor, source_path)
        validate_anchor(plan, read(source_path))
        validate_qa(draft, plan, read(source_path))
        save(folder / "QA_DRAFT.json", draft)
        require(plan_path.stat().st_mtime_ns <= (folder / "QA_DRAFT.json").stat().st_mtime_ns, "ANCHOR_PLAN_NOT_BEFORE_DRAFT")
    except Exception as exc:
        save(folder / "QA_AUTHOR_FAILURE.json", {"created_at_utc": now(), "root_id": root, "author": executor,
            "attempt": 1, "error": str(exc), "payload_binding": binding(folder / "QA_AUTHOR_PAYLOAD.json")})
        raise
    save(folder / "QA_AUTHOR_RECEIPT.json", {**receipt_from(context, ctx_path), "finished_at_utc": now(),
        "source_sha256": digest(source_path), "ordering_anchor_plan_sha256": digest(plan_path), "ordering_anchor_plan_binding": binding(plan_path),
        "ordering_anchor_plan_saved_before_QA_DRAFT": True, "qa_draft_sha256": digest(folder / "QA_DRAFT.json"),
        "payload_binding": binding(folder / "QA_AUTHOR_PAYLOAD.json"), "serializer_binding": binding(__file__),
        "payload_serialization_binding": binding(folder / "QA_SERIALIZATION_BINDING.json"),
        "timing_scope": "Wall clock from complete QA source rendering through frozen anchors and one first QA draft; includes scheduling, not inference latency or billed usage."})
    print(folder / "QA_DRAFT.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("scene-read", "scene-save", "qa-read", "qa-save", "anchor-save"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--scene", choices=sequential.SCENES)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--batch-dir", type=Path, default=MASTER, help="Alternate isolated fixture master; default is this v120 batch")
    args = parser.parse_args()
    if args.action.startswith("scene-") and not args.scene:
        parser.error("--scene is required for scene actions")
    if args.action.endswith("save") and not args.payload:
        parser.error("--payload is required for save actions")
    if not args.action.startswith("scene-") and args.scene:
        parser.error("--scene is only valid for scene actions")
    master = args.batch_dir.resolve()
    if args.action == "scene-read":
        make_read(master, args.root, args.executor, args.scene)
    elif args.action == "scene-save":
        scene_save(master, args.root, args.executor, args.scene, args.payload)
    elif args.action == "qa-read":
        make_read(master, args.root, args.executor)
    elif args.action == "anchor-save":
        anchor_save(master, args.root, args.executor, args.payload)
    else:
        qa_save(master, args.root, args.executor, args.payload)


if __name__ == "__main__":
    main()
