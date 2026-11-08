"""Record an assigned reader's independent prefix judgment; no semantic inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

CHECKS = {"time", "location", "actor_identity", "plan_vs_completion",
          "cross_scene_continuity", "semantic_repetition", "naturalness",
          "event_fidelity", "selection_alignment", "baseline_eligibility", "prefix_boundary"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def inputs(batch, root, executor):
    role_path = batch / "ROLE_ASSIGNMENTS.json"
    manifest_path = batch / "PREFIX_INPUTS.json"
    roles = [r for r in read(role_path)["roots"] if r["root_id"] == root]
    require(len(roles) == 1, "UNIQUE_ROLE_REQUIRED")
    role = roles[0]
    require(executor == role["prefix_reviewer"] == role["reviewer"], "NOT_ASSIGNED_REVIEWER")
    require(executor != role["author"], "AUTHOR_CANNOT_REVIEW")
    manifest = read(manifest_path)
    require(manifest["role_assignments_binding"] == bind(role_path), "ROLES_CHANGED")
    rows = [r for r in manifest["roots"] if r["root_id"] == root]
    require(len(rows) == 1, "UNIQUE_INPUT_SET_REQUIRED")
    row = rows[0]
    contents = []
    for item in row["inputs"]:
        path = Path(item["path"])
        require(sha(path) == item["sha256"], "INPUT_CHANGED:" + str(path))
        require(not any(x in path.name.upper() for x in ("QA_DRAFT", "PRIVATE", "GOLD", "SOURCE_REVIEW")), "PRIVATE_OR_PRIOR_JUDGMENT_INPUT")
        contents.append({"label": item["label"], "binding": bind(path), "content": read(path)})
    prefix = read(row["candidate_prefix_binding"]["path"])
    require(bind(row["candidate_prefix_binding"]["path"]) == row["candidate_prefix_binding"], "PREFIX_CHANGED")
    require(prefix["root_id"] == root, "PREFIX_ROOT")
    refs = [t["turn_ref"] for t in prefix["turns"]]
    require(len(refs) == len(set(refs)), "DUPLICATE_PREFIX_TURN")
    constraints = next(x["content"] for x in contents if x["label"] == "constraints")
    texts = constraints["continuation_constraints"]
    render = json.dumps({"root_id": root, "retained_scene_refs": row["retained_scene_refs"],
                         "proposed_retry_start_scene": row["proposed_retry_start_scene"],
                         "complete_inputs": contents}, ensure_ascii=False, indent=2) + "\n"
    return row, refs, texts, render, [bind(role_path), bind(manifest_path), bind(Path(__file__))] + [x["binding"] for x in contents]


def run(args):
    batch = Path(args.batch).resolve()
    row, refs, texts, render, bindings = inputs(batch, args.root, args.executor)
    folder = batch / "prefix_audits" / ("root" + args.root.rsplit("-", 1)[-1])
    folder.mkdir(parents=True, exist_ok=True)
    context_path = folder / "READ_CONTEXT.json"
    render_path = folder / "INPUT_RENDER.txt"
    output_path = folder / "PREFIX_SOURCE_REVIEW.json"
    require(not output_path.exists(), "REVIEW_ALREADY_EXISTS")
    if args.command == "read":
        require(not context_path.exists(), "READ_ALREADY_STARTED")
        with render_path.open("x", encoding="utf-8") as stream:
            stream.write(render)
        context = {"schema_version": "source_prefix_read_v1", "started_at_utc": now(),
                   "root_id": args.root, "executor": args.executor,
                   "input_bindings": bindings, "render_binding": bind(render_path),
                   "candidate_prefix_binding": row["candidate_prefix_binding"],
                   "retained_scene_refs": row["retained_scene_refs"],
                   "turn_refs_read": refs, "continuation_constraints": texts,
                   "reading_limitation": "Exact inputs are rendered and hash-bound. Full reading and privacy remain the executor's attestations; caller identity is not cryptographically authenticated."}
        save_new(context_path, context)
        print(render, end="")
        return
    context = read(context_path)
    require(context["executor"] == args.executor and context["root_id"] == args.root, "READ_EXECUTOR_ROOT")
    require(context["input_bindings"] == bindings, "INPUT_SET_CHANGED")
    require(context["render_binding"] == bind(render_path), "RENDER_CHANGED")
    require(render_path.read_text(encoding="utf-8") == render, "RENDER_NOT_EXACT")
    raw_path = Path(args.judgment).resolve()
    raw = read(raw_path)
    require(raw.get("executor") == args.executor, "RAW_EXECUTOR_MISMATCH")
    require(raw.get("root_id") == args.root, "RAW_ROOT_MISMATCH")
    attestation = raw.get("read_attestation", {})
    for key in ("full_baseline_read", "full_candidate_prefix_read", "all_constraints_read", "selection_read", "retained_native_events_read", "no_private_qa_or_prior_source_judgments_read"):
        require(attestation.get(key) is True, "ATTESTATION_REQUIRED:" + key)
    require(attestation.get("turn_refs_read") == refs, "FULL_TURN_COVERAGE_REQUIRED")
    require(attestation.get("continuation_constraints_read") == texts, "FULL_CONSTRAINT_COVERAGE_REQUIRED")
    require(raw.get("decision") in {"PASS", "HOLD"}, "DECISION_REQUIRED")
    require(set(raw.get("checks", {})) == CHECKS, "CHECK_SET_REQUIRED")
    for name, check in raw["checks"].items():
        require(check.get("status") in {"PASS", "WARNING", "FAIL"}, "CHECK_STATUS:" + name)
        require(isinstance(check.get("reason"), str) and check["reason"].strip(), "CHECK_REASON:" + name)
    require(isinstance(raw.get("findings"), list), "FINDINGS_REQUIRED")
    if raw["decision"] == "PASS":
        require(not any(x["status"] == "FAIL" for x in raw["checks"].values()), "PASS_WITH_FAILED_CHECK")
        require(not any(x.get("severity") == "BLOCKING" for x in raw["findings"]), "PASS_WITH_BLOCKING_FINDING")
    completed = now()
    require(datetime.fromisoformat(completed) >= datetime.fromisoformat(context["started_at_utc"]), "INVALID_CHRONOLOGY")
    output = {"schema_version": "source_prefix_review_v1", "created_at_utc": completed,
              "actual_executor": args.executor, "root_id": args.root,
              "read_context_binding": bind(context_path), "input_render_binding": bind(render_path),
              "raw_judgment_binding": bind(raw_path), "input_bindings": bindings,
              "candidate_prefix_binding": row["candidate_prefix_binding"],
              "retained_scene_refs": row["retained_scene_refs"], "judgment": raw,
              "semantic_judgment_supplied_by_executor": True, "content_reuse_approved": raw["decision"] == "PASS",
              "formal_generation_started": False, "private_access_proven_by_program": False,
              "internal_reading_proven_by_program": False}
    save_new(output_path, output)
    save_new(folder / "SAVE_RECEIPT.json", {"created_at_utc": completed, "actual_executor": args.executor,
              "read_context_binding": bind(context_path), "raw_judgment_binding": bind(raw_path),
              "output_binding": bind(output_path), "status": "RECORDED_ASSIGNED_PREFIX_JUDGMENT"})
    print(json.dumps({"output": str(output_path), "decision": raw["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["read", "save"])
    parser.add_argument("--batch", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--judgment")
    args = parser.parse_args()
    if args.command == "save" and not args.judgment:
        parser.error("--judgment required for save")
    run(args)
