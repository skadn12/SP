"""Record assigned agents' source-first judgments; never supply semantic answers.

The explicit executor is checked against ROLE_ASSIGNMENTS.json. This shared
filesystem cannot cryptographically authenticate the caller or prove that a
model read text, or did not previously read private answers. Those remain
explicit executor attestations. No model, Solver, training, or private design
reader is called by this helper.

Save inputs contain ``read_attestation`` plus the semantic fields below:
scene: decision/checks/findings/baseline_constraint_review;
asset: multiselect_stem_valid/statements/ordering_questions/findings;
public: questions/findings.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


DEFAULT_BATCH = Path(__file__).resolve().parent.parent
PROJECT = Path(__file__).resolve().parents[3]
SCENES = tuple(f"R02:{route}:S{i:02d}" for route in "AB" for i in (1, 2, 3))
CHECKS = {"time", "location", "actor_identity", "plan_vs_completion",
          "cross_scene_continuity", "semantic_repetition", "naturalness", "event_fidelity"}
STATUSES = {"PASS", "WARNING", "PASS_WITH_WARNING", "FAIL"}
OUTPUTS = {"scene": "SOURCE_REVIEW.json", "asset": "ASSET_SOURCE_JUDGMENTS.json",
           "public": "PUBLIC_SOURCE_JUDGMENTS.json"}
IDENTITY_LIMITATION = ("Explicit canonical executor checked against the bound role assignment; "
                      "shared-filesystem caller identity is not cryptographically authenticated.")
READ_LIMITATION = ("Full reading and source-first privacy are the executor's explicit attestations. "
                   "Rendering, hashes, and field checks cannot prove internal reading or prior private-file access.")


class ReviewError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path: Path | str) -> dict:
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha(p)}


def unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "DUPLICATE_JSON_KEY:" + key)
        result[key] = value
    return result


def read(path: Path | str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    require(isinstance(data, dict), "JSON_OBJECT_REQUIRED:" + str(path))
    return data


def save_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def canonical_executor(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"/root/[a-z0-9_]+(?:/[a-z0-9_]+)*", value) is not None


def resolve(unit: Path, value: str) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else unit / p).resolve()


def verify_binding(item: dict, unit: Path, label: str) -> Path:
    require(isinstance(item, dict) and isinstance(item.get("path"), str), label + ":BINDING_REQUIRED")
    p = resolve(unit, item["path"])
    require(sha(p) == item.get("sha256"), label + ":HASH_CHANGED")
    return p


def layout(batch: Path | str, root: str, executor: str) -> tuple:
    batch = Path(batch).resolve()
    require(re.fullmatch(r"cc-train-episode-\d+", root) is not None, "INVALID_ROOT")
    require(canonical_executor(executor), "EXPLICIT_CANONICAL_EXECUTOR_REQUIRED")
    roles_path = batch / "ROLE_ASSIGNMENTS.json"
    roles = read(roles_path).get("roots")
    require(isinstance(roles, list) and roles, "ROLE_ASSIGNMENTS_ROOTS_REQUIRED")
    require(all(isinstance(row, dict) for row in roles), "ROLE_ASSIGNMENT_OBJECT_REQUIRED")
    require(len({row.get("root_id") for row in roles}) == len(roles), "DUPLICATE_ROLE_ROOT")
    matches = [row for row in roles if row.get("root_id") == root]
    require(len(matches) == 1, "ROOT_NOT_ASSIGNED")
    role = matches[0]
    require(canonical_executor(role.get("author")) and canonical_executor(role.get("reviewer")), "CANONICAL_ROLE_IDENTITIES_REQUIRED")
    require(role["author"] != role["reviewer"], "ASSIGNED_REVIEWER_IS_AUTHOR")
    baseline_reviewers = role.get("baseline_reviewers")
    require(isinstance(baseline_reviewers, list) and len(baseline_reviewers) == 2 and
            len(set(baseline_reviewers)) == 2 and all(canonical_executor(x) for x in baseline_reviewers),
            "TWO_DISTINCT_BASELINE_REVIEWERS_REQUIRED")
    require(executor == role["reviewer"], "EXECUTOR_NOT_ASSIGNED_REVIEWER")
    short = root.rsplit("-", 1)[-1]
    unit = batch / "units" / ("root" + short)
    return batch, unit, unit / "roots" / ("root" + short), role, roles_path


def validate_turns(source: dict, root: str) -> dict:
    require(source.get("root_id") == root, "SOURCE_ROOT_MISMATCH")
    rows = source.get("turns")
    require(isinstance(rows, list) and rows, "SOURCE_TURNS_REQUIRED")
    for turn in rows:
        require(isinstance(turn, dict) and all(isinstance(turn.get(k), str) and turn[k].strip()
                                             for k in ("turn_ref", "speaker", "text")), "SOURCE_TURN_SCHEMA")
    turns = {turn["turn_ref"]: turn for turn in rows}
    require(len(turns) == len(rows), "DUPLICATE_SOURCE_TURN")
    return turns


def controller_state(unit: Path, root: str) -> dict:
    # The unchanged controller checks materialization hashes and exact prefixes.
    # record_reviews=False prevents this read helper from writing controller receipts.
    scripts = str(PROJECT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from codex_sequential_ab import inspect_root, load_config
    cfg = load_config(unit)
    ids = [row["root_id"] if isinstance(row, dict) else row for row in cfg["roots"]]
    require(ids == [root], "UNIT_MUST_CONTAIN_EXACTLY_ITS_ROOT")
    return inspect_root(unit, cfg, root, record_reviews=False)


def inputs(batch: Path | str, root: str, phase: str, executor: str, scene: str | None) -> dict:
    require(phase in OUTPUTS, "INVALID_REVIEW_PHASE")
    batch, unit, folder, role, role_path = layout(batch, root, executor)
    require(scene in SCENES if phase == "scene" else scene is None, "SCENE_ARGUMENT_MISMATCH")
    target_folder = folder / "scenes" / scene.removeprefix("R02:").replace(":", "_") if scene else folder
    output = target_folder / OUTPUTS[phase]
    require(not output.exists(), "REVIEW_ALREADY_EXISTS")
    source_path = target_folder / "CUMULATIVE_SOURCE.json"
    source = read(source_path)
    turns = validate_turns(source, root)
    constraints_path = unit / "inputs/CONTINUATION_CONSTRAINTS.json"
    constraints = read(constraints_path)
    require(constraints.get("root_id") == root and constraints.get("decision") in {"PASS", "WARNING"}, "BASELINE_NOT_ELIGIBLE")
    constraint_texts = constraints.get("continuation_constraints")
    require(isinstance(constraint_texts, list) and bool(constraint_texts) and all(isinstance(x, str) and x.strip() for x in constraint_texts), "CONSTRAINT_LIST_REQUIRED")
    refs = constraints.get("review_bindings")
    require(isinstance(refs, list) and len(refs) == 2, "TWO_BASELINE_REVIEW_BINDINGS_REQUIRED")
    paths = [role_path, unit / "RUN_CONFIG.json", constraints_path, source_path, Path(__file__).resolve()]
    baseline_path = verify_binding(constraints.get("baseline_binding"), unit, "BASELINE")
    require(baseline_path == (unit / "inputs/BASELINE.json").resolve(), "CONSTRAINT_BASELINE_PATH_MISMATCH")
    paths.append(baseline_path)
    baseline_reviewers = []
    for item in refs:
        p = verify_binding(item, unit, "BASELINE_REVIEW")
        baseline_review = read(p)
        actual = item.get("reviewer", baseline_review.get("reviewer"))
        require(actual == baseline_review.get("reviewer"), "BASELINE_REVIEW_IDENTITY_MISMATCH")
        baseline_reviewers.append(actual)
        paths.append(p)
    require(len(set(baseline_reviewers)) == 2 and set(baseline_reviewers) == set(role["baseline_reviewers"]), "BASELINE_REVIEWERS_NOT_ASSIGNED")
    state = controller_state(unit, root)
    public = native = selection = None
    native_binding = selection_binding = public_binding = asset_binding = None
    if phase == "scene":
        current = state["scenes"][SCENES.index(scene)]
        require(not state["root_stopped"] and current["status"] == "AWAITING_REVIEW", "SCENE_NOT_AWAITING_VALID_FIRST_REVIEW")
        draft_path = target_folder / "EXPANSION_DRAFT.json"
        author_record = read(draft_path)
        require(author_record.get("author") == role["author"], "DRAFT_AUTHOR_NOT_ASSIGNED")
        require(author_record["author"] != executor, "SELF_REVIEW_FORBIDDEN")
        paths.append(draft_path)
        if ":B:" in scene:
            reservations_path = unit / "SOURCE_RESERVATIONS.json"
            reservations = read(reservations_path)["allocation"]
            matches = [row for row in reservations if row["root_id"] == root and row["scene_ref"] == scene]
            require(len(matches) == 1, "NATIVE_RESERVATION_REQUIRED")
            row = matches[0]
            native_path = verify_binding({"path": row["source_item_path"], "sha256": row["source_item_sha256"]}, unit, "NATIVE_EVENT")
            native = read(native_path)
            require(isinstance(native.get("turns"), list) and native["turns"], "NATIVE_TURNS_REQUIRED")
            native_ids = [turn["turn_id"] for turn in native["turns"]]
            require(len(native_ids) == len(set(native_ids)), "DUPLICATE_NATIVE_TURN")
            native_binding = binding(native_path)
            paths.extend([reservations_path, native_path])
            if row.get("selection_path"):
                selection_path = verify_binding({"path": row["selection_path"], "sha256": row["selection_sha256"]}, unit, "SELECTION")
                selection = read(selection_path)
                selection_binding = binding(selection_path)
                paths.append(selection_path)
    else:
        require(state["full_expansion_complete"], "QA_REQUIRES_SIX_READY_SCENES")
        require(sha(source_path) == sha(folder / "scenes/B_S03/CUMULATIVE_SOURCE.json"), "FINAL_SOURCE_NOT_B3_BYTE_COPY")
        qa_receipt = folder / "QA_AUTHOR_RECEIPT.json"
        author_record = read(qa_receipt)
        require(author_record.get("author") == role["author"], "QA_AUTHOR_NOT_ASSIGNED")
        require(author_record["author"] != executor, "QA_SELF_REVIEW_FORBIDDEN")
        require(author_record.get("source_sha256") == sha(source_path), "QA_AUTHOR_SOURCE_HASH")
        paths.append(qa_receipt)
        public_path = folder / ("PUBLIC_ASSETS.json" if phase == "asset" else "PUBLIC_QUESTIONS.json")
        public = read(public_path)
        public_binding = binding(public_path)
        paths.append(public_path)
        require(not (folder / "PRIVATE_GOLD.json").exists(), "PRIVATE_GOLD_ALREADY_EXISTS_BEFORE_PUBLIC_REVIEW")
        if phase == "asset":
            require(public.get("root_id") == root and public.get("source_sha256") == sha(source_path), "ASSET_SOURCE_BINDING")
        else:
            asset_path = folder / "ASSET_SOURCE_JUDGMENTS.json"
            asset = read(asset_path)
            require(asset.get("source_sha256") == sha(source_path) and asset.get("reviewer") == executor, "PRIOR_ASSET_REVIEW_BINDING")
            asset_binding = binding(asset_path)
            paths.append(asset_path)
            for question in public.get("questions", []):
                require(question.get("root_id") == root and question.get("source_sha256") == sha(source_path), "PUBLIC_QUESTION_SOURCE_BINDING")
                require(resolve(unit, question["source_path"]) == source_path.resolve(), "PUBLIC_QUESTION_SOURCE_PATH")
    # Every predecessor review is bound, so later changes cannot evade read/save checks.
    for candidate in state["scenes"]:
        if candidate["status"] == "READY":
            paths.append(Path(candidate["folder"]) / "SOURCE_REVIEW.json")
    all_bindings = [binding(p) for p in dict.fromkeys(paths)]
    return {"batch": batch, "unit": unit, "folder": folder, "target_folder": target_folder, "output": output,
            "root": root, "phase": phase, "scene": scene, "executor": executor, "role": role,
            "source": source, "turns": turns, "constraints": constraints, "constraints_binding": binding(constraints_path),
            "source_binding": binding(source_path), "role_assignments_binding": binding(role_path),
            "public": public, "public_binding": public_binding, "asset_judgments_binding": asset_binding,
            "native": native, "native_binding": native_binding, "selection": selection, "selection_binding": selection_binding,
            "input_bindings": all_bindings, "author_record": author_record}


def read_directory(data: dict) -> Path:
    executor_hash = hashlib.sha256(data["executor"].encode()).hexdigest()[:16]
    label = data["scene"].replace(":", "_") if data["scene"] else "root"
    return data["unit"] / "evidence/reads" / f'{data["phase"]}_{label}_{executor_hash}'


def render(data: dict) -> str:
    lines = ["REVIEW_ASSIGNMENT", json.dumps({"root_id": data["root"], "phase": data["phase"],
             "scene_ref": data["scene"], "executor_id": data["executor"], "assigned_roles": data["role"]}, ensure_ascii=False),
             "ALL_CONTINUATION_CONSTRAINTS", json.dumps(data["constraints"], ensure_ascii=False, indent=2), "FULL_CUMULATIVE_SOURCE"]
    lines.extend(t["turn_ref"] + " | " + t["speaker"] + ": " + t["text"] for t in data["source"]["turns"])
    if data["phase"] == "scene":
        lines.extend(["DRAFT_NOTES", json.dumps({k: v for k, v in data["author_record"].items() if k != "turns"}, ensure_ascii=False, indent=2)])
    for label, value in (("NATIVE_EVENT_FULL", data["native"]), ("SELECTION_FULL", data["selection"]), ("PUBLIC_CONTENT_FULL", data["public"])):
        if value is not None:
            lines.extend([label, json.dumps(value, ensure_ascii=False, indent=2)])
    lines.extend(["EVIDENCE_LIMITATIONS", IDENTITY_LIMITATION, READ_LIMITATION])
    return "\n".join(lines) + "\n"


def start(batch: Path | str, root: str, phase: str, executor: str, scene: str | None = None) -> Path:
    data = inputs(batch, root, phase, executor, scene)
    directory = read_directory(data)
    directory.mkdir(parents=True, exist_ok=False)
    rendered = render(data)
    render_path = directory / "INPUT_RENDER.txt"
    with render_path.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    context = {"schema_version": "codex_assigned_read_context_v2", "root_id": root, "phase": phase, "scene_ref": scene,
               "executor_id": executor, "started_at_utc": now(), "source_binding": data["source_binding"],
               "constraints_binding": data["constraints_binding"], "role_assignments_binding": data["role_assignments_binding"],
               "public_binding": data["public_binding"], "native_event_binding": data["native_binding"],
               "selection_binding": data["selection_binding"], "asset_judgments_binding": data["asset_judgments_binding"],
               "input_render_binding": binding(render_path), "input_bindings": data["input_bindings"],
               "turn_refs_rendered": list(data["turns"]), "constraint_count_rendered": len(data["constraints"]["continuation_constraints"]),
               "identity_verification_scope": IDENTITY_LIMITATION, "read_attestation_scope": READ_LIMITATION}
    save_new(directory / "READ_CONTEXT.json", context)
    # Writing a render is not a semantic read. The actual agent must inspect this
    # output (or the complete saved render) before explicitly attesting at save.
    print(rendered, end="")
    print("READ_CONTEXT", directory / "READ_CONTEXT.json")
    return directory / "READ_CONTEXT.json"


def text(value: Any, where: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), where + ":NONEMPTY_TEXT_REQUIRED")


def bool_field(data: dict, name: str, where: str) -> None:
    require(type(data.get(name)) is bool, where + ":BOOLEAN_REQUIRED:" + name)


def validate_attestation(raw: dict, data: dict) -> dict:
    attestation = raw.get("read_attestation")
    require(isinstance(attestation, dict), "READ_ATTESTATION_REQUIRED")
    for name in ("full_source_read", "all_continuation_constraints_read"):
        require(attestation.get(name) is True, "EXPLICIT_READ_ATTESTATION_REQUIRED:" + name)
    if data["native"] is not None:
        require(attestation.get("native_event_full_read") is True, "EXPLICIT_NATIVE_READ_ATTESTATION_REQUIRED")
    if data["phase"] != "scene":
        require(attestation.get("private_answers_read_before_save") is False, "EXPLICIT_SOURCE_FIRST_ATTESTATION_REQUIRED")
    if data["phase"] == "public":
        require(attestation.get("final_public_questions_actually_reread") is True, "EXPLICIT_FINAL_PUBLIC_REREAD_REQUIRED")
    return deepcopy(attestation)


def validate_constraints(record: dict, data: dict) -> dict:
    require(isinstance(record, dict) and record.get("status") in STATUSES, "CONSTRAINT_REVIEW_STATUS")
    text(record.get("reason"), "CONSTRAINT_REVIEW_REASON")
    rows = record.get("constraint_results")
    constraints = data["constraints"]["continuation_constraints"]
    require(isinstance(rows, list) and len(rows) == len(constraints), "CONSTRAINT_COVERAGE_INCOMPLETE")
    for index, (row, constraint) in enumerate(zip(rows, constraints), 1):
        require(row.get("index") == index and row.get("constraint") == constraint, "CONSTRAINT_ORDER_OR_TEXT_CHANGED")
        require(row.get("status") in STATUSES, "CONSTRAINT_STATUS_INVALID")
        text(row.get("reason"), "CONSTRAINT_REASON")
        refs = row.get("evidence_turn_refs")
        require(isinstance(refs, list) and bool(refs) and len(refs) == len(set(refs)) and set(refs) <= set(data["turns"]), "CONSTRAINT_EVIDENCE_REFS")
    require(isinstance(record.get("new_blocking_findings"), list), "CONSTRAINT_BLOCKING_FINDINGS_REQUIRED")
    failed = any(row["status"] == "FAIL" for row in rows) or bool(record["new_blocking_findings"])
    require(not failed or record["status"] == "FAIL", "CONSTRAINT_FAILURE_NOT_AGGREGATED")
    require(record["status"] != "PASS" or all(row["status"] == "PASS" for row in rows), "CONSTRAINT_PASS_NOT_AGGREGATED")
    result = deepcopy(record)
    result.update(constraints_binding=data["constraints_binding"], baseline_gate_version=data["batch"].name,
                  inherited_baseline_decision=data["constraints"]["decision"])
    return result


def validate_scene(raw: dict, data: dict) -> dict:
    require(raw.get("decision") in {"READY", "HOLD"}, "SCENE_DECISION_INVALID")
    checks = raw.get("checks")
    require(isinstance(checks, dict) and set(checks) == CHECKS, "EIGHT_SCENE_CHECKS_REQUIRED")
    for name, value in checks.items():
        require(isinstance(value, dict) and value.get("status") in STATUSES, "CHECK_STATUS_INVALID:" + name)
        text(value.get("reason"), "CHECK_REASON:" + name)
    require(isinstance(raw.get("findings"), list), "FINDINGS_LIST_REQUIRED")
    constraints = validate_constraints(raw.get("baseline_constraint_review"), data)
    if raw["decision"] == "READY":
        require(all(value["status"] != "FAIL" for value in checks.values()), "READY_WITH_FAILED_CHECK")
        require(constraints["status"] != "FAIL", "READY_WITH_FAILED_CONSTRAINT")
        require(not any(item.get("severity") == "BLOCKING" for item in raw["findings"]), "READY_WITH_BLOCKING_FINDING")
    return {"decision": raw["decision"], "checks": deepcopy(checks), "findings": deepcopy(raw["findings"]),
            "baseline_constraint_review": constraints}


def citation(value: dict, turns: dict, where: str) -> None:
    require(isinstance(value, dict), where + ":CITATION_OBJECT_REQUIRED")
    turn = turns.get(value.get("turn_ref"))
    require(turn is not None, where + ":CITATION_TURN_INVALID")
    quote = value.get("quote")
    require(isinstance(quote, str) and bool(quote) and quote in turn["text"], where + ":CITATION_NOT_EXACT")
    require(value.get("speaker") == turn["speaker"], where + ":CITATION_SPEAKER_MISMATCH")


def statement(value: dict, proposition: str, turns: dict, where: str) -> None:
    require(isinstance(value, dict) and value.get("proposition") == proposition, where + ":PROPOSITION_MISMATCH")
    require(value.get("relation") in {"supports", "contradicts", "unknown"}, where + ":RELATION_INVALID")
    for name in ("scope_valid", "fact_distinct"):
        bool_field(value, name, where)
    text(value.get("reason"), where + ":REASON")
    citations = value.get("citations")
    require(isinstance(citations, list) and bool(citations), where + ":CITATIONS_REQUIRED")
    for item in citations:
        citation(item, turns, where)
    if len(citations) > 1 or "primary_citation" in value:
        require(value.get("primary_citation") in citations, where + ":EXPLICIT_REVIEWED_PRIMARY_REQUIRED")


def ordering(value: dict, question: dict, data: dict) -> None:
    where = question["candidate_id"]
    require(value.get("candidate_id") == where, where + ":CANDIDATE_MISMATCH")
    for name in ("stem_valid", "scope_valid", "coherent", "unique_order", "neutral_stem_ok"):
        bool_field(value, name, where)
    text(value.get("reason"), where + ":REASON")
    labels = {f"I{i}" for i in range(1, 7)}
    require(set(question.get("items", {})) == labels and set(value.get("items", {})) == labels, where + ":SIX_ITEM_JUDGMENTS_REQUIRED")
    for label, proposition in question["items"].items():
        item = value["items"][label]
        require(item.get("proposition") == proposition, where + label + ":PROPOSITION_MISMATCH")
        for name in ("supported", "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok", "single_event_ok"):
            bool_field(item, name, where + label)
        text(item.get("reason"), where + label + ":REASON")
        anchor = item.get("order_anchor")
        citation(anchor, data["turns"], where + label)
        turn = data["turns"][anchor["turn_ref"]]
        require(turn["text"].count(anchor["quote"]) == 1, where + label + ":ANCHOR_NOT_UNIQUE_SPAN")
        require(turn.get("scene_ref") in SCENES, where + label + ":ANCHOR_OUTSIDE_PUBLIC_SCOPE")
        refs = item.get("support_refs")
        require(isinstance(refs, list) and bool(refs) and len(refs) == len(set(refs)) and
                set(refs) <= set(data["turns"]) and anchor["turn_ref"] in refs, where + label + ":SUPPORT_REFS_INVALID")
        alternatives = item.get("alternative_matching_anchors")
        require(isinstance(alternatives, list), where + label + ":ALTERNATIVES_LIST_REQUIRED")
        for alternative in alternatives:
            citation(alternative, data["turns"], where + label + ":ALTERNATIVE")
    require(isinstance(value.get("derived_order"), list) and len(value["derived_order"]) == 6 and
            set(value["derived_order"]) == labels, where + ":DERIVED_ORDER_LABELS_INVALID")
    require(isinstance(value.get("alternative_orders"), list), where + ":ALTERNATIVE_ORDERS_REQUIRED")
    for alternative in value["alternative_orders"]:
        require(isinstance(alternative, list) and len(alternative) == 6 and set(alternative) == labels, where + ":ALTERNATIVE_ORDER_LABELS_INVALID")


def validate_qa(raw: dict, data: dict) -> dict:
    require(isinstance(raw.get("findings"), list), "FINDINGS_LIST_REQUIRED")
    if data["phase"] == "asset":
        bool_field(raw, "multiselect_stem_valid", "ASSET")
        public_statements = data["public"].get("statements")
        require(isinstance(public_statements, list), "PUBLIC_STATEMENTS_REQUIRED")
        propositions = {row["statement_id"]: row["proposition"] for row in public_statements}
        require(len(propositions) == len(public_statements), "DUPLICATE_PUBLIC_STATEMENT_ID")
        judgments = raw.get("statements")
        require(isinstance(judgments, dict) and set(judgments) == set(propositions), "STATEMENT_JUDGMENTS_INCOMPLETE")
        for sid, proposition in propositions.items():
            statement(judgments[sid], proposition, data["turns"], sid)
        questions = data["public"].get("ordering_questions")
        reviews = raw.get("ordering_questions")
        require(isinstance(questions, list) and isinstance(reviews, list), "ORDERING_QUESTIONS_LIST_REQUIRED")
        by_id = {row["candidate_id"]: row for row in reviews}
        require(len(by_id) == len(reviews) and set(by_id) == {q["candidate_id"] for q in questions}, "ORDERING_ASSET_REVIEWS_INCOMPLETE")
        for question in questions:
            ordering(by_id[question["candidate_id"]], question, data)
        return {name: deepcopy(raw[name]) for name in ("multiselect_stem_valid", "statements", "ordering_questions", "findings")}
    questions = data["public"].get("questions")
    require(isinstance(questions, list), "PUBLIC_QUESTIONS_REQUIRED")
    by_id = {question["candidate_id"]: question for question in questions}
    require(len(by_id) == len(questions), "DUPLICATE_PUBLIC_CANDIDATE")
    judgments = raw.get("questions")
    require(isinstance(judgments, dict) and set(judgments) == set(by_id), "PUBLIC_JUDGMENTS_INCOMPLETE")
    for cid, question in by_id.items():
        value = judgments[cid]
        require(value.get("candidate_id") == cid, "PUBLIC_JUDGMENT_CANDIDATE_MISMATCH")
        if question.get("question_type") == "multi_select":
            for name in ("stem_valid", "scope_valid", "fact_distinct", "no_entangled_options"):
                bool_field(value, name, cid)
            text(value.get("reason"), cid + ":REASON")
            require(set(question.get("options", {})) == set("ABCDEF") and set(value.get("options", {})) == set("ABCDEF"), "SIX_OPTION_JUDGMENTS_REQUIRED")
            for label, proposition in question["options"].items():
                statement(value["options"][label], proposition, data["turns"], cid + label)
        else:
            require(question.get("question_type") == "ordering", "UNSUPPORTED_QUESTION_TYPE")
            ordering(value, question, data)
    return {"questions": deepcopy(judgments), "findings": deepcopy(raw["findings"])}


def save_review(batch: Path | str, root: str, phase: str, executor: str,
                judgment: Path | str, scene: str | None = None) -> Path:
    data = inputs(batch, root, phase, executor, scene)
    directory = read_directory(data)
    context_path = directory / "READ_CONTEXT.json"
    context = read(context_path)
    require((context.get("root_id"), context.get("phase"), context.get("scene_ref"), context.get("executor_id")) ==
            (root, phase, scene, executor), "READ_CONTEXT_IDENTITY_MISMATCH")
    require(context.get("input_bindings") == data["input_bindings"], "READ_INPUTS_CHANGED")
    for item in [*context["input_bindings"], context["input_render_binding"]]:
        verify_binding(item, data["unit"], "READ_CONTEXT_INPUT")
    require(context["input_render_binding"] == binding(directory / "INPUT_RENDER.txt"), "READ_RENDER_PATH_CHANGED")
    require((directory / "INPUT_RENDER.txt").read_text(encoding="utf-8") == render(data), "READ_RENDER_NOT_EXACT")
    raw_bytes = Path(judgment).read_bytes()
    archive = directory / "RAW_JUDGMENT.json"
    # Preserve the first supplied judgment even if its schema is rejected. A
    # correction must never silently replace an earlier semantic judgment.
    with archive.open("xb") as stream:
        stream.write(raw_bytes)
    try:
        raw = read(archive)
        attestation = validate_attestation(raw, data)
        supplied = validate_scene(raw, data) if phase == "scene" else validate_qa(raw, data)
        finished = now()
        require(context["started_at_utc"] <= finished, "INVALID_READ_INTERVAL")
        role_names = {"scene": "non_author_complete_cumulative_source_and_native_event_reviewer",
                      "asset": "non_author_source_first_public_asset_reviewer",
                      "public": "non_author_source_first_final_public_question_reviewer"}
        review = {"schema_version": "codex_assigned_source_review_v2", "root_id": root,
                  "reviewer": executor, "review_role": role_names[phase], "backend": "CURRENT_CODEX_SESSION",
                  "exact_model_id": None, "external_semantic_certification": False,
                  "started_at_utc": context["started_at_utc"], "finished_at_utc": finished, "created_at_utc": finished,
                  "source_sha256": data["source_binding"]["sha256"], "full_source_read": True,
                  "turn_refs_read": list(data["turns"]), "all_continuation_constraints_read": True,
                  "constraints_binding": data["constraints_binding"], "role_assignments_binding": data["role_assignments_binding"],
                  "read_context_binding": binding(context_path), "input_render_binding": context["input_render_binding"],
                  "raw_judgment_binding": binding(archive), "helper_binding": binding(Path(__file__)),
                  "executor_read_attestation": attestation, "identity_verification_scope": IDENTITY_LIMITATION,
                  "read_attestation_scope": READ_LIMITATION, "semantic_review_is_programmatic": False, **supplied}
        if phase == "scene":
            review["scene_ref"] = scene
            if data["native"] is not None:
                review.update(native_event_full_read=True, native_event_path=data["native_binding"]["path"],
                              native_event_sha256=data["native_binding"]["sha256"],
                              native_turn_ids_read=[turn["turn_id"] for turn in data["native"]["turns"]],
                              source_id=data["author_record"].get("source_id"))
                if data["selection_binding"] is not None:
                    review.update(selection_path=data["selection_binding"]["path"], selection_sha256=data["selection_binding"]["sha256"])
        else:
            review.update(public_sha256=data["public_binding"]["sha256"], private_answers_read_before_save=False,
                          independent_of_question_author=True)
            if phase == "public":
                review.update(asset_judgments_sha256=data["asset_judgments_binding"]["sha256"], final_public_questions_actually_reread=True)
        save_new(data["output"], review)
        save_new(directory / "SAVE_RECEIPT.json", {"status": "RECORDED_EXECUTOR_JUDGMENT_NOT_EXTERNAL_CERTIFICATION",
                 "created_at_utc": now(), "executor_id": executor, "root_id": root, "phase": phase,
                 "read_context_binding": binding(context_path), "raw_judgment_binding": binding(archive),
                 "output_binding": binding(data["output"]), "semantic_judgment_supplied_by_executor": True,
                 "private_file_access_proven_by_program": False, "internal_reading_proven_by_program": False})
    except Exception as exc:
        save_new(directory / "SAVE_FAILURE.json", {"status": "FIRST_JUDGMENT_REJECTED_PRESERVED", "created_at_utc": now(),
                 "executor_id": executor, "raw_judgment_binding": binding(archive), "error_type": type(exc).__name__, "error": str(exc)})
        raise
    print(data["output"])
    return data["output"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("stage", choices=[f"{phase}-{action}" for phase in OUTPUTS for action in ("read", "save")])
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--executor", required=True, help="Actual executing canonical agent ID, checked against fixed assignment; not cryptographic authentication.")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--judgment", type=Path)
    args = parser.parse_args(argv)
    phase, action = args.stage.split("-")
    require((action == "save") == (args.judgment is not None), "SAVE_REQUIRES_JUDGMENT_READ_FORBIDS_IT")
    if action == "read":
        start(args.batch, args.root, phase, args.executor, args.scene)
    else:
        save_review(args.batch, args.root, phase, args.executor, args.judgment, args.scene)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
