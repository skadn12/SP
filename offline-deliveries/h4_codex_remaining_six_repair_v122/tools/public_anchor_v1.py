"""Strict, answer-free public ordering anchor inputs for v120 reviews.

Only structural/exact-source checks happen here. The assigned reviewer still
judges event support, ambiguity, and order independently from the complete
source. A full-turn hash uses SHA256 of UTF-8 text without normalization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCENES = tuple(f"R02:{route}:S{i:02d}" for route in "AB" for i in (1, 2, 3))
SCHEMA = "codex_public_ordering_anchor_plan_v1"
PLAN_FIELDS = {"schema_version", "root_id", "source_sha256", "scope_scene_refs", "slots"}
SLOT_FIELDS = {"slot_id", "candidate_id", "anchors"}
ANCHOR_FIELDS = {"item_id", "anchor_id", "turn_ref", "speaker", "full_turn_quote", "full_turn_sha256"}
RUNTIME_TOOLS = ("scene_controller_v120.py", "question_controller_v120.py", "author_v120.py", "review_v4.py",
                 "review_provenance_gate_v2.py", "scene_review_provenance_v1.py", "public_anchor_v1.py", "run_guarded_questions_v2.py", "historical_fixed_import_v122.py")
RUNTIME_SCRIPTS = ("codex_sequential_ab.py", "codex_controlled_batch.py", "codex_sequential_questions.py", "codex_scene_payload.py")
ASSET_FIELDS = {"root_id", "source_sha256", "scope_scene_refs", "multiselect_stem", "statements", "ordering_questions", "ordering_failures"}
QUESTION_FIELDS = {"candidate_id", "root_id", "question_type", "question", "source_path", "source_sha256"}


class AnchorError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AnchorError(message)


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_runtime(batch, caller=None, loaded_functions=None):
    """Check the frozen transitive runtime before rendering or dispatch.

    Actual loaded functions must have the exact bytes frozen for that tool;
    this also rejects a cached foreign implementation or replacement lambda.
    Shared-filesystem attestations are still not cryptographic authentication.
    """
    batch = Path(batch).resolve()
    freeze = json.loads((batch / "MASTER_FREEZE.json").read_text(encoding="utf-8"))
    rows = freeze.get("files")
    require(isinstance(rows, list), "RUNTIME_FREEZE_FILES_REQUIRED")
    expected = {}
    paths = [batch / "tools" / n for n in RUNTIME_TOOLS]
    paths += [batch / "scripts" / n for n in RUNTIME_SCRIPTS]
    paths += [batch / "ROLE_ASSIGNMENTS.json"]
    for path in paths:
        matches = [row for row in rows if isinstance(row, dict) and row.get("path") == str(path.resolve())]
        require(len(matches) == 1, "RUNTIME_UNIQUE_FROZEN_BINDING_REQUIRED:" + str(path))
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == matches[0].get("sha256"),
                "RUNTIME_FROZEN_HASH_MISMATCH:" + str(path))
        expected[path.name] = matches[0]["sha256"]
    actual = {"public_anchor_v1.py": Path(__file__).resolve()}
    if caller is not None:
        actual[Path(caller).name] = Path(caller).resolve()
    for name, function in (loaded_functions or {}).items():
        code = getattr(function, "__code__", None)
        require(code is not None, "RUNTIME_LOADED_FUNCTION_REQUIRED:" + name)
        actual[name] = Path(code.co_filename).resolve()
    for name, path in actual.items():
        require(name in expected and path == (batch / "tools" / name).resolve() and path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected[name],
                "RUNTIME_LOADED_CODE_MISMATCH:" + name + ":" + str(path))
    return paths


def public_question(question, ordering_only=False):
    require(isinstance(question, dict), "PUBLIC_QUESTION_OBJECT_REQUIRED")
    kind = question.get("question_type")
    require(kind in ({"ordering"} if ordering_only else {"ordering", "multi_select"}), "PUBLIC_QUESTION_TYPE")
    allowed = QUESTION_FIELDS | ({"items", "chronology_kind"} if kind == "ordering" else {"options"})
    required = QUESTION_FIELDS | ({"items"} if kind == "ordering" else {"options"})
    require(required <= set(question) <= allowed, "PUBLIC_QUESTION_FIELDS_NOT_PUBLIC_SCHEMA")
    require(all(isinstance(question[k], str) and question[k].strip() for k in QUESTION_FIELDS), "PUBLIC_QUESTION_TEXT_REQUIRED")
    if "chronology_kind" in question:
        require(question["chronology_kind"] == "conversation_order", "PUBLIC_CHRONOLOGY_KIND")
    key = "items" if kind == "ordering" else "options"
    values = question[key]
    labels = {f"I{i}" for i in range(1, 7)} if kind == "ordering" else set("ABCDEF")
    require(isinstance(values, dict) and set(values) == labels and
            all(isinstance(value, str) and value.strip() for value in values.values()), "PUBLIC_QUESTION_LABELS_OR_TEXT_INVALID")


def validate_public_schema(public, phase):
    require(isinstance(public, dict), "PUBLIC_OBJECT_REQUIRED")
    if phase == "asset":
        require({"root_id", "source_sha256", "statements", "ordering_questions"} <= set(public) <= ASSET_FIELDS,
                "PUBLIC_ASSET_FIELDS_NOT_PUBLIC_SCHEMA")
        require(all(isinstance(public[k], str) and public[k].strip() for k in ("root_id", "source_sha256")), "PUBLIC_ASSET_BINDING_TEXT_REQUIRED")
        if "scope_scene_refs" in public:
            require(public["scope_scene_refs"] == list(SCENES), "PUBLIC_ASSET_SCOPE")
        if "multiselect_stem" in public:
            require(isinstance(public["multiselect_stem"], str) and public["multiselect_stem"].strip(), "PUBLIC_ASSET_STEM_REQUIRED")
        statements = public["statements"]
        require(isinstance(statements, list), "PUBLIC_STATEMENTS_LIST_REQUIRED")
        for statement in statements:
            require(isinstance(statement, dict) and set(statement) == {"statement_id", "proposition"} and
                    all(isinstance(v, str) and v.strip() for v in statement.values()), "PUBLIC_STATEMENT_FIELDS_NOT_PUBLIC_SCHEMA")
        require(len({s["statement_id"] for s in statements}) == len(statements), "PUBLIC_STATEMENT_ID_DUPLICATE")
        failures = public.get("ordering_failures", [])
        require(isinstance(failures, list), "PUBLIC_ORDERING_FAILURES_LIST_REQUIRED")
        for failure in failures:
            require(isinstance(failure, dict) and set(failure) == {"question_type", "slot_id", "candidate_id", "reason", "failure"} and
                    all(isinstance(v, str) and v.strip() for v in failure.values()) and failure["question_type"] == "ordering",
                    "PUBLIC_ORDERING_FAILURE_FIELDS_NOT_PUBLIC_SCHEMA")
        questions = public["ordering_questions"]
    else:
        require(phase == "public" and set(public) == {"questions"}, "PUBLIC_QUESTIONS_FIELDS_NOT_PUBLIC_SCHEMA")
        questions = public["questions"]
    require(isinstance(questions, list), "PUBLIC_QUESTION_LIST_REQUIRED")
    for question in questions:
        public_question(question, ordering_only=phase == "asset")
    require(len({q["candidate_id"] for q in questions}) == len(questions), "PUBLIC_QUESTION_ID_DUPLICATE")


def validate_plan(plan, root, source, source_sha256):
    require(isinstance(plan, dict) and set(plan) == PLAN_FIELDS, "PLAN_FIELDS_NOT_EXACT_PUBLIC_SCHEMA")
    require(plan["schema_version"] == SCHEMA, "PLAN_SCHEMA")
    require(plan["root_id"] == root and plan["source_sha256"] == source_sha256, "PLAN_ROOT_OR_SOURCE_BINDING")
    require(plan["scope_scene_refs"] == list(SCENES), "PLAN_SCOPE")
    turns = {row["turn_ref"]: row for row in source["turns"]}
    slots = plan["slots"]
    require(isinstance(slots, list) and len(slots) == 2, "PLAN_TWO_SLOTS_REQUIRED")
    require(all(isinstance(slot, dict) and set(slot) == SLOT_FIELDS for slot in slots), "SLOT_FIELDS_NOT_EXACT_PUBLIC_SCHEMA")
    require({slot["slot_id"] for slot in slots} == {"1", "2"}, "PLAN_SLOT_IDS")
    candidates = [slot["candidate_id"] for slot in slots]
    require(all(isinstance(cid, str) and cid.strip() for cid in candidates) and len(set(candidates)) == 2, "PLAN_DISTINCT_CANDIDATES")
    all_ids = []
    for slot in slots:
        anchors = slot["anchors"]
        require(isinstance(anchors, list) and len(anchors) == 6, "SLOT_SIX_ANCHORS_REQUIRED")
        require(all(isinstance(a, dict) and set(a) == ANCHOR_FIELDS for a in anchors), "ANCHOR_FIELDS_NOT_EXACT_PUBLIC_SCHEMA")
        require({a["item_id"] for a in anchors} == {f"I{i}" for i in range(1, 7)}, "SLOT_ITEM_IDS")
        refs = [a["turn_ref"] for a in anchors]
        require(len(set(refs)) == 6, "SLOT_DISTINCT_TURNS")
        scenes = []
        for anchor in anchors:
            require(isinstance(anchor["anchor_id"], str) and anchor["anchor_id"].strip(), "ANCHOR_ID_REQUIRED")
            turn = turns.get(anchor["turn_ref"])
            require(turn is not None, "ANCHOR_SOURCE_TURN_MISSING")
            require(anchor["speaker"] == turn["speaker"], "ANCHOR_SPEAKER_MISMATCH")
            require(anchor["full_turn_quote"] == turn["text"], "ANCHOR_FULL_TURN_MISMATCH")
            require(anchor["full_turn_sha256"] == text_sha(turn["text"]), "ANCHOR_FULL_TURN_HASH_MISMATCH")
            scenes.append(turn.get("scene_ref"))
            all_ids.append(anchor["anchor_id"])
        require(set(scenes) == set(SCENES), "SLOT_ONE_ANCHOR_PER_SCENE")
    require(len(all_ids) == len(set(all_ids)), "PLAN_DUPLICATE_ANCHOR_ID")
    return plan


def anchor_rows(plan):
    return [{"slot_id": slot["slot_id"], "candidate_id": slot["candidate_id"], **anchor}
            for slot in plan["slots"] for anchor in slot["anchors"]]


def candidates(plan):
    return {slot["candidate_id"]: {a["item_id"]: a for a in slot["anchors"]} for slot in plan["slots"]}


def validate_public(plan, public, phase):
    validate_public_schema(public, phase)
    questions = public.get("ordering_questions") if phase == "asset" else public.get("questions")
    require(isinstance(questions, list), "PUBLIC_QUESTION_LIST_REQUIRED")
    if phase == "public":
        questions = [q for q in questions if q.get("question_type") == "ordering"]
    by_id = candidates(plan)
    ids = [q.get("candidate_id") for q in questions]
    require(len(ids) == len(set(ids)) and set(ids) <= set(by_id), "PUBLIC_ORDERING_NOT_IN_PLAN")
    for question in questions:
        require(set(question.get("items", {})) == set(by_id[question["candidate_id"]]), "PUBLIC_ITEM_IDS_NOT_IN_PLAN")
    return by_id


def validate_review_anchor(item, planned):
    require(isinstance(item, dict), "REVIEW_ANCHOR_OBJECT_REQUIRED")
    require(item.get("anchor_id") == planned["anchor_id"], "REVIEW_ANCHOR_ID_MISMATCH")
    anchor = item.get("order_anchor")
    require(isinstance(anchor, dict), "REVIEW_ORDER_ANCHOR_REQUIRED")
    require(anchor.get("turn_ref") == planned["turn_ref"], "REVIEW_ANCHOR_OUTSIDE_FROZEN_TURN")
    require(anchor.get("speaker") == planned["speaker"], "REVIEW_ANCHOR_SPEAKER_MISMATCH")
    quote = anchor.get("quote")
    full = planned["full_turn_quote"]
    require(isinstance(quote, str) and bool(quote.strip()) and quote in full, "REVIEW_ANCHOR_QUOTE_NOT_EXACT")
    require(full.count(quote) == 1, "REVIEW_ANCHOR_QUOTE_NOT_UNIQUE")
    require(item.get("frozen_full_turn_sha256") == planned["full_turn_sha256"], "REVIEW_FULL_TURN_HASH_MISMATCH")


def resolve_public_source(batch, unit, value):
    """Resolve only an unambiguous source path; never import a private exporter."""
    p = Path(value)
    if p.is_absolute():
        require(p.is_file(), "RECORDED_ABSOLUTE_SOURCE_MISSING")
        return p.resolve()
    project = Path(batch).resolve().parent.parent
    found = {(base / p).resolve() for base in (Path(unit), Path(batch), project) if (base / p).is_file()}
    require(len(found) == 1, "RECORDED_RELATIVE_SOURCE_MISSING_OR_AMBIGUOUS")
    return found.pop()


def prefix_input_paths(unit):
    """Enumerate source-only lineage actually consumed by the scene adapter.

    The adapter validates its semantic/import rules; read/save additionally
    bind all inspected prefix files so their contents cannot change mid-read.
    No question draft, private answer or Solver file is permitted here.
    """
    unit = Path(unit)
    path = unit / "PREFIX_IMPORT.json"
    require(path.is_file(), "PREFIX_IMPORT_INPUT_REQUIRED")
    doc = json.loads(path.read_text(encoding="utf-8"))
    paths = [path]
    def source_only(candidate):
        candidate = Path(candidate).resolve()
        name = candidate.name.upper()
        require("PRIVATE" not in name and "GOLD" not in name and "SOLVER" not in name and name != "QA_DRAFT.JSON",
                "PREFIX_PRIVATE_INPUT_FORBIDDEN")
        return candidate
    for key in ("prefix_review_binding", "candidate_prefix_binding", "prefix_read_receipt_binding"):
        paths.append(source_only(doc[key]["path"]))
    review = json.loads(source_only(doc["prefix_review_binding"]["path"]).read_text(encoding="utf-8"))
    for key in ("read_context_binding", "input_render_binding", "raw_judgment_binding", "candidate_prefix_binding"):
        paths.append(Path(review[key]["path"]))
    paths.extend(Path(item["path"]) for item in review["input_bindings"])
    for row in doc["scenes"]:
        for item in row["verbatim_artifacts"].values():
            paths.extend(Path(item[key]["path"]) for key in ("historical_binding", "imported_binding"))
    for candidate in paths:
        source_only(candidate)
        require(candidate.is_file(), "PREFIX_INPUT_FILE_MISSING:" + str(candidate))
    return list(dict.fromkeys(p.resolve() for p in paths))
