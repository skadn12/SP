from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Callable, Mapping, Sequence

from .backends import BackendResponse, QuestionerRequest
from .core import atomic_json, canonical_json, sha256_bytes
from .ledger import RequestLedger
from .questioner_contract import build_span_catalog, span_catalog_index


THREE_SCOPE_CANARY_VERSION = "three_scope_questioner_canary_v1"
THREE_SCOPE_PROMPT_VERSION = "spell_dialogue_mcq_3x_three_scope_lineage_canary_v1"
THREE_SCOPE_SCHEMA_VERSION = "questioner_three_scope_span_ids_v1"
THREE_SCOPE_CONTRACT_VERSION = "questioner_qa_scope_contract_v1"
SCHEMA_ONLY_RETRY_VERSION = "questioner_exactly_three_distractors_retry_v1"

PARENT_LOCAL = "parent_local"
CHILD_LOCAL = "child_local"
PARENT_CHILD_JOINT = "parent_child_joint"
SCOPE_ASSIGNMENTS = (PARENT_LOCAL, CHILD_LOCAL, PARENT_CHILD_JOINT)

SCOPED_ITEM_JSON_SCHEMA: dict[str, Any] = {
    "type":"object", "additionalProperties":False,
    "required":["qa_id", "qa_scope", "question", "correct_answer_text",
                "distractors", "answer_support_span_ids"],
    "properties":{
        "qa_id":{"type":"string", "minLength":1},
        "qa_scope":{"type":"string", "enum":list(SCOPE_ASSIGNMENTS)},
        "question":{"type":"string", "minLength":1},
        "correct_answer_text":{"type":"string", "minLength":1},
        "distractors":{"type":"array", "minItems":3, "maxItems":3,
                       "items":{"type":"string", "minLength":1}},
        "answer_support_span_ids":{"type":"array", "minItems":1,
                                   "items":{"type":"string", "minLength":1},
                                   "uniqueItems":True},
        "reasoning_steps":{"type":"array", "items":{"type":"string"}},
    },
}
SCOPED_BATCH_JSON_SCHEMA = {"type":"array", "minItems":3, "maxItems":3,
                            "items":SCOPED_ITEM_JSON_SCHEMA}

SCHEMA_ONLY_RETRY_SUFFIX = """Your previous batch was valid JSON, but one or more QA items did not contain exactly three distractors.

Regenerate the complete batch from scratch. Each QA item must contain exactly three distractors. Preserve the assigned qa_scope for each array position.

Do not repair or continue the previous response. Do not include Markdown, explanations, comments, or text outside the JSON array. Ensure all arrays and objects are fully closed before returning."""

FORMAT_ONLY_RETRY_SUFFIX = """Your previous response was not valid JSON.

Regenerate the complete batch from scratch as exactly one valid JSON array containing exactly three objects that follow the required schema and preserve the assigned qa_scope for each array position.

Do not repair or continue the previous response. Do not include Markdown, explanations, comments, or text outside the JSON array. Ensure all arrays and objects are fully closed before returning."""


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def schema_only_retry_suffix_hash() -> str:
    return _hash(SCHEMA_ONLY_RETRY_VERSION + SCHEMA_ONLY_RETRY_SUFFIX)


def format_only_retry_suffix_hash() -> str:
    return _hash(FORMAT_ONLY_RETRY_SUFFIX)


def three_scope_template_hash() -> str:
    material = (THREE_SCOPE_PROMPT_VERSION + THREE_SCOPE_SCHEMA_VERSION +
                inspect.getsource(build_three_scope_prompt) +
                canonical_json(SCOPED_BATCH_JSON_SCHEMA))
    return _hash(material)


def _turn_position(turns: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {str(turn["turn_id"]):index for index, turn in enumerate(turns)}


def compile_arc_lineage_views(*, outline: Mapping[str, Any],
                              parent_turns: Sequence[dict[str, Any]],
                              child_turns: Sequence[dict[str, Any]],
                              scene_contracts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    parent = [dict(turn) for turn in parent_turns]
    child = [dict(turn) for turn in child_turns]
    parent_positions = _turn_position(parent)
    anchors = {str(anchor["anchor_id"]):str(anchor["turn_id"])
               for anchor in outline.get("parent_anchors") or []}
    contracts_by_session = {
        f"session_{int(contract['scene_index']):02d}":dict(contract)
        for contract in scene_contracts
    }
    views = []
    for arc in outline.get("planned_event_arcs") or []:
        event_id = str(arc["event_id"])
        anchor_turn_ids = [anchors[str(anchor_id)] for anchor_id in
                           arc.get("parent_anchor_ids") or []]
        selected_parent_positions: set[int] = set()
        for turn_id in anchor_turn_ids:
            if turn_id not in parent_positions:
                raise ValueError("lineage_view_parent_anchor_missing")
            position = parent_positions[turn_id]
            session_id = parent[position]["session_id"]
            selected_parent_positions.add(position)
            for neighbor in (position - 1, position + 1):
                if (0 <= neighbor < len(parent) and
                        parent[neighbor]["session_id"] == session_id):
                    selected_parent_positions.add(neighbor)
        selected_parent = [parent[index] for index in sorted(selected_parent_positions)]
        child_sessions = {session_id for session_id, contract in contracts_by_session.items()
                          if event_id in contract.get("assigned_event_arc_ids", [])}
        selected_child = [turn for turn in child if turn["session_id"] in child_sessions]
        if not selected_parent or not selected_child:
            raise ValueError("lineage_view_partition_empty")
        expected_stages = [f"{event_id}:stage_{index:02d}" for index in range(1, 5)]
        covered_stages = sorted({stage for session_id in child_sessions
            for stage in contracts_by_session[session_id].get("assigned_event_stage_ids", [])
            if str(stage).startswith(f"{event_id}:")})
        if covered_stages != expected_stages:
            raise ValueError("lineage_view_event_stage_coverage_invalid")
        combined = selected_parent + selected_child
        catalog = build_span_catalog(combined)
        parent_span_ids = [f"{turn['turn_id']}#S0001" for turn in selected_parent]
        child_span_ids = [f"{turn['turn_id']}#S0001" for turn in selected_child]
        view = {"three_scope_canary_version":THREE_SCOPE_CANARY_VERSION,
            "scope_contract_version":THREE_SCOPE_CONTRACT_VERSION,
            "event_arc_id":event_id, "event_arc_summary":str(arc.get("summary") or ""),
            "parent_anchor_ids":list(arc.get("parent_anchor_ids") or []),
            "parent_anchor_turn_ids":anchor_turn_ids,
            "parent_bridge_turn_ids":[turn["turn_id"] for turn in selected_parent
                                      if turn["turn_id"] not in anchor_turn_ids],
            "child_session_ids":sorted(child_sessions),
            "covered_event_stage_ids":covered_stages,
            "selected_parent_turn_count":len(selected_parent),
            "selected_child_turn_count":len(selected_child),
            "selected_parent_span_ids":parent_span_ids,
            "selected_child_span_ids":child_span_ids,
            "combined_dialogue":combined, "span_catalog":catalog}
        view["view_sha256"] = sha256_bytes(canonical_json(view).encode())
        views.append(view)
    if [view["event_arc_id"] for view in views] != ["E1", "E2", "E3", "E4"]:
        raise ValueError("lineage_view_event_arc_set_invalid")
    return views


def build_three_scope_prompt(*, lineage_view: Mapping[str, Any],
                             assigned_gold_letters: Sequence[str],
                             solver_feedback: Sequence[Mapping[str, Any]]) -> str:
    letters = [str(letter).upper() for letter in assigned_gold_letters]
    if len(letters) != 3 or any(letter not in "ABCD" for letter in letters):
        raise ValueError("three_scope_target_letters_invalid")
    catalog = span_catalog_index(dict(lineage_view["span_catalog"]))
    parent_ids = set(lineage_view["selected_parent_span_ids"])
    child_ids = set(lineage_view["selected_child_span_ids"])
    compact_spans = [{"span_id":span_id,
                      "source_partition":"parent" if span_id in parent_ids else "child",
                      "speaker":span["speaker"], "exact_text":span["exact_text"]}
                     for span_id, span in catalog.items()]
    assignments = [{"item_index":index, "qa_scope":scope}
                   for index, scope in enumerate(SCOPE_ASSIGNMENTS)]
    feedback_fields = ("qa_id", "final_decision", "no_context_rewards",
                       "full_context_rewards", "diagnosis")
    feedback = [{key:item.get(key) for key in feedback_fields if key in item}
                for item in solver_feedback]
    payload = {"event_arc_id":lineage_view["event_arc_id"],
        "event_arc_summary_for_navigation_only":lineage_view["event_arc_summary"],
        "scope_assignments_by_array_position":assignments,
        "local_target_letters_do_not_output":letters,
        "lineage_span_catalog":compact_spans,
        "solver_feedback_for_quality_only":feedback}
    rules = [
        "Return exactly one valid JSON array containing exactly three new QA objects in the assigned array order.",
        "Copy the assigned qa_scope exactly into each object: item 0 parent_local, item 1 child_local, item 2 parent_child_joint.",
        "For parent_local, every answer_support_span_id must come from the parent partition.",
        "For child_local, every answer_support_span_id must come from the child partition.",
        "For parent_child_joint, select at least one parent span and at least one child span; removing either partition must make the answer underdetermined.",
        "The span catalog is the sole factual source. The event summary is navigation metadata, not answer evidence. Solver feedback is quality-only and not factual evidence.",
        "Choose the facts, question, correct answer, exactly three distractors, optional reasoning steps, and support span IDs yourself within the assigned scope.",
        "Each item must contain exactly three distinct distractors: never two, four, or more.",
        "Do not output options, answer letters, references, evidence values, supporting quotes, difficulty, reasoning type, turn IDs, session IDs, or fields outside the schema.",
        "Do not impose a fixed number of spans, sessions, or reasoning steps beyond the assigned partition scope.",
        "Avoid questions guessable without dialogue and avoid direct one-sentence lookup when a state change, temporal order, cause, commitment, correction, or consequence is available.",
        "Return JSON only, without Markdown, comments, code fences, or surrounding text.",
    ]
    return "\n\n".join([f"PROMPT_VERSION: {THREE_SCOPE_PROMPT_VERSION}",
        "THREE-SCOPE RULES\n- " + "\n- ".join(rules),
        "DYNAMIC LINEAGE INPUT\n" + json.dumps(payload, ensure_ascii=False,
                                                  sort_keys=True, indent=2),
        "BATCH JSON SCHEMA\n" + json.dumps(SCOPED_BATCH_JSON_SCHEMA,
                                             ensure_ascii=False, sort_keys=True, indent=2)])


def batch_distractor_schema_valid(parsed: Any) -> bool:
    return (isinstance(parsed, list) and len(parsed) == 3 and
            all(isinstance(item, dict) and isinstance(item.get("distractors"), list) and
                len(item["distractors"]) == 3 for item in parsed))


def validate_scope_item(item: Any, *, item_index: int,
                        lineage_view: Mapping[str, Any]) -> dict[str, Any]:
    expected = SCOPE_ASSIGNMENTS[item_index]
    parent_ids = set(lineage_view["selected_parent_span_ids"])
    child_ids = set(lineage_view["selected_child_span_ids"])
    if not isinstance(item, dict):
        return {"decision":"qa_scope_not_followed", "expected_scope":expected,
                "reported_scope":None, "reasons":["scope_item_not_object"],
                "selected_parent_span_count":0, "selected_child_span_count":0}
    span_ids = item.get("answer_support_span_ids")
    spans = span_ids if isinstance(span_ids, list) else []
    parent_count = sum(span_id in parent_ids for span_id in spans)
    child_count = sum(span_id in child_ids for span_id in spans)
    unknown_count = sum(span_id not in parent_ids | child_ids for span_id in spans)
    reasons = []
    if item.get("qa_scope") != expected:
        reasons.append("assigned_qa_scope_mismatch")
    if expected == PARENT_LOCAL and (not spans or child_count or unknown_count):
        reasons.append("parent_local_support_out_of_scope")
    elif expected == CHILD_LOCAL and (not spans or parent_count or unknown_count):
        reasons.append("child_local_support_out_of_scope")
    elif expected == PARENT_CHILD_JOINT and (
            parent_count < 1 or child_count < 1 or unknown_count):
        reasons.append("joint_support_missing_partition")
    return {"decision":"qa_scope_not_followed" if reasons else "qa_scope_followed",
        "expected_scope":expected, "reported_scope":item.get("qa_scope"),
        "reasons":sorted(set(reasons)),
        "selected_parent_span_count":parent_count,
        "selected_child_span_count":child_count,
        "selected_span_count":len(spans), "unknown_span_count":unknown_count}


def strip_scope_field(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    return {key:value for key,value in item.items() if key != "qa_scope"}


def run_scoped_questioner_batch(*, ledger: RequestLedger, backend: Any,
                                semantic_prompt: str, event_arc_id: str,
                                assigned_gold_letters: tuple[str, str, str],
                                lineage_view: Mapping[str, Any],
                                prompt_protocol: str,
                                backend_identity: Mapping[str, Any],
                                budget_limit: int = 2) -> dict[str, Any]:
    semantic_sha = _hash(semantic_prompt)
    attempts = []
    final_parsed: Any = None
    retry_suffix: str | None = None
    retry_reason: str | None = None
    for attempt_index in (0, 1):
        prompt = semantic_prompt if attempt_index == 0 else (
            semantic_prompt + "\n\n" + str(retry_suffix))
        request_id = f"questioner-{event_arc_id}-a{attempt_index:02d}"
        atomic_json(ledger.root.parent / f"raw/prompts/{request_id}.json", {
            "request_id":request_id, "attempt_index":attempt_index,
            "semantic_prompt_sha256":semantic_sha,
            "retry_reason":retry_reason if attempt_index else None,
            "retry_suffix_sha256":_hash(str(retry_suffix)) if attempt_index else None,
            "rendered_prompt_sha256":_hash(prompt), "prompt":prompt})
        request = QuestionerRequest(request_id, int(event_arc_id[1:]) - 1,
            assigned_gold_letters, prompt, _hash(prompt),
            tuple(turn["turn_id"] for turn in lineage_view["combined_dialogue"]),
            tuple(sorted({turn["session_id"] for turn in
                          lineage_view["combined_dialogue"]})),
            dict(lineage_view["span_catalog"]), tuple())
        metadata = {"event_arc_id":event_arc_id, "attempt_index":attempt_index,
            "assigned_gold_letters":list(assigned_gold_letters),
            "assigned_qa_scopes":list(SCOPE_ASSIGNMENTS),
            "semantic_prompt_sha256":semantic_sha,
            "lineage_view_sha256":lineage_view["view_sha256"],
            "prompt_protocol":prompt_protocol,
            "retry_of_request_id":f"questioner-{event_arc_id}-a00"
                if attempt_index else None,
            "retry_reason":retry_reason if attempt_index else None,
            "retry_mode":"full_batch_regeneration" if attempt_index else None,
            "schema_retry_suffix_sha256":_hash(str(retry_suffix))
                if attempt_index and retry_reason == "distractor_count_invalid" else None,
            "physical_request_ordinal":attempt_index + 1,
            "maximum_physical_request_count":2}
        response = ledger.invoke(request_id=request_id, intent="questioner_three_scope_batch",
            attempt=attempt_index + 1, prompt_sha256=request.prompt_sha256,
            call=lambda request=request: backend.generate_batch(request),
            budget_class="questioner", budget_limit=budget_limit,
            total_budget_limit=budget_limit, request_metadata=metadata,
            backend_identity=dict(backend_identity))
        try:
            parsed = json.loads(response.raw_text)
            parse_status = "json_parsed"
        except json.JSONDecodeError:
            parsed = None
            parse_status = "json_invalid"
        distractors_valid = batch_distractor_schema_valid(parsed)
        if parse_status == "json_invalid":
            decision = "retry_format_only" if attempt_index == 0 else "questioner_json_invalid_exhausted"
            if attempt_index == 0:
                retry_suffix, retry_reason = FORMAT_ONLY_RETRY_SUFFIX, "json_invalid"
        elif not distractors_valid:
            decision = "retry_schema_only" if attempt_index == 0 else "questioner_schema_invalid_exhausted"
            if attempt_index == 0:
                retry_suffix = SCHEMA_ONLY_RETRY_SUFFIX
                retry_reason = "distractor_count_invalid"
        else:
            decision = "continue_to_scope_and_minimal_admission"
            final_parsed = parsed
        ledger.record_decision(request_id, decision)
        state = ledger.read(request_id) or {}
        attempts.append({"attempt_index":attempt_index, "request_id":request_id,
            "parse_status":parse_status,
            "raw_object_count":len(parsed) if isinstance(parsed, list) else 0,
            "distractor_schema_valid":distractors_valid, "decision":decision,
            "raw_response_path":state.get("raw_response_path"),
            "raw_response_sha256":state.get("raw_response_sha256")})
        if decision == "continue_to_scope_and_minimal_admission" or attempt_index == 1:
            break
    final = attempts[-1]
    return {"event_arc_id":event_arc_id, "semantic_prompt_sha256":semantic_sha,
        "attempts":attempts, "physical_questioner_calls":len(attempts),
        "format_retry_used":any(row["decision"] == "retry_format_only" for row in attempts),
        "schema_retry_used":any(row["decision"] == "retry_schema_only" for row in attempts),
        "final_decision":final["decision"], "raw_batch":final_parsed,
        "content_admission_evaluated":final_parsed is not None}


def classify_joint_necessity(*, no_context_rewards: Sequence[int] | None,
                             parent_only_rewards: Sequence[int] | None,
                             child_only_rewards: Sequence[int] | None,
                             full_context_rewards: Sequence[int] | None) -> dict[str, Any]:
    def stable(values: Sequence[int] | None, target: int) -> bool:
        return isinstance(values, (list, tuple)) and len(values) == target and all(
            value == 1 for value in values)
    parent_sufficient = stable(parent_only_rewards, 3)
    child_sufficient = stable(child_only_rewards, 3)
    no_context_sufficient = stable(no_context_rewards, 3)
    full_sufficient = stable(full_context_rewards, 5)
    if parent_sufficient:
        decision = "parent_only_sufficient"
    elif child_sufficient:
        decision = "child_only_sufficient"
    elif not no_context_sufficient and full_sufficient:
        decision = "lineage_jointly_required"
    else:
        decision = "joint_necessity_uncertified"
    return {"decision":decision, "parent_only_sufficient":parent_sufficient,
        "child_only_sufficient":child_sufficient,
        "no_context_sufficient":no_context_sufficient,
        "full_context_sufficient":full_sufficient}
