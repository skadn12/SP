"""Opt-in Planner templates with typed relations and explicitly supplied feedback."""
from copy import deepcopy
import hashlib
import json

from ..h4_round1_r03_feedback_view_v2.renderers import normalize_parent
from ..h4_round1_r03_feedback_view_v2.templates import (
    EXPANSION_PLANNER_TEMPLATE_V4, PLANNER_DEVELOPMENT_GUIDANCE_V1,
    SHARED_FEEDBACK_RULES_V3, render,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import build_planner_prompt
from .input_views import project_supplement_inputs
from .planning import sha
from .prompts import TARGET_RULES, package
from .shared_planner_prompts import APPEND_TASK
from . import planner_relations


VERSION = "planner_minimal_relations_filtered_feedback_v1"
CAP9_VERSION = "planner_minimal_relations_filtered_feedback_target8_cap9_v1"

RELATION_DETAIL_RULES = """RELATION DETAIL IN EXISTING PLAN FIELDS
When choosing similar events, state updates, personal associations or cross-event
links, specify the involved facts and the distinguishing people, times, states or
relationships in the existing plan fields and interaction_beats. State what the
Realizer should make distinguishable in dialogue. Put necessary old facts in the
existing transition or state fields; do not merely ask to discuss further, increase
complexity or cause confusion. Keep third-person action summaries, not dialogue.
These are optional design directions, not requirements for every scene. Do not add
difficulty scores, confusion levels, planned QA answers or new story-data fields.

CONTINUITY DECLARED DURING THIS PLANNING RESPONSE
Use dependencies to identify candidates suitable for consecutive development,
their predecessor order, and any specific actual result required by a later scene.
Use natural_continuity with condition=null for a preferred natural sequence;
use hard_prerequisite with a concrete condition only when that actual result is
necessary. The two relation kinds are not interchangeable. Same-topic scenes,
shared participants and continuation labels alone do not establish dependencies.
History-established facts need no forced dependency on a new scene. Use existing
source_candidate_ref or simple output_ref pointers; the program owns formal IDs.
An empty relation list means undeclared, not proven independence. Do not add a
second group table or a duplicate graph. This preparation does not change batching.

TARGET COMPLETION ALIGNMENT
required_transition is a change to aim for, not an established fact. after_state is
a target and need not be fully reached within this scene. A new explicit decision,
arrangement or intermediate step can be valuable; reconfirming an old arrangement
is not new progress. An earlier partial result does not automatically block a
later scene. Only actual established prerequisites may support that later scene.
"""

INITIAL_RELATION_DETAIL_RULES = RELATION_DETAIL_RULES.replace(
    "History-established facts need no forced dependency on a new scene. Use existing\nsource_candidate_ref or simple output_ref pointers; the program owns formal IDs.",
    "History-established facts need no forced dependency on a new scene. In initial mode,\nset every source_candidate_ref to null and use only an earlier output_ref pointer.\nA selected_continuation continues its verified historical track; it does not need a\ncurrent-candidate dependency merely because it is a continuation. Leave dependencies\nempty unless a real earlier candidate in this response must precede it. Never self-reference.\nMemory event IDs such as ME01 and track IDs such as VT01 are not Planner candidate refs.")


def _feedback(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("explicit_planner_feedback_view_required")
    # The selected export adapter owns availability and source qualification.
    # This renderer preserves its explicit no-feedback state as well as records.
    json.dumps(value, allow_nan=False)
    return deepcopy(value)


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("planner_template_source_block_changed")
    return text.replace(old, new, 1)


def _aligned_initial_instructions(text):
    start = text.index("For all developments, before_state -> required_transition")
    end = text.index("Do not output development_mode, scene IDs", start)
    text = text[:start] + TARGET_RULES + "\n" + text[end:]
    old = ("and work circumstances. It must be a genuinely new event or state change, not a\n"
           "restated preference, personality explanation, past discussion, confirmation,\n"
           "generic chat, or merely something that may happen later.")
    new = ("and work circumstances. Aim for a genuinely new event, state change, explicit\n"
           "decision, arrangement or intermediate step. Repeating a preference, past\n"
           "discussion or an already established arrangement is not new progress.")
    return _replace_once(text, old, new)


def _identity(mode, *, max_new_candidates=8):
    cap9 = max_new_candidates == 9
    contract_version = planner_relations.CAP9_VERSION if cap9 else planner_relations.VERSION
    schema = (planner_relations.initial_output_schema(max_new_candidates=max_new_candidates, contract_version=contract_version) if mode == "initial"
              else planner_relations.output_schema())
    return {"output_contract": contract_version,
            "schema_sha256": sha(schema),
            "render_version": CAP9_VERSION if cap9 else VERSION, "execution_mode": mode,
            "strict_decoding_added": False,
            "development_guidance_version": "planner_development_guidance_v1"}


def render_initial_planner(*, feedback_view, dialogue, memory_overview=None,
                           memory_candidates=(), verified_tracks=(), speaker_registry=None,
                           memory_only_fast=True, expose_complete_memory=True, max_new_candidates=8):
    view = _feedback(feedback_view)
    base = build_planner_prompt(
        parent=normalize_parent(dialogue), memory_overview=dict(memory_overview or {}),
        candidates=list(memory_candidates), verified_tracks=list(verified_tracks),
        memory_only_fast=memory_only_fast, event_bucket_only=True,
        memory_source_mode="event_bucket_only", speaker_registry=speaker_registry,
        strict_speaker_refs=True, expose_complete_memory=expose_complete_memory,
        include_authoritative_transcript=True, max_continuations=3,
        max_new_development_candidates=max_new_candidates, max_plan_variants=1, scene_quota=6,
        output_contract_text=planner_relations.render_output_contract(mode="initial", max_new_candidates=max_new_candidates))
    # Touch only builder-owned instruction text, never dynamic history or Memory.
    marker = "COMPLETE MEMORAX MEMORY\n" if expose_complete_memory else "CANDIDATE MEMORY PLANNING PACKETS\n"
    prefix, separator, data_and_contract = base["prompt"].partition(marker)
    if not separator:
        raise ValueError("planner_initial_instruction_boundary_missing")
    prefix = _aligned_initial_instructions(prefix)
    role = ("Execution mode: initial. Return the versioned shared Planner object. "
            "Candidate source_candidate_ref must be null. Each continuation has a unique actual "
            "verified_track_ref from the menu and may have an empty dependencies array. Do not add a "
            "current-candidate dependency unless a real earlier candidate is required; never self-reference. "
            "Return at most three continuations and at most eight new-development candidates. "
            "If you have more ideas, rank them and omit the extras; never output a ninth candidate. "
            "Relations may reference only earlier output_ref rows; the initial source registry "
            "does not yet exist. Declare relations in this response, without an extra grouping call.")
    if max_new_candidates == 9:
        role = role.replace(
            "Return at most three continuations and at most eight new-development candidates. "
            "If you have more ideas, rank them and omit the extras; never output a ninth candidate. ",
            "Return at most three continuations. Target eight new-development candidates; fewer are allowed. "
            "The hard maximum is nine new-development candidates. An optional ninth candidate is tolerated "
            "only for a distinct, useful additional development; do not pad the pool or aim to fill nine. "
            "Rank candidates and omit further ideas; never output ten. ")
    suffix = render(EXPANSION_PLANNER_TEMPLATE_V4, {
        "ROLE_SPECIFIC_COMMON_FEEDBACK_VIEW": json.dumps(view, ensure_ascii=False, sort_keys=True, indent=2),
        "AUTHORITATIVE_CUMULATIVE_DIALOGUE": (
            "The authoritative cumulative transcript is included above by the active Planner builder."),
        "EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT": role,
    })
    text = prefix + separator + data_and_contract + "\n\n" + suffix + "\n\n" + INITIAL_RELATION_DETAIL_RULES
    if text.count(PLANNER_DEVELOPMENT_GUIDANCE_V1) != 1:
        raise ValueError("planner_guidance_not_exactly_once")
    template_version = CAP9_VERSION if max_new_candidates == 9 else VERSION
    built = package("initial_planner." + template_version, text, {})
    built.update(feedback_view=view, planner_contract_identity=_identity("initial", max_new_candidates=max_new_candidates),
                 planner_max_tokens=8192)
    return built


def _append_task():
    task = _replace_once(APPEND_TASK,
        "they are reference context, not the new action to output. Its empty dependencies\n"
        "is not a valid continuation dependency list. Derive the added phase and actual\n"
        "prerequisites from the selected source and the supplied history.",
        "they are reference context, not the new action to output. Derive the added phase\n"
        "and any declared natural continuity or actual prerequisites from the selected\n"
        "source and the supplied history. Empty dependencies means no relation declared.")
    task = _replace_once(task,
        "order. Each continuation requires at least one concrete prerequisite. Independent\n"
        "new developments may use an empty dependencies array. Each dependency references",
        "order. Declare natural continuity separately from hard prerequisites using\n"
        "relation_kind. Any candidate may have empty dependencies. Each relation references")
    task = _replace_once(task,
        "State the concrete condition that actual retained dialogue must establish.\n"
        "Dependency existence or source retention alone does not prove that condition.",
        "Only hard_prerequisite requires a condition that actual retained dialogue must\n"
        "establish; natural_continuity has condition=null. Neither source retention nor\n"
        "a declared dependency proves that a hard condition actually holds.")
    return task


def render_supplement(inputs, *, continuation_source_templates, feedback_view=None):
    if "shared_feedback_view" not in inputs:
        raise ValueError("supplement_inherited_feedback_required")
    view = _feedback(inputs["shared_feedback_view"])
    if feedback_view is not None and _feedback(feedback_view) != view:
        raise ValueError("supplement_feedback_differs_from_initial")
    projected, mapping = project_supplement_inputs(inputs, extra_reference_fields=("memory_navigation",))
    template = (SHARED_FEEDBACK_RULES_V3 + "\n" + TARGET_RULES + "\n" + _append_task()
                + "\n" + PLANNER_DEVELOPMENT_GUIDANCE_V1 + "\n" + RELATION_DETAIL_RULES
                + "\n" + planner_relations.render_output_contract(mode="append_only"))
    if template.count(PLANNER_DEVELOPMENT_GUIDANCE_V1) != 1:
        raise ValueError("planner_guidance_not_exactly_once")
    built = package("supplement." + VERSION, template + "\nCONTINUATION SOURCE TEMPLATES\n"
        + json.dumps(continuation_source_templates, ensure_ascii=False, indent=2), projected)
    identity = _identity("append_only")
    built.update(feedback_view=view, planner_contract_identity=identity,
                 input_projection_mapping=mapping, planner_max_tokens=8192)
    built["supplement_contract_identity"] = {
        **identity, "prompt_version": built["prompt_version"], "source_scope": inputs["source_scope"],
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "allowed_source_view": deepcopy(continuation_source_templates),
        "allowed_source_view_sha256": sha(continuation_source_templates)}
    return built
