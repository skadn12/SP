"""Versioned task text; shared feedback definition is imported, not forked."""
import json
from copy import deepcopy

from ..h4_round1_r03_feedback_view_v2.templates import SHARED_FEEDBACK_RULES_V3, REALIZER_CONTENT_GUIDANCE_V1
from ..h4_round1_r03_feedback_view_v2.renderers import render_expansion_planner
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import build_semantic_gate_context
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import (
    build_semantic_continuation_gate_prompt, REALIZER_PARTICIPANT_RULES,
    REALIZER_TEXT_FREE_STYLE_RULES, realizer_arc_turn_rules,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.realizer_style import build_realizer_style_context
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.gate_output_contract import FULL_PROMPT_VERSION
from .supplement_contract_alignment import (
    LEGACY_VERSION, ALIGNED_VERSION, CONTRACT_ALIGNMENT_RULES,
    select_supplement_version, allowed_source_view, contract_sidecar,
)

TARGET_RULES = """PLANNED TARGET / ACTUAL PROGRESS POLICY v1
The original goal and planned after_state/durable_outcome are desired targets,
not facts. Keep the original final goal unchanged, even if it remains partial
or not realized. A scene may establish genuine new facts, decisions, arrangements
or intermediate progress without completing that final goal. Repeating an old
intention or generic discussion is not progress. Do not invent past actions.
Memory state descriptions and Planner before_state are hypotheses/navigation;
the authoritative history and retained actual dialogue establish historical facts.
Keep actor ownership fixed. Never assume that an earlier retained scene completed
its goal: explicit dependency conditions require actual dialogue evidence.
Interaction beats are third-person action summaries, never prewritten dialogue.
"""

SUPPLEMENT_TEMPLATE = """You are the incremental Planner for ONE effective plan.
This is the sole pre-Gate supplement opportunity for this root/round. Return only
an incremental patch, never another full plan or alternative plan_variants.
Original plan version/SHA, stable candidate IDs, original goals, diagnostic gaps,
authoritative complete history, retained actual dialogue (possibly empty), budget
and actor/event bindings are supplied below. No retained text means no completed
R03 stages; do not invent any. Unrelated candidates and original goals are immutable.
Add zero or more follow-on phases for affected events only. Reuse the supplied
source_candidate_id/ME/VT identities; do not invent an event to bypass limits.
Use unique phase:<identifier> IDs. Every added phase names a predecessor phase
(initial:<source_candidate_id> or an earlier added phase) and a concrete condition
that must actually hold before execution. Several followups are allowed; no quota.
Each phase uses 3-7 action-summary interaction_beats, not dialogue. The supplied
budget is an upper bound, not an instruction to fill it. No safe supplement is legal.
Return exactly: {"status":"PATCH" or "NO_SUPPLEMENT", "original_plan_sha256":"...",
"phases":[{"phase_id":"phase:...", "source_candidate_id":"...",
"original_goal":"EXACT original goal", "required_transition":"new action",
"interaction_beats":["...","...","..."],
"dependencies":[{"phase_id":"initial:...", "condition":"actual prerequisite"}]}],
"reason":"..."}. NO_SUPPLEMENT requires an empty phases array.
"""

REALIZER_TEMPLATE = """You are the Natural Realizer, not a Planner or QA Writer.
Only output NEW dialogue for the supplied phase plans, in input order. Do not
copy history, plans or interaction beats verbatim. Never output complete history.
Preserve actor bindings and required participants. ACTOR BINDINGS refers to
actor_bindings; participants supplies required_participants when that field is
absent. Each required participant must speak in the corresponding scene.
New people follow the original declarations and bindings. Distinguish intention,
preparation, confirmation and completion; do not claim an unobserved final result.
Only use the selected phase's interaction_beats as action summaries, not scripts.
Let later scenes follow the retained_actual_prefix and what actually happens
in earlier output scenes in this batch. Do not rewrite that prefix, replay an
established event, or assume a planned after_state became true. Rejected earlier
attempts are not context. Keep explicit dependencies tied to actual evidence.
Exactly one output scene per input scene, at most two scenes in this request.
""" + REALIZER_PARTICIPANT_RULES + "\n" + realizer_arc_turn_rules(actual_progress_only=True) + """
Return ONLY JSON matching the scene-count example below. Its turn arrays are
abbreviated field examples; expand EVERY scene to the required turn count.
Do not add output fields. Scene/turn IDs are assigned locally.
""" + "\n" + REALIZER_CONTENT_GUIDANCE_V1 + "\n"


def package(role, text, data):
    prompt = text + "\n\nINPUT DATA\n" + json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
    return {"prompt_version": "incremental_retention_v1." + role,
            "prompt": prompt, "messages": [{"role": "user", "content": prompt}]}


def render_supplement(inputs, *, prompt_version=LEGACY_VERSION):
    from .input_views import project_supplement_inputs
    select_supplement_version({"supplement_prompt_version": prompt_version})
    projected, mapping = project_supplement_inputs(inputs)
    template = SHARED_FEEDBACK_RULES_V3 + "\n" + TARGET_RULES + SUPPLEMENT_TEMPLATE
    text = template
    if prompt_version == ALIGNED_VERSION:
        template += "\n" + CONTRACT_ALIGNMENT_RULES
        rows = allowed_source_view(projected)
        text = template + "\nALLOWED SUPPLEMENT SOURCES\n" + json.dumps(rows, ensure_ascii=False, indent=2)
    built = package("supplement.readiness_v2", text, projected)
    built["prompt_version"] = prompt_version
    built["input_projection_mapping"] = mapping
    if prompt_version == ALIGNED_VERSION:
        built["supplement_contract_identity"] = contract_sidecar(prompt_version, template, rows)
    return built


def render_realizer(*, parent, plan, mode="text_free", continuity_shape=None, promoted_prefix=()):
    if mode not in {"text_free", "full_parent"} or not 1 <= len(plan["scenes"]) <= 2:
        raise ValueError("realizer_mode_or_batch_limit")
    scene_count = len(plan["scenes"])
    count_rules = (
        f"This request contains exactly {scene_count} input scene plans. "
        f"Return exactly {scene_count} separate objects in the scenes array.\n"
        "Never merge scenes, omit an input scene, or stop after an earlier scene. "
        "Keep the output scenes in the same order as the input plans.\n"
        + "\n".join(
            f"Output scenes[{index}] must realize only planned_phases_not_facts[{index}]."
            for index in range(scene_count))
        + f"\nBefore returning, check len(scenes) == {scene_count} and verify that "
        "each scene has its own complete dialogue_turns array with exactly 14 "
        "items unless a longer 15-18 turn arc is required. Never fewer than 9.\n"
    )
    example = {"scenes": [{"dialogue_turns": [{"speaker": "...", "text": "..."}]}
                          for _ in range(scene_count)]}
    count_rules += "EXACT SCENE-COUNT JSON EXAMPLE\n" + json.dumps(example, ensure_ascii=False, indent=2) + "\nEND SCENE-COUNT EXAMPLE\n"
    data = {"root_id": parent["root_id"], "planned_phases_not_facts": deepcopy(plan["scenes"]),
            "retained_actual_prefix": list(promoted_prefix)}
    if mode == "full_parent":
        data["authoritative_history"] = parent["dialogue"]["turns"]
        boundary = "Full authoritative history is provided. Output only new text.\n"
    else:
        # Fixed field allowlist prevents caller accidentally re-injecting text.
        shape = dict(continuity_shape or {})
        permitted = {"speakers", "turn_count", "median_turn_words", "session_count"}
        if set(shape) - permitted or any(isinstance(v, (dict, list)) for k, v in shape.items() if k != "speakers"):
            raise ValueError("text_free_shape_fields")
        expected_speakers = sorted({r["speaker"] for r in parent["dialogue"]["turns"]})
        if shape.get("speakers", expected_speakers) != expected_speakers:
            raise ValueError("text_free_shape_speakers")
        data["text_free_continuity_shape"] = shape
        # Reuse the successful route's local statistics, never caller-supplied
        # nested style objects or historical wording.
        style = build_realizer_style_context(parent["dialogue"]["turns"])
        data["text_free_continuity_tail"] = style["continuity_tail"]
        data["text_free_style_profile"] = style["style_profile"]
        boundary = REALIZER_TEXT_FREE_STYLE_RULES + (
            " The text_free_continuity_tail and text_free_style_profile contain "
            "local form statistics only, not story facts. Plans constrain the "
            "selected developments; they do not establish historical facts.\n")
    return package("realizer.successful_prompt_alignment_v3." + mode,
                   TARGET_RULES + REALIZER_TEMPLATE + count_rules + boundary, data)


def render_gate(*, parent, gate_scene, candidates):
    context = build_semantic_gate_context(parent=parent, candidates=candidates,
        plan={"scenes": [gate_scene]}, memory_only_fast=False, minimum_pass_candidates=1)
    # Full actual history is included for this policy, not a Memory-only claim.
    context["authoritative_complete_history"] = parent["dialogue"]["turns"]
    built = build_semantic_continuation_gate_prompt(gate_context=context, prompt_version=FULL_PROMPT_VERSION)
    result = package("gate.full_history_contract_v2", TARGET_RULES + "\n" + built["prompt"], {})
    result["gate_contract_identity"] = built["gate_contract_identity"]
    return result


def render_initial_planner(*, planner_contract_version="legacy_initial_v1", **kwargs):
    """Default to the restored initial contract; later contracts stay explicit."""
    from .shared_planner_contract import VERSION, RENDER_VERSION, render_output_contract, output_schema
    from .planning import sha
    if planner_contract_version not in {VERSION, "legacy_initial_v1"}:
        raise ValueError("unknown_initial_planner_contract")
    shared = planner_contract_version == VERSION
    if shared:
        kwargs["strict_speaker_refs"] = True
        kwargs["planner_output_contract_text"] = render_output_contract()
        kwargs["role_contract"] = (
            "Execution mode: initial. Return the shared Planner output contract. "
            "Every source_candidate_ref must be null. Continuations require a real "
            "unique verified_track_ref from the supplied menu. At most three "
            "continuations and eight new developments. Dependencies may reference "
            "only earlier rows of this response through output_ref; process "
            "selected_continuations first, then new_development_candidates. "
            "Dependencies may be empty. No original candidate registry exists yet.")
    kwargs.update(planner_max_continuations=3, planner_max_new_candidates=8,
                  planner_scene_quota=6, planner_max_plan_variants=1, planner_max_tokens=8192)
    built = render_expansion_planner(**kwargs)
    text = built["prompt"]
    start = text.index("For all developments, before_state -> required_transition")
    end = text.index("Do not output development_mode, scene IDs", start)
    # Version-specific replacement block removes incompatible old requirements,
    # rather than adding a conflicting instruction to their end.
    text = text[:start] + TARGET_RULES + "\n" + text[end:]
    for old in ("The resulting state already true before scene end",
                "The resulting new state already true before scene end"):
        text = text.replace(old, "Original desired final target; not an observed fact")
    built = package("initial_planner" + ("." + VERSION if shared else ""), text, {})
    if shared:
        built["planner_contract_identity"] = {
            "output_contract": VERSION, "schema_sha256": sha(output_schema()),
            "render_version": RENDER_VERSION,
            "execution_mode": "initial", "strict_decoding_added": False}
    return built
