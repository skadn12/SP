"""Prompts for the single Planner and single round Realizer calls."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .contracts import build_planner_outline_pairs


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _transcript(parent: Mapping[str, Any]) -> str:
    return "\n".join(
        f"{row['turn_id']} | {row['speaker']} | {row['text']}"
        for row in parent["dialogue"]["turns"])


# Shared with the incremental route; these passages were present in the
# successful B_TEXT_FREE request. Historical default wording stays unchanged.
REALIZER_PARTICIPANT_RULES = """Use natural dialogue, concrete actions, information, decisions,
constraints, or consequences. Every required_participant must appear and must
keep the identity assigned by ACTOR BINDINGS. required_participants is a
coverage requirement, not a speaker allowlist. Additional established people
or incidental scene-local people such as a clerk or passerby may speak when
natural, but they must use a stable non-transport name, remain auxiliary, and
must not perform a bound core action or replace a required participant. Never
use `user` or `assistant` as a character name."""

REALIZER_TEXT_FREE_STYLE_RULES = (
    "Use the continuity shape only to preserve speaker alternation, "
    "turn rhythm, and sentence shape. Previous wording is intentionally "
    "omitted: do not reconstruct, quote, copy, or closely paraphrase any "
    "earlier exchange. Do not recreate an earlier emotional closing or "
    "support exchange merely to fill turns. Do not replay prior events or "
    "continue an old topic unless the selected plan explicitly requires "
    "it.")


def realizer_arc_turn_rules(*, actual_progress_only: bool = False) -> str:
    transition = ("actual progress established in this scene" if actual_progress_only
                  else "required\ntransition")
    return f"""Write a complete mini-arc for every scene: opening, interaction, the {transition}, its immediate consequence, and a closing beat. Even when the main
action finishes early, continue with concrete reaction, clarification,
decision, or consequence that belongs to the same scene; do not stop after the
first exchange. Use exactly 14 dialogue turns per scene by default (14 is the
target baseline; extend up to 18 only when the selected development genuinely
needs it). The absolute hard contract is 9 to 48 turns for EACH scene; an
8-turn scene is invalid. A scene may use one or more declared speakers; do not
add another speaker merely to satisfy a count. Do not copy any Parent sentence
verbatim."""


def build_event_verifier_prompt(*, parent: Mapping[str, Any],
                                candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_view = [{
        "memory_event_ref": row["memory_event_id"],
        "memory_event_key": row["memory_event_key"],
        "memory_event_display": row["memory_event_display"],
        "memory_content": row["memory_event_content"],
    } for row in candidates]
    transcript = "\n".join(
        f"{index} | {row['speaker']} | {row['text']}"
        for index, row in enumerate(parent["dialogue"]["turns"], 1))
    text = f"""You are the Transcript Event Verifier. Memorax event candidates
are recall hints only. The positioned Parent transcript is authoritative.

For each candidate directly supported by Parent, state only the latest state
that has actually happened. Keep intention, decision, action, and result
strictly distinct; never upgrade a plan into execution. Select the smallest
sufficient evidence: one position by default, two to four only when needed.
Four positions is a hard maximum. If the state cannot be proved with four or
fewer positions, omit that candidate. Never copy a whole session or a long
contiguous block. The state must be a concrete proposition with the real
participant, event, action, decision, or result from the transcript. Do not use
labels such as "latest state" or "current state" as the state value. Do not
plan future developments.

MEMORAX EVENT CANDIDATES
{_json(candidate_view)}

AUTHORITATIVE POSITIONED PARENT TRANSCRIPT
{transcript}

Output only this JSON object, with no markdown or extra keys. The empty form is
valid; when you include a track, fill all three fields with transcript-grounded
values:
{{"verified_tracks":[]}}

For every non-empty track, the object has EXACTLY these literal keys:

- `memory_event_ref`: copy exactly one `ME##` value from the event menu.
- `verified_current_state`: one concrete sentence describing what has already
  happened, naming the real participant and event/action/decision/result.
- `evidence_positions`: an array of one to four integer transcript positions.

Do not output `event_ref`, `intention`, `action`, `result`, `before_state`,
`after_state`, canonical turn IDs, or any other keys. Do not split a track into
multiple state fields. verified_tracks may be empty; the local program maps
positions to authoritative Parent turn refs."""
    return {"prompt_version": "transcript_event_verifier_prompt_v3",
            "prompt": text, "candidate_count": len(candidates),
            "parent_turn_count": len(parent["dialogue"]["turns"])}


def build_planner_prompt(*, parent: Mapping[str, Any], memory_overview: Mapping[str, Any],
                         candidates: Sequence[Mapping[str, Any]],
                         verified_tracks: Sequence[Mapping[str, Any]] | None = None,
                         memory_only_fast: bool = False,
                         event_bucket_only: bool = False,
                         memory_source_mode: str = "person_bucket",
                         speaker_registry: Mapping[str, Any] | None = None,
                         strict_speaker_refs: bool = False,
                         expose_complete_memory: bool = False,
                         include_authoritative_transcript: bool | None = None,
                         max_continuations: int = 3,
                         max_new_development_candidates: int = 4,
                         scene_quota: int = 3,
                         max_plan_variants: int = 1,
                         output_contract_text: str | None = None,
                         feedback_guidance: str | None = None) -> dict[str, Any]:
    if int(max_plan_variants) < 1:
        raise ValueError("max_plan_variants_must_be_positive")
    verified_tracks = list(verified_tracks or [])
    candidate_view = []
    for row in verified_tracks:
        if memory_only_fast:
            packet = dict(row.get("memory_planning_packet") or {})
            packet["verified_track_ref"] = row["verified_track_ref"]
            candidate_view.append(packet)
        else:
            candidate_view.append({
                "verified_track_ref": row["verified_track_ref"],
                "memory_event_ref": row["memory_event_ref"],
                "memory_event_key": row["memory_event_key"],
                "memory_event_display": row["memory_event_display"],
                "verified_current_state": row["verified_current_state"],
                "authoritative_evidence": row.get("evidence_source") or [],
            })
    authority = (
        "Memory provides non-authoritative navigation hints about people and "
        "events. Event-bucket rows are the main selectable event source; "
        "person-bucket identities, relationships, preferences, traits, and "
        "named event facts are auxiliary background. All Memory state is "
        "unverified. The complete cumulative Parent transcript is authoritative "
        "about what has happened. Memory silence never proves that an intention, "
        "decision, action, agreement, resolution, or result has not happened. "
        "Use only concrete event rows and canonical Parent speaker refs."
        if memory_only_fast else
        "Memorax is a non-authoritative event index. Event-bucket rows are the "
        "main candidate source and person-bucket facts are auxiliary context. "
        "The complete cumulative Parent transcript is authoritative about what "
        "has happened.")
    if memory_only_fast and memory_source_mode == "event_bucket_fallback":
        authority = (
            "Event-bucket fallback mode. The primary event-bucket "
            "view contained no concrete selectable event, so the local program "
            "exposes remaining concrete event-bucket keys and contents as "
            "unverified planning hints. Event-bucket key is the event identity; "
            "do not infer speaker identity from Memory labels such as user or "
            "assistant. Use only canonical Parent speaker refs from the "
            "registry. Memory remains non-authoritative and silence never "
            "proves that an intention, decision, action, agreement, resolution, "
            "or result has not happened. The local program and later Realizer "
            "checks remain the quality safeguards.")
    if event_bucket_only:
        authority += (
            " The selectable menu is event-bucket-first. Person-bucket facts "
            "remain auxiliary background only and cannot create a second event "
            "identity. The event-bucket keys and event-bucket contents are the "
            "main candidate material. Do not infer a speaker from Memory labels such as user; "
            "speakers come only from the canonical registry.")
    parent_input = ("CANONICAL PARENT SPEAKER REGISTRY\n" +
                    _json(speaker_registry or {"speakers": [
                        {"speaker_ref": f"SP{index:02d}", "canonical_name": name}
                        for index, name in enumerate(sorted({
                            str(row.get('speaker') or '') for row in
                            parent.get('dialogue', {}).get('turns', [])
                            if str(row.get('speaker') or '')}), 1)]}))
    if include_authoritative_transcript is None:
        include_authoritative_transcript = not memory_only_fast
    if include_authoritative_transcript:
        parent_input += "\n\nAUTHORITATIVE PARENT TRANSCRIPT\n" + _transcript(parent)
    speaker_field = "speaker_refs" if strict_speaker_refs else "participants"
    speaker_example = '["SP01", "SP02"]' if strict_speaker_refs else '["Exact Parent speaker", "Exact Parent speaker"]'
    actor_example = ('"actor_ref": "SP01"' if strict_speaker_refs
                     else '"actor": "Exact Parent speaker"')
    speaker_rule = (
        "Use only speaker_refs from the canonical registry. Do not output "
        "participants or translated/alias speaker names. Use actor_ref in "
        "actor_bindings; the local program resolves names. In every "
        "action-bearing free-text field (goal, transition, before/after state, "
        "and interaction beats), use the exact canonical name implied by the "
        "ref. Memorax labels such as `user`, `the user`, 用户, and `assistant` "
        "are transport roles with no character identity. Never copy them into "
        "speaker_refs, actor_ref, or action-bearing text, and never infer a "
        "canonical speaker from them. "
        "Do not use friend or another unbound role label for a different "
        "action subject. If a non-Parent person must act, declare that person "
        "in new_people and include the exact declared name in the relevant "
        "text. If Memory does not identify the actor with a canonical name, "
        "skip that continuation rather than guessing."
        if strict_speaker_refs else
        "Use exact Parent speaker names for participants and actor bindings."
    )
    complete_memory = (
        "COMPLETE MEMORAX MEMORY\n" + _json(memory_overview) + "\n\n"
        if expose_complete_memory else "")
    variant_instruction = ""
    if int(max_plan_variants) == 1:
        variant_instruction = """
This profile requires exactly one complete plan in this one Planner response.
That plan may contain multiple continuation entries and a ranked
new-development candidate pool. Do not generate a second alternative plan and
do not wrap multiple complete plans in an alternative-plan array. The local
program applies the existing Gate, selection, and scene-batch limits to this
single plan.
"""
    else:
        variant_instruction = f"""
This explicitly expanded profile permits up to {int(max_plan_variants)} complete
alternative plans from this one Planner call. Return them under one top-level
`plan_variants` array; each array item must have exactly the single-plan object
shape shown below (`round_goal`, `selected_continuations`, and
`new_development_candidates`). Do not put those keys at the top level alongside
`plan_variants`, do not exceed the configured variant limit, and do not repeat a
variant. The local program selects one returned variant and applies the normal
Gate/materialization contract; producing multiple variants does not bypass any
validation or create extra Planner calls.
"""
    variant_block = ("\n\n" + variant_instruction.strip()
                     if variant_instruction else "")
    output_header = (
        "Output only this JSON object. No markdown, no extra keys:"
        if int(max_plan_variants) == 1 else
        "Output only this JSON object. No markdown, no extra keys besides the "
        "top-level `plan_variants` array:"
    )
    single_plan_example = '''{
  "round_goal": "...",
  "selected_continuations": [
    {
      "verified_track_ref": "VT01",
      "new_people": [],
      "goal": "...",
      "SPEAKER_FIELD": ["SPEAKER_EXAMPLE", "SPEAKER_EXAMPLE"],
      "transition_from_previous": "...",
      "required_transition": "A concrete new state change that happens now",
      "after_state": "The resulting state already true before scene end",
      "actor_bindings": [
        {ACTOR_BINDING}
      ],
      "interaction_beats": ["...", "...", "..."]
    }
  ],
  "new_development_candidates": [
    {
      "source_basis": ["Concrete existing-world fact", "Another concrete basis"],
      "new_people": [{"name": "Elena", "relationship_basis": "new neighbor of an exact Parent speaker"}],
      "goal": "...",
      "SPEAKER_FIELD": ["SPEAKER_EXAMPLE", "SPEAKER_EXAMPLE"],
      "transition_from_previous": "...",
      "before_state": "A supported current-world state, without inventing absence",
      "required_transition": "A genuinely new event or state change that happens now",
      "after_state": "The resulting new state already true before scene end",
      "actor_bindings": [
        {ACTOR_BINDING}
      ],
      "interaction_beats": ["...", "...", "..."]
    }
  ]
}'''
    single_plan_example = (single_plan_example
                           .replace('"SPEAKER_FIELD": ["SPEAKER_EXAMPLE", "SPEAKER_EXAMPLE"]',
                                    f'"{speaker_field}": {speaker_example}')
                           .replace("{ACTOR_BINDING}",
                                    f'{{{actor_example}, "binding": "This speaker owns the key action."}}'))
    output_example = (single_plan_example if int(max_plan_variants) == 1 else
                      '{"plan_variants": [' + single_plan_example + ', ' +
                      single_plan_example + ']}')
    if output_contract_text is not None:
        if int(max_plan_variants) != 1 or not strict_speaker_refs:
            raise ValueError("custom_planner_contract_requires_single_plan_speaker_refs")
        output_header = ""
        output_example = output_contract_text
    text = f"""You are the single Planner for Continuation-First Controlled World
Expansion v1. {authority}

First select zero to {int(max_continuations)} Memory events that genuinely have a natural next
step. Never force a closed event to continue. Then provide ranked
new_development_candidates, ordered from highest to lowest quality. These are a
candidate pool, not final scenes. The new-development candidate pool has
a hard maximum of {int(max_new_development_candidates)} items; never exceed it.
The local program owns the fixed scene quota of {int(scene_quota)} and any independent
new-development slots: it sends the whole candidate pool to Gate, skips rejected
candidates, and materializes only PASS candidates in deterministic rank order.
Skipped and extra candidates never reach Realizer. Exact-fill arithmetic belongs
to the local program. Do not perform cross-field exact-fill arithmetic.

For every continuation, verified_track_ref must come from the Memory Planning
Packet menu and must be unique. Person event contents and person_event_facts are
unverified Memory summaries, not a verified latest state. Use person_event_facts
to understand which named person is associated with an event, but never map the
transport label "user" to a character. Do not output or rewrite event identity,
evidence positions, or evidence turn refs. Use related_people only as
background. Only canonical Parent speaker refs may be continuation
participants; a related person does not become a speaker. Never infer "X has not happened" merely because a
packet omits X. Continue beyond usable Memory hints; never repeat an existing
intention, decision, action,
agreement, resolution, or result.

Every new development must have one to four concrete source_basis facts grounded
in existing people, relationships, preferences, experiences, or current life
and work circumstances. It must be a genuinely new event or state change, not a
restated preference, personality explanation, past discussion, confirmation,
generic chat, or merely something that may happen later. New developments must
be meaningfully distinct from continuations and from each other. new_people is
a list of zero to two independently declared people.
Each entry contains only name and relationship_basis. One entry represents one
person, so never combine names such as "Elena and Marco". Do not force new
people. A new-development scene may use canonical Parent speaker refs plus exact
names declared in that scene's new_people list; no other participant is legal.
A declared new person is not a Parent speaker and must never be encoded as a
fabricated SP03 (or any other SPxx) ref. The local program automatically adds
every declared new-person name to scene participants. Therefore speaker_refs
must contain only real refs from the canonical Parent registry; the declared
new person may still speak in the realized scene.
A continuation must always use new_people=[] and cannot introduce a person.

For all developments, before_state -> required_transition -> after_state is the
frozen State Transition Contract. The transition and resulting state must be a
bounded mini-arc that can be completed and directly observed within THIS scene.
Do not make after_state a multi-scene or long-term destination. The local
program derives durable_outcome from after_state, so durable_outcome means the
smallest stable result already true at the end of this scene, not the final
outcome of a future storyline. A future possibility may be mentioned only as a
supporting consequence; it must never be the required end state.

Good bounded outcomes include: a new person is introduced and a concrete
connection is established; two people agree on a specific arrangement; contact
information is exchanged; a planned appointment is scheduled; a first action
is actually completed; or a concrete responsibility is accepted. Do not require
an entire future activity when the scene only needs to establish its immediate
condition. For example, "A and B agree to walk together on Saturday" is a
scene-completable outcome; "A and B complete a lasting weekly walking routine"
is an over-scoped future outcome unless the dialogue actually completes it now.

required_transition must actually happen in the Child scene; after_state must
already be true before the scene ends. It must describe a concrete event that
causes a clear state change. Generic discussion, chatting, reflection, or merely
mentioning a preference is not the main transition. A discussion may accompany
an explicit event such as meeting, exchanging contact information, agreeing to
participate, accepting responsibility, making a concrete decision, establishing
a relationship, or completing a transaction. {speaker_rule} Actor bindings
freeze who owns each key action. Three to seven interaction beats must enact
only this development. Each beat is a third-person action/interaction summary
(who proposes, decides, executes, or responds and what changes), never a
complete dialogue line or `Name: "..."`.
For a selected continuation, do not use "discusses", "shares", "asks for advice",
"gets a better understanding", or similar conversation as the required
transition by itself. The continuation must move the Memory event to a concrete
later state: for example, an already purchased item is used for a first trip,
a prior plan is carried out, a responsibility is accepted and completed, or a
new arrangement is explicitly made. Put any explanation or conversation in
supporting beats, and make the required_transition and after_state name the
new action, decision, arrangement, or result that is true at scene end.
Do not output development_mode, scene IDs, candidate IDs, new_development,
durable_outcome, or completion_conditions; the local program derives them.

{complete_memory}CANDIDATE MEMORY PLANNING PACKETS
{_json(candidate_view)}

{parent_input}{variant_block}

{output_header}
{output_example}"""
    if int(max_plan_variants) > 1:
        text += (
            "\n\nFINAL OUTPUT SHAPE OVERRIDE FOR THIS EXPANDED PROFILE\n"
            "The required top-level object is exactly {\"plan_variants\": "
            "[<single-plan-object>, ...]}; each item must use the complete "
            "single-plan shape shown above. Do not emit the single-plan keys "
            "alongside `plan_variants`."
        )
    # Feedback is an optional, append-only input layer.  Keeping this branch
    # after the historical prompt construction means feedback=None (and the
    # empty string) produce byte-for-byte the same prompt and metadata as the
    # first-round Planner call.  The feedback text is guidance only; the
    # Planner's existing JSON contract and local gates remain authoritative.
    if feedback_guidance and str(feedback_guidance).strip():
        text += "\n\n" + str(feedback_guidance).strip()
    prompt_version = ("memory_only_complete_memory_planner_prompt_v1" if expose_complete_memory else
                                "event_bucket_only_planner_prompt_v2_bounded_scene_outcome" if event_bucket_only and memory_only_fast else
                                "memory_context_fusion_planner_prompt_v3_event_bucket_fallback" if memory_only_fast and memory_source_mode == "event_bucket_fallback" else
                                "memory_context_fusion_planner_prompt_v2_bounded_scene_outcome_person_bucket_only" if strict_speaker_refs and memory_only_fast else
                                "memory_context_fusion_planner_prompt_v2_bounded_scene_outcome" if memory_only_fast
                                else "verified_track_programmatic_scene_quota_planner_prompt_v2_bounded_scene_outcome")
    if int(max_plan_variants) > 1:
        prompt_version += "_multi_plan_v1"
    return {"prompt_version": prompt_version,
            "prompt":text,"candidate_count":len(candidates),
            "event_bucket_only": bool(event_bucket_only),
            "memory_source_mode": str(memory_source_mode),
            "include_authoritative_transcript": bool(include_authoritative_transcript),
            "scene_quota": int(scene_quota),
            "max_continuations": int(max_continuations),
            "max_new_development_candidates": int(max_new_development_candidates),
            "max_plan_variants": int(max_plan_variants)}


def build_planner_candidate_repair_prompt(
    *, candidates: Sequence[Mapping[str, Any]],
    verified_tracks: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any], gate_validation: Mapping[str, Any],
    repair_targets: Sequence[Mapping[str, Any]] | None = None,
    speaker_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one bounded, ordered replacement request after Gate rejection."""
    passed_count = int(gate_validation.get("passed_candidate_count") or 0)
    minimum = int(gate_validation.get("minimum_pass_candidates") or 3)
    needed = (len(repair_targets) if repair_targets is not None else
              max(0, minimum - passed_count))
    rejected_refs = set(str(ref) for ref in
                        gate_validation.get("rejected_candidate_refs") or [])
    candidate_rows: list[dict[str, Any]] = []
    source_scenes = plan.get("candidate_pool") or plan.get("scenes") or []
    reviews = gate_validation.get("normalized_candidate_reviews") or []
    for scene in source_scenes:
        event_id = str(scene.get("memory_event_id") or "")
        for index, candidate in enumerate(
                scene.get("next_step_candidates") or [], 1):
            ref = f"{event_id}:C{index:02d}"
            if ref in rejected_refs:
                candidate_rows.append({
                    "candidate_ref": ref,
                    "required_transition": candidate.get("required_transition"),
                    "after_state": candidate.get("after_state"),
                    "gate_reason": next((review.get("reason") for review in
                        reviews if review.get("candidate_ref") == ref), ""),
                })
    memory_rows = []
    for row in verified_tracks:
        packet = dict(row.get("memory_planning_packet") or {})
        if packet:
            memory_rows.append(packet)
    registry = speaker_registry or {"speakers": []}
    speaker_rows = [row for row in registry.get("speakers") or []
                    if isinstance(row, Mapping) and row.get("speaker_ref")]
    output_candidate = {
        "source_basis": ["Concrete existing-world fact"],
        "new_people": [],
        "goal": "...",
        "speaker_refs": [str(row["speaker_ref"]) for row in speaker_rows[:2]],
        "transition_from_previous": "...",
        "before_state": "...",
        "required_transition": "A genuinely new event or state change that happens now",
        "after_state": "The smallest stable result already true before scene end",
        "actor_bindings": [{
            "actor_ref": str(speaker_rows[0]["speaker_ref"])
            if speaker_rows else "SP01",
            "binding": "This speaker owns the key action.",
        }],
        "interaction_beats": ["...", "...", "..."],
    }
    ordered_targets = [dict(row) for row in (repair_targets or [])]
    if not ordered_targets:
        ordered_targets = [{
            "target_ref": ref,
            "target_kind": "new_development_slot",
            "reason": "fill_missing_scene_slot",
        } for ref in sorted(rejected_refs)[:needed]]
    text = f"""You are repairing rejected Planner candidates. Generate exactly {needed} replacement candidate(s) in the exact target order below. This is one
bounded repair call; it is the only repair attempt for this round. Do not
repeat or rewrite a rejected transition. The local program assigns candidate
references from list position; do not output IDs.

For a target with `target_kind=retry_same_memory_event`, continue that same
Memory event beyond its verified current state with a concrete next action,
arrangement, consequence, or result. Do not repeat the rejected transition.
For a target with `target_kind=replace_new_development` or
`target_kind=new_development_slot`, create a different concrete development
grounded in the supplied Memory planning packets and existing-world facts. A
new person or event is allowed when it has a concrete connection and a
scene-completable outcome. Do not use generic themes, praise, reflection, or a
future-only plan. A rejected event marked non-retryable is intentionally absent
from the targets and must not be retried.

The replacement after_state must be the smallest stable state already true at
the end of this scene. Do not output evidence positions, Parent turn IDs, or
candidate IDs; the local program supplies those.

CANONICAL SPEAKER REGISTRY
{_json(registry)}

MEMORY PLANNING PACKETS
{_json(memory_rows)}

REJECTED CANDIDATES AND GATE REASONS
{_json(candidate_rows)}

ORDERED REPAIR TARGETS
{_json(ordered_targets)}

Output only this JSON object with exactly one key. The list length must be
{needed}, and each item must have exactly the keys shown:
{{"new_development_candidates": [{_json(output_candidate)}]}}"""
    return {
        "prompt_version": "memory_context_fusion_planner_candidate_targeted_repair_v2",
        "prompt": text,
        "needed_candidate_count": needed,
        "rejected_candidate_refs": sorted(rejected_refs),
        "repair_targets": ordered_targets,
    }


def build_realizer_prompt(*, parent: Mapping[str, Any], plan: Mapping[str, Any],
                          style_excerpt: Sequence[Mapping[str, Any]] | None = None,
                          style_profile: Mapping[str, Any] | None = None,
                          continuity_shape: Mapping[str, Any] | None = None,
                          promoted_prefix: Sequence[Mapping[str, Any]] | None = None,
                          include_full_parent: bool = True,
                          request_purpose: str = "normal_generation") -> dict[str, Any]:
    if request_purpose != "normal_generation":
        raise ValueError("realizer_prompt_wrong_request_purpose")
    compact = [{
        "memory_event_id": scene.get("memory_event_id"),
        "development_mode": scene.get("development_mode"),
        "source_basis": scene.get("source_basis"),
        "new_people": scene.get("new_people"),
        "verified_current_state": scene.get("verified_current_state"),
        "goal": scene.get("goal"),
        "required_participants": (
            scene.get("required_participants") or scene.get("participants")),
        "transition_from_previous": scene.get("transition_from_previous"),
        "CURRENT STATE": scene.get("before_state"),
        "REQUIRED TRANSITION": scene.get("required_transition"),
        "REQUIRED END STATE": scene.get("after_state"),
        "ACTOR BINDINGS": scene.get("actor_bindings"),
        "INTERACTION BEATS": scene.get("interaction_beats"),
    } for scene in plan["scenes"]]
    gain_policy = plan.get("development_policy") == "development_gain_v2"
    if gain_policy:
        for row, scene in zip(compact, plan["scenes"]):
            row["scene_ref"] = scene.get("scene_ref")
            row["EXPECTED NEW INFORMATION (not yet realized)"] = scene.get("expected_information")
            row["NECESSARY PRIOR SCENES"] = scene.get("depends_on_scene_refs", [])
    if include_full_parent:
        history_context = "FROZEN PARENT TRANSCRIPT\n" + _transcript(parent)
        context_instruction = (
            "The frozen transcript is authoritative history. Do not copy or "
            "replay it.")
        prompt_version = "memory_event_continuation_realizer_prompt_v6"
    else:
        excerpt = list(style_excerpt or [])
        profile = dict(style_profile or {})
        shape = dict(continuity_shape or {})
        if excerpt:
            # Compatibility for callers outside the live multi-round path.
            history_context = "STYLE AND CONTINUITY EXCERPT ONLY\n" + "\n".join(
                f"{row['turn_id']} | {row['speaker']} | {row['text']}"
                for row in excerpt)
        else:
            history_context = (
                "CONTINUITY SHAPE ONLY (previous wording omitted)\n" +
                _json(shape))
        history_context += "\n\nTEXT-FREE STYLE PROFILE\n" + _json(profile)
        context_instruction = REALIZER_TEXT_FREE_STYLE_RULES + (
            " The frozen scene plans are the only authority for what must happen.")
        prompt_version = "memory_event_continuation_realizer_continuity_shape_v1"
    prefix_rows = list(promoted_prefix or [])
    if prefix_rows:
        history_context += (
            "\n\nPROMOTED CHILD PREFIX FOR THIS BATCH\n" +
            "Only scenes already admitted by the local promotion gate are "
            "included below. Treat them as the immediate prior context; do "
            "not rewrite them.\n" +
            "\n".join(
                f"[{row.get('turn_id') or row.get('turn_ref')}] "
                f"{row.get('speaker')}: {row.get('text')}"
                for row in prefix_rows))
    scene_count = len(compact)
    output_shape = {"scenes": [
        {"dialogue_turns": [{"speaker": "...", "text": "..."}]}
        for _ in range(scene_count)]}
    text = f"""You are the Natural Realizer. Write one coherent append-only Child
batch for the frozen scene plans below. The input contains exactly {scene_count}
scene plan(s). The output scenes array MUST contain exactly {scene_count} scene objects,
one for each input plan in the same order. Never merge scenes, omit a scene, or
stop early. Treat every verified_current_state
    as already realized Parent history. Never replay or re-decide it. Continue from
    that state and let later scenes follow the actual prior Child scene. CURRENT
    STATE is already true before the Child scene begins. REQUIRED TRANSITION must
    actually happen during that scene. REQUIRED END STATE must already be true
    before the scene ends. Do not substitute the required transition with merely
    mentioning, discussing, considering, planning, promising, preparing, or waiting.
    ACTOR BINDINGS are frozen facts: never transfer a bound action or state to
    another participant. Use INTERACTION BEATS to realize only the selected
    candidate. Preserve natural dialogue and never quote these instructions or
    state fields verbatim. {REALIZER_PARTICIPANT_RULES}
{realizer_arc_turn_rules()} Return only JSON with the exact top-level key scenes; each scene
contains dialogue_turns with speaker and text only. Before returning, verify
    that len(scenes) == {scene_count} and that every dialogue_turns array has exactly 14
items unless a longer 15-18 turn arc is genuinely required.

{history_context}

FROZEN SCENE PLANS
{_json(compact)}

OUTPUT
{{
  "scenes": {json.dumps(output_shape["scenes"], ensure_ascii=False)}
}}
The single turn shown in each scene above illustrates turn object shape only;
generate 14-18 actual turns per scene and never fewer than 9.

{context_instruction} Before returning, verify that len(scenes) == {scene_count}."""
    if gain_policy:
        text = text.replace("Treat every verified_current_state\n    as already realized Parent history.",
                            "Treat Memory current-state summaries as context with their stated uncertainty;\n    a mentioned plan is not an already executed action.")
        text += """\nDEVELOPMENT GAIN v2: Establish the selected concrete expected information
in natural dialogue; do not replace it with generalized support or reflection.
The expected addition is a target, not evidence it happened. Preserve the exact
target state: agreeing to meet is not meeting; preparing to submit is not
submitting; gaining a specific understanding is not complete mastery. Necessary
prior scenes must actually establish prerequisites before a dependent scene
relies on them. Do not claim skipped or failed candidates occurred.
"""
        prompt_version += ":development_gain_v2"
    return {"prompt_version": prompt_version, "prompt": text,
            "style_excerpt_turn_count": (0 if include_full_parent else
                                           len(style_excerpt or [])),
            "continuity_shape_included": bool(not include_full_parent and
                                               continuity_shape is not None),
            "style_profile_included": bool(not include_full_parent and
                                            style_profile is not None),
            "full_parent_included": bool(include_full_parent),
            "promoted_prefix_turn_count": len(prefix_rows),
            "request_purpose": request_purpose,
            "scene_count": scene_count}


def build_semantic_continuation_gate_prompt(
    *, gate_context: Mapping[str, Any],
    prompt_version: str = "realization_aware_semantic_gate_prompt_v1",
) -> dict[str, Any]:
    from .gate_output_contract import LEGACY_PROMPT_VERSION, FULL_PROMPT_VERSION, full_history_output_contract
    if prompt_version not in {LEGACY_PROMPT_VERSION, FULL_PROMPT_VERSION}:
        raise ValueError("unknown_gate_prompt_version")
    full_contract = (full_history_output_contract(gate_context)
                     if prompt_version == FULL_PROMPT_VERSION else None)
    expected_refs = [str(value) for value in
                     gate_context.get("expected_candidate_refs") or []]
    checklist = "\n".join(
        f"{index}. {candidate_ref}"
        for index, candidate_ref in enumerate(expected_refs, 1))
    memory_only_fast = bool(gate_context.get("memory_only_fast"))
    has_global_memory_view = bool(gate_context.get("global_memory_state_view"))
    has_planner_state_contract = any(
        candidate.get("planner_current_state_stage")
        for scene in gate_context.get("scenes") or []
        if isinstance(scene, Mapping)
        for candidate in scene.get("candidates") or []
        if isinstance(candidate, Mapping))
    authority_block = (
        "Planner has already judged novelty "
        "against the supplied Memory current state. Audit only whether a "
        "candidate is a no-op or duplicate of that state. Memorax is the only "
        "state context in this Gate input. Do not request, invent, or cite "
        "Parent transcript turns."
        if memory_only_fast else
        "Audit only whether each candidate's required_transition has actually "
        "happened in the authoritative Parent, or whether its after_state is "
        "already explicitly true there. Memorax is a non-authoritative hint; "
        "Parent evidence is authoritative."
    )
    relation_block = (
        "For every candidate, classify its relation to the supplied Memory "
        "current state as exactly one of REALIZED, ONGOING, "
        "PLANNED_OR_INTENDED, RELATED_ONLY, or NONE. REALIZED means the "
        "candidate merely restates a completed transition explicitly stated "
        "in one or more cited Memory facts and must be rejected. ONGOING, "
        "PLANNED_OR_INTENDED, RELATED_ONLY, and NONE pass. Use "
        "supporting_memory_refs, never transcript turn refs."
        if memory_only_fast else
        "For every candidate, classify the most relevant Parent evidence as "
        "exactly one of REALIZED, PLANNED_OR_INTENDED, RELATED_ONLY, or NONE. "
        "REALIZED means the same actor/action/object transition actually "
        "happened, or the candidate after_state is already explicitly "
        "established. Only REALIZED is reject."
    )
    global_memory_block = (
        "A GLOBAL MEMORY STATE VIEW is included in the Gate input. Each "
        "memory_facts entry has a program-generated memory_ref. Compare "
        "every candidate against all explicit states in every bucket, not only "
        "the candidate's selected event. Reject only when the same completed "
        "decision, action, agreement, resolution, or result is already stated. "
        "A shared topic, person, plan, intention, or related event is not a "
        "duplicate. Memory is a current namespace hint, not Parent evidence; "
        "do not request or invent transcript refs."
        if has_global_memory_view else "")
    planner_state_block = (
        "Each candidate includes a Planner state contract grounded by valid "
        "Memory refs. Treat planner_current_state_stage and "
        "planner_current_state_summary as the primary current-state reading. "
        "Do not reclassify PROPOSED, IN_PROGRESS, AWAITING_CONFIRMATION, or "
        "READY_TO_FINALIZE as completed merely because a Memory fact shares "
        "the same actor, object, or topic. You may reject only when a cited "
        "Memory fact explicitly establishes that the candidate's exact "
        "required_transition or after_state is already completed."
        if has_planner_state_contract else "")
    text = f"""You are a strict Realization-Aware Semantic Continuation Gate
between Planner and Realizer. You are a candidate filter and a realization detector,
not a realization writer. {authority_block}

You must review EVERY candidate provided below. Return exactly one review for
every candidate_ref. Do not omit any continuation candidate, new-development
candidate, PASS candidate, or REJECT candidate. Preserve each candidate_ref
exactly as provided. The output review count must equal the input candidate
count. ME##:C## identifies an existing-Memory-event continuation candidate.
ND##:C## identifies a controlled-new-development candidate. Both are
first-class candidates and both MUST be reviewed.

REQUIRED CANDIDATE REFS
{checklist}

Your output must contain every candidate_ref above exactly once, preferably in
the displayed order.

{relation_block}
{global_memory_block}
{planner_state_block}
PLANNED_OR_INTENDED means the supplied state only describes an intention,
plan, preparation, proposal, invitation, or possibility; it is pass.
ONGOING means the supplied state says the transition is underway, pending,
being finalized, or not yet complete; it is pass.
RELATED_ONLY means a shared person, place, activity, topic, or background
without the same completed transition; it is pass. NONE is also pass.

The fixed mapping is REALIZED -> reject; all other relations -> pass. Plans
are not execution, hopes are not completion, proposals are not realization,
and related background is not a completed transition. A future intention later
executed is a legitimate continuation. Compare transition identity rather than
word overlap: actor, action, object/person, completion status, and resulting
state must establish substantially the same completed transition.

Examples: Parent hopes to introduce B to Elena someday and the candidate
actually introduces them -> PLANNED_OR_INTENDED/pass. Parent suggests cycling
later and the candidate completes a cycling trip -> PLANNED_OR_INTENDED/pass.
Parent showed a previous visitor around and the candidate meets new neighbor
Elena for the first time -> RELATED_ONLY/pass. Parent already introduced B to
Elena and exchanged contact information and the candidate repeats it ->
REALIZED/reject.

Check only these six duplicate categories:
- decision
- intention
- action
- agreement
- resolution
- result

Return reject only if the proposed required_transition was realized or the
after_state already holds in the supplied state, even when paraphrased. Do not reject
merely because the candidate is similar, related, discussed, planned, hoped
for, or proposed.
Do not repair, rewrite, improve, or suggest an alternative development. For a
reject, include a nonempty reason and cite one or more exact program-generated
memory_ref values whose text explicitly proves the completed duplicate. In
non-fast mode, cite exact Parent turn refs proving the duplicate. In fast mode,
supporting_memory_refs must contain only valid memory_ref values. For a pass,
supporting_memory_refs may be empty; omit reason because the local
program supplies the fixed PASS audit reason. Audit every
candidate_ref exactly once. Judge each candidate independently; do not select,
rank, repair, or rewrite it.

GATE INPUT
{_json(gate_context)}

OUTPUT ONLY THIS JSON OBJECT, WITH NO EXTRA KEYS
{{
  "candidate_reviews": [
    {{
      "candidate_ref": "ME01:C01",
      "parent_relation": "PLANNED_OR_INTENDED",
      "decision": "pass",
      "duplicate_with_parent": false,
      "supporting_memory_refs": []
    }},
    {{
      "candidate_ref": "ND01:C01",
      "parent_relation": "REALIZED",
      "decision": "reject",
      "duplicate_with_parent": true,
      "supporting_memory_refs": ["GM:E001:C001"],
      "reason": "The candidate repeats the cited Memory fact."
    }}
  ]
}}"""
    if full_contract is not None:
        # Versioned replacement, not an appended contradictory instruction.
        start = text.index("Do not repair, rewrite, improve, or suggest an alternative development.")
        text = (text[:start] + full_contract["rules"] + "\nGATE INPUT\n" + _json(gate_context)
                + "\n\nOUTPUT ONLY THIS JSON OBJECT, WITH NO EXTRA KEYS\n"
                + _json(full_contract["examples"][0])
                + "\n\nALTERNATIVE REALIZED/REJECT SHAPE (NOT A FACT CLAIM)\n"
                + _json(full_contract["examples"][1]))
        # Current-state Memory hints cannot override history in this mode.
        for block in (global_memory_block, planner_state_block):
            if block:
                text = text.replace(block, "Memory and Planner state descriptions are non-authoritative navigation; use the supplied history.")
        text = text.replace("exactly one of REALIZED, PLANNED_OR_INTENDED, RELATED_ONLY, or NONE.",
                            "exactly one of REALIZED, ONGOING, PLANNED_OR_INTENDED, RELATED_ONLY, or NONE.")
    result = {"prompt_version": prompt_version,
            "prompt": text,
            "scene_count": len(gate_context.get("scenes") or []),
            "expected_candidate_count": len(expected_refs),
            "expected_candidate_refs": expected_refs}
    if full_contract is not None:
        result["gate_contract_identity"] = full_contract
    return result


def build_planner_outline_semantic_prompt(
    *, plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Prompt one compact model call to review the complete candidate pool."""
    pairs = build_planner_outline_pairs(plan)
    text = f"""You are a narrow Planner Candidate-Pool Semantic Auditor.

Review only whether the two candidate outlines in each pair describe the same
narrative development. This is a duplicate check, not a quality review. The
pool includes ranked fallbacks that may not be in the first three scene slots.
The same Memory event may continue legitimately when it advances to a new
state. Different participants, actors, objects, actions, consequences, or
stages normally indicate distinct developments.

Mark `duplicate` only when both scenes would make the same concrete development
happen, even if the wording or Memory event IDs differ. Mark `distinct` when
one scene is a genuine next step, a different event, or a different consequence.
Mark `uncertain` when the supplied outlines are insufficient to decide. Do not
use topic similarity alone. Do not reject two scenes merely because they share
a person, location, activity, or broad theme.

Treat a start and completion of the same concrete action as `duplicate` when
they merely split one action into stages for this round. A continuation is
distinct only when it creates a materially new event, arrangement, result, or
consequence. The local program keeps the earlier ranked candidate and filters
a later conflicting candidate; you only classify the pair.

Return one review for every pair_ref exactly once, in the supplied order. The
program generates pair_ref; never invent or rename it. Keep reasons short and
specific. Memory provenance is context only; do not request Parent evidence,
turn IDs, or canonical refs. Do not rewrite any outline.

There are exactly {len(pairs)} pair_refs below. Return exactly one review for
each of these refs, including the final ref. Missing a ref is treated as
uncertain by the local program and can remove a candidate.

RANKED CANDIDATE-POOL PAIRS
{_json(pairs)}

OUTPUT ONLY THIS JSON OBJECT:
{{
  "pair_reviews": [
    {{"pair_ref": "P01", "decision": "distinct", "reason": "..."}}
  ]
}}"""
    return {
        "prompt_version": "planner_outline_semantic_audit_prompt_v2_candidate_pool",
        "prompt": text,
        "pair_count": len(pairs),
        "expected_pair_refs": [str(row["pair_ref"]) for row in pairs],
        "scene_count": len(plan.get("scenes") or []),
        "candidate_pool_count": len(plan.get("candidate_pool") or
                                    plan.get("scenes") or []),
    }
