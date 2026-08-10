"""Compact prompts for the simple append prototype."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from .contracts import PLANNER_SCHEMA_VERSION, REALIZER_SCHEMA_VERSION
from .qa import (
    MCQ_FOCUS_WRITER_SCHEMA_VERSION,
    MIN_FOCUS_ROUND_JOINT_WITH_HISTORY,
    MIN_FOCUS_ROUND_SUPPORTED,
    build_evidence_alias_registry,
)


PLANNER_PROMPT_VERSION = "simple_append_planner_prompt_v2"
REALIZER_PROMPT_VERSION = "simple_append_round_realizer_prompt_v3"
MCQ_PROMPT_VERSION = "simple_append_full_dialogue_mcq_batch_prompt_v18"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_result(prompt: str, *, version: str) -> dict[str, Any]:
    return {
        "prompt_version": version,
        "prompt": prompt,
        "prompt_sha256": _sha(prompt),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
    }


def _tail_prompt_view(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root_id": context["root_id"],
        "parent_dialogue_sha256": context["parent_dialogue_sha256"],
        "selected_session_ids": list(context["selected_session_ids"]),
        "parent_speakers": list(context["parent_speakers"]),
        "turns": [{
            "turn_id": row["turn_id"],
            "session_id": row["session_id"],
            "speaker": row["speaker"],
            "text": row["text"],
        } for row in context["selected_turns"]],
    }


def build_planner_prompt(tail_context: Mapping[str, Any]) -> dict[str, Any]:
    template_scenes = []
    for index in range(1, 4):
        template_scenes.append({
            "goal": f"distinct goal for scene {index}",
            "participants": ["exact speaker display name", "exact speaker display name"],
            "new_development": f"distinct concrete development {index}",
            "interaction_beats": ["beat 1", "beat 2", "beat 3"],
            "transition_from_previous": (
                "continues the parent tail" if index == 1
                else f"continues the actual outcome of scene {index - 1}"),
            "desired_outcome": f"distinct changed situation after scene {index}",
        })
    template = {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "round_goal": "...",
        "scenes": template_scenes,
    }
    prompt = "\n\n".join([
        "SIMPLE APPEND PLANNER v1. Return JSON only.",
        "Plan two or three sequential, substantial dialogue scenes that continue from the "
        "provided final parent sessions. Prefer three scenes and about 42-54 added turns in "
        "the later realization. Build one coherent arc with concrete events, decisions, "
        "constraints, information changes, or consequences. New people, events, and places "
        "are allowed when naturally connected through at least one parent speaker. Do not "
        "plan insertion, flashback placement, Memory operations, evidence, QA, IDs, markers, "
        "or fixed dialogue. Copy every existing Parent speaker name exactly from "
        "parent_speakers; do not singularize, abbreviate, translate, or otherwise rewrite it. "
        "Each scene needs two to five speakers and three to seven beats. The union of all "
        "scene participants is the complete speaking cast for the round. Every person who "
        "will speak anywhere in the realized dialogue must be declared by one exact display "
        "name in at least one participants list. This includes newly connected specialists, "
        "clients, managers, passersby, friends, or other newly introduced people; they do not "
        "need to exist in Memory first. If a new specialist will speak, name them now (for "
        "example, Specialist D) and include that exact name in participants. Do not describe "
        "a speaking person only as an unnamed role inside new_development or a beat. "
        "Scene 1 should continue the immediate parent-tail situation. Scene 2 must make a "
        "connected shift to a different event, person, location, practical problem, or "
        "information state rather than repeat scene 1's discussion. Scene 3 must show a "
        "result, consequence, convergence, or further changed situation rather than repeat "
        "advice about the same issue. Do not plan three scenes that merely revisit one topic.",
        "AUTHORITATIVE PARENT TAIL CONTEXT\n" + _json(_tail_prompt_view(tail_context)),
        "OUTPUT EXACTLY THIS SHAPE\n" + _json(template),
    ])
    return _prompt_result(prompt, version=PLANNER_PROMPT_VERSION)


def build_round_realizer_prompt(
    *, compiled_plan: Mapping[str, Any], tail_context: Mapping[str, Any],
) -> dict[str, Any]:
    roster_contract = compiled_plan.get("speaker_roster_contract") or {
        "round_speakers": list(dict.fromkeys(
            str(participant)
            for scene in compiled_plan.get("scenes") or []
            for participant in scene.get("participants") or []
        )),
        "scene_core_speakers": {
            str(scene["scene_ref"]): list(scene.get("participants") or [])
            for scene in compiled_plan.get("scenes") or []
        },
    }
    compiled_prompt_view = deepcopy(dict(compiled_plan))
    # The roster is shown once as a dedicated exact whitelist below instead
    # of being duplicated inside the compiled plan payload.
    compiled_prompt_view.pop("speaker_roster_contract", None)
    frozen = {
        "tail_context": _tail_prompt_view(tail_context),
        "compiled_plan": compiled_prompt_view,
        "frozen_speaker_roster": deepcopy(dict(roster_contract)),
    }
    output = {
        "realizer_schema_version": REALIZER_SCHEMA_VERSION,
        "scene_dialogues": {
            str(scene["scene_ref"]): [{
                "speaker": str((scene.get("participants") or ["..."])[index % len(
                    scene.get("participants") or ["..."])]),
                "text": "...",
            } for index in range(2)] for scene in compiled_plan.get("scenes") or []
        },
    }
    prompt = "\n\n".join([
        "SIMPLE APPEND ROUND REALIZER v2. Return JSON only.",
        "Write every planned scene in one scene_dialogues object. Each exact planned scene_ref "
        "is a key whose value is that scene's complete array of dialogue turns. Do not put a "
        "scene_ref inside a turn and do not nest one scene inside another. All scenes occur "
        "after the final parent turn. Continue naturally from the final parent sessions and "
        "let each scene assume the preceding generated scene has already happened. Never turn "
        "an action completed in an earlier scene back into a future intention. Aim for 14-18 "
        "turns per scene and 42-54 turns overall, but natural pacing matters more than "
        "hitting an exact count. Treat each scene's participants as its core focus. A speaker "
        "listed in another scene of this same round may enter when that is natural, but at "
        "least one core participant must speak in every scene. The frozen round_speakers list "
        "is an exact whitelist for every speaker field. Copy a speaker name exactly from that "
        "list. Never invent a speaker absent from the complete round cast, and never rename, "
        "abbreviate, or singularize one. If the plan mentions a non-cast person, they may be discussed "
        "but may not receive a dialogue turn. Listed new speakers may speak naturally without a "
        "formal introduction ritual. Realize concrete actions, "
        "events, information changes, decisions, disagreements, or consequences. Do not copy "
        "parent utterances, mechanically recite beats, emit Memory operations, evidence, QA, "
        "completion markers, turn IDs, nested scene objects, or any unrequested fields. Change "
        "the dramatic focus in each scene according to the plan; do not replay the same topic, "
        "question, advice, or emotional reaction across all scenes.",
        "FROZEN ROUND INPUT\n" + _json(frozen),
        "OUTPUT EXACTLY THIS SHAPE WITH EVERY PLANNED SCENE KEY\n" + _json(output),
    ])
    return _prompt_result(prompt, version=REALIZER_PROMPT_VERSION)


def build_mcq_batch_prompt(
    expanded_dialogue: Mapping[str, Any], *, target_questions: int = 8,
    maximum_questions: int | None = None,
    focus_round_index: int | None = None,
) -> dict[str, Any]:
    if target_questions < 8:
        raise ValueError("simple_append_mcq_target_out_of_range")
    if maximum_questions is not None and maximum_questions < target_questions:
        raise ValueError("simple_append_mcq_target_out_of_range")
    count_instruction = (
        f"Create at least {target_questions} grounded four-option candidate questions"
        if maximum_questions is None else
        f"Create {target_questions} to {maximum_questions} grounded four-option "
        "candidate questions")
    final_count_instruction = (
        f"Return at least {target_questions} distinct questions"
        if maximum_questions is None else
        f"Return {target_questions} to {maximum_questions} distinct questions")
    upper_count_contract = (
        "There is no question-count upper admission gate; stop when you have covered the "
        "dialogue without restating an earlier question."
        if maximum_questions is None else
        f"Do not exceed {maximum_questions} questions; prioritize option quality over "
        "producing extra candidates.")
    registry = build_evidence_alias_registry(expanded_dialogue)
    source_turns = list(expanded_dialogue.get("turns") or [])
    turns = [{
        "evidence_ref": entry["evidence_ref"],
        "speaker": row.get("speaker"),
        "text": row.get("text"),
        "origin": row.get("origin"),
        "scene_ref": row.get("scene_ref"),
    } for entry, row in zip(registry["entries"], source_turns)]
    parent_aliases = [row["evidence_ref"] for row in turns
                      if row["origin"] == "source_grounded"]
    child_aliases = [row["evidence_ref"] for row in turns
                     if row["origin"] == "generated_extension"]
    focus_aliases = ([row["evidence_ref"] for row in turns
                      if str(row.get("scene_ref") or "").startswith(
                          f"R{int(focus_round_index):02d}:")]
                     if focus_round_index is not None else child_aliases)
    if not parent_aliases or not child_aliases or not focus_aliases:
        raise ValueError("simple_append_mcq_evidence_partition_missing")
    earlier_aliases = [row["evidence_ref"] for row in turns
                       if row["evidence_ref"] not in set(focus_aliases)]
    focus_alias_set = set(focus_aliases)
    focus_turns = [row for row in turns
                   if row["evidence_ref"] in focus_alias_set]
    parent_turns = [row for row in turns
                    if row["origin"] == "source_grounded"]
    earlier_child_turns = [row for row in turns
                           if row["origin"] == "generated_extension"
                           and row["evidence_ref"] not in focus_alias_set]
    question_template = {
            "coverage_role": "current_round_only",
            "question_type": "causal",
            "question": "...",
            "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
            "correct_option": "A",
            "evidence_turn_refs": ["..."],
        }
    template_roles = (
        ["current_round_only"] * 2 +
        ["current_round_joint_history"] * 2 +
        ["parent_child"] * 2 +
        ["general"] * 2
    )
    template = {
        "mcq_focus_writer_schema_version": MCQ_FOCUS_WRITER_SCHEMA_VERSION,
        "questions": [
            {**deepcopy(question_template), "coverage_role": role}
            for role in template_roles
        ],
    }
    prompt = "\n\n".join([
        "FULL-DIALOGUE MCQ CANDIDATE WRITER v3. Return JSON only.",
        count_instruction +
        " from the complete dialogue. Generate each question, its uniquely correct "
        "answer, and its three high-quality distractors together. Prefer context-dependent "
        "reasoning over direct one-turn quotation or generic commonsense questions. Evidence "
        "origin is independent of question_type: "
        "a parent-child question may still be causal, state_change, temporal_sequence, or any "
        "other accurate reasoning type. Use at "
        "least 4 distinct question_type values from: causal, temporal_sequence, state_change, "
        "cross_turn_comparison, relationship_or_decision, cross_context_synthesis, "
        "information_visibility, information_revision, task_dependency_or_constraint. Use "
        "cross_context_synthesis only when the reasoning truly combines separated facts, not "
        "merely because evidence comes from Parent and Child. Use information_visibility for "
        "reasoning about who knew which information and when. Use information_revision for "
        "reasoning about corrected, superseded, or currently valid information. Use "
        "task_dependency_or_constraint for prerequisites, blockers, responsibility handoffs, "
        "or constraints on a task or plan. Every "
        "question_type must match what the question actually asks. Use only exact D### "
        "evidence_ref aliases shown beside the dialogue turns and include every turn needed "
        "to answer, not merely a nearby turn. Never construct or output a canonical Parent "
        "or Child turn ID; the program resolves D### aliases after generation.",
        "DIFFICULTY AND OPTION CONTRACT\n"
        "- Each question must have exactly one correct answer when the complete dialogue is read.\n"
        "- Do not ask which phrase was merely 'mentioned' when one option is an obvious quote.\n"
        "- Do not copy the correct answer wording into the question.\n"
        "- Require a cause, sequence, change, comparison, decision, or relation supported by turns.\n"
        "- First identify one factual answer slot for the item, such as person, action, object, "
        "time, cause, state, decision, or result. Every option must fill that same slot.\n"
        "- Build each distractor as a minimal counterfactual by changing only one material fact "
        "in the correct proposition. Keep it in the same event or story thread and make it "
        "plausible without the dialogue, while the complete dialogue still contradicts it. "
        "Do not swap in an unrelated event.\n"
        "- All four options must be similarly specific, fluent, and close in length. They must "
        "remain equally plausible to a solver that cannot see the dialogue.\n"
        "- Do not use generic or extreme distractors such as immediately confronting someone, "
        "ignoring everything, cutting off all contact, or focusing on only one issue unless that "
        "specific behavior is a genuine competing fact in this dialogue.\n"
        "- Use the same grammatical frame and comparable number of factual details in all four "
        "options. Do not make the correct option the only one that names the full event chain, "
        "all participants, a reason, or a consequence.\n"
        "- Do not make the gold uniquely longer, more qualified, more natural, or less absolute.\n"
        "- Avoid giveaway words such as always, never, obviously, merely, or only unless all four "
        "options use comparable qualifiers. Avoid distractors that can be discarded solely because "
        "they say unrelated, only, never, decreased, or omit a detail present in every plausible "
        "answer.\n"
        "- Before returning a question, hide the dialogue and answer key and compare only the four "
        "options. If one option is uniquely complete, cautious, fluent, or contextually plausible, "
        "rewrite all four options or omit that question.\n"
        "- A knowledgeable solver shown only the question and options should not reliably infer "
        "the answer without the dialogue. The correct_option letter is only a content binding; "
        "the program will reassign final answer positions.",
        "TOPIC AND PROPOSITION DIVERSITY (soft preference, not a rejection gate)\n"
        "- After generating two questions about the same underlying event, thread, decision, "
        "or outcome, prefer a different supported person, event, task, relationship, or "
        "historical phase when one is available. If no other reliable content exists, reuse "
        "the thread rather than inventing facts.\n"
        "- Do not merely paraphrase a question whose core gold proposition has already been "
        "used. Prefer a different supported fact. Reuse the same proposition only when the "
        "question requires materially different reasoning and evidence, not just different "
        "wording. These are generation preferences; the program does not silently delete or "
        "rewrite a paid candidate.",
        "AUTHORITATIVE SOURCE PARTITION CONTRACT\n" + _json({
            "root_id": expanded_dialogue.get("root_id"),
            "complete_dialogue_sha256": expanded_dialogue.get(
                "complete_dialogue_sha256"),
            "evidence_alias_contract": {
                "schema_version": registry["evidence_alias_schema_version"],
                "registry_sha256": registry["registry_sha256"],
                "parent_alias_range": [parent_aliases[0], parent_aliases[-1]],
                "child_alias_range": [child_aliases[0], child_aliases[-1]],
                "last_parent_evidence_ref": parent_aliases[-1],
                "first_child_evidence_ref": child_aliases[0],
                "focus_round_index": focus_round_index,
                "focus_alias_range": [focus_aliases[0], focus_aliases[-1]],
            },
        }),
        "CURRENT FOCUS ROUND — REQUIRED SOURCE FOR BOTH CURRENT-ROUND POOLS\n" +
        _json({
            "focus_round_index": focus_round_index,
            "focus_alias_range": [focus_aliases[0], focus_aliases[-1]],
            "turns": focus_turns,
        }),
        "ORIGINAL PARENT — REQUIRED HISTORY SOURCE FOR JOINT QUESTIONS\n" +
        _json({
            "alias_range": [parent_aliases[0], parent_aliases[-1]],
            "turns": parent_turns,
        }),
        "EARLIER GENERATED CHILD — OPTIONAL HISTORY, NOT A PARENT SUBSTITUTE\n" +
        _json({
            "alias_range": (
                [earlier_child_turns[0]["evidence_ref"],
                 earlier_child_turns[-1]["evidence_ref"]]
                if earlier_child_turns else None),
            "turns": earlier_child_turns,
        }),
        "FINAL HARD CHECKLIST\n" +
        final_count_instruction + " in the single questions "
        "array and no other fields. Every question must declare exactly one coverage_role: "
        "current_round_only, current_round_joint_history, parent_child, or general. Include "
        "exactly two current_round_only, exactly two current_round_joint_history, exactly "
        "two parent_child, and two to six general questions. Do not exceed a role's range "
        "and do not create additional current-round items after its two required items. "
        "A current_round_only item "
        "must cite one to four exact aliases "
        "exclusively from the current focus range. A current_round_joint_history item must "
        "cite at least one exact alias from the current focus range and at least one exact "
        "earlier alias outside that range; the earlier alias may be Original Parent or an "
        "earlier generated Child. A parent_child item must cite at least one exact Original "
        "Parent alias and at least one exact generated Child alias from any round. "
        "A general item may use any accurate evidence from the complete dialogue. "
        "coverage_role is not evidence: "
        "the program will independently resolve and verify every alias. " +
        upper_count_contract + " The entire batch will be "
        "rejected before Solver evaluation if the following current-round quotas are "
        "not met. At least "
        f"{MIN_FOCUS_ROUND_SUPPORTED} questions must "
        f"cite the current focus round ({focus_aliases[0]} through {focus_aliases[-1]}), and "
        f"at least {MIN_FOCUS_ROUND_JOINT_WITH_HISTORY} must combine current-focus "
        "evidence with earlier dialogue evidence "
        f"({earlier_aliases[0]} through {earlier_aliases[-1]} overall, excluding the focus "
        "range). In this source, Parent aliases are "
        f"{parent_aliases[0]} through {parent_aliases[-1]} and Child aliases are "
        f"{child_aliases[0]} through {child_aliases[-1]}. Prefer one to four "
        "evidence_turn_refs and copy each D### alias exactly; extra or "
        "invalid refs will be recorded as evidence warnings. Keep question types diverse, but "
        "diversity is a statistic rather "
        "than a schema gate. Do not restate an earlier question. Check every option before "
        "returning JSON.",
        "OUTPUT EXACTLY THIS SHAPE\n" + _json(template),
    ])
    return _prompt_result(prompt, version=MCQ_PROMPT_VERSION)
