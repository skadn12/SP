"""Append the current candidate task after frozen history and feedback.

This adds grounded generation guidance, not another semantic admission gate.
The supplied card must be the full validated card, including original evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


TAIL_MARKER = "CURRENT TASK AFTER ALL HISTORICAL DATA AND FROZEN FEEDBACK"
TARGET_MARKER = "TARGET_AND_ORIGINAL_CARD_JSON\n"
ROLES = ("planner", "realizer", "reviewer")
ROUTES = ("old_event_continuation", "dataset_new_event")


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def append_current_task(payload, proof, *, root, route, scene_id, role, history,
                        card, plan=None, realization=None, repair=None):
    """Return copied payload/proof with a final task bound to its exact inputs.

    Prior messages, full history and the frozen feedback receipt are preserved.
    The receipt covers the intermediate request; separate hashes bind the append
    and the final request. No source-specific roles or dialogue are supplied here.
    """
    if role not in ROLES:
        raise ValueError("current_task_role_invalid")
    if route not in ROUTES:
        raise ValueError("current_task_route_invalid")
    if route == ROUTES[1] and not isinstance(card, dict):
        raise ValueError("current_task_B_card_missing")
    result, result_proof = deepcopy(payload), deepcopy(proof)
    messages = result.get("messages", [])
    user_indices = [i for i, message in enumerate(messages) if message.get("role") == "user"]
    if not user_indices or not isinstance(messages[user_indices[-1]].get("content"), str):
        raise ValueError("current_task_user_text_missing")
    # Current callers have one user message. Appending a later message could put
    # the current task before other instructions and defeat its intended order.
    if user_indices[-1] != len(messages) - 1:
        raise ValueError("current_task_last_message_not_user")

    target = {
        "root_id": root,
        "route": route,
        "scene_id": scene_id,
        "current_task_role": role,
        "dataset_card_id": card.get("card_id") if card else None,
        "allowed_speakers": sorted({turn["speaker"] for turn in history}),
        "source_card_with_original_evidence": deepcopy(card),
        "draft_actor_mapping": deepcopy(plan.get("actor_mapping")) if isinstance(plan, dict) else None,
        "advisory_plan": deepcopy(plan),
        "new_scene_to_diagnose": deepcopy(realization) if role == "reviewer" else None,
        "mechanical_repair": deepcopy(repair),
    }
    tail = "\n\n" + TAIL_MARKER + "\n" + """Only the target identity below applies to this response. Earlier scene IDs and
old output objects belong to the supplied history. Read the complete history
above; preserve its actual words, commitments and action states. Do not replay
or paraphrase an old scene as the new output. Provisional candidate dialogue is
continuity context, not admitted training data or a certified reviewer account.
Only allowed_speakers may have direct speaking turns. Attribute other people's
actions explicitly without transferring them to a speaking character.
"""
    if card is not None:
        tail += """
Use the full source card and its ORIGINAL QUOTED EVIDENCE below to understand
the core interaction, including what each request or action is about. Source
evidence describes a separate source situation; it is NOT an earlier event or
conversation in this root's history. Introduce a compatible NEW occasion now.
Do not invent prior discussion or agreement to connect this occasion to history.
Adapt names, places, institutions and compatible values when useful, while
preserving the source's purpose, actor roles, action, object and action status.
Do not change the meaning of a request or object just to fit an old task. A
consideration, request, offer or agreement is not evidence of completed action.
Keep distinct actors' responsibilities and perspectives distinct. A valid draft
actor_mapping is guidance for this adaptation, not evidence that events occurred.
If there is no draft mapping, choose compatible allowed speakers for source roles
or explicitly attributed non-speaking participants; do not add mapping fields to
the required dialogue output. Produce the recognizable core interaction and
substantive new information, a decision or another supported development. Not
every optional source step or planned development must occur in this scene.
"""
    else:
        tail += """
Continue an event grounded in the actual history with substantive new action,
information, a decision or an outcome. The plan is advisory and cannot establish
historical facts. Not every planned development must occur in this scene.
"""
    if role == "planner":
        tail += """
Return one advisory plan using the planner JSON schema already specified above.
Use the exact target root_id, route, scene_id and dataset_card_id. Ground any
historical anchors in the full history. For B, map all source actor IDs without
changing their roles; do not output old dialogue or a completed-scene review.
"""
        ending = "END CURRENT TASK. Generate only the advisory plan now.\n"
    elif role == "realizer":
        tail += """
Write ONLY the NEW scene for this target. Use a faithful advisory plan if present;
otherwise compose directly from the history and original source evidence. Aim
for 12-16 natural turns; length is guidance. Return one JSON object with root_id,
route, scene_id, dataset_card_id, status, turns:[{speaker,text}],
feedback_topic_ids:[], no_scene_reason. Use status="scene"; if impossible, return
status="no_scene", turns:[], and an honest reason. Do not return a plan or review.
"""
        ending = "END CURRENT TASK. Generate only the new scene now.\n"
    else:
        tail += """
Diagnose ONLY new_scene_to_diagnose below, using the reviewer JSON schema already
specified above. Judge actual dialogue against history and original evidence,
not a plan's intended outcome. Cite exact NEW/history references for content
problems. Ambiguity is uncertain. Missing optional steps or plan coverage belongs
in observations, not an automatic core failure. Do not rewrite the scene or turn
an intention into a completion in your summary. This is advisory candidate
review, not automatic training admission.
"""
        ending = "END CURRENT TASK. Generate only the advisory review now.\n"
    tail += "\n" + TARGET_MARKER + json.dumps(target, ensure_ascii=False) + "\n" + ending
    result["messages"][user_indices[-1]]["content"] += tail
    result_proof.update({
        "feedback_receipt_scope": "intermediate_before_tail",
        "pre_tail_payload_sha256": digest(payload),
        "final_payload_sha256": digest(result),
        "source_card_with_evidence_sha256": digest(card),
        "raw_card_sha256": digest(card),
        "final_task_sha256": digest(tail),
        "current_task_role": role,
        "current_task_history_sha256": digest(history),
        "current_task_history_turns": len(history),
        "current_task_version": "v43_grounded_current_task",
    })
    return result, result_proof
