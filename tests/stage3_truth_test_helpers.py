"""Deterministic, non-repeating API-context fixtures for Truth-first tests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


_ACTORS = (
    "the workshop coordinator",
    "a visiting artist",
    "the neighborhood volunteer",
    "an experienced organizer",
    "the careful note taker",
    "a curious participant",
    "the venue manager",
    "a local designer",
    "the project mentor",
    "an attentive listener",
    "the planning assistant",
    "a community member",
    "the travel adviser",
    "a practical reviewer",
    "the materials lead",
    "an invited speaker",
    "the scheduling helper",
)
_ACTIONS = (
    "reviews",
    "reframes",
    "connects",
    "examines",
    "weighs",
    "clarifies",
    "organizes",
    "compares",
    "summarizes",
    "explores",
    "develops",
    "traces",
    "balances",
)
_OBJECTS = (
    "the practical schedule",
    "the travel arrangements",
    "the shared materials",
    "the room layout",
    "the budget notes",
    "the preparation checklist",
    "the neighborhood concerns",
    "the design choices",
    "the available equipment",
    "the follow through",
    "the group priorities",
    "the coordination details",
    "the creative possibilities",
    "the next discussion point",
    "the logistical tradeoffs",
    "the participant feedback",
    "the remaining questions",
    "the working timeline",
    "the resource plan",
)
_DETAILS = (
    "before anyone chooses the next task",
    "while everyone considers the practical consequences",
    "so the exchange keeps moving with purpose",
    "without losing the thread from the previous remark",
    "as the group prepares a thoughtful response",
    "before the conversation turns toward another concern",
    "while both speakers test the idea from different angles",
    "so each suggestion receives a concrete reaction",
    "as neighboring points begin to fit together",
    "before the participants settle on a useful follow up",
    "while the discussion stays grounded in shared experience",
    "so the speakers can refine their understanding together",
    "as the practical implications become easier to discuss",
    "before a new question opens the next part of the exchange",
    "while the group keeps its attention on workable choices",
    "so the response builds naturally on what came earlier",
    "as each person contributes a different perspective",
)


def context_sentence(index: int, topic: str) -> str:
    """Return a substantive topic-linked sentence with a unique composition."""

    actor = _ACTORS[index % len(_ACTORS)]
    action = _ACTIONS[(index * 5 + 3) % len(_ACTIONS)]
    object_text = _OBJECTS[(index * 7 + 2) % len(_OBJECTS)]
    detail = _DETAILS[(index * 11 + 1) % len(_DETAILS)]
    templates = (
        f"During the {topic} discussion, {actor} {action} {object_text} {detail}.",
        f"{actor.capitalize()} {action} {object_text}; the {topic} exchange stays focused {detail}.",
        f"With the {topic} conversation underway, {actor} {action} {object_text} {detail}.",
        f"Building on the prior remark, {actor} {action} {object_text} around {topic} {detail}.",
        f"The {topic} exchange continues as {actor} {action} {object_text} {detail}.",
        f"Answering the preceding point thoughtfully, {actor} {action} {object_text} near {topic} {detail}.",
    )
    return templates[index % len(templates)]


def flat_context_body(
    turn_slots: Sequence[Mapping[str, Any]], topic: str
) -> dict[str, Any]:
    """Build either current compact or legacy API context for a local plan."""

    sentence_index = 0
    turns: list[dict[str, Any]] = []
    for slot in turn_slots:
        if (
            "target_context_sentence_count" in slot
            and "lead_sentence_count" not in slot
        ):
            context: list[str] = []
            for _ in range(int(slot["target_context_sentence_count"])):
                context.append(context_sentence(sentence_index, topic))
                sentence_index += 1
            turns.append(
                {
                    "turn_slot_id": slot["turn_slot_id"],
                    "context_sentences": context,
                }
            )
            continue
        lead: list[str] = []
        for _ in range(int(slot["lead_sentence_count"])):
            lead.append(context_sentence(sentence_index, topic))
            sentence_index += 1
        follow: list[str] = []
        for _ in range(int(slot["follow_sentence_count"])):
            follow.append(context_sentence(sentence_index, topic))
            sentence_index += 1
        turns.append(
            {
                "turn_slot_id": slot["turn_slot_id"],
                "lead_in_sentences": lead,
                "follow_up_sentences": follow,
            }
        )
    return {"turns": turns}


__all__ = ["context_sentence", "flat_context_body"]
