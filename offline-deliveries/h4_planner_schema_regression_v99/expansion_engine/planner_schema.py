"""Single source of truth for the planner's route-specific output contract.

The planner has one scene slot, but the old-event and dataset-card routes do
not have the same scene fields.  Keeping the complete shape here prevents the
prompt's introductory schema and its retry reminder from drifting apart.
This module only renders instructions; validation remains in the existing
route validators.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any
import event_identity


OLD_EVENT_CONTINUATION = "old_event_continuation"
DATASET_NEW_EVENT = "dataset_new_event"
ROUTES = (OLD_EVENT_CONTINUATION, DATASET_NEW_EVENT)

def _state_transition() -> dict[str, Any]:
    return {
        "basis_fact_ids": ["<fact_id from latest_state>"],
        "new_event": "<specific new progress or event>",
        "temporal_placement": "<when this occurs relative to history>",
    }


def route_schema(route: str, card: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the complete output shape for *route* (with placeholders)."""
    if route not in ROUTES:
        raise ValueError(f"unknown planner route: {route}")
    scene: dict[str, Any] = {
        key: deepcopy(value) for key, value in (
            ("scene_id", "<target_scene_id>"),
            ("actors", ["<allowed speaker>", "<another allowed speaker>"]),
            ("source_anchors", [{"turn_ref": "<exact history turn_ref>",
                                  "quote": "<exact history substring>"}]),
            ("goal", "<concrete short goal>"),
            ("developments", ["<new development 1>", "<new development 2>" ]),
            ("feedback_topic_ids", []),
            ("state_transition", _state_transition()),
        )
    }
    if route == OLD_EVENT_CONTINUATION:
        scene.update({"dataset_card_id": None, "card_development_links": []})
        # Preserve the established validator order while making this route's
        # complete shape explicit in the prompt.
        ordered = {key: scene[key] for key in (
            "scene_id", "actors", "source_anchors", "goal", "developments",
            "dataset_card_id", "card_development_links", "feedback_topic_ids",
            "state_transition")}
    else:
        scene.update({
            "dataset_card_id": card.get("card_id", "<assigned card_id>") if card else "<assigned card_id>",
            "card_development_links": [{
                "card_step_ids": ["<actual step_id>"],
                "dependency_indices": [],
                "development_indices": [1],
            }],
            # Delegate the nested shape to the same source used by the
            # identity validator and mapping_context; never copy its fields.
            "event_identity": event_identity.identity_schema(
                card if card is not None else {"actors": [{"actor_id": "<card actor_id>"}]}
            ),
        })
        ordered = {key: scene[key] for key in (
            "scene_id", "actors", "source_anchors", "goal", "developments",
            "dataset_card_id", "card_development_links", "feedback_topic_ids",
            "state_transition", "event_identity")}
    return {
        "root_id": "<root_id>", "route": route,
        "scenes": [ordered], "shortfall_reason": "",
    }


def schema_text(route: str, card: dict[str, Any] | None = None) -> str:
    """Render the exact route shape used in both initial and retry prompts."""
    return json.dumps(route_schema(route, card), ensure_ascii=False, indent=2)


def route_prompt(route: str, card: dict[str, Any] | None = None) -> str:
    """Preserve the existing task with one authoritative complete route schema."""
    if route not in ROUTES:
        raise ValueError(f"unknown planner route: {route}")
    common_start = """Plan at most ONE new scene after this branch's cumulative_history.
This is one scene slot, not a request to fill three scenes. Return exactly
{root_id,route,scenes,shortfall_reason}. scenes has zero or one entry. For zero
scenes explain why; for one scene shortfall_reason must be the JSON string "".
shortfall_reason must always be a string, never null.
Use the complete route-specific scene schema below.
scene_id MUST equal target_scene_id.
actors: at least two distinct allowed_speakers. source_anchors: 1-3 exact
{turn_ref,quote} entries from cumulative_history. goal: concrete short string.
developments: 2-5 concrete achievable new changes, decisions or outcomes.
Only scene actors may speak directly in the later dialogue. Make every planned
development expressible by these actors. They may accurately report a third
party's action, but never require that third party to speak or join this scene.
feedback_topic_ids: at most four IDs from supplied previous-round feedback.
"""
    if route == DATASET_NEW_EVENT:
        route_task = """For dataset_new_event, introduce a distinct new event from the single supplied
dataset card; dataset_card_id MUST be its card_id. Transfer a scaffold's
relationships/conditions or a seed's premise, never source identities as facts.
The separate CARD MAPPING CONTRACT defines card_development_links and its three
reference spaces. Build the new event from that card, using history only to keep
the participants and circumstances coherent. An unrelated continuation of an
old donation, class, work task or trip does not instantiate a new event card.
For this route the assigned card determines the new event. Previous-round
feedback guides clarity and coverage; its historical example topics must not
replace the card's event. Prior history anchors justify continuity, not a return
to unfinished old tasks as the main event of this scene.
"""
    else:
        route_task = """For old_event_continuation, progress an existing event; dataset_card_id is null
and card_development_links is empty.
"""
    common_end = """Keep proposals, agreements, attempts and outcomes distinct. A plan need not
force all actions to finish, but every listed development must appear in the
new dialogue. Do not merely rephrase old events or create reassurance filler.
The other expansion branch is not part of this branch's history.
"""
    return (common_start + route_task + common_end
            + "\nCOMPLETE OUTPUT SCHEMA (authoritative for this route; return no other keys):\n"
            + schema_text(route, card)
            + "\nUse the actual supplied root, scene, history and latest-state IDs in place of placeholders.\n")


def retry_tail(route: str, errors: list[dict[str, Any]] | None = None,
               card: dict[str, Any] | None = None) -> str:
    """Render a precise repair tail, ending with the complete route schema."""
    if route not in ROUTES:
        raise ValueError(f"unknown planner route: {route}")
    normalized = errors if isinstance(errors, list) else []
    error_json = json.dumps(normalized, ensure_ascii=False, indent=2)
    source_rule = (
        "history and assigned card. Correct every listed path using its actual value and\n"
        "allowed value; do not silently drop a card step, dependency, development or\n"
        "negative requirement. Return the entire JSON response again."
        if route == DATASET_NEW_EVENT else
        "history. Correct every listed path using its actual value and allowed value;\n"
        "preserve the planned continuity and return the entire JSON response again."
    )
    return f"""
PLANNER RETRY — FIX THE LISTED ERRORS IN THE COMPLETE RESPONSE
The previous response is rejected. Preserve the same root, route, scene slot,
{source_rule}
EXACT VALIDATION ERRORS (machine-readable; do not paraphrase):
{error_json}

COMPLETE OUTPUT SCHEMA (authoritative; this is the final instruction):
{schema_text(route, card)}
"""


def error_list(repair: Any) -> list[dict[str, Any]]:
    """Extract only structured validator errors for the retry tail."""
    if not isinstance(repair, dict) or not isinstance(repair.get("errors"), list):
        return []
    return [deepcopy(item) for item in repair["errors"] if isinstance(item, dict)]

