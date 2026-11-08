"""Explicitly authored offline annotations, never a production repair helper."""
from copy import deepcopy
import event_identity


def comparison(card, history, scene):
    """Only migrate synthetic test envelopes to the v98 required shape.

    These generic positive labels have no claim to independent semantic truth.
    Real archived responses must not be upgraded with this helper in production.
    """
    people = list(dict.fromkeys(t["speaker"] for t in history if t.get("speaker")))
    if not people:
        people = scene.get("actors") or ["OFFLINE fixture participant"]
    evidence = [{"turn_ref": history[0]["turn_ref"], "quote": history[0]["text"][:70]}] if history else []
    result = {
        "purpose": {"source": "OFFLINE assigned function", "actual": "OFFLINE assigned function"},
        "start_state": {"source": "Pending interaction", "actual": "Pending interaction"},
        "end_state": {"source": "Interaction reported", "actual": "Interaction reported"},
        "roles": [{"actor_id": actor, "participant": people[i % len(people)], "role": "OFFLINE role"}
                  for i, actor in enumerate(event_identity.actor_ids(card))],
        "current_trigger": "OFFLINE new interaction",
        "nearest_old_event": {"summary": "OFFLINE annotated prior context", "history_evidence": evidence,
                              "difference": "OFFLINE newly supplied interaction"},
        "request_acceptance": {"required": False, "verdict": "compatible",
                               "request_evidence": [], "acceptance_evidence": []},
        "experience_state": {"source": "other", "actual": "other", "verdict": "compatible"},
    }
    return result


def identity(card, history, scene):
    result = comparison(card, history, scene)
    for key in ("purpose", "start_state", "end_state"):
        result[key] = result[key]["actual"]
    result.pop("request_acceptance"); result.pop("experience_state")
    return deepcopy(result)
