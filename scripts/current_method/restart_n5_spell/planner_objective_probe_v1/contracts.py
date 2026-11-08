from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from scripts.current_method.restart_n5_spell.scene_complexity_probe_v1.contracts import (
    Setting, validate_plan,
)


@dataclass(frozen=True)
class Objective:
    key: str
    label: str
    instruction: str


FIXED_SCENE_COUNT = 3
FIXED_MIN_BEATS = 3
FIXED_MAX_BEATS = 7

OBJECTIVES = {
    "O0": Objective(
        "O0", "current_completion_objective",
        "Advance the event coherently through the available scenes and produce a concrete bounded result. Do not force a reconsideration or revision when the event does not naturally require one.",
    ),
    "O1": Objective(
        "O1", "explicit_state_evolution_objective",
        "Build one coherent unresolved-to-revised-to-resolved chain. Scene 1 establishes a tentative open proposal. In scene 2, one concrete new trigger must cause reconsideration or revision and the chain must remain non-final. Scene 3 reaches the first final resolution. Do not fake an arbitrary reversal: the trigger and revision must be plausible continuations of the frozen event.",
    ),
}


def setting_for(objective_key: str) -> Setting:
    if objective_key not in OBJECTIVES:
        raise KeyError(objective_key)
    return Setting(
        objective_key, OBJECTIVES[objective_key].label,
        FIXED_SCENE_COUNT, FIXED_MIN_BEATS, FIXED_MAX_BEATS,
    )


ROLE_NAME_PATTERN = re.compile(
    r"\b(?:(?:New\s+)?(?:Classmates?|Co-workers?|Neighbors?|Friends?|Colleagues?)|"
    r"Specialist|Boss|Community\s+Organizer|Local\s+Park\s+Ranger)"
    r"(?:\s+[A-Z])?\b")


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def validate_objective_plan(
    value: Any, *, objective_key: str, case_id: str,
    valid_speakers: Sequence[str],
) -> dict[str, Any]:
    base = validate_plan(value, setting=setting_for(objective_key), case_id=case_id)
    errors = list(base["errors"])
    allowed = set(valid_speakers)
    unknown = sorted({match.group(0) for text in _strings(value)
                      for match in ROLE_NAME_PATTERN.finditer(text)
                      if match.group(0) not in allowed})
    if unknown:
        errors.append("noncanonical_speakers:" + ",".join(unknown))
    return {**base, "valid": not errors, "errors": errors,
            "noncanonical_speakers": unknown}
