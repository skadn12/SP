"""Unique Planner-only probe for the frozen v11 person/event growth contract.

This launcher deliberately reuses the established single-physical-request probe
runtime.  It only freezes the person/event growth profile into the Planner input
and adds that profile's coverage check to the normal Planner validation chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import planner_probe_11 as previous
from ..core import sha256_bytes
from .growth_profile_v11 import (
    frozen_person_event_growth_profile,
    validate_planner_growth_profile_v11,
)
from .live_contracts_v11 import build_planner_prompt_v11


runtime = previous.runtime
_base_build_planner_probe_input = runtime.build_planner_probe_input
_base_validate_planner_outline = runtime.validate_planner_outline_v11


def _build_profiled_planner_probe_input(input_path: Path) -> dict[str, Any]:
    values = _base_build_planner_probe_input(input_path)
    profile = frozen_person_event_growth_profile()
    prompt, frozen_input_sha = build_planner_prompt_v11(
        root_id=runtime.ROOT_ID,
        planning_state=values["planning_state"],
        source_anchor_registry=values["source_anchor_registry"],
        placement_choice_registry=values["placement_choice_registry"],
        action_contracts=values["shared_narrative_action_contracts"],
        action_availability_view=values["action_availability_view"],
        growth_profile=profile,
    )
    values.update({
        "growth_profile": profile,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "frozen_planner_input_sha256": frozen_input_sha,
    })
    return values


def _validate_profiled_planner_outline(
    outline: Mapping[str, Any], **kwargs: Any,
) -> dict[str, Any]:
    structural = _base_validate_planner_outline(outline, **kwargs)
    growth = validate_planner_growth_profile_v11(
        outline, profile=frozen_person_event_growth_profile())
    result = dict(structural)
    result["growth_profile_validation"] = growth
    result["errors"] = list(dict.fromkeys([
        *list(structural.get("errors") or []),
        *list(growth.get("errors") or []),
    ]))
    result["valid"] = structural.get("valid") is True and growth["valid"]
    return result


runtime.PROBE_VERSION = "v11_person_event_planner_probe_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Person/Event Growth Planner-only Probe 1"
runtime.PROBE_RUN_SUFFIX = "v11_person_event_planner_probe_1"
runtime.PREFLIGHT_VERSION = "v11_person_event_planner_probe_preflight_v1"
runtime.REQUEST_ID = "planner_v11_person_event_probe_01"
runtime.REPORT_BASENAME = "V11_PERSON_EVENT_PLANNER_PROBE_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_person_event_planner_probe_1"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "probe_11": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_11"
    ),
    "live_canary_15": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_slim_continuous_expansion_"
        "v11_live_canary_15"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "probe_11": "81e34471fc5fe1dd2435cce1cd05869538d6911c0388fd230262eace5885c016",
    "live_canary_15": "fca44d7b6e822ba9d6e8cf547d604d86b34008c47643f5f93212e536f31fc4b0",
}
runtime.build_planner_probe_input = _build_profiled_planner_probe_input
runtime.validate_planner_outline_v11 = _validate_profiled_planner_outline


if __name__ == "__main__":
    raise SystemExit(runtime.main())
