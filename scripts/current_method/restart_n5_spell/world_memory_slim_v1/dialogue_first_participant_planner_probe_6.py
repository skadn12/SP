"""Identity-frozen launcher for the disjoint-reference participant Planner probe 6."""

from __future__ import annotations

from pathlib import Path

from . import dialogue_first_participant_planner_probe_5 as previous


runtime = previous.runtime
runtime.PROBE_VERSION = "v11_dialogue_first_participant_planner_probe_6_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Dialogue-First Participant Planner-only Probe 6"
runtime.PROBE_RUN_SUFFIX = "v11_dialogue_first_participant_planner_probe_6"
runtime.PREFLIGHT_VERSION = "v11_dialogue_first_participant_probe_6_preflight_v1"
runtime.REQUEST_ID = "planner_v11_dialogue_first_participant_probe_06"
runtime.REPORT_BASENAME = "V11_DIALOGUE_FIRST_PARTICIPANT_PLANNER_PROBE_6_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_planner_probe_6"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "dialogue_first_participant_probe_5": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
        "planner_probe_5"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "dialogue_first_participant_probe_5": (
        "f87476acb598eac387d7451986fb50b5b08b8a6c01edc727ff02840058e3d6e1"
    ),
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
