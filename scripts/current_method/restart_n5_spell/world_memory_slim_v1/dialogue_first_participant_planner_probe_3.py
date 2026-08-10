"""Identity-frozen launcher for the alias-only participant Planner probe 3."""

from __future__ import annotations

from pathlib import Path

from . import dialogue_first_participant_planner_probe_2 as previous


runtime = previous.runtime
runtime.PROBE_VERSION = "v11_dialogue_first_participant_planner_probe_3_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Dialogue-First Participant Planner-only Probe 3"
runtime.PROBE_RUN_SUFFIX = "v11_dialogue_first_participant_planner_probe_3"
runtime.PREFLIGHT_VERSION = "v11_dialogue_first_participant_probe_3_preflight_v1"
runtime.REQUEST_ID = "planner_v11_dialogue_first_participant_probe_03"
runtime.REPORT_BASENAME = "V11_DIALOGUE_FIRST_PARTICIPANT_PLANNER_PROBE_3_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_planner_probe_3"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "dialogue_first_participant_probe_2": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
        "planner_probe_2"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "dialogue_first_participant_probe_2": (
        "9ec1139411667f5e37649280fb38de4a972a96e283996ee629b42a3101e04563"
    ),
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
