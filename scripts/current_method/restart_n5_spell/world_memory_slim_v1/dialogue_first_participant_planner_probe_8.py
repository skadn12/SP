"""Identity-frozen launcher for the disjoint-value-namespace Planner probe 8."""

from __future__ import annotations

from pathlib import Path

from . import dialogue_first_participant_planner_probe_7 as previous


runtime = previous.runtime
runtime.PROBE_VERSION = "v11_dialogue_first_participant_planner_probe_8_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Dialogue-First Participant Planner-only Probe 8"
runtime.PROBE_RUN_SUFFIX = "v11_dialogue_first_participant_planner_probe_8"
runtime.PREFLIGHT_VERSION = "v11_dialogue_first_participant_probe_8_preflight_v1"
runtime.REQUEST_ID = "planner_v11_dialogue_first_participant_probe_08"
runtime.REPORT_BASENAME = "V11_DIALOGUE_FIRST_PARTICIPANT_PLANNER_PROBE_8_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
    "planner_probe_8"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "dialogue_first_participant_probe_7": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
        "planner_probe_7"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "dialogue_first_participant_probe_7": (
        "2b1d2aea5cc3d82fbb738c9a444af1cd1193aba702f16634604b48f6ec6bd0c3"
    ),
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
