"""Identity-frozen launcher for the speaker/action-boundary Planner probe 7."""

from __future__ import annotations

from pathlib import Path

from . import dialogue_first_participant_planner_probe_6 as previous


runtime = previous.runtime
runtime.PROBE_VERSION = "v11_dialogue_first_participant_planner_probe_7_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Dialogue-First Participant Planner-only Probe 7"
runtime.PROBE_RUN_SUFFIX = "v11_dialogue_first_participant_planner_probe_7"
runtime.PREFLIGHT_VERSION = "v11_dialogue_first_participant_probe_7_preflight_v1"
runtime.REQUEST_ID = "planner_v11_dialogue_first_participant_probe_07"
runtime.REPORT_BASENAME = "V11_DIALOGUE_FIRST_PARTICIPANT_PLANNER_PROBE_7_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
    "planner_probe_7"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "dialogue_first_participant_probe_6": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
        "planner_probe_6"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "dialogue_first_participant_probe_6": (
        "7c8ed556f5ec6a19fb869c9ba6f1b0385d5300d63009c194191760c1ae06d8a3"
    ),
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
