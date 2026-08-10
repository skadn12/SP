"""Identity-frozen launcher for the alias-reuse participant Planner probe 4."""

from __future__ import annotations

from pathlib import Path

from . import dialogue_first_participant_planner_probe_3 as previous


runtime = previous.runtime
runtime.PROBE_VERSION = "v11_dialogue_first_participant_planner_probe_4_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Dialogue-First Participant Planner-only Probe 4"
runtime.PROBE_RUN_SUFFIX = "v11_dialogue_first_participant_planner_probe_4"
runtime.PREFLIGHT_VERSION = "v11_dialogue_first_participant_probe_4_preflight_v1"
runtime.REQUEST_ID = "planner_v11_dialogue_first_participant_probe_04"
runtime.REPORT_BASENAME = "V11_DIALOGUE_FIRST_PARTICIPANT_PLANNER_PROBE_4_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_planner_probe_4"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "dialogue_first_participant_probe_3": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
        "planner_probe_3"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "dialogue_first_participant_probe_3": (
        "cb074e719f70fff891c0fa31176bcf65246476a372b0b6bbb0f31ddcd0ea375d"
    ),
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
