"""Identity-frozen launcher for dialogue-first participant Planner probe 2."""

from __future__ import annotations

from pathlib import Path

from . import dialogue_first_participant_planner_probe_v11 as previous


runtime = previous.runtime
runtime.PROBE_VERSION = "v11_dialogue_first_participant_planner_probe_2_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Dialogue-First Participant Planner-only Probe 2"
runtime.PROBE_RUN_SUFFIX = "v11_dialogue_first_participant_planner_probe_2"
runtime.PREFLIGHT_VERSION = "v11_dialogue_first_participant_probe_2_preflight_v1"
runtime.REQUEST_ID = "planner_v11_dialogue_first_participant_probe_02"
runtime.REPORT_BASENAME = "V11_DIALOGUE_FIRST_PARTICIPANT_PLANNER_PROBE_2_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_planner_probe_2"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "dialogue_first_participant_probe_1": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
        "planner_probe_1"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "dialogue_first_participant_probe_1": (
        "69653522d9645b7e5297aeea97a245ae96ce823359f86e8f460d3c402fb40269"
    ),
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
