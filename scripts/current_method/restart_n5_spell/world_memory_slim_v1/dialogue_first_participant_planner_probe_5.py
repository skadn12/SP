"""Identity-frozen launcher for the action-declared participant Planner probe 5."""

from __future__ import annotations

from pathlib import Path

from . import dialogue_first_participant_planner_probe_4 as previous


runtime = previous.runtime
runtime.PROBE_VERSION = "v11_dialogue_first_participant_planner_probe_5_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Dialogue-First Participant Planner-only Probe 5"
runtime.PROBE_RUN_SUFFIX = "v11_dialogue_first_participant_planner_probe_5"
runtime.PREFLIGHT_VERSION = "v11_dialogue_first_participant_probe_5_preflight_v1"
runtime.REQUEST_ID = "planner_v11_dialogue_first_participant_probe_05"
runtime.REPORT_BASENAME = "V11_DIALOGUE_FIRST_PARTICIPANT_PLANNER_PROBE_5_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_planner_probe_5"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "dialogue_first_participant_probe_4": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
        "planner_probe_4"
    ),
    "live_canary_16": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_slim_continuous_expansion_"
        "v11_live_canary_16"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "dialogue_first_participant_probe_4": (
        "4ba22cffef25a940c5feb0af6acb0645db0887c642ad2a36d709ddb9805cfcb0"
    ),
    "live_canary_16": (
        "1b6fc0e04658afc08104240a79184704b4893262d477e1b5adc4cd4dd242b5e5"
    ),
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
