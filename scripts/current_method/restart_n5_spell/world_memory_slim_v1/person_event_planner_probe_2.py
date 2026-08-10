"""Identity-frozen launcher for person/event Planner-only probe 2."""

from __future__ import annotations

from pathlib import Path

from . import person_event_planner_probe_v11 as previous


runtime = previous.runtime
runtime.PROBE_VERSION = "v11_person_event_planner_probe_2_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Person/Event Growth Planner-only Probe 2"
runtime.PROBE_RUN_SUFFIX = "v11_person_event_planner_probe_2"
runtime.PREFLIGHT_VERSION = "v11_person_event_planner_probe_2_preflight_v1"
runtime.REQUEST_ID = "planner_v11_person_event_probe_02"
runtime.REPORT_BASENAME = "V11_PERSON_EVENT_PLANNER_PROBE_2_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_person_event_planner_probe_2"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "person_event_probe_1": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_person_event_planner_probe_1"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "person_event_probe_1": (
        "2bfb3faf8dbce246bcf7f58b359d5b6367403979836bc873722b9618b0cac63c"
    ),
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
