"""Identity-frozen launcher for the unique shared-action Planner probe 7."""

from __future__ import annotations

from pathlib import Path

from . import planner_probe_5 as runtime


runtime.PROBE_VERSION = "v11_shared_narrative_action_planner_probe_7_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Shared Narrative Action Planner-only Probe 7"
runtime.PROBE_RUN_SUFFIX = "v11_planner_probe_7"
runtime.PREFLIGHT_VERSION = "v11_planner_probe_7_preflight_v1"
runtime.REQUEST_ID = "planner_v11_probe_7_request_01"
runtime.REPORT_BASENAME = "V11_PLANNER_PROBE_7_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_7")
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "probe_5": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_5"),
    "probe_6": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_6"),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "probe_5": "14d6b65d34b48986fcd8a55a0ec32b2311ad9a87539c8b7197607f63619ee4d3",
    "probe_6": "9690e2dcb6cc0f9dd2afb074b9c413802daf0b265f6c13a5c8c70c84f3504ed0",
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
