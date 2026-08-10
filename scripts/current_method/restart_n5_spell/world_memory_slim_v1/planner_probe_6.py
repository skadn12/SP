"""Identity-frozen launcher for the unique v11 Planner-only probe 6."""

from __future__ import annotations

from pathlib import Path

from . import planner_probe_5 as runtime


runtime.PROBE_VERSION = "v11_planner_only_probe_6_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Planner-only Probe 6"
runtime.PROBE_RUN_SUFFIX = "v11_planner_probe_6"
runtime.PREFLIGHT_VERSION = "v11_planner_probe_6_preflight_v1"
runtime.REQUEST_ID = "planner_v11_probe_6_request_01"
runtime.REPORT_BASENAME = "V11_PLANNER_PROBE_6_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_6")
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "probe_5": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_5"),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "probe_5": "14d6b65d34b48986fcd8a55a0ec32b2311ad9a87539c8b7197607f63619ee4d3",
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
