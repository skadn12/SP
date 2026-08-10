"""Identity-frozen launcher for the unique duplicate-prevention Planner probe 10."""

from __future__ import annotations

from pathlib import Path

from . import planner_probe_9 as previous


runtime = previous.runtime
runtime.PROBE_VERSION = "v11_planner_duplicate_prevention_probe_10_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Planner Duplicate-Prevention Planner-only Probe 10"
runtime.PROBE_RUN_SUFFIX = "v11_planner_probe_10"
runtime.PREFLIGHT_VERSION = "v11_planner_probe_10_preflight_v1"
runtime.REQUEST_ID = "planner_v11_probe_10_request_01"
runtime.REPORT_BASENAME = "V11_PLANNER_PROBE_10_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_10"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "probe_9": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_9"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "probe_9": "d9b0111a64f36179ffc622d03919acb58db87d0c29f2d308e6228d90fe19c757",
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
