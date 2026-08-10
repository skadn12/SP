"""Identity-frozen launcher for the unique slim-outline Planner probe 11."""

from __future__ import annotations

from pathlib import Path

from . import planner_probe_10 as previous


runtime = previous.runtime
runtime.PROBE_VERSION = "v11_slim_outline_planner_probe_11_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Slim Outline Planner-only Probe 11"
runtime.PROBE_RUN_SUFFIX = "v11_planner_probe_11"
runtime.PREFLIGHT_VERSION = "v11_planner_probe_11_preflight_v1"
runtime.REQUEST_ID = "planner_v11_probe_11_request_01"
runtime.REPORT_BASENAME = "V11_PLANNER_PROBE_11_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_11"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "probe_10": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_10"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "probe_10": "ec46bdeda2ea275af31c8b764a5f6de61f4e3968ceb49b063dd56507868c6187",
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
