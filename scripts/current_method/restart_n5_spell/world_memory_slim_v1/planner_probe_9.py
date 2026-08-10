"""Identity-frozen launcher for the unique namespace/reference Planner probe 9."""

from __future__ import annotations

from pathlib import Path

from . import planner_probe_5 as runtime


runtime.PROBE_VERSION = "v11_planner_namespace_reference_probe_9_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Planner Namespace/Reference Planner-only Probe 9"
runtime.PROBE_RUN_SUFFIX = "v11_planner_probe_9"
runtime.PREFLIGHT_VERSION = "v11_planner_probe_9_preflight_v1"
runtime.REQUEST_ID = "planner_v11_probe_9_request_01"
runtime.REPORT_BASENAME = "V11_PLANNER_PROBE_9_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_9")
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "probe_5": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_5"),
    "probe_6": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_6"),
    "probe_7": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_7"),
    "probe_8": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_planner_probe_8"),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "probe_5": "14d6b65d34b48986fcd8a55a0ec32b2311ad9a87539c8b7197607f63619ee4d3",
    "probe_6": "9690e2dcb6cc0f9dd2afb074b9c413802daf0b265f6c13a5c8c70c84f3504ed0",
    "probe_7": "d2353a677915d6496edd878a61825bc6c34bdb6b43e6acbbc3d65e034ff685b8",
    "probe_8": "b101cdb4c82a927ccf8fa4f6cc7404b73f8abedc4c56f85c91d6201948d6d750",
}


if __name__ == "__main__":
    raise SystemExit(runtime.main())
