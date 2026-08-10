"""One-request live verification of the complete namespace-repair patch contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import dialogue_first_participant_namespace_repair_probe_9 as previous
from .planner_namespace_repair_v11 import (
    PLANNER_NAMESPACE_REPAIR_PROMPT_VERSION_V11,
)


runtime = previous.runtime
PROBE_9 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_namespace_repair_probe_9"
)


def _report(summary: Mapping[str, Any]) -> str:
    usage = summary["api_usage"]
    return f"""# V11 Planner Namespace-Repair Probe 10

- Final choice: `{summary['final_choice']}`
- Transport completed: `{str(summary['transport_completed']).lower()}`
- Finish reason: `{summary['finish_reason']}`
- JSON patch parsed: `{str(summary['json_parsed']).lower()}`
- Complete patch/reconstruction valid: `{str(summary['response_transform_valid']).lower()}`
- Reconstructed Planner schema valid: `{str(summary['planner_schema_valid']).lower()}`
- Compile / DAG / structural gate: `{str(summary['compile_valid']).lower()} / {str(summary['dag_valid']).lower()} / {str(summary['structural_gate_valid']).lower()}`
- Planner logical / physical requests: `{usage['roles']['planner']['logical_requests']} / {usage['roles']['planner']['physical_requests']}`
- Automatic retries: `0`
- Realizer / Round Validator / Evidence Committer: `0 / 0 / 0`
- Prompt / completion / total tokens: `{usage['prompt_tokens']} / {usage['completion_tokens']} / {usage['total_tokens']}`
- Latency seconds: `{summary['latency_seconds']}`
- Historical integrity: `{str(summary['post_run_integrity']['valid']).lower()}`

This probe made exactly one Planner namespace-repair request. The Probe 8
outline and repair spec were frozen before the request. The returned patch was
preserved verbatim and was either rejected at the complete-replacement boundary
or reconstructed and passed through the full local Planner chain. No downstream
role or complete canary was started.
"""


runtime.PROBE_VERSION = "v11_planner_namespace_repair_probe_10_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Planner Namespace-Repair Probe 10"
runtime.PROBE_RUN_SUFFIX = "v11_planner_namespace_repair_probe_10"
runtime.PREFLIGHT_VERSION = "v11_planner_namespace_repair_probe_10_preflight_v1"
runtime.REQUEST_ID = "planner_v11_namespace_repair_probe_10"
runtime.REQUEST_PROMPT_VERSION = PLANNER_NAMESPACE_REPAIR_PROMPT_VERSION_V11
runtime.REPORT_BASENAME = "V11_PLANNER_NAMESPACE_REPAIR_PROBE_10_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_namespace_repair_probe_10"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "planner_namespace_repair_probe_9": PROBE_9,
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "planner_namespace_repair_probe_9": (
        "9a05506a115d834748e0e777a87c13086e17cb633fe17db2dcbf138f5015e447"
    ),
}
runtime._report = _report


if __name__ == "__main__":
    raise SystemExit(runtime.main())
