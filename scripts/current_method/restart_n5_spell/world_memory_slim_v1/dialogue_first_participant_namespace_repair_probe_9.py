"""One-request live probe for the frozen Planner namespace repair contract."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..core import sha256_bytes
from . import dialogue_first_participant_planner_probe_8 as previous
from .planner_namespace_repair_v11 import (
    PLANNER_NAMESPACE_REPAIR_PROMPT_VERSION_V11,
    apply_planner_namespace_repairs_v11,
    build_planner_namespace_only_retry_prompt_v11,
    compile_planner_namespace_repair_spec_v11,
    planner_namespace_only_retry_eligible,
)


runtime = previous.runtime
_base_build = runtime.build_planner_probe_input
PROBE_8 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
    "planner_probe_8"
)


def _build_namespace_repair_probe_input(input_path: Path) -> dict[str, Any]:
    values = _base_build(input_path)
    outline = json.loads((PROBE_8 / "planner_response_parsed.json").read_text())
    authoritative_refs = {row["entity_ref"] for row in values["role_table"]}
    validation = runtime.validate_planner_outline_v11(
        outline,
        source_anchor_registry=values["source_anchor_registry"],
        placement_choice_registry=values["placement_choice_registry"],
        authoritative_entity_refs=authoritative_refs,
        authoritative_entities=values["planning_state"]["entities"],
        action_availability_view=values["action_availability_view"])
    if validation.get("valid") is True or not planner_namespace_only_retry_eligible(
            outline=outline, errors=validation.get("errors") or []):
        raise ValueError("probe_8_not_namespace_repair_eligible")
    spec = compile_planner_namespace_repair_spec_v11(
        outline=outline, errors=validation["errors"], validation=validation,
        action_contracts=values["shared_narrative_action_contracts"],
        action_availability_view=values["action_availability_view"])
    prompt = build_planner_namespace_only_retry_prompt_v11(spec)
    values.update({
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "frozen_planner_input_sha256": spec["repair_spec_sha256"],
        "namespace_repair_frozen_outline": outline,
        "namespace_repair_spec": spec,
        "namespace_repair_source_validation": validation,
        "extra_probe_artifacts": {
            "frozen_source_outline.json": outline,
            "namespace_repair_spec.json": spec,
            "source_outline_validation.json": validation,
        },
    })
    return values


def _prepare_namespace_repair_candidate(
    parsed: Mapping[str, Any], values: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, dict[str, Any], list[str]]:
    result = apply_planner_namespace_repairs_v11(
        frozen_outline=values["namespace_repair_frozen_outline"],
        response=parsed, spec=values["namespace_repair_spec"])
    audit = {key: deepcopy(value) for key, value in result.items()
             if key != "reconstructed_outline"}
    return result.get("reconstructed_outline"), audit, list(result["errors"])


def _report(summary: Mapping[str, Any]) -> str:
    usage = summary["api_usage"]
    return f"""# V11 Planner Namespace-Only Repair Probe 9

- Final choice: `{summary['final_choice']}`
- Transport completed: `{str(summary['transport_completed']).lower()}`
- JSON patch parsed: `{str(summary['json_parsed']).lower()}`
- Patch reconstruction valid: `{str(summary['response_transform_valid']).lower()}`
- Reconstructed Planner schema valid: `{str(summary['planner_schema_valid']).lower()}`
- Compile / DAG / structural gate: `{str(summary['compile_valid']).lower()} / {str(summary['dag_valid']).lower()} / {str(summary['structural_gate_valid']).lower()}`
- Planner logical / physical requests: `{usage['roles']['planner']['logical_requests']} / {usage['roles']['planner']['physical_requests']}`
- Automatic retries: `0`
- Realizer / Round Validator / Evidence Committer: `0 / 0 / 0`
- Prompt / completion / total tokens: `{usage['prompt_tokens']} / {usage['completion_tokens']} / {usage['total_tokens']}`
- Latency seconds: `{summary['latency_seconds']}`
- Historical integrity: `{str(summary['post_run_integrity']['valid']).lower()}`

This standalone probe made exactly one Planner namespace-repair request. The
Probe 8 outline was frozen, the raw patch was preserved, and local code created
a separate reconstructed outline before full validation. No downstream role or
complete canary was started.
"""


runtime.PROBE_VERSION = "v11_planner_namespace_repair_probe_9_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Planner Namespace-Only Repair Probe 9"
runtime.PROBE_RUN_SUFFIX = "v11_planner_namespace_repair_probe_9"
runtime.PREFLIGHT_VERSION = "v11_planner_namespace_repair_probe_9_preflight_v1"
runtime.REQUEST_ID = "planner_v11_namespace_repair_probe_09"
runtime.REQUEST_PROMPT_VERSION = PLANNER_NAMESPACE_REPAIR_PROMPT_VERSION_V11
runtime.REPORT_BASENAME = "V11_PLANNER_NAMESPACE_REPAIR_PROBE_9_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_namespace_repair_probe_9"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "dialogue_first_participant_probe_8": PROBE_8,
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "dialogue_first_participant_probe_8": (
        "ff03306f5337e804e3dc87ffaa82d3d50fce6d45a22468701cc64f9fd73d8ee6"
    ),
}
runtime.build_planner_probe_input = _build_namespace_repair_probe_input
runtime.prepare_candidate_response = _prepare_namespace_repair_candidate
runtime._report = _report


if __name__ == "__main__":
    raise SystemExit(runtime.main())
