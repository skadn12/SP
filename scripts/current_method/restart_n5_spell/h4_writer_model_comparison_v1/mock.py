"""Exercise all actual frozen inputs using synthetic responses in temporary directories."""
from __future__ import annotations
import copy
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ..backends import BackendResponse
from ..core import atomic_json, sha256_file
from ..h4_r02_ordering_temporal_contract_v1.dependencies import forbid_network
from ..h4_r02_ordering_temporal_contract_v1.test_runtime import synthetic_candidates
from . import runtime
from .prepare import verify
from .transport import build_payload


def exercise(run):
    run = Path(run)
    verify(run)
    outcomes = []
    with forbid_network():
        for scenario in ("success", "json_repair", "graph_reject", "timeout_stop"):
            with tempfile.TemporaryDirectory(prefix="h4_writer_mock_") as folder:
                target = Path(folder)
                for name in ("inputs", "policy_snapshot"):
                    shutil.copytree(run / name, target / name)
                for name in ("schedule.json", "execution_contract.json"):
                    shutil.copyfile(run / name, target / name)
                profiles = copy.deepcopy(runtime.read(run / "profiles.json"))
                for arm, profile in profiles.items():
                    # These settings describe mock fixtures only, never claim service support.
                    profile.update(model="MOCK:" + arm, base_url="https://offline.invalid/v1",
                        endpoint="https://offline.invalid/v1/chat/completions", output_limit=8192,
                        request_parameters={"temperature": 0.1, "max_tokens": 8192}, unresolved=[])
                    for path in (target / "policy_snapshot/primary_requests" / arm).glob("*/*/request.json"):
                        value = runtime.read(path)
                        value["payload"] = build_payload(profile, path.with_name("prompt.txt").read_text())
                        atomic_json(path, value)
                atomic_json(target / "profiles.json", profiles)
                atomic_json(target / "audit/frozen_manifest.json", {"fixture": "MOCK_TEMPORARY_ONLY"})
                calls = []

                class Client:
                    def __init__(self, row, profile):
                        self.row = row
                        self.config = SimpleNamespace(model=profile["model"], base_url=profile["base_url"])

                    def complete(self, *, prompt, **kwargs):
                        calls.append({**self.row, "prompt": prompt})
                        if scenario == "timeout_stop":
                            raise TimeoutError("synthetic timeout")
                        raw = synthetic_candidates(prompt)
                        if scenario == "json_repair" and "REPAIR THE FOLLOWING BATCH" not in prompt:
                            raw = "{invalid JSON"
                        if scenario == "graph_reject":
                            raw["candidates"][-1]["temporal_relations"] = []
                        return BackendResponse(raw if isinstance(raw, str) else json.dumps(raw),
                                               self.config.model, {"usage": None})

                try:
                    result = runtime.execute_schedule(target,
                        client_factory=lambda row, profile, callback: Client(row, profile),
                        verifier=lambda: None, capacity=lambda *args: None)
                except runtime.ExecutionStopped:
                    if scenario != "timeout_stop":
                        raise
                    result = runtime.read(target / "reports/runtime_summary.json")
                expected = {"success": 12, "json_repair": 24, "graph_reject": 24, "timeout_stop": 1}[scenario]
                if len(calls) != expected:
                    raise AssertionError("MOCK_CALL_COUNT_MISMATCH:" + scenario)
                for call in calls:
                    if "REPAIR THE FOLLOWING BATCH" not in call["prompt"]:
                        path = run / "policy_snapshot/primary_requests" / call["arm_id"] / call["root_id"] / call["slot_id"] / "prompt.txt"
                        if call["prompt"].encode() != path.read_bytes():
                            raise AssertionError("MOCK_ACTUAL_SOURCE_PRIMARY_MISMATCH")
                outcomes.append({"scenario": scenario, "status": "PASS", "mock_calls": len(calls),
                    "result": result, "external_calls": 0, "capacity_verification": "NOT_A_SERVICE_CAPACITY_TEST"})
    verify(run)
    result = {"status": "PASS", "scenarios": outcomes, "external_calls": 0,
              "code": runtime.read(run / "audit/frozen_manifest.json")["code"],
              "profiles_sha256": sha256_file(run / "profiles.json"),
              "all_twelve_real_primary_slots_exercised": True,
              "qualification": "Synthetic responses validate plumbing, not model quality or service compatibility."}
    atomic_json(run / "audit/mock_report.json", result)
    return result
