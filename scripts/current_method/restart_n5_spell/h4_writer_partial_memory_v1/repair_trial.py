"""Prepare a fixed, local-only old/new Writer comparison on the dated roots.

Both arms execute through runtime.execute and the existing shared generator.
No authorization or credential value is inherited from a historical run.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from ..backends import BackendResponse
from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt
from ..h4_writer_model_comparison_v1.prepare import _capacity
from ..h4_writer_model_comparison_v1.transport import build_payload, _reject_inline_credentials
from ..h4_r02_ordering_temporal_contract_v1.dependencies import forbid_network
from .prepare import PROJECT, ROOTS, code_closure, record, verify
from .runtime import read, write_rows
from .sources import render_dialogue

SOURCE = PROJECT / "restart-n5-runs/h4_dated_transcript_qwen14b_v1_prepare_1"
DEFAULT_RUN = PROJECT / "restart-n5-runs/h4_planner_writer_repair_v2_writer_prepare_1"
SCOPE = "writer_repair_comparison"
BUDGET = {"primary": 12, "repair_max": 12, "http_max": 24}
ARM_POLICIES = {
    "TEMPORAL_V1": {"ordering_policy": {"version": "ordering_temporal_contract_v1", "node_count": 4},
                    "admission_policy": "partial_v1"},
    "GROUNDED_V2": {"ordering_policy": {"version": "ordering_grounded_options_v2", "node_count": 4},
                    "admission_policy": "partial_grounded_v2"},
}


def expected_schedule():
    return [{"scope": SCOPE, "source_mode": "TRANSCRIPT_ONLY", "root_id": root,
             "slot_id": f"intent_{index+1}", "raw_index": index, "arm_id": arm,
             "order": ri * 6 + index * 2 + ai + 1}
            for ri, root in enumerate(ROOTS) for index in range(3)
            for ai, arm in enumerate(ARM_POLICIES)]


def validate_schedule(schedule):
    if schedule != expected_schedule():
        raise ValueError("WRITER_REPAIR_SCHEDULE_DRIFT")


def grouped_metrics(schedule, finals, raw_stages, costs, outcomes, *, logical_calls):
    """Account for every attempted call without treating program PASS as truth."""
    from .admission import duplicate_key
    result = []
    for root in ROOTS:
        for arm in ARM_POLICIES:
            def group(rows):
                return [x for x in rows if x.get("root_id") == root and x.get("arm_id") == arm]
            stages, accepted, calls = group(raw_stages), group(finals), group(costs)
            raw = []
            for stage in stages:
                payload = stage.get("raw")
                candidates = payload.get("candidates") if isinstance(payload, dict) else None
                if isinstance(candidates, list):
                    raw.extend(row for row in candidates if isinstance(row, dict)
                               and isinstance(row.get("question_type"), str))
            counts = Counter(row["question_type"] for row in raw)
            kept = Counter(row["question_type"] for row in accepted)
            unique = {kind: len({duplicate_key(row) for row in accepted if row["question_type"] == kind})
                      for kind in ("single_choice", "multi_select", "ordering")}
            usage = {}
            for field in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
                values = [c.get(field) for c in calls]
                unknown = sum(type(v) is not int for v in values)
                usage[field] = {"known_sum": sum(v for v in values if type(v) is int),
                                "unknown_calls": unknown,
                                "complete_total": "UNKNOWN" if unknown else sum(values)}
            slots = group(outcomes)
            decisions = Counter(x.get("status", "UNKNOWN") for x in slots)
            abstentions = sum("ordering" in (x.get("admission_summary") or {}).get("abstained_minimums", {}) for x in slots)
            attempts = group(logical_calls)
            result.append({"root_id": root, "arm_id": arm, "raw_yield": dict(counts),
                "program_accepted": dict(kept), "program_accepted_unique": unique,
                "logical_calls": len(attempts), "http_attempts": sum(c.get("http_attempts", 0) for c in calls),
                "primary_calls": sum(x.get("stage") == "primary" for x in attempts),
                "repair_calls": sum(x.get("stage") == "repair" for x in attempts),
                "unsent_primary_slots": len(group(schedule)) - sum(x.get("stage") == "primary" for x in attempts),
                "slot_statuses": dict(decisions), "legal_ordering_abstention_slots": abstentions,
                "usage": usage, "actual_returned_models": sorted({str(c.get("returned_model") or "UNKNOWN") for c in calls}),
                "content_diagnostic": "NOT_RUN", "content_pass_fail_undetermined": "NOT_ASSESSED",
                "event_state_actor_temporal_errors": "NOT_ASSESSED", "supported_unique_ordering": "UNKNOWN",
                "verified_money": "UNKNOWN", "cost_per_supported_question": "UNKNOWN",
                "qualification": "Program counts and raw usage only. Includes failed calls and all primary/repair costs. Semantic yield requires offline diagnosis."})
    return result


def prepare(run=DEFAULT_RUN, *, source=SOURCE):
    run, source = Path(run).resolve(), Path(source).resolve()
    if run.exists():
        raise ValueError("NEW_RUN_DIRECTORY_REQUIRED")
    frozen = read(source / "audit/frozen_manifest.json")
    originals = {str((source / x["path"]).resolve()): x for x in frozen["files"]}
    sources = {}

    def selected(path):
        path = Path(path).resolve()
        entry = record(path)
        if originals.get(str(path), {}).get("sha256") != entry["sha256"]:
            raise ValueError("SELECTED_FROZEN_SOURCE_DRIFT:" + str(path))
        sources[str(path)] = entry
        return path

    profile = deepcopy(read(selected(source / "profile.json")))
    _reject_inline_credentials(profile)
    profile.pop("user_service_exception", None)
    profile["status"] = "OFFLINE_PREPARED_WAITING_FOR_NEW_BUDGET_AUTHORIZATION"
    profile["arm_id"] = "SHARED_QWEN14B"
    atomic_json(run / "profile.json", profile)
    schedule = expected_schedule()
    identity, requests, budgets = [], [], []
    for root in ROOTS:
        original_folder = source / "inputs/TRANSCRIPT_ONLY" / root
        source_expanded = selected(original_folder / "expanded.json")
        source_contracts = selected(original_folder / "contracts.json")
        original = read(source_expanded)
        expanded = {**original, "dialogue_rendering": "conversation_dates_v1"}
        contracts = read(source_contracts)
        if expanded.get("root_id") != root or [x.get("intent_id") for x in contracts] != [f"intent_{i}" for i in range(1, 4)]:
            raise ValueError("FROZEN_ROOT_OR_INTENT_MISMATCH")
        dialogue = render_dialogue(expanded)
        folder = run / "inputs/TRANSCRIPT_ONLY" / root
        atomic_json(folder / "expanded.json", expanded)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_contracts, folder / "contracts.json")
        for row in (x for x in schedule if x["root_id"] == root):
            contract = contracts[row["raw_index"]]
            policy = ARM_POLICIES[row["arm_id"]]
            spec = build_prompt(expanded, contract, ordering_policy=policy["ordering_policy"])
            old_path = selected(source / "policy_snapshot/primary_requests/TRANSCRIPT_ONLY" / root / row["slot_id"] / "prompt.txt")
            old_prompt = old_path.read_text()
            if spec["prompt"].rsplit("DIALOGUE:\n", 1)[-1] != dialogue:
                raise ValueError("FULL_DATED_TRANSCRIPT_NOT_EXACT")
            if row["arm_id"] == "TEMPORAL_V1" and spec["prompt"] != old_prompt:
                raise ValueError("BASELINE_PROMPT_CHANGED")
            payload = build_payload(profile, spec["prompt"])
            path = run / "policy_snapshot/primary_requests" / row["arm_id"] / root / row["slot_id"]
            atomic_text(path / "prompt.txt", spec["prompt"])
            atomic_json(path / "request.json", {**row, **policy, "payload": payload})
            entry = {**row, **policy, "prompt_sha256": spec["prompt_sha256"],
                     "messages_sha256": sha256_bytes(canonical_json(payload["messages"]).encode()),
                     "turns_sha256": sha256_bytes(canonical_json(original["turns"]).encode()),
                     "dates_sha256": sha256_bytes(canonical_json(original["conversation_date_segments"]).encode()),
                     "contracts_sha256": record(source_contracts)["sha256"]}
            requests.append(entry)
            identity.append({**entry, "same_original_turn_objects": expanded["turns"] == original["turns"],
                "date_origin": "SYNTHETIC_ANNOTATION_NOT_RECOVERED_TIMESTAMP", "event_date_implied": False,
                "baseline_prompt_exact": spec["prompt"] == old_prompt,
                "input_metadata_change": "explicit shared conversation_dates_v1 renderer only",
                "comparison": "WRITER_REPAIR_COMBINATION; prompts intentionally differ by version"})
            capacity = _capacity(profile, payload["messages"], 8192, f'{row["arm_id"]}:{root}:{row["slot_id"]}')
            if capacity["status"] == "BLOCKED":
                raise ValueError("FULL_REQUEST_EXCEEDS_CONFIGURED_CAPACITY")
            budgets.append({**row, **capacity})
    # Freeze only the actually used tokenizer/provider evidence, never secrets.
    accounting = profile.get("token_accounting") or {}
    if accounting.get("evidence_path"):
        p = Path(accounting["evidence_path"]); sources[str(p)] = record(p)
    if accounting.get("local_tokenizer_path"):
        for p in Path(accounting["local_tokenizer_path"]).iterdir():
            if p.is_file(): sources[str(p)] = record(p)
    atomic_json(run / "schedule.json", schedule)
    atomic_json(run / "ordering_policy.json", {k: v["ordering_policy"] for k, v in ARM_POLICIES.items()})
    atomic_json(run / "admission_policy.json", {k: v["admission_policy"] for k, v in ARM_POLICIES.items()})
    atomic_json(run / "execution_contract.json", {"scope": SCOPE, "budget": BUDGET,
        "arm_policies": ARM_POLICIES, "real_execution_authorized": False,
        "per_primary_repair_max": 1, "http_attempts_per_logical_call": 1,
        "transport_retries": 0, "parameter_fallback": False, "automatic_resume": False,
        "technical_failure_policy": "STOP_ALL_UNSENT; NEVER_CONTENT_REPAIR",
        "seed_policy": profile["seed_policy"], "fresh_primary_both_arms": True,
        "model": profile["model"], "memory_calls": 0, "other_model_roles": [],
        "default_model_upgrade": False, "semantic_diagnostic": "NOT_RUN",
        "authorization_note": "Historical authorizations are not reused. Offline work only."})
    atomic_json(run / "audit/prompt_identity.json", identity)
    counts = [x["input_tokens"] for x in budgets]
    atomic_json(run / "audit/token_budget.json", {"budget": BUDGET, "requests": budgets,
        "primary_input_tokens_local_template": sum(counts) if all(type(x) is int for x in counts) else "UNKNOWN",
        "primary_output_ceiling": 12 * 8192, "all_output_ceiling": 24 * 8192,
        "repair_input_tokens": "DYNAMIC_OWN_PRIMARY_PLUS_FULL_FROZEN_INPUT; CHECK_BEFORE_SEND",
        "actual_usage": "UNKNOWN", "verified_cost": "UNKNOWN",
        "capacity_basis": "Existing local tokenizer and frozen provider configuration; no new service probe",
        "no_truncation": True, "actual_parameter_adoption": "UNKNOWN"})
    write_rows(run / "slot_call_manifest.jsonl", requests)
    atomic_text(run / "RUN_STATUS.md", "Offline preparation only. 12 proposed primary slots, at most 12 repairs / 24 HTTP attempts.\n"
        "Real model calls: 0. Content improvement: UNTESTED. Separate new budget authorization required.\n")
    files = sorted(p for p in run.rglob("*") if p.is_file())
    atomic_json(run / "audit/frozen_manifest.json", {"version": "writer_repair_comparison_v2",
        "files": [record(p, run) for p in files], "source_files": list(sources.values()),
        "code": [record(p) for p in code_closure()], "credentials_in_manifest": False})
    return {"status": "OFFLINE_PREPARED_WAITING_FOR_NEW_BUDGET_AUTHORIZATION", "budget": BUDGET,
            "primary_slots": len(schedule), "external_calls": 0, "identity": verify(run)}


def mock(run):
    """Exercise the exact bounded entry, using deliberately synthetic content."""
    from . import runtime
    from .mock import synthetic_candidates
    run = Path(run).resolve()
    verify(run)
    results = []
    for scenario in ("success", "abstention", "abstention_scms_repair", "bad_ordering", "invalid_json",
                     "timeout", "repair_timeout", "v2_abstention_repair_timeout", "nonterminal"):
        with tempfile.TemporaryDirectory(prefix="writer_repair_v2_") as tmp, forbid_network():
            target = Path(tmp) / run.name
            shutil.copytree(run, target, ignore=shutil.ignore_patterns("executions"))
            calls = []

            class Client:
                def __init__(self, row, profile, callback):
                    self.row, self.profile, self.callback = row, profile, callback
                    self.config = SimpleNamespace(model=profile["model"], base_url=profile["base_url"], temperature=0.1)

                def complete(self, *, prompt, **kwargs):
                    self.callback(build_payload(self.profile, prompt))
                    stage = "repair" if "REPAIR THE FOLLOWING BATCH" in prompt else "primary"
                    calls.append((self.row["arm_id"], stage))
                    if scenario == "timeout" or scenario in ("repair_timeout", "v2_abstention_repair_timeout") and stage == "repair":
                        raise TimeoutError("OFFLINE_SYNTHETIC_TIMEOUT")
                    expanded = read(target / "inputs/TRANSCRIPT_ONLY" / self.row["root_id"] / "expanded.json")
                    raw = synthetic_candidates(expanded, self.row["slot_id"], stage=stage)
                    grounded = self.row["arm_id"] == "GROUNDED_V2"
                    if grounded:
                        ordering = raw["candidates"][-1]
                        ordering["ordering_contract_version"] = "ordering_grounded_options_v2"
                        catalog = {t["turn_ref"]: t["text"] for t in expanded["turns"]}
                        for node in ordering["event_nodes"] + ordering["temporal_relations"]:
                            node["evidence_spans"] = [{"turn_ref": ref, "quote": catalog[ref]} for ref in node["evidence_refs"]]
                        for edge in ordering["temporal_relations"]:
                            edge["ordered_pair"] = [edge.pop("before"), edge.pop("after")]
                            edge.pop("temporal_basis")
                    if scenario in ("abstention", "abstention_scms_repair", "v2_abstention_repair_timeout") and grounded:
                        raw["candidates"] = raw["candidates"][:-1]
                        raw["abstentions"] = [{"question_type": "ordering", "status": "INSUFFICIENT_EVIDENCE",
                                                "reason": "OFFLINE_SYNTHETIC_SHORTAGE"}]
                        if scenario in ("abstention_scms_repair", "v2_abstention_repair_timeout") and stage == "primary":
                            raw["candidates"][0]["options"] = {}
                    if scenario == "bad_ordering" or scenario == "repair_timeout" and stage == "primary":
                        raw["candidates"][-1]["temporal_relations"] = []
                    text = "{" if scenario == "invalid_json" and stage == "primary" else json.dumps(raw)
                    return BackendResponse(text, "OFFLINE_SYNTHETIC", {"finish_reason": "length" if scenario == "nonterminal" else "stop", "usage": None})

            stopped = scenario in ("timeout", "repair_timeout", "v2_abstention_repair_timeout", "nonterminal")
            try:
                result = runtime.execute(target, SCOPE, authorized=True, allow_unknown_service=False, client_factory=Client)
                if stopped: raise AssertionError("EXPECTED_STOP")
            except runtime.ExecutionStopped:
                if not stopped: raise
                result = read(target / "executions" / SCOPE / "reports/summary.json")
            expected_calls = {"success": 12, "abstention": 12, "abstention_scms_repair": 18,
                              "bad_ordering": 24, "invalid_json": 24, "timeout": 1,
                              "repair_timeout": 2, "v2_abstention_repair_timeout": 3, "nonterminal": 1}[scenario]
            if len(calls) != expected_calls or result["http_attempts"] != expected_calls:
                raise AssertionError(f"MOCK_CALL_COUNT:{scenario}:{len(calls)}:{expected_calls}")
            expected_accepted = {"success": 48, "abstention": 42, "abstention_scms_repair": 42,
                                 "bad_ordering": 36, "invalid_json": 48, "timeout": 0,
                                 "repair_timeout": 3, "v2_abstention_repair_timeout": 6, "nonterminal": 0}[scenario]
            if result["program_accepted"] != expected_accepted:
                raise AssertionError(f"MOCK_ADMISSION:{scenario}:{result['program_accepted']}:{expected_accepted}")
            if scenario == "v2_abstention_repair_timeout":
                metrics = read(target / "executions" / SCOPE / "reports/root_arm_metrics.json")
                first = next(x for x in metrics if x["root_id"] == ROOTS[0] and x["arm_id"] == "GROUNDED_V2")
                if first["legal_ordering_abstention_slots"] != 1:
                    raise AssertionError("STOPPED_PRIMARY_ABSTENTION_LOST")
            results.append({"scenario": scenario, "status": "PASS", "synthetic_calls": len(calls),
                            "program_accepted": result["program_accepted"], "external_calls": 0})
    result = {"status": "PASS", "scenarios": results, "external_calls": 0,
              "qualification": "Synthetic structural/flow tests only; no evidence of model quality improvement."}
    atomic_json(run / "audit/mock_report.json", result)
    return result
