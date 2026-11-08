"""Qwen-specific entry snapshot; Writer, ledger and admission stay shared."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import stat

from writer_trial import runtime as shared_runtime, verify

BUDGETS = shared_runtime.BUDGETS
POLICY = shared_runtime.POLICY
ExecutionStopped = shared_runtime.ExecutionStopped
SlotLedger = shared_runtime.SlotLedger
ProfileClient = shared_runtime.ProfileClient
generation = shared_runtime.generation
normalize_stage = shared_runtime.normalize_stage
admit_stage = shared_runtime.admit_stage
read = shared_runtime.read
rows = shared_runtime.rows
write_rows = shared_runtime.write_rows
atomic_json = shared_runtime.atomic_json


def _qwen_credential(profile):
    """Use the existing environment/config reference only at live execution."""
    if (profile.get("api_key_env") != "H4_QWEN_API_KEY"
            or profile.get("credential_format") != "challenger_json"
            or profile.get("credential_file") != "/root/autodl-tmp/gxy-selfplay/config.json"):
        raise ValueError("QWEN_CREDENTIAL_REFERENCE_DRIFT")
    credential = os.environ.get("H4_QWEN_API_KEY")
    if not credential:
        descriptor = None
        try:
            descriptor = os.open(profile["credential_file"], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            info = os.fstat(descriptor)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o022):
                raise ValueError("unsafe credential source")
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = None
                credential = json.load(handle)["api"]["api_key"]
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("QWEN_RUNTIME_CREDENTIAL_SOURCE_INVALID") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
    if not isinstance(credential, str) or not credential or any(char.isspace() for char in credential):
        raise ValueError("QWEN_RUNTIME_CREDENTIAL_INVALID")
    return credential


def execute(run, scope, *, authorized=False, allow_unknown_service=False, client_factory=None):
    if not authorized:
        raise RuntimeError("EXPLICIT_SCOPE_AND_BUDGET_AUTHORIZATION_REQUIRED")
    if scope not in BUDGETS:
        raise ValueError("UNKNOWN_SCOPE")
    run = Path(run).resolve()
    verify(run)
    profile = read(run / "profile.json")
    if (profile.get("provider") != "siliconflow"
            or profile.get("model") != "Qwen/Qwen3-14B"
            or profile.get("base_url") != "https://api.siliconflow.cn/v1"
            or profile.get("endpoint") != "https://api.siliconflow.cn/v1/chat/completions"
            or profile.get("request_parameters") != {
                "temperature": 0.1, "max_tokens": 8192, "enable_thinking": False,
                "response_format": {"type": "json_object"}}):
        raise ValueError("FIXED_QWEN_WRITER_PROFILE_REQUIRED")
    if profile.get("user_service_exception"):
        raise ValueError("OLD_EXPERIMENT_EXCEPTION_CANNOT_AUTHORIZE_NEW_RUN")
    schedule = [row for row in read(run / "schedule.json") if row["scope"] == scope]
    if len(schedule) != BUDGETS[scope]["primary"]:
        raise ValueError("FROZEN_PRIMARY_COUNT_MISMATCH")
    if not allow_unknown_service and profile.get("unresolved"):
        raise RuntimeError("UNVERIFIED_SERVICE_CONFIGURATION_REQUIRES_EXPLICIT_EXCEPTION")
    credential = _qwen_credential(profile) if client_factory is None else None
    work = run / "executions" / scope
    work.mkdir(parents=True, exist_ok=True)
    with (work / "execution.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (work / "audit/execution_started.json").exists():
            raise ExecutionStopped("STARTED_SCOPE_CANNOT_AUTOMATICALLY_RESUME_OR_RESEND")
        atomic_json(work / "audit/execution_started.json", {"scope": scope,
            "budget": BUDGETS[scope], "allow_unknown_service": allow_unknown_service,
            "backend": "MOCK" if client_factory is not None else "LIVE",
            "reasoning": {"requested_enable_thinking": False, "service_adoption": "UNKNOWN"},
            "automatic_retry": False})
        pools, slot_outcomes, failure = {}, [], None
        for row in schedule:
            folder = run / "inputs" / row["source_mode"] / row["root_id"]
            expanded = read(folder / "expanded.json")
            contract = read(folder / "contracts.json")[row["raw_index"]]
            root = work / "arms" / row["arm_id"] / "roots" / row["root_id"]
            context = {"run_id": run.name + ":" + scope, "arm_id": row["arm_id"]}
            ledger = SlotLedger(run, work, row, profile, root, expanded, contract, allow_unknown_service)
            client = (client_factory(row, profile, ledger.before_send) if client_factory is not None else
                      ProfileClient(profile, credential=credential, before_send=ledger.before_send))
            key = (row["arm_id"], row["root_id"])
            pool = pools.setdefault(key, {"candidates": [], "records": [], "expanded": expanded,
                                         "root": root, "context": context})
            try:
                result = generation.execute_writer_batch(expanded=expanded, contract=contract,
                    root_dir=root, root_id=row["root_id"], ledger=ledger, client=client,
                    after_call=ledger.after_call, ordering_policy=POLICY, ordering_context=context,
                    writer_max_tokens=8192, writer_temperature=0.1, admission_policy="partial_v1")
                pool["candidates"].extend(result["candidates"])
                pool["records"].extend(result["generation_records"])
                slot_outcomes.append({**row, "status": "COMPLETED", "candidate_count": len(result["candidates"]),
                    "repair_requested": result["repair_requested"], "admission_summary": result["admission_summary"]})
            except BaseException as error:
                failure = type(error).__name__ + ":" + str(error) if isinstance(error, ExecutionStopped) else type(error).__name__
                # Preserve an already completed primary even if its repair failed technically.
                retained = []
                for stage, parsed in ledger.parsed.items():
                    ordering_context = {**context, "root_id": row["root_id"], "root_dir": root,
                        "stage": stage, "request_id": parsed["request_id"], "raw_sha256": parsed["raw_text_sha256"]}
                    normalized = normalize_stage(parsed["raw"], intent_id=row["slot_id"], source="contract_guided",
                        contract=contract, expanded=expanded, root_id=row["root_id"], stage=stage,
                        request_id=parsed["request_id"], parse_error=parsed["parse_error"],
                        ordering_policy=POLICY, ordering_context=ordering_context)
                    retained, acceptance = admit_stage(normalized, retained=retained)
                    pool["records"].append({"request_id": parsed["request_id"], "stage": stage,
                                             "validation": normalized, "minimum_acceptance": acceptance})
                pool["candidates"].extend(retained)
                slot_outcomes.append({**row, "status": "STOPPED", "reason": failure,
                                      "candidate_count": len(retained)})
                break
        finals, outcomes = [], []
        for (arm, root_id), pool in pools.items():
            if not pool["candidates"] and not pool["records"]:
                continue
            outcome = generation.finalize_writer_candidates(root_dir=pool["root"], root_id=root_id,
                candidates=pool["candidates"], generation_records=pool["records"], expanded=pool["expanded"],
                ordering_policy=POLICY, ordering_context=pool["context"], admission_policy="partial_v1")
            outcomes.append({"arm_id": arm, **outcome})
            finals.extend({"arm_id": arm, **candidate} for candidate in read(pool["root"] / "candidates_valid.json"))
        all_calls = [read(path) for path in sorted((work / "arms").glob("*/roots/*/transport_outcomes/*.json"))]
        raw_stages = []
        for pool in pools.values():
            for path in sorted((pool["root"] / "stage_calls").glob("*/*.json")):
                raw_stages.append(read(path))
        counts = Counter(row["stage"] for row in rows(work / "audit/logical_calls.jsonl"))
        summary = {"status": "STOPPED" if failure else "COMPLETED", "scope": scope,
            "primary_calls": counts["primary"], "repair_calls": counts["repair"],
            "logical_calls": sum(counts.values()), "http_attempts": len(rows(work / "audit/http_events.jsonl")),
            "unattempted_primary_slots": len(schedule)-counts["primary"], "technical_failure": failure,
            "program_accepted": len(finals), "semantic_diagnostic": "NOT_RUN",
            "memory_support": "NOT_ASSESSED" if scope == "memory_only" else "NOT_APPLICABLE",
            "transcript_support": "NOT_ASSESSED",
            "actual_money": "UNKNOWN", "root_outcomes": outcomes,
            "automatic_retries": 0, "external_calls": 0 if client_factory is not None else len(rows(work / "audit/http_events.jsonl"))}
        write_rows(work / "reports/final_candidates.jsonl", finals)
        write_rows(work / "reports/raw_stage_candidates.jsonl", raw_stages)
        write_rows(work / "reports/all_call_costs.jsonl", all_calls)
        atomic_json(work / "reports/slot_outcomes.json", slot_outcomes)
        atomic_json(work / "reports/summary.json", summary)
        if failure:
            raise ExecutionStopped(failure)
        return summary
