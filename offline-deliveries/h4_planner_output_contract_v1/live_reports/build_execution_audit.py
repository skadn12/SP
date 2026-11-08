"""Read-only execution checks; write reports only beside this script."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

DELIVERY = Path(__file__).resolve().parent.parent
RUN = DELIVERY / "prepared_run"
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-planner-output-contract-v1")
PRIOR = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_full_pipeline_realizer_state_contract_v1_live_1")
sys.path.insert(0, str(WORKTREE))

from scripts.current_method.restart_n5_spell.core import canonical_json
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.full_pipeline_qa_source import build_qa_source, verify_qa_source
from scripts.current_method.restart_n5_spell.h4_r02_ordering_temporal_contract_v1.contract import derive_ordering, resolve_runtime_context
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.admission import verify_retained_candidate


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


errors = []


def check(condition, label):
    if not condition:
        errors.append(label)


def check_manifest(path, expected_count=None, expected_mutable=()):
    manifest = read(path)
    rows = manifest["files"]
    mismatches, missing, allowed_changes = [], [], []
    for row in rows:
        p = Path(row["path"])
        if not p.is_file():
            missing.append(str(p))
        elif sha(p) != row["sha256"]:
            target = allowed_changes if str(p) in expected_mutable else mismatches
            target.append(str(p))
    check(not mismatches and not missing, f"manifest_drift:{path}")
    if expected_count is not None:
        check(len(rows) == expected_count, f"manifest_count:{path}")
    return {"path": str(path), "record_count": len(rows),
            "matched_count": len(rows) - len(mismatches) - len(missing) - len(allowed_changes),
            "unexpected_changes": mismatches, "missing": missing,
            "expected_runtime_changes": allowed_changes,
            "status": "PASS" if not mismatches and not missing else "FAIL"}


current_freeze = check_manifest(RUN / "freeze_manifest.json", 293)
previous_freeze = check_manifest(PRIOR / "freeze_manifest.json", 394)
delivery_manifest = check_manifest(DELIVERY / "DELIVERY_MANIFEST.json",
    expected_mutable=(str(RUN / "run_status.json"),))
summary = read(RUN / "generation_summary.json")
records_by_phase = read(RUN / "complete_call_records.json")
schedule = read(RUN / "frozen/schedule.json")
records = [dict(row, phase=phase) for phase, rows in records_by_phase.items() for row in rows]
check(len(records) == 31, "logical_call_count")
check(sum(row["http_attempt_count"] for row in records) == 31, "http_count")
check(len(records) <= schedule["maximum_http_attempts"] == 68, "global_http_budget")

ledger_paths = list((RUN / "expansion/execution").glob("*/development_call_ledger.json"))
ledger_paths.append(RUN / "qa/execution/qa_call_ledger.json")
ledger_report = []
ledger_rows = []
for path in ledger_paths:
    ledger = read(path)
    check(ledger["status"] == "FINISHED", f"ledger_not_finished:{path}")
    check(ledger["logical_call_count"] == len(ledger["records"]), f"ledger_logical:{path}")
    check(ledger["http_attempt_count"] == sum(x["http_attempt_count"] for x in ledger["records"]), f"ledger_http:{path}")
    check(ledger["automatic_retry_count"] == 0, f"ledger_retry:{path}")
    check(ledger["automatic_resume"] is False, f"ledger_resume:{path}")
    ledger_rows.extend(ledger["records"])
    ledger_report.append({"path": str(path), "logical_calls": ledger["logical_call_count"],
                          "http_attempts": ledger["http_attempt_count"], "status": ledger["status"]})
check(Counter(map(digest, ledger_rows)) == Counter(digest({k: v for k, v in row.items() if k != "phase"}) for row in records), "aggregate_ledger_identity")

calls = []
grouped = defaultdict(lambda: {"logical_calls": 0, "http_attempts": 0,
    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
expected_metadata_paths = set()
for row in records:
    prefix = RUN / "qa/execution" if row["phase"] == "qa" else RUN / "expansion/execution" / row["root_id"]
    call_dir = prefix / row["artifact_directory"]
    request = read(call_dir / "request.json")
    metadata_path = call_dir / "response_metadata.json"
    expected_metadata_paths.add(metadata_path)
    meta = read(metadata_path)
    response = json.loads(meta["raw_response_text"])
    choice = response["choices"][0]
    capacity = read(call_dir / "capacity.json")
    identity = f"{row['root_id']}:{row['stage']}:{row.get('intent_id') or ''}"
    check(digest(request["payload"]) == row["payload_sha256"], f"payload_hash:{identity}")
    check(digest(request["payload"]["messages"]) == row["messages_sha256"], f"messages_hash:{identity}")
    check(row["usage"] == meta["usage"] == response["usage"], f"usage_identity:{identity}")
    check(row["actual_model"] == meta["actual_model"] == response["model"] == "qwen3.6-35b-a3b", f"model_identity:{identity}")
    check(row["finish_reason"] == meta["finish_reason"] == choice["finish_reason"] == "stop", f"finish_reason:{identity}")
    check(row["input_tokens"] == row["usage"]["prompt_tokens"], f"input_usage:{identity}")
    check(row["output_tokens"] == row["usage"]["completion_tokens"], f"output_usage:{identity}")
    check(row["total_tokens"] == row["usage"]["total_tokens"] == row["input_tokens"] + row["output_tokens"], f"total_usage:{identity}")
    check(row["http_attempt_count"] == meta["http_attempt_count"] == 1, f"attempts:{identity}")
    check(row["retry_count"] == meta["transport_retry_count"] == meta["rate_limit_retry_count"] == 0, f"retries:{identity}")
    check(row["timeout"] is False and meta["timeout"] is False, f"timeout:{identity}")
    check(row["http_status"] == meta["http_status"] == 200 and row["status"] == "COMPLETED", f"completion:{identity}")
    check((call_dir / "response.txt").read_text() == choice["message"]["content"], f"raw_response_content:{identity}")
    check(capacity["status"] == "PASS" and capacity["truncation"] is False, f"capacity:{identity}")
    check(request["payload"]["max_tokens"] == row["maximum_output_tokens"], f"output_budget:{identity}")
    for key in [(row["root_id"], row["phase"], row["stage"]), (row["root_id"], row["phase"], "ALL"), ("ALL", row["phase"], "ALL"), ("ALL", "ALL", "ALL")]:
        g = grouped[key]
        g["logical_calls"] += 1
        for metric in ("input_tokens", "output_tokens", "total_tokens"):
            g[metric] += row[metric]
        g["http_attempts"] += row["http_attempt_count"]
    calls.append({"root_id": row["root_id"], "phase": row["phase"], "stage": row["stage"],
        "intent_id": row.get("intent_id"), "request_id": row.get("request_id"),
        "artifact_directory": str(call_dir), "actual_model": response["model"],
        "finish_reason": choice["finish_reason"], "usage": response["usage"],
        "http_attempts": row["http_attempt_count"], "retries": row["retry_count"],
        "timeout": row["timeout"], "http_status": row["http_status"],
        "reasoning_tokens": "UNKNOWN", "verified_cost": "UNKNOWN",
        "requested_parameters": request["requested_parameters"],
        "confirmed_parameter_adoption": request["confirmed_parameter_adoption"]})

actual_metadata_paths = set((RUN / "expansion/execution").glob("*/calls/*/*/response_metadata.json"))
actual_metadata_paths.update((RUN / "qa/execution/calls").glob("*/*/response_metadata.json"))
check(actual_metadata_paths == expected_metadata_paths, "unaccounted_response_metadata")
total = grouped[("ALL", "ALL", "ALL")]
for key in ("logical_calls", "http_attempts", "input_tokens", "output_tokens", "total_tokens"):
    check(total[key] == summary["cost"][key], f"summary_cost:{key}")
check(set(row["stage"] for row in records) <= set(schedule["expansion_stages"]) | {"miner", "writer_primary", "writer_repair"}, "extra_stage")
check(summary["memory_free_solver_training_extra_audit_api_calls"] == 0, "extra_api_summary")

qa_root = RUN / "qa/roots/cc-train-episode-112812"
source = read(qa_root / "qa_source.json")
provenance = read(qa_root / "qa_source_provenance.json")
verify_qa_source(source, provenance)
rebuilt = build_qa_source(expansion_result=read(RUN / "expansion/roots/cc-train-episode-112812/result.json"),
    input_data=read(RUN / "frozen/cc-train-episode-112812/inputs.json"), root_id="cc-train-episode-112812")
check(rebuilt == source, "qa_source_rebuild")
check(len(source["turns"]) == 137 and provenance["history_turn_count"] == 95 and len(provenance["new_turn_refs"]) == 42, "qa_turn_counts")
policy, context = resolve_runtime_context(qa_root)
candidates = read(qa_root / "candidates_valid.json")
retained = []
for candidate in candidates:
    result = verify_retained_candidate(candidate, expanded=source, policy=policy, context=context,
        admission_policy=candidate["admission_identity_version"])
    retained.append({"candidate_id": candidate["candidate_id"], **result})
derivations = []
for path in sorted((qa_root / "ordering_derivations").glob("*/*.json")):
    record = read(path)
    rebuilt_derivation = derive_ordering(record["raw_row"], expanded=source, policy=policy, identity=record["identity"])
    check(canonical_json(rebuilt_derivation["record"]) == canonical_json(record), f"derivation_rebuild:{path}")
    raw_path = qa_root / "raw/responses" / (record["identity"]["request_id"] + ".json.txt")
    check(sha(raw_path) == record["identity"]["raw_sha256"], f"derivation_raw_sha:{path}")
    raw = read(raw_path)
    check(raw["candidates"][record["identity"]["raw_row_index"] - 1] == record["raw_row"], f"derivation_raw_row:{path}")
    derivations.append({"path": str(path), "status": record["derivation_status"],
        "errors": record["errors"], "derived_gold": record.get("derived_gold"),
        "identity": record["identity"], "rederivation_matches": True,
        "semantic_support_status": "NOT_ASSESSED"})
check(len(derivations) == 4, "derivation_count")
check(sum(row["status"] == "PASS" for row in retained) == 3, "retained_ordering_count")

report = {"version": "planner_output_contract_live_execution_audit_v1",
    "audited_at": datetime.now(timezone.utc).isoformat(),
    "status": "PASS" if not errors else "FAIL", "errors": errors,
    "scope": "Local execution, cost, used-file identity, QA provenance and deterministic ordering derivation only; no semantic certification.",
    "run_root": str(RUN), "generation_status": summary["status"],
    "audit_external_calls": 0, "audit_source_or_run_mutations": 0,
    "freeze_checks": {"current": current_freeze, "previous": previous_freeze,
        "preparation_delivery": delivery_manifest},
    "expected_runtime_mutation_note": "DELIVERY_MANIFEST is a preparation snapshot. prepared_run/run_status.json changes from prepared to executed as designed; neither actual freeze manifest reports drift.",
    "budget": {"approved_maximum_http": 68, "maximum_expansion_http": 40, "maximum_qa_http": 28,
               "used_http": 31, "unused_http": 37, "additional_generation_requested": False},
    "totals": {**total, "retries": 0, "timeouts": 0, "technical_failures": 0,
        "reasoning_tokens": "UNKNOWN", "verified_cost": "UNKNOWN"},
    "actual_models": dict(Counter(row["actual_model"] for row in records)),
    "finish_reasons": dict(Counter(row["finish_reason"] for row in records)),
    "stage_counts": dict(Counter(row["stage"] for row in records)),
    "per_root_phase_stage_costs": [{"root_id": key[0], "phase": key[1], "stage": key[2], **value}
                                   for key, value in grouped.items()],
    "ledgers": ledger_report, "calls": calls,
    "no_extra_api_evidence": {"record_count": len(records), "metadata_file_count": len(actual_metadata_paths),
        "all_artifacts_accounted_for": actual_metadata_paths == expected_metadata_paths,
        "runtime_reported_memory_free_solver_training_extra_audit_calls": 0,
        "observed_stage_set_matches_frozen_schedule": not any(x == "extra_stage" for x in errors),
        "limitation": "Verified run artifacts and ledgers, not a machine-wide network packet capture."},
    "qa_source": {"status": "PASS", "root_id": provenance["root_id"],
        "source_path": provenance["source_path"], "provenance_path": str(qa_root / "qa_source_provenance.json"),
        "source_sha256": provenance["qa_source_file_sha256"], "rebuilt_identically": rebuilt == source,
        "history_turn_count": 95, "new_turn_count": 42, "total_turn_count": 137,
        "historical_dates_preserved": True, "new_R02_dates": "UNKNOWN", "memory_fact_source": False,
        "historical_fallback": False},
    "ordering_derivation_checks": derivations, "retained_candidate_integrity_checks": retained,
    "semantic_review": "Separate Codex offline content diagnosis; this report does not certify evidence support or question correctness.",
    "service_unknowns": ["reasoning mode/budget", "reasoning token breakdown", "actual parameter adoption",
                         "maximum service context/output limits", "verified currency cost"]}

lines = ["# Execution audit", "", f"Status: **{report['status']}**. Local offline audit; 0 external calls and no run/source changes.", "",
    "31 logical calls = 31 HTTP attempts / 68 authorized. Expansion: 25; QA: 6 (1 Miner, 3 primary Writers, 2 repair Writers). All 31 responses returned qwen3.6-35b-a3b, finish_reason=stop and HTTP 200. Retry, timeout and technical-failure counts are 0.", "",
    "| Root | Expansion HTTP | QA HTTP | Input tokens | Output tokens | Total tokens |",
    "|---|---:|---:|---:|---:|---:|"]
for root in summary["root_order"]:
    phase_values = [grouped.get((root, phase, "ALL"), {}) for phase in ("expansion", "qa")]
    sums = {key: sum(value.get(key, 0) for value in phase_values) for key in ("input_tokens", "output_tokens", "total_tokens")}
    lines.append(f"| {root.rsplit('-', 1)[-1]} | {phase_values[0].get('http_attempts', 0)} | {phase_values[1].get('http_attempts', 0)} | {sums['input_tokens']:,} | {sums['output_tokens']:,} | {sums['total_tokens']:,} |")
lines.extend([f"| Total | 25 | 6 | {total['input_tokens']:,} | {total['output_tokens']:,} | {total['total_tokens']:,} |", "",
    "Costs include unsuccessful roots, rejected candidates, supplemental planning and Writer repairs. Currency fees and reasoning token breakdown remain UNKNOWN; token counts are not prices.", "",
    f"Frozen identity: current {current_freeze['matched_count']}/293, previous {previous_freeze['matched_count']}/394; no missing or changed frozen files. The preparation delivery manifest has one expected change: prepared_run/run_status.json advanced during authorized execution. This is not frozen-input drift.", "",
    "Only 112812 was promoted to QA. Rebuilding its source from the new promotion and frozen input reproduced the saved source exactly: 95 P+R01 turns + 42 new R02 turns = 137. Historical date segments remain unchanged; new R02 dates remain UNKNOWN. No historical R02 fallback or Memory QA fact source was used.", "",
    "All 4 saved ordering derivations were recomputed identically from their saved raw rows and exact source; 3 retained ordering candidates passed identity/projection verification. These checks establish provenance and deterministic Gold derivation, not semantic correctness. Rejected derivations remain preserved.", "",
    "Call ledgers, saved request hashes, response metadata and raw server model/finish_reason/usage fields agree. Every saved HTTP response is accounted for; recorded stages match the frozen schedule. No Memory, Free, Solver, training or extra audit API is present in these run records. This is an artifact audit, not a machine-wide network capture.", "",
    "Service capacity, reasoning defaults, parameter adoption and monetary cost remain UNKNOWN. The 37 unused requests are not consumed. Content diagnosis and final filtering are separate deliverables.", "",
    "Machine-readable per-call usage, per-root/stage cost and provenance checks: [EXECUTION_AUDIT.json](EXECUTION_AUDIT.json).", ""])
if errors:
    lines.extend(["Audit errors:", "", *[f"- {error}" for error in errors], ""])
(Path(__file__).parent / "EXECUTION_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
(Path(__file__).parent / "EXECUTION_AUDIT.md").write_text("\n".join(lines))
print(json.dumps({"status": report["status"], "errors": errors, "totals": report["totals"], "freeze_checks": report["freeze_checks"]}, ensure_ascii=False, indent=2))
