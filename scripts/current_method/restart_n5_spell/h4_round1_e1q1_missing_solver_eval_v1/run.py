"""Run and close the missing E1Q1 canonical Solver evaluations.

This runner is deliberately narrower than the paired Round-1 runner. Its
input pool is the frozen 158-candidate missing manifest, and its only live
work is the two trained merged policies under the frozen B0 K=3 contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..h4_round0_b0_k3_difficulty_v1.contract import build_prompt
from ..h4_round0_b0_k3_difficulty_v1.gate import PROGRAM_PASS, build_solver_contract, gate_candidate
from ..h4_round0_b0_k3_difficulty_v1.io_utils import load_json, load_jsonl, verify_sha_sidecar
from ..h4_round0_b0_k3_difficulty_v1 import candidate_request_count
from ..h4_round1_paired_blind_solver_eval_v1 import run as paired
from . import EXISTING_SOLVER_RUN, GENERATION_RUN, IDENTITY_REPORT, OUTPUT_RUN, PROBE_VERSION, REROUTE_RUN


MODELS = paired.MODELS
MISSING_MANIFEST = REROUTE_RUN / "missing_solver_evaluation_manifest.jsonl"
MISSING_PLAN = REROUTE_RUN / "missing_solver_request_plan.json"
ROUTING = REROUTE_RUN / "e1q1_multilabel_routing.jsonl"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact(path: Path, value: Any) -> str:
    if path.suffix == ".jsonl":
        text = "".join(canonical(row) + "\n" for row in value)
    elif isinstance(value, str):
        text = value if value.endswith("\n") else value + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def require_source(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing_source:{path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0] != sha256_file(path):
        raise RuntimeError(f"source_sidecar_mismatch:{path}")


def load_candidates() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    for path in (MISSING_MANIFEST, MISSING_PLAN, ROUTING):
        require_source(path)
    plan = load_json(MISSING_PLAN)
    if int(plan.get("request_count", -1)) != 158:
        raise RuntimeError(f"missing_plan_count:{plan.get('request_count')}")
    ids = set(str(value) for value in plan.get("candidate_ids") or [])
    manifest = load_jsonl(MISSING_MANIFEST)
    if len(manifest) != 158 or {str(row["candidate_id"]) for row in manifest} != ids:
        raise RuntimeError("missing_manifest_plan_id_mismatch")
    routing = {str(row["candidate_id"]): row for row in load_jsonl(ROUTING)}
    rows: list[dict[str, Any]] = []
    pools: dict[tuple[str, str], dict[str, Any]] = {}
    dialogues: dict[tuple[str, str], dict[str, Any]] = {}
    roots: set[str] = set()
    for root_dir in sorted(GENERATION_RUN.glob("cc-train-episode-*")):
        candidate_path = root_dir / "candidates_valid_scoped.json"
        dialogue_path = root_dir / "dialogue_source.json"
        if not candidate_path.is_file() or not dialogue_path.is_file():
            continue
        require_source(candidate_path); require_source(dialogue_path)
        root = root_dir.name
        dialogue = load_json(dialogue_path)
        selected = [row for row in load_json(candidate_path) if str(row.get("candidate_id")) in ids]
        if selected:
            roots.add(root)
            pool = {str(row["candidate_id"]): dict(row) for row in selected}
            pools[("feedback", root)] = pool
            dialogues[("feedback", root)] = dialogue
            rows.extend({**dict(row), "branch": "feedback"} for row in selected)
    if len(rows) != 158 or {str(row["candidate_id"]) for row in rows} != ids:
        raise RuntimeError(f"selected_candidate_count_or_ids:{len(rows)}")
    if any(str(row["candidate_id"]) not in routing for row in rows):
        raise RuntimeError("selected_candidate_missing_routing_row")
    return rows, pools, dialogues, routing


def build_inputs(rows: list[dict[str, Any]], pools: dict[tuple[str, str], dict[str, Any]], dialogues: dict[tuple[str, str], dict[str, Any]], tokenizer: Any, run_root: Path) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    gates: dict[str, dict[str, Any]] = {}
    (run_root / "source_pools").mkdir(parents=True, exist_ok=True)
    for root in sorted({str(row["root_id"]) for row in rows}):
        pool = pools[("feedback", root)]
        pool_path = run_root / "source_pools" / f"{root}.json"
        write_artifact(pool_path, list(pool.values()))
    for row in rows:
        cid, root = str(row["candidate_id"]), str(row["root_id"])
        candidate = pools[("feedback", root)][cid]
        dialogue = dialogues[("feedback", root)]
        gate = gate_candidate(candidate, dialogue, tokenizer, root_id=root)
        if gate.get("gate_status") != PROGRAM_PASS:
            raise RuntimeError(f"program_gate_failed:{cid}:{gate.get('program_errors')}")
        pool_path = run_root / "source_pools" / f"{root}.json"
        dialogue_path = GENERATION_RUN / root / "dialogue_source.json"
        gate.update({
            "candidate_pool_path": str(pool_path.resolve()), "candidate_pool_sha256": sha256_file(pool_path),
            "dialogue_path": str(dialogue_path.resolve()), "dialogue_sha256": sha256_file(dialogue_path),
            "branch": "feedback",
        })
        gates[cid] = gate
    return gates, pools


def preflight(run_root: Path = OUTPUT_RUN) -> dict[str, Any]:
    run_root = run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"output_exists_or_nonempty:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    rows, pools, dialogues, _ = load_candidates()
    tokenizer = AutoTokenizer.from_pretrained(str(paired.BASE_MODEL), local_files_only=True)
    gates, pools = build_inputs(rows, pools, dialogues, tokenizer, run_root)
    protocol_path = EXISTING_SOLVER_RUN / "solver_protocol.json"
    require_source(protocol_path)
    protocol = load_json(protocol_path)
    expected_protocol = build_solver_contract(tokenizer, model_path=paired.BASE_MODEL, tokenizer_path=paired.BASE_MODEL)
    if canonical(protocol) != canonical(expected_protocol):
        raise RuntimeError("solver_protocol_drift")
    qtypes = Counter(str(row["question_type"]) for row in rows)
    per_model = sum(candidate_request_count(row) for row in rows)
    if per_model != 4002:
        raise RuntimeError(f"request_count_not_closed:{per_model}")
    identity, models = paired._identity_preflight()
    write_artifact(run_root / "model_identity_report.json", identity)
    write_artifact(run_root / "solver_protocol.json", protocol)
    write_artifact(run_root / "candidate_manifest.jsonl", rows)
    write_artifact(run_root / "program_gate_results.jsonl", list(gates.values()))
    report = {
        "status": "PASS", "probe_version": PROBE_VERSION, "candidate_count": len(rows),
        "root_count": len({str(row["root_id"]) for row in rows}), "question_type_counts": dict(sorted(qtypes.items())),
        "per_model_request_count": per_model, "all_model_request_count": per_model * len(MODELS),
        "models": list(models), "base_model_used_as_solver": False, "tokenizer": str(paired.BASE_MODEL),
        "max_model_len": 16384, "temperatures": protocol.get("temperatures"), "rollouts_per_position": 3,
        "program_gate_pass_count": len(gates), "program_gate_reject_count": 0,
        "dev_test_accessed": False, "solver_started": False, "online_calls": 0,
    }
    write_artifact(run_root / "preflight_report.json", report)
    return report


def live(run_root: Path, gpu_memory_utilization: float = 0.80) -> dict[str, Any]:
    run_root = run_root.resolve()
    preflight_report = load_json(run_root / "preflight_report.json")
    if preflight_report.get("status") != "PASS":
        raise RuntimeError("solver_preflight_required")
    rows, pools, dialogues, _ = load_candidates()
    tokenizer = AutoTokenizer.from_pretrained(str(paired.BASE_MODEL), local_files_only=True)
    gates, pools = build_inputs(rows, pools, dialogues, tokenizer, run_root)
    protocol = load_json(run_root / "solver_protocol.json")
    paired.BRANCHES = ("feedback",)
    identity, models = paired._identity_preflight()
    model_results: dict[str, Any] = {}
    from ..h4_round0_b0_k3_difficulty_v1.io_utils import RunLock
    with RunLock(run_root):
        for model_name in MODELS:
            result = paired._run_model(
                model_name=model_name, model_identity=models[model_name], rows=rows, gates=gates,
                pools=pools, dialogues=dialogues, tokenizer=tokenizer, protocol=protocol,
                run_root=run_root, gpu_memory_utilization=gpu_memory_utilization,
            )
            model_results[model_name] = result["model_summary"]
    report = {
        "status": "E1Q1_MISSING_CANONICAL_SOLVER_COMPLETE", "candidate_count": len(rows),
        "models": list(MODELS), "model_summaries": model_results,
        "request_count": sum(branch["request_count"] for summary in model_results.values() for branch in summary["branches"].values()),
        "technical_failure_count": sum(branch["technical_failure_count"] for summary in model_results.values() for branch in summary["branches"].values()),
        "dev_test_accessed": False, "base_model_used_as_solver": False, "semantic_auditor_calls": 0,
        "online_calls": sum(branch["request_count"] for summary in model_results.values() for branch in summary["branches"].values()),
    }
    write_artifact(run_root / "missing_solver_live_report.json", report)
    return report


def load_new_difficulty(run_root: Path, model: str) -> list[dict[str, Any]]:
    path = run_root / model / "feedback" / "difficulty_manifest.jsonl"
    require_source(path)
    return load_jsonl(path)


def close(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    live_report = load_json(run_root / "missing_solver_live_report.json")
    if live_report.get("technical_failure_count") != 0:
        raise RuntimeError("technical_failure_nonzero")
    routing = {str(row["candidate_id"]): row for row in load_jsonl(ROUTING)}
    evaluated_candidates = {str(row["candidate_id"]): row for row in load_jsonl(run_root / "candidate_manifest.jsonl")}
    plan = load_json(MISSING_PLAN)
    new_rows = {model: load_new_difficulty(run_root, model) for model in MODELS}
    if any(len(new_rows[model]) != 158 for model in MODELS):
        raise RuntimeError("new_solver_manifest_count_mismatch")
    existing = {
        model: load_jsonl(EXISTING_SOLVER_RUN / model / "feedback" / "difficulty_manifest.jsonl")
        for model in MODELS
    }
    if any(len(existing[model]) != 73 for model in MODELS):
        raise RuntimeError("existing_solver_manifest_count_mismatch")
    combined = {model: sorted(existing[model] + new_rows[model], key=lambda row: str(row["candidate_id"])) for model in MODELS}
    if any(len({str(row["candidate_id"]) for row in combined[model]}) != 231 for model in MODELS):
        raise RuntimeError("combined_solver_ids_not_231")

    complete = []
    for row in load_jsonl(ROUTING):
        cid = str(row["candidate_id"])
        status = str(row.get("solver_coverage", {}).get("solver_eval_status"))
        if status == "CANDIDATE_COMPLETE":
            result_status = "PREVIOUSLY_COMPLETE"
        elif status == "MISSING_BOTH":
            result_status = "NEW_CANONICAL_COMPLETE"
        else:
            result_status = "REPRESENTATIVE_COVERED_ALIAS"
        rep = row.get("solver_coverage", {}).get("cluster_representative_candidate_id")
        complete.append({
            "candidate_id": cid, "root_id": row.get("root_id"), "question_type": (evaluated_candidates.get(cid) or {}).get("question_type") or row.get("question_type"),
            "normalized_experiment_arm": row.get("normalized_experiment_arm"), "split": row.get("split"),
            "solver_completion_status": result_status, "representative_candidate_id": rep,
            "representative_source": "E1Q0" if rep and rep not in routing else ("E1Q1" if rep else None),
            "solver_prompt_fingerprint": row.get("solver_prompt_fingerprint"),
            "feedback_target_hit": row.get("feedback_target_hit"), "off_target": row.get("off_target"),
            "clean_complete": cid in {str(r["candidate_id"]) for r in combined["M_clean_policy"]},
            "inclusive_complete": cid in {str(r["candidate_id"]) for r in combined["M_inclusive_policy"]},
        })
    status_counts = Counter(row["solver_completion_status"] for row in complete)
    if status_counts != Counter({"PREVIOUSLY_COMPLETE": 73, "NEW_CANONICAL_COMPLETE": 158, "REPRESENTATIVE_COVERED_ALIAS": 7}):
        raise RuntimeError(f"completion_status_not_closed:{status_counts}")

    cluster_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        cluster_groups[str(row.get("solver_prompt_fingerprint") or "UNAVAILABLE")].append(row)
    cluster_map = []
    representatives = []
    for fingerprint, members in sorted(cluster_groups.items()):
        member_ids = sorted(row["candidate_id"] for row in members)
        external_reps = sorted({str(row["representative_candidate_id"]) for row in members if row.get("representative_candidate_id") and row["representative_candidate_id"] not in {item["candidate_id"] for item in members}})
        represented = [row for row in members if row["solver_completion_status"] in {"PREVIOUSLY_COMPLETE", "NEW_CANONICAL_COMPLETE"}]
        representative = sorted(represented, key=lambda row: row["candidate_id"])[0] if represented else members[0]
        representative_id = external_reps[0] if external_reps else representative["candidate_id"]
        for row in members:
            cluster_map.append({
                "candidate_id": row["candidate_id"], "solver_prompt_fingerprint": fingerprint,
                "cluster_representative_candidate_id": representative_id,
                "cluster_member_ids": member_ids, "cluster_member_count": len(member_ids),
                "solver_completion_status": row["solver_completion_status"],
                "alias": row["candidate_id"] != representative_id,
            })
        representatives.append({
            "solver_prompt_fingerprint": fingerprint, "representative_candidate_id": representative_id,
            "member_ids": member_ids, "member_count": len(member_ids),
            "representative_completion_status": representative["solver_completion_status"],
        })
    technical = {
        "technical_failure_count": int(live_report.get("technical_failure_count") or 0),
        "per_model": {model: {"technical_failure_count": sum(int(row.get("technical_failure_count") or 0) for row in new_rows[model]), "candidate_count": len(new_rows[model])} for model in MODELS},
        "technical_exclusion_rows": [row["candidate_id"] for model in MODELS for row in new_rows[model] if row.get("technical_exclusion_reason")],
        "status": "PASS" if int(live_report.get("technical_failure_count") or 0) == 0 else "FAIL",
    }
    qtypes = Counter(str(row["question_type"]) for row in new_rows[MODELS[0]])
    expected_per_model = sum(candidate_request_count(row) for row in new_rows[MODELS[0]])
    observed_per_model = {model: sum(int(row.get("request_count") or 0) for row in new_rows[model]) for model in MODELS}
    closure = {
        "version": PROBE_VERSION, "status": "E1Q1_SOLVER_COVERAGE_CLOSED",
        "raw_e1q1_candidates": 238, "previously_complete_candidates": 73,
        "new_canonical_representatives": 158, "representative_covered_aliases": 7,
        "closure_equation": "238 = 73 + 158 + 7", "closure_equation_pass": True,
        "independent_prompt_cluster_count": len(representatives), "cluster_alias_count": sum(1 for row in cluster_map if row["alias"]),
        "technical_failure_count": technical["technical_failure_count"], "semantic_auditor_calls": 0,
        "dev_test_accessed": False, "excluded_e0q0_count": 276, "excluded_e1q0_count": 254,
        "new_qtype_counts": dict(sorted(qtypes.items())), "expected_new_requests_per_model": expected_per_model,
        "observed_new_requests_per_model": observed_per_model, "observed_new_requests_total": sum(observed_per_model.values()),
        "request_formula": "24*N_single + 30*N_multi + 24*N_ordering",
    }
    report = {
        "version": PROBE_VERSION, "status": "E1Q1_MISSING_SOLVER_EVALUATION_COMPLETE",
        "next_status": "READY_FOR_QUESTION_BANK_LIFECYCLE_FREEZE", "coverage_closure": closure,
        "technical_status": technical["status"], "online_calls": sum(observed_per_model.values()),
        "source_manifests_rewritten": False, "feedback_target_hit_policy": "UNKNOWN_RETAINED",
    }
    report_md = "\n".join([
        f"# {PROBE_VERSION}", "", "E1Q1 missing canonical Solver evaluation is closed.", "",
        "- Closure: `238 = 73 + 158 + 7`", f"- Independent Solver prompt clusters: {len(representatives)}",
        f"- New canonical representatives: 158; new requests total: {sum(observed_per_model.values())}",
        f"- Technical failures: {technical['technical_failure_count']}",
        "- Models: M_clean_policy and M_inclusive_policy; frozen B0 K=3 contract.",
        "- `feedback_target_hit` remains `UNKNOWN`; no semantic auditor was called.", "",
        "Next step: Question Bank lifecycle and replay-weight freeze.", "",
    ])
    for model in MODELS:
        write_artifact(run_root / ("clean_complete_difficulty_manifest.jsonl" if model == "M_clean_policy" else "inclusive_complete_difficulty_manifest.jsonl"), combined[model])
    write_artifact(run_root / "e1q1_solver_coverage_closure.json", closure)
    write_artifact(run_root / "e1q1_complete_candidate_manifest.jsonl", complete)
    write_artifact(run_root / "solver_prompt_cluster_map.jsonl", cluster_map)
    write_artifact(run_root / "cluster_representative_manifest.jsonl", representatives)
    write_artifact(run_root / "technical_failure_report.json", technical)
    write_artifact(run_root / "solver_request_closure.json", {
        "version": PROBE_VERSION, "new_canonical_candidate_count": 158,
        "question_type_counts": dict(sorted(qtypes.items())), "expected_requests_per_model": expected_per_model,
        "observed_requests_per_model": observed_per_model, "observed_requests_total": sum(observed_per_model.values()),
        "excluded_previously_complete": 73, "excluded_aliases": 7, "excluded_e0q0": 276, "excluded_e1q0": 254,
        "excluded_dev_test": True, "formula": closure["request_formula"], "request_count_closed": observed_per_model == {model: expected_per_model for model in MODELS},
    })
    write_artifact(run_root / "final_missing_e1q1_solver_report.json", report)
    write_artifact(run_root / "final_missing_e1q1_solver_report.md", report_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "live", "close", "all"), required=True)
    parser.add_argument("--run-root", type=Path, default=OUTPUT_RUN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.phase in {"preflight", "all"}:
        print(json.dumps(preflight(args.run_root), ensure_ascii=False, indent=2), flush=True)
    if args.phase in {"live", "all"}:
        if not args.live:
            raise RuntimeError("solver_live_requires_explicit_live")
        print(json.dumps(live(args.run_root, args.gpu_memory_utilization), ensure_ascii=False, indent=2), flush=True)
    if args.phase in {"close", "all"}:
        print(json.dumps(close(args.run_root), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
