#!/usr/bin/env python3
"""Run only the audited single-arc typed Dependency Planner stage."""

from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path
from transformers import AutoTokenizer

from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .ledger import RequestLedger
from .lineage_three_stage_canary import ApiBudget, run_single_object_task, artifact_sha256_manifest
from .live_backends import LiveSemanticRepairBackend, OpenAICompatibleChallengerClient, load_challenger_config
from .run_canary import create_source_snapshot, project_diff_sha256
from .run_outline_dependency_mainline import PROJECT, MODULE_ROOT, _source, _view
from .single_arc_typed_dependency import build_prompt, materialize, schema_for, validate, VERSION
from . import run_dependency_conditioned_child_canary as base

DEFAULT_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_single_arc_typed_dependency_canary_v1")
DEFAULT_AUDIT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_outline_arc_dependency_feasibility_audit_v1")
CONTRACT = MODULE_ROOT / "prompts/single_arc_typed_dependency_planner_v1.md"

def _read(path: Path): return json.loads(path.read_text(encoding="utf-8"))

def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(); p.add_argument("--run-root",type=Path,default=DEFAULT_ROOT); p.add_argument("--audit-root",type=Path,default=DEFAULT_AUDIT); p.add_argument("--api-config",type=Path,default=base.DEFAULT_API_CONFIG); p.add_argument("--tokenizer",type=Path,default=base.DEFAULT_TOKENIZER); a=p.parse_args(argv)
    root=a.run_root.resolve()
    if root.exists(): raise FileExistsError(f"run_root_exists:{root}")
    audit=_read(a.audit_root/"reports/outline_arc_dependency_feasibility_audit.json")
    if audit["feasible_arc_count"] < 2: raise RuntimeError("typed_planner_threshold_not_met")
    source=_source(); candidates={r["arc_id"]:r for r in audit["items"]}; arcs={r["event_id"]:r for r in source["outline"]["planned_event_arcs"]}; anchors={r["anchor_id"]:r for r in source["outline"]["parent_anchors"]}; turns={r["turn_id"]:r for r in source["parent_dialogue"]}
    feasible=[candidates[k] for k in ("E1","E2","E3","E4") if candidates[k]["feasible"]]
    prompts=[build_prompt(arcs[c["arc_id"]],[anchors[x] for x in c["candidate_parent_anchor_ids"]],c,CONTRACT.read_text()) for c in feasible]
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True); prompt_tokens=[len(tok.encode(x,add_special_tokens=False)) for x in prompts]
    if any(x+1024>32768 for x in prompt_tokens): raise RuntimeError("typed_prompt_overflow")
    api=load_challenger_config(a.api_config)
    if api.model != "Qwen/Qwen3-14B": raise RuntimeError("unexpected_challenger_model")
    root.mkdir(parents=True); snapshot=create_source_snapshot(PROJECT,root,False)
    manifest={"run_id":root.name,"mode":VERSION,"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip(),"project_diff_sha256":project_diff_sha256(PROJECT),**snapshot,"source_feasibility_audit_sha256":sha256_file(a.audit_root/"reports/outline_arc_dependency_feasibility_audit.json"),"source_outline_sha256":source["outline_sha256"],"source_parent_anchor_sha256":source["anchor_sha256"],"prompt_file_sha256":sha256_file(CONTRACT),"schema_sha256_by_arc":{c["arc_id"]:sha256_bytes(canonical_json(schema_for(c)).encode()) for c in feasible},"normal_request_cap":len(feasible),"format_retry_cap":len(feasible),"physical_cap":2*len(feasible),"questioner_changes":0,"distractor":0,"solver":0,"training":0}
    atomic_json(root/"config/run_manifest.json",manifest); atomic_json(root/"config/resolved_canary_config.json",{"model":api.model,"temperature":api.temperature,"prompt_tokens_by_arc":dict(zip([c["arc_id"] for c in feasible],prompt_tokens)),"infeasible_arcs_skipped_without_api":[k for k in ("E1","E2","E3","E4") if not candidates[k]["feasible"]]})
    ledger=RequestLedger(root,sha256_file(root/"config/run_manifest.json"),"live"); budget=ApiBudget(normal_caps={"dependency_planner":len(feasible)},format_retry_cap=len(feasible),physical_cap=2*len(feasible)); backend=LiveSemanticRepairBackend(OpenAICompatibleChallengerClient(api),1024)
    items=[]
    for c,prompt in zip(feasible,prompts):
        arc=c["arc_id"]; atomic_text(root/f"prompts/single_arc_typed_dependency/{arc}.txt",prompt)
        task=run_single_object_task(run_root=root,ledger=ledger,backend=backend,budget=budget,stage="dependency_planner",logical_request_id=f"typed-dependency-{arc}",semantic_prompt=prompt,lineage_view=_view([], {"arc_id":arc,"audit":c}),validator=lambda raw,c=c:validate(raw,c),schema_retry_decisions=("typed_dependency_schema_invalid",))
        row={"arc_id":arc,"request":{k:v for k,v in task.items() if k not in {"accepted_raw","accepted_validation"}}}
        if task["accepted"] and task["accepted_raw"].get("status")=="valid":
            aid=task["accepted_raw"]["parent_anchor_ids"][0]; anchor=anchors[aid]; dep=materialize(task["accepted_raw"],c,anchor,turns[anchor["turn_id"]]); row.update({"decision":"typed_dependency_semantic_pass","dependency":dep}); atomic_json(root/f"frozen/typed_dependencies/{arc}.json",dep)
        elif task["accepted"]: row["decision"]="no_valid_dependency"
        else: row["decision"]=task["final_decision"]
        items.append(row)
    for arc in ("E1","E2","E3","E4"):
        if not candidates[arc]["feasible"]: items.append({"arc_id":arc,"decision":"skipped_zero_api_infeasible"})
    items.sort(key=lambda x:x["arc_id"]); passed=sum(x["decision"]=="typed_dependency_semantic_pass" for x in items)
    usage=base._usage(ledger); report={"canary_version":VERSION,"feasibility_count":audit["feasible_arc_count"],"items":items,"semantic_pass":passed,"selected_for_downstream":[x["arc_id"] for x in items if x["decision"]=="typed_dependency_semantic_pass"][:2],"api_usage":usage,"downstream_authorized":passed>=2}
    atomic_json(root/"reports/single_arc_typed_dependency_canary.json",report); atomic_json(root/"reports/api_usage.json",usage); atomic_json(root/"reports/artifact_sha256.json",artifact_sha256_manifest(root))
    checks={"git_head_match":subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip()==manifest["git_head"],"project_diff_match":project_diff_sha256(PROJECT)==manifest["project_diff_sha256"],"audit_sha_match":sha256_file(a.audit_root/"reports/outline_arc_dependency_feasibility_audit.json")==manifest["source_feasibility_audit_sha256"]}
    atomic_json(root/"reports/post_run_provenance.json",{"checks":checks,"authoritative_release_provenance":all(checks.values()),"provenance_mixed":not all(checks.values())})
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
