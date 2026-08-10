#!/usr/bin/env python3
"""Run Child Expander, Realizer, and unchanged constrained QA Writer for E1/E2."""

from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path

from .core import atomic_json, atomic_text, sha256_file
from .ledger import RequestLedger
from .lineage_three_stage_canary import ApiBudget, artifact_sha256_manifest, run_single_object_task
from .live_backends import LiveSemanticRepairBackend, OpenAICompatibleChallengerClient, load_challenger_config
from .outline_dependency_mainline import (apply_anti_leak_gate, build_qa_prompt,
    validate_child_graph, validate_qa, validate_realizer)
from .run_canary import create_source_snapshot, project_diff_sha256
from .run_outline_dependency_mainline import PROJECT, MODULE_ROOT, PROMPTS, _source, _view
from .single_arc_typed_dependency import (build_child_expander_prompt, build_realizer_prompt,
    validate_child_graph_with_semantic_split)
from . import run_dependency_conditioned_child_canary as base

DEFAULT_SOURCE=Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_single_arc_typed_dependency_canary_v1")
DEFAULT_ROOT=Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_typed_dependency_downstream_canary_v1")

def _read(p:Path): return json.loads(p.read_text(encoding="utf-8"))
def _tree(root:Path): return {str(p.relative_to(root)):sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--source-root",type=Path,default=DEFAULT_SOURCE); p.add_argument("--run-root",type=Path,default=DEFAULT_ROOT); p.add_argument("--api-config",type=Path,default=base.DEFAULT_API_CONFIG); a=p.parse_args(argv); root=a.run_root.resolve()
    if root.exists(): raise FileExistsError(f"run_root_exists:{root}")
    source_report=_read(a.source_root/"reports/single_arc_typed_dependency_canary.json")
    selected=source_report["selected_for_downstream"]
    if len(selected)<2: raise RuntimeError("fewer_than_two_typed_dependencies")
    deps=[_read(a.source_root/f"frozen/typed_dependencies/{arc}.json") for arc in selected[:2]]
    source=_source(); arcs={x["event_id"]:x for x in source["outline"]["planned_event_arcs"]}
    source_tree_before=_tree(a.source_root); root.mkdir(parents=True); snapshot=create_source_snapshot(PROJECT,root,False)
    api=load_challenger_config(a.api_config); manifest={"run_id":root.name,"mode":"typed_dependency_downstream_canary_v1","git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip(),"project_diff_sha256":project_diff_sha256(PROJECT),**snapshot,"source_typed_root":str(a.source_root),"source_typed_report_sha256":sha256_file(a.source_root/"reports/single_arc_typed_dependency_canary.json"),"source_typed_tree_before":source_tree_before,"selected_arcs":selected[:2],"questioner_prompt_modified":False,"ordinary_questioner":0,"distractor":0,"solver":0,"training":0,"normal_request_cap":6,"format_retry_cap":6,"physical_cap":12}
    atomic_json(root/"config/run_manifest.json",manifest); atomic_json(root/"frozen/accepted_typed_dependencies.json",deps)
    ledger=RequestLedger(root,sha256_file(root/"config/run_manifest.json"),"live"); budget=ApiBudget(normal_caps={"child_expander":2,"realizer":2,"dependency_qa_writer":2},format_retry_cap=6,physical_cap=12); client=OpenAICompatibleChallengerClient(api)
    outputs=[]
    for dep in deps:
        did=dep["dependency_id"]; arc=arcs[dep["arc_id"]]
        cp=build_child_expander_prompt(arc,dep,PROMPTS["child"].read_text()); atomic_text(root/f"prompts/child_expander_{did}.txt",cp)
        child=run_single_object_task(run_root=root,ledger=ledger,backend=LiveSemanticRepairBackend(client,2048),budget=budget,stage="child_expander",logical_request_id=f"typed-child-{did}",semantic_prompt=cp,lineage_view=_view([],{"dependency_id":did,"side":"B"}),validator=lambda raw,d=dep:validate_child_graph_with_semantic_split(raw,d),schema_retry_decisions=("outline_child_graph_schema_invalid",))
        row={"dependency_id":did,"arc_id":dep["arc_id"],"child_request":{k:v for k,v in child.items() if k not in {"accepted_raw","accepted_validation"}}}
        if not child["accepted"]: row["decision"]="child_graph_reject"; outputs.append(row); continue
        graph=child["accepted_raw"]; atomic_json(root/f"frozen/child_graphs/{did}.json",graph)
        rp=build_realizer_prompt(graph,PROMPTS["realizer"].read_text()); atomic_text(root/f"prompts/realizer_{did}.txt",rp)
        real=run_single_object_task(run_root=root,ledger=ledger,backend=base._FormalRealizerBatchAdapter(client),budget=budget,stage="realizer",logical_request_id=f"typed-realizer-{did}",semantic_prompt=rp,lineage_view=_view([],{"dependency_id":did,"child_graph":graph}),validator=lambda raw,d=dep,g=graph:validate_realizer(raw,d,g),schema_retry_decisions=("outline_realizer_schema_invalid",))
        row.update({"child_graph":graph,"realizer_request":{k:v for k,v in real.items() if k not in {"accepted_raw","accepted_validation"}}})
        if not real["accepted"]: row["decision"]="realizer_reject"; outputs.append(row); continue
        gate=apply_anti_leak_gate(real["accepted_raw"],dep,graph); row.update({"realizer_output":real["accepted_raw"],"anti_leak_gate":gate})
        if gate["decision"]!="outline_realization_local_gate_pass": row["decision"]=gate["decision"]; outputs.append(row); continue
        parent=dep["resolved_parent_turns"]; child_turns=gate["planned_event_evidence_turns"]
        qp=build_qa_prompt(dep,parent,child_turns,PROMPTS["qa"].read_text()); atomic_text(root/f"prompts/qa_writer_{did}.txt",qp)
        qa=run_single_object_task(run_root=root,ledger=ledger,backend=LiveSemanticRepairBackend(client,1024),budget=budget,stage="dependency_qa_writer",logical_request_id=f"typed-qa-{did}",semantic_prompt=qp,lineage_view=_view([*parent,*child_turns],{"dependency_id":did}),validator=lambda raw,d=dep:validate_qa(raw,d),schema_retry_decisions=("outline_qa_schema_invalid",))
        row.update({"qa_request":{k:v for k,v in qa.items() if k not in {"accepted_raw","accepted_validation"}},"qa":qa["accepted_raw"],"decision":"qa_accept" if qa["accepted"] else qa["final_decision"]}); outputs.append(row)
    usage=base._usage(ledger); accepted=sum(x["decision"]=="qa_accept" for x in outputs)
    report={"canary_version":"typed_dependency_downstream_canary_v1","items":outputs,"child_expansions":len(deps),"anti_leak_pass":sum(x.get("anti_leak_gate",{}).get("decision")=="outline_realization_local_gate_pass" for x in outputs),"qa_writer_outputs":accepted,"api_usage":usage,"ordinary_questioner_generation":0,"distractor":0,"solver":0,"training":0,"semantic_repair":0}
    atomic_json(root/"reports/typed_dependency_downstream_canary.json",report); atomic_json(root/"reports/api_usage.json",usage)
    lines=["# Typed dependency downstream review",""]
    for x in outputs:
        lines += [f"## {x['dependency_id']} / {x['arc_id']}","",f"Decision: `{x['decision']}`",""]
        for t in x.get("anti_leak_gate",{}).get("child_dialogue_with_aliases",[]): lines.append(f"- `{t['alias']}` {t['speaker']}: {t['text']}")
        lines += ["",f"QA: `{json.dumps(x.get('qa'),ensure_ascii=False)}`",""]
    atomic_text(root/"review/typed_dependency_downstream_review.md","\n".join(lines)); atomic_json(root/"reports/artifact_sha256.json",artifact_sha256_manifest(root))
    checks={"git_head_match":subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip()==manifest["git_head"],"project_diff_match":project_diff_sha256(PROJECT)==manifest["project_diff_sha256"],"source_typed_unchanged":_tree(a.source_root)==source_tree_before}
    atomic_json(root/"reports/post_run_provenance.json",{"checks":checks,"authoritative_release_provenance":all(checks.values()),"provenance_mixed":not all(checks.values())})
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
