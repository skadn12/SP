"""Resume a rejected scene review with only schema-normalization additions.

The first RAW_JUDGMENT and SAVE_FAILURE are immutable.  This adapter is limited
to the two observed mechanical failures: string findings are wrapped as
nonblocking WARNING findings, and a missing ``new_blocking_findings`` field is
added as an empty list.  It supplies no semantic review and refuses any other
change before delegating validation to the bound review_v3 engine.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, sys
from pathlib import Path

def require(ok, msg):
    if not ok: raise ValueError(msg)

def load(path):
    path=Path(path).resolve(); require(path.name == "review_v3.py", "SUPPORTED_BOUND_ENGINE_REQUIRED")
    sys.path.insert(0, str(path.parent)); spec=importlib.util.spec_from_file_location("scene_resume_bound_engine", path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def norm(raw):
    """Return the only schema-normalized representation allowed by this adapter."""
    out=copy.deepcopy(raw)
    findings=out.get("findings")
    if isinstance(findings,list) and all(isinstance(x,str) and x.strip() for x in findings):
        out["findings"]=[{"severity":"WARNING","reason":x} for x in findings]
    require(isinstance(out.get("findings"),list), "FINDINGS_LIST_REQUIRED")
    b=out.get("baseline_constraint_review")
    require(isinstance(b,dict), "CONSTRAINT_REVIEW_REQUIRED")
    if "new_blocking_findings" not in b: b["new_blocking_findings"]=[]
    return out

def resume(batch, root, executor, scene, judgment, engine_path, check_only=False):
    eng=load(engine_path)
    data=eng.inputs(batch,root,"scene",executor,scene)
    directory=eng.read_directory(data); context_path=directory/"READ_CONTEXT.json"
    context=eng.read(context_path)
    require((context.get("root_id"),context.get("phase"),context.get("scene_ref"),context.get("executor_id"))==(root,"scene",scene,executor),"READ_CONTEXT_IDENTITY_MISMATCH")
    require(context.get("input_bindings")==data["input_bindings"],"ORIGINAL_READ_INPUTS_CHANGED")
    for item in [*context["input_bindings"],context["input_render_binding"]]: eng.verify_binding(item,data["unit"],"ORIGINAL_READ_CONTEXT_INPUT")
    require(context["input_render_binding"]==eng.binding(directory/"INPUT_RENDER.txt"),"RENDER_PATH_CHANGED")
    require((directory/"INPUT_RENDER.txt").read_text(encoding="utf-8")==eng.render(data),"ORIGINAL_RENDER_CHANGED")
    archive=directory/"RAW_JUDGMENT.json"; failure_path=directory/"SAVE_FAILURE.json"
    failure=eng.read(failure_path)
    require(failure.get("status")=="FIRST_JUDGMENT_REJECTED_PRESERVED" and failure.get("executor_id")==executor,"ONLY_RECORDED_MECHANICAL_FAILURE")
    require(failure.get("raw_judgment_binding")==eng.binding(archive),"ORIGINAL_RAW_BINDING_CHANGED")
    original=eng.read(archive); supplied=eng.read(judgment); corrected=norm(supplied); expected=norm(original)
    require(corrected==expected,"ONLY_SCHEMA_NORMALIZATION_ALLOWED")
    attestation=eng.validate_attestation(corrected,data); semantic=eng.validate_scene(corrected,data)
    provenance={"schema_version":"scene_review_schema_resume_v1","original_raw_judgment_binding":eng.binding(archive),"original_save_failure_binding":eng.binding(failure_path),"corrected_submission_binding":eng.binding(judgment),"bound_validation_engine":eng.binding(engine_path),"resume_adapter_binding":eng.binding(__file__),"normalizations_allowed":["findings_string_to_warning_reason_objects","missing_new_blocking_findings_to_empty_list"],"semantic_dictionary_unchanged":True,"private_file_access_proven_by_program":False}
    if check_only:return {"status":"MECHANICAL_SCENE_RESUME_CHECK_PASS",**provenance}
    outdir=directory/"scene_review_resume_v1"; outdir.mkdir(exist_ok=False)
    corrected_archive=outdir/"CORRECTED_JUDGMENT.json"; corrected_archive.write_text(json.dumps(corrected,ensure_ascii=False,indent=2)+"\n")
    finished=eng.now()
    review={"schema_version":"codex_assigned_source_review_v2","root_id":root,"reviewer":executor,"review_role":"non_author_complete_cumulative_source_and_native_event_reviewer","backend":"CURRENT_CODEX_SESSION","exact_model_id":None,"external_semantic_certification":False,"started_at_utc":context["started_at_utc"],"finished_at_utc":finished,"created_at_utc":finished,"source_sha256":data["source_binding"]["sha256"],"full_source_read":True,"turn_refs_read":list(data["turns"]),"all_continuation_constraints_read":True,"constraints_binding":data["constraints_binding"],"role_assignments_binding":data["role_assignments_binding"],"read_context_binding":eng.binding(context_path),"input_render_binding":context["input_render_binding"],"raw_judgment_binding":eng.binding(corrected_archive),"helper_binding":eng.binding(engine_path),"executor_read_attestation":attestation,"identity_verification_scope":eng.IDENTITY_LIMITATION,"read_attestation_scope":eng.READ_LIMITATION,"semantic_review_is_programmatic":False,**semantic}
    review["scene_ref"]=scene
    if data["native"] is not None:
        review.update(native_event_full_read=True,native_event_path=data["native_binding"]["path"],native_event_sha256=data["native_binding"]["sha256"],native_turn_ids_read=[t["turn_id"] for t in data["native"]["turns"]],source_id=data["author_record"].get("source_id"))
        if data["selection_binding"] is not None: review.update(selection_path=data["selection_binding"]["path"],selection_sha256=data["selection_binding"]["sha256"])
    eng.save_new(data["output"],review)
    eng.save_new(outdir/"SAVE_RECEIPT.json",{"status":"RECORDED_CORRECTED_SCENE_SCHEMA_NOT_EXTERNAL_CERTIFICATION","created_at_utc":eng.now(),"executor_id":executor,"root_id":root,"phase":"scene","scene_ref":scene,"read_context_binding":eng.binding(context_path),"output_binding":eng.binding(data["output"]),"corrected_archive_binding":eng.binding(corrected_archive),**provenance})
    return {"status":"SCENE_SCHEMA_RESUME_SAVED","output":str(data["output"])}

def main():
    p=argparse.ArgumentParser();p.add_argument("--batch",type=Path,required=True);p.add_argument("--root",required=True);p.add_argument("--scene",required=True);p.add_argument("--executor",required=True);p.add_argument("--judgment",type=Path,required=True);p.add_argument("--engine",type=Path,required=True);p.add_argument("--check-only",action="store_true");a=p.parse_args();print(json.dumps(resume(a.batch,a.root,a.executor,a.scene,a.judgment,a.engine,a.check_only),ensure_ascii=False))
if __name__=="__main__": main()
