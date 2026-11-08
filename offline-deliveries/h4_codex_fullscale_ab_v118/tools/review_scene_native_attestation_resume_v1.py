from __future__ import annotations
import copy, json, importlib.util, sys
from pathlib import Path

def req(x,m):
    if not x: raise ValueError(m)
def load(path):
    path=Path(path).resolve(); req(path.name=='review_v3.py','BOUND_ENGINE')
    sys.path.insert(0,str(path.parent)); spec=importlib.util.spec_from_file_location('native_resume_engine',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def resume(batch,root,executor,scene,judgment,engine_path):
    eng=load(engine_path); data=eng.inputs(batch,root,'scene',executor,scene)
    directory=eng.read_directory(data); ctxp=directory/'READ_CONTEXT.json'; ctx=eng.read(ctxp)
    req((ctx.get('root_id'),ctx.get('phase'),ctx.get('scene_ref'),ctx.get('executor_id'))==(root,'scene',scene,executor),'CTX_ID')
    req(ctx.get('input_bindings')==data['input_bindings'],'INPUTS_CHANGED')
    for item in [*ctx['input_bindings'],ctx['input_render_binding']]: eng.verify_binding(item,data['unit'],'CTX_INPUT')
    req(ctx['input_render_binding']==eng.binding(directory/'INPUT_RENDER.txt'),'RENDER_PATH')
    req((directory/'INPUT_RENDER.txt').read_text()==eng.render(data),'RENDER_CHANGED')
    archive=directory/'RAW_JUDGMENT.json'; failure=directory/'SAVE_FAILURE.json'; fr=eng.read(failure)
    req(fr.get('status')=='FIRST_JUDGMENT_REJECTED_PRESERVED' and fr.get('error')=='EXPLICIT_NATIVE_READ_ATTESTATION_REQUIRED','EXPECTED_FAILURE')
    req(fr.get('raw_judgment_binding')==eng.binding(archive),'RAW_BINDING')
    original=eng.read(archive); corrected=eng.read(judgment)
    expected=copy.deepcopy(original); expected['read_attestation']['native_event_full_read']=True
    req(corrected==expected,'ONLY_NATIVE_ATTESTATION_ADDITION_ALLOWED')
    att=eng.validate_attestation(corrected,data); supplied=eng.validate_scene(corrected,data)
    outdir=directory/'native_attestation_resume_v1'; outdir.mkdir(exist_ok=False)
    corr=outdir/'CORRECTED_JUDGMENT.json'; corr.write_text(json.dumps(corrected,ensure_ascii=False,indent=2)+'\n')
    finished=eng.now()
    review={'schema_version':'codex_assigned_source_review_v2','root_id':root,'reviewer':executor,'review_role':'non_author_complete_cumulative_source_and_native_event_reviewer','backend':'CURRENT_CODEX_SESSION','exact_model_id':None,'external_semantic_certification':False,'started_at_utc':ctx['started_at_utc'],'finished_at_utc':finished,'created_at_utc':finished,'source_sha256':data['source_binding']['sha256'],'full_source_read':True,'turn_refs_read':list(data['turns']),'all_continuation_constraints_read':True,'constraints_binding':data['constraints_binding'],'role_assignments_binding':data['role_assignments_binding'],'read_context_binding':eng.binding(ctxp),'input_render_binding':ctx['input_render_binding'],'raw_judgment_binding':eng.binding(corr),'helper_binding':eng.binding(engine_path),'executor_read_attestation':att,'identity_verification_scope':eng.IDENTITY_LIMITATION,'read_attestation_scope':eng.READ_LIMITATION,'semantic_review_is_programmatic':False,**supplied,'scene_ref':scene,'native_event_full_read':True,'native_event_path':data['native_binding']['path'],'native_event_sha256':data['native_binding']['sha256'],'native_turn_ids_read':[t['turn_id'] for t in data['native']['turns']],'source_id':data['author_record'].get('source_id')}
    if data['selection_binding'] is not None: review.update(selection_path=data['selection_binding']['path'],selection_sha256=data['selection_binding']['sha256'])
    eng.save_new(data['output'],review)
    prov={'schema_version':'scene_native_attestation_resume_v1','original_raw_judgment_binding':eng.binding(archive),'original_save_failure_binding':eng.binding(failure),'corrected_submission_binding':eng.binding(judgment),'bound_validation_engine':eng.binding(engine_path),'resume_adapter_binding':eng.binding(__file__),'only_added_field':'read_attestation.native_event_full_read','added_value':True,'semantic_dictionary_unchanged':True,'private_file_access_proven_by_program':False}
    eng.save_new(outdir/'SAVE_RECEIPT.json',{'status':'RECORDED_CORRECTED_NATIVE_ATTESTATION_NOT_EXTERNAL_CERTIFICATION','created_at_utc':eng.now(),'executor_id':executor,'root_id':root,'phase':'scene','scene_ref':scene,'read_context_binding':eng.binding(ctxp),'output_binding':eng.binding(data['output']),'corrected_archive_binding':eng.binding(corr),**prov})
    print(json.dumps({'status':'SCENE_NATIVE_ATTESTATION_RESUME_SAVED','output':str(data['output'])}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--root',required=True);p.add_argument('--scene',required=True);p.add_argument('--executor',required=True);p.add_argument('--judgment',type=Path,required=True);p.add_argument('--engine',type=Path,required=True);a=p.parse_args();resume(a.batch,a.root,a.executor,a.scene,a.judgment,a.engine)
