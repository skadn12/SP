"""Materialize only newly accepted v97 scenes; preserve failures and exact bindings.

Read-only toward the frozen expansion engine/run. This stage makes no semantic
READY decision and cannot substitute historical content when expansion fails.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import uuid

V97 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V97 / 'qa_engine'))
import batch_scope
RUN = batch_scope.upstream_run()
ENGINE = RUN.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def immutable_bytes(path, data):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes()==data,'materialized_file_changed:'+str(path))
        return
    temporary=path.parent/('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_bytes(data)
        os.link(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def write(path, value):
    immutable_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def verify_record(path):
    record=read(path)
    for row in record['bound_files']:
        require(bind(row['path'])==row,'provenance_binding_changed:'+row['path'])
    return record


def stage(directory):
    proof=read(directory/'PROOF.json')
    terminal=None; verdict=None
    for name in ('ERROR.json','ACCEPTANCE.json','REVIEW.json','VALIDATION.json'):
        path=directory/name
        if path.exists():
            terminal=path;verdict=read(path);break
    if terminal is None:
        status='RESPONSE_WITHOUT_STAGE_VERDICT' if (directory/'RESPONSE.json').exists() else 'INCOMPLETE'
    elif terminal.name=='ERROR.json':status='TRANSPORT_FAILURE'
    elif terminal.name=='VALIDATION.json' and proof['role']=='dataset_extractor':
        status='PASS' if verdict.get('accepted_cards') else 'REJECT_NO_ACCEPTED_CARD'
    else:status=verdict.get('status','UNKNOWN')
    return {'request_directory':str(directory),'role':proof['role'],
            'root_id':proof.get('root_id'),'source_id':proof.get('source_id'),
            'scene_id':proof.get('scene_id'),'status':status,'verdict':verdict,
            'http_attempt_count':len(list(directory.glob('HTTP_ATTEMPT*.json'))),
            'bound_files':[bind(p) for p in sorted(directory.glob('*.json'))]}


def runtime():
    sys.path.insert(0,str(ENGINE))
    spec=importlib.util.spec_from_file_location('v97_materializer_frozen_runtime',ENGINE/'runner.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.verify(RUN)
    audit = read(RUN / 'FINAL_AUDIT.json')
    require(audit['status'] == 'PASS' and audit['failures'] == [] and audit['semantic_certified'] is False,
            'upstream_structural_audit_required_not_semantic_ready')
    return r


def materialize_once():
    r=runtime(); manifest=read(RUN/'MANIFEST.json');allocation=read(RUN/'UNIQUE_EVENT_ALLOCATION.json')
    completed=[];new=[]
    require(set(batch_scope.selected_roots()) <= set(manifest['roots']), 'selected_roots_not_in_upstream_run')
    for root in batch_scope.selected_roots():
        folder=RUN/'roots'/root/'B'
        if not (folder/'RESULT.json').exists() or not (folder/'cumulative_dialogue.json').exists():
            continue
        short=root.rsplit('-',1)[-1]
        provenance_path=V97/'evidence/expansion_provenance'/('root'+short+'_PROVENANCE.json')
        failure_path=V97/'evidence/expansion_failures'/('root'+short+'_FAILURE.json')
        already=provenance_path if provenance_path.exists() else failure_path if failure_path.exists() else None
        if already:
            completed.append(verify_record(already));continue
        result=read(folder/'RESULT.json');source_path=RUN/'sources'/(root+'.json')
        before=read(source_path);output_path=folder/'cumulative_dialogue.json';after=read(output_path)
        require(result['root_id']==root and result['route']==r.ROUTES[1],'root_or_route_changed')
        require(after['turns'][:len(before['turns'])]==before['turns'],'historical_prefix_changed')
        suffix=after['turns'][len(before['turns']):]
        require(len(suffix)==result['added_turns'],'added_turns_mismatch')
        require(len({t['turn_ref'] for t in after['turns']})==len(after['turns']),'duplicate_turn_ref')
        require(all(t['turn_ref'].startswith('R02:B:') for t in suffix),'new_suffix_ref_mismatch')
        require(r.base.digest(after['turns'])==result['output_history_sha256'],'output_history_hash_mismatch')
        source_ids={x['source_id'] for x in allocation['assignments'][root]}
        stages=[]
        for directory in sorted((RUN/'requests').iterdir()):
            if not (directory/'PROOF.json').exists():continue
            proof=read(directory/'PROOF.json')
            if proof.get('root_id')==root or proof.get('source_id') in source_ids:
                stages.append(stage(directory))
        require(all(s['status']!='INCOMPLETE' for s in stages),'completed_root_has_incomplete_request')
        stage_path=V97/'evidence/stage_index'/('root'+short+'_STAGES.json')
        stage_index={'version':'v97_expansion_root_stage_index_v1','root_id':root,
            'accepted_scenes':result['accepted_scenes'],'failed_scene_entries':result['failures'],
            'stages':stages,'successful_stage_count':sum(s['status']=='PASS' for s in stages),
            'failed_or_rejected_stage_count':sum(s['status']!='PASS' for s in stages),
            'requests_reexecuted':False,'network_requests':0}
        write(stage_path,stage_index)
        bindings={str(p.resolve()):bind(p) for p in folder.glob('*.json')}
        for path in (source_path,RUN/'MANIFEST.json',RUN/'SOURCE_RESERVATIONS.json',
                     RUN/'UNIQUE_EVENT_ALLOCATION.json',RUN/'dataset_cards_result.json',
                     RUN/'REPORT.json',RUN/'FINAL_AUDIT.json',
                     ENGINE/'SOURCE_REGISTRY.json',stage_path,Path(__file__)):
            bindings[str(path.resolve())]=bind(path)
        for binding in manifest['files']:
            require(bind(binding['path']) == binding, 'frozen_upstream_input_changed:' + binding['path'])
            bindings[binding['path']] = binding
        for item in stages:
            for b in item['bound_files']:bindings[b['path']]=b
        record={'version':'v97_materialized_expansion_provenance_v1','root_id':root,
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'run':str(RUN),
            'input_source':bind(source_path),'accepted_output':bind(output_path),
            'historical_input_turns':len(before['turns']),'output_turns':len(after['turns']),
            'new_turn_count':len(suffix),'new_turn_refs':[t['turn_ref'] for t in suffix],
            'accepted_scene_count':result['accepted_scenes'],'expansion_failures':result['failures'],
            'structural_audit':bind(RUN/'FINAL_AUDIT.json'),
            'reserved_dataset_sources':allocation['assignments'][root],
            'stage_index':bind(stage_path),'historical_prefix_preserved':True,
            'fresh_full_source_consistency_review_required':True,
            'source_consistency_decision':None,'question_answer_certification':False,
            'training_allowed':False,'historical_content_fallback':False}
        if result['accepted_scenes']>0:
            require(suffix,'accepted_scene_has_no_new_turns')
            expected=list(before['turns']);accepted_requests=[]
            for item in result['scene_metadata']:
                scene_id=item['scene_id'];scene=read(folder/(scene_id+'.json'))
                expected=r.base.append_scene(expected,root,r.ROUTES[1],scene)
                attempts=read(folder/(scene_id+'.ATTEMPTS.json'));last=attempts[-1]
                require(last['status']=='PASS','accepted_scene_without_pass_acceptance')
                request_dir=Path(last['request']);review_dir=Path(last['review_request'])
                require(read(request_dir/'ACCEPTANCE.json')==last,'acceptance_audit_mismatch')
                review=read(review_dir/'REVIEW.json');proof=read(review_dir/'PROOF.json')
                require(review['status']=='PASS','scene_review_not_pass')
                require(proof['realization_sha256']==r.base.digest(scene),'review_does_not_bind_accepted_scene')
                response=read(request_dir/'RESPONSE.json')
                parsed=r.parse_object(response['text'])
                require(r.base.validate_realization(parsed,root,r.ROUTES[1],read(folder/(scene_id+'.PLAN.json'))['scenes'][0])==scene,
                        'accepted_scene_not_actual_realizer_response')
                accepted_requests.append({'scene_id':scene_id,'realizer_request':str(request_dir),
                    'review_request':str(review_dir),'realization_sha256':r.base.digest(scene)})
            require(expected==after['turns'],'materialized_output_not_exact_accepted_scene_chain')
            destination=V97/'sources'/('root'+short+'_SOURCE.json')
            immutable_bytes(destination,output_path.read_bytes())
            require(bind(destination)['sha256']==bind(output_path)['sha256'],'source_copy_not_exact')
            bindings[str(destination.resolve())]=bind(destination)
            record.update(status='EXPANSION_ACCEPTED_AWAITING_FULL_SOURCE_REVIEW',
                          source=bind(destination),output_source=bind(destination),accepted_scene_requests=accepted_requests)
            record['bound_files']=list(bindings.values());write(provenance_path,record)
        else:
            require(not suffix,'failed_expansion_has_new_turns')
            record.update(status='EXPANSION_SHORTFALL_NO_SOURCE_MATERIALIZED',source=None,
                          accepted_scene_requests=[])
            record['bound_files']=list(bindings.values());write(failure_path,record)
        completed.append(record);new.append({'root_id':root,'status':record['status'],
            'source':record['source'],'provenance_path':str(provenance_path if record['source'] else failure_path),
            'stage_index_path':str(stage_path),'new_turn_count':record['new_turn_count']})
    final=(RUN/'REPORT.json').exists()
    if final:
        require(len(completed)==len(batch_scope.selected_roots()),'final_report_without_all_selected_materialized_root_outcomes')
        index_path=V97/'evidence/EXPANSION_MATERIALIZATION_COMPLETE.json'
        index={'version':'v97_expansion_materialization_complete_v1','status':'COMPLETE',
            'report':bind(RUN/'REPORT.json'),'roots':[{'root_id':x['root_id'],'status':x['status'],
            'new_turn_count':x['new_turn_count'],'source':x['source'],'stage_index':x['stage_index']} for x in completed],
            'accepted_roots':sum(x['source'] is not None for x in completed),
            'failed_roots':sum(x['source'] is None for x in completed),
            'selected_roots':list(batch_scope.selected_roots()),
            'upstream_roots_not_materialized':[root for root in manifest['roots'] if root not in batch_scope.selected_roots()],
            'source_ready_certified':False,'training_allowed':False,'network_requests':0}
        write(index_path,index)
    return {'new_materialized_roots':new,'completed_root_count':len(completed),'run_complete':final}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        result=materialize_once()
        if result['new_materialized_roots'] or result['run_complete'] or not args.watch:
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        if result['run_complete'] or not args.watch:break
        time.sleep(10)
