"""Bind selected native events and baseline constraints before v117 authoring."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

HERE = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_screened_ab_validation_v117')
PROJECT = HERE.parent.parent
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}
def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n')
def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f: f.write(Path(source).read_bytes())

def main():
    plan = read(HERE/'PRECALL_PLAN.json')
    gate = read(HERE.parent/'h4_codex_baseline_screen_v116/SELECTED_ROOTS.json')
    assert [r['root_id'] for r in plan['roots']] == [r['root_id'] for r in gate['roots']]
    allocations=[]
    for root, screened in zip(plan['roots'], gate['roots']):
        rid=root['root_id']; short=rid.rsplit('-',1)[1]
        selection_path=HERE/'selection'/('root'+short+'.json')
        selected=read(selection_path); baseline=read(root['baseline_path'])
        assert root['continuation_constraints']==screened['continuation_constraints']
        assert selected['full_baseline_read'] is True and selected['root_id']==rid
        assert selected['baseline_sha256']==root['baseline_sha256']==sha(root['baseline_path'])
        assert selected['baseline_turn_refs_read']==[t['turn_ref'] for t in baseline['turns']]
        poolp=HERE/'pre_inputs'/('root'+short+'_EVENT_POOL.json')
        pool={e['dialogue_id']:e for e in read(poolp)['events']}
        assert selected['partition']['event_pool_sha256']==sha(poolp)
        assert selected['continuation_constraints']==screened['continuation_constraints']
        assert len(selected['allocations'])==3
        for n,allocation in enumerate(selected['allocations'],1):
            assert allocation['root_id']==rid and allocation['scene_ref']==f'R02:B:S{n:02}'
            eid=allocation['source_id'].removeprefix('multiwoz22:train:')
            event=pool[eid]
            assert allocation['source_id'] not in plan['consumed_source_ids']
            assert allocation['full_event_utterances_read'] is True
            assert allocation['native_turn_ids_read']==[t['turn_id'] for t in event['turns']]
            if 'native_event' in allocation: assert allocation['native_event']==event
            eventpath=HERE/'inputs'/f'root{short}_B_S{n:02}_EVENT.json'
            save(eventpath,event)
            allocations.append({k:v for k,v in allocation.items() if k!='native_event'} | {
                'source_item_path':str(eventpath),'source_item_sha256':sha(eventpath),
                'status':'PERMANENTLY_RESERVED_BEFORE_AUTHORING',
                'selection_path':str(selection_path),'selection_sha256':sha(selection_path)})
    assert len(allocations)==12 and len({a['source_id'] for a in allocations})==12
    save(HERE/'SOURCE_RESERVATIONS.json',{
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'batch_id':HERE.name,
        'allocation':allocations,'permanent_including_failed_and_unexecuted_slots':True,
        'source_pool_path':plan['source_pool_path'],'source_pool_sha256':plan['source_pool_sha256'],
        'consumed_source_ids_before_this_batch':plan['consumed_source_ids']})
    feedback_path=HERE.parent/'h4_codex_scaled_production_v111/inputs/TRAINING_FEEDBACK.json'
    records=[{'root_id':r['root_id'],'candidate_count':r.get('candidate_count'),'aggregate':r.get('aggregate')}
             for r in read(feedback_path)['records'] if r['root_id'] in {r['root_id'] for r in plan['roots']}]
    save(HERE/'inputs/TRAINING_FEEDBACK.json',{
        'version':'v117_training_only_historical_aggregate_projection','upstream_path':str(feedback_path),
        'upstream_sha256':sha(feedback_path),'records':records,'root_order':[r['root_id'] for r in plan['roots']],
        'current_validation_feedback_used':False,
        'usage':'Training-side historical aggregate projection only; current roots and event choices fixed before this projection. No dev/test responses or current Solver feedback.'})
    cfg={k:v for k,v in plan.items() if k not in ('status','groups','primary_metrics')}
    cfg.update(controller_version='codex_sequential_ab_v1',route_roles={'A':'old_event_continuation','B':'dataset_new_event'},
        sequence='P+R01 -> accepted A1,A2,A3 -> accepted B1,B2,B3 -> one cumulative root',scenes_per_route_target=3,
        failure_policy='Stop dependent later scenes for failing root; no rewrite/replacement; retain failure denominator.',
        baseline_choice=plan['root_selection'],question_contract='Six-fact family, one fact per new scene, source-first reviewed true/false variants before Gold2/3/4/5 plus6control; two ordering slots with one anchor per new scene.',
        seed_namespace=HERE.name,multiselect_target_cardinalities=[2,3,4,5],gold6_control_per_ready_root=1,
        ordering_slots_per_ready_root=2,qwen_transport_used=False,codex_cli_used=False,solver_run=True,training_run=False,
        asset_review_required_before_composition=True,public_review_required_before_private_gold=True,
        semantic_review_is_programmatic=False,immutable_after_save=True,source_substitution_allowed=False,
        serialized_turn_metadata_required=True,lossless_exported_citations_required=True,
        multicitation_explicit_semantic_primary_required=True)
    save(HERE/'RUN_CONFIG.json',cfg)
    sys.path.insert(0,str(PROJECT/'scripts'))
    from codex_sequential_ab import load_config,load_baseline,reservation_for,SCENES
    checked=load_config(HERE)
    for root in plan['roots']:
        load_baseline(HERE,checked,root['root_id'])
        for scene in SCENES:
            if ':B:' in scene: reservation_for(HERE,checked,root['root_id'],scene)
    code=[]
    for name in ('codex_sequential_ab.py','codex_sequential_questions.py','codex_scene_payload.py','codex_controlled_batch.py'):
        source=PROJECT/'scripts'/name; target=HERE/'evidence/preexecution_code'/name
        copy(source,target);code.append({**bind(source),'snapshot_path':str(target),'snapshot_sha256':sha(target)})
    save(HERE/'evidence/PREEXECUTION_CODE_BINDINGS.json',{
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_DIALOGUE_AUTHORING',
        'files':code,'tooling_freeze':'Evaluation/lossless exporter tooling freezes before question authoring.'})
    files=[HERE/name for name in ('PRECALL_PLAN.json','RUN_CONFIG.json','SOURCE_RESERVATIONS.json',
        'EXPOSURE_AUDIT.json','evidence/CONTENT_CONTRACT.md','evidence/HISTORICAL_PROTECTION.json','evidence/PREEXECUTION_CODE_BINDINGS.json')]
    files+=sorted((HERE/'inputs').glob('*'))+sorted((HERE/'selection').glob('*'))
    files+=[PROJECT/'scripts'/name for name in ('codex_sequential_ab.py','codex_sequential_questions.py','codex_scene_payload.py','codex_controlled_batch.py')]
    save(HERE/'EXECUTION_INPUTS_FROZEN.json',{
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'READY_FOR_SERIALIZED_A3_THEN_B3',
        'files':[bind(p) for p in files],'root_denominator':4,'scene_denominator':24,'training_started':False,
        'controller_config_validation':'PASS_4_BASELINES_AND12_RESERVATIONS',
        'baseline_gate_bound':True,'all_continuation_constraints_bound':True})
    print('INPUTS_FROZEN',len(files),len(allocations))

if __name__=='__main__': main()
