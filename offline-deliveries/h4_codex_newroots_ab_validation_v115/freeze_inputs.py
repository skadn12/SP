"""Freeze reviewed allocations before any v115 dialogue authoring."""
from bootstrap import HERE, P, read, sha, write, copy
from datetime import datetime, timezone
import sys

stamp = datetime.now(timezone.utc).isoformat()
plan = read(HERE / 'PRECALL_PLAN.json')
roots = plan['roots']
allocations = []
for root in roots:
    rid = root['root_id']; short = rid.rsplit('-', 1)[1]
    selected = read('/tmp/v115_selection_' + short + '.json')
    baseline = read(root['baseline_path'])
    assert selected['root_id'] == rid and selected['full_baseline_read'] is True
    assert selected['baseline_sha256'] == root['baseline_sha256']
    assert selected['baseline_turn_refs_read'] == [t['turn_ref'] for t in baseline['turns']]
    poolp = HERE / 'pre_inputs' / ('root' + short + '_EVENT_POOL.json')
    pool = {e['dialogue_id']: e for e in read(poolp)['events']}
    assert selected['partition']['event_pool_sha256'] == sha(poolp)
    assert len(selected['allocations']) == 3
    sp = HERE / 'selection' / ('root' + short + '.json')
    copy(__import__('pathlib').Path('/tmp/v115_selection_' + short + '.json'), sp)
    for n, allocation in enumerate(selected['allocations'], 1):
        assert allocation['root_id'] == rid and allocation['scene_ref'] == f'R02:B:S{n:02}'
        eid = allocation['source_id'].removeprefix('multiwoz22:train:')
        event = pool[eid]
        assert allocation['source_id'] not in plan['consumed_source_ids']
        assert allocation['full_event_utterances_read'] is True
        assert allocation['native_turn_ids_read'] == [t['turn_id'] for t in event['turns']]
        ep = HERE / 'inputs' / f'root{short}_B_S{n:02}_EVENT.json'
        write(ep, event)
        allocations.append({k:v for k,v in allocation.items() if k != 'native_event'} | {
            'source_item_path': str(ep), 'source_item_sha256': sha(ep),
            'status':'PERMANENTLY_RESERVED_BEFORE_AUTHORING',
            'selection_path': str(sp), 'selection_sha256':sha(sp)})
assert len({a['source_id'] for a in allocations}) == 12
write(HERE / 'SOURCE_RESERVATIONS.json', {
    'created_at_utc': stamp, 'batch_id': HERE.name, 'allocation': allocations,
    'permanent_including_failed_and_unexecuted_slots': True,
    'source_pool_path': plan['source_pool_path'], 'source_pool_sha256': plan['source_pool_sha256'],
    'consumed_source_ids_before_this_batch': plan['consumed_source_ids']})
feedback_path = HERE.parent / 'h4_codex_scaled_production_v111/inputs/TRAINING_FEEDBACK.json'
oldfb=read(feedback_path)
records=[{'root_id':item['root_id'],'candidate_count':item.get('candidate_count'),'aggregate':item.get('aggregate')}
         for item in oldfb['records'] if item['root_id'] in {r['root_id'] for r in roots}]
write(HERE/'inputs/TRAINING_FEEDBACK.json', {
    'version':'v115_training_only_historical_aggregate_projection',
    'upstream_path':str(feedback_path),'upstream_sha256':sha(feedback_path),'records':records,
    'root_order':[r['root_id'] for r in roots], 'current_validation_feedback_used':False,
    'usage':'Training-side historical aggregates only; source/root choices already fixed. No dev/test responses or current Solver feedback.'})
cfg={k:v for k,v in plan.items() if k not in ['status','groups','primary_metrics']}
cfg.update(controller_version='codex_sequential_ab_v1',route_roles={'A':'old_event_continuation','B':'dataset_new_event'},
    sequence='P+R01 -> accepted A1,A2,A3 -> accepted B1,B2,B3 -> one cumulative root', scenes_per_route_target=3,
    failure_policy='Stop dependent later scenes for failing root; no rewrite/replacement; retain failure denominator.',
    baseline_choice=plan['root_selection'], question_contract='Six-fact family, one fact per new scene, source-first reviewed true/false variants before Gold2/3/4/5 plus6control; two ordering slots with one anchor per new scene.',
    seed_namespace=HERE.name,multiselect_target_cardinalities=[2,3,4,5],gold6_control_per_ready_root=1,
    ordering_slots_per_ready_root=2,qwen_transport_used=False,codex_cli_used=False,solver_run=True,training_run=False,
    asset_review_required_before_composition=True,public_review_required_before_private_gold=True,
    semantic_review_is_programmatic=False,immutable_after_save=True,source_substitution_allowed=False,
    serialized_turn_metadata_required=True,lossless_exported_citations_required=True,
    multicitation_explicit_semantic_primary_required=True)
write(HERE/'RUN_CONFIG.json',cfg)
sys.path.insert(0,str(P/'scripts'))
from codex_sequential_ab import load_config,load_baseline,reservation_for,SCENES
checked=load_config(HERE)
for r in roots:
    load_baseline(HERE,checked,r['root_id'])
    for scene in SCENES:
        if ':B:' in scene:reservation_for(HERE,checked,r['root_id'],scene)
code=[]
for name in ['codex_sequential_ab.py','codex_sequential_questions.py','codex_scene_payload.py','codex_controlled_batch.py']:
    p=P/'scripts'/name;dst=HERE/'evidence/preexecution_code'/name;copy(p,dst)
    code.append({'path':str(p),'sha256':sha(p),'snapshot_path':str(dst),'snapshot_sha256':sha(dst)})
write(HERE/'evidence/PREEXECUTION_CODE_BINDINGS.json',{'created_at_utc':stamp,'status':'FROZEN_BEFORE_DIALOGUE_AUTHORING','files':code,'tooling_freeze':'Evaluation/lossless exporter tooling freezes before question authoring.'})
files=[HERE/x for x in ['PRECALL_PLAN.json','RUN_CONFIG.json','SOURCE_RESERVATIONS.json','EXPOSURE_AUDIT.json','evidence/CONTENT_CONTRACT.md','evidence/HISTORICAL_PROTECTION.json','evidence/PREEXECUTION_CODE_BINDINGS.json']]
files += sorted((HERE/'inputs').glob('*')) + sorted((HERE/'selection').glob('*')) + [P/'scripts'/n for n in ['codex_sequential_ab.py','codex_sequential_questions.py','codex_scene_payload.py','codex_controlled_batch.py']]
write(HERE/'EXECUTION_INPUTS_FROZEN.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'READY_FOR_SERIALIZED_A3_THEN_B3','files':[{'path':str(p),'sha256':sha(p)} for p in files], 'root_denominator':4,'scene_denominator':24,'training_started':False,'controller_config_validation':'PASS_4_BASELINES_AND12_RESERVATIONS'})
print('INPUTS_FROZEN',len(files),len(allocations))
