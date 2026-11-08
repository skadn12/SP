"""Publish the completed pre-expansion screen, without starting generation."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT/'scripts'))
from codex_baseline_screen import finalize_screen

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(data,f,ensure_ascii=False,indent=2); f.write('\n')
def bind(path): return {'path': str(path.resolve()), 'sha256': sha(path)}

def main():
    plan = read(HERE/'SCREEN_PLAN.json')
    tools = [PROJECT/'scripts/codex_baseline_screen.py', PROJECT/'tests/test_codex_baseline_screen.py', Path(__file__)]
    snapshots = []
    for tool in tools:
        target = HERE/'evidence/tooling_code'/tool.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output: output.write(tool.read_bytes())
        snapshots.append({**bind(tool), 'snapshot': bind(target)})
    save(HERE/'TOOLING_FREEZE.json', {'created_at_utc':now(), 'status':'FROZEN_BEFORE_AGGREGATION', 'files':snapshots})
    paths = [HERE/'reviews'/('root'+rid.rsplit('-',1)[1])/(role+'.json') for rid in plan['candidate_order'] for role in ['primary','secondary']]
    # The gate validates every review before creating either final output.
    finalize_screen(HERE/'SCREEN_PLAN.json', paths, HERE)
    results, selected = read(HERE/'SCREEN_RESULTS.json'), read(HERE/'SELECTED_ROOTS.json')
    save(HERE/'REVIEW_COMPARISON.json', {
        'created_at_utc':now(), 'scope':'Descriptive same-model reader comparison, not external semantic certification',
        'roots': [{'root_id':rid, 'decisions':[read(p)['decision'] for p in paths if read(p)['root_id']==rid],
                   'different_overall_decisions':len({read(p)['decision'] for p in paths if read(p)['root_id']==rid})>1,
                   'admission_disagreement':len({read(p)['decision']=='HOLD' for p in paths if read(p)['root_id']==rid})>1}
                  for rid in plan['candidate_order']]})
    bound = [bind(HERE/name) for name in ['SCREEN_PLAN.json','BASELINE_POLICY.md','SCREEN_RESULTS.json','SELECTED_ROOTS.json','REVIEW_COMPARISON.json','TOOLING_FREEZE.json']]
    save(HERE/'NEXT_BATCH_HANDOFF.json', {
        'created_at_utc':now(), 'status':'BASELINE_GATE_COMPLETED_ROOT_SELECTION_FIXED_GENERATION_NOT_STARTED',
        'screen_bindings':bound, 'selection_authority':'SELECTED_ROOTS.json',
        'generation_root_replacement_allowed':False,
        'required_rules':{'sequence':['R02:A:S01','R02:A:S02','R02:A:S03','R02:B:S01','R02:B:S02','R02:B:S03'],
            'turns_per_scene':{'min':12,'max':16}, 'independent_full_source_ready_before_next_scene':True,
            'author_attempts_per_scene':1,'semantic_repairs':0,'source_replacements_after_reservation':0,
            'bind_all_selected_root_continuation_constraints':True,
            'new_speaking_characters':False, 'question_gold_targets':[2,3,4,5],
            'gold6_control_per_complete_root':1, 'ordering_per_complete_root':2,
            'source_first_asset_and_public_question_reviews':True,'lossless_evidence_export':True,
            'solver_protocol':'v15_full_only_random_mcq6x3_order12x1_v1',
            'models':['M0','M_clean_policy'], 'solver_automatic_retries':False,
            'selection':'complete-valid 0 < mean_partial < 1; exclude Gold6 control'},
        'source_event_selection':'Not started; rescan all existing reservations including v115 before event allocation. Preserve permanent reservations and full native-event reads.',
        'held_roots':'Retain full original and both reviews; do not replace or rewrite to rescue this screen.',
        'existing_training_exposure':True,'unseen_holdout_claimed':False,
        'generation_started':False,'solver_started':False,'training_started':False,'pool_merge_started':False})
    # Verify sealed history after all review work and aggregation.
    protected = read(HERE/'HISTORICAL_PROTECTION.json')['files']
    for row in protected:
        assert sha(row['path'])==row['sha256'], row['path']
    save(HERE/'HISTORICAL_AUDIT.json', {'created_at_utc':now(),'status':'PASS','errors':[],
        'verified_file_count':len(protected),'protection_binding':bind(HERE/'HISTORICAL_PROTECTION.json')})
    print('SCREEN_AGGREGATED', json.dumps({'result_keys':list(results),'selected_keys':list(selected)}))

if __name__=='__main__': main()
