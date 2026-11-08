#!/usr/bin/env python3
"""Validate extraction, pair controls, provenance and unchanged Train upstream."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
WEEK = OUT.parent


def load(name):
    return json.loads((OUT/name).read_text())


def lines(name):
    return [json.loads(x) for x in (OUT/name).read_text().splitlines() if x]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []
    def check(name, ok, detail=None):
        checks.append({'name': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})
    f = lines('MATERIAL_SHORTFALLS.jsonl')
    p = lines('PAIRED_MATERIAL_TARGETS.jsonl')
    r = lines('ROOT_EXTENSION_REQUESTS.jsonl')
    by_f = {x['feedback_id']: x for x in f}
    by_p = {x['material_target_id']: x for x in p}
    upstream_rows = []
    for d in ('day3_production_v1','day3_production_batch2_v1'):
        upstream_rows.extend(json.loads((WEEK/d/'exports/SLOT_LEDGER.json').read_text())['rows'])
    true_short = {x.get('candidate_slot_id',x.get('candidate_id')) for x in upstream_rows if x['state']=='SHORTFALL'}
    check('42_exact_original_shortfalls_no_hold_or_generation_hold', len(f)==42 and {x['candidate_slot_id'] for x in f} == true_short)
    check('unique_feedback_ids', len(by_f)==len(f))
    check('40_joint_2_revision', Counter(x['gap_category'] for x in f)==Counter({'NONREDUNDANT_CONDITIONAL_JOINT_RELATION':40,'EXPLICIT_SAME_OBJECT_REVISION_CHAIN':2}))
    check('all_train_roots', all(x['root_id'].startswith('cc-train-episode-') for x in f))
    check('21_matched_targets_20_merged_roots', len(p)==21 and len(r)==20 and len({x['root_id'] for x in r})==20)
    for x in f:
        for key in ('source_binding','ledger_binding','original_shortfall_binding'):
            b=x[key]
            check(x['feedback_id']+'/'+key, digest(b['path'])==b['sha256'])
        ledger = json.loads(Path(x['ledger_binding']['path']).read_text())
        row=ledger['rows'][int(x['ledger_binding']['json_pointer'].split('/')[-1])]
        check(x['feedback_id']+'/original_row', row['state']=='SHORTFALL' and row['root_id']==x['root_id'] and row['arm']==x['arm'] and row['slot_id']==x['slot_id'] and row['target_gold_count']==x['target_gold_count_provenance_only'])
        check(x['feedback_id']+'/exact_reason_record', json.loads(Path(x['original_shortfall_binding']['path']).read_text())==x['original_shortfall'])
        check(x['feedback_id']+'/not_solver_weakness', x['solver_weakness_evidence'] is False and not x['solver_response_bindings'] and x['formal_B_targeted_quota_eligible_from_this_feedback'] is False and x['current_frozen_slot_remains_shortfall'] is True)
        source=json.loads(Path(x['source_binding']['path']).read_text())
        turn_lookup={t['turn_ref']:t['text'] for t in source['turns']}
        for i,a in enumerate(x['recorded_source_anchors']):
            check(x['feedback_id']+'/anchor'+str(i), a['turn_ref'] in turn_lookup and a['quote'] in turn_lookup[a['turn_ref']])
    for x in p:
        refs=[by_f[k] for k in x['feedback_ids']]
        check(x['material_target_id']+'/two_arms_same_slot_source_and_target', len(refs)==2 and {y['arm'] for y in refs}=={'A','B'} and {y['slot_id'] for y in refs}=={x['original_slot_id']} and {y['source_binding']['sha256'] for y in refs}=={x['source_binding']['sha256']} and {y['target_relation_id'] for y in refs}=={x['target_relation_id']} and {y['target_gold_count_provenance_only'] for y in refs}=={x['target_gold_count_provenance_only']})
    claimed=[]
    for x in r:
        targets=[by_p[k] for k in x['material_target_ids']]
        claimed.extend(x['material_target_ids'])
        check(x['request_id']+'/targets_merge_same_root', all(y['root_id']==x['root_id'] for y in targets) and set(x['feedback_ids'])=={k for y in targets for k in y['feedback_ids']})
        layout=x['extension_layout']
        check(x['request_id']+'/two_routes_three_scenes', layout['route_ids']==['A','B'] and layout['scenes_per_route']==3 and layout['scenes_total_if_executed']==6 and layout['route_AB_is_distinct_from_question_arm_AB'] is True)
        check(x['request_id']+'/future_only_shared_source', x['common_to_question_generation_arms']==['A','B'] and x['generation_executed'] is False and x['current_day3_outputs_changed'] is False and x['status']=='READY_FOR_PLANNING_NOT_EXECUTION_FROZEN')
    check('each_paired_target_claimed_once', Counter(claimed)==Counter(by_p.keys()))
    for b in load('READ_MANIFEST.json')['content_inputs']:
        path=Path(b['path'])
        check('input_unchanged/'+str(path), digest(path)==b['sha256'])
    seal=load('UPSTREAM_SEAL.json')['files']
    oldpaths={x['path'] for x in seal}
    current={str(z) for d in ('day3_production_v1','day3_production_batch2_v1') for z in (WEEK/d).rglob('*') if z.is_file()}
    check('upstream_file_set_unchanged', oldpaths==current)
    for b in seal:
        check('upstream_hash/'+str(Path(b['path']).relative_to(WEEK)), digest(b['path'])==b['sha256'])
    check('prompt_is_present', (OUT/'PROMPT_TEMPLATE_ZH.md').is_file())
    def forbidden_author_fields(value):
        forbidden=[]
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('gold','gold_count','target_gold_count','target_gold_count_provenance_only','answer','answers','answer_labels'):
                    forbidden.append(k)
                forbidden.extend(forbidden_author_fields(v))
        elif isinstance(value,list):
            for v in value:
                forbidden.extend(forbidden_author_fields(v))
        return forbidden
    check('author_root_requests_contain_no_answer_or_gold_fields',not forbidden_author_fields(r))
    inherited=load('INHERITED_ROUTE_POLICY.json')
    check('exact_inherited_route_roles_and_sequence',inherited['route_roles']=={'A':'old_event_continuation','B':'dataset_new_event'} and inherited['historical_sequence']=='P+R01 -> accepted A1,A2,A3 -> accepted B1,B2,B3 -> one cumulative root' and inherited['scenes_per_route_target']==3)
    for b in inherited['authoritative_bindings']:
        check('inherited_route_binding/'+b['path'], digest(b['path'])==b['sha256'])
    policy=load('POLICY.json')
    check('no_executed_generation_solver_or_training', all(policy[k]==0 for k in ('executed_dialogue_extensions','executed_new_questions','solver_calls','training_runs')))
    failed=[x for x in checks if x['status']=='FAIL']
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if not failed else 'FAIL', 'checks':len(checks),'failures':failed,'unchanged_upstream_files':len(seal),'shortfall_rows':len(f),'paired_targets':len(p),'root_requests':len(r),'mechanical_limits':'Checks provenance/counts/pair controls/quote existence; does not certify future dialogue naturalness, relation truth, training value or semantic independence.','details':checks}
    (OUT/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    summary=load('SUMMARY.json')
    summary['validation']={'path':str(OUT/'VALIDATION.json'),'sha256':digest(OUT/'VALIDATION.json'),'status':result['status'],'checks':len(checks),'unchanged_upstream_files':len(seal)}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'All current package files except this manifest itself. Prior checker defect retained as history, not an upstream content failure.','files':[{'path':str(z),'sha256':digest(z),'bytes':z.stat().st_size} for z in sorted(OUT.iterdir()) if z.is_file() and z.name!='MANIFEST.json']}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
