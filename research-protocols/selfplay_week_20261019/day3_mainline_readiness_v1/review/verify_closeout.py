#!/usr/bin/env python3
"""Read-only checks of bounded Day3 readiness deliverables; writes audit only."""
import json, hashlib, datetime
from pathlib import Path
RUN = Path(__file__).resolve().parents[1]
WEEK = RUN.parent
CHECKS = []
def read(path):
    return json.loads(Path(path).read_text())
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(name, passed, details=None):
    CHECKS.append(dict(name=name, passed=bool(passed), details=details))
def main():
    freeze = read(RUN/'EXECUTION_FREEZE.json')
    for binding in freeze['input_bindings']:
        path = Path(binding['path'])
        if path.name == 'CURRENT_STATUS.json':
            path = RUN/'navigation_history/CURRENT_STATUS.json'
        check('frozen_input:'+binding['path'], path.exists() and digest(path)==binding['sha256'] and path.stat().st_size==binding['bytes'])
    outputs = sorted(set(path for stream in freeze['streams'] for path in stream['outputs']))
    for path in outputs:
        p = Path(path)
        check('required_output:'+p.name, p.is_file())
        if not p.is_file():
            continue
        if p.suffix == '.json':
            read(p)
        elif p.suffix == '.jsonl':
            for line in p.read_text().splitlines():
                if line.strip():
                    json.loads(line)
    primary = read(RUN/'review/PRIMARY_INVENTORY_CHECK.json')
    check('content187_inventory', primary['content_accepted_count']==187 and primary['public_private_ids_match'])
    check('unique88_pairs', primary['pairs_count']==primary['unique_pair_slots']==88)
    check('all_pair_basics_match', all(all(v for v in row.values() if isinstance(v,bool)) for row in primary['pair_inventory_checks']))
    coverage = read(RUN/'review/CONTENT187_COVERAGE_VERIFICATION.json')
    check('both_arm_minimum_diversity', all(x['minimum_two_MC_classes'] and x['minimum_two_ordering_profiles'] and x['all_record_bindings_pass'] and x['all_qa_pass'] and x['all_targets_implemented'] for x in coverage['summary'].values()))
    amendment = read(RUN/'review/EXISTING_SCOPE_AMENDMENT_VERIFICATION.json')
    check('existing_prospective50pct_amendment_bound', all(amendment['checks'].values()))
    admission_path = RUN/'common_eval/HISTORICAL184_ADMISSION.json'
    if admission_path.exists():
        admission = read(admission_path)
        expected = next(s for s in freeze['streams'] if s['id']=='CE-184')['fixed_candidate_ids']
        ids = [r['candidate_id'] for r in admission['records']]
        check('historical184_exact_ids', ids==expected and len(set(ids))==184)
        check('historical184_concrete_dispositions', all(r['status'] in ['PASS','HOLD','EXCLUDE'] and r['reason'] for r in admission['records']))
        check('historical_controls27_separate', sum(r['reporting_role']=='GOLD6_CONTROL_ONLY' for r in admission['records'])==27)
        check('historical_all_bound_checks_pass', all(all(r['checks'].values()) for r in admission['records']))
        from collections import Counter
        check('historical_disposition_counts_reconcile', dict(Counter(r['status'] for r in admission['records']))==admission['counts'])
        for binding in admission.get('evidence_bindings',[]):
            path = Path(binding['path'])
            check('historical_evidence_binding:'+path.name, path.is_file() and digest(path)==binding['sha256'])
    order_path = RUN/'common_eval/ORDERING_FOUR_PLAN_ADDENDUM.json'
    if order_path.exists():
        ordering=read(order_path)
        expected=[p['plan_id'] for p in next(s for s in freeze['streams'] if s['id']=='CE-ORDER-4')['fixed_plans']]
        check('ordering_exact4_fixed_plans', [p['plan_id'] for p in ordering['plans']]==expected)
        check('ordering_all_stopped_with_specific_overlap', all(p['disposition']=='EXCLUDE' and p['decisive_record_id'] and p['decisive_necessary_shared_atom'] for p in ordering['plans']))
    for name in ['TRAIN_POOL_A.jsonl','TRAIN_POOL_B.jsonl']:
        path=RUN/'training'/name
        check('final_unadmitted_pool_empty:'+name, path.exists() and path.stat().st_size==0)
    budget_path=RUN/'training/TRAINING_BUDGET_FREEZE.json'
    if budget_path.exists():
        budget=read(budget_path)
        check('current_mix_policy_explicit', budget['current_protocol_mix_policy']['candidate_mix_target_if_q0']=={'new_structure':0.75,'audited_legacy':0.0,'existing_round2_foundation':0.25})
        check('final_schedule_not_vacuous_pass', budget['final_budget']['final_schedule_validated'] is False and budget['final_budget']['input_difference_le5percent'] is None)
    upstream = read(RUN/'review/ALL_UPSTREAM_MANIFESTS_VERIFICATION.json')
    for group in upstream['groups']:
        if not group.get('manifest_path'):
            continue
        mp = Path(group['manifest_path'])
        check('upstream_manifest:'+group['directory'], digest(mp)==group['manifest_sha256'])
        for binding in read(mp)['files']:
            path = Path(binding['path'])
            path = path if path.is_absolute() else mp.parent/path
            check('upstream_file:'+str(path), path.is_file() and digest(path)==binding['sha256'] and ('bytes' not in binding or path.stat().st_size==binding['bytes']))
    decision_path = RUN/'READINESS_DECISION.json'
    if decision_path.exists():
        decision = read(decision_path)
        check('no_training_start', decision.get('training_start_go') is False)
        check('no_false_full_day_completion', decision.get('full_day_complete') is False)
        check('zero_new_solver_calls', decision.get('new_solver_calls')==0)
        check('zero_new_training_runs', decision.get('new_training_runs')==0)
    result = dict(schema_version='day3_mainline_closeout_technical_check_v1', checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), checks_total=len(CHECKS), checks_passed=sum(x['passed'] for x in CHECKS), all_pass=all(x['passed'] for x in CHECKS), failures=[x for x in CHECKS if not x['passed']], checks=CHECKS, scope='Technical completeness and immutable input/source binding; separate semantic independent audit required.')
    (RUN/'review/TECHNICAL_CLOSEOUT_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
    raise SystemExit(0 if result['all_pass'] else 1)
if __name__ == '__main__':
    main()
