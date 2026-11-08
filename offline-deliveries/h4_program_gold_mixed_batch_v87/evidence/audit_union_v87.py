"""Independent standard-library audit of new export and cumulative inventory."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys
import traceback

HOME = Path(__file__).resolve().parents[1]
PREVIOUS = HOME.parent / 'h4_reason_before_gold_mixed_batch_v86'
OUT = HOME / 'evidence/UNION_AUDIT.json'
sys.dont_write_bytecode = True
CHECKS, FILES = [], {}


def guard(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen','os.system','os.posix_spawn'}:
        raise RuntimeError('audit_forbids_external:' + event)
    if event == 'open' and isinstance(args[0], (str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve(); mode, flags=args[1:3]
        writing=((isinstance(mode,str) and any(c in mode for c in 'wax+')) or
                 (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing and path != OUT:
            raise RuntimeError('audit_forbids_artifact_write:' + str(path))


sys.addaudithook(guard)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path=Path(path).resolve(); row={'path':str(path),'sha256':sha(path)}
    FILES[str(path)]=row
    return row


def read(path):
    bind(path)
    return json.loads(Path(path).read_text())


def require(value, label, detail=None):
    row={'check':label,'status':'PASS' if value else 'FAIL'}
    if detail is not None:
        row['detail']=detail
    CHECKS.append(row)
    if not value:
        raise AssertionError(label)


def whole_set(q):
    field='options' if q['question_type']=='multi_select' else 'items'
    return (q['root_id'],q['question_type'],tuple(sorted(' '.join(t.split()).casefold() for t in q[field].values())))


def verify_named_binding(record, path, label):
    require(record==bind(path),label)


def same_counts(actual, expected):
    """Zero buckets may be written explicitly or omitted; no extra count passes."""
    return isinstance(actual, dict) and all(type(n) is int and n >= 0 for n in actual.values()) and {
        k:v for k,v in actual.items() if v
    } == {k:v for k,v in expected.items() if v}


def audit():
    result=read(HOME/'evidence/RESULT_AUDIT.json')
    generation=read(HOME/'evidence/GENERATION_AUDIT.json')
    require(result['status']=='PASS' and generation['status'] in ('PASS','PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS'), 'current_result_and_generation_audits_pass')
    require(read(PREVIOUS/'evidence/RESULT_AUDIT_v2.json')['status']=='PASS' and read(PREVIOUS/'evidence/UNION_AUDIT.json')['status']=='PASS', 'previous_result_and_union_audits_pass')
    config=read(HOME/'CONFIG.json')
    controls={s['parent_candidate_id'] for s in config['multiselect_config']['slots']}
    selected=read(HOME/'SELECTED_CANDIDATES.json')['questions']
    require(all(q['candidate_id'] not in controls or (q['question_type']=='multi_select' and len(q['gold'])==6) for q in selected), 'controls_only_historical_multiselect_gold6')
    expected_new=[q for q in selected if q['candidate_id'] not in controls]
    require([q['candidate_id'] for q in expected_new]==result['selected_new_ids'], 'export_expected_from_independent_policy_selection')
    current_path=HOME/'AUDITED_NEW_CANDIDATES.json'; current=read(current_path)
    require(current['questions']==expected_new and current['new_question_count']==len(expected_new), 'current_export_exact_new_selected_questions')
    require(current['historical_controls_excluded']==[q['candidate_id'] for q in selected if q['candidate_id'] in controls], 'selected_controls_excluded_from_export')
    verify_named_binding(current['independent_result_audit'],HOME/'evidence/RESULT_AUDIT.json','current_export_audit_binding')
    verify_named_binding(current['paired_report'],HOME/'MIXED_EVALUATION_REPORT.json','current_export_paired_report_binding')
    verify_named_binding(current['content_review'],HOME/'BATCH_REVIEW.COMPLETE.json','current_export_content_review_binding')
    require(all(current[k] is False for k in ('training_allowed','training_started','formal_training_admission','merged_into_baseline')), 'current_export_not_training_or_baseline_admission')
    previous_path=PREVIOUS/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json'; previous=read(previous_path)
    require(previous['candidate_count']==len(previous['questions'])==42, 'historical_cumulative_forty_two_preserved')
    require(Counter(q['question_type'] for q in previous['questions'])=={'multi_select':36,'ordering':6}
            and {str(g):sum(q['question_type']=='multi_select' and len(q['gold'])==g for q in previous['questions']) for g in range(2,6)}
            =={'2':9,'3':7,'4':9,'5':11}, 'historical_v86_type_and_multiselect_gold_inventory_preserved')
    sources=[previous_path,current_path]
    expected=[]; seen={}; duplicate_records=[]; expected_members=[]
    for path,data in ((previous_path,previous),(current_path,current)):
        require(data['training_allowed'] is False and data['formal_training_admission'] is False, 'source_pool_no_formal_admission:' + str(path))
        for q in data['questions']:
            require(q['candidate_id'] not in controls and not (q['question_type']=='multi_select' and len(q['gold'])==6), 'source_pool_has_no_historical_controls:' + q['candidate_id'])
            key=whole_set(q)
            if key in seen:
                duplicate_records.append({'candidate_id':q['candidate_id'],'matches':seen[key]})
                continue
            seen[key]=q['candidate_id']; expected.append(q)
            expected_members.append({'candidate_id':q['candidate_id'],'source':str(path),'root_id':q['root_id'],'gold_count':len(q['gold'])})
    cumulative=read(HOME/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json')
    require(not duplicate_records and len({q['candidate_id'] for q in expected})==len(expected), 'no_duplicate_whole_set_or_identity_in_union')
    require(cumulative['questions']==expected and cumulative['candidate_count']==len(expected) and cumulative['duplicates_removed']==duplicate_records, 'cumulative_exact_previous_plus_current')
    require(cumulative['source_pools']==[bind(p) for p in sources], 'cumulative_exact_source_bindings')
    require(cumulative['members']==expected_members, 'cumulative_member_provenance_preserved')
    for record in cumulative['independent_batch_audits']:
        require(read(record['path'])['status']=='PASS' and bind(record['path'])==record, 'cumulative_batch_audit_binding:' + record['path'])
    require(any(Path(r['path']).resolve()==HOME/'evidence/RESULT_AUDIT.json' for r in cumulative['independent_batch_audits']), 'cumulative_includes_current_result_audit')
    require(all(cumulative[k] is False for k in ('training_allowed','training_started','formal_training_admission','merged_into_baseline')), 'cumulative_no_training_or_baseline_admission')
    expected_gold={str(g):sum(q['question_type']=='multi_select' and len(q['gold'])==g for q in expected) for g in range(2,6)}
    expected_type=dict(Counter(q['question_type'] for q in expected))
    require(same_counts(cumulative['gold_distribution'],expected_gold), 'cumulative_gold_distribution_multiselect_only')
    require(same_counts(cumulative['type_distribution'],expected_type), 'cumulative_type_distribution')
    require(cumulative['root_distribution']==dict(Counter(q['root_id'] for q in expected)), 'cumulative_root_distribution')
    new_gold={str(g):sum(q['question_type']=='multi_select' and len(q['gold'])==g for q in expected_new) for g in range(2,6)}
    require(same_counts(current['gold_distribution'],new_gold) and same_counts(current['type_distribution'],dict(Counter(q['question_type'] for q in expected_new))), 'current_export_type_and_multi_gold_counts')
    baseline=read(config['baseline_pool'])['questions']; baseline_keys={whole_set(q) for q in baseline}
    overlap=[q['candidate_id'] for q in expected if whole_set(q) in baseline_keys]
    require(not overlap,'no_cumulative_whole_set_overlap_with_baseline',overlap)
    protected=read(HOME/'PRESERVATION_BEFORE.json')
    require(len(protected['files'])==10226, 'union_all_10226_previous_files_protected')
    for path,digest in protected['files'].items():
        require(sha(path)==digest,'previous_file_preserved:' + path)
    return {'status':'PASS','current_new_count':len(expected_new),'previous_cumulative_count':len(previous['questions']),
        'cumulative_count':len(expected),'type_distribution':expected_type,'multiselect_gold_distribution':expected_gold,
        'root_distribution':dict(Counter(q['root_id'] for q in expected)),'whole_set_duplicates_removed':duplicate_records,
        'historical_controls_in_union':0,'baseline_whole_set_overlap':overlap,'protected_previous_file_count':len(protected['files']),
        'limitations':['Whole-set novelty does not certify new independent facts or semantic novelty.',
            'This inventory neither recertifies the baseline nor admits candidates to training.',
            'Ordering has six ordered items; it is not included in multiselect Gold-count statistics.']}


def main():
    if OUT.exists():
        raise RuntimeError('refuse_overwrite:' + str(OUT))
    result={'version':'v87_independent_export_union_audit_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'new_model_calls':0,'new_http_calls':0,'training_performed':False,'frozen_artifacts_modified':False,
        'semantic_recertification_claimed':False}
    try:
        result.update(audit())
    except BaseException as error:
        result['status']='FAIL'
        CHECKS.append({'check':'audit_execution_exception','status':'FAIL','detail':{'error':repr(error),'traceback':traceback.format_exc()}})
    result['checks']=CHECKS;result['check_count']=len(CHECKS)
    result['failed_checks']=[r for r in CHECKS if r['status']=='FAIL']
    result['files']=list(FILES.values())+[bind(__file__)]
    with OUT.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'path':str(OUT),'status':result['status'],'checks':len(CHECKS),'failed_checks':result['failed_checks']},ensure_ascii=False))
    if result['status']!='PASS':
        raise SystemExit(1)


if __name__=='__main__':
    main()
