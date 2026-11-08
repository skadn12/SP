"""CPU-only bound summary of actual frozen Train diagnostic requests/results."""
from __future__ import annotations
import collections
from fractions import Fraction
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_prepare as b
import run_v2 as rt

EXPOSURES = ('ALL', 'OWN_ARM_TRAINED_QUESTION', 'OPPOSITE_ARM_QUESTION_SHARED_ROOT_MAY_SHARE_FACTS')
TYPES = ('ALL', 'multi_select', 'ordering')


def fraction(v):
    return {'fraction': str(v), 'value': float(v)}


def direct_groups(model, exposure_role):
    result = {}
    for typ in TYPES:
        for exposure in EXPOSURES:
            qs = [q for q in model['questions'] if (typ == 'ALL' or q['question_type'] == typ)
                and (exposure == 'ALL' or q['model_training_exposure'][exposure_role] == exposure)]
            byroot = collections.defaultdict(list)
            for q in qs:
                byroot[q['root_id']].append(Fraction(q['exact_calls'], q['denominator_calls']))
            question_equal = sum((Fraction(q['exact_calls'], q['denominator_calls']) for q in qs), Fraction()) / len(qs)
            root_equal = sum((sum(v, Fraction()) / len(v) for v in byroot.values()), Fraction()) / len(byroot)
            planned = sum(q['denominator_calls'] for q in qs)
            valid = sum(q['valid_calls'] for q in qs)
            exact = sum(q['exact_calls'] for q in qs)
            result[typ + '/' + exposure] = {'questions':len(qs), 'roots':len(byroot), 'planned_calls':planned,
                'valid_calls':valid, 'invalid_calls':planned-valid, 'exact_calls':exact,
                'question_equal_exact':fraction(question_equal), 'root_equal_exact':fraction(root_equal),
                'call_pooled_exact_secondary':fraction(Fraction(exact, planned))}
    return result


def brief_groups(readout, exposure_role):
    result = {}
    for typ in TYPES:
        for exposure in EXPOSURES:
            qs = [q for q in readout['questions'] if (typ == 'ALL' or q['question_type'] == typ)
                and (exposure == 'ALL' or q['model_training_exposure'][exposure_role] == exposure)]
            valid = sum(q['schema_valid'] for q in qs)
            exact = sum(q['diagnostic_final_exact_fixed_denominator'] for q in qs)
            result[typ + '/' + exposure] = {'planned_questions':len(qs), 'roots':len({q['root_id'] for q in qs}),
                'strict_valid':valid, 'strict_invalid':len(qs)-valid, 'strict_valid_items':6*valid,
                'final_exact_fixed_denominator':exact, 'final_wrong_among_strict_valid':valid-exact,
                'strict_valid_rate':fraction(Fraction(valid,len(qs))),
                'final_exact_fixed_denominator_rate':fraction(Fraction(exact,len(qs)))}
    return result


def branch_usage(role, branch):
    run = HERE / 'models' / role / branch
    freeze = b.read(run/'RAW_RESPONSE_FREEZE.json')
    rt.check(freeze['files'] + [freeze['request_freeze'],freeze['execution_manifest']])
    complete = b.read(run/'COMPLETE.json')
    assert complete['status']=='COMPLETE' and complete['failure'] is None
    rows = [b.read(r['path']) for r in freeze['files']]
    assert len(rows)==complete['actual_submitted_requests']==complete['returned_requests']
    for row in rows:
        assert row['status']=='RETURNED' and not row['output_validation_errors']
        assert row['metadata']['prompt_token_count']==len(row['prompt_token_ids'])
        assert row['metadata']['completion_token_count']==len(row['all_completions'][0]['token_ids'])
    return {'planned_requests':freeze['planned_request_count'],
        'actual_submitted_requests':complete['actual_submitted_requests'], 'returned_requests':complete['returned_requests'],
        'input_tokens':sum(r['metadata']['prompt_token_count'] for r in rows),
        'output_tokens':sum(r['metadata']['completion_token_count'] for r in rows),
        'wall_seconds':complete['wall_seconds'], 'model_load_wall_seconds':complete['model_load_wall_seconds'],
        'wall_scope':'Actual branch executor COMPLETE wall, including model load, request submission, raw commits and close; not manual review or orchestration gaps.',
        'complete':b.binding(run/'COMPLETE.json'), 'raw_freeze':b.binding(run/'RAW_RESPONSE_FREEZE.json'),
        'started':b.binding(run/'STARTED.json')}


def main():
    rt.verify()
    models, usages = {}, []
    baseline = b.read(HERE/'private/S0_BASELINE_BINDINGS.json')
    rt.check(baseline['selected_raw_responses'] + [baseline['gold_binding'], baseline['manifest'], baseline['raw_response_freeze']])
    for role in ('A1','B1'):
        run=HERE/'models'/role
        scores=b.read(run/'private/DIRECT_SCORES.json')
        readout=b.read(run/'private/SEMANTIC_READOUT.json')
        evidence=b.read(run/'private'/('A1_SEMANTIC_AUDIT.json' if role=='A1' else 'B1_SEMANTIC_AUDIT_V2.json'))
        groups=direct_groups(scores['models'][role],role)
        for name,group in groups.items():
            assert abs(group['question_equal_exact']['value']-scores['models'][role]['groups'][name]['question_equal_exact'])<1e-14
        usage={branch:branch_usage(role,branch) for branch in ('direct','brief_evidence')}
        usages.extend(usage.values())
        models[role]={'endpoint':b.binding(run/'EXECUTION_MANIFEST.json'),
            'direct_scores':b.binding(run/'private/DIRECT_SCORES.json'),'direct':groups,
            'brief_evidence_mechanical_readout':b.binding(run/'private/SEMANTIC_READOUT.json'),
            'brief_evidence_secondary_not_primary':brief_groups(readout,role),
            'source_semantic_audit':b.binding(run/'private'/('A1_SEMANTIC_AUDIT.json' if role=='A1' else 'B1_SEMANTIC_AUDIT_V2.json')),
            'source_semantic_dimensions':evidence['aggregate'] if role=='A1' else evidence['separate_error_dimensions'],
            'usage':usage}
    a=b.read(HERE/'models/A1/private/DIRECT_SCORES.json')
    z=b.read(HERE/'models/B1/private/DIRECT_SCORES.json')
    assert a['models']['S0']['requests']==z['models']['S0']['requests']
    oldrows=[b.read(x['path']) for x in baseline['selected_raw_responses']]
    assert len(oldrows)==360 and len({r['request_id'] for r in oldrows})==360
    s0={'status':'REUSED_EXACT_OLD360_RESPONSES_NO_NEW_S0_REQUESTS',
        'raw_binding_packet':b.binding(HERE/'private/S0_BASELINE_BINDINGS.json'),
        'reused_direct_responses':360, 'new_calls':0, 'brief_evidence_same_prompt_baseline':'UNAVAILABLE_NOT_RECOMPUTED',
        'direct_groups_relative_to_A1':direct_groups(a['models']['S0'],'A1'),
        'direct_groups_relative_to_B1':direct_groups(a['models']['S0'],'B1'),
        'historical_reused_input_tokens':sum(r['metadata']['prompt_token_count'] for r in oldrows),
        'historical_reused_output_tokens':sum(r['metadata']['completion_token_count'] for r in oldrows),
        'historical_selected360_wall_seconds':None,
        'usage_note':'Historical token counts identify reused payloads, excluded from all new Day5 usage. Selected subset wall cannot be attributed from whole historical batch wall.'}
    usage={'new_model_requests':sum(u['actual_submitted_requests'] for u in usages),
        'new_returned_requests':sum(u['returned_requests'] for u in usages),
        'direct_requests':720, 'brief_evidence_requests':48,
        'new_input_tokens':sum(u['input_tokens'] for u in usages),
        'new_output_tokens':sum(u['output_tokens'] for u in usages),
        'summed_actual_branch_wall_seconds':sum(u['wall_seconds'] for u in usages),
        'summed_actual_model_load_wall_seconds':sum(u['model_load_wall_seconds'] for u in usages),
        'wall_note':'Sum of4 actual nonoverlapping executor branches; excludes external process startup/orchestration gaps and CPU source review.',
        'retries':0, 'new_S0_calls':0, 'DEV_TEST_calls_in_this_diagnostic':0}
    assert usage['new_model_requests']==usage['new_returned_requests']==768
    eligibility=b.read(HERE/'models/B1/SECOND_WAVE_ELIGIBILITY_V2.json')
    assert eligibility['B1_reproduced'] is False and eligibility['second_wave_formal_B_GO'] is False
    probe={'shared_day5_day6_cap':24,'actual_probe_requests':0,'remaining_shared_cap':24,
        'release':'NOT_EXECUTED_NOT_RELEASED_SOURCE_QUALIFIED_MIN3ROOT3FAMILY_NOT_MET',
        'old_general_reserve':'RETIRED_NOT_EXECUTED',
        'narrow_reserve':b.binding(HERE/'NARROW_PROBE_CPU_FREEZE.json'),
        'retired_reserve_addendum':b.binding(HERE/'PROBE_RESERVE_ADDENDUM_V2.json')}
    summary={'schema_version':'day5_train_diagnosis_summary_v1', 'created_at_utc':rt.now(),
        'status':'ACTUAL_TRAIN_DIAGNOSIS_AND_REVIEW_CLOSED_FORMAL_FEEDBACK_NO_GO',
        'scope':{'questions':24,'train_roots':6,'question_types':{'multi_select':12,'ordering':12},
            'source_arms':{'A':12,'B':12},'own_arm_questions_per_model':12,'opposite_arm_questions_per_model':12,
            'selection':'Frozen paired slot selection without reading diagnostic scores.',
            'interpretation':'Both own and opposite arm share training roots/source/fact families. Opposite arm is not independent heldout generalization.'},
        'primary_metric':'Question-equal direct exact accuracy. Root-equal and call-pooled also reported with exact rational fractions. MC18 and ordering12 calls/question make pooled call rate a different weighting.',
        'exposure_note_for_S0':'S0 exposure groups merely index later A1/B1 arm assignment; no S0 treatment difference.',
        'S0':s0,'models':models,'actual_new_usage':usage,'controlled_probe':probe,
        'error_interpretation':'Strict invalid brief outputs are not rescued. Literal quote/reference, semantic judgment and final-answer errors are separate; direct answers remain primary. Visible brief reasons are not evidence of faithful internal CoT.',
        'feedback':{'snapshot':b.binding(HERE/'models/B1/FEEDBACK_SNAPSHOT_V2.json'),
            'decision':b.binding(HERE/'models/B1/SECOND_WAVE_ELIGIBILITY_V2.json'),
            'qualified_narrow_failure_questions':1,'qualified_narrow_failure_roots':1,'qualified_narrow_failure_fact_families':1,
            'required_roots':3,'required_fact_families':3,'B1_reproduced':False,'second_wave_formal_B_GO':False,
            'A1_release_to_A_generator':False,'inherit_S0_reproduced':False,'fact_memory_adopted':False,'generation_context':'full_history'},
        'closure':{'train_diagnosis_records_closed':True,'day5_full_GO':False,
            'next_wave_feedback':'FORMAL_TARGETED_B_NO_GO',
            'meaning':'Actual training/DEV/Train diagnosis records can close with a negative feedback eligibility result. This summary does not authorize Day6 production or assert positive feedback benefit.'},
        'code':b.binding(Path(__file__))}
    b.save(HERE/'DIAGNOSIS_SUMMARY.json',summary)
    print(__import__('json').dumps({'status':summary['status'],'actual_new_usage':usage,
        'S0_all':s0['direct_groups_relative_to_A1']['ALL/ALL'],
        'A1_all':models['A1']['direct']['ALL/ALL'],'B1_all':models['B1']['direct']['ALL/ALL']}))


if __name__=='__main__':
    main()
