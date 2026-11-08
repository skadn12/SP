#!/usr/bin/env python3
"""Future unstarted shape/unused calls: exact known lossless blind schemas.

The frozen v1 adapter and all live drivers stay byte-identical. Only the two
already observed R03 key aliases and explicit call-global true/false read
declarations are adapted in RAM. No Gold, evidence, decision or budget changes.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import question_driver as q
import review_schema_adapter as v1
import question_candidate_shape_continuation as shape

EXPECTED_V1='7fb0938e8bf007ee62bfdbe6938fd26baff28568c35b7eec92eafa59124f44ec'
DRIVERS={
    'shape':('question_candidate_shape_continuation','70b44b835a314fc9a89c2fb88533f7e03dbb0c6d2ca047b22bd58b9613ad3796'),
    'unused':('question_unfilled_slot_continuation','0bf0fedd04e85d23edc1df0d013ef0f8f82c2303d2ffd885a6fafa315621ea34'),
    'unused_failed':('unused_failed_lineage_adapter','161cdbd348b9057164d10039accef1eb843754790185999b70b34ee13061eb56')}
ASSIGNMENTS={'shape':{('179186','F'),('77882','G'),('136202','F')},
             'unused':{('77882','F')},'unused_failed':{('109164','F')}}
MAPS={frozenset(('decision','removal_reason','r03_turn_refs')):{'removal_reason':'rationale','r03_turn_refs':'turn_refs'},
      frozenset(('decision','removal_reason','turn_refs')):{'removal_reason':'rationale'}}
CANONICAL=frozenset(('decision','rationale','turn_refs'))
DECLARATIONS={'complete_source_read':True,'solver_results_read':False}
ORIGINAL_V1_INSTALL=v1.install


def verify_code():
    if hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest()!=EXPECTED_V1:
        raise RuntimeError('Frozen v1 RAM adapter drift')
    for name,expected in DRIVERS.values():
        module=importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen future driver helper drift: '+name)


def normalize_blind_schema(raw):
    """Exactly invert the observed aliases and explicit-global-to-missing copy.

    Invalid or unproven read declarations, ambiguous collections, R03 key
    mixtures/extras and conflicting values fail closed. The original complete
    object is recoverable exactly; all semantic values remain untouched.
    """
    if raw is None:return None,[]
    if not isinstance(raw,dict) or not isinstance(raw.get('reviews'),list) or not raw['reviews']:
        raise RuntimeError('Only the observed nonempty reviews object schema is supported')
    has_global=any(key in raw for key in DECLARATIONS)
    allowed={'reviews',*DECLARATIONS} if has_global else {'reviews'}
    if set(raw)!=allowed:
        raise RuntimeError('Missing, extra, ambiguous or unproven call-scope declaration')
    if has_global and any(raw.get(key) is not value for key,value in DECLARATIONS.items()):
        raise RuntimeError('Call-global declarations must be explicit true/false exactly')
    rows=raw['reviews']
    keys=[row.get('question_key') if isinstance(row,dict) else None for row in rows]
    if not all(isinstance(key,str) and key for key in keys) or len(set(keys))!=len(keys):
        raise RuntimeError('Review question keys must be present and unique')
    # Validate the entire scope before mutating the deep copy. Never fill absent
    # declarations from a different row or infer a read from useful-looking Gold.
    for row in rows:
        for key,value in DECLARATIONS.items():
            if key in row and row[key] is not value:
                raise RuntimeError('Conflicting or unproven per-record read declaration')
            if key not in row and not has_global:
                raise RuntimeError('Missing per-record declaration has no explicit global source')
        r03=row.get('r03_necessity')
        if not isinstance(r03,dict) or frozenset(r03) not in {CANONICAL,*MAPS}:
            raise RuntimeError('R03 missing, extra, mixed or unproven key shape')
        mapping=MAPS.get(frozenset(r03),{})
        reason=r03.get(next((k for k,v in mapping.items() if v=='rationale'),'rationale'))
        refs=r03.get('r03_turn_refs' if 'r03_turn_refs' in r03 else 'turn_refs')
        if (not isinstance(reason,str) or not reason.strip() or not isinstance(refs,list) or not refs
                or not all(isinstance(ref,str) and ref for ref in refs)):
            raise RuntimeError('Empty or invalid rationale/refs are not repairable')
    value=copy.deepcopy(raw);changes=[]
    for index,row in enumerate(value['reviews']):
        original_r03=copy.deepcopy(row['r03_necessity']);mapping=MAPS.get(frozenset(original_r03),{})
        if mapping:
            row['r03_necessity']={mapping.get(key,key):copy.deepcopy(item) for key,item in original_r03.items()}
            changes.append({'kind':'exact_observed_r03_key_alias','review_index':index,'question_key':row['question_key'],
                'mapping':mapping,'authored_original_value':original_r03,'normalized_value':copy.deepcopy(row['r03_necessity']),
                'decision_and_all_values_changed':False})
        for key,expected in DECLARATIONS.items():
            if key not in row:
                row[key]=copy.deepcopy(raw[key])
                changes.append({'kind':'explicit_call_scope_attestation_copy','review_index':index,'question_key':row['question_key'],
                    'field':key,'original_field_absent':True,'source_global_path':[key],'source_global_value':raw[key],
                    'normalized_value':row[key],'global_original_retained':True})
    reversed_value=copy.deepcopy(value)
    for change in reversed(changes):
        row=reversed_value['reviews'][change['review_index']]
        if change['kind']=='explicit_call_scope_attestation_copy':
            if row[change['field']] is not change['source_global_value']:raise RuntimeError('Declaration copy changed value')
            del row[change['field']]
        else:
            reverse={v:k for k,v in change['mapping'].items()}
            restored={reverse.get(key,key):copy.deepcopy(item) for key,item in row['r03_necessity'].items()}
            if restored!=change['authored_original_value']:raise RuntimeError('R03 mapping is not exactly reversible')
            row['r03_necessity']=restored
    if reversed_value!=raw:raise RuntimeError('Full authored object does not round-trip exactly')
    return value,changes


def install(launch_binding=None):
    """Clone frozen v1 installation with only the proven pure normalizer hook.

    Its candidate normalizer, G validation-only exception, gates, contexts and
    logging policy remain the same. Logging honestly names this new wrapper.
    """
    verify_code()
    namespace=dict(vars(v1));namespace.update(__file__=__file__,normalize_blind_aliases=normalize_blind_schema)
    namespace['_publish_normalization']=types.FunctionType(v1._publish_normalization.__code__,namespace,
        v1._publish_normalization.__name__,v1._publish_normalization.__defaults__,v1._publish_normalization.__closure__)
    installer=types.FunctionType(ORIGINAL_V1_INSTALL.__code__,namespace,ORIGINAL_V1_INSTALL.__name__,
        ORIGINAL_V1_INSTALL.__defaults__,ORIGINAL_V1_INSTALL.__closure__)
    result=installer(launch_binding)
    result.update({'v2_wrapper_binding':q.binding(__file__),'frozen_v1_adapter_binding':q.binding(v1.__file__),
        'only_added_scope':'two observed exact R03 aliases and explicit-global-to-missing-row declarations',
        'original_v1_install_code_sha256':hashlib.sha256(ORIGINAL_V1_INSTALL.__code__.co_code).hexdigest()})
    return result


def parse_target(driver,argv):
    if driver=='unused_failed':
        if argv!=['run']:raise RuntimeError('The consumed-failure wrapper has only its exact fixed run assignment')
        return '109164','F'
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('command',choices=['run']);parser.add_argument('--root',required=True);parser.add_argument('--arm',choices=q.ARMS,required=True)
    parser.add_argument('--input-work' if driver=='shape' else '--work-dir',required=True)
    args=parser.parse_args(argv);number=q.root_number(args.root)
    if (number,args.arm) not in ASSIGNMENTS[driver]:raise RuntimeError('Future adapter assignment is outside its exact authorized targets')
    work=Path(getattr(args,'input_work',None) or args.work_dir).resolve();work.relative_to(BASE)
    if work!=BASE/'question_plan_continuation'/number/args.arm:raise RuntimeError('Future input-work differs from actual frozen continuation')
    return number,args.arm


def target_stage_requests(driver,number,arm):
    requests=shape.stage_requests(q.root_id(number),arm)
    suffix='candidate_shape_01' if driver=='shape' else 'unused_content_revision_01'
    return [row for rows in requests.values() for row in rows if suffix in row['call_id']]


def self_check():
    verify_code();selection=BASE/'SOURCE_SELECTION_FINAL.json';q.SOURCE_FREEZE_PATH=BASE/'SOURCE_FREEZE.json'
    _,sources=q.load_selection(selection)
    fixtures=[('179186','BLIND_GOLD.json','CANDIDATES_DRAFT.json',8),
              ('136202','BLIND_GOLD_revision_01.json','CANDIDATES_REVISION_01.json',24),
              ('109164','BLIND_GOLD_revision_01.json','CANDIDATES_REVISION_01.json',21)]
    fixture_results=[];actual=[]
    for number,raw_name,candidate_name,count in fixtures:
        work=BASE/'question_plan_continuation'/number/'G';raw=q.read(work/raw_name);candidates=q.read(work/candidate_name)
        normalized,changes=normalize_blind_schema(raw);assert len(changes)==count
        source=sources[q.root_id(number)]
        prior,errors_before=q.normalize_blind(copy.deepcopy(raw),candidates,source['source'])
        after,errors_after=q.normalize_blind(copy.deepcopy(normalized),candidates,source['source']);assert not errors_after,errors_after
        assert {k:v['independent_gold'] for k,v in prior.items()}=={k:v['independent_gold'] for k,v in after.items()}
        assert raw==q.read(work/raw_name)
        actual.append((number,raw,candidates,source))
        fixture_results.append({'root_id':q.root_id(number),'raw_binding':q.binding(work/raw_name),
            'candidate_binding':q.binding(work/candidate_name),'record_count':len(candidates),'reversible_changes':count,
            'prior_parser_error_count':len(errors_before),'adapted_parser_errors':[]})
    sample=actual[1][1]
    for defect in ('missing_global','partial_global','global_conflict','global_string','row_conflict','row_string',
                   'missing_r03','extra_r03','alias_conflict','empty_reason','extra_top','duplicate_key'):
        bad=copy.deepcopy(sample)
        if defect=='missing_global':del bad['complete_source_read'];del bad['solver_results_read']
        elif defect=='partial_global':del bad['solver_results_read']
        elif defect=='global_conflict':bad['complete_source_read']=False
        elif defect=='global_string':bad['solver_results_read']='false'
        elif defect=='row_conflict':bad['reviews'][0]['solver_results_read']=True
        elif defect=='row_string':bad['reviews'][0]['complete_source_read']='true'
        elif defect=='missing_r03':del bad['reviews'][0]['r03_necessity']['removal_reason']
        elif defect=='extra_r03':bad['reviews'][0]['r03_necessity']['extra']=True
        elif defect=='alias_conflict':bad['reviews'][0]['r03_necessity']['rationale']='ENGINEERING_CONFLICT_FIXTURE'
        elif defect=='empty_reason':bad['reviews'][0]['r03_necessity']['removal_reason']=''
        elif defect=='extra_top':bad['read_all_claim']=True
        else:bad['reviews'][0]['question_key']=bad['reviews'][1]['question_key']
        try:normalize_blind_schema(bad)
        except RuntimeError:pass
        else:raise AssertionError('Rejected missing/conflicting/extra fixture was repaired: '+defect)
    # Explicit same-scope row declarations are retained; only absent fields are
    # copied. A canonical all-row response needs no artificial global claims.
    canonical,_=normalize_blind_schema(sample)
    canonical.pop('complete_source_read');canonical.pop('solver_results_read')
    unchanged,empty=normalize_blind_schema(canonical);assert unchanged==canonical and not empty
    mixed=copy.deepcopy(sample);mixed['reviews'][0].update(DECLARATIONS)
    mapped,mixed_changes=normalize_blind_schema(mixed);assert len(mixed_changes)==22
    before={number:shape.stage_requests(q.root_id(number),'G') for number,_,_,_ in actual}
    old=(q.normalize_blind,q.normalize_candidates,q.validate_g_prompt);clones=[]
    cache=BASE/'runtime_cache/review_schema_adapter_v2_selfcheck';cache.mkdir(parents=True,exist_ok=True)
    installation=install()
    try:
        with tempfile.TemporaryDirectory(prefix='v2_',dir=cache) as temp:
            base=Path(temp)
            for number,raw,candidates,source in actual:
                module=importlib.import_module('question_plan_schema_continuation_v3');process,_=module.build_driver();namespace=process.__globals__
                assert namespace['normalize_blind'] is q.normalize_blind
                trace=[];work=base/number;work.mkdir()
                call_name='BLIND_GOLD_CALL.json' if number=='179186' else 'BLIND_GOLD_CALL_revision_01.json'
                receipt=q.read(BASE/'question_plan_continuation'/number/'G'/call_name)
                class Mock:
                    def call(self,key,prompt,**kwargs):
                        trace.append(key)
                        payload=shape.request_payload(prompt)
                        assert payload['complete_source_and_warnings']==q.source_view(source)
                        if key.startswith('q20_gold_'):
                            assert payload['anonymous_questions']==[q.anonymize_candidate(c)[0] for c in candidates]
                            return copy.deepcopy(raw),copy.deepcopy(receipt)
                        if number=='109164':
                            assert next(c for c in payload['review_cases'] if c['question_key']=='Qc148b6e2c27f50')['saved_independent_gold']['r03_necessity']['decision']=='FAIL'
                        raise RuntimeError('ENGINEERING_MOCK_FIRST_VERIFIER_ONLY_NO_DISPATCH')
                namespace['cc']=Mock();reviewer=namespace['continuation_review_batch']
                blind,review,calls,audits=reviewer(q.root_id(number),candidates,source,selection,work,'engineering_v2')
                assert len(trace)==2 and trace[1].startswith('q20_verify_')
                assert len(blind)==len(candidates) and all(r['decision']=='HOLD' for r in review.values())
                assert q.read(work/'BLIND_GOLD_engineering_v2.json')==raw
                assert q.read(work/'BLIND_GOLD_VALIDATION_engineering_v2.json')['errors']==[]
                log=q.read(work/'MECHANICAL_REVIEW_SCHEMA_NORMALIZATION_engineering_v2.json')
                assert log['adapter_binding']==q.binding(__file__) and log['exact_round_trip_verified'] is True
                assert len(log['changes'])==next(x['reversible_changes'] for x in fixture_results if x['root_id']==q.root_id(number))
                assert not (work/'CONTENT_VERIFIER_CALL_engineering_v2.json').exists()
                clones.append({'root_id':q.root_id(number),'Gold_records':len(blind),'mock_verifier_reached_once':True,'mock_no_response_HOLD_count':len(review)})
    finally:q.normalize_blind,q.normalize_candidates,q.validate_g_prompt=old
    assert {number:shape.stage_requests(q.root_id(number),'G') for number,_,_,_ in actual}==before
    return {'status':'PASS','semantic_calls_started':0,'semantic_admission_asserted':False,
        'real_fixture_checks':fixture_results,'negative_fixture_count':12,'canonical_and_mixed_explicit_declarations_checked':True,
        'actual_saved_source_and_anonymous_questions_exactly_bound_in_mock':True,'frozen_review_clones':clones,
        'saved_G109_MC4_FAIL_preserved':True,'frozen_v1_and_old_artifacts_modified':False,'installation':installation}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-check');launch=sub.add_parser('launch');launch.add_argument('--driver',choices=DRIVERS,required=True)
    launch.add_argument('driver_args',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.command=='self-check':print(json.dumps(self_check(),ensure_ascii=False));return 0
    verify_code();argv=args.driver_args[1:] if args.driver_args[:1]==['--'] else args.driver_args
    number,arm=parse_target(args.driver,argv)
    if (BASE/'CONTENT_FREEZE.json').exists() or q.cc.stop_generation():raise RuntimeError('No new production stage after generation cutoff')
    name,_=DRIVERS[args.driver];module=importlib.import_module(name)
    key=q.digest({'driver':args.driver,'argv':argv})[:24];path=BASE/'review_schema_adapter_v2/launches'/f'{key}.json'
    if not path.exists() and target_stage_requests(args.driver,number,arm):
        raise RuntimeError('Target semantic stage already started before this v2 launcher; no retrofit or replay allowed')
    receipt={'schema_version':'fgr20_ram_review_schema_v2_launch_v1','driver':args.driver,'root_id':q.root_id(number),'arm':arm,
        'driver_binding':q.binding(module.__file__),'adapter_binding':q.binding(__file__),'frozen_v1_adapter_binding':q.binding(v1.__file__),
        'driver_argv':argv,'future_unstarted_stage_only':True,'raw_and_old_validation_and_review_artifacts_preserved':True,
        'extra_semantic_budget':0,'semantic_admission_asserted':False,'parent_must_schedule_wrapper_families_as_separate_stages':True}
    q.publish(path,receipt);old_argv=sys.argv;old_install=v1.install
    try:
        if args.driver=='unused_failed':
            # This frozen wrapper installs its RAM adapter internally. Replace
            # only that callable in this new process and bind the honest v2 log.
            v1.install=lambda original_lineage_launch=None:install(q.binding(path))
        else:install(q.binding(path))
        sys.argv=[str(Path(module.__file__)),*argv]
        return module.main()
    finally:sys.argv=old_argv;v1.install=old_install


if __name__=='__main__':raise SystemExit(main())
