"""Audit one immutable v89 mixed shard, independently of production scoring.

inputs: after freeze/prepare-evaluation, before the first Solver.
results: after both Solver runs, report, runtime receipt and RESULT_DETAILS.
Export admission is downstream of this result PASS and audited separately.
Every invocation writes a new evidence record; no artifact is overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode=True
AUDITOR_HOME=Path(__file__).resolve().parent
AUDITOR_NAMES=('audit_shard.py','audit_contract_v89.py','audit_results_core_v89.py','audit_generation_core_v89.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def next_output(home,stage):
    stem='INPUT_AUDIT' if stage=='inputs' else 'RESULT_AUDIT'
    path=home/'evidence'/f'{stem}.json'; attempt=1
    while path.exists():
        attempt+=1; path=home/'evidence'/f'{stem}.attempt{attempt:02}.json'
    return path,attempt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',required=True,type=Path)
    parser.add_argument('--stage',required=True,choices=('inputs','results'))
    args=parser.parse_args(); home=args.home.resolve()
    if not (home/'SHARD_ENGINE_INSTALLATION.json').is_file():
        raise ValueError('installed_shard_required')
    output,attempt=next_output(home,args.stage)
    opened=set()
    def guard(event,values):
        if event.startswith('socket.') or event in {'subprocess.Popen','os.system','os.posix_spawn'}:
            raise RuntimeError('audit_forbids_external:'+event)
        if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(values[0])).resolve(); mode,flags=values[1:3]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and path!=output and path!=Path('/dev/null') and not path.is_relative_to(Path('/tmp')):
                raise RuntimeError('audit_forbids_artifact_write:'+str(path))
            if not writing:
                opened.add(str(path))
                if args.stage=='inputs' and path.is_relative_to(home/'solver') and 'responses' in path.parts:
                    raise RuntimeError('input_audit_forbids_solver_response_read:'+str(path))
    sys.addaudithook(guard)
    sys.path.insert(0,str(home)); sys.path.insert(0,str(AUDITOR_HOME))
    results=importlib.import_module('audit_results_core_v89'); results.HOME=home
    a=results.Audit()
    report={'version':'v89_independent_shard_audit_v1','created_at_utc':now(),'phase':'inputs_only' if args.stage=='inputs' else 'paired_results',
            'attempt':attempt,'home':str(home),'new_model_or_http_calls':0,'training_performed':False,
            'independent_score_and_selection_implementation':True,'native_score_or_selection_functions_used':False,
            'generation_validator_replay_plus_independent_reconstruction':True,
            'source_answer_semantics_recertified_by_this_audit':False,'frozen_artifacts_modified':False,
            'network_and_subprocess_blocked':True,'downstream_typed_export_requires_separate_audit':True}
    try:
        results.ROOTS=tuple(s['root_id'] for s in a.read(home/'CONFIG.json')['multiselect_config']['slots'])
        auditor_files=[a.bind(AUDITOR_HOME/name) for name in AUDITOR_NAMES]
        results.selfcheck(a)
        cached={}
        if args.stage=='inputs':
            a.check(not any((home/f'{side}_EVALUATION.STARTED.json').exists() for side in ('TRAINED','BASE')),
                    'input_audit_before_any_solver_started')
            generation=importlib.import_module('audit_generation_core_v89')
            generation.HOME=home; generation.ROOTS=results.ROOTS
            generation.read=a.read; generation.bind=a.bind
            try:
                findings=generation.audit()
                report['generation_audit']=findings
            finally:
                a.checks.extend(generation.CHECKS)
            a.check(not any(c['status']=='FAIL' for c in generation.CHECKS),'generation_audit_complete')
        else:
            candidates=[]
            for path in sorted((home/'evidence').glob('INPUT_AUDIT*.json')):
                row=a.read(path)
                if row.get('status')=='PASS' and row.get('phase')=='inputs_only':
                    candidates.append((row['completed_at_utc'],path,row))
            if not candidates:
                raise ValueError('required_preexecution_input_audit_missing')
            _,path,prior=max(candidates,key=lambda r:r[0])
            a.check(prior['auditor_files']==auditor_files,'same_auditor_bytes_for_inputs_and_results')
            a.check(results.dt(prior['completed_at_utc'])<results.dt(a.read(home/'TRAINED_EVALUATION.STARTED.json')['time']),
                    'passing_input_audit_completed_before_first_solver_stage')
            for row in prior['files']:
                a.check(a.bind(row['path'])==row,'preexecution_audited_file_unchanged:'+row['path'])
            report['passing_input_audit']=a.bind(path)
            report['generation_audit']=prior['generation_audit']
            cached={r['path']:r for r in prior.get('checkpoint_verification',[])}
        ctx=results.audit_inputs(a,cached)
        report.update(source_admission_and_parent_findings=ctx['admission_findings'],candidate_count=len(ctx['questions']),
                      content_gold_distribution=ctx['gold_distribution'],expected_calls_per_model=ctx['expected_calls'],
                      ordering_candidate_count=len(ctx['ordering_pool']),controls=sorted(ctx['controls']))
        if args.stage=='results':
            report.update(results.audit_results(a,ctx))
    except BaseException as error:
        a.check(False,'audit_execution_exception',{'error':repr(error),'traceback':traceback.format_exc()})
    report['status']='FAIL' if any(c['status']=='FAIL' for c in a.checks) else 'PASS'
    report['checks']=a.checks; report['failed_checks']=[c for c in a.checks if c['status']=='FAIL'];report['check_count']=len(a.checks)
    report['checkpoint_verification']=list(a.weights.values())
    report['auditor_files']=[a.bind(AUDITOR_HOME/name) for name in AUDITOR_NAMES]
    report['files']=sorted(a.files.values(),key=lambda r:r['path'])
    report['opened_files']=sorted(opened)
    report['limitations']=['This audit reconstructs saved evidence and actual Solver scoring; it does not independently recertify source semantics.',
        'Source READY, exact full-turn bounds and name policies do not by themselves certify supported propositions, actors or unique order.',
        'Shared parent/source families are dependencies; exact whole-set novelty is not independent fact novelty.',
        'Global historical preservation and final typed export/union admission are audited separately by the pilot orchestrator.']
    report['completed_at_utc']=now()
    with output.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'path':str(output),'status':report['status'],'checks':report['check_count'],'failed_checks':report['failed_checks']},ensure_ascii=False))
    if report['status']!='PASS':
        raise SystemExit(1)


if __name__=='__main__':
    main()
