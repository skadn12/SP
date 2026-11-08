"""P1 preparation/live orchestration through the two authorized original adapters."""
from __future__ import annotations
import argparse
import importlib.util
import subprocess
from pathlib import Path
from .freeze import DEFAULT,ARMS,CONTROL,REPO,RUNS,load,source,write
from .qa_run import verify_schedule
from .prompt_audit import audit
from .analysis import review,compare
from ..core import atomic_json

PYTHON='/root/miniconda3/bin/python'
SOLVER_MODULE='scripts.current_method.restart_n5_spell.h4_round1_r03_difficulty_followup_v1.run'

def prepare(run):
    verify_schedule(run)
    prompt=audit(run)
    if prompt['status']!='PASS_COMPLETE' or prompt['errors']:raise RuntimeError('qa_not_complete_or_prompt_drift')
    atomic_json(run/'audit/prompt_parity_complete.json',prompt)
    review(run)
    path=RUNS/'prepare_aligned_r02_solver_eval.py'
    spec=importlib.util.spec_from_file_location('original_parameterized_solver_adapter',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    budgets=[]
    for arm in ARMS:
        output=run/arm/'solver'
        if not output.exists():adapter.prepare(run/arm/'qa',output,CONTROL)
        pre=load(output/'preflight_report.json')
        if pre['status']!='PASS':raise RuntimeError('solver_preflight_not_pass')
        for citation in pre['sources']:
            if source(citation['path'])['sha256']!=citation['sha256']:raise RuntimeError('solver_frozen_source_drift')
        for name,digest in pre['files'].items():
            if source(output/name)['sha256']!=digest:raise RuntimeError('solver_frozen_file_drift')
        budgets.append({'arm':arm,'candidate_count':pre['candidate_count'],'question_type_counts':pre['question_type_counts'],'requests_per_model':pre['request_count_per_model'],'all_models_request_count':pre['all_models_request_count'],'preflight':source(output/'preflight_report.json')})
    write(run/'solver_budget_manifest.json',{'status':'PASS','rows':budgets,'all_models_both_arms_request_count':sum(x['all_models_request_count'] for x in budgets),'budget_method':'original candidate_request_count over every program-valid candidate; no hard-coded expected QA count','control':str(CONTROL)})
    print({'status':'SOLVER_INPUTS_FROZEN','budgets':budgets},flush=True)

def live(run):
    budgets=load(run/'solver_budget_manifest.json')
    assert budgets['status']=='PASS'
    for arm in ARMS:
        output=run/arm/'solver'
        logfile=run/'audit'/f'{arm}_solver_live.log'
        # The original live runner verifies identities/sources, checks idle GPU,
        # resumes exact requests, loads and releases only its own models.
        with logfile.open('a') as log:
            result=subprocess.run([PYTHON,'-u','-m',SOLVER_MODULE,'--phase','live','--output',str(output)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        attempt=len(list((run/'audit').glob(f'{arm}_solver_process_result_*.json')))+1
        write(run/'audit'/f'{arm}_solver_process_result_{attempt:02d}.json',{'returncode':result.returncode,'log':str(logfile),'requested_phase':'live'})
        if result.returncode:raise RuntimeError(f'original_solver_live_failed:{arm}:see:{logfile}')
        assert load(output/'comparison.json')['status']=='COMPLETE'
        print({'arm':arm,'solver_models_complete':3},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=DEFAULT);parser.add_argument('--phase',choices=['prepare','live','compare'],required=True);args=parser.parse_args();{'prepare':prepare,'live':live,'compare':compare}[args.phase](args.run.resolve())
