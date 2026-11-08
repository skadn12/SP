"""Run/resume the bounded v15 feedback -> expansion -> QA -> training-pool chain.

Complete phases are reused after their own artifact checks. An unfinished phase
that has already started is never automatically repeated (no duplicate spending).
There is no training command. Logs and the final pool live in this version only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
D=HERE.parent
PY=D.parent/'.venv-local4b/bin/python'
EXP=D/'h4_feedback_training_pool_v15_expansion'
QA=D/'h4_feedback_training_pool_v15_qa'
BASELINES=[D/'h4_r02_yield_pilot_v12/runs/contract_smoke_2roots_01_qa/SELECTED_QUESTIONS.json',
           D/'h4_r02_question_contract_v14/runs/frozen_v12_2roots_01/SELECTED_QUESTIONS.json']


def command(args, label, logs):
    logfile=logs/(label+'.log')
    if logfile.exists():
        raise RuntimeError('phase_log_exists_without_completion_review_required:'+str(logfile))
    print(json.dumps({'stage':label,'status':'RUNNING','log':str(logfile)},ensure_ascii=False),flush=True)
    with logfile.open('x') as stream:
        subprocess.run([str(PY),*map(str,args)],cwd=str(D.parent),stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'stage':label,'status':'COMPLETE'}),flush=True)


def guard_started(path, completed):
    if path.exists() and not completed.exists():
        raise RuntimeError('unfinished_started_phase_needs_inspection_no_automatic_replay:'+str(path))


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_paths(run_name):
    require(isinstance(run_name,str) and run_name not in {'','.','..'}
            and Path(run_name).name==run_name and '\x00' not in run_name,
            'single_directory_run_name_required')
    paths=[]
    for owner,subfolder,name in ((EXP,'runs',run_name),(QA,'runs',run_name),
                                (HERE,'runs',run_name+'_training_pool'),(HERE,'logs',run_name)):
        parent=(owner/subfolder).resolve()
        path=(parent/name).resolve()
        require(parent.is_relative_to(owner.resolve()) and path.parent==parent,
                'run_path_escapes_expected_parent')
        paths.append(path)
    return tuple(paths)


def verify_training_pool(output, expected_inputs):
    """Validate saved outputs and exact inputs before a resume can claim success."""
    output=Path(output).resolve()
    manifest,report=read(output/'MANIFEST.json'),read(output/'REPORT.json')
    require(manifest.get('version')==report.get('version')=='h4_feedback_training_pool_v15'
            and manifest.get('immutable_write_once') is True
            and manifest.get('training_started') is False and report.get('training_started') is False,
            'existing_pool_version_or_execution_scope_mismatch')
    require(report.get('status')=='IMMUTABLE_TRAINING_POOL_PREPARED_NOT_TRAINED'
            and manifest.get('report')==report and manifest.get('inputs')==report.get('inputs'),
            'existing_pool_manifest_report_mismatch')
    files=manifest.get('files')
    require(isinstance(files,list) and len({row['path'] for row in files})==len(files),
            'existing_pool_file_manifest_invalid')
    named={row['path'] for row in files}
    require({'TRAINING_POOL.jsonl','REPORT.json','SUMMARY.md'}<=named,
            'existing_pool_required_artifact_missing')
    for row in files:
        relative=Path(row['path'])
        path=(output/relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(output) and path.is_file(),
                'existing_pool_file_path_invalid:'+str(relative))
        require(sha(path)==row['sha256'],'existing_pool_file_hash_changed:'+str(relative))
    actual={str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()}
    require(actual==named|{'MANIFEST.json'},'existing_pool_unbound_or_missing_files')
    expected=[(str(Path(path).resolve()),role) for path,role in expected_inputs]
    inputs=manifest['inputs']
    require([(row['path'],row['role']) for row in inputs]==expected,
            'existing_pool_input_paths_or_roles_mismatch')
    input_counts=Counter()
    for row in inputs:
        require(sha(row['path'])==row['sha256'],'existing_pool_input_hash_changed:'+row['path'])
        source=read(row['path'])
        require(len(source['questions'])==row['selected_count'],'existing_pool_input_count_changed')
        input_counts[row['role']]+=row['selected_count']
    require(report['input_existing_selected']==input_counts['existing']
            and report['input_new_selected']==input_counts['new']
            and report['input_selected_total']==sum(input_counts.values()),'existing_pool_input_counts_mismatch')
    rows=[json.loads(line) for line in (output/'TRAINING_POOL.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(rows)==report['total_unique']==report['existing_unique']+report['new_unique_added']
            and len({row['canonical_question_id'] for row in rows})==len(rows),
            'existing_pool_unique_counts_mismatch')
    require(report['exact_duplicates_excluded']==len(report['duplicates'])==report['input_selected_total']-len(rows),
            'existing_pool_dedup_counts_mismatch')
    counts,per_root=Counter(),{}
    for row in rows:
        require(row['origin'] in {'existing','new'} and row['question_type'] in {'multi_select','ordering'},
                'existing_pool_row_scope_invalid')
        counts[row['origin']]+=1
        counts[row['question_type']]+=1
        root_counts=per_root.setdefault(row['root_id'],{'existing':0,'new':0,'multi_select':0,'ordering':0,'total':0})
        root_counts[row['origin']]+=1
        root_counts[row['question_type']]+=1
        root_counts['total']+=1
        original=row['original_selected_question']
        field='options' if row['question_type']=='multi_select' else 'items'
        require(all(row[key]==original[key] for key in ('candidate_id','root_id','question_type','question','gold',field)),
                'existing_pool_question_or_gold_changed')
        require(row['training_target']=='complete_canonical_gold' and row['canonical_exposure_units']==1
                and row['loss_weight_from_partial_score'] is False
                and row['evaluated_permutations_are_separate_training_examples'] is False,
                'existing_pool_training_target_or_weight_mismatch')
        context=(output/'contexts'/(row['source_sha256']+'.json')).resolve()
        require(Path(row['source_path']).resolve()==context and sha(context)==row['source_sha256']
                and read(context)==row['full_context'], 'existing_pool_context_binding_changed')
    require(counts['existing']==report['existing_unique'] and counts['new']==report['new_unique_added']
            and counts['multi_select']==report['selected_multi_select'] and counts['ordering']==report['selected_ordering']
            and per_root==report['per_root'],'existing_pool_report_counts_mismatch')
    require(report['training_calls']==report['generation_calls']==report['solver_calls']==0
            and report['ground_used_for_selection'] is False,'existing_pool_execution_scope_mismatch')
    return report


def report_complete(output, report):
    counts={key:report[key] for key in ('existing_unique','new_unique_added','total_unique',
        'selected_multi_select','selected_ordering','exact_duplicates_excluded')}
    print(json.dumps({'status':'TRAINING_POOL_PREPARED','training_started':False,'pool':str(output),**counts},
                     ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--scenes',type=int,default=1)
    parser.add_argument('--slots',type=int,default=4)
    args=parser.parse_args()
    expansion,qa,out,logs=run_paths(args.run_name)
    expected_inputs=[(path,'existing') for path in BASELINES]+[(qa/'SELECTED_QUESTIONS.json','new')]
    if (out/'MANIFEST.json').exists():
        em,qm=read(expansion/'MANIFEST.json'),read(qa/'MANIFEST.json')
        require(em['scenes_per_route_target']==args.scenes,'existing_expansion_scene_budget_mismatch')
        require(qm['slots_per_root_route_type']==args.slots and Path(qm['expansion_run']).resolve()==expansion,
                'existing_qa_source_or_budget_mismatch')
        report_complete(out,verify_training_pool(out,expected_inputs))
        return
    logs.mkdir(parents=True,exist_ok=True)
    if not (expansion/'MANIFEST.json').exists():
        command([EXP/'runner.py','prepare','--run',expansion,'--scenes',args.scenes,'--routes','both'],'01-expansion-prepare',logs)
    em=json.loads((expansion/'MANIFEST.json').read_text())
    if em['scenes_per_route_target']!=args.scenes:
        raise ValueError('existing_expansion_scene_budget_mismatch')
    if not (expansion/'REPORT.json').exists():
        guard_started(expansion/'LIVE_STARTED.json',expansion/'REPORT.json')
        command([EXP/'runner.py','live','--run',expansion],'02-expansion-live',logs)
    if not (expansion/'FINAL_AUDIT.json').exists():
        command([EXP/'audit_run.py','--run',expansion],'03-expansion-audit',logs)
    if json.loads((expansion/'FINAL_AUDIT.json').read_text())['status']!='PASS':
        raise ValueError('expansion_audit_not_pass')
    if not (qa/'MANIFEST.json').exists():
        command([QA/'qa.py','prepare','--run',qa,'--expansion-run',expansion,'--slots',args.slots],'04-qa-prepare',logs)
    qm=json.loads((qa/'MANIFEST.json').read_text())
    if qm['slots_per_root_route_type']!=args.slots or Path(qm['expansion_run'])!=expansion:
        raise ValueError('existing_qa_source_or_budget_mismatch')
    def generate(branch):
        directory=qa/branch/'generation'
        if not (directory/'REPORT.json').exists():
            guard_started(directory/'STARTED.json',directory/'REPORT.json')
            command([QA/'qa.py','generate','--run',qa,'--branch',branch],'05-qa-generate-'+branch,logs)
    with ThreadPoolExecutor(max_workers=2) as workers:
        jobs=[workers.submit(generate,branch) for branch in 'AB']
        for job in jobs:job.result()
    for branch in 'AB':
        solver=qa/branch/'solver'
        if not (solver/'manifest.json').exists() and not (solver/'SKIPPED.json').exists():
            command([QA/'qa.py','solver-preflight','--run',qa,'--branch',branch],'06-preflight-'+branch,logs)
        if (solver/'SKIPPED.json').exists():continue
        if not (solver/'COMPLETE.json').exists():
            guard_started(solver/'STARTED.json',solver/'COMPLETE.json')
            command([QA/'qa.py','solver-live','--run',qa,'--branch',branch],'07-solver-'+branch,logs)
        if not (solver/'RESULTS.json').exists():
            command([QA/'qa.py','solver-report','--run',qa,'--branch',branch],'08-results-'+branch,logs)
    if not (qa/'REPORT.json').exists():
        command([QA/'qa.py','deliver','--run',qa],'09-qa-delivery',logs)
    if not (qa/'AUDIT.json').exists():
        command([QA/'audit_run.py','--run',qa],'10-qa-audit',logs)
    if json.loads((qa/'AUDIT.json').read_text())['status']!='PASS':
        raise ValueError('qa_audit_not_pass')
    if not (out/'MANIFEST.json').exists():
        command([HERE/'assemble_pool.py',*[x for path in BASELINES for x in ('--input',path)],
                 '--new-input',qa/'SELECTED_QUESTIONS.json','--output',out],'11-training-pool',logs)
    report_complete(out,verify_training_pool(out,expected_inputs))


if __name__=='__main__':main()
