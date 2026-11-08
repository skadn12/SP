"""Cross-check Day 1 packages and reference bytes, without running models."""
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import ast
import hashlib
import importlib.util
import json

BASE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    checks=[]
    def check(ok,name):checks.append({'name':name,'pass':bool(ok)})
    protocol=read(BASE/'PROTOCOL_V1.json')
    roots=read(BASE/'evaluation/ROOT_SELECTION.json')
    boundary=read(BASE/'evaluation/ROOT_BOUNDARY.json')
    whitelist=read(BASE/'TRAIN_ROOTS.json')['root_ids']
    check(len(whitelist)==70 and set(whitelist)==set(boundary['split_root_ids']['train']),'70_TRAIN_WHITELIST_EXACT')
    ordered=sorted(whitelist)
    for key,expected in [('wave1',ordered[:20]),('wave2',ordered[20:30]),('memory_pilot',ordered[:6])]:
        check(roots[key]['root_ids']==expected,'SCORE_FREE_ROOT_SELECTION:'+key)
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        check(not(set(boundary['split_root_ids'][a])&set(boundary['split_root_ids'][b])),'DISJOINT:'+a+':'+b)
    check([len(boundary['split_root_ids'][k]) for k in ('train','dev','test')]==[70,15,16],'AUTHORITATIVE_70_15_16')
    for key,expected in [('new_solver_inference',False),('new_training',False),('new_dialogue_expansion',False)]:
        check(protocol['execution_this_package'][key] is expected,'NO_EXPENSIVE_EXECUTION:'+key)
    check(protocol['arms']['A']['reads_solver_failure_memory'] is False,'A_NO_DIAGNOSIS_FEEDBACK')
    check(protocol['arms']['B']['reads_solver_failure_memory'] is True,'B_FEEDBACK_DEFINED')
    for rel in [protocol['legacy_reuse']['queue'],protocol['root_schedule']['file'],protocol['diagnosis']['taxonomy'],protocol['diagnosis']['schema'],protocol['evaluation']['policy']]:
        check((BASE/rel).is_file(),'REFERENCED_CONFIG_EXISTS:'+rel)
    slots=read(BASE/'SLOT_PLAN.json')
    check(slots['protocol_sha256']==sha(BASE/'PROTOCOL_V1.json'),'SLOTS_BIND_FINAL_PROTOCOL')
    check(slots['root_selection_sha256']==sha(BASE/'evaluation/ROOT_SELECTION.json'),'SLOTS_BIND_ROOT_SELECTION')
    check(len(slots['slots'])==360==len({x['candidate_slot_id'] for x in slots['slots']}),'360_UNIQUE_SLOT_IDS_NOT_GENERATED_QUESTIONS')
    for arm in ['A','B']:
        arm_slots=[s for s in slots['slots'] if s['arm']==arm]
        check(Counter(s['question_type'] for s in arm_slots)=={'multi_select':90,'ordering':90},'90_90_SLOTS:'+arm)
        check(Counter(s['stage'] for s in arm_slots)=={1:120,2:60},'120_60_WAVES:'+arm)
        check(Counter(s['target_gold_count'] for s in arm_slots if s['question_type']=='multi_select')=={2:23,3:23,4:22,5:22},'GOLD_QUOTA:'+arm)
        check(all(not s['question_generated'] and s['source_binding'] is None for s in arm_slots),'SLOTS_NOT_MISREPORTED_AS_GENERATED:'+arm)
    check(sum(s['generation_mode']=='diagnosis_targeted' for s in slots['slots'])==126,'B_126_TARGETED_SLOTS')
    public=read(BASE/'reuse/public/QUESTIONS.json')['questions']
    inv=read(BASE/'reuse/INVENTORY.json')['questions']
    check(len(public)==len(inv)==113,'113_LEGACY_REVIEW_CANDIDATES')
    check(Counter(q['question_type'] for q in public)=={'multi_select':51,'ordering':62},'51_MC_62_ORDER')
    check(len({q['root_id'] for q in public})==56,'56_LEGACY_TRAIN_ROOTS')
    allowed_public={'candidate_id','root_id','question_type','question','options','items','source_binding'}
    for q in public:
        check(set(q)==allowed_public,'PUBLIC_WHITELIST:'+q['candidate_id'])
        check(q['root_id'] in whitelist,'LEGACY_TRAIN_ONLY:'+q['candidate_id'])
        sb=q['source_binding'];p=Path(sb['path'])
        check(p.is_file() and p.stat().st_size==sb['bytes'] and sha(p)==sb['sha256'],'LEGACY_SOURCE_BYTES:'+q['candidate_id'])
    for rel in ['diagnosis/VALIDATION.json','evaluation/VALIDATION.json','reuse/VALIDATION_RECHECK.json']:
        d=read(BASE/rel);check(d['status']=='PASS','CHILD_VALIDATION_PASS:'+rel)
    d=read(BASE/'diagnosis/VALIDATION.json')
    for rel,rec in d['files'].items():
        check(sha(BASE/'diagnosis'/rel)==rec['sha256'],'DIAGNOSIS_VALIDATION_BINDING:'+rel)
    spec=importlib.util.spec_from_file_location('day1_diagnosis_validator',BASE/'diagnosis/validate_diagnosis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    memory=read(BASE/'WEAKNESS_MEMORY_B_v000.json')
    check(not module.validate(memory,set(whitelist)),'INITIAL_EMPTY_MEMORY_VALID')
    check(not memory['causes'] and not memory['failure_cases'] and not memory['probes'],'NO_FAKE_DISCOVERED_CAUSES')
    for k in ['taxonomy','allowed_train_roots_manifest']:
        rec=memory[k];check(sha(rec['path'])==rec['sha256'],'MEMORY_REF_BYTES:'+k)
    next_plan=read(BASE/'diagnosis_inputs/INPUT_PLAN.json')
    next_public=read(BASE/'diagnosis_inputs/PUBLIC_QUESTIONS.json')['questions']
    next_private=read(BASE/'diagnosis_inputs/PRIVATE_QUESTIONS.json')['questions']
    check(len(next_public)==len(next_private)==next_plan['question_count']==39,'39_NEXT_STAGE_DIAGNOSIS_INPUTS')
    check(len({q['root_id'] for q in next_public})==next_plan['root_count']==20,'20_TRAIN_ROOTS_FOR_NEXT_DIAGNOSIS')
    check(Counter(q['question_type'] for q in next_public)=={'multi_select':19,'ordering':20},'19_MC_20_ORDER_NEXT_INPUTS')
    check(next_plan['planned_direct_calls']==19*18+20*12==582,'582_PLANNED_NOT_EXECUTED_CALLS')
    check(next_plan['new_solver_calls']==0 and next_plan['use_as_final_eval'] is False,'NEXT_INPUTS_ARE_UNEXECUTED_TRAIN_FEEDBACK')
    for pub,priv in zip(next_public,next_private):
        expected={k:v for k,v in priv.items() if k in {'candidate_id','root_id','question_type','question','source_path','source_sha256','options','items'}}
        check(pub==expected,'NEXT_INPUT_PUBLIC_WHITELIST:'+pub['candidate_id'])
        check(pub['root_id'] in whitelist and sha(pub['source_path'])==pub['source_sha256'],'NEXT_INPUT_TRAIN_SOURCE:'+pub['candidate_id'])
    bindings=read(BASE/'MODEL_BINDINGS.json')
    for key,model in bindings['models'].items():
        for rec in model['files']:
            p=Path(rec['path'])
            check(p.is_file() and p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],'MODEL_BOUND_BYTES:'+key+':'+p.name)
    ledger=read(BASE/'reviews/ADMISSION_LEDGER.json')
    freeze=read(BASE/'reviews/RECONSTRUCTION_FREEZE.json')
    check(ledger['reconstruction_freeze_sha256']==sha(BASE/'reviews/RECONSTRUCTION_FREEZE.json'),'LEDGER_BINDS_SOURCE_RECONSTRUCTIONS')
    for rec in freeze['reviews']:
        check(sha(BASE/rec['path'])==rec['sha256'],'FROZEN_SOURCE_REVIEW:'+rec['path'])
    check(len(ledger['questions'])==113 and len({q['candidate_id'] for q in ledger['questions']})==113,'113_UNIQUE_FINAL_DECISIONS')
    check(set(q['candidate_id'] for q in ledger['questions'])==set(q['candidate_id'] for q in public),'FINAL_DECISIONS_COVER_ALL_CANDIDATES')
    check(dict(Counter(q['status'] for q in ledger['questions']))==ledger['status_counts'],'FINAL_STATUS_ARITHMETIC')
    check(all(not q['training_admitted'] for q in ledger['questions']),'CONTENT_REVIEW_NOT_TRAINING_RUNTIME_ADMISSION')
    golds={q['candidate_id']:q['original_question_record']['gold'] for q in inv}
    for row in ledger['questions']:
        check(row['original_gold_unchanged']==golds[row['candidate_id']],'ORIGINAL_GOLD_PRESERVED:'+row['candidate_id'])
        if row['status']=='PASS_INTERNAL_CONTENT':
            check(not row['review_record_errors'] and not row['conservative_holds'] and row['gold_agreement'] is True,'PASS_ALL_GATES:'+row['candidate_id'])
    for p in sorted(BASE.rglob('*.py')):
        ast.parse(p.read_text());check(True,'PYTHON_SYNTAX:'+str(p.relative_to(BASE)))
    errors=[x['name'] for x in checks if not x['pass']]
    report={'status':'PASS' if not errors else 'FAIL','created_at_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'check_count':len(checks),'error_count':len(errors),'errors':errors,'checks':checks,
            'new_solver_calls':0,'training_runs':0,
            'limits':['Mechanical cross-checks and declared role boundaries do not prove source semantic truth or OS access isolation.',
                      'HOLD/FAIL content decisions are valid outcomes, not mechanical validation failures.']}
    with (BASE/'VALIDATION.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ['status','check_count','error_count','errors']},ensure_ascii=False))
    return not errors


if __name__=='__main__':raise SystemExit(0 if main() else 1)
