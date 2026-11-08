#!/usr/bin/env python3
"""Versioned OET1 driver guard; v1 frozen source/prompts remain unchanged.

This entry does not run unless explicitly invoked. It adds exact request/raw/
parsed/terminal/RESULT checks and rejects any actual tool event before returning
a model response to source/question commit code. Fresh semantic attempts are
still owned by the original finite three-worker v1 scheduler.
"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
driver=module('oet1_frozen_driver_v1',BASE/'scripts/production_driver.py')
integrity=module('oet1_production_integrity',BASE/'scripts/production_integrity.py')
original_call=driver.call

def call(call_id,prompt):
    result_path=BASE/'cli_calls'/call_id/'RESULT.json'
    if result_path.exists():integrity.validate_call_result(result_path,prompt)
    response,result=original_call(call_id,prompt)
    try:
        checked,_=integrity.validate_call_result(result_path,prompt)
        assert checked==result,'RETURNED_RESULT_RECEIPT_MISMATCH'
    except Exception as error:
        rejection=result_path.with_name('PROVENANCE_REJECTION_V2.json')
        if not rejection.exists():
            integrity.exclusive_json(rejection,{'schema_version':'oet1_cli_provenance_rejection_v2','created_at_utc':integrity.now(),
                'status':'FAIL','error':str(error),'result_binding':integrity.bind(result_path),
                'semantic_retry_permitted':False,'original_call_and_return_preserved':True})
        raise
    return response,result

driver.call=call

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--action',choices=['first-wave','second-wave','questions-first-wave','questions-all','verify-only'],default='first-wave');ap.add_argument('--gate');args=ap.parse_args()
    before=integrity.audit(repair_metadata=True)
    assert not before['failures'],('PRODUCTION_INTEGRITY_FAILED',before['failures'])
    if args.action=='verify-only':
        print(json.dumps({'status':before['status'],'completed_calls':before['completed_calls'],'repairs':before['mechanical_repairs']},ensure_ascii=False));return
    assert not before['active_calls'],'EXISTING_UNSEALED_CLI_CALLS_REQUIRE_RECOVERY_OR_CURRENT_DRIVER_COMPLETION'
    # One finite v2 scheduler at a time. The legacy live v1 has no such lock,
    # so the unsealed-call guard above also prevents overlap with its calls.
    scheduler_lock=(BASE/'production_driver_v2.lock').open('a+')
    fcntl.flock(scheduler_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    roots=driver.read(BASE/'PLAN.json')['new_train_roots']
    if args.action=='second-wave':
        assert args.gate,'SECOND_WAVE_REQUIRES_FROZEN_GATE_PATH'
        frozen_gate=driver.read(args.gate);qm=module('oet1_current_gate',BASE/'scripts/questions.py');current_gate=qm.engineering_gate()
        assert frozen_gate['status']==current_gate['status']=='PASS','SECOND_WAVE_GATE_NOT_PASS'
        for key in ['checks','decisions','overall','by_root','by_slot']:assert frozen_gate[key]==current_gate[key],('SECOND_WAVE_GATE_DRIFT',key)
        driver.run_wave(roots[4:],'SECOND_FOUR_V2')
    elif args.action=='first-wave':driver.run_wave(roots[:4],'FIRST_FOUR_V2')
    else:
        from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
        selected=roots[:4] if args.action=='questions-first-wave' else roots
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(driver.question_unit,r,a):(r,a) for r in selected for a in ['GENERIC','TARGETED'] if (BASE/'sources'/r/a/'FINAL_SOURCE.json').exists()}
            while futures:
                done,_=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    root,arm=futures.pop(future)
                    try:result=future.result()
                    except Exception as error:result={'status':'FAIL','error':str(error)}
                    driver.log('QUESTION_UNIT_TERMINAL_V2',root=root,arm=arm,result=result)
                driver.status(heartbeat={'at_utc':driver.now(),'active_question_units':[list(x) for x in futures.values()],'driver':'v2'})
    after=integrity.audit(repair_metadata=True)
    out=BASE/'audit'/('PRODUCTION_INTEGRITY_'+args.action.replace('-','_')+'_'+integrity.now().replace(':','').replace('.','')+'.json')
    integrity.exclusive_json(out,after)
    print(json.dumps({'status':after['status'],'audit_binding':integrity.bind(out)},ensure_ascii=False))
if __name__=='__main__':main()
