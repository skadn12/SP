#!/usr/bin/env python3
"""Continue to final closeout once the already-running source worker freezes."""
import os,time,traceback
from codex_client import BASE,atomic,bind,load,now,start_token,status

def main():
 while not (BASE/'SOURCE_FREEZE.json').exists():
  worker=load(BASE/'RUN_STATUS.json').get('worker',{})
  if worker.get('pid') and start_token(worker['pid'])!=worker.get('start_token'):raise RuntimeError('source worker stopped before freeze')
  time.sleep(5)
 atomic(BASE/'process_registry'/'pipeline.json',{'pid':os.getpid(),'start_token':start_token(os.getpid()),'kind':'pipeline','started_at_utc':now()})
 from question_resume import run as questions
 from solver_resume import run as solver
 from analysis_resume import run as analysis
 from closeout_resume import run as closeout
 try:
  questions();solver();analysis()
 except Exception as e:
  atomic(BASE/'PIPELINE_FAILURE.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});status('pipeline_failed',pipeline_error=repr(e))
 finally:closeout()
if __name__=='__main__':main()
