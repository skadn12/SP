#!/usr/bin/env python3
"""Seal only day2/fact_memory preparation; never seal or mutate parent/day1."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 target=HERE/'PREPARATION_FREEZE.json'
 if target.exists():raise SystemExit('Already sealed; do not overwrite.')
 v=json.loads((HERE/'VALIDATION_SEALED.json').read_text());assert v['status']=='PASS'
 assert v['actual_generation_scenes']==0 and v['actual_model_calls']==0 and v['memory_semantic_pass']==0
 protocol=json.loads((HERE/'PILOT_PROTOCOL.json').read_text());assert protocol['generation']['maximum_memory_update_review_calls']==12
 rows=[]
 for p in sorted(HERE.rglob('*')):
  if not p.is_file() or p.name=='PREPARATION_FREEZE.json' or '__pycache__' in p.parts or p.suffix=='.pyc':continue
  rows.append({'path':str(p.relative_to(HERE)),'sha256':sha(p),'bytes':p.stat().st_size})
 manifest={'status':'PREPARATION_FROZEN_ONLY','created_at_utc':datetime.now(timezone.utc).isoformat(),
  'scope':str(HERE),'does_not_seal_parent_day2_or_day1':True,'file_count':len(rows),'files':rows,
  'roots':6,'source_turns':1095,'memory_candidate_facts':144,'memory_semantic_pass':0,'memory_coverage':'unknown',
  'planned_scenes':54,'actual_scenes':0,'actual_runtime_model_calls':0,'mechanical_checks':v['check_count'],
  'pending_execution_gates':['6 initial memory semantic reviews at exact snapshot hash',
   'Backend identity/settings/usage preflight and exclusive arm prompt payloads',
   'Hash-bound accepted-branch receipts and source-first memory update maintenance review before scenes2/3',
   'Actual54scene generation/review including all failures and costs; no pilot outcome yet']}
 with target.open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
 for r in rows:assert sha(HERE/r['path'])==r['sha256']
 print(json.dumps({k:v for k,v in manifest.items() if k not in ['files','pending_execution_gates']}))
if __name__=='__main__':main()
