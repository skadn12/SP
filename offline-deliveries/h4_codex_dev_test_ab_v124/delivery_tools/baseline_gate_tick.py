"""Parent-only nonsemantic baseline aggregation; never author a review."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess, sys
B=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(B/'tools'))
from phase_gate_v124 import require_phase
def read(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def now():return datetime.now(timezone.utc).isoformat()
def tick():
 if not (B/'evidence/DEV_PHASE_CLOSED.json').is_file():
  print(json.dumps({'status':'TEST_CLOSED_NO_SOURCE_READS'}));return
 for row in read(B/'MANIFEST.json')['roots']:
  if row['split']!='test':continue
  require_phase(B,row['root_id'])
  u=Path(row['unit_path']);out=u/'inputs/CONTINUATION_CONSTRAINTS.json'
  if out.exists():continue
  inputs=[u/'baseline_reviews'/('reader'+str(i+1))/'REVIEW.json' for i in range(2)]
  if not all(p.is_file() for p in inputs):continue
  folder=u/'evidence/PARENT_STAGES';folder.mkdir(parents=True,exist_ok=True)
  receipt=folder/'BASELINE_GATE_RUN.json';lock=folder/'BASELINE_GATE_RUN.lock'
  if receipt.exists():continue
  try:lock.mkdir()
  except FileExistsError:continue
  cmd=[sys.executable,'-B',str(B/'tools/baseline.py'),'gate','--root',row['root_id'],'--executor','/root']
  start=now();result=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
  value={'schema_version':'v124_parent_baseline_gate_receipt_v1','root_id':row['root_id'],'executor':'/root','started_at_utc':start,'finished_at_utc':now(),'command':cmd,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'input_bindings':[bind(p) for p in inputs],'output_binding':bind(out) if out.is_file() else None,'semantic_judgment_created':False,'execution_script_binding':bind(Path(__file__))}
  with receipt.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
  lock.rmdir();print(json.dumps({'root_id':row['root_id'],'returncode':result.returncode,'decision':read(out)['decision'] if out.is_file() else None,'stderr':result.stderr},ensure_ascii=False),flush=True)
if __name__=='__main__':tick()
