"""Rejection checks only; temporary synthetic metadata, no model or generation."""
import importlib.util
import json
from pathlib import Path
import tempfile
spec=importlib.util.spec_from_file_location('prepare_next',Path(__file__).with_name('prepare_next_generation.py'))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
H=Path(__file__).resolve().parents[1]
cases=[]
with tempfile.TemporaryDirectory(prefix='handoff_prepare_rejections_') as tmp:
 p=Path(tmp)
 def rejected(name,state,scope='exploratory'):
  output=p/(name+'_output')
  try:mod.prepare(state,output,scope)
  except mod.GateRejected as error:
   assert not output.exists(),name+' unexpectedly created output'
   cases.append({'case':name,'status':'REJECTED_EXPECTED','reason':str(error),'output_created':False});return
  raise AssertionError(name+' unexpectedly accepted')
 rejected('missing_state',p/'does_not_exist.json')
 state=json.loads((H/'HANDOFF_STATE.json').read_text())
 state['artifacts']['feedback_snapshot']['sha256']='0'*64
 broken=p/'bad_sha_state.json';broken.write_text(json.dumps(state))
 rejected('changed_bound_sha',broken)
 rejected('formal_scope_without_reproduced',H/'HANDOFF_STATE.json','formal')
 result={'status':'PASS','synthetic_rejection_tests_only':True,'new_solver_calls':0,'new_roots':0,'new_questions':0,'cases':cases}
 out=Path(__file__).with_name('SELFTEST_PREPARE_NEXT_RESULT.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
