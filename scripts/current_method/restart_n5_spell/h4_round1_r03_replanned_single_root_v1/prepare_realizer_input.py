"""Offline Realizer-input adapter for a Gate-complete initial cap9 run."""
import argparse,json
from pathlib import Path
from copy import deepcopy
from ..h4_round1_r03_incremental_plan_retention_v1 import planner_relations
from ..h4_round1_r03_incremental_plan_retention_v1.prompts import render_realizer

def prepare(run_dir,bundle_dir,output):
 run_dir,bundle_dir,output=map(lambda p:Path(p).resolve(),(run_dir,bundle_dir,output)); output.mkdir(parents=True,exist_ok=False)
 b=json.loads((bundle_dir/'prepared/initial/bundle.json').read_text()); raw=json.loads((run_dir/'planner/raw_response.json').read_text())
 comp=planner_relations.compile_initial_output(raw,parent=b['parent'],candidates=b['candidates'],verified_tracks=b['verified_tracks'],max_new_candidates=9)
 if not comp['valid']: raise ValueError('planner_compile_invalid:'+str(comp['errors']))
 decisions=json.loads((run_dir/'gate/candidate_decisions.json').read_text()); passed={x['candidate_id'] for x in decisions if x.get('decision')=='pass'}
 registry=comp['prepared']['candidate_registry']; ordered=[ref for ref in registry if ref in passed][:6]
 if len(ordered)!=6: raise ValueError('realizer_requires_six_passed_materialized_scenes')
 scenes=[deepcopy(registry[ref]['scene']) for ref in ordered]
 batches=[]
 for i in range(0,6,2):
  built=render_realizer(parent=b['parent'],plan={'scenes':scenes[i:i+2]},mode='text_free')
  req={'role':'realizer','call_id':f'realizer-{i//2+1:03d}','messages':built['messages'],'max_tokens':8192,'prompt_version':built['prompt_version'],'identity':{'root_id':b['identity']['root_id'],'round_index':3,'batch_index':i//2+1,'candidate_refs':ordered[i:i+2]},'model':json.loads((run_dir/'model_config.json').read_text())['model'],'temperature':json.loads((run_dir/'model_config.json').read_text())['temperature'],'thinking':False}
  batches.append(req); (output/f'{i//2+1:03d}_request.json').write_text(json.dumps(req,ensure_ascii=False,indent=2)+'\n'); (output/f'{i//2+1:03d}_prompt.txt').write_text(built['prompt'])
 manifest={'version':'h4_round1_r03_realizer_input_adapter_cap9_v1','status':'OFFLINE_REALIZER_INPUT_PREPARED_NOT_GENERATED','root_id':b['identity']['root_id'],'source_run':str(run_dir),'planner_sha256':planner_relations.sha(raw),'gate_decisions_sha256':planner_relations.sha(decisions),'selected_candidate_refs':ordered,'scene_count':6,'batch_count':3,'input_mode':'text_free','historical_qa_included':False,'gold_included':False,'solver_included':False,'relation_sidecar':json.loads((run_dir/'planner/relation_metadata.json').read_text()),'external_calls':{'realizer':0,'qa':0,'solver':0,'training':0,'memorax_write':0}}
 (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'); return manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);print(json.dumps(prepare(p.parse_args().run_dir,p.parse_args().bundle,p.parse_args().output),ensure_ascii=False))
if __name__=='__main__': main()
