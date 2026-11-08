"""Run the preserved first-round generator as a no-feedback control on six R03 roots."""
import argparse, json
from dataclasses import replace
from pathlib import Path
from ..core import canonical_json, sha256_bytes, sha256_file, atomic_json, atomic_text
from ..live_backends import load_challenger_config, OpenAICompatibleChallengerClient
from ..ledger import RequestLedger
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_qa, GENERATOR_VERSION, GENERATOR_TEMPERATURE

RUNS=Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs')
SOURCE=RUNS/'h4_round1_r03_feedback_batch_generation_revision1_live_2'
ROOTS=('cc-train-episode-122711','cc-train-episode-254852','cc-train-episode-136202','cc-train-episode-94289','cc-train-episode-198480','cc-train-episode-197635')

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=RUNS/'h4_round1_r03_first_round_no_feedback_6roots_live_1'); p.add_argument('--live',action='store_true'); a=p.parse_args()
 if a.output.exists() and any(a.output.iterdir()): raise RuntimeError('output_exists')
 a.output.mkdir(parents=True)
 rows=[]
 for root in ROOTS:
  src=SOURCE/root/'dialogue_source.json'
  d=json.loads(src.read_text()); rows.append({'root_id':root,'dialogue_path':str(src),'dialogue_sha256':sha256_file(src),'turn_count':len(d['turns'])})
 plan={'version':'r03_first_round_no_feedback_control_v1','roots':list(ROOTS),'generator_version':GENERATOR_VERSION,'temperature':GENERATOR_TEMPERATURE,'feedback_context':None,'history_visible_to_generator':False,'source_manifest':rows,'planned_primary_requests_per_root':7,'notes':'Preserved first-round K3 generator; no feedback_context is passed.'}
 atomic_json(a.output/'control_manifest.json',plan); atomic_text(a.output/'control_manifest.json.sha256',sha256_file(a.output/'control_manifest.json')+'  control_manifest.json\n')
 if not a.live:
  print(json.dumps({'status':'OFFLINE_PLAN_ONLY','root_count':6})); return
 cfg=load_challenger_config(Path('/root/autodl-tmp/gxy-selfplay/config.json'))
 client=OpenAICompatibleChallengerClient(replace(cfg,temperature=GENERATOR_TEMPERATURE,enable_thinking=False,timeout_seconds=max(180,int(cfg.timeout_seconds))))
 reports=[]
 try:
  for row in rows:
   root=row['root_id']; rd=a.output/root; rd.mkdir()
   d=json.loads(Path(row['dialogue_path']).read_text())
   ledger=RequestLedger(rd,sha256_bytes(canonical_json(plan).encode()),'live')
   report=generate_root_qa(expanded=d,root_dir=rd,root_id=root,api_config=Path('/root/autodl-tmp/gxy-selfplay/config.json'),ledger=ledger,client=client,feedback_context=None)
   reports.append({'root_id':root,**report,'request_count':ledger.summary()['request_count']})
   atomic_json(a.output/'progress.json',{'completed_roots':len(reports),'total_roots':6,'roots':reports})
 finally:
  if hasattr(client, 'close'):
   client.close()
 final={'status':'COMPLETE','control':'first_round_no_feedback','root_count':6,'roots':reports,'total_candidates':sum(int(x.get('candidate_count') or 0) for x in reports),'total_valid_candidates':sum(int(x.get('valid_candidate_count') or 0) for x in reports),'total_requests':sum(int(x.get('request_count') or 0) for x in reports),'question_type_policy':'2 single + 1 multi + 1 ordering per guided request; 4 single + 2 multi + 2 ordering per free request; repairs conditional','semantic_audit_calls':0,'solver_calls':0,'training_calls':0}
 atomic_json(a.output/'final_report.json',final); atomic_text(a.output/'final_report.json.sha256',sha256_file(a.output/'final_report.json')+'  final_report.json\n'); print(json.dumps(final,ensure_ascii=False))
if __name__=='__main__': main()
