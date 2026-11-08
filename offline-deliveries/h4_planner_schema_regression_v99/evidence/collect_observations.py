"""Summarize immutable actual traces; no engine import and no provider calls."""
from pathlib import Path
import hashlib,json
from collections import Counter

B=Path(__file__).resolve().parents[1]
R=B/'regression'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

report=read(R/'REPORT.json');rows=[];inputs=outputs=0;roles=Counter();http=Counter()
for d in sorted((R/'requests').iterdir()):
 p=read(d/'PROOF.json');roles[p['role']]+=1
 response=read(d/'RESPONSE.json') if (d/'RESPONSE.json').exists() else None
 v=read(d/'VALIDATION.json') if (d/'VALIDATION.json').exists() else None
 m=response['metadata'] if response else {}
 inputs+=m.get('input_tokens',0);outputs+=m.get('output_tokens',0)
 for h in sorted(d.glob('HTTP_RESULT_*.json')):http[str(read(h).get('http_status'))]+=1
 identity=None
 if p['role']=='planner' and response:
  try:
   scene=json.loads(response['text'])['scenes']
   identity=bool(scene and isinstance(scene[0].get('event_identity'),dict))
  except (ValueError,KeyError,IndexError,TypeError):pass
 rows.append({'directory':str(d),'root_id':p['regression_cell_root'],'arm':p['experiment_arm'],
  'role':p['role'],'plan_identity_present':identity,'validation':v,
  'input_tokens':m.get('input_tokens'),'output_tokens':m.get('output_tokens'),
  'finish_reason':m.get('finish_reason'),'response_model':m.get('returned_model'),
  'capacity_status':p['capacity']['status'],'local_truncation':p['capacity']['truncation']})
arms={}
for arm in ('control','treatment'):
 cs=[c for c in report['cells'] if c['arm']==arm]
 first=[next(x for x in rows if x['root_id']==c['root_id'] and x['arm']==arm and x['role']=='planner') for c in cs]
 arms[arm]={'cells':len(cs),'first_identity_present':sum(x['plan_identity_present'] is True for x in first),
  'first_structural_pass':sum(x['validation'] and x['validation']['status']=='PASS' for x in first),
  'valid_nonempty_plan_within_budget':sum(c['structurally_valid_nonempty_plan'] for c in cs),
  'event_review_reached':sum(c['event_review_reached'] for c in cs),
  'event_review_structural_pass':sum(c['event_review_structurally_valid'] for c in cs),
  'event_review_pass_without_replan':sum(c['event_review_structurally_valid'] and c['event_review_requires_replan'] is False for c in cs),
  'event_review_requires_replan':sum(c['event_review_requires_replan'] is True for c in cs)}
preserved=read(B/'PRESERVATION_BEFORE.json');changed=[x['path'] for x in preserved['files'] if sha(x['path'])!=x['sha256']]
write(B/'evidence/PRESERVATION_AFTER.json',{'status':'PASS' if not changed else 'FAIL','checked_files':len(preserved['files']),'changed':changed})
out={'status':'COMPLETE','classification':'exposed regression, no fresh yield','report_sha256':sha(R/'REPORT.json'),
 'arms':arms,'logical_requests':report['logical_requests'],'http_attempts':report['http_attempts'],
 'roles':dict(roles),'http_status_counts':dict(http),'input_tokens':inputs,'output_tokens':outputs,'total_tokens':inputs+outputs,
 'token_scope':'Received successful completion metadata; no provider errors silently excluded from HTTP counts.',
 'requests':rows,'fresh_source_count':0,'new_dialogue_turns':0,'new_questions':0,'selected_questions':0,'training_jobs':0,
 'paired_comparison_scope':'First Planner requests; retries use branch-specific actual responses and original arm feedback.'}
write(B/'evidence/ACTUAL_REGRESSION_OBSERVATIONS.json',out)
print(json.dumps({k:v for k,v in out.items() if k!='requests'},ensure_ascii=False))
