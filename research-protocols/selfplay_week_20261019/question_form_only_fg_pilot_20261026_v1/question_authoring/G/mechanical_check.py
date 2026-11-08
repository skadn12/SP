import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
B=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/question_form_only_fg_pilot_20261026_v1/question_authoring/G')
def bind(p):
 b=Path(p).read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
x=json.loads((B/'CANDIDATES_DRAFT.json').read_text()); plans=json.loads((B/'TASK_PLANS_FIXED.json').read_text()); pm={p['candidate_id']:p for p in plans}; errors=[]; lengths=[]; qcs=[]; ocs=[]; ec=0
assert len(x)==20
for c in x:
 source=Path(c['source_binding']['path']); s=json.loads(source.read_text()); ts={t['turn_ref']:t for t in s['turns']}; w=json.loads((B/f"SOURCE_WARNINGS_{c['root_id'].split('-')[-1]}.json").read_text()); p=pm[c['candidate_id']]
 if c['source_binding']!=bind(source):errors.append([c['candidate_id'],'source_binding'])
 if c['source_status']!=s['source_status'] or c['warnings']!=w['warnings']:errors.append([c['candidate_id'],'status_or_warnings'])
 if set(c['options'])!=set('ABCDEF') or set(c['item_analysis'])!=set('ABCDEF'):errors.append([c['candidate_id'],'labels'])
 if len(c['gold'])!=int(c['slot_id'][2:]):errors.append([c['candidate_id'],'Gold_quota'])
 if len(set(c['options'].values()))!=6:errors.append([c['candidate_id'],'duplicate_option'])
 if c['preserved_plan_candidate_id']!=c['candidate_id'] or c['preserved_plan_operation']!=p['planned_operation'] or c['preserved_source_relations']!=p['source_relations']:errors.append([c['candidate_id'],'fixed_plan'])
 qc=len(c['question'].split());qcs.append(qc)
 if not 30<=qc<=90:lengths.append([c['candidate_id'],'question',qc])
 for k,o in c['options'].items():
  oc=len(o.split());ocs.append(oc)
  if not 10<=oc<=35:lengths.append([c['candidate_id'],k,oc])
  ia=c['item_analysis'][k]
  if ia['should_select']!=(k in c['gold']):errors.append([c['candidate_id'],k,'Gold_relation'])
  if not ia['evidence']:errors.append([c['candidate_id'],k,'missing_evidence'])
  for e in ia['evidence']:
   ec+=1;t=ts.get(e['turn_ref'])
   if not t or e['speaker']!=t['speaker'] or e['quote'] not in t['text']:errors.append([c['candidate_id'],k,'quote'])
  if not ia['rationale'] or (not ia['should_select'] and not ia['changed_relation']):errors.append([c['candidate_id'],k,'missing_analysis'])
 if not c['r03_necessity']:errors.append([c['candidate_id'],'missing_necessity'])
for b in json.loads((B/'RECEIPTS_FREEZE.json').read_text())['receipts']:
 if bind(b['path'])!=b:errors.append(['frozen_receipt_changed',b['path']])
report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'author_id':'form_only_author_g','candidate_binding':bind(B/'CANDIDATES_DRAFT.json'),'candidate_count':len(x),'option_count':sum(len(c['options']) for c in x),'evidence_quote_count':ec,'question_word_count_range':[min(qcs),max(qcs)],'option_word_count_range':[min(ocs),max(ocs)],'checks':['20 fixed candidate IDs and preserved plan operation/source relations','A–F option and item-analysis labels','Gold quotas MC2..MC6 = 2..6 and should_select agreement','source path, SHA256, bytes, source_status and exact warnings preserved','every evidence quote is a verbatim substring at the stated turn and speaker','question and option approximate length ranges','all frozen receipt hashes unchanged'],'errors':errors,'length_flags':lengths,'result':'PASS' if not errors and not lengths else 'FAIL','semantic_review':'Not an independent review; independent content review remains pending.','production_gate':'Not read or run by isolated author; parent will execute existing audit_candidate.'}
(B/'MECHANICAL_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
assert not errors and not lengths,(errors,lengths)
print(json.dumps(report,ensure_ascii=False,indent=2))
