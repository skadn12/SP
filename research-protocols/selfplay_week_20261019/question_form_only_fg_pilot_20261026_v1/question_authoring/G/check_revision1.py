import json,hashlib,re
from pathlib import Path
from datetime import datetime,timezone
B=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/question_form_only_fg_pilot_20261026_v1/question_authoring/G')
def bind(p):
 b=Path(p).read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def canonical(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
old=json.loads((B/'CANDIDATES_DRAFT.json').read_text());new=json.loads((B/'CANDIDATES_REVISION1.json').read_text());op={c['candidate_id']:c for c in old};inp=json.loads((B/'REVISION1_INPUT.json').read_text());target={i['candidate_id'] for i in inp['issues']};errs=[];lengths=[];quote_count=0;span_count=0;cache={};qcs=[];ocs=[]
if len(new)!=20:errs.append('candidate_count')
changed={c['candidate_id'] for c in new if c!=op[c['candidate_id']]}
if changed!=target:errs.append('changed_candidate_scope')
for c in new:
 cid=c['candidate_id'];o=op[cid];path=c['source_binding']['path']
 if path not in cache:cache[path]=json.loads(Path(path).read_text())
 s=cache[path];ts={t['turn_ref']:t for t in s['turns']}
 if c['source_binding']!=bind(path):errs.append([cid,'source_binding'])
 if c['source_status']!=o['source_status'] or c['warnings']!=o['warnings']:errs.append([cid,'warnings_status'])
 if c['gold']!=o['gold'] or len(c['gold'])!=int(c['slot_id'][2:]):errs.append([cid,'Gold'])
 if list(c['options'])!=list('ABCDEF') or list(c['item_analysis'])!=list('ABCDEF'):errs.append([cid,'schema'])
 qc=len(c['question'].split());qcs.append(qc)
 if not 30<=qc<=90:lengths.append([cid,'question',qc])
 for lab,opt in c['options'].items():
  oc=len(opt.split());ocs.append(oc)
  if not 10<=oc<=35:lengths.append([cid,lab,oc])
  a=c['item_analysis'][lab]
  if a['should_select']!=(lab in c['gold']):errs.append([cid,lab,'should_select'])
  for e in a['evidence']:
   quote_count+=1;t=ts.get(e['turn_ref'])
   if not t or e['speaker']!=t['speaker'] or e['quote'] not in t['text']:errs.append([cid,lab,'quote'])
 if c['slot_id']=='MC4':
  for lab,p in c['evidence_unit_preservation'].items():
   span_count+=1;text=ts[p['turn_ref']]['text'];oldspan=op[cid]['options'][lab].split('“',1)[1][:-1]
   if p['locator_type']=='full_turn':resolved=text
   elif p['locator_type']=='final_sentence':resolved=text.rsplit('. ',1)[-1]
   elif p['locator_type']=='clause_to_turn_end':
    matches=list(re.finditer(r'\bthen\b',text));assert len(matches)==1;resolved=text[matches[0].start():]
   else:raise AssertionError(p['locator_type'])
   if not (oldspan==resolved==p['original_exact_span']==p['resolved_revised_span']==c['item_analysis'][lab]['evidence'][0]['quote']):errs.append([cid,lab,'changed_evidence_unit'])
   if c['options'][lab]!=p['revised_public_locator'] or p['original_span_sha256']!=hashlib.sha256(resolved.encode()).hexdigest() or p['resolved_span_sha256']!=p['original_span_sha256']:errs.append([cid,lab,'span_hash_locator'])
for issue in inp['issues']:
 if canonical(op[issue['candidate_id']])!=issue['initial_candidate_sha256']:errs.append(['initial_candidate_fingerprint',issue['candidate_id']])
freeze=json.loads((B/'INITIAL_DRAFT_FREEZE.json').read_text())
if bind(B/'CANDIDATES_DRAFT.json')!=freeze['candidate_binding']:errs.append('initial_draft_modified')
for p in json.loads((B/'RECEIPTS_FREEZE.json').read_text())['receipts']:
 if bind(p['path'])!=p:errs.append(['receipt_modified',p['path']])
log=json.loads((B/'REVISION1_CHANGELOG.json').read_text())
if bind(B/'REVISION1_INPUT.json')!=log['revision_input_binding']:errs.append('revision_input_modified')
if bind(B/'CANDIDATES_REVISION1.json')!=log['revision_binding']:errs.append('revision_binding_mismatch')
report={'arm':'G','author_id':'form_only_author_g','checked_at_utc':datetime.now(timezone.utc).isoformat(),'revision_binding':bind(B/'CANDIDATES_REVISION1.json'),'candidate_count':20,'changed_candidates':len(changed),'structurally_identical_untargeted_candidates':20-len(changed),'option_count':sum(len(c['options']) for c in new),'evidence_quote_count':quote_count,'exact_original_unit_preservations_checked':span_count,'Gold_cardinality_and_should_select_unchanged':True,'initial_draft_input_and_frozen_receipts_unchanged':True,'question_word_count_range':[min(qcs),max(qcs)],'option_word_count_range':[min(ocs),max(ocs)],'errors':errs,'length_flags':lengths,'result':'PASS' if not errs and not lengths else 'FAIL','scope':'Local quote/hash/schema/cardinality and evidence-span checks only; no independent semantic re-review, production gate or Solver/model calls.'}
(B/'REVISION1_MECHANICAL_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2));assert not errs and not lengths
