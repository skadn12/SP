"""Merge signed reviews and additive annotations; performs no semantic judging."""
from qa_common import D,P,read,sha,binding,dump,resolve
from copy import deepcopy
from datetime import datetime,timezone
from collections import Counter
import json,sys
reviews={};overrides={};provenance={};canonical={}
publics={read(p)['candidate_id']:p for p in (D/'candidates').glob('*/*/public/*.json')}
for p in sorted((D/'qa/independent').rglob('COMPARISON.json')):
 c=read(p)
 for row in c['rows']:
  cid=row['candidate_id'];assert cid not in reviews,('duplicate independent QA',cid)
  row=deepcopy(row);row['reviewer']=c['reviewer'];row['reviewed_at']=c['created_at_utc'];row['comparison_binding']=binding(p)
  for key in ['independent_reconstruction','public_binding','private_binding']:
   row[key]['path']=str(resolve(row[key]['path']))
  row['addenda_bindings']=[];reviews[cid]=row;provenance[cid]=[binding(p)]
  pub=publics[cid];a=read(pub.parent.parent/'private'/pub.name)
  if read(pub)['question_type']=='ordering':
   it=a.get('items',{})
   overrides[cid]={'occurrences_by_label':{k:deepcopy(it[k]['all_mentions'] if k in it else a['all_mentions'][k]) for k in 'ABCDEF'}}
  rr=read(row['independent_reconstruction']['path'])
  canonical[cid]=next(x for x in rr['rows'] if x['candidate_id']==cid).get('canonical_event_keys',{})

def norm(o,source):
 t=o['turn_ref'];quote=source[t]['text']
 assert not o.get('quote') or o['quote'] in quote,('bad overlay quote',t)
 return dict(turn_ref=t,quote=quote,included_for_event_key=o.get('included_for_event_key',o.get('included')),rationale=o.get('rationale',o.get('reason','Explicit signed semantic annotation.')))
def src(cid):return {t['turn_ref']:t for t in read(P/'sources'/f"{read(publics[cid])['root_id']}.json")['turns']}
def replace_full(cid,bylabel,p):
 s=src(cid)
 for label,occ in bylabel.items():
  assert label in 'ABCDEF'
  overrides[cid]['occurrences_by_label'][label]=[norm(o,s) for o in occ]
 if binding(p) not in provenance[cid]:provenance[cid].append(binding(p))
def update_occ(cid,label,o,p):
 s=src(cid);o=norm(o,s);occ=overrides[cid]['occurrences_by_label'][label]
 occ[:]=[x for x in occ if x['turn_ref']!=o['turn_ref']];occ.append(o)
 if binding(p) not in provenance[cid]:provenance[cid].append(binding(p))
# A middle reviewers' signed comparison contains the full final occurrence table.
for cid,row in reviews.items():
 notes=row.get('annotation_notes',[])
 if isinstance(notes,dict):notes=[notes]
 for note in notes:
  if isinstance(note,dict) and 'occurrences_by_label' in note:replace_full(cid,note['occurrences_by_label'],row['comparison_binding']['path'])
# Main review groups are disjoint. Original files are never overwritten.
for p in [D/'qa/ANNOTATION_OVERRIDES.json',D/'qa/ANNOTATION_OVERRIDES_A_TAIL.json',D/'qa/ANNOTATION_OVERRIDES_A_EXTRA.json']:
 if p.exists():
  for cid,row in read(p)['rows'].items():replace_full(cid,row['occurrences_by_label'],p)
# B tail signed delta syntax; ignore explicitly non-candidate metadata keys.
for p in sorted((D/'qa/independent/review_B').glob('*/ANNOTATION_OVERRIDES.json')):
 j=read(p);root=j['root_id']
 for local,bylabel in j['occurrences_by_label'].items():
  if not local.startswith('ordering-'):continue
  cid=f'B-week1-stage1-{root}-{local}'
  for label,ops in bylabel.items():
   occ=overrides[cid]['occurrences_by_label'][label]
   occ[:]=[o for o in occ if o['turn_ref'] not in ops.get('remove_turn_refs',[])]
   for key in ['add','changes','replace']:
    for o in ops.get(key,[]):update_occ(cid,label,o,p)
# Later full-source occurrence review can demote an earlier comparison PASS.
for p in sorted((D/'qa/independent/review_B_early').glob('*/ANNOTATION_SEMANTIC_REVIEW.json')):
 for r in read(p)['rows']:
  cid=r['candidate_id'];row=reviews[cid]
  assert sha(row['public_binding']['path'])==r['public_binding']['sha256']
  row['addenda_bindings'].append(binding(p));provenance[cid].append(binding(p))
  row['content_status']=r['final_content_status'];row['target_status']=r['final_target_status']
  for finding in r.get('semantic_findings_and_overlays',[]):
   if finding.get('status_override')=='HOLD':
    assert row['content_status']=='HOLD'
    row['independent_semantic_rationale']+=' | '+finding['reason']
   if finding.get('independently_revised_earliest_anchor'):row['anchor_agreement']=False
   if finding.get('reviewer_canonical_key_correction'):canonical[cid][finding['label']]=finding['reviewer_canonical_key_correction']
   for o in finding.get('mention_overrides',[]):update_occ(cid,finding['label'],o,p)
   for o in finding.get('additional_mentions',[]):
    if 'included_for_event_key' in o:update_occ(cid,o['label'],o,p)
ep=D/'qa/independent/review_B_early/EFFECTIVE_DECISIONS.json'
if ep.exists():
 for r in read(ep)['rows']:
  cid=r['candidate_id'];row=reviews[cid]
  row.update(content_status=r['content_status'],target_status=r['target_status'],gold_agreement=r['gold_agreement'],anchor_agreement=r['anchor_agreement'],independent_semantic_rationale=r['rationale'])
  row['addenda_bindings'].append(binding(ep));provenance[cid].append(binding(ep))
  canonical[cid].update(r.get('canonical_event_keys_overrides',{}))
for cid,table in overrides.items():
 for label,occ in table['occurrences_by_label'].items():
  assert all(isinstance(x.get('included_for_event_key'),bool) for x in occ),(cid,label)
  assert len({x['turn_ref'] for x in occ})==len(occ),(cid,label,'duplicate occurrence')
  occ.sort(key=lambda x:next(i for i,t in enumerate(read(P/'sources'/f"{read(publics[cid])['root_id']}.json")['turns']) if t['turn_ref']==x['turn_ref']))
missing=set(publics)-set(reviews)
if missing:raise RuntimeError('Independent QA missing: '+str(sorted(missing)))
assert len(publics)==175
out=dict(schema_version='BATCH2_EFFECTIVE_SIGNED_QA_V1',created_at_utc=datetime.now(timezone.utc).isoformat(),rows=reviews,occurrence_overrides=overrides,canonical_event_keys=canonical,provenance=provenance,counts=dict(Counter(x['content_status'] for x in reviews.values())),limits='Mechanical merge of explicit signed judgments. Post-seal semantic holds override earlier PASS. Originals preserved. No solver, difficulty or causal effect assessed.')
if '--dry-run' not in sys.argv:dump(D/'qa/EFFECTIVE_REVIEW_INDEX.json',out)
print(json.dumps(dict(counts=out['counts'],reviews=len(reviews),ordering_overlays=len(overrides),dry_run='--dry-run' in sys.argv)))
