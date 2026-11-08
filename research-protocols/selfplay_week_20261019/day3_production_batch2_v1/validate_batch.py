"""Check hashes, frozen quotas, source spans and review lineage, no semantic QA."""
from pathlib import Path
from collections import Counter
from itertools import combinations
import json,hashlib,sys
import jsonschema
sys.path.insert(0,str(Path(__file__).resolve().parent/'qa'))
from qa_common import resolve
D=Path(__file__).resolve().parent
P=D.parent/'day3_production_v1'
def read(p):return json.loads(resolve(p).read_text())
def sha(p):return hashlib.sha256(resolve(p).read_bytes()).hexdigest()
errors=[];checks=0
def check(ok,message):
 global checks
 checks+=1
 if not ok:errors.append(message)
f=read(D/'EXECUTION_FREEZE.json')
old={s['candidate_slot_id']:s for s in read(P/'EXECUTION_FREEZE.json')['slots']}
slots={s['candidate_slot_id']:s for s in f['slots']}
check(len(slots)==216,'216 inherited slots expected')
for cid,s in slots.items():
 for key in ['slot_id','root_id','arm','question_type','local_slot','target_gold_count','design_id']:
  check(s[key]==old[cid][key],f'{cid} inherited {key} changed')
 check(sha(s['source_binding']['path'])==s['source_binding']['sha256'],f'{cid} source changed')
manifest=read(P/'FINAL_MANIFEST.json')
check(sha(P/'FINAL_MANIFEST.json')==f['sealed_first_batch']['sha256'],'sealed first-batch manifest itself changed')
entries=manifest.get('files',manifest.get('rows',manifest.get('artifacts',[])))
if isinstance(entries,dict):entries=[{'path':k,'sha256':v if isinstance(v,str) else v['sha256']} for k,v in entries.items()]
for entry in entries:
 ep=Path(entry['path'])
 if not ep.is_absolute():ep=P/ep
 check(sha(ep)==entry['sha256'],f'Sealed first-batch artifact changed {ep}')
for srp in sorted((D/'source_reviews').glob('cc-*.json')):
 sr=read(srp)
 check(sr['actual_read_turn_count']==len(read(sr['source_binding']['path'])['turns']),f'{srp} read count')
 check(sha(sr['source_binding']['path'])==sr['source_binding']['sha256'],f'{srp} source hash')
 check(sr['full_history_scope']['all_parent_R01_R02_semantically_read'],f'{srp} incomplete source read')
schema=read(P/'design/PUBLIC_DESIGN_LIBRARY.json')['schemas']['candidate_record']
validator=jsonschema.Draft202012Validator(schema,format_checker=jsonschema.FormatChecker())
directory=D/'final_records' if (D/'final_records').exists() else D/'records'
paths=sorted(directory.glob('*.json'))
public_ids={read(p)['candidate_id'] for p in (D/'candidates').glob('*/*/public/*.json')}
check(len(public_ids)==175,'expected all175 actual authored candidates')
check({p.stem for p in paths}==public_ids,'incomplete or extra normalized/final record coverage')
effective_path=D/'qa/EFFECTIVE_REVIEW_INDEX.json'
dedup_path=D/'dedup/DECISIONS.json'
effective=read(effective_path)
dedup={r['candidate_id']:r for r in read(dedup_path)['rows']}
check(set(effective['rows'])==public_ids,'incomplete effective QA coverage')
check(set(dedup)==public_ids,'incomplete semantic dedup coverage')
for path in paths:
 r=read(path);cid=r['candidate_id'];s=slots[cid]
 q=effective['rows'][cid];dd=dedup[cid]
 prov=read(D/'record_provenance'/(cid+'.json'))
 check(prov['effective_review_index']['sha256']==sha(effective_path),'stale normalized effective QA')
 check(prov['dedup']['sha256']==sha(dedup_path),'stale normalized dedup')
 for x in prov['signed_review_and_annotation_sources']:
  check(sha(x['path'])==x['sha256'],'changed signed annotation/review source')
 check((r['qa_review']['status']=='PASS')==(q['content_status']=='PASS'),'record contradicts effective content QA')
 check((r['qa_review']['gold_independently_reconstructed']=='PASS')==q['gold_agreement'],'record contradicts independent Gold agreement')
 check((r['target_realization']['target_implemented']=='YES')==(q['target_status']=='PASS'),'record contradicts effective target QA')
 expected_accept=q['content_status']=='PASS' and q['target_status']=='PASS' and dd['duplicate_review']=='PASS' and not dd.get('within_arm_duplicate',False)
 check(r['disposition']['state'].startswith('ACCEPT')==expected_accept,'record bypasses current QA/dedup gate')
 check(r['fact_family']=={k:dd[k] for k in ['family_id','atomic_relation_keys','related_candidate_ids','reuse_class','duplicate_review']},'record differs from signed fact-family decision')
 for e in validator.iter_errors(r):check(False,f'{cid} schema {list(e.path)} {e.message}')
 check(r['root_id']==s['root_id'] and r['arm']==s['arm'],'record slot identity changed')
 check(r['source_review']['status']=='PASS','source not admitted')
 check(r['budget']['initial_used']==1 and r['budget']['content_clarification_used']<=1,'budget exceeded')
 check(r['budget']['solver_calls']==0,'unexpected solver')
 source=read(r['source_binding']['source_path']);turns={t['turn_ref']:(i,t) for i,t in enumerate(source['turns'])}
 refs={x['ref_id']:x for x in r['evidence_span']['evidence_refs']}
 for e in refs.values():
  i,t=turns[e['turn_id']];pos=e['source_position']
  check(pos['global_turn_index']==i,'source position mismatch')
  check(e['quote']==t['text'][pos['span_start']:pos['span_end']],'quote mismatch')
 for h in r['generation_history']:
  check(h['actual_cap_support']=='UNKNOWN' and h['actual_output_tokens'] is None and h['request_id'] is None,'unsupported provider metadata')
  check(sha(h['raw_response_path'])==h['candidate_content_sha256'],'initial draft hash changed')
 if r['question_type']=='multi_select':
  m=r['multi_select'];g=[x['option_id'] for x in m['options'] if x['source_verdict']=='SUPPORTED']
  if r['qa_review']['status']=='PASS':
   check(len(g)==s['target_gold_count'],'MC inherited gold count changed')
   check(set(g)==set(m['gold_option_ids']),'MC evidence/gold mismatch')
  else:
   check(not r['disposition']['state'].startswith('ACCEPT'),'held MC entered accepted pool')
   if set(g)!=set(m['gold_option_ids']):check(m['gold_count_verified']=='FAIL','held MC discrepancy unflagged')
  if r['qa_review']['status']=='PASS':check(all(x['source_verdict'] in ['SUPPORTED','REFUTED'] for x in m['options']),'unrefuted accepted distractor')
  if r['disposition']['state'].startswith('ACCEPT') and s['design_id']=='MC_FINAL_STATE_REVISION':
   check(len(set(r['evidence_span']['turn_ids']))>=2,'accepted revision lacks cross-turn source chain')
 else:
  o=r['ordering'];anchors={x['item_id']:refs[x['selected_ref_id']] for x in o['items']}
  positions={k:(x['source_position']['global_turn_index'],x['source_position']['span_start']) for k,x in anchors.items()}
  check(len(set(positions.values()))==6,'ordering position collision')
  check(sorted(positions,key=positions.get)==o['gold_item_ids'],'ordering gold is not source order')
  groups=Counter(x['scene_id'] for x in anchors.values())
  check(sorted(groups.values(),reverse=True)==o['scene_event_counts'],'wrong scene profile')
  within=sum(anchors[a]['scene_id']==anchors[b]['scene_id'] for a,b in combinations(anchors,2))
  check(within==o['within_scene_pair_count'],'wrong within-scene pairs')
  for item in o['items']:
   included=[refs[x['ref_id']] for x in item['all_occurrences'] if x['included_for_event_key']]
   check(bool(included),'no included occurrences')
   if included and r['qa_review']['status']=='PASS':check(positions[item['item_id']]==min((x['source_position']['global_turn_index'],x['source_position']['span_start']) for x in included),'not earliest included event')
   if r['qa_review']['status']!='PASS':check(not r['disposition']['state'].startswith('ACCEPT'),'held ordering entered accepted pool')
 if r['disposition']['state'].startswith('ACCEPT'):
  check(r['qa_review']['status']=='PASS' and r['target_realization']['target_implemented']=='YES','accepted without content and target PASS')
  check(r['fact_family']['duplicate_review']=='PASS','accepted without dedup PASS')
  check(r['disposition']['accepted_content_sha256']==r['generation_history'][-1]['candidate_content_sha256'],'accepted hash mismatch')
 if r['arm']=='A':check(r['disposition']['feedback_binding_location'] is None,'A feedback leak')
for cp in (D/'qa/independent').rglob('COMPARISON.json'):
 c=read(cp);sp=Path(c['reconstruction_seal']['path'])
 check(sha(sp)==c['reconstruction_seal']['sha256'],'QA reconstruction seal changed')
 seal=read(sp);rp=Path(seal['binding']['path'])
 check(sha(rp)==seal['binding']['sha256'],'QA independent reconstruction changed')
 for row in c['rows']:
  for key in ['private_binding','public_binding']:
   check(sha(row[key]['path'])==row[key]['sha256'],f"QA-bound {key} changed")
  if row['content_status']=='PASS':check(not row['mechanical_comparison_errors'],'comparison errors with PASS')
report={'status':'PASS' if not errors else 'FAIL','checks':checks,'records':len(paths),'source_reviews':len(list((D/'source_reviews').glob('cc-*.json'))),
 'sealed_first_batch_entries_checked':len(entries),'errors':errors,
 'scope':'Mechanical consistency and byte lineage only. Semantic decisions are signed independent full-source/public reviews; this script does not establish difficulty or feedback utility.'}
print(json.dumps(report,ensure_ascii=False,indent=2))
if '--output' in sys.argv:
 p=Path(sys.argv[sys.argv.index('--output')+1])
 with p.open('x') as fp:json.dump(report,fp,ensure_ascii=False,indent=2);fp.write('\n')
sys.exit(bool(errors))
