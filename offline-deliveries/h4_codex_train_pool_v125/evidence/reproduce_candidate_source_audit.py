import json, hashlib, unicodedata, re
from pathlib import Path
from collections import Counter, defaultdict
BASE=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries')
FILES={
'v118':BASE/'h4_codex_fullscale_ab_v118/delivery/all/SELECTED_CANDIDATES.json',
'v120':BASE/'h4_codex_scene_retry_v120/delivery/TRAINING_CANDIDATES_V1.json',
'v121':BASE/'h4_codex_all_failed_repair_v121/delivery/TRAINING_CANDIDATES_V1.json',
'v122':BASE/'h4_codex_remaining_six_repair_v122/delivery/TRAINING_CANDIDATES_V1.json'}
loaded={}; initial={}; issues=[]; checks=Counter(); explicit_bindings={}; sources={}; allqs=[]
def digest(p):
 p=Path(p)
 if not p.exists():return None
 return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 p=Path(p)
 if p not in loaded:initial[str(p)]=digest(p);loaded[p]=json.loads(p.read_text())
 return loaded[p]
def issue(kind,**kw):issues.append({'kind':kind,**kw})
def bind(p,h,location):
 p=Path(p);got=digest(p);explicit_bindings[(str(p),h)]=got
 if got!=h:issue('binding_mismatch',location=location,path=str(p),expected=h,actual=got)
 else: checks['sha_bindings']+=1
def bindings(v,loc):
 if isinstance(v,dict):
  if isinstance(v.get('path'),str) and re.fullmatch('[0-9a-f]{64}',str(v.get('sha256',''))):bind(v['path'],v['sha256'],loc)
  for k,x in v.items():bindings(x,loc+'/'+k)
 elif isinstance(v,list):
  for i,x in enumerate(v):bindings(x,loc+'/'+str(i))
def norm(s):
 s=unicodedata.normalize('NFKC',s).casefold().translate(str.maketrans({'’':"'",'“':'"','”':'"'}))
 return re.sub(r'[^\w\s]','',re.sub(r'\s+',' ',s)).strip()
def cites(v,loc=''):
 if isinstance(v,dict):
  if all(isinstance(v.get(k),str) for k in ['turn_ref','speaker','quote']):yield loc,v
  for k,x in v.items():yield from cites(x,loc+'/'+k)
 elif isinstance(v,list):
  for i,x in enumerate(v):yield from cites(x,loc+'/'+str(i))
report={'input_files':{k:str(p) for k,p in FILES.items()},'batches':{},'duplicates':{},'issues':issues}
for batch,p in FILES.items():
 d=read(p);bindings(d,str(p));qs=d.get('questions',d.get('candidates',[]));allqs.extend((batch,q) for q in qs)
 report['batches'][batch]={'candidate_count':len(qs),'root_count':len({q['root_id'] for q in qs}),'type_counts':dict(Counter(q['question_type'] for q in qs)),'gold_counts':dict(Counter(str(len(q['gold'])) for q in qs if q['question_type']=='multi_select')),'input_sha256':digest(p)}
 for q in qs:
  cid=q['candidate_id'];sp=Path(q['source_path']); bind(sp,q['source_sha256'],cid+'/source')
  if not sp.exists():continue
  src=read(sp);ts={t['turn_ref']:t for t in src['turns']};sources[str(sp)]=src
  if src.get('root_id') not in [None,q['root_id']]:issue('source_root_mismatch',candidate=cid,source_root=src.get('root_id'),root=q['root_id'])
  if q.get('role')=='control6' or (q['question_type']=='multi_select' and len(q['gold'])==6):issue('control_in_training',candidate=cid)
  for loc,c in cites(q):
   checks['citation_occurrences']+=1
   t=ts.get(c['turn_ref'])
   if t is None:issue('citation_ref_absent',candidate=cid,location=loc,citation=c)
   elif t['speaker']!=c['speaker'] or not c['quote'] or c['quote'] not in t.get('text',t.get('utterance','')):issue('citation_text_or_speaker',candidate=cid,location=loc,citation=c,source=t)
  key='options' if q['question_type']=='multi_select' else 'items';evkey='option_evidence' if key=='options' else 'item_evidence'
  if len(q[key])!=6 or set(q[key])!=set(q[evkey]):issue('label_evidence_keys',candidate=cid)
  if len(q['gold'])!=len(set(q['gold'])) or not set(q['gold'])<=set(q[key]):issue('gold_labels',candidate=cid)
  if len({norm(x) for x in q[key].values()})!=6:issue('duplicate_option_text_in_question',candidate=cid)
  if key=='options':
   for label,ev in q[evkey].items():
    if ev.get('proposition') not in [None,q[key][label]]:issue('evidence_proposition',candidate=cid,label=label)
    expected='supports' if label in q['gold'] else 'contradicts'
    if ev.get('relation')!=expected:issue('gold_relation',candidate=cid,label=label,expected=expected,actual=ev.get('relation'))
  else:
   if set(q['gold'])!=set(q[key]):issue('ordering_incomplete_gold',candidate=cid)
   for label,ev in q[evkey].items():
    if ev.get('proposition') not in [None,q[key][label]]:issue('item_proposition',candidate=cid,label=label)
    for k in ['supported','actor_ok','modality_ok','state_ok','single_event_ok']:
     if k in ev and ev[k] is not True:issue('ordering_evidence_flag',candidate=cid,label=label,flag=k)
    if 'frozen_full_turn_sha256' in ev:
     t=ts[ev['order_anchor']['turn_ref']];h=hashlib.sha256(t['text'].encode()).hexdigest()
     if h!=ev['frozen_full_turn_sha256']:issue('anchor_fullturn_sha',candidate=cid,label=label,actual=h,expected=ev['frozen_full_turn_sha256'])
   anchors=q.get('order_anchor_spans',{})
   if set(anchors)!=set(q[key]):issue('order_anchor_labels',candidate=cid)
   else:
    pos={r:i for i,r in enumerate(ts)}
    derived=sorted(q[key],key=lambda k:(pos.get(anchors[k]['turn_ref'],-1),ts.get(anchors[k]['turn_ref'],{}).get('text','').find(anchors[k]['quote'])))
    if derived!=q['gold']:issue('ordering_source_position',candidate=cid,derived=derived,gold=q['gold'])
  ex=q.get('evidence_export',{}).get('input_sha256',{})
  for filename,h in ex.items():bind(sp.parent/filename,h,cid+'/evidence_export/'+filename)
  pgpath=sp.parent/'PRIVATE_GOLD.json';pubpath=sp.parent/'PUBLIC_QUESTIONS.json'
  if pgpath.exists() and pubpath.exists():
   pg=read(pgpath);pub=read(pubpath);gp={x['candidate_id']:x for x in pg['questions']};pq={x['candidate_id']:x for x in pub['questions']}
   if cid in gp and gp[cid]['gold']!=q['gold']:issue('private_gold_changed',candidate=cid)
   if cid in pq:
    for f in ['root_id','question_type','question',key,'source_path','source_sha256']:
     if pq[cid].get(f)!=q.get(f):
      oldpath=pq[cid].get(f)
      basedir=next((x for x in sp.parents if (x/'MASTER_FREEZE.json').exists()),sp.parent)
      if f=='source_path' and oldpath and Path(oldpath).resolve()==Path(q[f]).resolve():checks['equivalent_repo_relative_to_absolute_source_paths']+=1
      elif f=='source_path' and oldpath and (basedir/oldpath).resolve()==Path(q[f]).resolve():checks['equivalent_batch_relative_to_absolute_source_paths']+=1
      else:issue('public_payload_changed',candidate=cid,field=f)
   elif 'v114-' not in cid:issue('candidate_missing_sibling_public',candidate=cid)
  elif 'v114-' not in cid:issue('missing_sibling_gold_or_public',candidate=cid,source=str(sp))
# Canonical public content ignores arbitrary label names and display order.
ids=defaultdict(list);publicsig=defaultdict(list);answeredsig=defaultdict(list);families=defaultdict(list);roots=defaultdict(set)
for b,q in allqs:
 cid=q['candidate_id'];ids[cid].append(b);roots[q['root_id']].add(b)
 key='options' if q['question_type']=='multi_select' else 'items';texts=q[key]
 sig=(q['root_id'],q['question_type'],tuple(sorted(norm(t) for t in texts.values())))
 publicsig[sig].append(cid)
 ans=tuple(sorted(norm(texts[k]) for k in q['gold'])) if key=='options' else tuple(norm(texts[k]) for k in q['gold'])
 answeredsig[(sig,ans)].append(cid)
 if key=='options':
  # These root-level variants intentionally share one fact family, not independent examples.
  scenes=tuple(sorted({c['turn_ref'].rsplit(':T',1)[0] for _,c in cites(q['option_evidence'])}))
  families[(q['root_id'],scenes)].append(cid)
report['duplicates']={'candidate_ids':{k:v for k,v in ids.items() if len(v)>1},'canonical_public_content_groups':[v for v in publicsig.values() if len(v)>1],'canonical_content_and_answer_groups':[v for v in answeredsig.values() if len(v)>1],'roots_across_batches':{k:sorted(v) for k,v in roots.items() if len(v)>1},'multi_select_family_count_by_root':len({k[0] for k in families}),'multi_select_family_groups':[{'root':k[0],'citation_scenes':k[1],'candidate_ids':v} for k,v in families.items()]}
hold=read(BASE/'h4_codex_dev_test_ab_v124/MANIFEST.json');holdroots={r['root_id'] for r in hold['roots']}
report['v124_disjoint']={'denominator':len(holdroots),'training_roots':len(roots),'overlapping_roots':sorted(holdroots&set(roots))}
overlay=read(BASE/'h4_codex_wording_minfix_v114/RECOMMENDED_CANDIDATES.json');ov={q['candidate_id']:q for q in overlay['questions']};selected={q['candidate_id']:q for b,q in allqs if b=='v118'}
report['v114_overlay']={'quarantined_ids':overlay['quarantined_original_ids'],'quarantine_present':sorted(set(overlay['quarantined_original_ids'])&set(ids)),'expected_overlay_count':len(ov),'present_overlay_count':len(set(ov)&set(selected)),'not_exactly_equal':[cid for cid,q in ov.items() if selected.get(cid)!=q]}
# Check that every source-review citation survives export. A longer exported source
# quote may contain a reviewed shorter span; this is preservation, not a new quote.
review_preservation=[]
for batch,q in allqs:
 cid=q['candidate_id'];sp=Path(q['source_path']);jp=sp.parent/'PUBLIC_SOURCE_JUDGMENTS.json'
 if not jp.exists():
  if cid in ov:checks['overlay_instead_of_sibling_source_review']+=1
  else:issue('missing_source_review',candidate=cid)
  continue
 js=read(jp)['questions'];js=js if isinstance(js,dict) else {x['candidate_id']:x for x in js}
 if cid not in js:issue('candidate_missing_source_review',candidate=cid);continue
 j=js[cid];key='options' if q['question_type']=='multi_select' else 'items';evkey='option_evidence' if key=='options' else 'item_evidence'
 for label,ev in q[evkey].items():
  orig=j[key][label];oldc=list(cites(orig));exportc=list(cites(ev))
  for loc,c in oldc:
   if not any(c['turn_ref']==x['turn_ref'] and c['speaker']==x['speaker'] and c['quote'] in x['quote'] for _,x in exportc):
    review_preservation.append({'candidate':cid,'label':label,'omitted_reviewed_citation':c})
  if key=='options' and orig['relation']!=ev['relation']:issue('source_review_relation_changed',candidate=cid,label=label)
report['source_review_citation_omissions']=review_preservation
selection_checks={}
from fractions import Fraction
for batch,fp in FILES.items():
 d=read(fp);qs=d['questions'];errors=[]
 if batch!='v118':
  pool=read(d['source_pool_binding']['path'])['questions'];pool={q['candidate_id']:q for q in pool}
  results=read(d['selection_results_binding']['path']);res={q['candidate_id']:q for q in results['questions']}
  for q in qs:
   cid=q['candidate_id']
   if q!=pool.get(cid):errors.append({'candidate':cid,'problem':'not_exact_pool_question'})
   r=res.get(cid,{})
   if not r.get('selected_by_policy') or not 0<Fraction(r.get('full_partial_mean','0'))<1 or r.get('actual_full_calls')!=r.get('expected_full_calls') or r.get('data_errors'):errors.append({'candidate':cid,'problem':'selection_policy_evidence'})
  selection_checks[batch]={'questions_checked':len(qs),'selected_pool_exact_and_policy_errors':errors,'global_call_set_errors':results.get('call_set_errors')}
 else:
  final=read(BASE/'h4_codex_fullscale_ab_v118/delivery/all/FINAL_STATUS.json');prov=final['reused_provenance'];group=defaultdict(list)
  for q in qs:
   if q['root_id'] in prov:
    old=read(prov[q['root_id']]['selection_artifact']['path']);old={x['candidate_id']:x for x in old['questions']}
    if q!=old.get(q['candidate_id']):errors.append({'candidate':q['candidate_id'],'problem':'historical_selection_changed'})
   else:group['new'].append(q)
  res={}
  for w in ['validation','production']:
   rp=BASE/'h4_codex_fullscale_ab_v118/evaluations'/w/'run/trained/RESULTS.json'
   res.update({x['candidate_id']:x for x in read(rp)['questions']})
  for q in group['new']:
   cid=q['candidate_id'];r=res.get(cid,{})
   if not r.get('selected_by_policy') or not 0<Fraction(r.get('full_partial_mean','0'))<1 or r.get('actual_full_calls')!=r.get('expected_full_calls') or r.get('data_errors'):errors.append({'candidate':cid,'problem':'selection_policy_evidence'})
  selection_checks[batch]={'questions_checked':len(qs),'new_policy_questions':len(group['new']),'historical_preserved_questions':len(qs)-len(group['new']),'errors':errors}
report['selection_checks']=selection_checks
# Exact ordering anchor-set duplication catches alternate public wording on the
# very same frozen source acts; it does not claim arbitrary paraphrase detection.
osig=defaultdict(list)
for batch,q in allqs:
 if q['question_type']=='ordering':
  sig=(q['root_id'],tuple(sorted((x['turn_ref'],norm(x['quote'])) for x in q['order_anchor_spans'].values())))
  osig[sig].append(q['candidate_id'])
report['duplicates']['same_ordering_anchor_set_groups']=[v for v in osig.values() if len(v)>1]
# Confirm each candidate/source/review artifact read directly from a sealed batch
# matches its COMPLETE_MANIFEST entry, without invoking any batch tool.
manifest_coverage={}
for batch,path in FILES.items():
 bd=next(p for p in path.parents if (p/'COMPLETE_MANIFEST.json').exists())
 mp=bd/'COMPLETE_MANIFEST.json';md=read(mp);entries={x['path']:x['sha256'] for x in md['files']}
 hits=[p for p in initial if p in entries]
 for p in hits:bind(p,entries[p],str(mp))
 manifest_coverage[batch]={'candidate_file_in_manifest':str(path) in entries,'accessed_files_verified':len(hits),'manifest_sha256':digest(mp)}
report['sealed_manifest_coverage']=manifest_coverage
historical_coverage={}
for oldbatch in ['h4_codex_serialized_ab_validation_v113','h4_codex_wording_minfix_v114','h4_codex_newroots_ab_validation_v115','h4_codex_screened_ab_validation_v117']:
 bd=BASE/oldbatch;mp=bd/'COMPLETE_MANIFEST.json';md=read(mp)
 entries={str((Path(x['path']) if Path(x['path']).is_absolute() else bd/x['path']).resolve()):x['sha256'] for x in md['files']}
 hits=[p for p in initial if p in entries]
 for p in hits:bind(p,entries[p],str(mp))
 historical_coverage[oldbatch]={'accessed_files_verified':len(hits),'manifest_sha256':digest(mp),'source_review_files_verified':len([p for p in hits if p.endswith('PUBLIC_SOURCE_JUDGMENTS.json')])}
report['historical_manifest_coverage']=historical_coverage
# Independently apply the explicit metadata overlay to a deep copy, then audit
# effective records. The original selected files remain the immutable input.
import copy
overlaypath=Path('/tmp/v125_evidence_overlay.json');effective_errors=[];effectiveqs=[(b,copy.deepcopy(q)) for b,q in allqs]
if overlaypath.exists():
 eo=read(overlaypath);bindings(eo,str(overlaypath));eqs={q['candidate_id']:q for b,q in effectiveqs};origqs={q['candidate_id']:q for b,q in allqs}
 for cor in eo['corrections']:
  cid=cor['candidate_id'];label=cor['label'];q=eqs[cid]
  if q['option_evidence'][label]!=cor['original_evidence']:effective_errors.append({'candidate':cid,'label':label,'problem':'overlay_preimage_mismatch'})
  if eo['original_candidate_records'][cid]['record']!=origqs[cid]:effective_errors.append({'candidate':cid,'problem':'full_original_record_mismatch'})
  q['option_evidence'][label]=copy.deepcopy(cor['effective_evidence'])
 for batch,q in effectiveqs:
  cid=q['candidate_id'];orig=origqs[cid];evkey='option_evidence' if q['question_type']=='multi_select' else 'item_evidence'
  if {k:v for k,v in q.items() if k!=evkey}!={k:v for k,v in orig.items() if k!=evkey}:effective_errors.append({'candidate':cid,'problem':'non_evidence_field_changed'})
  sp=Path(q['source_path']);ts={t['turn_ref']:t for t in read(sp)['turns']}
  for loc,c in cites(q[evkey]):
   t=ts.get(c['turn_ref'])
   if t is None or c['speaker']!=t['speaker'] or c['quote'] not in t['text']:effective_errors.append({'candidate':cid,'problem':'effective_citation_mismatch','location':loc})
  jp=sp.parent/'PUBLIC_SOURCE_JUDGMENTS.json'
  if jp.exists():
   jj=read(jp)['questions'];jj=jj if isinstance(jj,dict) else {x['candidate_id']:x for x in jj};key='options' if q['question_type']=='multi_select' else 'items'
   for label,ev in q[evkey].items():
    origreview=jj[cid][key][label];exportc=list(cites(ev))
    for loc,c in cites(origreview):
     if not any(c['turn_ref']==x['turn_ref'] and c['speaker']==x['speaker'] and c['quote'] in x['quote'] for _,x in exportc):effective_errors.append({'candidate':cid,'label':label,'problem':'effective_review_citation_missing','citation':c})
  if evkey=='option_evidence':
   for label,ev in q[evkey].items():
    if ev.get('relation')!=('supports' if label in q['gold'] else 'contradicts'):effective_errors.append({'candidate':cid,'label':label,'problem':'effective_gold_relation'})
 report['effective_evidence_overlay']={'path':str(overlaypath),'sha256':digest(overlaypath),'correction_count':len(eo['corrections']),'full_original_candidate_records_retained':len(eo['original_candidate_records']),'errors':effective_errors,'status':'PASS' if not effective_errors else 'FAIL','all_363_public_text_gold_membership_sources_unchanged':not any(x.get('problem')=='non_evidence_field_changed' for x in effective_errors),'all_reviewed_citations_preserved_after_effective_overlay':not any(x.get('problem')=='effective_review_citation_missing' for x in effective_errors)}
report['scope_limitations']=['Mechanical bindings, exact citation matching, source-review preservation, ID/content canonicalization and frozen selection provenance checked; no new Solver or training run.','No claim of exhaustive semantic equivalence detection for arbitrary paraphrases, source factuality, external model certification or held-out quality; related Gold-family variants are counted as correlated examples.','Raw legacy evidence has four citation omissions on two candidates; effective metadata overlay is required for lossless evidence preservation. Mandatory v114 overlay must remain applied.']
# v118 content set carries the explicit mandatory overlay SHA, plus historical per-root selection bindings.
for path in [BASE/'h4_codex_fullscale_ab_v118/delivery/all/CONTENT_QUESTIONS.json',BASE/'h4_codex_fullscale_ab_v118/delivery/all/FINAL_STATUS.json']:
 bindings(read(path),str(path))
report['checks']=dict(checks);report['explicit_distinct_bindings']=len(explicit_bindings)
report['unique_sources']=len(sources);report['all_root_ids']=sorted(roots)
report['total_candidates']=len(allqs);report['type_counts']=dict(Counter(q['question_type'] for b,q in allqs));report['gold_counts']=dict(Counter(str(len(q['gold'])) for b,q in allqs if q['question_type']=='multi_select'))
report['inputs_changed_during_audit']=[p for p,h in initial.items() if digest(p)!=h]
Path('/tmp/v125_independent_candidates_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['duplicates','all_root_ids','issues']},indent=2))
print('DUPLICATE_SUMMARY',json.dumps({k:v for k,v in report['duplicates'].items() if k!='multi_select_family_groups'},indent=2))
print('ISSUES',json.dumps(issues,indent=2))
