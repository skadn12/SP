#!/usr/bin/env python3
"""Experiment-only freeze of direct Codex first drafts; does not change QA algorithms."""
import argparse,hashlib,json,random
from pathlib import Path
from datetime import datetime,timezone
E=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
p=argparse.ArgumentParser();p.add_argument('--author',required=True);p.add_argument('--batch',required=True);p.add_argument('--arm',default='C');a=p.parse_args()
b=Path(a.batch);author=Path(a.author);b.mkdir(parents=True,exist_ok=True)
sources={s['root_id']:s for s in read(E/'SOURCE_FREEZE.json')['sources']}
drafts=read(author/'CANDIDATES_FIRST_DRAFT.json');candidates=drafts['candidates'];assert len(candidates)<=5
target_file=author/'TARGETS_FROZEN.json';target_doc=read(target_file);target_file_hash=sha(target_file)
target_hash_definition=target_doc.get('target_hash_definition')
if target_hash_definition:
 expected_definition='SHA-256 of the targets array encoded as UTF-8 JSON with sorted keys, compact separators, and ensure_ascii=false.'
 if target_hash_definition!=expected_definition:raise ValueError('unsupported_target_hash_definition')
 canonical=json.dumps(target_doc['targets'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
 computed_target_hash=hashlib.sha256(canonical).hexdigest();target_hash_mode='canonical_targets_array'
 declared_target_hash=target_doc.get('targets_sha256')
 if declared_target_hash!=computed_target_hash:raise ValueError('target_document_hash_mismatch')
 target_hash=computed_target_hash
else:
 target_hash=target_file_hash;computed_target_hash=None;declared_target_hash=None;target_hash_mode='target_file_bytes'
bindings=[{'path':str(f),'sha256':sha(f)} for f in sorted(author.iterdir()) if f.is_file()]
items=[];mechanical=[];collected=[]
for number,c in enumerate(candidates,1):
 c=dict(c);c['arm']=a.arm;s=sources[c['root_id']];source=read(s['path']);turns={t['turn_ref']:t for t in source['turns']};err=[]
 if c['source_sha256']!=s['sha256'] or sha(s['path'])!=s['sha256'] or c['source_path']!=s['path']:err.append('source_binding')
 if drafts.get('targets_sha256')!=target_hash or c.get('targets_sha256',target_hash)!=target_hash:err.append('target_hash_mismatch')
 ids=[o['option_id'] for o in c['options']];gold=c['gold_option_ids'];typ=c['question_type']
 if len(set(ids))!=len(ids):err.append('duplicate_option_id')
 if len({o['text'] for o in c['options']})!=len(ids):err.append('duplicate_option_text')
 if typ=='multi_select':
  if not 2<=len(gold)<=6 or not len(gold)<len(ids) or not set(gold)<=set(ids) or len(set(gold))!=len(gold):err.append('invalid_multi_gold')
 elif typ=='ordering':
  if len(gold)!=len(ids) or set(gold)!=set(ids):err.append('invalid_order_gold')
 elif typ=='partial_order':
  pairs=c.get('gold_pairs',[])
  if not pairs or any(len(x)!=2 or x[0]==x[1] or not set(x)<=set(ids) for x in pairs) or len({tuple(x) for x in pairs})!=len(pairs):err.append('invalid_partial_gold')
  if not c.get('query_scope') or c.get('answer_semantics')!='all_and_only_entailed_pairs_in_query_scope':err.append('undefined_partial_scope')
 else:err.append('unsupported_type')
 audit={o['option_id']:o for o in c['option_audit']}
 if set(audit)!=set(ids):err.append('incomplete_option_audit')
 for oid,row in audit.items():
  if typ=='multi_select' and row.get('correct')!=(oid in gold):err.append('truth_gold_mismatch:'+oid)
  if typ=='ordering' and row.get('gold_position')!=gold.index(oid)+1:err.append('position_gold_mismatch:'+oid)
 evidence=list(c['evidence'])
 for row in c['option_audit']:evidence.extend(row.get('evidence',[]))
 for row in c.get('pair_audit',[]):evidence.extend(row.get('evidence',[]))
 for ev in evidence:
  t=turns.get(ev['turn_ref'])
  if t is None:err.append('unknown_turn:'+ev['turn_ref']);continue
  if not ev.get('quote') or ev['quote'] not in t['text']:err.append('quote_not_exact:'+ev['turn_ref'])
  if 'origin'in ev and ev['origin']!=t.get('origin'):err.append('origin_mismatch:'+ev['turn_ref'])
 rid=f'Q{number:02d}';opts=list(c['options']);rng=random.Random(int(hashlib.sha256(('capacity-review:'+c['candidate_id']).encode()).hexdigest(),16));rng.shuffle(opts)
 labels=[chr(65+i) for i in range(len(opts))] if typ=='multi_select' else [f'I{i+1}' for i in range(len(opts))]
 mapping={k:o['option_id'] for k,o in zip(labels,opts)};rev={v:k for k,v in mapping.items()};display=[{'id':k,'text':o['text']} for k,o in zip(labels,opts)]
 if typ=='partial_order':stable_gold=c['gold_pairs'];gl=[[rev[u],rev[v]] for u,v in stable_gold]
 else:
  stable_gold=gold;gl=[rev[x] for x in gold]
  if typ=='multi_select':gl.sort()
 packet={'review_id':rid,'reviewer_mode':'same_model_blind','root_id':c['root_id'],'question_type':typ,'question':c['question'],'options':display,'source':{'path':s['path'],'sha256':s['sha256'],'turn_count':s['turn_count'],'scope':'complete frozen source, user-authorized research; origins/warnings retained; cited-turn quality review','warnings':source.get('warnings',[])}}
 if typ=='partial_order':
  packet.update(query_scope=c['query_scope'],query_pairs=[[rev[u],rev[v]] for u,v in c.get('query_pairs',[])],answer_semantics=c['answer_semantics'])
 write(b/'blind_packets'/f'{rid}.json',packet)
 item={**c,'number':number,'review_id':rid,'options':display,'label_to_option_id':mapping,'gold_labels':gl,'gold_option_ids':stable_gold,'source_turn_count':s['turn_count'],'author_path':str(author/'CANDIDATES_FIRST_DRAFT.json'),'author_sha256':sha(author/'CANDIDATES_FIRST_DRAFT.json')}
 items.append(item);collected.append(c);mechanical.append({'candidate_id':c['candidate_id'],'mechanical_pass':not err,'errors':err})
assert len({i['candidate_id'] for i in items})==len(items)
write(b/'SOURCE_FREEZE.json',{'sources':[sources[r] for r in sorted({i['root_id'] for i in items})]})
write(b/'ITEMS_PRIVATE.json',items);write(b/'ALL_CANDIDATES_PRIVATE.json',collected);write(b/'MECHANICAL_REVIEW.json',{'items':mechanical});write(b/'AUTHOR_BINDINGS.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'files':bindings,'target_hash_binding':{'mode':target_hash_mode,'author_declared_sha256':drafts.get('targets_sha256'),'computed_targets_sha256':computed_target_hash,'target_document_declared_sha256':declared_target_hash,'target_file_sha256':target_file_hash,'valid':drafts.get('targets_sha256')==target_hash}})
print(json.dumps({'batch':str(b),'candidates':len(items),'mechanical_PASS':sum(i['mechanical_pass'] for i in mechanical),'errors':[i for i in mechanical if i['errors']]}))
