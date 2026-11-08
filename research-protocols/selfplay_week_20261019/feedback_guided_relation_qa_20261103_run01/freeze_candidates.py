#!/usr/bin/env python3
"""Freeze Codex-authored drafts and make answer-free, arm-blind review packets."""
import hashlib
import json
from pathlib import Path
import random
from datetime import datetime, timezone
HERE = Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=HERE/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    sources={s['root_id']:s for s in read(HERE/'SOURCE_FREEZE.json')['sources']}
    collected=[];mechanical=[];bindings=[]
    for arm in ['F','G']:
        base=HERE/('author_'+arm)
        drafts=read(base/'CANDIDATES_FIRST_DRAFT.json')
        targets_hash=sha(base/'TARGETS_FROZEN.json')
        assert len(drafts['candidates'])<=4
        assert len({c['root_id'] for c in drafts['candidates']})==len(drafts['candidates'])
        for name in ['CANDIDATES_FIRST_DRAFT.json','TARGETS_FROZEN.json','TARGETS_FREEZE.json','EXPLORATION.json','AUTHOR_INPUT.json']:
            bindings.append({'path':str(base/name),'sha256':sha(base/name)})
        for candidate in drafts['candidates']:
            c=dict(candidate);c['arm']=arm
            s=sources[c['root_id']]
            assert c['source_path']==s['path'] and c['source_sha256']==s['sha256']==sha(s['path'])
            assert c['targets_sha256']==targets_hash
            source=read(s['path']);turns={t['turn_ref']:t for t in source['turns']}
            errors=[]
            ids=[o['option_id'] for o in c['options']]
            if len(set(ids))!=len(ids):errors.append('duplicate_option_id')
            if len({o['text'] for o in c['options']})!=len(ids):errors.append('duplicate_text')
            gold=c['gold_option_ids']
            if not gold or len(set(gold))!=len(gold) or not set(gold)<=set(ids):errors.append('invalid_gold')
            if c['question_type']=='multi_select':
                if not 2<=len(gold)<=6 or not len(gold)<len(ids):errors.append('cardinality_outside_frozen_contract')
            elif c['question_type']=='ordering':
                if set(gold)!=set(ids):errors.append('incomplete_gold_order')
            else:errors.append('unsupported_type')
            audits={r['option_id']:r for r in c['option_audit']}
            if set(audits)!=set(ids):errors.append('incomplete_option_audit')
            for oid,row in audits.items():
                if c['question_type']=='multi_select' and row.get('correct')!=(oid in gold):errors.append('truth_gold_mismatch:'+oid)
                if c['question_type']=='ordering' and row.get('gold_position')!=gold.index(oid)+1:errors.append('order_gold_mismatch:'+oid)
            all_evidence=list(c['evidence'])
            for row in c['option_audit']:all_evidence.extend(row['evidence'])
            mapping=[]
            for ev in all_evidence:
                ref=ev['turn_ref'];t=turns.get(ref)
                if t is None:errors.append('unknown_turn:'+ref);continue
                if not ev.get('quote') or ev['quote'] not in t['text']:errors.append('quote_not_exact:'+ref)
                if 'origin' in ev and ev['origin']!=t.get('origin'):errors.append('origin_mismatch:'+ref)
                mapping.append({'root_id':c['root_id'],'turn_ref':ref,'scene_id':t.get('scene_ref') or ':'.join(ref.split(':')[:-1]),'origin':t.get('origin','UNKNOWN'),'source_path':s['path'],'source_sha256':s['sha256'],'quote':ev['quote']})
            c.update(author_candidate_path=str(base/'CANDIDATES_FIRST_DRAFT.json'),author_candidate_sha256=sha(base/'CANDIDATES_FIRST_DRAFT.json'),target_path=str(base/'TARGETS_FROZEN.json'),target_sha256=targets_hash,source_turn_count=len(turns))
            collected.append(c);mechanical.append({'candidate_id':c['candidate_id'],'mechanical_pass':not errors,'errors':errors,'evidence_turn_mapping':mapping})
    assert len(collected)<=8 and len({c['candidate_id'] for c in collected})==len(collected)
    # Gold-blind packet order and item permutations.
    rng=random.Random(2026110371);rng.shuffle(collected)
    items=[]
    for index,c in enumerate(collected,1):
        qid=f'B{index:02d}';options=list(c['options'])
        local=random.Random(int(hashlib.sha256(('review-display-v1:'+c['candidate_id']).encode()).hexdigest(),16));local.shuffle(options)
        labels=[chr(65+i) for i in range(len(options))] if c['question_type']=='multi_select' else [f'I{i+1}' for i in range(len(options))]
        mapping={l:o['option_id'] for l,o in zip(labels,options)};reverse={v:k for k,v in mapping.items()}
        display=[{'id':l,'text':o['text']} for l,o in zip(labels,options)]
        gold=[reverse[g] for g in c['gold_option_ids']]
        if c['question_type']=='multi_select':gold.sort()
        source=read(c['source_path'])
        packet={'review_id':qid,'reviewer_mode':'same_model_blind','root_id':c['root_id'],'question_type':c['question_type'],'question':c['question'],'options':display,'source':{'path':c['source_path'],'sha256':c['source_sha256'],'turn_count':c['source_turn_count'],'scope':'complete frozen dialogue; origins and warnings retained; user-authorized research; per-question cited-turn review','warnings':source.get('warnings',[])}}
        write('blind_packets/'+qid+'.json',packet)
        item={**c,'number':index,'review_id':qid,'options':display,'label_to_option_id':mapping,'gold_labels':gold,'source_turn_count':c['source_turn_count']}
        items.append(item)
    write('ALL_CANDIDATES_PRIVATE.json',collected)
    write('ITEMS_PRIVATE.json',items)
    write('MECHANICAL_REVIEW.json',{'items':mechanical})
    write('AUTHOR_BINDINGS.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'files':bindings})
    print(json.dumps({'candidates':len(items),'mechanical_pass':sum(x['mechanical_pass'] for x in mechanical),'review_packets':[x['review_id'] for x in items]}))
if __name__=='__main__':main()
