"""Deterministic public projection of immutable Codex authored facts/items."""
import json
import hashlib
import random
import sys
from datetime import datetime,timezone
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def stamp():return datetime.now(timezone.utc).isoformat()
def build(short):
    folder=BASE/'roots'/f'root{short}'
    output=folder/'PUBLIC_QUESTIONS.json'
    if output.exists() or not (folder/'QA_DRAFT.json').exists():return
    review=read(folder/'SOURCE_REVIEW.json')
    assert review['decision']=='READY'
    path=folder/'CUMULATIVE_SOURCE.json';source=read(path);source_hash=sha(path)
    draft=read(folder/'QA_DRAFT.json');facts=draft['facts']
    assert len(facts)==6
    assert len({f['fact_id'] for f in facts})==6
    turns={t['turn_ref']:t for t in source['turns']}
    errors=[]
    for f in facts:
        assert f['true_text'].strip() and f['false_text'].strip()
        for c in f['evidence']:
            if c['turn_ref'] not in turns or not c['quote'] or c['quote'] not in turns[c['turn_ref']]['text']:
                errors.append({'fact_id':f['fact_id'],'error':'evidence_not_exact','citation':c})
    questions=[];design=[]
    for cardinality in (6,2,3,4,5):
        rng=random.Random(f'v104-fixed-{short}-{cardinality}')
        truth=set(rng.sample(range(6),cardinality))
        order=list(range(6));rng.shuffle(order)
        options={label:(facts[i]['true_text'] if i in truth else facts[i]['false_text']) for label,i in zip('ABCDEF',order)}
        opaque=hashlib.sha256(f'v104:{short}:{cardinality}'.encode()).hexdigest()[:12]
        cid=f'v104-root{short}-ms-{opaque}'
        questions.append({'candidate_id':cid,'root_id':source['root_id'],'question_type':'multi_select','question':draft['stem'],'options':options,'source_path':str(path),'source_sha256':source_hash})
        design.append({'candidate_id':cid,'labels_to_fact_ids':{k:facts[i]['fact_id'] for k,i in zip('ABCDEF',order)},'chosen_true_fact_ids':[facts[i]['fact_id'] for i in sorted(truth)],'intended_cardinality':cardinality,'role':'control6' if cardinality==6 else 'new_child'})
    plan=read(folder/'ORDERING_ANCHOR_PLAN.json')
    slots=plan['slots'];ordering=draft['ordering']
    for slot in slots:
        anchors=slot['anchors'];assert len(anchors)==6
        assert len({a['turn_ref'] for a in anchors})==6
        for a in anchors:
            assert turns[a['turn_ref']]['speaker']==a['speaker']
            assert turns[a['turn_ref']]['text']==a['full_turn_quote']
        item=next((x for x in ordering if str(x['slot_id'])==str(slot['slot_id'])),None)
        if not item or item.get('failure'):
            errors.append({'slot_id':slot['slot_id'],'error':'ordering_slot_uncreated','reason':None if not item else item['failure']});continue
        its=item['items'];assert len(its)==6
        assert {i['item_id'] for i in its}=={f'I{n}' for n in range(1,7)}
        assert {i['anchor_id'] for i in its}=={a['anchor_id'] for a in anchors}
        by_id={a['anchor_id']:a for a in anchors}
        for i in its:
            if not i['support_quote'] or i['support_quote'] not in by_id[i['anchor_id']]['full_turn_quote']:
                errors.append({'slot_id':slot['slot_id'],'item_id':i['item_id'],'error':'ordering_support_not_exact'})
        cid=f"v104-root{short}-ordering{slot['slot_id']}"
        questions.append({'candidate_id':cid,'root_id':source['root_id'],'question_type':'ordering','chronology_kind':'conversation_order','question':'Order the statements, using left-to-right clause order within a turn, as they APPEAR IN THE SUPPLIED DIALOGUE within the R02:B:S01 scene.','items':{i['item_id']:i['text'] for i in its},'source_path':str(path),'source_sha256':source_hash})
        design.append({'candidate_id':cid,'role':'ordering','slot_id':slot['slot_id'],'item_anchor_ids':{i['item_id']:i['anchor_id'] for i in its}})
    save(output,{'questions':questions})
    save(folder/'PRIVATE_DESIGN.json',{'source_sha256':source_hash,'qa_draft_sha256':sha(folder/'QA_DRAFT.json'),'anchor_plan_sha256':sha(folder/'ORDERING_ANCHOR_PLAN.json'),'design':design,'mechanical_errors':errors,'private_gold_not_constructed_yet':True})
    save(folder/'PUBLIC_GENERATION_RECEIPT.json',{'created_at_utc':stamp(),'question_count':len(questions),'public_sha256':sha(output),'mechanical_errors':errors,'no_writer_repairs':True,'assembly_seed_policy':'v104-fixed-<root>-<cardinality>; immutable script chooses subsets and label shuffles without Solver','gold_construction':'deferred until independent source judgments saved'})
    print(short,len(questions),'questions',len(errors),'errors')

def private(short):
    folder=BASE/'roots'/f'root{short}'
    output=folder/'PRIVATE_GOLD.json'
    if output.exists():return
    judgments=folder/'QUESTION_SOURCE_JUDGMENTS.json'
    assert judgments.exists(),'source_first_review_required'
    design=read(folder/'PRIVATE_DESIGN.json');draft=read(folder/'QA_DRAFT.json')
    source=read(folder/'CUMULATIVE_SOURCE.json');positions={t['turn_ref']:n for n,t in enumerate(source['turns'])}
    questions=read(folder/'PUBLIC_QUESTIONS.json')['questions']
    answers=[]
    for d in design['design']:
        if d['role'] in ('control6','new_child'):
            gold=[k for k,v in d['labels_to_fact_ids'].items() if v in d['chosen_true_fact_ids']]
            evidence={k:next(f for f in draft['facts'] if f['fact_id']==v)['evidence'] for k,v in d['labels_to_fact_ids'].items()}
        else:
            slot=next(s for s in read(folder/'ORDERING_ANCHOR_PLAN.json')['slots'] if str(s['slot_id'])==str(d['slot_id']))
            anchors={a['anchor_id']:a for a in slot['anchors']}
            gold=sorted(d['item_anchor_ids'],key=lambda k:positions[anchors[d['item_anchor_ids'][k]]['turn_ref']])
            evidence={k:{'turn_ref':anchors[v]['turn_ref'],'quote':anchors[v]['full_turn_quote']} for k,v in d['item_anchor_ids'].items()}
        answers.append({'candidate_id':d['candidate_id'],'role':d['role'],'gold':gold,'evidence':evidence})
    save(output,{'created_at_utc':stamp(),'source_first_judgments_sha256':sha(judgments),'public_sha256':sha(folder/'PUBLIC_QUESTIONS.json'),'answers':answers,'derivation':'Frozen original design and source positions only; independent judgments do not alter intended answers.'})
    print(short,'private Gold constructed after source review')
if __name__=='__main__':
    fn=private if '--private' in sys.argv else build
    for short in ('14304','20432','24719','38601'):
        if not '--private' in sys.argv or (BASE/'roots'/f'root{short}'/'QUESTION_SOURCE_JUDGMENTS.json').exists():fn(short)
