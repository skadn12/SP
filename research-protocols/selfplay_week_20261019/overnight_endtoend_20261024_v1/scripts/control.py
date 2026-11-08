#!/usr/bin/env python3
"""OET1 append/review control; does not call a model or author any dialogue."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

BASE=Path(__file__).resolve().parents[1]
SCENES=[f'OET1:{r}:S{i:02}' for r in 'AB' for i in range(1,4)]
CHECKS=['time','location','actor_identity','plan_vs_completion','cross_scene_continuity','semantic_repetition','naturalness','event_fidelity']
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def bind(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def verify(b):
    assert bind(b['path'])==b,('BINDING_CHANGED',b['path'])
    return read(b['path'])
def write(p,d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def unit(root,arm):
    assert arm in ['GENERIC','TARGETED']
    assert root in read(BASE/'PLAN.json')['new_train_roots']
    return BASE/'sources'/root/arm
def commit(root,arm,scene):
    u=unit(root,arm); packet=read(u/'SOURCE_AUTHOR_INPUT.json'); i=SCENES.index(scene)
    prev=u/SCENES[i-1].replace(':','_')/'CUMULATIVE_SOURCE.json' if i else Path(packet['source_binding']['path'])
    if i:
        receipt=read(prev.parent/'ACCEPTED.json')
        assert receipt['status']=='READY' and receipt['source']==bind(prev)
        assert receipt['review']==bind(prev.parent/'REVIEW.json')
    else: verify(packet['source_binding'])
    source=read(prev); sdir=u/scene.replace(':','_'); draft=read(sdir/'DRAFT.json')
    assert draft['root_id']==root and draft['scene_ref']==scene
    assert draft['input_source_binding']==bind(prev) and draft['author_input_binding']==bind(u/'SOURCE_AUTHOR_INPUT.json')
    if i>=3: assert draft['native_event_binding']==packet['native_events'][i-3]['native_event_binding']
    turns=draft['turns']; assert len(turns)==14
    for n,t in enumerate(turns,1):
        assert t['turn_ref']==f'{scene}:T{n:03}' and t['scene_ref']==scene
        assert t['speaker'] in packet['speakers'] and t['text'].strip()
    assert not set(t['turn_ref'] for t in source['turns'])&set(t['turn_ref'] for t in turns)
    full={**source,'schema_version':'oet1_cumulative_source_v1','scene_ref':scene,'condition':arm,
        'input_source_binding':bind(prev),'draft_binding':bind(sdir/'DRAFT.json'),
        'turns':source['turns']+turns,'new_turn_count':14,'oet1_added_turn_count':14*(i+1)}
    cp=sdir/'CUMULATIVE_SOURCE.json'
    if cp.exists(): assert read(cp)==full
    else: write(cp,full)
    return {'source':bind(cp),'draft':bind(sdir/'DRAFT.json'),'needs_independent_review':True}
def accept(root,arm,scene):
    u=unit(root,arm); s=u/scene.replace(':','_'); d=read(s/'DRAFT.json'); source=read(s/'CUMULATIVE_SOURCE.json'); r=read(s/'REVIEW.json')
    assert r['source_binding']==bind(s/'CUMULATIVE_SOURCE.json') and r['draft_binding']==bind(s/'DRAFT.json')
    assert r['reviewer_id']!=d['author_id'] and r['independent_of_source_author'] is True
    assert r['full_source_read'] is True and r['turn_refs_read']==[t['turn_ref'] for t in source['turns']]
    assert set(CHECKS)<=set(r['checks'])
    assert all(r['checks'][k]['status'] in ['PASS','WARNING','PASS_WITH_WARNING'] and r['checks'][k]['reason'].strip() for k in CHECKS)
    assert r['decision']=='READY' and not any(x.get('severity')=='BLOCKING' for x in r.get('findings',[]))
    receipt={'status':'READY','created_at_utc':now(),'review':bind(s/'REVIEW.json'),'source':bind(s/'CUMULATIVE_SOURCE.json')}
    if (s/'ACCEPTED.json').exists():
        old=read(s/'ACCEPTED.json'); assert old['review']==receipt['review'] and old['source']==receipt['source']; return old
    write(s/'ACCEPTED.json',receipt)
    if scene==SCENES[-1]: write(u/'FINAL_SOURCE.json',source)
    return receipt
def pending():
    out=[]
    for p in sorted((BASE/'sources').glob('*/*/OET1_*/CUMULATIVE_SOURCE.json')):
        if not (p.parent/'REVIEW.json').exists():
            d=read(p);out.append({'path':str(p),'root_id':d['root_id'],'condition':d['condition'],'scene_ref':d['scene_ref'],'turn_count':len(d['turns'])})
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['commit','accept','pending']);ap.add_argument('--root');ap.add_argument('--arm');ap.add_argument('--scene');a=ap.parse_args()
    print(json.dumps(pending() if a.action=='pending' else globals()[a.action](a.root,a.arm,a.scene),ensure_ascii=False))
