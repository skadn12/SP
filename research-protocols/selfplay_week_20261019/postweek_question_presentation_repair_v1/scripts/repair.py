"""Immutable, shared stem-only repair of the 56 frozen FBC1 drafts.

No author regeneration, source expansion, answer changes, or solver selection.
The complete old question and source/operation review bindings are retained.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[1]
PARENT = BASE.parent / 'postweek_feedback_control_4root_v1'
WHICH = r'\b(which)\s+(?:two|three|four|five|six|[2-6])\s+'
EXACT = r'\s*Select\s+exactly\s+(?:two|three|four|five|six|[2-6])\s*\.'
NEUTRAL = 'Which of the following statements are supported by the record? Select all that apply.'
SLOTS = ['MC2','MC3','MC4','MC5','MC6','ORDER1','ORDER2']

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(p,d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def seed(cid, branch):
    return int.from_bytes(hashlib.sha256(('fbc1:20261023:'+cid+':'+branch).encode()).digest()[:4],'big') % 2147483647

def transform():
    protocol=read(BASE/'REPAIR_PROTOCOL.json')
    assert protocol['status']=='FROZEN_BEFORE_REPAIR'
    rows=[]; seeds={}
    for root in protocol['roots']:
        for arm in ['GENERIC','TARGETED']:
            old_review=PARENT/'question_reviews'/root/arm/'REVIEW.json'
            rr=read(old_review)
            assert rr['full_source_read'] and not rr['solver_scores_read']
            for slot in SLOTS:
                p=PARENT/'questions'/root/arm/(slot+'.json'); q=read(p)
                before=q['question']; after=before; n=m=0
                if slot=='MC6': after=NEUTRAL
                elif slot.startswith('MC'):
                    after,n=re.subn(WHICH,lambda x:x.group(1)+' ',before,flags=re.I)
                    after,m=re.subn(EXACT,'',after,flags=re.I)
                    assert n==1 and m<=1
                assert not re.search(WHICH,after,re.I)
                assert not re.search(EXACT,after,re.I)
                if slot.startswith('MC'): assert not re.search(r'all.correct|audit.control|record.audit.control',after,re.I)
                cid='QPR1_'+hashlib.sha256(q['candidate_id'].encode()).hexdigest()[:16]
                nq={**q,'candidate_id':cid,'question':after,
                    'presentation_parent_binding':bind(p),
                    'presentation_transform':{'version':'qpr1_stem_only_v1','editor_id':'/root/program',
                        'created_at_utc':now(),'which_count_removed':n,'exact_count_removed':m,
                        'mc6_neutral_stem':slot=='MC6','author_regeneration':False,
                        'gold_used_for_text_replacement':False,'slot_used_for_neutral_mc6_rule':slot=='MC6'}}
                assert {k:v for k,v in nq.items() if k not in ['candidate_id','question','presentation_parent_binding','presentation_transform']} == {k:v for k,v in q.items() if k not in ['candidate_id','question']}
                assert q['source_binding']==bind(q['source_binding']['path'])
                dest=BASE/'questions'/root/arm/(slot+'.json'); save(dest,nq)
                oldrow=next(x for x in rr['questions'] if x['slot_id']==slot)
                assert oldrow['submitted_gold_matches'] is True and oldrow['checks']['grounding']=='PASS' and oldrow['checks']['unique_key']=='PASS'
                rows.append({'root_id':root,'condition':arm,'slot_id':slot,'candidate_id':cid,
                    'parent_candidate_id':q['candidate_id'],'original':bind(p),'repaired':bind(dest),
                    'original_content_review':bind(old_review),'carried_operation_status':oldrow['checks']['actual_operation'],
                    'before':before,'after':after,'changed':before!=after,'which_count_removed':n,
                    'exact_count_removed':m,'mc6_neutral_stem':slot=='MC6',
                    'source_options_items_gold_evidence_display_unchanged':True})
                seeds[cid]={b:seed(q['candidate_id'],b) for b in ['direct','reason']}
    assert len(rows)==56
    save(BASE/'TRANSFORM_RECEIPT.json',{'status':'REPAIRED_PENDING_INDEPENDENT_DELTA_REVIEW',
        'created_at_utc':now(),'records':rows,'new_sources':0,'author_attempts_added':0,
        'candidate_count':len(rows),'changed_stems':sum(x['changed'] for x in rows),
        'which_count_removed':sum(x['which_count_removed'] for x in rows),
        'exact_count_removed':sum(x['exact_count_removed'] for x in rows),
        'neutral_mc6_stems':sum(x['mc6_neutral_stem'] for x in rows),
        'options_items_keys_display_sources_unchanged':True,
        'old_grounding_and_key_reviews_carried_not_new_semantic_reviews':True,
        'solver_scores_used_for_repair':False,'protocol_binding':bind(BASE/'REPAIR_PROTOCOL.json'),
        'script_binding':bind(__file__)})
    save(BASE/'inputs/PUBLIC_SAMPLING_SEEDS.json',{'seeds':seeds,'rule':'Same request-specific FBC1 parent seeds; no Gold used.'})
    print(json.dumps({'questions':len(rows),'changed_stems':sum(x['changed'] for x in rows),'which_removed':sum(x['which_count_removed'] for x in rows),'exact_removed':sum(x['exact_count_removed'] for x in rows)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['transform']); a=ap.parse_args(); transform()
