#!/usr/bin/env python3
"""Pilot bookkeeping only; never generates or edits experimental candidates."""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone
import pilot_artifacts as pa

E = Path(__file__).resolve().parents[1]
SLOT_FIELDS = ('slot_id', 'natural_information_need', 'precise_query_scope', 'facts',
               'semantic_answer', 'answer_program', 'required_relations',
               'alternative_sufficient_paths', 'unresolved_relations', 'source_scope_note')

def opportunities():
    return pa.read(E / 'manifests/SCHEDULE_RESTRICTED.json')['opportunities']

def folder(n):
    return E / 'isolated_authors' / next(r['opportunity_id'] for r in opportunities()
                                         if int(r['opportunity_id'].split('_')[1]) == n)

def selector(n):
    d = folder(n)
    i, s = pa.read(d / 'INPUT.json'), pa.read(d / 'NEW_SLOTS.json')
    freeze = pa.read(d / 'SLOTS_FREEZE.json')
    # Freeze schema differs between isolated authors; bind original bytes here too.
    assert len(s['slots']) <= 3
    out = E / 'selector_packets' / f'SEL_{n:02d}.json'
    packet = {k: i[k] for k in ('source_path', 'source_sha256', 'source_turn_count', 'neutral_path')}
    packet.update(coverage_path=str(E / 'manifests/selector_baseline' / (i['root_id'] + '.json')),
                  operation_summary_path=str(E / 'manifests/SELECTOR_OPERATION_SUMMARY.json'),
                  slots=[{k: x[k] for k in SLOT_FIELDS if k in x} for x in s['slots']],
                  slots_sha256=pa.digest(d / 'NEW_SLOTS.json'),
                  output_path=str(E / 'selector_results' / f'SEL_{n:02d}.json'))
    pa.write_new(out, packet)
    return {'packet': str(out), 'sha256': pa.digest(out)}

def scan():
    originals = {1: 'author_raw/OPP_01_ac7f8020', 3: 'drafts/OPP_03_bd999ccb',
                 5: 'drafts/OPP_05_bb97416d', 6: 'author_raw/OPP_06_cd4999d3'}
    rows = []
    for r in opportunities():
        n = int(r['opportunity_id'].split('_')[1]); d = folder(n)
        if n in originals: d = E / originals[n]
        row = dict(r)
        if n in (2,4):
            row.update(status='INVALIDATED_ROLE_CONTAMINATION_NO_RETRY')
        else:
            draft = d / 'CANDIDATES_FIRST_DRAFT.json'
            if draft.exists():
                row['draft_path'] = str(draft)
                row['mechanical'] = pa.audit(draft)
                if row['mechanical']['status'] == 'NO_CANDIDATE':
                    row['status'] = 'NO_CANDIDATE_FROZEN'
                else:
                    code = 'BQ_' + pa.digest(draft)[:12]
                    packet = E / 'review/fresh_blind_packets' / (code + '.json')
                    if not packet.exists(): pa.blind(draft)
                    row.update(blind_id=code, blind_packet=str(packet), status='DRAFT_FROZEN')
                    review = E / 'review/fresh_blind_results' / code / 'REVIEW.json'
                    if review.exists(): row.update(status='BLIND_REVIEW_FROZEN', review_path=str(review))
            elif (d / 'NEW_SLOTS.json').exists():
                row.update(status='SLOTS_FROZEN', slots_path=str(d / 'NEW_SLOTS.json'))
                sel = E / 'selector_results' / f'SEL_{n:02d}.json'
                if sel.exists(): row.update(status='SELECTION_FROZEN', selection_path=str(sel))
            else: row['status'] = 'UNSTARTED_OR_AUTHOR_RUNNING'
        if n in (3,5): row['protocol_interpretation'] = 'DESCRIPTIVE_ONLY_DEVIATING_SELECTION_AND_IMPLEMENTER'
        if n == 1: row['protocol_interpretation'] = 'DESCRIPTIVE_ONLY_POST_DRAFT_TARGET_AND_HASH_REWRITE_FROM_PROVENANCE_AUDIT'
        rows.append(row)
    value={'created_utc':datetime.now(timezone.utc).isoformat(), 'opportunities':rows,
           'no_training':True, 'no_exports':True}
    # Mutable controller index only; never used as a pre-generation experimental freeze.
    (E / 'manifests/LIVE_PROGRESS.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return [{'n':int(r['opportunity_id'].split('_')[1]),'group':r['group'],'status':r['status'],
             'mechanical':r.get('mechanical',{}).get('status'),'blind_id':r.get('blind_id')} for r in rows]

def review_status():
    scan()
    rows=[]
    for r in pa.read(E / 'manifests/LIVE_PROGRESS.json')['opportunities']:
        if 'blind_id' not in r: continue
        d=E/'review/fresh_blind_results'/r['blind_id']
        if not (d/'REVIEW.json').exists(): continue
        c=pa.candidate(r['draft_path']); first=pa.read(d/'FIRST_ANSWER.json'); rev=pa.read(d/'REVIEW.json')
        recovered=first.get('recovered_answer_option_ids',first.get('independent_answer',{}).get('selected_option_ids'))
        gold=c.get('gold_option_ids',c.get('gold'))
        if c['question_type']=='partial_order':
            recovered=first.get('recovered_gold_pairs'); gold=c.get('gold_pairs')
            agree=None if recovered is None else sorted(map(tuple,recovered))==sorted(map(tuple,gold))
        elif recovered is None: agree=None
        elif c['question_type']=='ordering': agree=recovered==gold
        else: agree=sorted(recovered)==sorted(gold)
        status=rev.get('quality_status',rev.get('quality',{}).get('verdict'))
        rows.append({'opportunity_id':r['opportunity_id'],'candidate_id':c['candidate_id'],
                     'blind_id':r['blind_id'],'quality_status':status,'gold_agreement':agree,
                     'mechanical_status':r['mechanical']['status'],
                     'draft_path':r['draft_path'],'draft_sha256':pa.digest(r['draft_path']),
                     'first_answer_path':str(d/'FIRST_ANSWER.json'),
                     'first_answer_sha256':pa.digest(d/'FIRST_ANSWER.json'),
                     'review_path':str(d/'REVIEW.json'),'review_sha256':pa.digest(d/'REVIEW.json'),
                     'protocol_interpretation':r.get('protocol_interpretation','NO_KNOWN_DEVIATION_PENDING_FINAL_RUNTIME_AUDIT')})
    (E/'manifests/LIVE_REVIEW_STATUS.json').write_text(json.dumps({'items':rows},ensure_ascii=False,indent=2)+'\n')
    return [{k:r[k] for k in ('candidate_id','quality_status','gold_agreement','mechanical_status')} for r in rows]

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode', choices=('scan','selector','review-status'));p.add_argument('n',nargs='?',type=int)
    a=p.parse_args();print(json.dumps({'scan':scan,'review-status':review_status}.get(a.mode,lambda:selector(a.n))(),ensure_ascii=False,indent=2))
