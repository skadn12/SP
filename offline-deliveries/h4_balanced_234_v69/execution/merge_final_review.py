from pathlib import Path
import json,hashlib,datetime
R=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_balanced_234_v69')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'EVALUATION_PREPARATION.STARTED.json').exists()
pool=read(R/'POOL.json'); questions={q['candidate_id']:q for q in pool['questions']}
rows=[];sources=[]
for g in ('a','b'):
    jp=R/'final_review'/('group_'+g+'_SOURCE_JUDGMENTS.json')
    ap=R/'final_review'/('group_'+g+'_AUDIT.json')
    j=read(jp); a=read(ap)
    assert j['gold_read_before_save'] is False and j['solver_scores_used'] is False
    assert a['solver_scores_used'] is False and a['pool_sha256']==sha(R/'POOL.json')
    assert a['source_judgments_sha256']==sha(jp)
    judges={q['candidate_id']:q for q in j['questions']}
    assert set(judges)=={q['candidate_id'] for q in a['questions']}
    for ar in a['questions']:
        q=questions[ar['candidate_id']]; sj=judges[ar['candidate_id']]
        assert ar['root_id']==sj['root_id']==q['root_id']
        assert ar['source_sha256']==sj['source_sha256']==q['source_sha256']==sha(Path(q['source_path']))
        assert set(sj['options'])==set('ABCDEF')
        turns={t['turn_ref']:t for t in read(Path(q['source_path']))['turns']}
        for v in sj['options'].values():
            assert v['relation'] in ('supports','contradicts','unknown')
            for c in v['citations']:
                assert c['turn_ref'] in turns and c['quote'] in turns[c['turn_ref']]['text'], (q['candidate_id'],c)
        if ar['core_status']=='CORE_USABLE':
            assert sj['stem_valid'] is True
            assert all(v['relation']!='unknown' for v in sj['options'].values())
            assert {k for k,v in sj['options'].items() if v['relation']=='supports'}==set(q['gold'])
        rows.append(ar)
    sources.extend([{'path':str(f),'sha256':sha(f)} for f in (jp,ap)])
assert len(rows)==len(questions) and {q['candidate_id'] for q in rows}==set(questions)
# Protocol exclusions were declared before assembly and any solver outcomes.
protocol_path=R/'execution/PROTOCOL_EXCLUSIONS.json'
if protocol_path.exists():
    protocol=read(protocol_path)
    assert protocol['before_assembly'] is True and protocol['before_solver'] is True and protocol['solver_scores_used'] is False
    selections=read(R/'ASSEMBLY_RECEIPTS.json')['selections']
    excluded_slots={row['slot_id'] for row in protocol['exclusions']}
    excluded_ids={row['candidate_id'] for row in selections if row['slot_id'] in excluded_slots}
    statuses={row['candidate_id']:row['core_status'] for row in rows}
    assert all(statuses[qid]=='KNOWN_SUBSTANTIVE_ISSUE' for qid in excluded_ids)
    sources.append({'path':str(protocol_path),'sha256':sha(protocol_path)})
value={'pool_sha256':sha(R/'POOL.json'),'solver_scores_used':False,'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'independent_review_files':sources,'questions':rows}
with (R/'FINAL_CONTENT_REVIEW.json').open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'questions':len(rows),'usable':sum(q['core_status']=='CORE_USABLE' for q in rows)}))
