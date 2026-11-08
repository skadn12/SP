import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

A=Path(__file__).parent
E=A.parent/'feedback_closed_loop_20261024_v1'
ROOTS=['218519','222937','225575','229686']
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mapped_label(can, mapping):
    if isinstance(can,list):return [mapping[k] for k in can]
    if '<' in can:return '<'.join(mapping[k] for k in can.split('<'))
    return mapping[can]
checks=[]
def check(name, ok, detail=None):
    checks.append(dict(name=name,pass_=bool(ok),detail=detail))
    if not ok: raise AssertionError((name,detail))

bindings=read(A/'INPUT_BINDINGS.json')
for b in bindings['bindings']:
    check('binding '+b['path'],sha(b['path'])==b['sha256'])
manifest=read(E/'FINAL_MANIFEST.json')
for b in manifest['files']:
    p=E/b['path']
    check('sealed_file '+b['path'],p.stat().st_size==b['bytes'] and sha(p)==b['sha256'])
official={r['request_id']:r for r in read(E/'execution/SCORES.json')['records'] if r['arm']=='G'}
all_rows=[]; all_responses=[]; cards=[]; gold_concerns=[]
for root in ROOTS:
    B=A/'reviews'/root
    packet=read(B/'ERROR_REVIEW_PACKET.json')
    rp=B/('ERROR_REVIEW_EFFECTIVE.json' if root=='218519' else 'ERROR_REVIEW.json')
    review=read(rp); cr=read(B/'CARD_MATCH_REVIEW.json')
    cp=read(B/'CARD_MATCH_PACKET.json')
    source=read(packet['source_binding']['path'])
    turns={t['turn_ref']:t['text'] for t in source['turns']}
    orig=read(B/'ERROR_REVIEW.json')
    check(root+' phase1 hash unchanged',sha(cr['phase1_binding']['path'])==cr['phase1_binding']['sha256'])
    check(root+' source hash',sha(packet['source_binding']['path'])==packet['source_binding']['sha256'])
    check(root+' 8 questions',len(review['records'])==len(cr['records'])==len(packet['records'])==8)
    byid={r['candidate_id']:r for r in review['records']}; cby={r['candidate_id']:r for r in cr['records']}
    for row in packet['records']:
        c=row['candidate']; rid=c['candidate_id']; rr=byid[rid]; card=cby[rid]
        check(rid+' question identity',rr['slot_id']==c['slot_id'] and rr['question_type']==c['question_type'])
        check(rid+' two repeats',len(rr['responses'])==len(row['responses'])==len(card['responses'])==2)
        if rr['gold_concern']:gold_concerns.append(dict(candidate_id=rid,details=rr['gold_concern_details']))
        rev={r['request_id']:r for r in rr['responses']}; cardrev={r['request_id']:r for r in card['responses']}
        joined=[]
        for response in row['responses']:
            req=response['request_id']; x=rev[req]; z=cardrev[req]; score=official[req]; st=score['strict']
            check(req+' official agreement',x['strict_exact']==st['exact'] and x['response_valid']==st['valid'] and score==response['score'])
            check(req+' raw hash',sha(response['raw_binding']['path'])==response['raw_binding']['sha256'])
            check(req+' score separation',(x['primary_failure_code'] is None)==bool(st['exact']) and (not x['final_failure_codes'] if st['exact'] else bool(x['final_failure_codes'])))
            check(req+' codes known',all(code in packet['taxonomy'] for code in x['final_failure_codes']+x['reasoning_error_codes']+([x['primary_failure_code']] if x['primary_failure_code'] else [])))
            for idx,ev in enumerate(x['evidence']):
                can=ev['canonical_label_or_pair']; disp=ev['display_label_or_pair']
                expected=mapped_label(can,response['presentation']['canonical_to_display'])
                check(req+f' phase1 evidence {idx} display',expected==disp)
                check(req+f' phase1 evidence {idx} raw quote',bool(ev['raw_quote']) and ev['raw_quote'] in response['raw_output'])
                for sc in ev['source_checks']:
                    check(req+f' evidence {idx} source '+sc['turn_ref'],sc['turn_ref'] in turns and bool(sc['source_quote']) and sc['source_quote'] in turns[sc['turn_ref']])
            for idx,ev in enumerate(z['evidence']):
                can=ev['canonical_label_or_pair']; disp=ev['display_label_or_pair']
                expected=mapped_label(can,response['presentation']['canonical_to_display'])
                check(req+f' phase2 evidence {idx} display',expected==disp)
                check(req+f' phase2 evidence {idx} raw quote',bool(ev['raw_quote']) and ev['raw_quote'] in response['raw_output'])
                check(req+f' phase2 evidence {idx} source',all(k in turns for k in ev['source_refs']))
            check(req+' target claim admissibility',not z['final_mechanism_error_confirmed'] or (card['target_realization']=='PASS' and not card['control'] and not st['exact'] and z['visible_mechanism_status']=='CONFIRMED'))
            check(req+' HOLD excluded',card['target_realization']!='HOLD' or not (z['final_mechanism_error_confirmed'] or z['reason_only_target_error'] or z['citation_only_target_error']))
            pub=response['public_request']
            check(req+' FULL THINK_OFF',pub['condition']=='FULL' and pub['mode']=='THINK_OFF' and pub['enable_thinking'] is False)
            merged=dict(root_id=root,candidate_id=rid,slot_id=c['slot_id'],task_operation=c['task_operation'],request_id=req,official_score=st,contract_errors=score['errors'],review=x,card_match=z,raw_binding=response['raw_binding'],review_path=str(rp))
            joined.append(merged);all_responses.append(merged)
        all_rows.append(dict(root_id=root,candidate_id=rid,slot_id=c['slot_id'],task_operation=c['task_operation'],question_type=c['question_type'],candidate=c,content_review=row['content_review'],review=rr,card_match=card,responses=joined,review_path=str(rp)))
        cards.append(card)

check('64 G responses exact coverage',len(all_responses)==64 and {r['request_id'] for r in all_responses}==set(official))
check('32 G questions',len(all_rows)==32 and len({r['candidate_id'] for r in all_rows})==32)
def response_stats(rows):
    return dict(responses=len(rows),exact=sum(r['official_score']['exact'] for r in rows),valid=sum(r['official_score']['valid'] for r in rows),citation_defect=sum(r['review']['citation_defect'] for r in rows),wrong=sum(not r['official_score']['exact'] for r in rows))
def question_stats(rows):
    both=sum(all(x['official_score']['exact'] for x in r['responses']) for r in rows)
    any_=sum(any(x['official_score']['exact'] for x in r['responses']) for r in rows)
    same=0
    for r in rows:
        a,b=[x['official_score']['prediction'] for x in r['responses']]
        same+=set(a or [])==set(b or []) if r['question_type']=='multi_select' else a==b
    return dict(questions=len(rows),both_correct=both,any_correct=any_,both_wrong=len(rows)-any_,same_canonical_prediction=same)

by_slot={slot:dict(**response_stats([x for x in all_responses if x['slot_id']==slot]),**question_stats([r for r in all_rows if r['slot_id']==slot])) for slot in ['MC2','MC3','MC4','MC5','MC6','ORDER1','ORDER2','ORDER3']}
by_root={root:dict(**response_stats([x for x in all_responses if x['root_id']==root]),**question_stats([r for r in all_rows if r['root_id']==root])) for root in ROOTS}
mc=[x for x in all_responses if x['slot_id'] in ['MC2','MC3','MC4','MC5']]
order=[x for x in all_responses if x['slot_id'].startswith('ORDER')]
noncontrol=[x for x in all_responses if x['slot_id']!='MC6']
primary=Counter(r['review']['primary_failure_code'] for r in all_responses if not r['official_score']['exact'])
final_codes=Counter(code for r in all_responses for code in set(r['review']['final_failure_codes']))
reason_codes=Counter(code for r in all_responses for code in set(r['review']['reasoning_error_codes']))
wrong_unknown=[r['request_id'] for r in all_responses if r['review']['primary_failure_code']=='UNKNOWN_CAUSE']
realized=[c for c in cards if c['target_realization']=='PASS' and not c['control']]
card_noncontrol=[c for c in cards if not c['control']]
confirmed=[c for c in realized if any(r['final_mechanism_error_confirmed'] for r in c['responses'])]
target_summary=dict(noncontrol_questions=len(card_noncontrol),realized_questions=len(realized),hold_questions=len(card_noncontrol)-len(realized),confirmed_final_error_questions=len(confirmed),both_repeat_confirmed_questions=sum(all(r['final_mechanism_error_confirmed'] for r in c['responses']) for c in realized),confirmed_final_error_responses=sum(r['final_mechanism_error_confirmed'] for c in realized for r in c['responses']),reason_only_responses=sum(r['reason_only_target_error'] for c in cards for r in c['responses']),citation_only_responses=sum(r['citation_only_target_error'] for c in cards for r in c['responses']),post_hoc_exploratory=True,comparable_to_F_prespecified_target_rate=False)
summary=dict(schema_version='g_solver_error_summary_v1',analysis_date='2026-10-25',scope='G, FULL, B1, THINK_OFF; frozen responses reused',new_solver_calls=0,training=False,overall=dict(**response_stats(all_responses),**question_stats(all_rows)),noncontrol=response_stats(noncontrol),mc_gold_2_to_5=dict(**response_stats(mc),false_positive=sum(x['official_score']['false_positive'] for x in mc),false_negative=sum(x['official_score']['false_negative'] for x in mc)),ordering=dict(**response_stats(order),pairwise_correct=sum(x['official_score']['pairwise_correct'] or 0 for x in order),pairwise_total=sum(x['official_score']['pairwise_total'] or 0 for x in order),valid_conditioned_pairwise_total=sum(x['official_score']['pairwise_total'] or 0 for x in order if x['official_score']['valid'])),control_mc6=response_stats([x for x in all_responses if x['slot_id']=='MC6']),by_slot=by_slot,by_root=by_root,primary_failure_codes=dict(primary),nonexclusive_final_failure_codes=dict(final_codes),nonexclusive_reasoning_codes=dict(reason_codes),wrong_unknown=wrong_unknown,citation_defect=dict(all=sum(x['review']['citation_defect'] for x in all_responses),among_correct=sum(x['review']['citation_defect'] and x['official_score']['exact'] for x in all_responses),among_wrong=sum(x['review']['citation_defect'] and not x['official_score']['exact'] for x in all_responses)),correct_with_non_citation_reason_errors=sum(x['official_score']['exact'] and bool(set(x['review']['reasoning_error_codes'])-{'EVIDENCE_SCOPE_OR_CITATION_BINDING_ERROR'}) for x in all_responses),gold_concerns=gold_concerns,original_card_exploratory_match=target_summary,frozen_experiment_unchanged=dict(manifest_sha256=sha(E/'FINAL_MANIFEST.json'),files_verified=len(manifest['files'])),interpretation_limits=['同题两展示是重复测量，不是64道独立题。','主分类为可见输出描述，不证明隐藏内部过程或长上下文致错。','自动合同有效不等于引用语义正确；引用缺陷不会自动重写原exact。','旧卡映射是事后且非盲审，不能与F预先定义target率作因果比较。'])
(A/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(A/'VALIDATION.json').write_text(json.dumps(dict(status='PASS',check_count=len(checks),checks=checks,new_solver_calls=0,training=False),ensure_ascii=False,indent=2)+'\n')
with (A/'QUESTION_INDEX.csv').open('w',newline='') as f:
    fields=['root_id','candidate_id','slot_id','task_operation','gold','perm00_prediction','perm00_exact','perm01_prediction','perm01_exact','perm00_primary','perm01_primary','review_path']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in all_rows:
        x,y=r['responses'];w.writerow(dict(root_id=r['root_id'],candidate_id=r['candidate_id'],slot_id=r['slot_id'],task_operation=r['task_operation'],gold=' '.join(r['candidate']['gold']),perm00_prediction=' '.join(x['official_score']['prediction'] or []),perm00_exact=x['official_score']['exact'],perm01_prediction=' '.join(y['official_score']['prediction'] or []),perm01_exact=y['official_score']['exact'],perm00_primary=x['review']['primary_failure_code'],perm01_primary=y['review']['primary_failure_code'],review_path=r['review_path']))
with (A/'RESPONSE_INDEX.jsonl').open('w') as f:
    for r in all_responses:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
