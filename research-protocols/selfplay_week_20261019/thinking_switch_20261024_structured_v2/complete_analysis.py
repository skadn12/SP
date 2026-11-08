"""Additive descriptive reporting and fixed-subset audit packet, after raw freeze."""
from pathlib import Path
import collections, hashlib, json, statistics
HERE=Path(__file__).resolve().parent; E=HERE/'execution'
def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def check(b): assert bind(b['path'])==b,b['path']
def mean(xs): return statistics.mean(xs) if xs else None

def cell(records):
    n=len(records);mc=[r for r in records if r['strict']['question_type']=='multi_select'];order=[r for r in records if r['strict']['question_type']=='ordering']
    ps=[];rs=[];fs=[]
    for r in mc:
        s=r['strict'];pred=set(s['prediction']);gold=set(s['gold']);tp=len(pred&gold);fp=len(pred-gold);fn=len(gold-pred)
        ps.append(tp/(tp+fp) if tp+fp else 0);rs.append(tp/(tp+fn));fs.append(2*tp/(2*tp+fp+fn))
    pairs=sum(r['strict']['pairwise_total'] for r in order)
    return {'requests':n,'valid':sum(r['strict']['valid'] for r in records),'correct':sum(r['strict']['exact'] for r in records),'exact_accuracy':sum(r['strict']['exact'] for r in records)/n if n else None,'answer_readable_correct':sum(r['answer_readable_score']['exact'] for r in records),'length_truncated':sum(r['finish_reason']=='length' for r in records),'mean_output_tokens':mean([r['output_tokens'] for r in records]),'total_output_tokens':sum(r['output_tokens'] for r in records),'nonempty_thinking_blocks':sum(r['thinking_boundary']=='CLOSED_NONEMPTY' for r in records),'mean_exact_thinking_tokens':mean([r['exact_thinking_tokens'] for r in records]),'mean_backend_latency_seconds':mean([r['backend_latency_seconds'] for r in records if r['backend_latency_seconds'] is not None]),'mc_macro_precision':mean(ps),'mc_macro_recall':mean(rs),'mc_macro_f1':mean(fs),'mc_false_positive':sum(r['strict']['false_positive'] for r in mc),'mc_false_negative':sum(r['strict']['false_negative'] for r in mc),'prediction_allselect':sum(len(r['strict']['prediction'])==len(r['public']['labels']) and r['strict']['valid'] for r in mc),'ordering_pairwise_correct':sum(r['strict']['pairwise_correct'] for r in order),'ordering_pairwise_total':pairs,'ordering_pairwise_accuracy':sum(r['strict']['pairwise_correct'] for r in order)/pairs if pairs else None,'mc_item_final_mismatch':sum(r['mechanical'].get('supported_set_equals_final') is False for r in mc)}

frozen=read(E/'RAW_RESPONSE_FREEZE.json');score=read(E/'SCORES.json');assert score['raw_freeze_binding']==bind(E/'RAW_RESPONSE_FREEZE.json')
for b in frozen['bindings']: check(b)
public={r['request_id']:r for r in read(E/'PUBLIC_REQUESTS.json')['requests']};records=score['records']
for r in records:
    r['public']=public[r['request_id']];raw=read(r['raw_binding']['path']);ids=raw.get('output_token_ids',[])
    r['exact_thinking_tokens']=ids.index(151668)-ids.index(151667)-1 if ids.count(151667)==ids.count(151668)==1 and ids.index(151667)<ids.index(151668) else 0
    r['truncation_stage'] = ('THINK_NOT_CLOSED' if 151667 in ids and 151668 not in ids else 'AFTER_THINK_FINAL' if 151668 in ids else 'FINAL_WITHOUT_THINK') if r['finish_reason']=='length' else None
modes=('THINK_OFF','THINK_ON');main=[];groups={'ALL':lambda s:True,'MC_ALL':lambda s:s['question_type']=='multi_select','MC2_5_PRIMARY':lambda s:s['gold_count'] in (2,3,4,5),'MC6_CONTROL':lambda s:s['gold_count']==6,'ORDERING':lambda s:s['question_type']=='ordering'}
for mode in modes:
    for group,predicate in groups.items():main.append({'mode':mode,'group':group,**cell([r for r in records if r['mode']==mode and predicate(r['strict'])])})
breakdowns={}
for field in ('root_id','task_operation','variant_id','gold_count'):
    values=sorted({r['strict'][field] for r in records if r['strict'][field] is not None},key=str)
    breakdowns[field]=[{'mode':m,field:v,**cell([r for r in records if r['mode']==m and r['strict'][field]==v])} for m in modes for v in values]
byq=collections.defaultdict(dict)
for r in records:byq[(r['mode'],r['strict']['candidate_id'])][r['strict']['variant_id']]=r
stability=[]
for (mode,cid),q in sorted(byq.items()):
    a,b=q['perm_00']['strict'],q['perm_01']['strict'];valid=a['valid'] and b['valid']
    same=(set(a['prediction'])==set(b['prediction']) if a['question_type']=='multi_select' else a['prediction']==b['prediction']) if valid else False
    stability.append({'mode':mode,'candidate_id':cid,'both_valid':valid,'canonical_prediction_same':same,'both_exact':a['exact'] and b['exact'],'correct_displays':int(a['exact'])+int(b['exact'])})
bybase=collections.defaultdict(dict)
for r in records:bybase[r['base_request_id']][r['mode']]=r
pairs=[]
for key,m in bybase.items():
    a,b=m['THINK_OFF'],m['THINK_ON'];ae,be=a['strict']['exact'],b['strict']['exact']
    pairs.append({'base_request_id':key,'candidate_id':a['strict']['candidate_id'],'variant_id':a['strict']['variant_id'],'off_correct':ae,'on_correct':be,'transition':'BOTH_CORRECT' if ae and be else 'ON_GAIN' if be else 'ON_LOSS' if ae else 'BOTH_WRONG'})
qpair=[]
for cid in sorted({r['strict']['candidate_id'] for r in records}):
    off=next(r for r in stability if r['mode']=='THINK_OFF' and r['candidate_id']==cid);on=next(r for r in stability if r['mode']=='THINK_ON' and r['candidate_id']==cid)
    qpair.append({'candidate_id':cid,'off_correct_displays':off['correct_displays'],'on_correct_displays':on['correct_displays'],'delta_correct_displays':on['correct_displays']-off['correct_displays']})
result={'source_score_binding':bind(E/'SCORES.json'),'raw_freeze_binding':bind(E/'RAW_RESPONSE_FREEZE.json'),'unique_questions':30,'roots':4,'main':main,'breakdowns':breakdowns,'display_stability':stability,'display_stability_summary':[{'mode':m,'questions':30,'both_valid':sum(r['both_valid'] for r in stability if r['mode']==m),'canonical_prediction_same':sum(r['canonical_prediction_same'] for r in stability if r['mode']==m),'both_exact':sum(r['both_exact'] for r in stability if r['mode']==m)} for m in modes],'paired_display_transitions':dict(collections.Counter(p['transition'] for p in pairs)),'display_pairs':pairs,'question_pairs':qpair,'paired_question_direction':{'gain':sum(p['delta_correct_displays']>0 for p in qpair),'loss':sum(p['delta_correct_displays']<0 for p in qpair),'tie':sum(p['delta_correct_displays']==0 for p in qpair)},'thinking_boundary_counts':{m:dict(collections.Counter(r['thinking_boundary'] for r in records if r['mode']==m)) for m in modes},'raw_run_wall_seconds':frozen['wall_seconds'],'semantic_audit':'PENDING_FIXED_SUBSET','limits':'Train-only 4 roots; 30 unique questions with dependent displays; no new training/feedback causality. Inherited mc_allselect_* aggregation is deliberately excluded.'}
save(E/'COMPARISON.json',result)
save(E/'COST_AND_FINISH_DETAILS.json', {'per_mode': [{'mode':m,'median_output_tokens':statistics.median([r['output_tokens'] for r in records if r['mode']==m]),'max_output_tokens':max(r['output_tokens'] for r in records if r['mode']==m),'finish_reason_counts':dict(collections.Counter(r['finish_reason'] for r in records if r['mode']==m)),'truncation_stage_counts':dict(collections.Counter(r['truncation_stage'] for r in records if r['mode']==m and r['truncation_stage'] is not None))} for m in modes], 'length_cases':[{'request_id':r['request_id'],'mode':r['mode'],'stage':r['truncation_stage'],'output_tokens':r['output_tokens'],'exact_thinking_tokens':r['exact_thinking_tokens'],'raw_binding':r['raw_binding']} for r in records if r['finish_reason']=='length'], 'measurement_note':'Exact thinking tokens count the original token IDs strictly between generated think markers; wall time includes the shared model load and all mixed batches. A truncation after the closing marker is not reported as unfinished thinking.'})
selection=read(E/'SEMANTIC_AUDIT_SELECTION.json')['records'];fixed=[s for s in selection if s['candidate_id']];packet=[]
for i,s in enumerate(fixed):
    matched=[r for r in records if r['strict']['candidate_id']==s['candidate_id'] and r['strict']['variant_id']==s['variant_id']];assert len(matched)==2
    row=matched[0]['public'];check(row['source_binding']);src=read(row['source_binding']['path']);tp=HERE/'semantic_audit/sources'/(row['root_id']+'.txt')
    if not tp.exists():
        tp.parent.mkdir(parents=True,exist_ok=True)
        with tp.open('x') as f:
            for t in src['turns']:f.write(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}\n")
    packet.append({'candidate_id':s['candidate_id'],'reviewer_group':'a' if i<3 else 'b','root_id':row['root_id'],'variant_id':s['variant_id'],'question_type':row['question_type'],'operation':row['task_operation'],'question':row['question'],'display':row['display'],'source_binding':row['source_binding'],'source_transcript_binding':bind(tp),'turns':len(src['turns']),'display_gold':matched[0]['strict']['display_gold'],'responses':[{k:r[k] for k in ('request_id','mode','thinking_text','final_text','parsed','strict','errors','thinking_boundary','raw_binding')} for r in sorted(matched,key=lambda r:r['mode'])]})
save(HERE/'semantic_audit/SEMANTIC_AUDIT_PACKET.json',{'selection_binding':bind(E/'SEMANTIC_AUDIT_SELECTION.json'),'score_binding':bind(E/'SCORES.json'),'requests':len(packet)*2,'cases':packet,'limits':'Non-author independent AI, Gold visible, shared workspace; all sources read; fixed subset only; not internal-faithfulness certification.'})
save(HERE/'semantic_audit/REVIEW_CONTRACT.json',{'selection':'First 3 nonmissing fixed cases group a; last 3 group b. No outcome-based replacement.','per_response_fields':['request_id','mode','final_correct','final_reasons_supported','citations_supported','item_final_consistent','thinking_factuality','thinking_adds_verifiable_error_beyond_own_final','thinking_adds_verifiable_correct_check_beyond_own_final','item_checks','notes'],'status_values':['PASS','FAIL','UNKNOWN','NA'],'new_evidence_status':['OBSERVED','NOT_OBSERVED','UNASSESSABLE','NA'],'instructions':['Read the complete bound source and both complete response texts for each assigned question.','Check all displayed items against the precise question and source; include turn references for factual claims.','Judge thinking separately from final JSON; more text does not imply more valid reasoning.','A new-error observation requires a concrete checkable wrong step absent the same response final JSON, with a short quote and source-based correction.','Correct extra computations may improve transparency without giving new failure diagnosis; report these separately.','No thinking block means NA, never proof of hidden cognition.','Gold is visible; retain any source/Gold concern and do not silently rewrite it.','Conclusions only cover the fixed six-question subset and are exploratory.']})
print(json.dumps({'main':main,'paired':result['paired_display_transitions'],'question_direction':result['paired_question_direction'],'thinking':result['thinking_boundary_counts'],'audit_cases':len(packet)},ensure_ascii=False))
