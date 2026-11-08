"""Independently recompute new-branch results from frozen raw JSON; no model run."""
import collections, hashlib, importlib.util, json, math, re, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
P=Path(__file__).resolve().parents[1]; E=P/'execution'
def read(p):return json.loads(Path(p).read_text())
def binding(p):
    p=Path(p).resolve();raw=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
spec=importlib.util.spec_from_file_location('independent_result_receipts',P/'runner.py')
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
prep,rows,tokens=r.public();freeze=read(E/'RAW_RESPONSE_FREEZE.json');r.checked_raw(freeze,rows,tokens)
plan=read(P/'PILOT_PLAN.json')
r.paths_equal(prep['private_bindings'],r.private_paths(plan),'INDEPENDENT_PRIVATE_SET')
for b in prep['private_bindings']:r.check(b)
candidates={c['candidate_id']:c for c in read(P/'pilot/CANDIDATES_FINAL.json')}
private=r.validate_private(rows,read(E/'PRIVATE_SCORING.json')['requests'],candidates)
official={x['request_id']:x for x in read(E/'SCORES.json')['records']}
summary=read(E/'ANALYSIS_SUMMARY.json')
records=[];differences=[]
def unique_keys(pairs):
    assert len(pairs)==len(set(k for k,v in pairs))
    return dict(pairs)
def same(a,b):
    return a==b or (isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isclose(a,b,abs_tol=1e-12))
for row in rows:
    raw=read(E/'raw'/(row['request_id']+'.json'));info=private[row['request_id']]
    assert raw['status']=='RETURNED' and raw['finish_reason']=='stop' and not raw['capture_errors']
    # The actual outputs are direct JSON, so no runner split/score implementation is needed.
    parsed=json.loads(raw['raw_output'],object_pairs_hook=unique_keys)
    boxed=parsed.get('final_answer');match=re.fullmatch(r'<box>([A-Z](?: [A-Z])*)?</box>',boxed) if isinstance(boxed,str) else None
    labels=(match[1] or '').split() if match else []
    label_set=set(row['labels'])
    valid=bool(match) and bool(labels) and len(labels)==len(set(labels)) and set(labels)<=label_set
    valid=valid and (len(labels)==1 if row['question_type']=='single_choice' else set(labels)==label_set)
    pred=[info['presentation']['display_to_canonical'][x] for x in labels if x in label_set]
    gold=info['presentation']['canonical_gold'];exact=valid and pred==gold
    refs=parsed.get('turn_refs');ref_ok=isinstance(refs,list) and all(isinstance(x,str) for x in refs)
    ref_ok=ref_ok and len(refs)<=8 and len(refs)==len(set(refs)) and set(refs)<=set(info['visible_refs'])
    partial,lcs,cp,ap=None,None,None,None
    if row['question_type']=='ordering' and valid:
        # Inversions of Gold ranks and LIS equal pair concordance and LCS respectively.
        ranks=[gold.index(x) for x in pred];n=len(gold);ap=n*(n-1)//2
        cp=ap-sum(ranks[i]>ranks[j] for i in range(n) for j in range(i+1,n));partial=cp/ap
        best=[]
        for i,v in enumerate(ranks):best.append(1+max([best[j] for j in range(i) if ranks[j]<v] or [0]))
        lcs=max(best)/n
    cert=info['dependency_certificate'];target=cert.get('target_option',cert.get('target_pair'));ret=None
    if valid:ret=pred==[target] if row['question_type']=='single_choice' else pred.index(target[0])<pred.index(target[1])
    out={k:row[k] for k in ('request_id','candidate_id','question_type','variant_id','condition')}
    out.update(answer_valid=valid,exact=exact,prediction=pred,gold=gold,reference_compliant=bool(ref_ok),
        pairwise_accuracy=partial,normalized_lcs=lcs,concordant_pairs=cp,all_pairs=ap,full_gold_target_retained=ret,
        abstention=bool(match) and not labels,duplicate=bool(labels) and len(labels)!=len(set(labels)),
        missing=[x for x in row['labels'] if x not in labels] if row['question_type']=='ordering' else [],
        reason=parsed.get('reason'),answer_errors=official[row['request_id']]['answer_errors'])
    records.append(out)
    for key in ('answer_valid','exact','prediction','gold','reference_compliant','pairwise_accuracy','normalized_lcs','concordant_pairs','all_pairs','full_gold_target_retained'):
        if not same(out[key],official[row['request_id']][key]):differences.append({'request_id':row['request_id'],'field':key,'independent':out[key],'official':official[row['request_id']][key]})
def mean(xs):return statistics.mean(xs) if xs else None
def aggregate(xs):
    orders=[x for x in xs if x['pairwise_accuracy'] is not None]
    targets=[x['full_gold_target_retained'] for x in xs if x['full_gold_target_retained'] is not None]
    return {'requests':len(xs),'exact_count':sum(x['exact'] for x in xs),'exact_accuracy':mean([int(x['exact']) for x in xs]),
        'valid':sum(x['answer_valid'] for x in xs),'invalid':sum(not x['answer_valid'] for x in xs),
        'pairwise_accuracy':mean([x['pairwise_accuracy'] for x in orders]),'normalized_lcs':mean([x['normalized_lcs'] for x in orders]),
        'partial_denominator':len(orders),'concordant_pairs_pooled':sum(x['concordant_pairs'] for x in orders),
        'all_pairs_pooled':sum(x['all_pairs'] for x in orders),
        'target_retained':sum(bool(x) for x in targets),'target_valid_denominator':len(targets),'target_retention':mean([int(x) for x in targets]),
        'reference_compliant':sum(x['reference_compliant'] for x in xs),
        'valid_answer_bad_reference':sum(x['answer_valid'] and not x['reference_compliant'] for x in xs)}
aggregates=[{'question_type':kind,'condition':cond,**aggregate([x for x in records if x['question_type']==kind and x['condition']==cond])}
    for kind in ('single_choice','ordering') for cond in sorted(set(x['condition'] for x in records))]
per_item=[{'candidate_id':cid,'question_type':next(x['question_type'] for x in records if x['candidate_id']==cid),'condition':cond,
           **aggregate([x for x in records if x['candidate_id']==cid and x['condition']==cond])}
    for cid in sorted(set(x['candidate_id'] for x in records)) for cond in ('FULL','NO_CONTEXT','ABLATE_G1','ABLATE_G2','ONLY_G1','ONLY_G2')]
for a in aggregates:
    official_summary=next(x for x in summary['summaries'] if x['question_type']==a['question_type'] and x['condition']==a['condition'])
    for k in ('requests','exact_count','exact_accuracy','pairwise_accuracy','normalized_lcs','partial_denominator','reference_compliant'):
        if not same(a[k],official_summary[k]):differences.append({'summary':(a['question_type'],a['condition']),'field':k})
    for ik,ok in [('valid','answer_valid'),('invalid','answer_invalid'),('target_retention','full_gold_target_retained')]:
        if not same(a[ik],official_summary[ok]):differences.append({'summary':(a['question_type'],a['condition']),'field':ok})
by={(x['candidate_id'],x['variant_id'],x['condition']):x for x in records}
paired=[]
for full in records:
    if full['condition']!='FULL':continue
    no=by[(full['candidate_id'],full['variant_id'],'NO_CONTEXT')]
    paired.append({'candidate_id':full['candidate_id'],'question_type':full['question_type'],'variant_id':full['variant_id'],
        'exact_full_minus_no':int(full['exact'])-int(no['exact']),
        'both_valid':full['answer_valid'] and no['answer_valid'],
        'pairwise_delta':full['pairwise_accuracy']-no['pairwise_accuracy'] if full['pairwise_accuracy'] is not None and no['pairwise_accuracy'] is not None else None,
        'lcs_delta':full['normalized_lcs']-no['normalized_lcs'] if full['normalized_lcs'] is not None and no['normalized_lcs'] is not None else None})
paired_summary=[]
for kind in ('single_choice','ordering'):
    xs=[x for x in paired if x['question_type']==kind];part=[x for x in xs if x['pairwise_delta'] is not None]
    paired_summary.append({'question_type':kind,'requests':len(xs),'exact_delta':mean([x['exact_full_minus_no'] for x in xs]),
        'pairwise_paired_delta':mean([x['pairwise_delta'] for x in part]),'lcs_paired_delta':mean([x['lcs_delta'] for x in part]),'partial_paired_denominator':len(part)})
invalid=[{k:x[k] for k in ('request_id','candidate_id','condition','prediction','missing','answer_errors')} for x in records if not x['answer_valid']]
valid_bad_refs=[{k:x[k] for k in ('request_id','exact','pairwise_accuracy','prediction')} for x in records if x['answer_valid'] and not x['reference_compliant']]
assert not differences, differences
findings=[
{'topic':'denominators','finding':'135/135 returned; 135 parsed direct JSON. Exact denominator includes all requests. All 21 invalid answers are incomplete SD-006 orders; no abstentions/duplicates. Ordering partial/target denominators exclude these responses, and differ across conditions.','caution':'Full ordering 7/9 valid versus No 8/9; 89.05% versus72.50% are valid-response macro means, not all-request scores or paired-complete effects. Paired partial Full-No is +10.476 points over7 complete pairs.'},
{'topic':'reference separation','finding':f'{len(valid_bad_refs)} answer-valid responses have noncompliant references and retain their independent answer scores. Reference failures never become answer failures in this recomputation.'},
{'topic':'no-context route priors','candidate_id':'SD-002','finding':'Full and No both exact0/3 and pairwise5/6 per response; LCS3/4. No-context explanations explicitly invoke usual route times/distances. High partial accuracy therefore cannot establish dialogue use. SD-002 Full arithmetic/reasons are also inconsistent with its chosen order.'},
{'topic':'no-context procedural shortcut','candidate_id':'SD-004','finding':'No-context exact1/3, pairwise5/6, LCS3/4; explanations place shopping last using question wording and routine sequence. Full exact2/3 is a one-response gain; deleting G2 preserves certified target3/3 and exact2/3. This is shortcut risk and lack of empirical dependency evidence, not a logical-bypass proof.'},
{'topic':'single-choice floor','finding':'SD-001 and SD-003 Full and No exact0/3 each. Full fails cannot support a useful ablation-success claim. Low No alone does not establish absence of shortcuts.'},
{'topic':'five-item incompleteness','candidate_id':'SD-006','finding':'Full only1/3 complete (0/3 exact); both deletion groups0/3 complete. Target retention for these deletions is null with denominator0, not0%. No-context2/3 complete. Report missing-item failures separately from source-dependent reasoning.'},
{'topic':'ablation interpretation','finding':'All altered-context values are FULL-Gold answer retention. Retained target under removed evidence can arise from guessing/semantics; falls can arise from incompleteness. Neither establishes independent logical necessity.'},
{'topic':'unit','finding':'2 admitted single-choice questions and3 ordering questions from3 selected roots; three permutations are repeated measures. No question-type superiority, training gain or general-quality conclusion is warranted.'}
]
result={'schema_version':'independent_single_order_result_check_v1','status':'PASS_EXACT_RECOMPUTATION','created_utc':datetime.now(timezone.utc).isoformat(),
    'method':'Reparse every actual raw JSON independently; reconstruct stable IDs from verified frozen presentation; count Gold-rank inversions and LIS instead of calling score_one/LCS; compare all records and summary fields.',
    'no_model_rerun':True,'frozen_inputs_modified':False,'old_questions_or_scores_read':False,
    'strict_raw_receipts':'PASS complete prior capture/model/submission/batch checker and returned batch-index equality',
    'input_bindings':[binding(E/x) for x in ('PREPARATION.json','PUBLIC_REQUESTS.json','PRIVATE_SCORING.json','SCORES.json','ANALYSIS_SUMMARY.json','RAW_RESPONSE_FREEZE.json')],
    'requests':len(records),'returned':freeze['returned'],'failure':freeze['failure'],'record_discrepancies':differences,
    'type_condition_recomputation':aggregates,'per_item_full_no_and_target':per_item,'paired_full_no':paired_summary,'paired_records':paired,
    'invalid_answer_count':len(invalid),'invalid_answers':invalid,'valid_answer_noncompliant_references':valid_bad_refs,
    'findings':findings,'records':records}
(P/'audit/INDEPENDENT_RESULT_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'requests':len(records),'discrepancies':len(differences),
'full_no':[a for a in aggregates if a['condition'] in ('FULL','NO_CONTEXT')],'paired':paired_summary,
'invalid':len(invalid),'valid_bad_refs':len(valid_bad_refs)},ensure_ascii=False,indent=2))
