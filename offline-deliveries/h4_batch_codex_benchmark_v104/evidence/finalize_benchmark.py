"""Validate frozen artifacts and record every outcome. No model or answer repair."""
import hashlib, itertools, json, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
PROJECT=BASE.parent.parent
ROOTS=('14304','20432','24719','38601')
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_sha(p): return hashlib.sha256(json.dumps(read(p),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def stamp(): return datetime.now(timezone.utc).isoformat()
def save(p,x):
    assert not p.exists(),str(p)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def norm(s): return re.sub(r'\s+','',s.casefold())
def error(out,code,**detail): out.append({'code':code,**detail})
def exact_citation(c,turns,issues,label):
    t=turns.get(c.get('turn_ref'))
    if not t or not c.get('quote') or c['quote'] not in t['text']:
        error(issues,'CITATION_NOT_EXACT',label=label,citation=c)
    elif c.get('speaker')!=t['speaker']:
        error(issues,'CITATION_SPEAKER_MISMATCH',label=label,citation=c)
def resolve(a,turns,positions):
    t=turns[a['turn_ref']];q=a['quote']
    assert q and t['text'].count(q)==1,'anchor must have one exact occurrence'
    start=t['text'].index(q)
    return (positions[a['turn_ref']],start,start+len(q))

# This is a concrete application of the existing v98/v103 single-event rule.
# It is separate from source entailment: both actions are supported, but two
# completed physical actions cannot be presented as one allowed ordering item.
protocol_file=PROJECT/'offline-deliveries/h4_batch_restore_v103/qa_engine/ordering_generation.py'
assert 'Select ONE event or speech act' in protocol_file.read_text()
protocol_failures={
 'v104-root24719-ordering01':[{'code':'ORDERING_MULTIPLE_EVENTS_IN_ONE_ITEM','label':'I5','proposition':'Emily fills the glass and drinks the water.','reason':'Filling the glass and drinking its water are two completed physical actions. Both are supported, but the existing writer contract requires one event or speech act per item. Neither source, item nor original Gold was repaired.','rule_source':str(protocol_file),'rule_sha256':sha(protocol_file),'rule_excerpt':'Select ONE event or speech act inside that complete turn, not a summary of every'}],
 'v104-root20432-ordering01':[{'code':'ORDERING_ACTOR_ATTRIBUTION_UNRESOLVED','label':'I4','proposition':'Liam reports having washed the tomatoes.','source_turn_ref':'R02:B:S01:T003','source_quote':"I've got the lettuce and tomatoes washed",'reason':'The source directly reports a prepared result, whereas reports having washed attributes the washing action to Liam. That personal-agent reading is natural in the meal-preparation context but is not explicitly fixed; under the existing whole-proposition and actor-preservation rule, the final reviewer conservatively withholds this candidate. The earlier PASS judgment and subsequent independent WARNING are preserved; the item and original Gold are not repaired.','rule_source':str(BASE/'evidence/QA_POLICY.md'),'rule_sha256':sha(BASE/'evidence/QA_POLICY.md'),'rule_excerpt':'supports：完整来源在题干范围内蕴含整句，包括人物'}]
}
save(BASE/'evidence/ORDERING_PROTOCOL_AUDIT.json',{'created_at_utc':stamp(),'rule_changed':False,'scope':'Final application of the existing one-event and whole-proposition actor-preservation rules. Original source-first reviews are preserved even when final adjudication is more conservative.','failures':protocol_failures})

reservations=read(BASE/'SOURCE_RESERVATIONS.json')['allocation']
duplicate=read(BASE/'evidence/HISTORICAL_DUPLICATE_AUDIT.json')
outcomes=[];root_stats=[];selected_public=[];selected_answers=[];controls_public=[];controls_answers=[]
all_public=[];all_answers=[];full_text=[];new_text=[]
for short in ROOTS:
    folder=BASE/'roots'/('root'+short)
    source_path=folder/'CUMULATIVE_SOURCE.json'
    source=read(source_path);turns={t['turn_ref']:t for t in source['turns']}
    positions={t['turn_ref']:i for i,t in enumerate(source['turns'])}
    public=read(folder/'PUBLIC_QUESTIONS.json')['questions']
    judgments=read(folder/'QUESTION_SOURCE_JUDGMENTS.json')
    goldfile=read(folder/'PRIVATE_GOLD.json')
    golds={x['candidate_id']:x for x in goldfile['answers']}
    designs={x['candidate_id']:x for x in read(folder/'PRIVATE_DESIGN.json')['design']}
    reviews={x['candidate_id']:x for x in judgments['questions']}
    source_review=read(folder/'SOURCE_REVIEW.json')
    generation=read(folder/'PUBLIC_GENERATION_RECEIPT.json')
    reservation=next(x for x in reservations if x['root_id']==source['root_id'])
    baseline_path=Path(reservation['baseline_path']);baseline=read(baseline_path)
    shared=[]
    # setup_benchmark.py freezes baseline file bytes but event JSON canonical
    # content. Preserve those definitions instead of comparing unlike encodings.
    if sha(baseline_path)!=reservation['baseline_sha256'] or canonical_sha(Path(reservation['source_item_path']))!=reservation['source_item_sha256']:
        error(shared,'INPUT_RESERVATION_HASH_MISMATCH')
    if source['turns'][:len(baseline['turns'])]!=baseline['turns']:
        error(shared,'HISTORICAL_PREFIX_CHANGED')
    if len(turns)!=len(source['turns']): error(shared,'SOURCE_TURN_REFS_NOT_UNIQUE')
    if source_review.get('decision')!='READY' or source_review.get('full_source_read') is not True or source_review.get('source_sha256')!=sha(source_path):
        error(shared,'SOURCE_NOT_REVIEWED_READY')
    if not {'time','location','actor_identity','plan_vs_completion','cross_scene_continuity','semantic_repetition'}.issubset(source_review.get('checks',{})):
        error(shared,'SOURCE_REVIEW_CHECKS_INCOMPLETE')
    if judgments.get('source_sha256')!=sha(source_path) or judgments.get('public_sha256')!=sha(folder/'PUBLIC_QUESTIONS.json'):
        error(shared,'QUESTION_REVIEW_HASH_MISMATCH')
    if judgments.get('full_source_read') is not True or judgments.get('turn_refs_read')!=[t['turn_ref'] for t in source['turns']]:
        error(shared,'QUESTION_REVIEW_READ_COVERAGE_INCOMPLETE')
    if judgments.get('private_answers_read_before_save') is not False:
        error(shared,'SOURCE_FIRST_SEQUENCE_NOT_ATTESTED')
    if goldfile.get('source_first_judgments_sha256')!=sha(folder/'QUESTION_SOURCE_JUDGMENTS.json') or goldfile.get('public_sha256')!=sha(folder/'PUBLIC_QUESTIONS.json'):
        error(shared,'PRIVATE_GOLD_REVIEW_BINDING_MISMATCH')
    ids={q['candidate_id'] for q in public}
    if len(ids)!=len(public) or ids!=set(reviews) or ids!=set(golds) or ids!=set(designs):
        error(shared,'CANDIDATE_SET_NOT_COMPLETE')
    if generation['mechanical_errors']: error(shared,'GENERATOR_MECHANICAL_ERRORS',details=generation['mechanical_errors'])
    root_results=[]
    seen=set()
    for q in public:
        cid=q['candidate_id'];r=reviews[cid];g=golds[cid];design=designs[cid]
        issues=list(shared)
        if q['root_id']!=source['root_id'] or q['source_sha256']!=sha(source_path) or q['source_path']!=str(source_path):
            error(issues,'PUBLIC_SOURCE_BINDING_MISMATCH')
        allowed={'candidate_id','root_id','question_type','question','source_path','source_sha256','options' if q['question_type']=='multi_select' else 'items'}
        if q['question_type']=='ordering': allowed.add('chronology_kind')
        if set(q)!=allowed:error(issues,'PUBLIC_FIELDS_INVALID',fields=sorted(q))
        if r.get('stem_valid') is not True or r.get('scope_valid') is not True:error(issues,'STEM_OR_SCOPE_INVALID')
        content=q.get('options',q.get('items'))
        if len(content)!=6 or any(not isinstance(v,str) or not v.strip() for v in content.values()): error(issues,'SIX_NONEMPTY_ITEMS_REQUIRED')
        if len({norm(v) for v in content.values()})!=6:error(issues,'DUPLICATED_PROPOSITION')
        signature=(q['question_type'],tuple(sorted(norm(v) for v in content.values())))
        if signature in seen:error(issues,'DUPLICATE_CURRENT_ROOT_QUESTION')
        seen.add(signature)
        if re.search(r'\b(?:he|she|him|her|his|hers|himself|herself)\b',' '.join(content.values()),re.I):error(issues,'FORBIDDEN_GENDER_PRONOUN')
        supported=[];unknown=[];combination_orders=[]
        if q['question_type']=='multi_select':
            if set(content)!=set('ABCDEF') or set(r.get('options',{}))!=set('ABCDEF'):error(issues,'MULTISELECT_LABEL_SET_INVALID')
            for label,text in content.items():
                j=r['options'][label]
                if j.get('proposition')!=text:error(issues,'REVIEW_PROPOSITION_MISMATCH',label=label)
                if j.get('scope_valid') is not True or j.get('fact_distinct') is not True:error(issues,'OPTION_SCOPE_OR_DISTINCTNESS_FAIL',label=label)
                relation=j.get('relation')
                if relation=='supports':supported.append(label)
                elif relation=='unknown':
                    unknown.append(label);error(issues,'UNRESOLVED_OPTION',label=label,proposition=text,reason=j.get('reason'))
                elif relation!='contradicts':error(issues,'INVALID_OPTION_RELATION',label=label)
                if not j.get('citations') or not j.get('reason'):error(issues,'OPTION_EVIDENCE_MISSING',label=label)
                for c in j.get('citations',[]):exact_citation(c,turns,issues,label)
            if r.get('duplicate_or_entangled_options'):error(issues,'ENTANGLED_OPTIONS',details=r['duplicate_or_entangled_options'])
            if supported!=g['gold']:error(issues,'PRIVATE_GOLD_DISAGREES_WITH_SOURCE',derived=supported,original_gold=g['gold'])
            if len(g['gold'])!=design['intended_cardinality']:error(issues,'INTENDED_CARDINALITY_MISMATCH')
        else:
            expected={f'I{i}' for i in range(1,7)}
            if set(content)!=expected or set(r.get('items',{}))!=expected:error(issues,'ORDERING_LABEL_SET_INVALID')
            for field in ['coherent','unique_order','neutral_stem_ok']:
                if r.get(field) is not True:error(issues,'ORDERING_QUESTION_BOOLEAN_FAILED',field=field)
            stem_match=re.search(r'APPEAR IN THE SUPPLIED DIALOGUE within the (\S+) scene\.$',q['question'])
            scope=stem_match.group(1) if stem_match else None
            if scope!='R02:B:S01':error(issues,'ORDERING_PUBLIC_SCOPE_INVALID')
            choices=[]
            for label,text in content.items():
                j=r['items'][label]
                if j.get('proposition')!=text:error(issues,'REVIEW_PROPOSITION_MISMATCH',label=label)
                for field in ['supported','actor_ok','modality_ok','state_ok','neutral_stem_ok']:
                    if j.get(field) is not True:error(issues,'ORDERING_ITEM_BOOLEAN_FAILED',label=label,field=field)
                if not j.get('support_refs') or any(ref not in turns for ref in j.get('support_refs',[])):error(issues,'ORDERING_SUPPORT_REFS_INVALID',label=label)
                primary=j.get('order_anchor',{})
                if primary.get('turn_ref')!=g['evidence'][label]['turn_ref']:error(issues,'INDEPENDENT_ANCHOR_OUTSIDE_FROZEN_TURN',label=label)
                candidates=[primary]+j.get('alternative_matching_anchors',[])
                resolved=[]
                for a in candidates:
                    try:
                        z=resolve(a,turns,positions)
                        if turns[a['turn_ref']]['scene_ref']!=scope:raise AssertionError('anchor outside public scope')
                        resolved.append((label,z))
                    except (KeyError,AssertionError,ValueError) as exc:error(issues,'ORDERING_ANCHOR_INVALID',label=label,detail=str(exc))
                choices.append(resolved)
            combinations=list(itertools.product(*choices))
            for combo in combinations:
                ordered=sorted(combo,key=lambda pair:pair[1][:2])
                overlap=any(a[1][0]==b[1][0] and a[1][2]>b[1][1] for a,b in zip(ordered,ordered[1:]))
                if overlap:error(issues,'ORDERING_ANCHORS_OVERLAP')
                combination_orders.append(tuple(x[0] for x in ordered))
            unique=set(combination_orders)
            if len(unique)!=1:error(issues,'ORDERING_NOT_UNIQUE',orders=[list(o) for o in sorted(unique)])
            derived=list(combination_orders[0]) if combination_orders else None
            if derived!=r.get('derived_order') or derived!=g['gold']:error(issues,'ORDERING_PRIVATE_GOLD_DISAGREEMENT',derived=derived,review_order=r.get('derived_order'),original_gold=g['gold'])
            if r.get('alternative_orders') and any(list(o)!=derived for o in r['alternative_orders']):error(issues,'REVIEW_REPORTS_ALTERNATIVE_ORDER')
            issues.extend(protocol_failures.get(cid,[]))
        # The historical audit reports exact duplicate matches. No semantic novelty
        # certification is inferred from a lack of exact matches.
        for match in duplicate.get('current_vs_historical_exact_duplicates',[]):
            if match.get('candidate_id')==cid:error(issues,'HISTORICAL_EXACT_DUPLICATE',match=match)
        status='PASS' if not issues else 'FAIL'
        out={'candidate_id':cid,'root_id':source['root_id'],'question_type':q['question_type'],'role':g['role'],'intended_gold_count':len(g['gold']) if q['question_type']=='multi_select' else None,'status':status,'issues':issues,'original_gold':g['gold'],'independent_supported_labels':supported if q['question_type']=='multi_select' else None,'unknown_labels':unknown,'enumerated_ordering_anchor_combinations':len(combination_orders) if q['question_type']=='ordering' else None,'public_path':str(folder/'PUBLIC_QUESTIONS.json'),'private_gold_path':str(folder/'PRIVATE_GOLD.json'),'source_review_path':str(folder/'SOURCE_REVIEW.json'),'question_review_path':str(folder/'QUESTION_SOURCE_JUDGMENTS.json'),'training_eligible':False}
        root_results.append(out);outcomes.append(out);all_public.append(q);all_answers.append(g)
        if status=='PASS':
            if g['role']=='control6':controls_public.append(q);controls_answers.append(g)
            else:selected_public.append(q);selected_answers.append(g)
    stat={'root_id':source['root_id'],'source_id':reservation['source_id'],'baseline_turns':len(baseline['turns']),'new_turns':len(source['turns'])-len(baseline['turns']),'cumulative_turns':len(source['turns']),'source_decision':source_review['decision'],'historical_prefix_exact':source['turns'][:len(baseline['turns'])]==baseline['turns'],'candidate_count':len(root_results),'passed_candidates':sum(x['status']=='PASS' for x in root_results),'failed_candidates':sum(x['status']=='FAIL' for x in root_results),'new_multiselect_by_gold':{str(k):sum(x['status']=='PASS' and x['role']=='new_child' and x['intended_gold_count']==k for x in root_results) for k in [2,3,4,5]},'gold6_controls':sum(x['status']=='PASS' and x['role']=='control6' for x in root_results),'ordering_passed':sum(x['status']=='PASS' and x['question_type']=='ordering' for x in root_results),'ordering_failed':sum(x['status']=='FAIL' and x['question_type']=='ordering' for x in root_results)}
    save(folder/'FINAL_CROSSCHECK.json',{'created_at_utc':stamp(),'shared_issues':shared,'source_sha256':sha(source_path),'public_sha256':sha(folder/'PUBLIC_QUESTIONS.json'),'judgments_sha256':sha(folder/'QUESTION_SOURCE_JUDGMENTS.json'),'private_gold_sha256':sha(folder/'PRIVATE_GOLD.json'),'summary':stat,'outcomes':root_results})
    root_stats.append(stat)
    full_text.append('\n=== '+source['root_id']+' ===\n')
    new_text.append('\n=== '+source['root_id']+' / NEW SUFFIX ===\n')
    for i,t in enumerate(source['turns']):
        line=f"[{t['turn_ref']}] {t['speaker']}: {t['text']}\n"
        full_text.append(line)
        if i>=len(baseline['turns']):new_text.append(line)

before=read(BASE/'evidence/HISTORICAL_PRESERVATION_BEFORE.json')
changes=[]
for entry in before['files']:
    p=Path(entry['path'])
    if not p.exists() or sha(p)!=entry['sha256']:changes.append({'path':str(p),'expected':entry['sha256'],'actual':sha(p) if p.exists() else None})
save(BASE/'evidence/HISTORICAL_PRESERVATION_AFTER.json',{'checked_at_utc':stamp(),'files_checked':len(before['files']),'changes':changes,'all_tracked_v103_bytes_unchanged':not changes})
assert not changes,'Do not deliver a run that mutated v103'
start=read(BASE/'BENCHMARK_STARTED.json')
finish=stamp();elapsed=(datetime.fromisoformat(finish)-datetime.fromisoformat(start['started_at_utc'])).total_seconds()
accounting={'schema_version':'v104_codex_direct_accounting_v1','started_at_utc':start['started_at_utc'],'content_crosscheck_completed_at_utc':finish,'wall_seconds_through_content_crosscheck':elapsed,'authoring_window':{'first_precise_author_start':'2026-10-15T14:00:28+00:00','last_author_finished':'2026-10-15T14:13:28+00:00','parallel_wall_seconds':780,'limitations':'Includes reading/tool overhead and compaction; primary root38601 author start is minute-estimated. This window is not pure inference latency.'},'candidate_count':len(outcomes),'content_passed_count':sum(x['status']=='PASS' for x in outcomes),'content_failed_count':sum(x['status']=='FAIL' for x in outcomes),'new_multiselect_candidates':16,'new_multiselect_passed':sum(x['status']=='PASS' and x['role']=='new_child' for x in outcomes),'new_multiselect_gold_distribution':{str(k):sum(x['status']=='PASS' and x['role']=='new_child' and x['intended_gold_count']==k for x in outcomes) for k in [2,3,4,5]},'gold6_controls_passed':len(controls_public),'ordering_candidates':8,'ordering_passed':sum(x['status']=='PASS' and x['question_type']=='ordering' for x in outcomes),'new_content_candidates_count':len(selected_public),'roots':root_stats,'failure_code_counts':dict(Counter(i['code'] for x in outcomes for i in x['issues'])),'new_turns_total':sum(r['new_turns'] for r in root_stats),'source_ready_roots':sum(r['source_decision']=='READY' for r in root_stats),'model_backend':start['author_backend'],'qwen_http_calls':0,'codex_cli_used':False,'production_transport_equivalent':False,'exact_model_id':start['exact_model_id'],'exact_codex_usage_tokens':None,'exact_codex_cost':None,'usage_limitation':start['usage_limitation'],'solver_run':False,'training_run':False,'training_eligible':False,'source_selection':'Four exposed roots, compatible events selected purposively; not random holdout.','same_model_review':True,'writer_repairs':0,'source_substitutions':0,'historical_semantic_deduplication_certified':False,'review_recovery_receipt':str(BASE/'evidence/REVIEW_RECOVERY_RECEIPT.json')}
save(BASE/'FINAL_BATCH_ACCOUNTING.json',accounting)
save(BASE/'ALL_OUTCOMES_INDEX.json',{'created_at_utc':stamp(),'all_candidates':outcomes,'note':'All original public questions, author drafts, answers, and failures are retained. FAIL candidates are not admitted.'})
save(BASE/'SELECTED_PUBLIC_QUESTIONS.json',{'questions':selected_public,'status':'CONTENT_REVIEWED_CANDIDATES_NOT_TRAINING_APPROVED'})
save(BASE/'SELECTED_PRIVATE_GOLD.json',{'answers':selected_answers})
save(BASE/'CONTROL_PUBLIC_QUESTIONS.json',{'questions':controls_public})
save(BASE/'CONTROL_PRIVATE_GOLD.json',{'answers':controls_answers})
save(BASE/'NEW_CONTENT_CANDIDATES.json',{'status':'CONTENT_REVIEWED_NOT_SOLVER_OR_TRAINING_APPROVED','questions':selected_public,'private_answers':selected_answers,'same_root_variants_are_correlated':True})
save(BASE/'REJECTED_CANDIDATES.json',{'candidates':[{'public_question':next(q for q in all_public if q['candidate_id']==o['candidate_id']),'original_answer':next(g for g in all_answers if g['candidate_id']==o['candidate_id']),'outcome':o} for o in outcomes if o['status']=='FAIL']})
for name,lines in [('CUMULATIVE_DIALOGUES.txt',full_text),('NEW_DIALOGUES.txt',new_text)]:
    path=BASE/name;assert not path.exists();path.write_text(''.join(lines))
print(json.dumps(accounting,ensure_ascii=False,indent=2))
