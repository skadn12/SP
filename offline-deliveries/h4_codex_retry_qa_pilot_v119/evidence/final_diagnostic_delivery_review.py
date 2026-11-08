#!/usr/bin/env python3
"""Independent, no-inference delivery audit. Does not import report or solver code."""
import collections, hashlib, json, re, sys
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
B=Path(__file__).resolve().parents[1]
D=B/'delivery/diagnostic_v2'
errors=[]
def check(ok,label):
    if not ok: errors.append(label)
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(actual,expected,label):
    if actual != expected:errors.append({'check':label,'actual':actual,'expected':expected})
def ratio(n,d):
    if not d:return None
    x=F(n,d);return {'fraction':str(x),'value':float(x)}
def mean(xs):return sum(xs,F())/len(xs)
r=load(D/'RETRY_DIAGNOSTIC_REPORT.json'); md=(D/'RETRY_DIAGNOSTIC_REPORT.md').read_text()
inputs={}
def bind(p):
    p=Path(p).resolve();inputs[str(p)]=sha(p)
    return {'path':str(p),'sha256':inputs[str(p)]}
for p in D.rglob('*'):
    if p.is_file():bind(p)
for label,doc in [('report',r),('independent_solver_audit',load(B/'evaluations/diagnostic/INDEPENDENT_SOLVER_AUDIT.json'))]:
    for b in doc['input_bindings']:
        p=Path(b['path']);same(sha(p),b['sha256'],label+':binding:'+str(p));bind(p)
external=load(B/'evaluations/diagnostic/INDEPENDENT_SOLVER_AUDIT.json')
bind(B/'evaluations/diagnostic/INDEPENDENT_SOLVER_AUDIT.json')
ix=load(r['v118_delivery_index_binding']['path']);same(sha(r['v118_delivery_index_binding']['path']),r['v118_delivery_index_binding']['sha256'],'v118 original index binding')
original_roots={x['root_id']:x for x in ix['roots']}
same(external['status'],'PASS','independent audit status');same(external['errors'],[],'independent audit errors')
same(external['training_started'],False,'independent audit no training')
same(external['same_inherited_model_external_certification'],False,'independent audit identity limitation')
for k,v in {'report_executor':'/root','report_tool_author':'/root/final_public_review','evaluation_scope':'CONTENT_ONLY_DIAGNOSTIC','batch_status':'CONTENT_AUDITED_PROCEDURE_HOLD','batch_procedural_success':False,'training_pool_eligible':False,'training_started':False,'training_pool_merged':False,'native_selection_confers_training_eligibility':False,'attempted_roots':3,'procedural_success_roots':0,'training_eligible_roots':0,'training_eligible_questions':0,'content_nonempty_roots':3,'read_only_inputs':True,'original_files_overwritten':False}.items():same(r[k],v,'report:'+k)
all_public={};pass_ids=set();root_details={}
for root in r['roots']:
    short=root['root_id'].split('-')[-1];p=B/f'units/root{short}/roots/root{short}'
    src=load(p/'CUMULATIVE_SOURCE.json');pub=load(p/'PUBLIC_QUESTIONS.json')['questions'];gold=load(p/'PRIVATE_GOLD.json')['questions'];cross=load(p/'FINAL_CROSSCHECK.json')['results'];comp=load(p/'COMPOSITION_RECEIPT.json');disp=load(p/'DIAGNOSTIC_CONTENT_DISPOSITION.json')
    for fn in ['CUMULATIVE_SOURCE.json','PUBLIC_QUESTIONS.json','PRIVATE_GOLD.json','FINAL_CROSSCHECK.json','COMPOSITION_RECEIPT.json','DIAGNOSTIC_CONTENT_DISPOSITION.json','FINAL_BLIND_SOURCE_REVIEW.json']:bind(p/fn)
    gm={q['candidate_id']:q for q in gold};cm={q['candidate_id']:q for q in cross}
    same(set(gm),set(cm),'gold/crosscheck question set:'+short)
    same(set(cm),{q['candidate_id'] for q in pub},'public/crosscheck question set:'+short)
    for q in pub:
        cid=q['candidate_id'];x=cm[cid];g=gm[cid];all_public[cid]={'question':q,'gold':g,'cross':x,'root':root['root_id']}
        meta=root['metadata'][cid]
        for k in ['question_type','root_id']:same(meta[k],q[k],'metadata:'+cid+':'+k)
        for k in ['status','issues','role']:same(meta[k],x[k],'metadata:'+cid+':'+k)
        same(meta['gold'],g['gold'],'metadata gold:'+cid);same(meta['gold_cardinality'],len(g['gold']) if q['question_type']=='multi_select' else None,'metadata gold count:'+cid)
        same(meta['training_pool_eligible'],False,'metadata eligibility:'+cid)
        if x['status']=='CONTENT_PASS':pass_ids.add(cid)
    passes=[q for q in pub if cm[q['candidate_id']]['status']=='CONTENT_PASS']
    ms=[q for q in passes if q['question_type']=='multi_select'];orders=[q for q in passes if q['question_type']=='ordering']
    counts={'planned_question_slots':7,'public_composed':len(pub),'composition_unfilled':len(comp['unfilled_slots']),'final_content_excluded':len(pub)-len(passes),'content_pass':len(passes),'pass_multi_select':len(ms),'pass_ordering':len(orders),'pass_multiselect_gold_counts':dict(sorted(collections.Counter(str(len(gm[q['candidate_id']]['gold'])) for q in ms).items()))}
    same(root['counts'],counts,'root counts:'+short)
    same(sorted(root['content_pass_candidate_ids']),sorted(q['candidate_id'] for q in passes),'root pass ids:'+short)
    same(root['composition_unfilled_slots'],comp['unfilled_slots'],'unfilled slots:'+short)
    same(root['procedural_success'],False,'root procedural hold:'+short);same(root['training_pool_eligible'],False,'root training hold:'+short)
    same(root['source_reused_byte_identically_from_v118'],True,'root source reuse:'+short);same(root['new_dialogue_expansion_performed_in_v119'],False,'root no expansion:'+short)
    same(root['final_excluded_candidates'],[root['metadata'][q['candidate_id']] for q in pub if cm[q['candidate_id']]['status']=='EXCLUDED'],'root excluded metadata:'+short)
    for k,v in root['v118_original_delivery_state'].items():same(v,original_roots[root['root_id']][k],'retained v118 state:'+short+':'+k)
    same(disp['status'],'CONTENT_AUDITED_PROCEDURE_HOLD','disposition:'+short)
    turns=src['turns'];r02=[t for t in turns if (t.get('scene_ref') or '').startswith('R02:')]
    scenes=dict(sorted(collections.Counter(t['scene_ref'] for t in r02).items()))
    for k,v in {'turns':len(turns),'r02_turns':len(r02),'r02_scene_turn_counts':scenes,'dialogue_token_count':None}.items():same(root['dialogue'][k],v,'dialogue:'+short+':'+k)
    old=B.parent/'h4_codex_fullscale_ab_v118'/f'units/root{short}/roots/root{short}/CUMULATIVE_SOURCE.json'
    same(sha(p/'CUMULATIVE_SOURCE.json'),sha(old),'source unchanged v118:'+short)
    root_details[root['root_id']]={'dialogue_turns':len(turns),'r02_turns':len(r02),'counts':counts,'procedure_status':disp['status'],'training_pool_eligible':False}
tot={'planned_question_slots':21,'public_composed':len(all_public),'composition_unfilled':sum(x['counts']['composition_unfilled'] for x in root_details.values()),'final_content_excluded':len(all_public)-len(pass_ids),'content_pass':len(pass_ids),'pass_multi_select':sum(all_public[c]['question']['question_type']=='multi_select' for c in pass_ids),'pass_ordering':sum(all_public[c]['question']['question_type']=='ordering' for c in pass_ids),'pass_multiselect_gold_counts':dict(sorted(collections.Counter(str(len(all_public[c]['gold']['gold'])) for c in pass_ids if all_public[c]['question']['question_type']=='multi_select').items()))}
same(r['totals'],tot,'batch totals');same(tot,{'planned_question_slots':21,'public_composed':19,'composition_unfilled':2,'final_content_excluded':2,'content_pass':17,'pass_multi_select':15,'pass_ordering':2,'pass_multiselect_gold_counts':{'2':3,'3':3,'4':3,'5':3,'6':3}},'required design counts')
group_names=['all','multi_select','multi_select_gold2_to_5','multi_select_control6','ordering']+['multi_select_gold'+str(i) for i in range(2,7)]
def groups(row):
    gs=['all',row['type']]
    if row['type']=='multi_select':gs.extend(['multi_select_gold'+str(row['gold_n']),'multi_select_control6' if row['gold_n']==6 else 'multi_select_gold2_to_5'])
    return gs
def metrics(rows):
    ms=[x for x in rows if x['type']=='multi_select'];o=[x for x in rows if x['type']=='ordering'];cs=sorted({x['cid'] for x in rows})
    ex=sum(x['exact'] for x in rows);n=len(rows);sump=sum((x['partial'] for x in rows),F())
    fp_calls=sum(x['fp']>0 for x in ms);over=sum(x['pred_n']>x['gold_n'] for x in ms);alls=sum(x['pred_n']==6 for x in ms)
    return {'calls':n,'candidates':len(cs),'exact_calls':ex,'exact_rate':ratio(ex,n),'partial_mean':ratio(sump,n),'option_confusion_counts':{'calls':len(ms),'tp':sum(x['tp'] for x in ms),'fp':sum(x['fp'] for x in ms),'fn':sum(x['fn'] for x in ms),'all_select_calls':alls,'all_select_rate':ratio(alls,len(ms)),'false_positive_calls':fp_calls,'false_positive_call_rate':ratio(fp_calls,len(ms)),'cardinality_overselect_calls':over,'cardinality_overselect_rate':ratio(over,len(ms)),'cardinality_underselect_calls':sum(x['pred_n']<x['gold_n'] for x in ms),'predicted_cardinality_counts':dict(sorted(collections.Counter(str(x['pred_n']) for x in ms).items()))},'ordering_pair_confusion_counts':{'calls':len(o),'correct_pairs':sum(x['tp'] for x in o),'inverted_pairs':sum(x['fp'] for x in o)},'candidate_macro_exact_rate':ratio(sum((F(sum(x['exact'] for x in rows if x['cid']==c),sum(x['cid']==c for x in rows)) for c in cs),F()),len(cs)),'candidate_macro_partial_mean':ratio(sum((mean([x['partial'] for x in rows if x['cid']==c]) for c in cs),F()),len(cs))}
def grouped(rows):return {g:metrics([x for x in rows if g in groups(x)]) for g in group_names}
side_results={};manifests={};golds={};computed={};selection={}
for side in ['base','trained']:
    p=B/'evaluations/diagnostic/run'/side;m=load(p/'manifest.json');g=load(p/'gold.json');bindings=load(p/'canonical_bindings.json')['candidates'];native=load(p/'RESULTS.json');sr=r['solvers'][side]
    manifests[side]=m;golds[side]=g;reqs={x['request_id']:x for x in m['requests']};files=sorted((p/'responses').glob('*.json'));rows=[]
    same(len(reqs),294,'request count:'+side);same(len(files),294,'response count:'+side);same(set(g),pass_ids,'evaluation only pass ids:'+side)
    same(m['random_display'],{'version':'independent_seeded_permutation_v2','seed':2026100800,'gold_blind':True,'cyclic_rebinding':False,'shape_filter':False},'random display:'+side)
    same(native['call_set_errors'],[],'native call set:'+side);same(native['ground_calls'],0,'no ground:'+side)
    for cb in bindings:
        cid=cb['candidate_id'];labels=all_public[cid]['gold']['gold'];expected=[cb['content_id_by_label'][x] for x in labels]
        same(g[cid],expected,'evaluation gold from saved gold:'+side+':'+cid)
    seen=[]
    for f in files:
        a=load(f);rid=a['request_id'];seen.append(rid);q=reqs[rid];cid=q['candidate_id'];gold=g[cid];qt=q['question_type'];sc=a['score'];label=side+':'+rid
        for k in ['candidate_id','root_id','question_type','mode','slot','seed']:same(a[k],q[k],label+':'+k)
        same(a['variant_id'],q['variant']['variant_id'],label+':variant');same(a['solver_model'],m['solver_model'],label+':model');same(a['model_path'],m['model'],label+':path')
        same(a['evaluation_valid'],True,label+':valid');same(a['data_errors'],[],label+':errors')
        for k,v in {'sampling_seed':q['seed'],'prompt_token_count':q['prompt_tokens'],'temperature':.8,'top_p':.9,'max_tokens':32,'structured_output_version':'finite_choice_set_v1','structured_output_api':'structured_outputs=StructuredOutputsParams(choice)','finish_reason':'stop'}.items():same(a['metadata'][k],v,label+':metadata:'+k)
        match=re.fullmatch(r'\s*<box>([^<>]+)</box>\s*',a['raw_output']);check(match is not None,label+':single_box')
        labels=[x.strip() for x in match.group(1).split(',')];mapping=q['variant']['display_to_content'];check(len(labels)==len(set(labels)) and all(x in mapping for x in labels),label+':unique_valid_labels')
        pred=[mapping[x] for x in labels]
        if qt=='multi_select':
            tp=len(set(pred)&set(gold));fp=len(set(pred)-set(gold));fn=len(set(gold)-set(pred));exact=set(pred)==set(gold);partial=F(tp,len(gold))-(F(fp,6-len(gold)) if len(gold)<6 else F());stored_pred=sorted(pred)
        else:
            check(len(pred)==6 and set(pred)==set(gold),label+':complete_order');positions={x:i for i,x in enumerate(gold)};fp=sum(positions[pred[i]]>positions[pred[j]] for i in range(6) for j in range(i+1,6));tp=15-fp;fn=0;exact=pred==gold;partial=F(tp-fp,15);stored_pred=pred
        expected_score={'parse_status':'success','parse_reason':None,'scored':True,'display_prediction':labels,'content_prediction':stored_pred,'content_gold':gold,'tp':tp,'fp':fp,'fn':fn,'exact_match':exact,'partial_score':str(partial),'scoring_status':'SCORED','partial_score_fraction':str(partial)}
        same(sc,expected_score,label+':complete score')
        rows.append({'cid':cid,'root':q['root_id'],'type':qt,'gold_n':len(gold),'pred_n':len(pred),'exact':exact,'partial':partial,'tp':tp,'fp':fp,'fn':fn,'variant':a['variant_id'],'slot':a['slot']})
    same(set(seen),set(reqs),'complete request coverage:'+side);same(len(seen),len(set(seen)),'no duplicate response:'+side)
    gs=grouped(rows);same(sr['grouped_metrics'],gs,'all grouped metrics:'+side);same(sr['calls'],294,'solver calls:'+side)
    nmap={q['candidate_id']:q for q in native['questions']};selected=[]
    for cid in sorted(pass_ids):
        sub=[x for x in rows if x['cid']==cid];qt=sub[0]['type'];nper,nslot=(6,3) if qt=='multi_select' else (12,1)
        same({(x['variant'],x['slot']) for x in sub},{(f'perm_{i:02d}',j) for i in range(nper) for j in range(nslot)},'native slots:'+side+':'+cid)
        pm={f'perm_{i:02d}':mean([x['partial'] for x in sub if x['variant']==f'perm_{i:02d}']) for i in range(nper)};mmean=mean(list(pm.values()));admit=0<mmean<1
        reason='FULL_PARTIAL_IN_OPEN_INTERVAL' if admit else ('FULL_PARTIAL_PERFECT_NO_NEW_HARD_ITEM' if mmean==1 else 'FULL_PARTIAL_NONPOSITIVE')
        expected={'policy_version':'r02_groundless_full_only_random_position_v2','ground_used':False,'selected_by_policy':admit,'selection_reason':reason,'full_partial_mean':str(mmean),'permutation_means':{k:str(v) for k,v in pm.items()},'permutation_score_min':str(min(pm.values())),'permutation_score_max':str(max(pm.values())),'permutation_score_range':str(max(pm.values())-min(pm.values())),'full_exact_rate':str(F(sum(x['exact'] for x in sub),len(sub))),'context_gain':None,'data_errors':[],'candidate_id':cid,'root_id':sub[0]['root'],'question_type':qt,'expected_full_calls':nper*nslot,'actual_full_calls':len(sub)}
        same(nmap[cid],expected,'native selection:'+side+':'+cid);same(sr['by_candidate'][cid],{'metrics':metrics(sub),'native_policy':expected},'report candidate:'+side+':'+cid)
        if admit:selected.append(cid)
    same(sorted(native['selected_questions']),selected,'native selected ids:'+side)
    for root in r['roots']:
        root_id=root['root_id'];rg=grouped([x for x in rows if x['root']==root_id]);same(sr['by_root'][root_id],rg,'solver root groups:'+side+':'+root_id);same(root['paired_solver_by_root'][side],rg,'root paired groups:'+side+':'+root_id)
        root_details[root_id].setdefault('paired_metrics',{})[side]=rg
    same(external['results'][side],{'calls':294,'candidates':17,'errors':0},'independent audit counts:'+side)
    computed[side]=gs;selection[side]=selected;side_results[side]={'calls':len(rows),'candidates':len(pass_ids),'grouped_metrics':gs,'native_selected_candidate_ids':selected}
same(manifests['base']['requests'],manifests['trained']['requests'],'paired immutable requests/prompts/seeds/permutations');same(golds['base'],golds['trained'],'paired immutable gold')
for group in group_names:
    for k,field in [('exact_trained_minus_base','exact_rate'),('partial_trained_minus_base','partial_mean')]:
        d=F(computed['trained'][group][field]['fraction'])-F(computed['base'][group][field]['fraction']);same(r['paired_trained_minus_base'][group][k],ratio(d,1),'paired delta:'+group+':'+k)
sel=r['trained_policy_diagnostic_selection'];controls=sorted(c for c in selection['trained'] if all_public[c]['gold']['role']=='control6');noncontrols=sorted(set(selection['trained'])-set(controls))
same(sel['control6_candidate_ids'],controls,'trained control ids');same(sel['non_control_candidate_ids'],noncontrols,'trained noncontrol ids');same(sel['control6_count'],3,'trained control count');same(sel['non_control_count'],13,'trained noncontrol count');same(sel['training_recommendation'],False,'native no recommendation');same(sel['training_pool_eligible'],False,'native no eligibility')
same(sel['non_control_type_counts'],dict(collections.Counter(all_public[c]['question']['question_type'] for c in noncontrols)),'selected noncontrol type counts')
same(sel['non_control_ms_gold_counts'],dict(collections.Counter(str(len(all_public[c]['gold']['gold'])) for c in noncontrols if all_public[c]['question']['question_type']=='multi_select')),'selected noncontrol gold counts')
for root in r['roots']:
    root_id=root['root_id'];same(root['trained_policy_non_control_selected_ids'],[c for c in noncontrols if all_public[c]['root']==root_id],'root noncontrol selection:'+root_id);same(root['trained_policy_control6_selected_ids'],[c for c in controls if all_public[c]['root']==root_id],'root control selection:'+root_id)
same(r['always_all_select_multiselect_baseline'],{'exact_rate':ratio(1,5),'partial_mean':ratio(1,5),'gold2_to_5_exact_and_partial':ratio(0,1),'control6_exact_and_partial':ratio(1,1)},'always all-select baseline')
for side,num in [('base',127),('trained',146)]:same(computed[side]['multi_select_gold2_to_5']['exact_calls'],num,'headline exact:'+side);same(computed[side]['multi_select_gold2_to_5']['calls'],216,'headline denominator:'+side);same(computed[side]['ordering']['exact_calls'],0,'ordering exact:'+side);same(computed[side]['ordering']['calls'],24,'ordering denominator:'+side)
for group in ['multi_select_gold2_to_5','multi_select_gold2','multi_select_gold3','multi_select_gold4','multi_select_gold5','multi_select_control6','ordering']:
    a=computed['base'][group];b=computed['trained'][group];line=f"| {group} | {a['candidates']} | {a['exact_rate']['value']:.2%} | {b['exact_rate']['value']:.2%} | {a['partial_mean']['value']:.4f} | {b['partial_mean']['value']:.4f} |";check(line in md,'markdown metric row:'+group)
for phrase in ['流程成功 0/3','training_pool_eligible=false','计划 21 槽；compose 未填 2；final 排除 2；最终内容通过 17（多选 15，排序 2）','control6 是全选对照','可入训练池数量仍为 0','EXPANSION_STOPPED','QA 程序 HOLD']:
    check(phrase in md,'markdown scope:'+phrase)
# All hashes must still match after the read-only inspection.
for p,h in inputs.items():same(sha(p),h,'unchanged during audit:'+p)
result={'schema_version':'final_diagnostic_delivery_review_v1','executor':'/root/final_blind_review','created_at_utc':datetime.now(timezone.utc).isoformat(),'target_delivery':'delivery/diagnostic_v2','status':'PASS_CONTENT_DIAGNOSTIC_DELIVERY_PROCEDURE_HOLD_RETAINED' if not errors else 'ERRORS','errors':errors,'method':'Independently parsed all 588 raw responses, mapped display labels to immutable content IDs, recomputed exact/partial/option or pair confusion, native policy and all report aggregate/candidate/root groups. No report or evaluator code imported. All bound input hashes rechecked.','no_model_inference':True,'reports_modified':False,'formal_promotion':False,'procedure_hold_retained':True,'training_pool_eligible':False,'training_started':False,'training_pool_merged':False,'totals':tot,'per_root':root_details,'solvers':side_results,'independent_solver_audit_consistent':not any('independent audit' in str(e) for e in errors),'report_binding_count':len(r['input_bindings']),'independent_audit_binding_count':len(external['input_bindings']),'input_bindings':[{'path':p,'sha256':h} for p,h in sorted(inputs.items())]}
print(json.dumps(result,ensure_ascii=False,indent=2))
sys.exit(bool(errors))
