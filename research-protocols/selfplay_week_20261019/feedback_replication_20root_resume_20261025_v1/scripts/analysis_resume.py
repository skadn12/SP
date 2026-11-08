#!/usr/bin/env python3
"""Paired root analyses with explicit missingness and unchanged sealed-original scores."""
from __future__ import annotations
import collections,csv,json,pathlib,statistics
import numpy as np
from codex_client import BASE,atomic,bind,load,now,status
from question_resume import SLOTS

def mean(xs):return statistics.mean(xs) if xs else None

def root_comparisons(records,roots,slots,metric='strict'):
 by=collections.defaultdict(list)
 for r in records:
  if r['slot_id'] in slots:by[(r['root_id'],r['slot_id'],r['arm'])].append(r)
 result=[]
 for root in roots:
  eligible=[]
  for slot in slots:
   groups=[by[(root,slot,a)] for a in ['F','G']]
   if all(len(g)==2 and len({r['variant_id'] for r in g})==2 and all(r.get('returned') for r in g) and all(isinstance(r.get(metric),dict) for r in g) for g in groups):
    vals={a:mean([float(r[metric]['exact']) for r in g]) for a,g in zip(['F','G'],groups)}
    eligible.append({'slot_id':slot,**vals,'difference':vals['F']-vals['G']})
  result.append({'root_id':root,'eligible_common_slots':len(eligible),'complete_primary_slots':len(eligible)==len(slots),'F':mean([e['F'] for e in eligible]),'G':mean([e['G'] for e in eligible]),'difference':mean([e['difference'] for e in eligible]),'slots':eligible})
 return result

def bootstrap(rows):
 values=[x['difference'] for x in rows if x['difference'] is not None]
 if not values:return {'eligible_roots':0,'mean_difference':None,'ci95':None,'resamples':10000,'seed':20261025}
 rng=np.random.default_rng(20261025);a=np.asarray(values);boot=np.mean(rng.choice(a,size=(10000,len(a)),replace=True),axis=1)
 return {'eligible_roots':len(a),'mean_difference':float(a.mean()),'ci95':[float(x) for x in np.quantile(boot,[0.025,0.975])],'resamples':10000,'seed':20261025,'unit':'paired root; both arms and both presentations retained'}

def summary(records):
 out=[]
 for arm in ['F','G']:
  for group,slots in [('PRIMARY',[s for s in SLOTS if s!='MC6']),('MC2-5',['MC2','MC3','MC4','MC5']),('ORDERING',['ORDER1','ORDER2','ORDER3']),('MC6_CONTROL',['MC6']),('ALL',list(SLOTS))]:
   rr=[r for r in records if r['arm']==arm and r['slot_id'] in slots];n=len(rr);ss=[r['strict'] for r in rr];valid=[s for s in ss if s['valid']];mc=[s for s in ss if s['question_type']=='multi_select'];order=[s for s in ss if s['question_type']=='ordering'];vorder=[s for s in order if s['valid']]
   precision=[];recall=[];f1=[]
   for s in mc:
    pred=set(s['prediction']);gold=set(s['gold']);tp=len(pred&gold);p=tp/len(pred) if pred else 0.;r=tp/len(gold) if gold else 0.;precision.append(p);recall.append(r);f1.append(2*p*r/(p+r) if p+r else 0.)
   answer_only=[r['answer_only'] for r in rr if isinstance(r.get('answer_only'),dict)]
   out.append({'arm':arm,'group':group,'returned_presentations':n,'valid_contract_presentations':len(valid),'strict_exact_count':sum(s['exact'] for s in ss),'strict_exact_accuracy':mean([float(s['exact']) for s in ss]),'answer_only_presentations':len(answer_only),'answer_only_exact_accuracy':mean([float(s['exact']) for s in answer_only]) if len(answer_only)==n else None,'mc_macro_precision':mean(precision),'mc_macro_recall':mean(recall),'mc_macro_f1':mean(f1),'mc_false_positive_labels':sum(s['false_positive'] for s in mc),'mc_false_negative_labels':sum(s['false_negative'] for s in mc),'mc_overselection_response_rate':mean([float(len(s['prediction'])>len(s['gold'])) for s in mc]),'mc_underselection_response_rate':mean([float(len(s['prediction'])<len(s['gold'])) for s in mc]),'ordering_pairwise_correct':sum(s['pairwise_correct'] for s in order),'ordering_pairwise_total_strict':sum(s['pairwise_total'] for s in order),'ordering_pairwise_total_contract_valid':sum(s['pairwise_total'] for s in vorder),'ordering_pairwise_strict':sum(s['pairwise_correct'] for s in order)/sum(s['pairwise_total'] for s in order) if order else None,'ordering_pairwise_contract_valid_only':sum(s['pairwise_correct'] for s in vorder)/sum(s['pairwise_total'] for s in vorder) if vorder else None})
 return out

def run():
 sf=load(BASE/'SOURCE_FREEZE.json');qf=load(BASE/'QUESTION_FREEZE.json');raw=load(BASE/'execution/RAW_RESPONSE_FREEZE.json');new=load(BASE/'execution/SCORES.json')['records']
 manifest=load(BASE/'RESUME_MANIFEST.json');orig=pathlib.Path(manifest['original_four_experiment']);old_scores=load(orig/'execution/SCORES.json');old=[]
 for r in old_scores['records']:
  s=r['strict'];old.append({'root_id':s['root_id'],'arm':r['arm'],'slot_id':r['slot_id'],'variant_id':s['variant_id'],'strict':s,'answer_only':None,'returned':True,'request_id':r['request_id']})
 roots=[r['root_id'] for r in sf['outcomes']];oldroots=sorted({r['root_id'] for r in old});primary=[s for s in SLOTS if s!='MC6']
 comparisons={}
 for name,rr,cohort in [('new16',new,roots),('original4_readonly',old,oldroots),('combined20_secondary',new+old,roots+oldroots)]:
  cr=root_comparisons(rr,cohort,primary);comparisons[name]={'planned_roots':len(cohort),'root_results':cr,'primary':bootstrap(cr),'summaries':summary(rr),'complete_seven_slot_root_sensitivity':bootstrap([r for r in cr if r['complete_primary_slots']]),'available_common_slot_pooled_descriptive':mean([s['difference'] for r in cr for s in r['slots']]),'MC2_5_secondary':bootstrap(root_comparisons(rr,cohort,['MC2','MC3','MC4','MC5'])),'ordering_secondary':bootstrap(root_comparisons(rr,cohort,['ORDER1','ORDER2','ORDER3']))}
  if name=='new16':comparisons[name]['answer_only_same_paired_cohort']=bootstrap(root_comparisons(rr,cohort,primary,metric='answer_only'))
 # Content validity and returned status determine eligibility, never correctness or target labels.
 returned_ids={r['candidate_id'] for r in new};production=[]
 for arm in ['F','G']:
  slots=[s for s in qf['slots'] if s['arm']==arm]
  production.append({'arm':arm,'planned_roots':16,'planned_slots':128,'initial_mechanically_valid':sum(bool(s.get('initial_mechanically_valid')) for s in slots),'initial_content_valid':sum((s.get('initial_audit') or {}).get('status')=='PASS' for s in slots),'final_content_valid':sum(s['status']=='PASS' for s in slots),'missing_content_slots':sum(s['status']!='PASS' for s in slots),'slots_with_any_return':sum(s['candidate_id'] in returned_ids for s in slots),'returned_presentations':sum(r['arm']==arm for r in new),'planned_presentations':256})
 consistency=[]
 by=collections.defaultdict(list)
 for r in new:by[r['candidate_id']].append(r)
 for cid,rr in by.items():
  if len(rr)==2:
   valid=all(r['strict']['valid'] for r in rr)
   preds=[r['strict']['prediction'] for r in rr]
   if rr[0]['strict']['question_type']=='multi_select':preds=[sorted(p) for p in preds]
   consistency.append({'candidate_id':cid,'both_exact':all(r['strict']['exact'] for r in rr),'at_least_one_exact':any(r['strict']['exact'] for r in rr),'both_contract_valid':valid,'canonical_predictions_identical':preds[0]==preds[1] if valid else None})
 strata={}
 for group in ['slot_id','root_id','variant_id']:
  strata[group]={str(v):summary([r for r in new if r[group]==v]) for v in sorted({r[group] for r in new})}
 strata['gold_count']={str(v):summary([r for r in new if r['strict'].get('gold_count')==v]) for v in range(2,7)}
 report={'schema_version':'fgr20_resume_analysis_v1','at':now(),'training':False,'difference_direction':'F minus G: negative means F harder for fixed B1; positive means F easier','no_nonresponse_imputation':True,'source_pass_roots':sf['pass_count'],'source_failed_roots':16-sf['pass_count'],'question_valid_slots':qf['valid_slots'],'question_missing_slots':qf['missing_slots'],'solver_returned':raw['returned_count'],'solver_missing_among_prepared':raw['missing_count'],'solver_missing_of_512_planned':512-raw['returned_count'],'comparisons':comparisons,'production':production,'presentation_consistency':consistency,'strata':strata,'sealed_original_evidence':{'scores':bind(orig/'execution/SCORES.json'),'comparison':bind(orig/'DIFFICULTY_COMPARISON.json'),'raw_freeze':bind(orig/'execution/RAW_RESPONSE_FREEZE.json'),'rerun':False},'limitations':['Only frozen selected train roots; parent sources historically seen during M-clean training.','Question yield and source failures can restrict the paired cohort; missing roots retained.','Two display presentations are not independent observations.','Root bootstrap describes this selected sample, not a population guarantee.','MC6 is an all-true control excluded from the primary endpoint.','Lower F exact accuracy alone does not establish feedback mechanism or training benefit.','Independent semantic entailment of solver citations and mechanism-error labels was not completed; exact scoring is mechanical.','Combined20 is secondary and includes the four exploratory sealed roots; no independent-confirmation claim.','Original4 answer-only metric was not rederived; combined answer-only not reported.']}
 atomic(BASE/'ANALYSIS.json',report)
 path=BASE/'ROOT_DIFFERENCES.csv'
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=['cohort','root_id','eligible_common_slots','F','G','difference']);w.writeheader()
  for cohort,d in comparisons.items():
   for r in d['root_results']:w.writerow({'cohort':cohort,**{k:r[k] for k in w.fieldnames if k!='cohort'}})
 def fmt(x):return '缺失' if x is None else f'{x:.4f}'
 lines=['# F/G 20-root 复现恢复结果','', '固定 B1；FULL THINK_OFF；无训练；旧四根只读复用。F−G 为负表示 F 题更难。缺失回答不计为错误。','', '|样本|计划根数|成对有效根数|根均值 F−G|根 bootstrap 95% 区间|','|---|---:|---:|---:|---|']
 for name,d in comparisons.items():
  b=d['primary'];ci=b['ci95'];lines.append(f"|{name}|{d['planned_roots']}|{b['eligible_roots']}|{fmt(b['mean_difference'])}|{('['+fmt(ci[0])+', '+fmt(ci[1])+']') if ci else '缺失'}|")
 lines+=['','|组|计划语义槽|最终内容通过|缺失内容槽|返回展示/计划展示|','|---|---:|---:|---:|---:|']
 for r in production:lines.append(f"|{r['arm']}|128|{r['final_content_valid']}|{r['missing_content_slots']}|{r['returned_presentations']}/256|")
 lines+=['','|样本|组|范围|返回展示|严格 exact|仅答案 exact|','|---|---|---|---:|---:|---:|']
 for name,d in comparisons.items():
  for r in d['summaries']:lines.append(f"|{name}|{r['arm']}|{r['group']}|{r['returned_presentations']}|{fmt(r['strict_exact_accuracy'])}|{fmt(r['answer_only_exact_accuracy'])}|")
 lines+=['','来源通过：'+str(sf['pass_count'])+'/16。所有失败根、槽及原始调用均保留。','结论：当前结果只描述此固定模型与有效成对槽的题目难度；不足以据此声称反馈收益或训练收益。','', '限制：来源/内容失败减少覆盖率；旧四根为探索背景；未完成 Solver 引文语义和窄机制逐项复核。']
 (BASE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
 status('analysis',analysis=bind(BASE/'ANALYSIS.json'),eligible_new16_primary_roots=comparisons['new16']['primary']['eligible_roots'])
 return report
if __name__=='__main__':run()
