#!/usr/bin/env python3
"""Recompute final-answer scores from raw designated field and semantic IDs without reused scorer."""
import json,re,hashlib
from fractions import Fraction
from pathlib import Path
HERE=Path(__file__).resolve().parent
S=HERE/'solver'
def read(p):return json.loads(Path(p).read_text())
def main():
 requests=read(S/'REQUESTS.json');raws=read(S/'RAW_ANSWERS.json');scores=read(S/'SCORES_FINAL_ANSWER_ONLY.json')
 existing={x['request_id']:x for x in scores['requests']};rows=[];differences=[]
 assert len(requests)==len(raws)==len(existing)
 for req,raw in zip(requests,raws):
  assert req['request_id']==raw['request_id']
  valid=True;err=[];labels=[]
  try:
   assert raw['status']=='RETURNED' and raw['finish_reason'] in ['stop','eos']
   def pairs(ps):
    d={}
    for k,v in ps:
     if k in d:raise ValueError('duplicate key')
     d[k]=v
    return d
   obj=json.loads(raw['raw_prediction'],object_pairs_hook=pairs)
   assert isinstance(obj,dict) and isinstance(obj.get('final_answer'),str)
   match=re.fullmatch(r'<box>([A-Z0-9]+(?: [A-Z0-9]+)*)</box>',obj['final_answer']);assert match
   labels=match.group(1).split(' ');mapping=req['display_to_option_id']
   assert labels and len(labels)==len(set(labels)) and set(labels)<=set(mapping)
   predicted=[mapping[x] for x in labels];gold=req['gold_option_ids']
   if req['question_type']=='multi_select':
    n=len(mapping);g=len(gold);tp=len(set(predicted)&set(gold));fp=len(set(predicted)-set(gold));fn=g-tp
    exact=set(predicted)==set(gold);j=Fraction(tp,g)-Fraction(fp,n-g)
   else:
    assert set(predicted)==set(mapping.values())
    goldpos={v:i for i,v in enumerate(gold)};pairs_n=len(gold)*(len(gold)-1)//2
    correct=sum(goldpos[predicted[i]]<goldpos[predicted[k]] for i in range(len(predicted)) for k in range(i+1,len(predicted)))
    exact=predicted==gold;j=Fraction(2*correct-pairs_n,pairs_n);tp=correct;fp=pairs_n-correct;fn=None
  except Exception as e:
   valid=False;exact=False;j=Fraction(0);predicted=[];tp=fp=fn=None;err.append(type(e).__name__+': '+str(e))
  old=existing[req['request_id']]
  if old['valid']!=valid or old['exact_correct']!=exact or Fraction(old['J_fraction'])!=j: differences.append(req['request_id'])
  rows.append({'request_id':req['request_id'],'candidate_id':req['candidate_id'],'context_mode':req['context_mode'],'valid':valid,'prediction_option_ids':predicted,'exact_correct':exact,'J_fraction':str(j),'TP':tp,'FP':fp,'FN':fn,'errors':err})
 out={'status':'PASS' if not differences else 'FAIL','policy':'independent raw designated final JSON field; no reason-based answer repair; semantic-ID scoring; no reused parsing/scoring implementation','requests_checked':len(rows),'mismatches':differences,'rows':rows}
 with (HERE/'SCORE_RECOMPUTATION_AUDIT.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
