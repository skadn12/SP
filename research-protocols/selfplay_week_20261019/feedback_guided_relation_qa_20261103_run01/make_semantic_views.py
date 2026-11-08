#!/usr/bin/env python3
"""Lossless semantic-label view of the designated raw JSON fields, Full only."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def main():
 items=read(P/'ITEMS_PRIVATE.json');requests=read(P/'solver/REQUESTS.json');raws=read(P/'solver/RAW_ANSWERS.json')
 scores={x['request_id']:x for x in read(P/'solver/SCORES.json')['requests']}
 byroot={}
 for c in items:
  view={k:c[k] for k in ['candidate_id','review_id','root_id','arm','question_type','question','gold_option_ids','source_path','source_sha256']}
  view['semantic_options']=[{'option_id':c['label_to_option_id'][o['id']],'text':o['text']} for o in c['options']];view['full_calls']=[]
  byroot.setdefault(c['root_id'],{'root_id':c['root_id'],'source_path':c['source_path'],'source_sha256':c['source_sha256'],'items':[]})['items'].append(view)
  for r,raw in zip(requests,raws):
   assert r['request_id']==raw['request_id']
   if r['candidate_id']!=c['candidate_id'] or r['context_mode']!='full_context':continue
   scored=scores[r['request_id']];obj=scored.get('parsed_output');row={'request_id':r['request_id'],'variant_id':r['variant_id'],'repeat_index':r['repeat_index'],'display_to_option_id':r['display_to_option_id'],'protocol_valid':scored['protocol_valid'],'protocol_errors':scored['protocol_errors'],'final_answer_valid':scored['final_answer_valid'],'answer_only_score':scored.get('answer_only_score'),'finish_reason':raw.get('finish_reason'),'raw_prediction':raw['raw_prediction'],'semantic_item_evidence':{}}
   if isinstance(obj,dict):
    row['answer_reason']=obj.get('answer_reason');row['final_answer']=obj.get('final_answer')
    for label,e in obj.get('item_evidence',{}).items():row['semantic_item_evidence'][r['display_to_option_id'].get(label,'UNKNOWN:'+label)]=e
   view['full_calls'].append(row)
 out=P/'semantic_inputs';out.mkdir(exist_ok=True)
 for root,d in byroot.items():
  d['bindings']=[{'path':str(P/n),'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()} for n in ['ITEMS_PRIVATE.json','solver/REQUESTS.json','solver/RAW_ANSWERS.json','solver/SCORES.json','TARGET_REALIZATION_PRE_SOLVER.json']]
  with (out/(root.split('-')[-1]+'.json')).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2)
 print({k:sum(len(i['full_calls']) for i in d['items']) for k,d in byroot.items()})
if __name__=='__main__':main()
