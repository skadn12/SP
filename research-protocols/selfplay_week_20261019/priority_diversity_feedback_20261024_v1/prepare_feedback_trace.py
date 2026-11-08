"""Trace actual Challenger consumption without upgrading exploratory cards."""
import json
import hashlib
from collections import Counter
from pathlib import Path

P=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text())
def bind(path):
    p=Path(path).resolve();b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(name,value):
    with (P/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    freeze=read(P/'CONTENT_FREEZE.json')
    for b in freeze['bindings']:assert bind(b['path'])==b
    cs=read(P/'CANDIDATES_PRIVATE.json')
    reviews={r['candidate_id']:r for r in read(P/'CONTENT_REVIEW_AI.json')['reviews']}
    ledger=read(P/'SLOT_LEDGER.json')['slots']
    feedback=read(P/'FEEDBACK_INPUT_FREEZE.json')
    entries=[]
    for c in cs:
        r=reviews[c['candidate_id']]
        entries.append({'candidate_id':c['candidate_id'],'root_id':c['root_id'],
         'card_id':c['feedback_target']['card_id'],'card_status_preserved':'EXPLORATORY_OBSERVATION_NOT_REPRODUCED',
         'task_operation':c['task_operation'],'planned_prediction':c['feedback_target']['predicted_error'],
         'realization':c['feedback_target']['source_specific_realization'],
         'positive_counterexample':c['feedback_target']['positive_counterexample'],
         'competing_explanation':c['feedback_target']['competing_explanation'],
         'author_id':c['author_id'],'independent_reviewer_id':r['reviewer_id'],
         'content_decision':r['decision'],'realization_review':r['feedback_realization'],
         'actually_generated':True,'formal_weakness_admission':False})
    cards=[]
    for card in feedback['cards']:
        selected=[e for e in entries if e['card_id']==card['card_id']]
        cards.append({'card_id':card['card_id'],'input_status':card['status'],
         'actual_generated_questions':len(selected),
         'content_admitted_questions':sum(e['content_decision']=='PASS' for e in selected),
         'roots':sorted({e['root_id'] for e in selected}),
         'candidate_ids':[e['candidate_id'] for e in selected],
         'original_historical_card_not_modified':True,'original_challenger_consumed_flag_not_overwritten':True,
         'current_batch_consumption_evidenced_in_separate_receipts':True})
    counts={'planned':len(ledger),'submitted':len(cs),'admitted':sum(r['decision']=='PASS' for r in reviews.values()),
      'initial_review':dict(Counter(r.get('initial_status','MATERIAL_SHORTFALL') for r in ledger)),
      'final_status':dict(Counter(r['status'] for r in ledger)),
      'operations':dict(Counter(c['task_operation'] for c in cs if reviews[c['candidate_id']]['decision']=='PASS')),
      'mc_gold':dict(Counter(str(len(c['gold'])) for c in cs if c['question_type']=='multi_select' and reviews[c['candidate_id']]['decision']=='PASS')),
      'roots':[]}
    for rid in sorted({r['root_id'] for r in ledger}):
        rows=[r for r in ledger if r['root_id']==rid]
        counts['roots'].append({'root_id':rid,'planned':len(rows),'final_status':dict(Counter(r['status'] for r in rows)),
          'initial_status':dict(Counter(r.get('initial_status','MATERIAL_SHORTFALL') for r in rows))})
    save('FEEDBACK_CONSUMPTION_TRACE.json',{'counts':counts,'cards':cards,'questions':entries,
       'author_consumption':freeze['author_consumption'],
       'claims':{'actual_prior_card_consumption_and_new_question_generation':True,
                 'solver_weakness_formal_admission':False,'causal_training_gain':False,
                 'next_generation_after_current_solver_feedback':'NOT_YET_CONSUMED'},
       'bindings':[bind(P/n) for n in ['CONTENT_FREEZE.json','CANDIDATES_PRIVATE.json','CONTENT_REVIEW_AI.json',
                   'SLOT_LEDGER.json','FEEDBACK_INPUT_FREEZE.json','FEEDBACK_OUTCOME_RULES.json']]})
    print(json.dumps(counts,ensure_ascii=False))
if __name__=='__main__':main()
