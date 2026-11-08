"""Apply the unchanged v77 full-source review contract to new root inventory."""
from pathlib import Path
import importlib.util
from common import HERE, bind, read, need, now, write, verify_bindings
GATE = HERE.parent / 'h4_source_consistency_minfix_v77/source_gate.py'
ROOTS = tuple('cc-train-episode-'+r for r in ('103866','109164','112812','135466'))
def gate():
    s=importlib.util.spec_from_file_location('_v87_unchanged_v77_source_gate',GATE)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def reviews():
    inventory=read(HERE/'evidence/SOURCE_INVENTORY.json')
    rows={r['root_id']:r for r in inventory['root_inventory']}
    need(set(rows)==set(ROOTS),'source_inventory_root_coverage')
    expected={HERE/'source_reviews'/('root'+r.split('-')[-1]+'_REVIEW.json') for r in ROOTS}
    need(set((HERE/'source_reviews').glob('*_REVIEW.json'))==expected,'source_review_coverage')
    result=[]
    for root in ROOTS:
        row={'root_id':root,'source':rows[root]['source'], 'review_path':str(HERE/'source_reviews'/('root'+root.split('-')[-1]+'_REVIEW.json'))}
        review=gate().validate_review(row)
        result.append({'root_id':root,'decision':review['decision'],'source':row['source'],
          'review':bind(row['review_path']),'turn_count':len(review['turn_refs_read']),
          'unit_count':len(review['units']), 'blocking_findings':[f['finding_id'] for f in review['findings'] if f['severity']=='BLOCKING'],
          'warning_findings':[f['finding_id'] for f in review['findings'] if f['severity']=='WARNING']})
    return result

def candidate_reviews():
    inventory = read(HERE / 'evidence/SOURCE_INVENTORY.json')
    result = []
    records = inventory.get('candidate_reviews', [])
    need(len(records) == 1 and records[0]['root_id'] == 'cc-train-episode-210046', 'candidate_source_coverage')
    for row in records:
        need(bind(row['source']['path']) == row['source'] and bind(row['review']['path']) == row['review'],
             'candidate_source_binding_changed')
        value = gate().validate_review({'root_id': row['root_id'], 'source': row['source'], 'review_path': row['review']['path']})
        need(value['decision'] == 'HOLD', 'candidate_hold_changed')
        result.append({'root_id': row['root_id'], 'source': row['source'], 'review': row['review'],
                       'decision': value['decision'], 'turn_count': len(value['turn_refs_read']),
                       'blocking_findings': [f['finding_id'] for f in value['findings'] if f['severity'] == 'BLOCKING']})
    return result

def finalize():
    need(not (HERE/'SOURCE_ADMISSION.COMPLETE.json').exists(),'admission_no_replay')
    rows=reviews();candidates=candidate_reviews();summary={'version':'v87_source_admission_v1','completed_at_utc':now(),'roots':rows,
      'candidate_reviews':candidates, 'candidate_held_roots':[r['root_id'] for r in candidates if r['decision']=='HOLD'],
      'selected_source_reviews_reused_from_v86':True,
      'ready_roots':[r['root_id'] for r in rows if r['decision']=='READY'],
      'held_roots':[r['root_id'] for r in rows if r['decision']=='HOLD'],
      'historical_held_roots_unchanged':['cc-train-episode-231899','cc-train-episode-138343'],
      'full_source_policy':'hold_entire_full_source_if_any_blocking_unit',
      'source_edits':[],'question_answer_certification':False,'training_allowed':False}
    write(HERE/'SOURCE_ADMISSION_SUMMARY.json',summary)
    files={GATE,HERE/'source_admission.py',HERE/'evidence/SOURCE_INVENTORY.json',HERE/'SOURCE_ADMISSION_SUMMARY.json'}
    for r in rows+candidates:files|={Path(r['source']['path']),Path(r['review']['path'])}
    result={'version':'v87_source_admission_freeze_v1','status':'FROZEN','completed_at_utc':now(),
      'ready_roots':summary['ready_roots'],'held_roots':summary['held_roots'],
      'candidate_held_roots':summary['candidate_held_roots'],'training_allowed':False,
      'files':[bind(p) for p in sorted(files)]}
    write(HERE/'SOURCE_ADMISSION.COMPLETE.json',result);return result

def verify():
    record=read(HERE/'SOURCE_ADMISSION.COMPLETE.json');need(record['status']=='FROZEN' and record['training_allowed'] is False,'source_admission_contract')
    verify_bindings(record['files']);rows=reviews();saved=read(HERE/'SOURCE_ADMISSION_SUMMARY.json')
    need(rows==saved['roots'] and candidate_reviews()==saved['candidate_reviews'],'source_review_changed')
    need(record['candidate_held_roots']==saved['candidate_held_roots'],'candidate_source_decisions_changed')
    need(record['ready_roots']==saved['ready_roots'] and record['held_roots']==saved['held_roots'],'source_decisions_changed')
    return saved
if __name__=='__main__':
    import sys,json
    print(json.dumps(finalize() if sys.argv[1]=='finalize' else verify(),ensure_ascii=False))
