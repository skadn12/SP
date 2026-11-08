"""Bind the source reviewer records to the unchanged private children."""
from pathlib import Path
import sys
HOME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME))
from common import load_multi,read,write,sha,now
context=load_multi('review_schema').validate_final_sources(HOME/'multiselect')
crosscheck=read(HOME/'evidence/PRIVATE_CROSSCHECK.json')
rows={r['candidate_id']:r for r in crosscheck['questions']}
assert set(rows)==set(context['questions'])
for group in context['groups']:
 j=group['judgment']
 assert j['current_gold_counts_known_before_review'] is True
 assert j['prior_current_variant_judgment_exposure'] is True
 judged=[]
 for qid in group['rows']:
  r=rows[qid]
  judged.append({'candidate_id':qid,'root_id':r['root_id'],'source_sha256':r['source_sha256'],
   'core_status':'KNOWN_SUBSTANTIVE_ISSUE' if r['errors'] else 'CORE_USABLE',
   'reason':('; '.join(r['errors']) if r['errors'] else 'Saved whole-question source relations match unchanged private Gold labels and exact evidence. Stem, scope and distinctness pass; no edits. The source reviewer disclosed inherited Gold-count and prior semantic-judgment exposure.')})
 record={'group':group['group'],'created_at_utc':now(),
  'pool_sha256':sha(HOME/'multiselect/POOL.json'),
  'source_judgments_sha256':group['binding']['sha256'],
  'current_solver_scores_read':False,'solver_scores_used':False,
  'source_first_for_current_children':True,'child_gold_read_before_save':False,
  'current_gold_counts_known_before_review':True,'prior_current_variant_judgment_exposure':True,
  'fresh_blind_child_review':False,
  'source_first_boundary':'Saved before reviewer opened private Gold label arrays and private pool; counts and prior semantic judgments were already exposed and are disclosed.',
  'source_judgments_bytes_changed_after_private_read':False,
  'current_pool_read_after_both_group_source_judgments_save':True,
  'first_private_pool_read_at_utc':crosscheck['first_private_pool_read_at_utc'],
  'both_source_judgments_saved_bindings':crosscheck['both_source_judgments_saved_bindings'],
  'question_or_gold_edits':[],'training_allowed':False,'questions':judged}
 write(HOME/'multiselect/final_review'/f"group_{group['group']}_AUDIT.json",record)
print({'questions':len(rows),'usable':sum(not r['errors'] for r in rows.values())})
