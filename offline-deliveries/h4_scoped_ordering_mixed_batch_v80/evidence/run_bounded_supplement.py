from pathlib import Path
import sys
HOME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME))
from common import load_multi, write, now, bind
engine=load_multi('asset_generation')
run=HOME/'multiselect/assets/root130221_family'
targets,feedback,paths=engine.supplement_spec(run)
write(HOME/'evidence/SUPPLEMENT_DECISION.json',{
 'created_at_utc':now(),'family_id':run.name,'target_fact_ids':targets,
 'reason':'Only three distinct contradiction variants survived the complete-source initial review; one authorized bounded supplement targets the rejected or missing facts.',
 'initial_feedback':feedback,'initial_review_files':[bind(p) for p in paths],
 'current_solver_scores_read':False,'semantic_retries':0,
 'max_supplements_for_this_family':1,'max_generation_http_batch':6})
engine.prepare_supplement(run)
print(engine.generate_round(run,'supplement'))
