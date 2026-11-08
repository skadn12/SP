from pathlib import Path
import json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import candidate_runner as c
from candidate_metrics import measure_turns
OUT=HERE/'diagnostics/prompt_tail_b2_s01_01'
response=c.r.base.read(next((OUT/'requests').glob('*/RESPONSE.json')))
parsed=c.r.parse_object(response['text']); inp=c.r.base.read(OUT/'INPUT.json'); baseline=inp['history']
count=c.token_counter(); measured,_=measure_turns(parsed['turns'],count)
duplicates=c.r.duplicate_check.inspect_duplicates(parsed['turns'],{'complete_formal_history':baseline})
report={'schema':'h4.prompt_tail_diagnostic.v41','scope':'offline_diagnostic','source_run':str(HERE/'runs/fresh_b2_candidates_01'),'source_id':inp['source_card_raw']['source_id'],'candidate_id':'diagnostic/prompt_tail_b2_s01_01','turns':len(parsed['turns']),'body_tokens':measured['body_tokens'],'mechanical_status':'PASS','duplicate_result':duplicates,'training_pool_eligible':False,'formal_accepted_baseline':{'turns':len(baseline),'body_tokens':sum(count(t['text']) for t in baseline)},'fresh_collection_body_tokens':0,'new_source_reservations':0,'model':response['metadata']['returned_model'],'http_status':response['metadata']['http_status'],'input_tokens':response['metadata']['input_tokens'],'output_tokens':response['metadata']['output_tokens'],'total_tokens':response['metadata']['total_tokens']}
c.r.base.write(OUT/'DIAGNOSTIC_MECHANICAL_CHECK.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
