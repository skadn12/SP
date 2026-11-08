from pathlib import Path
from datetime import datetime
import json,sys
ROOT_DIR=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT_DIR))
import grounded_tail_probe as g
c=g.c
first=g.PREVIOUS
second=ROOT_DIR/'diagnostics/grounded_tail_s01_01'
out=ROOT_DIR/'results'
rows=[]
for label,folder,disposition in [('tail_only',first,'quarantined'),('original_evidence_and_roles',second,'retained_pending_review')]:
 request=next(folder.glob('requests/*/REQUEST.json')).parent
 response=c.r.base.read(request/'RESPONSE.json')
 parsed=c.r.parse_object(response['text'])
 inputs=c.r.base.read(folder/'INPUT.json')
 card=inputs.get('source_card',inputs.get('source_card_raw'))
 history=inputs['history']
 check=g.validate_response(response['text'],{'source_card':card,'history':history})
 row={'candidate_id':label,'route':g.ROUTE,'disposition':disposition,'realization':parsed,
      'provenance':'diagnostic_replay','raw_review':None,'training_pool_eligible':False,
      'disposition_origin':'independent_assistant_content_analysis_not_program_or_model_verdict',
      'semantic_certified':False,'request':str(request),'source_run':str(g.SOURCE),
      'mechanical':check,'body_tokens':sum(c.token_counter()(t['text']) for t in parsed['turns'])}
 rows.append(row)
count=c.token_counter()
baseline=c.r.base.read(g.SOURCE/'FORMAL_BASELINE.json')['turns']
metrics=c.metrics.summarize_records(rows,count_text_tokens=count,baseline_turns=baseline,scope='offline_diagnostic')
usage={'input_tokens':0,'output_tokens':0,'total_tokens':0}
previous=c.r.base.read(g.SOURCE/'requests/010-cc-train-episode-104814-B-S02-realizer-2/HTTP_RESULT_01.json')['http_completed_at_utc']
transport=[]
for row in rows:
 d=Path(row['request']);response=c.r.base.read(d/'RESPONSE.json');meta=response['metadata']
 for k in usage:usage[k]+=meta[k]
 begin=c.r.base.read(d/'THROTTLE_01.json')['http_started_at_utc'];end=c.r.base.read(d/'HTTP_RESULT_01.json')['http_completed_at_utc']
 gap=(datetime.fromisoformat(begin)-datetime.fromisoformat(previous)).total_seconds()
 assert gap>=35
 assert len(list(d.glob('HTTP_ATTEMPT*.json')))==1
 assert meta['http_status']==200 and meta['returned_model']==c.r.MODEL
 transport.append({'label':row['candidate_id'],'started':begin,'completed':end,'gap_after_previous_recorded_response_seconds':gap,
   'http_status':meta['http_status'],'retry_count':0,'usage':{k:meta[k] for k in usage}})
 previous=end
 payload=c.r.base.read(d/'REQUEST.json');proof=c.r.base.read(d/'PROOF.json')
 assert proof['payload_sha256']==c.r.base.digest(payload)
 assert c.r.base.read(d/'HTTP_RESPONSE_01.json')['text']==response['text']
 context=json.JSONDecoder().raw_decode(payload['messages'][0]['content'].split('\nCANDIDATE_INPUT_JSON\n',1)[1])[0]
 assert context['cumulative_history']==baseline
 assert not (d.parent.parent/'SOURCE_RESERVATIONS.json').exists()
c.r.base.write(out/'DIAGNOSTIC_RECORDS.json',rows)
c.r.base.write(out/'TOKEN_AND_YIELD_REPORT.json',metrics)
c.r.base.write(out/'TRANSPORT_AND_USAGE.json',{'logical_requests':2,'http_attempts':2,'retry_count':0,'model':c.r.MODEL,'usage':usage,'requests':transport})
c.r.base.write(out/'EXECUTION_LIMITATIONS.json',{'first_diagnostic':{
 'shared_lock_held':False,'runtime_throttle_gap_observed_seconds':transport[0]['gap_after_previous_recorded_response_seconds'],
 'recorded_concurrent_request_overlap_found':False,'full_concurrency_guarantee':False,
 'code_binding':'retrospective archived exact executed script; no pre-execution code bindings',
 'feedback_receipt':'intermediate payload before final task append; actual payload separately bound in PROOF and HTTP_ATTEMPT',
 'scope_label':'offline_diagnostic is legacy accounting label; this diagnosis DID call live HTTP'},
 'second_diagnostic':{'shared_lock_held':True,'code_and_inputs_frozen_before_live':True,
 'feedback_receipt':'explicit intermediate scope plus separate before/after payload hashes'},
 'training_pool_eligible':False})
c.r.base.write(out/'RESULT_SUMMARY.json',{'diagnostic_calls':2,'new_source_reservations':0,
 'raw_diagnostic_body_tokens':metrics['diagnostic_replay']['generated']['body_tokens'],
 'pending_diagnostic_body_tokens':metrics['diagnostic_replay']['retained']['body_tokens'],
 'quarantined_diagnostic_body_tokens':metrics['diagnostic_replay']['quarantined']['body_tokens'],
 'formal_new_A_body_tokens':0,'formal_new_B_body_tokens':0,'training_admitted_body_tokens':0,
 'formal_history':{'turns':132,'body_tokens':3672},
 'isolated_candidate_history':{'turns':146,'body_tokens':4091,'not_formal_or_training':True},
 'usage':usage,'new_source_occupied_count':len(c.r.source_inventory.collect_prior()['accepted_uses'])})
text=['# 第二次诊断生成的原始对白','', '14轮、419正文token；复用既有来源，仅作诊断候选，不表示训练准入。','']
for i,t in enumerate(rows[1]['realization']['turns'],1):text.extend([f"**T{i:03d} {t['speaker']}**: {t['text']}",''])
(out/'CANDIDATE_DIALOGUE.md').write_text('\n'.join(text),encoding='utf-8')
print(json.dumps(c.r.base.read(out/'RESULT_SUMMARY.json'),ensure_ascii=False,indent=2))
