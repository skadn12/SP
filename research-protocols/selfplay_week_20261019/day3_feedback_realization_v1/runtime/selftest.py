"""CPU-only structural checks with old public fixtures; no candidate or inference."""
import copy
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import evaluate as ev
from transformers import AutoTokenizer
ev.c.environment()
old=ev.WEEK/'day3_targeted_4root_v1/exports/PUBLIC_QUESTIONS.json'
public=ev.c.read(old)['questions']
fixtures=[next(q for q in public if q['question_type']=='multi_select'),next(q for q in public if q['question_type']=='ordering' and len(q['items'])==3),next(q for q in public if q['question_type']=='ordering' and len(q['items'])==6)]
policy={'namespace':'train.feedback.synthetic_runtime_selftest','allowed_roots':['cc-train-episode-103866','cc-train-episode-104814','cc-train-episode-109164','cc-train-episode-112812'],
 'planned_slots':[{'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_type':q['question_type'],'item_count':len(q['options'] if q['question_type']=='multi_select' else q['items'])} for q in fixtures],
 'direct':{'display_permutations':3,'samples_per_permutation':2,'max_output_tokens':32,'display_seed':2026102101,'sampling_seed':2026102100},
 'diagnostic':{'per_question':1,'max_output_tokens':3072,'display_variant':'perm_00','sample':0,'seed_policy':'same_as_paired_direct'},
 'budget':{'max_direct_calls':18,'max_diagnostic_calls':3},'runtime':{'max_model_len':16384,'batch_size':8,'gpu_memory_utilization':.8},
 'model':{'name':'S0_M_clean','model':str(ev.c.MODEL),'tokenizer':str(ev.c.TOKENIZER)},'temperature':.8,'top_p':.9,'enable_thinking':False,'automatic_retry':False,'source_truncation':False}
ev.policy_check(policy)
tokenizer=AutoTokenizer.from_pretrained(str(ev.c.TOKENIZER),local_files_only=True)
rows,prompts,tokens,maps=ev.make_plan(policy,fixtures,tokenizer,ev.c.rt.source_check)
assert len(rows)==21 and sum(r['branch']=='direct' for r in rows)==18 and sum(r['branch']=='evidence' for r in rows)==3
for q in fixtures:
 records=[r for r in rows if r['candidate_id']==q['candidate_id']]
 direct=[r for r in records if r['branch']=='direct'];diag=next(r for r in records if r['branch']=='evidence')
 assert len(direct)==6 and len({r['variant_id'] for r in direct})==3
 for variant in {'perm_00','perm_01','perm_02'}:assert {r['slot'] for r in direct if r['variant_id']==variant}=={0,1}
 paired=next(r for r in direct if r['request_id']==diag['direct_request_id'])
 assert paired['seed']==diag['seed'] and paired['display']==diag['display']
 field='options' if q['question_type']=='multi_select' else 'items';canonical=maps['public_to_canonical'][q['candidate_id']]
 synthetic_gold=list(q[field])[:2] if field=='options' else list(q[field])
 for row in direct:
  variant=maps['requests'][row['request_id']];inverse={v:k for k,v in variant['display_to_content'].items()}
  answer=[inverse[canonical[label]] for label in synthetic_gold]
  if field=='options': answer.sort()
  text='<box>'+','.join(answer)+'</box>'
  raw={'status':'RETURNED','raw_output':text,'metadata':{'finish_reason':'stop'},'response_committed_at_utc':ev.c.now()}
  result=ev.legacy_score.score_one(row,raw,variant,canonical,synthetic_gold)
  assert result['valid'] and result['exact'] and (set(result['canonical_prediction'])==set(synthetic_gold) if field=='options' else result['canonical_prediction']==synthetic_gold), result
for row in rows:assert row['prompt_tokens']+row['max_output_tokens']<=16384
bad=copy.deepcopy(policy);bad['diagnostic']['max_output_tokens']=4096
try:ev.policy_check(bad)
except Exception:pass
else:raise AssertionError('Changed diagnostic budget was not rejected')
result={'status':'PASS','created_at_utc':ev.c.now(),'synthetic_cpu_only':True,'real_candidate_output':False,'source_fixtures':'3 old public questions, no private Gold used; invented scoring labels only for mapping test','solver_calls':0,'gpu_started':False,'fixture_questions':3,'synthetic_direct_requests':18,'synthetic_diagnostic_requests':3,'all_synthetic_gold_scores_exact':True,'direct_3x2_and_paired_diagnostic_verified':True,'changed_output_budget_rejected':True,'minimum_context_headroom':min(16384-r['prompt_tokens']-r['max_output_tokens'] for r in rows),'script_binding':ev.c.bind(Path(__file__)),'evaluate_binding':ev.c.bind(Path(ev.__file__))}
ev.save_new(ev.HERE/'SELFTEST_RESULT.json',result)
print(json.dumps(result))
