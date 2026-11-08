"""Build an explicit reviewed feedback handoff; never repair model content."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import json,hashlib,copy
H=Path(__file__).resolve().parent;P=H.parent/'day3_targeted_4root_v1';R=H/'runtime/evidence_v1'
def read(p):return json.loads(p.read_text())
def bind(p):
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(p,x):
 assert not p.exists(),str(p)+' exists'
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+chr(10))
coverage=read(H/'COVERAGE_PLAN.json')['items'];bycid={x['candidate_id']:x for x in coverage}
models=read(R/'manifest.json')['model_bindings'];parsed=read(R/'PARSED_RESPONSES.json');px={x['candidate_id']:x for x in parsed['questions']}
review_rows=[]
for name in ['MC_SEMANTIC_REVIEWS.jsonl','ORDER_SEMANTIC_REVIEWS.jsonl']:
 f=H/'review'/name
 for ln in f.read_text().splitlines():
  raw=json.loads(ln);x=copy.deepcopy(raw);x['review_record_origin']=bind(f);x['original_review_status']=x['review_status'];x['original_review_dimensions']={k:x[k] for k in ['quote_check','semantic_support_check','judgment_answer_consistency_check']}
  if x['review_status']=='NOT_REVIEWABLE':
   for k in x['original_review_dimensions']:x[k]='NOT_APPLICABLE'
  else:
   for k,v in x['original_review_dimensions'].items():
    x[k]={'pass':'PASS','fail':'FAIL','partial':'HOLD','unknown':'HOLD'}.get(v,v)
   if x['review_status']=='VALID_REVIEWED':x['review_status']='HOLD'
   elif x['quote_check']=='HOLD' and any(not e['exact_substring'] for e in px[x['candidate_id']]['citation_checks']):x['quote_check']='FAIL'
  x['normalization_note']='Review enum adaptation only; partial remains incomplete (HOLD), except partly inexact quotation becomes strict quote FAIL. Original reviewer record bound and retained.'
  review_rows.append(x)
reviews={x['candidate_id']:x for x in review_rows};assert set(reviews)==set(bycid)
ordered=[reviews[x['candidate_id']] for x in coverage]
f=H/'SEMANTIC_REVIEWS.jsonl';assert not f.exists();f.write_text(''.join(json.dumps(x,ensure_ascii=False)+chr(10) for x in ordered))
version='train.feedback.targeted4root.evidence.v1'
groups=[
 ('NUMERIC_SOURCE_TO_JUDGMENT',['T4R-R1-MC2','T4R-R2-MC1','T4R-R4-MC3'],
  '本批三道数值关系诊断中存在可由源数值反驳的短运算或选项判断：08:08−07:17=51分钟；11:30−10:45=45分钟；51分钟小于1小时，09:17+51=10:08，不能据此支持10:17。',
  '同日明确日程、明确数值和所问区间；区分抽取错误、运算错误和把正确计算对应到错误选项。',
  '仅在缺必要素材时自然延续补齐开始/结束时间、同日与计划状态；按共同规则保留全文，不为每个错误人工写出算式或答案。',
  '计算比较多选；正确关系与相邻但错误的时间差/目标时刻干扰项并列，不能只改Gold数量。',
  '在另行冻结的同root/family对照中保持source/映射稳定，数值条件改变后逐项计算与最终选项都一致，且原错项不再被选。'),
 ('CONDITION_AND_ANSWER_ALIGNMENT',['T4R-R3-MC1'],
  '本批该题出现reason说明未获准转发，但judgment仍supported，最终答案与逐项判断不完全一致；源规则要求school-only和对当前未改版本的明确许可。',
  '授权与school-only为合取；版本变动与许可不能跨版本继承。',
  '需要自然明确的版本、许可范围和变动状态；缺什么补什么，不把当前源中的未完成说成已完成。',
  '条件合取多选；错误项一次仅违背一项条件，证据可从全文核验。',
  '固定条件组合的后续探针中，judgment/reason/final三者一致，满足全部条件者选入，任一必要条件缺失者拒绝。'),
 ('SOURCE_FACT_REJECTION',['T4R-R4-MC1'],
  '该题最终答案正确，但一项拒绝理由否认了原文已出现的details请求；答案正确不能保证解释有依据。',
  '判断是否回答精确问题，不能将相关但非所问的真实事实说成原文不存在。',
  '保留请求与其后续状态的自然上下文，不为凑完整解释修改已审事实。',
  '证据选择多选；区分事实为真、与问题相关、满足问题全部限制。',
  '对真实但非充分证据的项，能用相关性/充分性拒绝，而不杜撰源中不存在的说法。'),
 ('REQUESTED_VS_CONFIRMED_CLOCK',['T4R-R3-ORD2'],
  '该时刻排序把失败预约请求19:15混为18:15，反转请求时刻与已确认18:15，实际应17:11→17:58→18:15→19:15。',
  '题目要求比较时间标记而非叙述次序，必须保留请求、不可用、确认的不同状态。',
  '补料只在缺状态链时提供自然请求—失败—替代—确认，明确同日可比时钟。',
  '明确时刻排序，报告顺序与时钟顺序可不同，每项绑定对应计划/请求状态。',
  '保持时钟值和状态，改变报告顺序后仍按17:11、17:58、18:15、19:15排序，且不称未成功请求已完成。'),
 ('REPORT_ORDER_CONSTRAINT',['T4R-R4-ORD1'],
  '该六项报告排序将源T005/T007和T011/T013两对关系反转，部分事实定位正确并未传递到最终全序。',
  '命名场景内首次报告顺序，不能用常见步骤替换实际报告锚点。',
  '只需要自然且明确的多事项报告，保留首次/重复提及边界，避免仅靠项目编号推断。',
  '叙述重建排序；每项有唯一可审核锚点，最终顺序检查必要序对。',
  '固定证据锚点的后续探针中，输出全序满足全部明确先后关系；若定位和顺序均正确，则本地现象未复现。'),
 ('CITATION_FIDELITY_SCOPE',['T4R-R1-ORD2','T4R-R1-ORD3','T4R-R2-ORD2','T4R-R3-ORD1','T4R-R3-ORD3','T4R-R4-ORD2'],
  '这些有效诊断存在轮号错位、改写当逐字引文或引文与项状态不符；其中最终排序可能正确，故仅作引用质量负反馈。',
  '短quote应原样出现在指定turn；speaker不拼入quote；请求与确认引用分别对应其项目。',
  '不为引用格式缺陷生成额外对话；先修执行格式与核验接口，完整事实保持原样。',
  '这是诊断执行/审核反馈，不自动构成出题难度或训练侧能力目标。',
  '同一来源的后续已冻结诊断中，短quote和turn_ref逐字准确且支持所判项目；只答案正确不能通过引用门。')]
entries=[]
for key,ids,claim,trigger,material,design,falsify in groups:
 assert all(reviews[c]['review_status']=='FAIL' and reviews[c]['source_verified_error'] for c in ids)
 entries.append({'feedback_id':'T4E-'+key,'version':version,'status':'supported_negative','operational_claim':claim,'source_verified_operational_feedback':True,'formal_reproduced':False,'evidence_candidate_ids':ids,'root_ids':sorted({bycid[c]['root_id'] for c in ids}),'fact_family_ids':sorted({bycid[c]['fact_family_id'] for c in ids}),'observed_trigger_definition':trigger,'dialogue_material_needed':material,'question_relation_and_design':design,'expected_observable_failure':claim,'falsification_condition':falsify,'competing_explanations':['Single fixed mapping and one diagnostic sample; sampling and prompt differences remain.','Evidence-format workload and extra output may contribute.','Visible rationale is not verified internal reasoning.'],'source_review_binding':bind(H/'SEMANTIC_REVIEWS.jsonl'),'response_bindings':[px[c]['raw_binding'] for c in ids],'same_solver_model_bindings':models,'qualified_for_formal_B_target_quota':False,'probe_status':'NOT_EXECUTED_NO_NEW_REPRODUCED_CLAIM','intended_feedback_consumer':'diagnostic_interface_quality' if key=='CITATION_FIDELITY_SCOPE' else 'exploratory_train_challenger','no_dev_test_feedback':True})
dispositions=[]
for c in [x['candidate_id'] for x in coverage]:
 v=reviews[c];ids=[e['feedback_id'] for e in entries if c in e['evidence_candidate_ids']]
 if v['review_status']=='NOT_REVIEWABLE':
  d={'status':'invalid_diagnostic','reason':'Strict diagnostic schema invalid; retain full raw and failed conditions, do not repair/extract answer for primary comparison.','feedback_ids':[],'excluded_from_usable_feedback':True}
 elif v['review_status']=='HOLD':
  d={'status':'review_hold','reason':'Final answer correct but source evidence chain partial; no supported answer-error weakness from this response.','feedback_ids':[],'excluded_from_usable_feedback':True}
 elif v['review_status']=='PASS':
  d={'status':'no_supported_weakness','reason':'Positive control: source, rationale and answer pass; no negative weakness claimed.','feedback_ids':[]}
 else:
  d={'status':'negative_evidence','reason':'Completed source-verified review of observable model error; use narrowly, not as a new reproduced/internal-cause claim.','feedback_ids':ids,'source_verified_operational_feedback':True}
 dispositions.append({'candidate_id':c,**d})
feedback={'schema_version':'reviewed_batch_feedback_snapshot_v1','version':version,'parent_version':'day3_targeted_four_root.material_and_shortcut_feedback','created_at_beijing':datetime.now(timezone(timedelta(hours=8))).isoformat(),'faithful_internal_trace_claim':False,'unverified_rationale_training_target':False,'entries':entries,'question_dispositions':dispositions,'old_day2_reproduced_scope_preserved':'Existing narrow ordering feedback remains historical admissible evidence; these6 local cards do not overwrite it.','generation_handoff':{'ready_for_exploratory_plan':True,'feedback_loop_closed':False,'actual_input_bindings':[],'target_realization_review_bindings':[],'feedback_to_plan_to_candidate':[],'formal_target_realization_gate_passed':False,'next_batch_started':False,'reason':'Reviewed feedback packet is available. Prospective next-generation input must bind this snapshot; no new candidate or target-realization evidence exists yet.'}}
save(H/'FEEDBACK_SNAPSHOT.json',feedback)
state={'schema_version':'diagnostic_handoff_state_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'expected_question_count':23,'expected_direct_calls':342,'prospective_all_question_diagnostic_rule':True,'question_admission_changed_by_diagnostic':False,'primary_evaluation':'direct','artifacts':{'accepted_questions':bind(P/'exports/PUBLIC_QUESTIONS.json'),'execution_freeze':bind(R/'PREFLIGHT_FREEZE.json'),'direct_baseline_binding':bind(H/'DIRECT_BASELINE_BINDING.json'),'coverage_plan':bind(H/'COVERAGE_PLAN.json'),'responses':bind(H/'DIAGNOSTIC_RESPONSES.jsonl'),'semantic_reviews':bind(H/'SEMANTIC_REVIEWS.jsonl'),'feedback_snapshot':bind(H/'FEEDBACK_SNAPSHOT.json'),'raw_response_freeze':bind(R/'RAW_RESPONSE_FREEZE.json'),'independent_postrun_audit':bind(H/'review/INDEPENDENT_DIAGNOSTIC_POSTRUN.json')}}
save(H/'HANDOFF_STATE.json',state)
print(json.dumps({'semantic_review_counts':dict(Counter(v['review_status'] for v in reviews.values())),'feedback_cards':len(entries),'question_dispositions':dict(Counter(x['status'] for x in dispositions))},ensure_ascii=False))
