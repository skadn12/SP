#!/usr/bin/env python3
"""Build evidence-derived OET1 reports and immutable delivery inventories.

No generation, training, evaluation, or finalization runs at import time. All
outputs use exclusive creation. DEV/Test item files are hashed, never parsed;
only known aggregate freezes/results/summaries are read as JSON.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
PHASES=['freeze_rules_and_balanced_protocol','different_root_paired_source_extension',
        'independent_source_and_question_review','freeze_matched_training_pool',
        'paired_training_with_metrics_and_dev_every20steps','frozen_evaluation_and_delivery']
ARMS=['GENERIC','TARGETED'];MODELS=['B1','B1_GENERIC','B1_TARGETED']
SLOTS=['MC2','MC3','MC4','MC5','MC6','ORDER1','ORDER2']
SCENES=[f'OET1:{route}:S{i:02}' for route in 'AB' for i in range(1,4)]
LIVE_NAMES={'TASK_STATUS.json','SUPERVISOR_STATUS.json','supervisor.log','supervisor.pid','supervisor.lock',
            'PROGRESS.jsonl','PRODUCTION_DRIVER.log','FINALIZE_REQUEST','STOP_REQUEST','production_driver_v2.lock'}
EXCLUDE_PARTS={'worker_events','__pycache__','.git'}
FINAL_NAMES={'FINAL_ARTIFACT_MANIFEST.json','ARTIFACTS_FOR_AUDIT.json'}

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def binding(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def verify(value):
    expected={k:value[k] for k in ['path','sha256','bytes']}
    assert binding(value['path'])==expected,('BOUND_FILE_CHANGED',value['path'])
    return Path(value['path'])
def bindings(value):
    if isinstance(value,dict):
        if {'path','sha256','bytes'}<=value.keys():yield {k:value[k] for k in ['path','sha256','bytes']}
        for child in value.values():yield from bindings(child)
    elif isinstance(value,list):
        for child in value:yield from bindings(child)
def verify_bindings(value):
    for b in bindings(value):verify(b)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
def write_text(path,text):
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
def load_optional(path,errors,label):
    if path is None or not Path(path).is_file():return None
    try:return read(path)
    except Exception as error:errors.append({'scope':label,'error':str(error)});return None
def evidence(path):return str(Path(path).resolve()) if path and Path(path).is_file() else None
def attempt_check(label,errors,fn):
    try:fn();return True
    except Exception as error:errors.append({'scope':label,'error':str(error)});return False
def need(value,message):
    if not value:raise ValueError(message)

def source_counts(roots,errors):
    units=[];counts=Counter();all_bindings=[];ambiguities=[];nonready_reviews=[];original_failures=[];mechanical_recoveries=[]
    for root in roots:
        for arm in ARMS:
            unit=BASE/'sources'/root/arm;accepted=0;ancestor_failed=False;scene_rows=[]
            packet=load_optional(unit/'SOURCE_AUTHOR_INPUT.json',errors,f'{root}:{arm}:packet')
            previous=Path(packet['source_binding']['path']) if packet else None
            for scene in SCENES:
                sd=unit/scene.replace(':','_');ap=sd/'ACCEPTED.json';cp=sd/'CUMULATIVE_SOURCE.json';rp=sd/'REVIEW.json';dp=sd/'DRAFT.json'
                row={'scene':scene,'status':'NOT_ATTEMPTED'}
                failure=sd/'FAILURE.json';recovery=sd/'ENGINEERING_RECOVERY.json'
                if failure.exists():original_failures.append({'root_id':root,'condition':arm,'scene':scene,'failure_binding':binding(failure),'accepted_after_explicit_engineering_recovery':ap.exists() and recovery.exists()})
                if recovery.exists():
                    restored=read(recovery);mechanical_recoveries.append({'root_id':root,'condition':arm,'scene':scene,'recovery_binding':binding(recovery),
                        'no_new_semantic_request':restored.get('no_new_semantic_request'),'raw_semantic_fields_unchanged':restored.get('raw_semantic_fields_unchanged')})
                if rp.exists():
                    reviewed=read(rp)
                    if reviewed.get('source_or_gold_ambiguity_failure') is True:
                        ambiguities.append({'root_id':root,'condition':arm,'scene':scene,'review_binding':binding(rp),'failure_codes':reviewed.get('failure_codes',[]),'findings':reviewed.get('findings',[])})
                    if reviewed.get('decision') in ['FAIL','HOLD']:
                        codes=reviewed.get('failure_codes',[])
                        category='PACKAGING_METADATA_HOLD' if 'PREFIX_BOUNDARY_METADATA_MISMATCH' in codes else 'CONTENT_TEMPORAL_BINDING_FAIL' if 'UNRESOLVED_PROJECT_YEAR_BINDING' in codes else 'OTHER_NONREADY_REVIEW'
                        nonready_reviews.append({'root_id':root,'condition':arm,'scene':scene,'decision':reviewed['decision'],'review_binding':binding(rp),'failure_codes':codes,'failure_category':category,'original_decision_preserved':True})
                if ap.exists():
                    def check():
                        r=read(ap);need(r['status']=='READY','SOURCE_RECEIPT_NOT_READY');verify_bindings(r)
                        need(r['source']==binding(cp) and r['review']==binding(rp),'SOURCE_RECEIPT_BINDING_MISMATCH')
                        review=read(rp);draft=read(dp);source=read(cp)
                        need(review['decision']=='READY' and review['reviewer_id']!=draft['author_id'] and review['independent_of_source_author'] is True,'SOURCE_REVIEW_NOT_INDEPENDENT_READY')
                        need(review['source_binding']==binding(cp) and review['draft_binding']==binding(dp),'SOURCE_REVIEW_CONTENT_DRIFT')
                        need(previous is not None and source['turns']==read(previous)['turns']+draft['turns'],'SOURCE_PREFIX_DRIFT')
                        need(len(draft['turns'])==14,'SOURCE_SCENE_TURN_BUDGET_DRIFT')
                    good=attempt_check(f'{root}:{arm}:{scene}',errors,check)
                    row['status']='ACCEPTED' if good else 'FAIL_INTEGRITY'
                    row['receipt_path']=str(ap.resolve());all_bindings.append(binding(ap))
                    if good:accepted+=1
                    previous=cp
                elif (sd/'FAILURE.json').exists():row['status']='FAIL';ancestor_failed=True
                elif dp.exists() or cp.exists() or rp.exists():row['status']='UNACCEPTED_ATTEMPT'
                elif ancestor_failed:row['status']='SHORTFALL_FAILED_ANCESTOR'
                counts[row['status']]+=1;scene_rows.append(row)
            final=unit/'FINAL_SOURCE.json';complete=accepted==6 and final.is_file()
            if complete:
                complete=attempt_check(f'{root}:{arm}:final',errors,lambda:need(final.read_bytes()==(unit/'OET1_B_S03/CUMULATIVE_SOURCE.json').read_bytes(),'FINAL_SOURCE_DIFFERS_FROM_LAST_ACCEPTED_SCENE'))
            units.append({'root_id':root,'condition':arm,'planned_scenes':6,'accepted_scenes':accepted,
                'source_completed':complete,'final_source_path':evidence(final),'scenes':scene_rows})
    return {'planned_sources':16,'completed_sources':sum(u['source_completed'] for u in units),
        'planned_scenes':96,'scene_status_counts':dict(counts),'accepted_scenes':counts['ACCEPTED'],
        'failed_scenes':counts['FAIL']+counts['FAIL_INTEGRITY'],
        'unaccepted_attempts':counts['UNACCEPTED_ATTEMPT'],
        'shortfall_or_not_attempted_scenes':counts['NOT_ATTEMPTED']+counts['SHORTFALL_FAILED_ANCESTOR'],
        'source_or_gold_ambiguity_failures':ambiguities,'source_review_nonready':nonready_reviews,
        'review_failure_category_counts':dict(Counter(r['failure_category'] for r in nonready_reviews)),
        'original_failure_records':original_failures,'explicit_mechanical_recoveries':mechanical_recoveries,
        'units':units,'receipt_bindings':all_bindings}

def question_counts(roots,freeze,errors):
    frozen_rows=freeze.get('decisions',[]) if freeze else []
    expected={(r,a,s) for r in roots for a in ARMS for s in SLOTS}
    mapped={(r['root_id'],r['condition'],r['slot_id']):r for r in frozen_rows}
    if freeze:
        attempt_check('question_freeze_planned_denominator',errors,lambda:need(len(frozen_rows)==112 and set(mapped)==expected and freeze.get('planned_slots')==112,'112_PLANNED_SLOT_FREEZE_MISMATCH'))
    rows=[];reviewed_units=0;semantic_batch_responses=0
    for root in roots:
        for arm in ARMS:
            rp=BASE/'question_reviews'/root/arm/'REVIEW.json';review=load_optional(rp,errors,f'{root}:{arm}:question_review') or {}
            rr={r['slot_id']:r for r in review.get('questions',[])}
            if len(rr)==7:reviewed_units+=1
            call=BASE/'cli_calls'/f'{root}__{arm}__QUESTIONS_AUTHOR'/'RESULT.json'
            if call.exists():semantic_batch_responses+=1
            for slot in SLOTS:
                qp=BASE/'questions'/root/arm/f'{slot}.json';sp=qp.with_name(f'{slot}_SHORTFALL.json');key=(root,arm,slot)
                base=mapped.get(key,{})
                status=base.get('status') or rr.get(slot,{}).get('decision') or ('SHORTFALL' if not qp.exists() else 'FAIL')
                row={'root_id':root,'condition':arm,'slot_id':slot,'status':status,
                    'candidate_or_shortfall_file_exists':qp.exists() or sp.exists(),
                    'independent_review_present':slot in rr,
                    'source_or_gold_ambiguity_failure':bool(base.get('source_or_gold_ambiguity_failure')) or bool(rr.get(slot,{}).get('source_or_gold_ambiguity_failure'))}
                for field in ['question_binding','review_binding','shortfall_binding','reason','mechanical_error','unit_contract_errors']:
                    if field in base:row[field]=base[field]
                rows.append(row)
    def summary(part):
        return {'planned':len(part),'admitted':sum(r['status']=='ADMIT' for r in part),
            'failed':sum(r['status'] in ['FAIL','EXACT_WHOLE_DUPLICATE'] for r in part),
            'shortfall':sum(r['status']=='SHORTFALL' for r in part),
            'other_terminal_or_missing':sum(r['status'] not in ['ADMIT','FAIL','EXACT_WHOLE_DUPLICATE','SHORTFALL'] for r in part),
            'candidate_or_shortfall_files':sum(r['candidate_or_shortfall_file_exists'] for r in part),
            'independent_review_rows':sum(r['independent_review_present'] for r in part)}
    return {'planned_slots':112,**summary(rows),'reviewed_units':reviewed_units,
        'semantic_batch_responses':semantic_batch_responses,'planned_author_batches':16,
        'source_or_gold_ambiguity_failures':sum(r['source_or_gold_ambiguity_failure'] for r in rows),
        'by_root':{root:summary([r for r in rows if r['root_id']==root]) for root in roots},
        'by_arm':{arm:summary([r for r in rows if r['condition']==arm]) for arm in ARMS},
        'by_slot':{slot:summary([r for r in rows if r['slot_id']==slot]) for slot in SLOTS},'decisions':rows}

def training_arm(arm,path,schedule,evaluation,errors):
    data=load_optional(path,errors,f'training:{arm}');out={'status':'NOT_COMPLETED','adapter_path':None,
        'result_path':evidence(path),'validation_passed':False}
    if data is None:return out,None
    adapter=data.get('adapter_path') or data.get('adapter',{}).get('path')
    out['adapter_path']=str(Path(adapter).resolve()) if adapter and Path(adapter).is_file() else None
    def check():
        need(data['status']=='COMPLETED' and data['arm']==arm and data['logical_model']=='B1_'+arm,'TRAINING_RESULT_NOT_COMPLETED_ROLE')
        need(data.get('new_independent_lora') is True and data.get('source_truncation') is False and data.get('checkpoint_selection')=='FIXED_FINAL_NO_DEV_SELECTION','TRAINING_LINEAGE_OR_NO_TRUNCATION_FAILED')
        verify_bindings(data)
        need(out['adapter_path'] is not None and Path(out['adapter_path']).stat().st_size>=100000,'REAL_DURABLE_ADAPTER_WEIGHT_FILE_REQUIRED')
        need(data['adapter']==binding(out['adapter_path']),'ADAPTER_BINDING_MISMATCH')
        final=data['fixed_final_optimizer_step'];need(isinstance(final,int) and final>0 and data['completed_optimizer_steps']==final,'FIXED_FINAL_STEP_NOT_REACHED')
        expected=sorted({0,final}|set(range(20,final+1,20)))
        need(data['dev_steps']==expected and len(data['dev_summaries'])==len(expected),'ACTUAL_ANSWER_DEV_EVERY20_OR_FINAL_MISSING')
        for step,value in zip(expected,data['dev_summaries']):
            dev=read(verify(value));need(dev['status']=='COMPLETED' and dev['scope']=='periodic' and dev['optimizer_step']==step,'PERIODIC_DEV_SUMMARY_MISMATCH')
            need(dev['metrics']['overall']['requests']==evaluation['periodic_request_count'],'PERIODIC_DEV_PLANNED_DENOMINATOR_CHANGED');verify_bindings(dev)
        need(schedule is not None and data['schedule_freeze']==binding(schedule['_path']),'TRAINING_SCHEDULE_NOT_COMMON_FREEZE')
        need(data['completed_microsteps']==schedule['arms'][arm]['microsteps'],'TRAINING_MICROSTEP_BUDGET_MISMATCH')
        need(data['evaluation_freeze']==binding(evaluation['_path']),'TRAINING_EVALUATION_FREEZE_MISMATCH')
        need(Path(data['resume_state']['path']).is_file() and data['resume_state']['bytes']>100000,'DURABLE_FULL_RESUME_STATE_MISSING')
        committed=read(data['committed_optimizer_metrics']['path'])['metrics']
        need([r['optimizer_step'] for r in committed]==list(range(1,final+1)),'OPTIMIZER_METRIC_STEP_GAP')
        for row in committed:
            need(row['gradient_finite'] is True and row['updated_parameters_finite'] is True,'NONFINITE_TRAINING_GRADIENT_OR_PARAMETERS')
            need(all(math.isfinite(row[k]) for k in ['optimizer_step_mean_loss','learning_rate','gradient_norm_before_clip']),'NONFINITE_TRAINING_METRICS')
    passed=attempt_check(f'training:{arm}',errors,check)
    out.update(status='COMPLETED' if passed else 'INVALID_OR_INCOMPLETE',validation_passed=passed,
        completed_optimizer_steps=data.get('completed_optimizer_steps'),fixed_final_optimizer_step=data.get('fixed_final_optimizer_step'),
        completed_microsteps=data.get('completed_microsteps'),mean_train_loss=data.get('mean_train_loss'),
        starting_checkpoint=data.get('starting_checkpoint'),initial_lora_sha256=data.get('initial_lora_sha256'),
        input_tokens=data.get('input_tokens'),supervised_tokens=data.get('supervised_tokens'),
        input_token_relative_difference=data.get('input_token_relative_difference'),
        formal_token_fairness_within_5pct=data.get('formal_token_fairness_within_5pct'),
        dev_steps=data.get('dev_steps'),wandb=data.get('wandb'),resume_state=data.get('resume_state'),
        committed_optimizer_metrics=data.get('committed_optimizer_metrics'),result_binding=binding(path))
    return out,data

def build_report(args):
    errors=[];plan=read(BASE/'PLAN.json');roots=plan['new_train_roots']
    docs={name:load_optional(path,errors,name) for name,path in paths(args).items()}
    for name,doc in docs.items():
        if isinstance(doc,dict):doc['_path']=str(paths(args)[name].resolve())
    protocol=docs['protocol'];gate=docs['gate'];integrity=docs['integrity'];qfreeze=docs['question_freeze'];pool=docs['pool'];schedule=docs['schedule'];evaluation=docs['evaluation'];common=docs['common'];expansion_stop=docs['expansion_stop']
    source=source_counts(roots,errors);questions=question_counts(roots,qfreeze,errors)
    def protocol_check():
        need(protocol and protocol['status']=='FROZEN_BEFORE_AUTHORING' and protocol['roots']==roots,'FORMAL_BALANCED_PROTOCOL_MISSING');verify_bindings(protocol)
        need(protocol['planned']=={'roots':8,'sources':16,'scenes':96,'question_slots':112},'FORMAL_PLANNED_BUDGET_CHANGED')
    protocol_good=attempt_check('protocol',errors,protocol_check)
    gate_keys={'overall_at_least_90pct','every_root_at_least_85pct','every_slot_at_least_75pct','no_source_or_gold_ambiguity_failure'}
    gate_good=bool(gate and gate.get('status')=='PASS' and gate_keys<=set(gate.get('checks',{})) and all(gate['checks'][key] is True for key in gate_keys) and gate.get('planned_slots')==56)
    if gate:attempt_check('engineering_gate_bindings',errors,lambda:verify_bindings(gate))
    integrity_good=bool(integrity and integrity.get('verdict')=='PASS' and not integrity.get('failures') and not integrity.get('actual_tool_events') and not integrity.get('active_calls'))
    if integrity:integrity_good=attempt_check('production_integrity_bindings',errors,lambda:verify_bindings(integrity)) and integrity_good
    if docs['prefix_diagnostic']:attempt_check('source_prefix_metadata_diagnostic_bindings',errors,lambda:verify_bindings(docs['prefix_diagnostic']))
    integrity_complete=False
    if integrity_good:
        recorded={(row['root_id'],row['condition']):row for row in integrity.get('sources',[])}
        integrity_complete=all((root,arm) in recorded and recorded[(root,arm)].get('validated_accepted_scenes')==6 and
            recorded[(root,arm)].get('final_source')==binding(BASE/'sources'/root/arm/'FINAL_SOURCE.json')
            for root in roots for arm in ARMS if (BASE/'sources'/root/arm/'FINAL_SOURCE.json').is_file()) and len(recorded)==16 and all(row.get('validated_accepted_scenes')==6 for row in recorded.values())
    export_good=bool(qfreeze and docs['export'] and qfreeze.get('planned_slots')==112 and len(qfreeze.get('decisions',[]))==112)
    if docs['export']:export_good=attempt_check('112_slot_export',errors,lambda:verify_bindings(docs['export'])) and export_good
    pool_good=False;pool_counts={};pool_layers={};pool_mode=None
    if pool:
        def pool_check():
            need(pool['status']=='FROZEN_BEFORE_TRAINING' and pool['new_controls_excluded'] is True,'TRAINING_POOL_NOT_FROZEN_PRIMARY_ONLY')
            need(pool['planned_new_slots']==112 and pool['solver_scores_read_for_selection'] is False,'TRAINING_POOL_BOUNDARY_FAILED')
            records=pool['records'];need(records,'EMPTY_TRAINING_POOL')
            keys=Counter((r['paired_slot_key'],r['condition']) for r in records)
            need(all(keys[(key,arm)]==1 for key in pool['matched_paired_slot_keys'] for arm in ARMS),'UNMATCHED_TRAINING_POOL')
            need(all(r['split']=='train' and r['slot']!='MC6' for r in records),'POOL_SPLIT_OR_MC6_VIOLATION');verify_bindings(pool)
        pool_good=attempt_check('training_pool',errors,pool_check);pool_mode=pool.get('mode')
        pool_counts={arm:sum(r['condition']==arm for r in pool.get('records',[])) for arm in ARMS}
        pool_layers={arm:dict(Counter(r['historical_exposure'] for r in pool.get('records',[]) if r['condition']==arm)) for arm in ARMS}
    if schedule:attempt_check('schedule_freeze',errors,lambda:verify_bindings(schedule))
    training={};raw_results={}
    for arm,name in [('GENERIC','generic_result'),('TARGETED','targeted_result')]:
        training[arm],raw_results[arm]=training_arm(arm,paths(args)[name],schedule,evaluation,errors)
    paired=False
    if all(training[a]['validation_passed'] for a in ARMS):
        def pair_check():
            g,t=[raw_results[a] for a in ARMS]
            need(g['initial_lora_sha256']==t['initial_lora_sha256'],'PAIRED_INITIAL_LORA_NOT_EQUAL')
            need(g['starting_checkpoint']==t['starting_checkpoint']==protocol['training']['starting_checkpoint'],'PAIRED_B1_START_NOT_EQUAL')
            need(g['schedule_freeze']==t['schedule_freeze'] and g['completed_optimizer_steps']==t['completed_optimizer_steps'] and g['supervised_tokens']==t['supervised_tokens'],'PAIRED_OPTIMIZER_OR_TOKEN_BUDGET_MISMATCH')
            need(training['GENERIC']['adapter_path']!=training['TARGETED']['adapter_path'],'TWO_ARMS_CANNOT_SHARE_ADAPTER_FILE')
        paired=attempt_check('paired_training_fairness',errors,pair_check)
    common_good=False
    if common:
        def common_check():
            need(common['status']=='COMPLETED' and common['models']==MODELS and common['initial_lora_equal'] is True and common['fixed_final_endpoint'] is True,'COMMON_THREE_MODEL_EVALUATION_NOT_COMPLETED')
            need(paired,'COMMON_EVALUATION_WITHOUT_VALID_PAIRED_TRAINING');verify_bindings(common)
            need(common['evaluation_freeze']==binding(paths(args)['evaluation']),'COMMON_EVALUATION_FREEZE_MISMATCH')
            for role in MODELS:
                summary=read(common['summaries'][role]['path'])
                need(summary['status']=='COMPLETED' and summary['scope']=='final' and summary['identity']['logical_model']==role,'COMMON_EVALUATION_ROLE_MISMATCH')
                need(summary['metrics']['overall']['requests']==evaluation['final_request_count'],'COMMON_EVALUATION_REQUEST_DENOMINATOR_CHANGED')
                need(summary['evaluation_freeze']==common['evaluation_freeze'],'COMMON_MODEL_PROMPT_FREEZE_MISMATCH');verify_bindings(summary)
                if role!='B1':need(summary['identity']['adapter']==raw_results[role[3:]]['adapter'],'EVALUATION_ADAPTER_IDENTITY_MISMATCH')
            for arm in ARMS:need(common['training_results'][arm]==binding(paths(args)[arm.lower()+'_result']),'COMMON_TRAINING_RESULT_BINDING_MISMATCH')
        common_good=attempt_check('common_evaluation',errors,common_check)
    fits={};fit_good=True
    for arm,name in [('GENERIC','generic_fit'),('TARGETED','targeted_fit')]:
        fit=docs[name];good=False
        if fit:
            def fit_check():
                need(fit['status']=='COMPLETED' and fit['scope']=='TRAIN_FIT_SEPARATE_FROM_DEV','TRAIN_FIT_MISSING_OR_MIXED_WITH_DEV');verify_bindings(fit)
                need(fit['identity']['training_result']==binding(paths(args)[arm.lower()+'_result']),'TRAIN_FIT_ENDPOINT_MISMATCH')
            good=attempt_check(f'train_fit:{arm}',errors,fit_check)
        fits[arm]={'status':'COMPLETED' if good else 'NOT_COMPLETED','summary_path':evidence(paths(args)[name]),'metrics':fit.get('metrics') if fit else None};fit_good &= good
    source_good=source['completed_sources']==16 and source['accepted_scenes']==96 and integrity_good and integrity_complete
    review_good=export_good and questions['reviewed_units']==16 and questions['independent_review_rows']==112 and questions['source_or_gold_ambiguity_failures']==0 and gate_good and integrity_good
    phases={
        PHASES[0]:{'status':'COMPLETED' if protocol_good else 'BLOCKED','evidence_paths':[evidence(paths(args)['protocol']),evidence(BASE/'PRODUCTION_INPUT_FREEZE_V1.json')]},
        PHASES[1]:{'status':'COMPLETED' if source_good else 'PARTIAL','evidence_paths':[evidence(paths(args)['integrity']),evidence(BASE/'inputs/PRODUCTION_PREPARATION_RECEIPT.json')],'completed_sources':source['completed_sources'],'planned_sources':16,'completed_scenes':source['accepted_scenes'],'planned_scenes':96},
        PHASES[2]:{'status':'COMPLETED' if review_good else 'PARTIAL','evidence_paths':[evidence(paths(args)['gate']),evidence(paths(args)['question_freeze']),evidence(paths(args)['export'])],'admitted':questions['admitted'],'planned_slots':112},
        PHASES[3]:{'status':'COMPLETED' if pool_good else 'BLOCKED','evidence_paths':[evidence(paths(args)['pool']),evidence(paths(args)['schedule'])],'scope':pool_mode},
        PHASES[4]:{'status':'COMPLETED' if paired else 'PARTIAL','evidence_paths':[training[a]['result_path'] for a in ARMS],'scope':pool_mode},
        PHASES[5]:{'status':'COMPLETED' if common_good and fit_good else 'PARTIAL','evidence_paths':[evidence(paths(args)['common'])]+[fits[a]['summary_path'] for a in ARMS],
            'closeout_audit':'Required after this report is frozen; final manifest will reject COMPLETED without a PASS bound to this exact report hash.'}}
    main_complete=all(p['status']=='COMPLETED' for p in phases.values()) and pool_mode!='QFP2_ONLY_PILOT_NEW_ROOT_GATE_FAILED' and not errors and expansion_stop is None
    overall='COMPLETED' if main_complete else 'PARTIAL' if any(p['status']=='COMPLETED' for p in phases.values()) else 'BLOCKED'
    if args.status_override:overall=args.status_override;main_complete=False
    blockers=[]
    if expansion_stop:blockers.append('真实来源语义失败已触发不可变扩批停止凭据；后四root不再自动生产，新增root端到端主目标未完成。')
    if not gate_good:blockers.append('前四新root工程门未通过或尚无可验证冻结；不得自动扩到后四。')
    if not source_good:blockers.append(f"本次新root完整来源仅{source['completed_sources']}/16，验收场景{source['accepted_scenes']}/96。")
    if not review_good:blockers.append(f"本次题目独立审核/导出未完整满足主目标：准入{questions['admitted']}/112，独立审核行{questions['independent_review_rows']}/112。")
    if not pool_good:blockers.append('尚无通过校验的正式匹配训练池。')
    if not paired:blockers.append('两臂同起点固定终点训练及每20步真实Dev证据未全部通过校验。')
    if not common_good:blockers.append('尚无通过校验的B1、B1_GENERIC、B1_TARGETED同题共同最终评估。')
    if not fit_good:blockers.append('两臂训练拟合诊断尚未完整完成或未与Dev分开。')
    if errors:blockers.append('存在文件/绑定/预算验证错误；详情见validation_errors。')
    conclusions=['本轮仅为单seed探索性配对实验；反馈为经核验Train卡片的跨root抽象目标迁移，不是已经证实的正式模型弱点。',
        '不同于前小试的新扩充roots均曾被M-clean历史看到；共同Dev为历史暴露回归，不支持未见root泛化或因果收益主张。',
        'MC2–5、排序与MC6控制分开报告；MC6排除主训练和主收益。训练拟合分数不证明泛化，可见解释不等同忠实内部CoT。']
    if pool_mode=='QFP2_ONLY_PILOT_NEW_ROOT_GATE_FAILED':conclusions.insert(0,'新root工程门未通过，本次训练即便完成也仅为QFP2历史数据pilot；新增root端到端主目标未完成。')
    report={'schema_version':'oet1_final_delivery_report_v1','created_at_utc':now(),'overall_status':overall,'objective_completed':main_complete,
        'phases':phases,'planned':{'roots':8,'sources':16,'scenes':96,'question_slots':112},'source_production':source,'question_production':questions,
        'engineering_gate':{'status':gate.get('status') if gate else 'MISSING','path':evidence(paths(args)['gate']),'summary':{k:gate[k] for k in ['checks','overall','by_root','by_slot','question_ambiguity_failures','source_ambiguity_bindings'] if gate and k in gate}},
        'expansion_stop':{'triggered':expansion_stop is not None,'receipt_path':evidence(paths(args)['expansion_stop']),'receipt_binding':binding(paths(args)['expansion_stop']) if expansion_stop else None,'reason':expansion_stop},
        'source_prefix_metadata_diagnostic':{'path':evidence(paths(args)['prefix_diagnostic']),
            'failure_categories':docs['prefix_diagnostic'].get('failure_categories') if docs['prefix_diagnostic'] else None,
            'all_prior_turn_objects_preserved':docs['prefix_diagnostic'].get('all_observed_prior_turn_objects_exactly_preserved') if docs['prefix_diagnostic'] else None,
            'original_fail_hold_decisions_changed':False,'failed_sources_admitted':False},
        'training_pool':{'mode':pool_mode,'freeze_path':evidence(paths(args)['pool']),'schedule_path':evidence(paths(args)['schedule']),'question_counts':pool_counts,'historical_exposure_layers':pool_layers,
            'input_token_relative_difference':schedule.get('input_token_relative_difference') if schedule else None,'formal_token_fairness_within_5pct':schedule.get('formal_token_fairness_within_5pct') if schedule else None},
        'training_arms':training,'common_evaluation':{'status':'COMPLETED' if common_good else 'NOT_COMPLETED','summary_path':evidence(paths(args)['common']),'models':MODELS,
            'metrics':common.get('metrics') if common_good else None,'paired_by_root':common.get('paired_by_root') if common_good else None,'descriptive_deltas':common.get('descriptive_deltas') if common_good else None,
            'planned_question_count':evaluation.get('question_count') if evaluation else None,'planned_requests_per_model':evaluation.get('final_request_count') if evaluation else None,'actual_models_verified':MODELS if common_good else []},
        'training_fit_separate':fits,'validation_errors':errors,'blockers':blockers,'conclusion_boundaries':conclusions,
        'remaining_work':blockers,'closeout_audit':{'required':True,'path':str(BASE/'INDEPENDENT_CLOSEOUT_AUDIT.json'),'report_hash_must_match':True,'audited_report_must_not_change':True},
        'evidence_paths':{k:evidence(p) for k,p in paths(args).items()},'builder_binding':binding(Path(__file__)),
        'report_completion_scope':'COMPLETED is a candidate final result until the separate independent closeout audit PASS and exact report hash are included by the final manifest.'}
    return report

def chinese_report(report):
    q=report['question_production'];s=report['source_production'];lines=[f"本轮状态：**{report['overall_status']}**；新增root端到端目标完成：{'是' if report['objective_completed'] else '否'}。\n",
        f"计划8个不同于前小试的Train roots、16份配对来源、96场、112个题槽。实际完整来源{s['completed_sources']}/16，验收场景{s['accepted_scenes']}/96；题目准入{q['admitted']}/112，失败{q['failed']}，SHORTFALL {q['shortfall']}，其他/未完整终态{q['other_terminal_or_missing']}。首次分母保留，未按solver成绩补题。\n",
        f"前四root工程门：{report['engineering_gate']['status']}。训练池范围：{report['training_pool']['mode'] or '尚未冻结'}；每臂题量：{json.dumps(report['training_pool']['question_counts'],ensure_ascii=False)}。\n"]
    for text in report['conclusion_boundaries']:lines.append(text+'\n')
    categories=report['source_production']['review_failure_category_counts']
    if categories:
        lines.append('来源非READY结果分类：`'+json.dumps(categories,ensure_ascii=False)+'`。内容时间绑定失败与包装计数/HOLD分别统计；即使独立机械诊断证明原文prefix完整，也保留原审核FAIL/HOLD与来源停止后继，不救成准入。\n')
    lines.append('各阶段实际状态：\n')
    lines.extend(f"- {key}：{report['phases'][key]['status']}\n" for key in PHASES)
    lines.append('\n两臂模型与训练证据：\n')
    for arm in ARMS:
        row=report['training_arms'][arm]
        lines.append(f"- {arm}：{row['status']}；固定优化器步数 {row.get('completed_optimizer_steps')} / {row.get('fixed_final_optimizer_step')}；adapter `{row['adapter_path']}`；结果 `{row['result_path']}`；Dev记录步 {row.get('dev_steps')}。\n")
    metrics=report['common_evaluation']['metrics']
    if metrics:
        lines.append('\n共同Dev各模型指标如下；MC6只作独立控制，历史其他Gold数类型保留在机器报告。\n\n| 模型 | MC2–5 exact | MC2–5 partial | 排序 exact | 排序 pairwise | MC6 exact | 展示一致性 |\n|---|---:|---:|---:|---:|---:|---:|\n')
        def val(value):return 'NA' if value is None else f'{value:.4f}'
        for model in MODELS:
            m=metrics[model];t=m.get('by_task',{});mc=t.get('MC2_5_PRIMARY',{});order=t.get('ORDERING',{});ctrl=t.get('MC6_CONTROL',{})
            lines.append(f"| {model} | {val(mc.get('exact'))} | {val(mc.get('partial'))} | {val(order.get('exact'))} | {val(order.get('pairwise'))} | {val(ctrl.get('exact'))} | {val(m.get('display_consistency',{}).get('rate'))} |\n")
        lines.append('\n描述性差值：`'+json.dumps(report['common_evaluation']['descriptive_deltas'],ensure_ascii=False)+'`。没有明显收益或证据不足均为允许结论；不得据此追加训练或改测试范围。\n')
    else:lines.append('\n共同最终评估未完成；没有可报告的三模型同题收益指标。\n')
    lines.append('\n训练拟合与Dev结果分别保存在机器报告；新题与QFP2历史层按真实cohort分开，缺失层不补造分母。W&B状态以各臂RESULT为准，offline或SDK失败均不声称同步。\n')
    if report['blockers']:
        lines.append('\n未完成项与可靠下一步：\n');lines.extend('- '+item+'\n' for item in report['blockers'])
    lines.append('\n完整路径、按root/arm/slot计数、失败和绑定错误见同目录 FINAL_REPORT.json。此报告冻结后由全新独立只读CLI审查；最终manifest必须绑定审计与本报告真实hash，审计后不得再改报告。\n')
    return '\n'.join(lines)

def paths(args):
    return {'protocol':Path(args.protocol),'gate':Path(args.gate),'integrity':Path(args.production_integrity),
        'question_freeze':Path(args.question_freeze),'export':Path(args.export_freeze),'pool':Path(args.pool_freeze),
        'schedule':Path(args.schedule_freeze),'evaluation':Path(args.evaluation_freeze),
        'generic_result':Path(args.generic_result),'targeted_result':Path(args.targeted_result),
        'common':Path(args.common_summary),'generic_fit':Path(args.generic_fit),'targeted_fit':Path(args.targeted_fit),
        'expansion_stop':Path(args.expansion_stop_receipt),'prefix_diagnostic':Path(args.prefix_diagnostic)}

def live_excluded(path):
    p=Path(path);relative=p.relative_to(BASE)
    return bool(set(relative.parts)&EXCLUDE_PARTS) or p.name in LIVE_NAMES or (p.name.startswith('PRODUCTION_DRIVER') and p.suffix=='.log') or p.name.endswith(('.pid','.lock','.tmp')) or '.tmp.' in p.name or '.pending.' in str(relative)

def assert_no_live_data_process():
    for start in (BASE/'cli_calls').glob('*/attempt_*/START.json'):
        if start.with_name('TERMINAL.json').exists():continue
        pp=start.with_name('PID.json')
        if not pp.exists():continue
        pid=read(pp).get('pid')
        if pid:
            try:os.kill(pid,0)
            except ProcessLookupError:continue
            except PermissionError:pass
            raise RuntimeError(f'LIVE_DATA_CLI_CANNOT_FREEZE_DELIVERY:{pid}:{start.parent}')

def selected_artifacts(args, *,include_audit):
    assert_no_live_data_process();selected={}
    for p in sorted(BASE.rglob('*')):
        if not p.is_file() or p.is_symlink() or live_excluded(p) or p.name in FINAL_NAMES:continue
        if not include_audit and (p.name=='INDEPENDENT_CLOSEOUT_AUDIT.json' or any('closeout' in part.lower() for part in p.relative_to(BASE).parts)):continue
        selected[str(p.resolve())]=binding(p)
    # Known aggregate documents carry original external model/data/code bindings.
    # Do not parse any referenced item-level file to discover further contents.
    aggregate_paths={str(p.resolve()):p for p in paths(args).values()}
    # A report frozen with explicit nondefault inputs remains sufficient for
    # manifest discovery even if the later command omits those same overrides.
    report_path=BASE/'FINAL_REPORT.json'
    if report_path.is_file():
        for value in read(report_path).get('evidence_paths',{}).values():
            if value:aggregate_paths[str(Path(value).resolve())]=Path(value)
    for p in aggregate_paths.values():
        if not p.is_file():continue
        selected[str(p.resolve())]=binding(p)
        for b in bindings(read(p)):
            target=Path(b['path']).resolve()
            if target==BASE/'FINAL_ARTIFACT_MANIFEST.json':continue
            if target.is_relative_to(BASE) and live_excluded(target):continue
            verify(b);selected[str(target)]=b
    for value in args.extra_evidence:
        p=Path(value).resolve();need(p.is_file(),'EXTRA_EVIDENCE_MUST_BE_REAL_FILE:'+str(p));selected[str(p)]=binding(p)
    return [selected[key] for key in sorted(selected)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['report','audit-inventory','manifest'],required=True)
    defaults={'protocol':'PROTOCOL_FREEZE_V1.json','gate':'FIRST_WAVE_ENGINEERING_GATE.json','production_integrity':'PRODUCTION_ENGINEERING_AUDIT_FINAL.json',
        'question_freeze':'QUESTION_FREEZE.json','export_freeze':'inputs/EXPORT_FREEZE.json','pool_freeze':'training_pool/TRAINING_POOL_FREEZE.json',
        'schedule_freeze':'training/schedule/SCHEDULE_FREEZE.json','evaluation_freeze':'training/evaluation/EVALUATION_FREEZE.json',
        'generic_result':'training/runs/GENERIC/RESULT.json','targeted_result':'training/runs/TARGETED/RESULT.json',
        'common_summary':'training/COMMON_EVALUATION_SUMMARY.json','generic_fit':'training/train_fit/GENERIC/SUMMARY.json','targeted_fit':'training/train_fit/TARGETED/SUMMARY.json',
        'expansion_stop_receipt':'EXPANSION_STOP_RECEIPT.json','prefix_diagnostic':'SOURCE_PREFIX_METADATA_DIAGNOSTIC_V1.json'}
    for name,value in defaults.items():p.add_argument('--'+name.replace('_','-'),default=str(BASE/value))
    p.add_argument('--status-override',choices=['PARTIAL','BLOCKED']);p.add_argument('--extra-evidence',action='append',default=[]);args=p.parse_args()
    if args.action=='report':
        need(not (BASE/'FINAL_REPORT.json').exists() and not (BASE/'FINAL_REPORT_ZH.md').exists(),'REPORT_ALREADY_SEALED_PRESERVE_OLD_BYTES_AND_REAUDIT_ANY_NEW_VERSION')
        report=build_report(args);write(BASE/'FINAL_REPORT.json',report);write_text(BASE/'FINAL_REPORT_ZH.md',chinese_report(report))
        print(json.dumps({'overall_status':report['overall_status'],'objective_completed':report['objective_completed'],'report':binding(BASE/'FINAL_REPORT.json'),'zh_report':binding(BASE/'FINAL_REPORT_ZH.md')},ensure_ascii=False))
    elif args.action=='audit-inventory':
        report=read(BASE/'FINAL_REPORT.json');need((BASE/'FINAL_REPORT_ZH.md').is_file(),'CHINESE_REPORT_MISSING')
        artifacts=selected_artifacts(args,include_audit=False)
        write(BASE/'ARTIFACTS_FOR_AUDIT.json',{'schema_version':'oet1_immutable_closeout_audit_inventory_v1','created_at_utc':now(),
            'report_sha256':sha(BASE/'FINAL_REPORT.json'),'overall_status':report['overall_status'],'artifacts':artifacts,
            'excluded_live_files':sorted(LIVE_NAMES),'excluded_live_directories':sorted(EXCLUDE_PARTS),
            'independent_closeout_session_must_use_separate_directory':True,'item_level_evaluation_files_parsed':False})
        print(json.dumps({'inventory':binding(BASE/'ARTIFACTS_FOR_AUDIT.json'),'artifacts':len(artifacts)},ensure_ascii=False))
    else:
        report=read(BASE/'FINAL_REPORT.json');audit=read(BASE/'INDEPENDENT_CLOSEOUT_AUDIT.json');inventory=read(BASE/'ARTIFACTS_FOR_AUDIT.json')
        need(audit.get('verdict') in ['PASS','FAIL','HOLD'] and audit.get('report_sha256')==sha(BASE/'FINAL_REPORT.json'),'CLOSEOUT_AUDIT_NOT_BOUND_TO_EXACT_REPORT')
        need(inventory['report_sha256']==sha(BASE/'FINAL_REPORT.json'),'AUDITED_INVENTORY_REPORT_CHANGED');verify_bindings(inventory)
        if report['overall_status']=='COMPLETED':
            need(report['objective_completed'] is True and all(report['phases'][phase]['status']=='COMPLETED' for phase in PHASES),'COMPLETED_PHASES_OR_OBJECTIVE_MISMATCH')
            need(audit['verdict']=='PASS','FAILED_CLOSEOUT_REQUIRES_PARTIAL_OR_BLOCKED_REPORT_AND_NEW_AUDIT')
            need(report['common_evaluation']['models']==MODELS,'COMMON_MODEL_ORDER_MISMATCH')
            for arm in ARMS:
                need(Path(report['training_arms'][arm]['adapter_path']).is_file() and Path(report['training_arms'][arm]['adapter_path']).stat().st_size>=100000,'REAL_ADAPTER_REQUIRED_FOR_COMPLETED')
        artifacts=selected_artifacts(args,include_audit=True);index={a['path']:a for a in artifacts}
        for value in inventory['artifacts']:need(index.get(str(Path(value['path']).resolve()))==value,'FROZEN_AUDIT_ARTIFACT_CHANGED_OR_OMITTED:'+value['path'])
        index[str((BASE/'ARTIFACTS_FOR_AUDIT.json').resolve())]=binding(BASE/'ARTIFACTS_FOR_AUDIT.json')
        required=[BASE/'FINAL_REPORT.json',BASE/'FINAL_REPORT_ZH.md',BASE/'INDEPENDENT_CLOSEOUT_AUDIT.json']
        for arm in ARMS:
            required.extend(Path(report['training_arms'][arm][key]) for key in ['adapter_path','result_path'] if report['training_arms'][arm].get(key))
        if report['common_evaluation'].get('summary_path'):required.append(Path(report['common_evaluation']['summary_path']))
        need(all(str(path.resolve()) in index for path in required),'REQUIRED_REAL_DELIVERY_ARTIFACT_OMITTED')
        write(BASE/'FINAL_ARTIFACT_MANIFEST.json',{'schema_version':'oet1_final_artifact_manifest_v1','created_at_utc':now(),
            'overall_status':report['overall_status'],'report_sha256':sha(BASE/'FINAL_REPORT.json'),'independent_closeout_verdict':audit['verdict'],
            'artifacts':[index[key] for key in sorted(index)],'self_included':False,'live_files_included':False,
            'reconstruction':'Existing B1 merged checkpoint plus each new durable custom adapter; all new resume state, logs, data, code and raw evidence are bound without copying old models.'})
        print(json.dumps({'manifest':binding(BASE/'FINAL_ARTIFACT_MANIFEST.json'),'overall_status':report['overall_status'],'artifacts':len(index)},ensure_ascii=False))
if __name__=='__main__':main()
