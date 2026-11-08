"""Close the preparation milestone only; never marks actual evaluation ready."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib
import json

HERE=Path(__file__).resolve().parent
WEEK=HERE.parent

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
    p=Path(p).resolve()
    return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def save(p,obj):
    p=Path(p)
    if p.exists():
        history=HERE/'history'/(p.name+'.'+sha(p)[:16])
        if not history.exists(): history.write_bytes(p.read_bytes())
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    now=datetime.now(timezone.utc).isoformat()
    status=read(HERE/'PREPARATION_STATUS.json')
    require(status['reviewed_roots']==31 and status['reviewed_slots']==186,'ALL31_SOURCE_REVIEWS_REQUIRED')
    require(status['structural_consistency_error_count']==0,'REVIEW_STRUCTURAL_ERRORS')
    require(all(r['explicit_report_nonreport_partition_records']==r['inventory_records'] for r in status['root_reviews']),'REPORT_NONREPORT_PARTITIONS_REQUIRED')
    for split in ['dev','test']:
        running=read(HERE/'source_admission'/split/'RUNNING.json')
        require(running.get('status') in ['COMPLETE','COMPLETED','COMPLETE_MATERIAL_REVIEW_ONLY','COMPLETE_SOURCE_MATERIAL_REVIEW_ONLY','COMPLETE_MATERIAL_ADMISSION_PENDING_EXCLUSION','COMPLETE_SOURCE_REVIEW_EXCLUSION_PENDING','COMPLETE_SOURCE_FIRST_MATERIAL_REVIEW_ONLY','COMPLETE_SOURCE_SEMANTIC_REVIEW_ONLY','COMPLETE_MATERIAL_REVIEW_EXCLUSION_PENDING','MATERIAL_REVIEW_COMPLETE_FAMILY_EXCLUSION_PENDING'],'SOURCE_REVIEW_STILL_RUNNING:'+split+':'+str(running.get('status')))
    preflight=read(HERE/'audit/SOURCE_AND_MAPPING_PREFLIGHT.json')
    require(preflight['status']=='STRUCTURAL_AND_SYNTHETIC_MAPPING_PASS_ONLY','CPU_PREFLIGHT_FAILED')
    require(len(preflight['sources'])==31 and len(preflight['structural_shortfall_slots'])==13,'PREFLIGHT_COVERAGE')
    require(sha(HERE/'prepare_source_preflight.py')==preflight['script_binding']['sha256'],'PREFLIGHT_SCRIPT_CHANGED')
    boundary=read(HERE/'boundary/OUTPUT_BINDINGS.json')
    for key in ['manifest','index','coverage','heldout_root_local_gaps']:
        row=boundary[key]; require(sha(row['path'])==row['sha256'],'BOUNDARY_OUTPUT_CHANGED:'+key)
    coverage=read(HERE/'boundary/COVERAGE.json')
    require(not coverage['missing_inputs_and_unresolved_references'],'BOUNDARY_INPUT_GAPS')
    local=read(HERE/'boundary/ROOT_LOCAL_CLEARANCE_GAPS.json')
    independent=read(HERE/'audit/INDEPENDENT_PREPARATION_AUDIT.json')
    require(independent.get('status') in ['PASS_WITH_LIMITS','PREPARATION_PASS_WITH_LIMITS','PASS_PREPARATION_ONLY'],'INDEPENDENT_PREPARATION_AUDIT_NOT_PASS')
    require(not any(f.get('status') == 'OPEN' for f in independent.get('findings',[])), 'INDEPENDENT_FINDINGS_STILL_OPEN')
    for row in independent.get('input_bindings',[]):
        require(sha(row['path'])==row['sha256'], 'AUDIT_INPUT_CHANGED:'+row['path'])
    release_validation=read(HERE/'author_release/INTERNAL_RELEASE_VALIDATION.json')
    require(release_validation.get('status') in ['PASS','PASS_RELEASE_SANITIZATION_ONLY','PASS_WITH_LIMITS','PASS_EXPORT_SANITIZATION_AND_INTEGRITY_ONLY'], 'SAFE_AUTHOR_RELEASE_VALIDATION_REQUIRED')
    require(release_validation['record_count']==4629 and release_validation['immutable_semantic_fields_equal_all_rows'] and release_validation['raw_bindings_unchanged'], 'RELEASE_INTEGRITY_REQUIRED')
    require(release_validation['internal_path_leaks']==0 and release_validation['performance_or_selection_metadata_leaks']==0, 'RELEASE_METADATA_LEAK')
    for name, row in release_validation['outputs'].items():
        target=HERE/'author_release'/name
        require(sha(target)==row['sha256'] and target.stat().st_size==row['bytes'], 'SAFE_RELEASE_CHANGED:'+name)
    by_type={q:dict(Counter(r['status'] for r in status['slot_ledger'] if r['question_type']==q)) for q in ['multi_select','ordering']}
    gates={
      'historical_input_discovery_and_no_score_extraction':'COMPLETE_WITH_EXPLICIT_MAPPING_GAPS',
      'full_source_material_review_31_roots_186_slots':'COMPLETE_REVIEWER_ASSESSMENT_NOT_CONTENT_ADMISSION',
      'source_hash_structure_and_synthetic_display_checks':'PASS_SCOPED_CPU_CHECKS_ONLY',
      'source_fact_semantic_normalization_and_old_format_reconstruction':'PENDING',
      'five_category_per_candidate_family_exclusion':'PENDING',
      'eligible_family_inventory_and_deterministic_selection':'PENDING',
      'actual_public_questions_private_Gold_independent_QA':'PENDING',
      'actual_full_prompt_tokenization_and_request_model_freeze':'PENDING',
      'day4_symmetric_training_pool_and_resource_freeze':'PENDING'}
    summary={
      'schema_version':'common_eval_preparation_milestone_v1','created_at_utc':now,
      'status':'PREPARATION_MILESTONE_COMPLETE_SEMANTIC_CLEARANCE_AND_CONTENT_FREEZE_PENDING',
      'roots':31,'abstract_slots':186,'material_decisions':status['material_decisions'],
      'material_decisions_by_type':by_type,
      'inventory_records':sum(r['inventory_records'] for r in status['root_reviews']),
      'structural_shortfall_slots':13,'source_text_token_range_without_question_or_template':[min(r['source_text_tokens_without_question_or_template'] for r in preflight['sources']),max(r['source_text_tokens_without_question_or_template'] for r in preflight['sources'])],
      'synthetic_mapping_roundtrips':sum(r['answer_roundtrip_checks'] for r in preflight['mapping_checks']),
      'exclusion_index':{'records':coverage['unique_export_records'],
          'source_versions':coverage['source_versions_verified'],'roots':coverage['roots'],
          'global_unique_unresolved_records':local['global_unique_unresolved_records'],
          'heldout_records':sum(r['unique_candidate_source_family_records'] for r in local['roots']),
          'heldout_exact_support_mapped':sum(r['exact_support_mapped_records'] for r in local['roots']),
          'heldout_legacy_mapping_unresolved':sum(r['unresolved_records'] for r in local['roots']),
          'semantic_near_duplicate_clearance':'PENDING_FOR_ALL_CANDIDATES'},
      'gates':gates,'full_evaluation_ready':False,'training_start_go':False,
      'actual_new_evaluation_questions':0,'actual_evaluation_requests':0,
      'new_solver_calls':0,'new_training_runs':0,
      'next_action':'Resolve source-grounded semantic fact mappings, prioritize31 heldout old-format gaps, then perform all five exclusions before eligible family selection, question generation and independent source-first QA.',
      'constraints':['Do not rerun Day2 diagnostic39 or regenerate sealed Train production.',
          'Existing Train187 accepted candidates are unchanged and are not yet frozen training pools.',
          'Explicitly unsupported slots remain shortages; ambiguous feasibility remains HOLD.',
          'Historical exposure and semantic overlap are not cleared by root separation or hashes.',
          'No new-root, difficulty gain, feedback utility or training gain claim is supported here.'],
      'bindings':{'preparation_status':bind(HERE/'PREPARATION_STATUS.json'),
          'source_mapping_preflight':bind(HERE/'audit/SOURCE_AND_MAPPING_PREFLIGHT.json'),
          'boundary_outputs':bind(HERE/'boundary/OUTPUT_BINDINGS.json'),
          'independent_audit':bind(HERE/'audit/INDEPENDENT_PREPARATION_AUDIT.json'),
          'prior_seals_and_late_audit':bind(HERE/'audit/PRIOR_SEALS_AND_LATE_AUDIT_ADDENDUM.json'),
          'request_policy':bind(HERE/'protocol/REQUEST_POLICY_V1.json'),
          'content_contract':bind(HERE/'protocol/CONTENT_RECORD_CONTRACT.json'),
          'author_release_manifest':bind(HERE/'author_release/EXCLUSION_MANIFEST.json'),
          'author_release_index':bind(HERE/'author_release/EXCLUSION_INDEX.jsonl'),
          'author_release_validation':bind(HERE/'author_release/INTERNAL_RELEASE_VALIDATION.json'),
          'audit_repair_addendum':bind(HERE/'protocol/AUDIT_REPAIR_ADDENDUM.json'),
          'historical_schema_provenance_disposition':bind(HERE/'audit/HISTORICAL_SCHEMA_REFERENCE_PROVENANCE.json')}}
    save(HERE/'SUMMARY.json',summary)
    counts=status['material_decisions']; ex=summary['exclusion_index']
    report=f'''# 第三天共同评估：准备阶段结果

完成的是来源审核与无成绩索引准备，不是实际评估题/Gold/request冻结；不能据此启动第四天训练。

## 已完成

- 完整阅读31个固定DEV/TEST来源的parent+R01，审核186个原定题位；记录{summary['inventory_records']}条事实库存。
- 材料可支持待排除：{counts.get('MATERIAL_SUPPORTED_PENDING_EXCLUSION',0)}；材料歧义待核：{counts.get('MATERIAL_AMBIGUOUS_HOLD',0)}；来源缺额：{counts.get('SOURCE_SHORTFALL',0)}。这些不是接受题数。
- 31份来源SHA/结构通过；13题位显式scene数不足。仅Dialogue块长度3268—4876 token，尚未包含题目和chat模板，不能冒充完整request零截断通过。
- 动态4/5/6项六种题型组合完成{summary['synthetic_mapping_roundtrips']}次答案映射还原和有限语法空间核查；检查使用合成内容，新增评估题为0。
- 历史索引{ex['records']}条，覆盖101root/197源版本。31heldout有{ex['heldout_records']}条：{ex['heldout_exact_support_mapped']}条显式支持映射，{ex['heldout_legacy_mapping_unresolved']}条旧格式待语义复原。映射不等于语义排重PASS。
- 原两批3058个封存条目SHA一致；晚到审计追加绑定，A148314歧义继续HOLD；无旧封存改写。

## 本次修复

已修正public字段为options/items二选一；原boundary manifest继续内部保留，作者只可读取author_release安全副本，输入绑定去除内部路径及表现编码，并重绑定索引。根/整源历史曝光与实际被考事实族分开；不把全源每turn都当已考题。Gold数量不自动对应同样数量的独立修订链。parent引用保留完整turn_ref，结构scene证据与事实turn证据分开。首次报告与疑问提及分开，保留逐版审核记录。

历史溯源限制：TEST记录曾读取的3958字节schema helper已被后续版本覆盖，原实际读取hash保持不变，未找到可重放快照。该项作为显式溯源限制保留；不声称TEST读取后来包含DEV来源信息的脚本，不重新推定语义审核完成。

## 尚未完成

先补必要事实规范化和旧格式支持映射，再逐候选完成五类语义排除及同义/修订检查；之后按预定hash规则选合格族、写题和私有Gold、独立source-first重建、核完整prompt长度并冻结请求和模型身份。当前尚无任何实际新评估题、Solver调用或新训练。

所有root仍为历史REGRESSION_ONLY；不声称新root泛化、难度提升、反馈收益或训练收益。TEST只在最终模型冻结后评估，DEV按冻结政策作汇总报告，checkpoint使用预定训练终点。

主要入口：SUMMARY.json、PREPARATION_STATUS.json、boundary/ROOT_LOCAL_CLEARANCE_GAPS.json（内部审核）、author_release/EXCLUSION_INDEX.jsonl和EXCLUSION_MANIFEST.json（无成绩作者发布包；尚未交付）、source_admission/dev和test逐root记录。
'''
    (HERE/'REPORT_ZH.md').write_text(report)
    # Update only live navigation pointers, preserving their previous bytes.
    for name in ['CURRENT_STATUS.json','DAY3_LATEST.json']:
        path=WEEK/name; current=read(path); old=sha(path)
        hist=HERE/'history'/(name+'.BEFORE_COMMON_EVAL_PREPARATION.'+old[:16])
        if not hist.exists(): hist.write_bytes(path.read_bytes())
        current['updated_at_utc']=now
        if 'updated_at_beijing' in current:
            current['updated_at_beijing']=datetime.fromisoformat(now).astimezone(timezone(timedelta(hours=8))).isoformat()
        current['status']='DAY3_COMMON_EVAL_PREPARATION_COMPLETE_SEMANTIC_CONTENT_FREEZE_PENDING'
        current['common_evaluation_preparation']={'summary':bind(HERE/'SUMMARY.json'),'report':bind(HERE/'REPORT_ZH.md'),
             'reviewed_roots':31,'reviewed_slots':186,'actual_questions_frozen':0,
             'fact_family_semantic_clearance':'PENDING','training_start_go':False}
        current['next_action']=summary['next_action']
        if name=='CURRENT_STATUS.json':
            current.setdefault('daily_planning',{})['day3']='TRAIN_PRODUCTION_COMPLETE_COMMON_EVAL_PREPARATION_COMPLETE_SEMANTIC_CONTENT_FREEZE_PENDING'
        path.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','roots','abstract_slots','material_decisions','exclusion_index','full_evaluation_ready','training_start_go']},ensure_ascii=False))

if __name__=='__main__':main()
