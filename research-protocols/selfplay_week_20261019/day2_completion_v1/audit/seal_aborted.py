"""Build and seal this infrastructure-aborted attempt; never declares Day2 complete."""
from pathlib import Path
import argparse,datetime,hashlib,json
WORK=Path(__file__).resolve().parent.parent;MAN=WORK/'COMPLETE_MANIFEST.json'
def load(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def save(p,value):
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def entries():return [{'path':str(p.relative_to(WORK)),'sha256':bind(p)['sha256'],'bytes':p.stat().st_size} for p in sorted(WORK.rglob('*')) if p.is_file() and p!=MAN and '__pycache__' not in p.parts and p.suffix!='.pyc']
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['seal','verify']);args=parser.parse_args()
    if args.mode=='seal':
        audit=load(WORK/'audit/RESULTS_AUDIT.json');scores=load(WORK/'pilot/SCORES.json');assert audit['status']=='PASS'
        fail=audit['operational_failure_analysis'];now=datetime.datetime.now(datetime.timezone.utc).isoformat()
        started=load(WORK/'pilot/STARTED.json')['started_at_utc'];finished=load(WORK/'pilot/COMPLETE.json')['finished_at_utc'];makespan=(datetime.datetime.fromisoformat(finished)-datetime.datetime.fromisoformat(started)).total_seconds()
        costs={a:{k:scores['arms'][a]['api_cost'][k] for k in ('api_attempt_count','known_input_tokens','known_output_tokens','known_total_tokens','sum_call_wall_seconds')} for a in scores['arms']}
        status={'schema_version':'day2_infrastructure_aborted_attempt_v1','status':'INFRASTRUCTURE_METADATA_ABORTED','closed_at_utc':now,'attempt_closed':True,'day2_all_goals_complete':False,'memory_pilot_valid_effect_comparison_completed':False,'technical_execution_and_audit_complete':True,'reason':'Frozen client required returned tool_choice none; gateway returned auto/null despite empty requested tools, causing operational stops. No actual tool use observed.','diagnosis_phase':'COMPLETE_WITH_NEGATIVE_AND_LIMITED_RESULTS','confirmed_internal_causes':0,'fact_memory':{'roots':6,'arms':3,'planned_scene_slots':54,'terminal_scene_slots':54,'official_generated_candidates':fail['official_candidate_response_files'],'raw_candidate_json_returns':fail['raw_candidate_json_returned_calls'],'complete_source_review_responses':fail['complete_source_review_response_files'],'accepted_by_arm':{a:x['accepted_initial_scenes'] for a,x in scores['arms'].items()},'effect_verdict':'INCONCLUSIVE_INFRASTRUCTURE_ABORT','cost_status':'NO_COST_CONCLUSION','memory_adopted':False,'main_pipeline_context':'full_history'},'failures':fail,'actual_api_cost':costs,'pilot_makespan_seconds':makespan,'audit':{'status':'PASS','check_count':audit['check_count'],'old_files_verified':audit['old_files_verified'],'binding':bind(WORK/'audit/RESULTS_AUDIT.json')},'new_training_questions':0,'new_training_runs':0,'next_action':'Separate pre-frozen v2 compatibility correction, all original6 roots and3 arms from original prefix, no v1 accepted-prefix continuation or selected retries.','evidence':{k:bind(WORK/p) for k,p in {'scores':'pilot/SCORES.json','freeze':'EXECUTION_FREEZE.json','amendment':'EXECUTION_AMENDMENT.json','diagnosis':'diagnosis/DIAGNOSIS_CLOSEOUT.json'}.items()},'legacy_generic_builders_executed':False}
        save(WORK/'FINAL_STATUS.json',status)
        report=['# 事实记忆试验v1：接口元数据阻断','',
          '**本轮已留痕并封存，但第二天全部目标尚未完成，memory三臂比较仍不可判。** 独立技术审核PASS只说明冻结、隔离输入、54槽位、费用和失败保存可核验，不表示记忆有效或内容质量下降。','',
          '固定6个Train root、3种上下文、每分支3场景。请求始终tools=[]、tool_choice=none；部分网关响应返回tools=null、tool_choice=auto，被事前冻结client拦截。43次API尝试中16次因此失败，覆盖6个root、16个分支；14次导致场景HOLD，另外2次发生在已接受场景后的记忆维护。实际工具活动观察数为0，返回输出只有message/reasoning，tool_usage全0。这是控制字段兼容性异常，不能作为模型弱点或内容错误。','',
          '另有2份recent2审核返回PASS，但给出prefix_quote同时prefix_turn_index=null，缺失配对证据索引，被原定审核格式规则拒绝。它们属于审核记录格式HOLD。没有已正式审核的内容FAIL，也不由此声称内容无问题。','',
          '| 上下文 | 接受/18 | HOLD/18 | 停止后MISSING/18 |','|---|---:|---:|---:|']
        for a in ('recent2','recent2_plus_prefix_memory','full_history'):
            x=scores['arms'][a]['status_counts'];report.append(f"| {a} | {x['PASS']} | {x['HOLD']} | {x['MISSING']} |")
        report+=['',
          '实际保留8份官方候选对话；原始API还返回过被控制字段拦截的候选JSON，共13次，均保留在RAW_RESPONSE。完整来源审核4份，模型均给PASS，但其中2份审核记录格式未达标，因此最终仅2场接受。三个分支方式没有完成可比较的连续3场，表内数字只能描述流程完成率。','',
          f"试验实际起止约{makespan:.2f}秒；并发调用wall之和不能作实验总时长。已知实际provider usage（包括失败尝试与网关指令）：",'',
          '| 上下文 | API尝试 | 输入tokens | 输出tokens | 调用wall总秒 |','|---|---:|---:|---:|---:|']
        for a,c in costs.items():report.append(f"| {a} | {c['api_attempt_count']} | {c['known_input_tokens']} | {c['known_output_tokens']} | {c['sum_call_wall_seconds']:.3f} |")
        report+=['',
          '初始6份内部记忆审核的实际独立模型usage/时延未知。完整成本结论仍为NO_COST_CONCLUSION；主流程保持full_history。本轮不能证明memory提升、降低质量或节省成本。','',
          '独立结果审核4701项PASS，3688个旧封存文件哈希不变。此前诊断与受控检验已可按负结果收口，确认内部原因仍为0；本轮没有训练或新训练题目。','',
          '下一步另建v2并事前冻结兼容性修复：保持请求tools=[]/none，继续阻断非空returned tools、实际工具调用、非零tool_usage、跨响应链和污染；仅不因returned tool_choice=auto本身判实际工具活动。明确审核字段配对要求后，用同一6root×3臂从原始prefix完整重跑。v1失败、费用和全部候选继续保留。','',
          '可复查：[最终状态](FINAL_STATUS.json)、[技术结果审核](audit/RESULTS_AUDIT.json)、[逐槽位评分](pilot/SCORES.json)、[全部原始调用](pilot/branches/)、[已完成诊断](diagnosis/REPORT_ZH.md)。','']
        (WORK/'REPORT_ZH.md').write_text('\n'.join(report))
        (WORK/'README_ZH.md').write_text('# v1接口阻断试验归档\n\n先看[结果说明](REPORT_ZH.md)和[最终状态](FINAL_STATUS.json)。本轮status=INFRASTRUCTURE_METADATA_ABORTED，day2_all_goals_complete=false。\n\n技术审计PASS≠完成有效memory比较。全部54槽位、43次API调用、8份候选、4份完整审核及失败均保留；不采用memory、不作质量或成本收益结论。\n\n本目录按audit/seal_aborted.py封存；原通用build_completion.py/seal_completion.py未执行，它们假定Day2整体完成，不用于本次接口中止结果。验证命令：python -B audit/seal_aborted.py verify。完整文件及SHA256见COMPLETE_MANIFEST.json。\n')
        (WORK/'PROGRESS.json').write_text(json.dumps({'status':'INFRASTRUCTURE_METADATA_ABORTED','updated_at_utc':now,'day2_all_goals_complete':False,'attempt_closed':True,'terminal_scene_slots':54,'accepted_by_arm':status['fact_memory']['accepted_by_arm'],'memory_effect_conclusion':'INCONCLUSIVE','cost_status':'NO_COST_CONCLUSION','main_pipeline_context':'full_history'},ensure_ascii=False,indent=2)+'\n')
        manifest={'schema_version':'day2_aborted_attempt_seal_v1','status':'SEALED_INFRASTRUCTURE_METADATA_ABORTED','created_at_utc':now,'day2_all_goals_complete':False,'exclusions':['COMPLETE_MANIFEST.json self-reference','__pycache__ and .pyc interpreter caches'],'files':entries()}
        save(MAN,manifest)
    manifest=load(MAN);assert manifest['day2_all_goals_complete'] is False and entries()==manifest['files'],'SEALED_BYTES_OR_FILESET_CHANGED'
    print(json.dumps({'status':'PASS','seal_status':manifest['status'],'files':len(manifest['files']),'manifest':bind(MAN)},ensure_ascii=False))
if __name__=='__main__':main()
