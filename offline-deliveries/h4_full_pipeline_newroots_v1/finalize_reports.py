"""Build offline reports from the completed, immutable four-root run; no API calls."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

BASE = Path('/root/autodl-tmp/gxy-selfplay')
RUN = BASE / 'restart-n5-runs/h4_full_pipeline_newroots_v1_live_1'
REPORTS = RUN / 'reports'
DELIVERY = BASE / 'offline-deliveries/h4_full_pipeline_newroots_v1'
PKG = BASE / 'worktrees/h4-full-pipeline-newroots-v1/scripts/current_method/restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1'


def read(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def link(label, path):
    return f'[{label}]({path})'


def main():
    gen = read(RUN / 'generation_summary.json')
    audit = read(REPORTS / 'execution_audit.json')
    costs = read(REPORTS / 'complete_cost_audit.json')
    # These are separate diagnostic artifacts, not new production admission checks.
    diagnosis = read(REPORTS / 'expansion_content_diagnosis.json')
    assert audit['status'] == 'PASS'
    assert gen['cost']['http_attempts'] == 4
    assert gen['cost']['total_tokens'] == 60074
    assert audit['observed_counts']['technical_failures'] == 0
    assert len(gen['root_order']) == 4
    when = datetime.now(timezone.utc).isoformat()
    rows = []
    outputs = [
        '# 四条 Planner 原始输出', '',
        '以下逐字收录实际 response.txt；错误字段和错误 ID 均保留。菜单仅作定位导航，Memory 并非已认证事实。'
        '所有内容仍是计划候选，没有任何一条被写成新 R02 对白。', '',
    ]
    for result in gen['expansion_outcomes']:
        root = result['root_id']
        root_dir = RUN / 'expansion/roots' / root
        call_dir = RUN / 'expansion/execution' / root / 'calls' / root / 'planner'
        output = read(root_dir / 'planner/output.json')
        validation = read(root_dir / 'planner/validation.json')
        menu = read(root_dir / 'event_verifier/materialized_tracks.json')['verified_tracks']
        raw = (call_dir / 'response.txt').read_text()
        rows.append({
            'root_id': root,
            'status': result['status'],
            'last_stage': 'planner_continuation',
            'raw_continuation_candidates': len(output['selected_continuations']),
            'raw_new_development_candidates': len(output['new_development_candidates']),
            'required_new_candidate_count': validation['required_new_candidate_count'],
            'require_full_scene_quota': validation['require_full_scene_quota'],
            'raw_contract_errors': validation['errors'],
            'new_expansion_promoted': False,
            'qa_status': 'NOT_EVALUATED',
            'qa_generated_candidates': {'SC': 0, 'MS': 0, 'ordering': 0},
            'qa_program_accepted': {'SC': 0, 'MS': 0, 'ordering': 0},
            'qa_content_labels': {'PASS': 0, 'FAIL': 0, 'UNDETERMINED': 0},
            'qa_reviewed_candidate_denominator': 0,
            'qa_accuracy': None,
            'qa_deduplicated_supported_yield': {'SC': 0, 'MS': 0, 'ordering': 0},
            'qa_error_counts': None,
            'qa_error_counts_reason': 'NO_QA_OUTPUT_TO_EVALUATE',
            'cost': result['cost'],
        })
        outputs += [f'## {root}', '',
                    link('原始响应', call_dir / 'response.txt') + ' · ' +
                    link('完整请求', call_dir / 'prompt.txt') + ' · ' +
                    link('程序验收', root_dir / 'planner/validation.json'), '',
                    f'原始响应 SHA256：`{sha256((call_dir / "response.txt").read_bytes()).hexdigest()}`', '',
                    '| 可选 ID | 原菜单事件 |', '| --- | --- |']
        for item in menu:
            display = str(item.get('memory_event_display') or item.get('memory_event_key') or '').replace('|', '\\|').replace('\n', ' ')
            outputs.append(f'| {item["verified_track_ref"]} | {display} |')
        outputs += ['', '```json', raw.rstrip('\n'), '```', '']
    (REPORTS / 'ALL_PLANNER_OUTPUTS.md').write_text('\n'.join(outputs))

    summary = {
        'version': 'full_pipeline_newroots_v1_final',
        'status': 'CLOSED_AFTER_OFFLINE_DIAGNOSIS',
        'closed_at': when,
        'execution_status': gen['status'],
        'experiment_result': 'ALL_FOUR_ROOTS_BLOCKED_AT_PLANNER_CONTRACT',
        'diagnosis_method': 'CODEX_OFFLINE_DIAGNOSIS_NOT_INDEPENDENT_HUMAN_CERTIFICATION',
        'run_root': str(RUN),
        'sample': {
            'root_order': gen['root_order'],
            'selection': 'ALL_FOUR_REMAINING_ROOTS_OF_HISTORICAL_SIX_ROOT_ROUTE; NOT_RANDOM',
            'historical_two_roots_excluded_from_current_denominators': True,
            'full_P_R01_turns': 413, 'dated_segments': 32,
            'date_origin': 'SYNTHETIC_ANNOTATION_NOT_RECOVERED_TIMESTAMP',
            'new_R02_generated': False,
        },
        'root_results': rows,
        'blocking_mechanisms': [
            {'kind': 'PROMPT_VALIDATOR_QUOTA_MISMATCH', 'roots': gen['root_order'][1:3],
             'root_count': 2, 'denominator': 4,
             'details': 'require_full_scene_quota=False is ignored by the new_development_slots=None minimum pool-size branch; prompt gives max4, forbids quota padding, and does not request remaining+1 backups.'},
            {'kind': 'OUT_OF_MENU_CONTINUATION_ID', 'roots': [gen['root_order'][0]],
             'root_count': 1, 'denominator': 4,
             'details': 'VT09 and VT10 absent from VT01–VT08 menu; missing memory/state errors cascade from lookup failure.'},
            {'kind': 'MISSING_REQUIRED_BEFORE_STATE', 'roots': [gen['root_order'][3]],
             'root_count': 1, 'denominator': 4, 'affected_candidates': 2,
             'details': 'Each new candidate lacks before_state. Duplicate shape errors are generated at two validator levels; person_ref is not a required model-output field.'},
        ],
        'all_or_nothing_planner_admission': True,
        'semantic_diagnosis': 'COMPLETED_FOR_RAW_PLANNER_CANDIDATES_ONLY',
        'downstream_quality': 'NOT_EVALUATED',
        'stage_calls': audit['stage_calls'],
        'cost': gen['cost'],
        'approved_maximum_http': 56,
        'unused_http_allowance': 52,
        'additional_calls_after_generation': 0,
        'technical_failures': 0,
        'actual_models': audit['actual_model_counts'],
        'finish_reasons': audit['finish_reason_counts'],
        'effective_question_yield': 0,
        'cost_per_effective_question': 'UNDEFINED_ZERO_YIELD',
        'finite_per_question_cost': None,
        'freeze_verification': audit['freeze'],
        'input_verification': audit['input_freeze'],
        'forbidden_stage_calls': audit['forbidden_stage_calls'],
        'unknowns': audit['unknowns'],
        'next_priority': 'Align partial-scene Planner admission with prompt and quarantine local malformed candidates; preserve evidence/identity validation; resolve new-person actor interface before new paid run.',
        'no_production_fix_or_rerun_in_this_report_step': True,
        'report_paths': {name: str(REPORTS / name) for name in [
            'FINAL_REPORT.md', 'FILE_INDEX.md', 'ALL_PLANNER_OUTPUTS.md',
            'expansion_content_diagnosis.md', 'expansion_content_diagnosis.json',
            'execution_audit.json', 'complete_cost_audit.json']},
    }
    write_json(REPORTS / 'FINAL_SUMMARY.json', summary)

    report = f'''# 新增四条完整流程筛查：最终报告

学校模型 `qwen3.6-35b-a3b` 的四条正式请求均正常完成，但四条都在 **Planner 本地合同验收**处停止。没有进入 Gate、Outline、Realizer、Miner 或 Writer，因此本轮测到的是上游准入卡点，**排序、SC/MS 内容质量均未评估**。不能将本轮题目产量为 0 写成 Writer 正确率为 0%。

本批 4 次逻辑调用 = 4 次 HTTP，0 重试、0 技术失败，消耗 60,074 tokens。批准上限为 56，剩余 52 未使用；失败根没有重采样。执行及输入完整性审计 PASS 不代表生成内容 PASS。

## 实际结果与根因

| root | 原始延续候选＋新发展候选 | 首个阻塞 | 判断 |
| --- | ---: | --- | --- |
| 254852 | 2＋2 | 引用 VT09、VT10，菜单仅有 VT01–VT08 | 模型使用菜单外 ID；随后出现的 memory/state 查找失败是级联错误 |
| 136202 | 2＋1 | 程序要求至少 2 个新候选 | 现有 3 个总候选仍被旧备选数量规则拒绝；提示未要求这个最小数 |
| 94289 | 1＋1 | 程序要求至少 3 个新候选 | 当前路线允许不足 3 个场景，却仍受旧候选池最小数约束 |
| 197635 | 2＋2 | 两个新候选都缺 before_state | 必填字段漏填；重复 shape 错误不是多种独立错误 |

共有 13 个计划候选（7 个延续、6 个新发展），完整 Planner 响应接受 0/4，新 R02 晋级 0/4。当前验收在根级整批拒绝：局部候选错误或候选池数量不足会阻止该 root 的全部候选进入 Gate。候选还需内容检查，不能据其他候选结构合格便直接放行。

**可确定的工程问题是候选数量约定不一致。** 当前 profile 设置 `require_full_scene_quota=False`，workflow 也把最少场景数设为 1。然而 `new_development_slots=None` 分支仍执行 `min(3, remaining_quota + 1)`，没有检查该开关。136202 与 94289 的实际 validation 唯一错误均为 `insufficient_new_development_candidates`；只说明这一步的拒绝原因，不能推断移除该检查后整个流程就会成功。

实际提示只规定延续最多 3 个、新候选最多 4 个，并说场景配额归程序处理、不要凑配额，没有声明上述备选最小数。相关位置：{link('contracts.py 数量检查', str(PKG / 'contracts.py') + ':1463')}、{link('workflow.py 最小场景数', str(PKG / 'workflow.py') + ':350')}、{link('workflow.py 根级停止', str(PKG / 'workflow.py') + ':633')}。

197635 的 `before_state` 缺失分别触发两层 shape 检查，再触发 empty_before_state；`new_people` 当前只要求 name 和 relationship_basis，**没有要求模型填写 person_ref**。不能把它误诊成缺人物 ID。254852 的 continuation before_state 则由程序从菜单取回，空值源自非法 VT 引用，不是另一种独立漏填。

## 内容上还能看到什么

这是 Codex 对冻结原文与原始 Planner 候选的离线诊断，没有新增审核 API，也不是独立人工认证。候选中的 after_state 是预期目标；仅因它使用完成时，不能判为虚构已发生事实。需要核对目标、动作和当前状态是否相互一致。

- 197635 的 VT08 实际对应高中聚会，候选却用它绑定工作坊，说明“ID 存在”仍不等于事件身份正确。另一个直接矛盾是动作写“会提前规划路线并准备问题”，after_state 却写“已规划好／已准备好”。这属于同一计划内部的状态不一致，尚未成为新对白或排序题错误。
- 94289 将 James/SP02 放进 speaker_refs，同时在该人物 binding 中写明 James 不在场。新导师需要行动，但当前 actor_ref 接口仍围绕旧人物注册表，后续存在人物绑定风险。
- 136202 新发展以另一候选预计完成的预约为前提；人物列表加入医生，场景动作却主要是 Emily 向 James 转述，需要检查候选依赖和实际参与者。未经 Gate/Realizer，不能声称依赖已满足或新对白已经犯错。
- 部分候选确实提出预约、发布邀请、开始写日记等具体变化，但本批没有实现这些计划，无法确认实际新增量、重复程度或 QA 可用性。

完整逐候选依据、继承的历史矛盾和风险分类见 {link('Planner 离线诊断', REPORTS / 'expansion_content_diagnosis.md')}。本批为历史六条路线剩余的全部四条便利样本，不是随机抽样；4/4 卡住仅描述这四条当前流程，不可外推全数据集失败率，也不可归因于模型能力一个因素。

## 日期与输入

四条使用完整 P+R01，共 413 条对白、32 段日期头，正文、人物、引用和次序保留。日期标为 `SYNTHETIC_ANNOTATION_NOT_RECOVERED_TIMESTAMP`：是会话日期的合成标注，不是恢复的真实时间，也不等于对白提到的每个事件发生日。已有墙体状态、学校任职状态和伴侣身份矛盾没有靠日期改写。

实际请求与准备请求一致，295 个冻结文件和 57 个输入文件 SHA 一致。日期计划解释没有发送给模型。本批没有生成新 R02，也没有测日期对排序的净效果；加日期不能修复字段漏填、错误事件 ID 或准入规则冲突。

## 产量与完整成本

| root | HTTP | input tokens | output tokens | total tokens |
| --- | ---: | ---: | ---: | ---: |
| 254852 | 1 | 14,047 | 1,403 | 15,450 |
| 136202 | 1 | 13,990 | 1,217 | 15,207 |
| 94289 | 1 | 14,178 | 983 | 15,161 |
| 197635 | 1 | 12,842 | 1,414 | 14,256 |
| 合计 | **4** | **55,057** | **5,017** | **60,074** |

四次实际模型均为 `qwen3.6-35b-a3b`，HTTP 200、finish_reason=stop。请求使用 temperature=0.25、max_tokens=4096、JSON；服务是否严格采用所有参数、默认 reasoning、推理 token 分项、最大服务容量与可核实金额仍为 UNKNOWN。全部失败计划成本已计入，历史实验成本未混入。

| 指标 | 结果 | 解释 |
| --- | ---: | --- |
| 排序、SC、MS 原始产量 | 各 0 | 未调用 Writer |
| 程序接受／去重后有据 QA 产量 | 各 0 | 没有候选进入准入与内容评价 |
| 内容 PASS／FAIL／UNDETERMINED | 各 0，分母 0 | 状态为 NOT_EVALUATED，正确率 null |
| QA 状态、actor、时间边错误数 | 未评估 | 没有输出可检查，不写成“零错误” |
| repair／重试／超时 | 0／0／0 | Planner 合同失败未转成内容 repair |
| 每道有效题成本 | 未定义 | 有效产量为 0，不给有限单题成本 |

Memory 只复用本地冻结快照；远程 Memory、Free、Solver、训练、额外审核 API 均为 0。本轮已收口停止。

## 下一步优先处理

先修 Planner 的提示与准入一致性：部分场景路线不再被旧最小备选数阻断；保留 ID 和必填字段检查，将能安全定位的坏候选隔离，并显式处理候选依赖，避免一个局部错误连带丢弃所有其他候选。全坏或无法识别的响应仍应停止，不能删除检查后宣称计划已正确。

同时明确新人物声明与 actor 绑定的统一接口，保留人物实际参与、事件身份和状态一致性的内容检验。先用这些冻结响应做离线回放，证明不再错误整批拒绝且坏 ID／缺字段不会放行，再做独立获授权的新生成验证。**本次报告步骤没有修改被冻结生产逻辑，没有手工修补模型响应，也没有追加请求。**

## 人工查看入口

- {link('四条完整原始模型输出及事件菜单', REPORTS / 'ALL_PLANNER_OUTPUTS.md')}
- {link('逐阶段输入输出路径', REPORTS / 'FILE_INDEX.md')}
- {link('机器可读最终汇总', REPORTS / 'FINAL_SUMMARY.json')}
- {link('调用与输入完整性审计', REPORTS / 'execution_audit.json')}
- {link('完整成本及实际返回元数据', REPORTS / 'complete_cost_audit.json')}
- {link('原始生成汇总（保留当时 PENDING 状态）', RUN / 'generation_summary.json')}
'''
    (REPORTS / 'FINAL_REPORT.md').write_text(report)
    status = dict(gen)
    status.update({
        'status': 'CLOSED_AFTER_OFFLINE_DIAGNOSIS',
        'generation_status': gen['status'],
        'semantic_quality': 'PLANNER_DIAGNOSED; DOWNSTREAM_NOT_EVALUATED',
        'closed_at': when,
        'final_summary_path': str(REPORTS / 'FINAL_SUMMARY.json'),
        'final_report_path': str(REPORTS / 'FINAL_REPORT.md'),
        'additional_api_calls': 0,
    })
    write_json(RUN / 'run_status.json', status)
    readme_path = DELIVERY / 'README.md'
    original = readme_path.read_text()
    if '## 正式执行结果（追加）' not in original:
        readme_path.write_text(original + f'''
## 正式执行结果（追加）

以上“离线准备／等待确认”是授权前的历史记录。用户批准 56 次 HTTP 上限后，本轮于 2026-09-23 正式执行并收口：4 次 Planner 请求正常返回，四条均被 Planner 本地合同拒绝，未进入 Gate、续写和出题。实际 60,074 tokens、0 重试、0 技术失败。排序／SC/MS 质量 NOT_EVALUATED。

两条卡在旧最小候选数与部分场景路线冲突；另外两条分别出现菜单外 ID、必填 before_state 缺失。全部原始响应保留，没有补样或使用剩余额度重跑。

{link('最终报告', REPORTS / 'FINAL_REPORT.md')} · {link('原始模型输出', REPORTS / 'ALL_PLANNER_OUTPUTS.md')} · {link('实际产物索引', REPORTS / 'FILE_INDEX.md')}。
''')
    print(json.dumps({'status': status['status'], 'http': 4, 'tokens': 60074,
                      'report': str(REPORTS / 'FINAL_REPORT.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
