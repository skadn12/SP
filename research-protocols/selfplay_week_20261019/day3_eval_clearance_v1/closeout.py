#!/usr/bin/env python3
"""Publish the checked finite-v1 result and update navigational status only."""
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def read(path):
    return json.loads(path.read_text())


def bind(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def write_new(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    stamp = datetime.now(timezone.utc)
    checked = read(HERE / 'audit/FINAL_ARTIFACT_CHECK.json')
    assert str(checked['status']).startswith('PASS'), checked['status']
    ledger = read(HERE / 'CONTENT_AND_REQUEST_LEDGER.json')
    admission = read(HERE / 'content/CONTENT_ADMISSION.json')
    freeze = read(HERE / 'requests/REQUEST_FREEZE.json')
    recovery = read(HERE / 'NEXT_RECOVERY_QUEUE.json')
    prior = read(HERE / 'audit/PRIOR_PRODUCTION_SEAL_RECHECK_FINAL.json')
    assert prior['status'] == 'PASS'
    grouped = defaultdict(Counter)
    split_counts = defaultdict(Counter)
    for row in ledger['slots']:
        grouped[row['question_type']][row['status']] += 1
        split_counts[row['split']][row['status']] += 1
    requests = [json.loads(s) for s in (HERE / 'requests/REQUESTS.jsonl').read_text().splitlines()]
    questions = []
    for row in admission['items']:
        questions.append({k: row[k] for k in ['candidate_id', 'root_id', 'original_split',
                                            'question_type', 'status', 'public_question',
                                            'private_author_record']})
    roots = {r['root_id'] for r in requests}
    status = 'DAY3_COMMON_EVAL_V1_REQUESTS_FROZEN_COVERAGE_INSUFFICIENT'
    summary = {
        'schema_version': 'day3_finite_common_eval_clearance_summary_v1',
        'created_at_utc': stamp.isoformat(), 'status': status,
        'scope': 'Finite prospectively frozen screening inventory, actual content admission and CPU-only request freeze; no model evaluation.',
        'reviewed_heldout_roots': 31, 'planned_slots': 186,
        'supported_slots_screened': 60, 'finite_plans_screened': 107,
        'screening_full_source_turns': 2939,
        'original_material_dispositions': {'supported_pending_exclusions': 60, 'material_ambiguous_hold': 98, 'source_shortfall': 28},
        'final_slot_counts': ledger['counts'],
        'final_slot_counts_by_type': dict(grouped), 'final_slot_counts_by_split': dict(split_counts),
        'actual_public_questions': len(questions), 'content_accepted': admission['accepted_count'],
        'content_held': admission['held_count'], 'content_mechanical_checks': admission['mechanical_checks'],
        'accepted_roots': len(roots), 'accepted_question_types': sorted({r['question_type'] for r in requests}),
        'actual_questions': questions, 'request_count': len(requests),
        'display_variants': len(read(HERE / 'requests/DISPLAY_MAPS.json')),
        'prompt_tokens_min': min(r['prompt_tokens'] for r in requests),
        'prompt_tokens_max': max(r['prompt_tokens'] for r in requests),
        'reserved_answer_tokens': 32, 'maximum_model_length': 16384,
        'zero_truncation': True, 'length_holds': freeze['length_holds'],
        'all_answer_space_mapping_roundtrips': sum(r['all_answer_space_roundtrips'] for r in freeze['length_and_mapping_checks']),
        'historical_role': 'REGRESSION_ONLY', 'unseen_root_claim': False,
        'actual_model_identity_frozen': False, 'full_evaluation_ready': False,
        'day3_complete': False, 'training_start_go': False,
        'new_solver_calls': 0, 'new_training_runs': 0, 'new_dialogue_expansions': 0,
        'train_production_unchanged': {'roots': 20, 'planned_slots': 240, 'drafts': 195,
            'accepted_candidates': 187, 'content_hold': 8, 'material_shortfalls': 42,
            'generation_intercepts': 3, 'matched_slot_pairs': 88, 'training_pool_frozen': False},
        'limits': [
            'Four MC questions from three historically exposed roots cannot support multi-type A/B/selfplay-effect conclusions; ordering coverage is zero.',
            'The bounded 107-plan inventory is not an exhaustive proof that no other source-supported fact families exist.',
            '144 HOLD is not 144 bad questions: 98 retain material ambiguity, 46 screened slots retain related-mapping or other unresolved checks.',
            'Reconstruction bytes/hash were sealed and reported before private reads; the separate seal manifest was created later and no exact first-private-read timestamp was captured.',
            'Local screening count/status text retains earlier-stage metadata; final selection and all186 content/request ledger are the current authorities.',
            'No source expansion, Solver difficulty, feedback benefit or training benefit was measured in this closure.'
        ],
        'next_action': 'Use the exact candidate-specific recovery queue, prioritizing ordering and new roots. Record a versioned, score-blind addendum; keep existing v1 request bytes and unrelated HOLD unchanged. If no supported cleared ordering remains, prepare a separately frozen evaluation source version rather than appending to the current locked BASELINE.',
        'artifacts': {name: bind(HERE / rel) for name, rel in {
            'screening_freeze': 'SCREENING_FREEZE.json',
            'family_selection': 'FAMILY_SELECTION.json',
            'all186_final_ledger': 'CONTENT_AND_REQUEST_LEDGER.json',
            'content_admission': 'content/CONTENT_ADMISSION.json',
            'source_first_reconstruction': 'content/qa/SOURCE_FIRST_RECONSTRUCTION.json',
            'source_first_seal': 'content/qa/FIRST_PASS_SEAL.json',
            'independent_content_review': 'content/qa/FINAL_CONTENT_REVIEW.json',
            'request_freeze': 'requests/REQUEST_FREEZE.json',
            'independent_final_check': 'audit/FINAL_ARTIFACT_CHECK.json',
            'prior_production_seal_recheck': 'audit/PRIOR_PRODUCTION_SEAL_RECHECK_FINAL.json',
            'next_recovery_queue': 'NEXT_RECOVERY_QUEUE.json',
            'next_recovery_explanation': 'NEXT_RECOVERY_ZH.md'
        }.items()}
    }
    write_new(HERE / 'SUMMARY.json', summary)
    report = '''# 共同评估有限库存 v1：实际题目与请求冻结结果

本轮完成候选事实族筛查、真实题目和私有 Gold、源优先独立内容审核、完整公开输入冻结。最终 **4 道多选题、3 个 root、72 条待执行请求**；排序题为 0。**第三天尚未完成，当前不能支撑正式多题型 A/B/selfplay 效果比较，也未启动训练。**

## 实际保留与覆盖

| 原计划题位 | 冻结题目 | HOLD | 素材不足 | 事实族排除 |
|---|---:|---:|---:|---:|
| 多选 93 | 4 | 84 | 2 | 3 |
| 排序 93 | 0 | 60 | 26 | 7 |
| 合计 186 | 4 | 144 | 28 | 10 |

DEV 为 2 题/2 roots；TEST 为 2 题/1 root。四题覆盖 186 个原题位的 2.15%，覆盖 31 个原 root 中的 3 个。原来源、root、划分、场景范围与设计未被替换；所有 root 仍是历史暴露来源，仅作回归评估，不能称作未见 root 测试。

| 题号 | root | 题型 | Gold 数量 |
|---|---|---|---:|
| D3-EVAL-DEV-001 | 105961 | 多选 | 2 |
| D3-EVAL-DEV-008 | 129761 | 多选 | 2 |
| D3-EVAL-TEST-055 | 216742 | 多选 | 2 |
| D3-EVAL-TEST-056 | 216742 | 多选 | 3 |

## 已完成的实际检查

预先冻结的有限库存覆盖 60 个素材支持题位、107 个计划，三个审核分组共读 2,939 条完整来源 turn。全五项排除通过后才按预锁 SHA 选择事实族；未依据 Solver 分数选题。这是有限库存筛查，不能据此证明其他可用题族不存在。

独立内容审核者先只读四道公开题和三个完整原始 BASELINE（100、104、92 turns），封存其答案和逐项证据，再读取私有记录比较。四题 Gold 全部匹配；24 个选项均核查，68 条作者证据的 turn、speaker、scene、index、quote 和 Unicode span 全通过。父代理实际入选检查共 237 项，通过。未修改题面、Gold 或来源，也未使用措辞补修。

独立答案重建文件本身在首次私有读取前已写入、核验并向父代理报告 SHA；单独的 FIRST_PASS_SEAL 清单是在私有读取后补写，已如实记录。首次私有读取的精确工具时间未捕获，不能把清单时间冒充为读取前封存时间。DEV008 的限定条件修正在选族前完成，过程与原文件备份另有保留；未保存的短暂中间写入已在修正日志披露。

四题每题生成 6 个唯一选项排列，每排列 3 个采样请求，共 24 个排列、72 条请求。所有非空答案子集均作映射往返核验，共 1,512 次；未向输入构建器提供私有 Gold。使用绑定 tokenizer 检查完整 chat prompt，长度 4,050–4,505 tokens，另预留 32 个输出 token，未截断。原 186 题位索引用于种子，不在入选后重编号。

另有独立代理复核最终绑定、逐题输入、排列、种子、token 长度及 186 位账目，见 [最终产物检查](audit/FINAL_ARTIFACT_CHECK.json)。旧训练侧两批生产的 3,058 个封存文件复查未变化。请求已冻结，但模型身份尚未冻结；72 是待执行请求数，本轮实际 Solver 调用为 0。TEST 仍只允许最终模型在身份冻结后执行。

## 当前问题与下一步

144 个 HOLD 中，98 个继承素材歧义，46 个是本轮筛查后仍有必要事实映射等疑点的题位。HOLD 表示尚不能可靠放行，不等于题目已证明错误。另有 10 个题位的有限库存计划全部明确排除，28 个保留来源短缺。旧筛查文件里的局部计数和 PENDING 文本是阶段快照，最终状态以 [186 位裁决表](CONTENT_AND_REQUEST_LEDGER.json) 为准。

实际瓶颈是可用独立事实族及排序来源覆盖。先按 [定点补审队列](NEXT_RECOVERY_ZH.md) 处理确切候选关联的历史记录，优先排序和新 root。补审可能确认复用排除，不能预设一定恢复。DEV056 照片/笔记与签名关系只是多选候选，完整关联记录多于提示中的两条，不能只审两条就放行。不重复全局 993 条旧格式审计。

后续结果以不看分数的新版本 addendum 记录，当前 v1 的题目与请求保持封存。若来源内找不到足够、可排重的排序事实，应另外冻结评估来源版本和设计；当前锁定 BASELINE 不追加后缀补题。

训练侧仍为 20 roots、187 个接受候选、88 个配对题位，尚未组为冻结训练池。42 个素材缺额已整理成 20 个共同扩充请求，但没有实际执行扩充。这类生产反馈可指导新版本素材生成；不能代替 Solver 弱点反馈，也不能据现有文件证明扩充、反馈或 selfplay 有效。

## 文件入口

- [完整机器摘要](SUMMARY.json)
- [实际题目目录](content/public/) 与 [私有作者 Gold/证据目录](content/private/)
- [独立审核结果](content/qa/FINAL_CONTENT_REVIEW.json)
- [实际请求冻结说明](requests/REQUEST_FREEZE.json) 与 [请求列表](requests/REQUESTS.jsonl)
- [原事实族选择记录](FAMILY_SELECTION.json) 与 [全部题位最终裁决](CONTENT_AND_REQUEST_LEDGER.json)
'''
    assert not (HERE / 'REPORT_ZH.md').exists()
    (HERE / 'REPORT_ZH.md').write_text(report)

    names = ['CURRENT_STATUS.json', 'DAY3_LATEST.json', 'DAY3_REMAINING_ZH.md', 'README_ZH.md']
    hist = HERE / 'navigation_history'
    hist.mkdir(exist_ok=False)
    backups = []
    for name in names:
        source = WEEK / name
        data = source.read_bytes()
        (hist / name).write_bytes(data)
        backups.append({'original_path': str(source), 'backup': bind(hist / name)})
    summary_ref, report_ref = bind(HERE / 'SUMMARY.json'), bind(HERE / 'REPORT_ZH.md')
    current = read(WEEK / 'CURRENT_STATUS.json')
    current.update(status=status, updated_at_utc=stamp.isoformat(),
                   updated_at_beijing=stamp.astimezone(timezone(timedelta(hours=8))).isoformat(),
                   next_action=summary['next_action'])
    current['daily_planning']['day3'] = status
    current['day3']['common_evaluation_actual_questions_frozen'] = len(questions)
    current['day3']['full_day_complete'] = False
    current['common_evaluation_preparation'].update(
        actual_questions_frozen=len(questions),
        fact_family_semantic_clearance='FINITE_V1_COMPLETE_OTHERS_HOLD_OR_EXCLUDED',
        training_start_go=False, current_clearance_summary=summary_ref,
        preparation_bound_reports_are_historical=True)
    current['common_evaluation_clearance'] = {
        'summary': summary_ref, 'report': report_ref, 'final_slot_counts': ledger['counts'],
        'actual_questions_frozen': len(questions), 'roots': len(roots),
        'request_count': len(requests), 'solver_calls': 0, 'ordering_questions': 0,
        'full_evaluation_ready': False, 'actual_model_identity_frozen': False,
        'historical_role': 'REGRESSION_ONLY', 'training_start_go': False}
    current['files']['day3_common_evaluation_clearance_report'] = report_ref
    current['files']['day3_common_evaluation_clearance_summary'] = summary_ref
    current['before_execution_freeze_remaining'] = [
        'Sufficient fixed common evaluation coverage, especially ordering/new roots; current four MC questions are insufficient for multi-type A/B claims.',
        'Day4 matched training pools, old-data fraction q and symmetric resources/settings freeze.',
        'Actual future model/checkpoint/runtime identity and permitted execution freeze.'
    ]
    current['execution_progress_v7'] = {
        'scope': 'Finite107-plan common-eval screen, four independently accepted actual MC questions,72 CPU-built requests frozen.',
        'new_generated_questions': 4, 'accepted_questions': 4, 'frozen_requests': 72,
        'new_solver_calls': 0, 'new_training_runs': 0, 'new_dialogue_expansions': 0,
        'sealed_prior_production_files_unchanged': prior['total_files_checked'], 'day3_complete': False}
    (WEEK / 'CURRENT_STATUS.json').write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n')
    latest = read(WEEK / 'DAY3_LATEST.json')
    latest.update(status=status, updated_at_utc=stamp.isoformat(), next_action=summary['next_action'],
                  full_day_complete=False, common_evaluation_actual_questions_frozen=len(questions),
                  common_evaluation_clearance=current['common_evaluation_clearance'])
    latest['common_evaluation_preparation'].update(current['common_evaluation_preparation'])
    (WEEK / 'DAY3_LATEST.json').write_text(json.dumps(latest, ensure_ascii=False, indent=2) + '\n')
    (WEEK / 'DAY3_REMAINING_ZH.md').write_text('''# 第三天：当前结果与剩余任务

第三天尚未完成。训练侧 20 roots 候选生产已完成；共同评估 v1 本次实际冻结 **4 道多选题、3 roots、72 条待执行请求**，排序为 0。当前覆盖不足以支撑正式多题型 A/B 比较，模型身份也尚未冻结。本轮 Solver、训练、新对话扩充均为 0。

## 已经完成

- 训练侧原 240 位：187 个接受候选、8 个内容 HOLD、42 个素材缺额、3 个生成拦截，88 个配对题位；尚不是冻结训练池。
- 共同评估 31 roots、186 位保留：4 个请求冻结、144 个 HOLD、28 个素材不足、10 个事实族排除。60 个素材支持题位的有限库存 107 个计划已筛查，不声称自然语言穷举。
- 四题经过题面/全文独立答案重建先封存、私有 Gold 后比较、全部选项事实边界核查。6 排列×3 采样/题；完整输入 4,050–4,505 tokens，无截断。
- 请求冻结、独立产物检查与原生产 3,058 个文件封存复查完成。当前来源全为历史暴露 root，仅作回归评估。

完整结果见[本轮报告](day3_eval_clearance_v1/REPORT_ZH.md)与[机器摘要](day3_eval_clearance_v1/SUMMARY.json)。HOLD 不等于已证明题目有错；局部筛查阶段旧 PENDING/计数不替代[186 位最终裁决](day3_eval_clearance_v1/CONTENT_AND_REQUEST_LEDGER.json)。

## 下一步执行顺序

1. 按[定点补审队列](day3_eval_clearance_v1/NEXT_RECOVERY_ZH.md)处理真实候选关联的旧必要事实映射，优先排序与未入选 root。不得把无检索命中当 PASS，不重复全局 993 条审计；明确复用则保留排除。
2. 将补审结果记录为看模型分数前的新版本 addendum；当前四题 v1 与请求保持原样。若可用排序覆盖仍不足，另外冻结评估来源版本与设计，不给当前锁定 BASELINE 追加后缀。
3. 有足够固定共同评估后，再执行第四天的两臂训练池组合、旧题比例 q、对称训练资源及最终模型身份冻结。TEST 仅最终模型执行，不作中途筛选。

## 素材反馈与扩充

[素材反馈包](day3_material_feedback_v1/REPORT_ZH.md)保留 42 个原缺额：40 个联合条件/否定、2 个显式修订，合并为 21 个两组配对目标、20 个共同扩充请求；实际扩充仍为 0。

后续共享 Train 来源扩充继承旧事件延续 3 场、数据集新事件 3 场的两条路线，顺序累计生成和审核；这与出题实验 A/B 组是不同维度。执行前绑定新来源版本、事件、模型和预算，全文审核通过后才出题。原来源和缺额不改；扩充作者不接收未来评估题或 Gold。

素材短缺属于生产反馈，已复现 Solver 弱点属于模型反馈；二者可以分别指导后续扩充，但不能混称 selfplay 收益。当前不报告新题难度、训练或反馈收益。
''')
    (WEEK / 'README_ZH.md').write_text('''# 当前方案与进度

当前采用[修订方案 v2](PLAN_V2_ZH.md)与[逐日执行细则 v2.1](DAILY_EXECUTION_V2_ZH.md)。方案文字反映修订时点；执行状态以[CURRENT_STATUS.json](CURRENT_STATUS.json)和[DAY3_LATEST.json](DAY3_LATEST.json)为准。

第 2 天在限定排序反馈范围内完成接入验收，多选关系仍是探索假设。第 3 天完成 20 个 Train root 的候选生产与审核；共同评估 v1 已冻结 **4 道多选题、3 roots、72 条待执行请求**，排序为 0。**第三天尚未完成，尚不能支撑正式多题型 A/B 比较；本轮没有运行 Solver 或训练。**

- [本轮共同评估真实结果](day3_eval_clearance_v1/REPORT_ZH.md)：186 位裁决、四题独立审核、实际请求冻结与限制。
- [下一批定点补审队列](day3_eval_clearance_v1/NEXT_RECOVERY_ZH.md)：优先排序和新 roots；只审具体相关旧记录。
- [第三天剩余任务](DAY3_REMAINING_ZH.md)
- [素材缺口反馈包](day3_material_feedback_v1/REPORT_ZH.md)：42 个原缺额，20 个下一版本共享 Train 扩充请求，尚未实际扩充。
- [训练侧两批合并报告](day3_production_batch2_v1/REPORT_ZH.md)：187 个接受候选，尚未冻结训练池。
- [共同评估准备阶段历史报告](day3_common_evaluation_v1/REPORT_ZH.md)

已封存实验、来源与生产目录保持原样。当前评估来源是历史暴露 root，仅作回归检查；素材反馈、Solver 反馈与训练效果分别记录。导航修改前版本保存在 day3_eval_clearance_v1/navigation_history/。
''')
    write_new(HERE / 'NAVIGATION_UPDATE.json', {
        'created_at_utc': stamp.isoformat(), 'status': status, 'old_versions': backups,
        'current_versions': [bind(WEEK / name) for name in names],
        'summary': summary_ref, 'report': report_ref,
        'historical_sealed_experiments_modified': False})
    print(json.dumps({'status': status, 'questions': len(questions), 'requests': len(requests),
                      'day3_complete': False, 'new_solver_calls': 0}))


if __name__ == '__main__':
    main()
