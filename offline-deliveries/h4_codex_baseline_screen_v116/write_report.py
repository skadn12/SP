"""Describe the actual screen separately from future generation."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, text):
    with path.open('x') as output: output.write(text)

def main():
    results = read(HERE/'SCREEN_RESULTS.json')
    selected = read(HERE/'SELECTED_ROOTS.json')
    audit = read(HERE/'DELIVERY_AUDIT.json')
    assert audit['status']=='PASS' and audit['errors']==[]
    tests = read(HERE/'UNIT_TEST_RUN.json')
    assert tests['status']=='PASS' and tests['exit_code']==0 and tests['tests_run']>0
    for binding in tests['bindings']+[tests['log']]:
        assert sha(binding['path'])==binding['sha256']
    comparison = read(HERE/'REVIEW_COMPARISON.json')['roots']
    comparison = {r['root_id']:r for r in comparison}
    state_counts = Counter(r['decision'] for r in results['roots'])
    reads = [read(binding['path']) for row in results['roots'] for binding in row['reviews']]
    findings = [(r['root_id'], finding) for r in reads for finding in r['findings']]
    started = read(HERE/'SCREEN_PLAN.json')['created_at_utc']
    finished = datetime.now(timezone.utc).isoformat()
    elapsed = (datetime.fromisoformat(finished)-datetime.fromisoformat(started)).total_seconds()
    lines = ['# v116 基线一致性筛查结果', '',
        f'已完成固定8个候选的16份全文审核，按规则选定{len(selected["roots"])}个供下一批扩充的root。筛查结果为：'+
        '、'.join(f'{key} {state_counts[key]}' for key in ['PASS','WARNING','HOLD'])+'。', '',
        '本轮实施的是扩充前基线门禁；未扩充、未出新题、未调用Solver、未训练或合并训练池。v115的20道算法候选及全部历史结果保持原样。', '',
        '|root|两位读者结论|聚合结论|选择结果|', '|---|---|---|---|']
    for row in results['roots']:
        decisions = comparison[row['root_id']]['decisions']
        disposition = '下一批固定root' if row['selected'] else ('保留并阻断扩充' if not row['eligible'] else '合格备用；因四个上限未选')
        lines.append(f'|{row["root_id"].rsplit("-",1)[1]}|'+ ' / '.join(decisions)+f'|{row["decision"]}|{disposition}|')
    lines += ['', f'候选按clean-training数字顺序取首八个结构合格项；双审后再按原顺序取最多四个。未追加候选，未用当前生成或Solver结果筛root。筛查分母8，不能将选中{len(selected["roots"])}个写成随机{len(selected["roots"])}root的成功率或生成成功率。', '',
        f'每个发现保存逐字turn/speaker/quote及理由；程序校验计划/来源哈希、完整原顺序覆盖、两位不同读者、分类处置及HOLD优先、时间解释所需证据、后续约束无损合并。{tests["tests_run"]}项单元测试通过，覆盖引用伪造、删覆盖、降级HOLD、重复读者、缺额不补与禁止覆写等情形；执行记录及日志绑定见UNIT_TEST_RUN.json。机械独立复算见DELIVERY_AUDIT.json。', '']
    held_ids = {row['root_id'] for row in results['roots'] if not row['eligible']}
    lines += ['阻断依据（以下分别保留两位读者原判断，不作事后改稿）：', '']
    blockers = [(rid,f) for rid,f in findings if rid in held_ids and f['disposition']=='HOLD']
    for rid,finding in blockers:
        lines.append(f'- {rid.rsplit("-",1)[1]}：{finding["reason"]}')
        for citation in finding['citations']:
            lines.append(f'  - `{citation["turn_ref"]}` / {citation["speaker"]}: “{citation["quote"]}”')
    warning_note = '选中root的WARNING并未消失：' if any(r['decision']=='WARNING' for r in selected['roots']) else ''
    lines += ['', warning_note+'SELECTED_ROOTS.json合并保存两位读者的全部continuation_constraints。后续作者必须逐条读取这些限制；不能凭空补年龄、性别、身份背景、完成状态或隐藏时间线，也不能把人物自述升级为外部证实事实。', '',
        '两位读者为同一继承模型且有共享项目背景，不构成异模型复核、严格双盲或外部语义认证。检查只针对原文内部一致性；没有HOLD不等于人物事实逐句100%正确。所有候选均已有训练/Codex曝光；与dev/test在root、原始source_id及source_record_sha256上无重叠，但不是未见holdout。', '',
        '下一批直接消费SELECTED_ROOTS.json及其全部约束，在生成前另冻root/scene分母和未消费事件。沿用A1→A2→A3→B1→B2→B3、每场12–16条、逐场全文READY、一次首稿、无语义重写或换源；仍联合生成Gold2/3/4/5、Gold6对照与两道排序，并使用原v15配对Solver。详细交接为NEXT_BATCH_HANDOFF.json。', '',
        f'计划冻结至本报告生成墙钟{elapsed/60:.2f}分钟，包含并行读源、实现和测试门禁、等待与机械复核；不是纯模型生成或推理延迟。精确Codex模型ID、计费tokens及费用未提供。', '']
    write(HERE/'REPORT_ZH.md','\n'.join(lines)+'\n')
    write(HERE/'READ_FIRST.md','# v116 阅读入口\n\n本轮已完成扩充前的基线一致性筛查。\n\n先看[结果说明](REPORT_ZH.md)，固定下一批root与全部限制见[SELECTED_ROOTS.json](SELECTED_ROOTS.json)。\n\n原对话位于inputs/root*_BASELINE.json，双审首稿及引用位于reviews/root*/{primary,secondary}.json；规则为BASELINE_POLICY.md，下一批交接为NEXT_BATCH_HANDOFF.json。未生成新扩充或题目，未训练或入池。\n')
    status = {'status':'COMPLETE_BASELINE_SCREEN_NEXT_ROOT_SELECTION_FIXED', 'completed_at_utc':finished,
              'candidate_roots':8,'review_count':16,'decision_counts':dict(state_counts),
              'selected_root_ids':[r['root_id'] for r in selected['roots']],
              'held_root_ids':sorted(held_ids),'unit_tests_passed':tests['tests_run'],
              'historical_files_verified_unchanged':read(HERE/'HISTORICAL_AUDIT.json')['verified_file_count'],
              'generation_started':False,'solver_started':False,'training_started':False,'pool_merge_started':False,
              'external_semantic_certification':False,'wall_clock_plan_to_report_seconds':elapsed,
              'bindings':[{'path':str((HERE/name).resolve()),'sha256':sha(HERE/name)} for name in
                          ['REPORT_ZH.md','SCREEN_RESULTS.json','SELECTED_ROOTS.json','DELIVERY_AUDIT.json','NEXT_BATCH_HANDOFF.json','UNIT_TEST_RUN.json']]}
    write(HERE/'FINAL_STATUS.json',json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:status[k] for k in ['decision_counts','selected_root_ids','held_root_ids']}))

if __name__=='__main__': main()
