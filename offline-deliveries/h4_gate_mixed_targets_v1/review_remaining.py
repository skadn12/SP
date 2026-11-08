"""Compile the five completed offline reviews with unchanged source evidence."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUN = BASE / 'prepared_run'
NOTES = [
    (112812, 'gate', 'PC-C001:C01', 'UNDETERMINED_TARGET_GRANULARITY',
     '日历安装与测试',
     '原文明确计划设置日历并用下一批预约测试；新reason也承认这个计划，却以未实际完成为主要理由将两个目标都写为unknown。缺少完成证据本身不能排除planned。完整候选还包含成功测试特定预约这一复合结果，所以不把整句必须为planned当作已认证真值。这里是计划信息在结构字段中丢失的风险，而非误判已经完成。',
     [54, 56, 57, 60, 61, 62, 63]),
    (112812, 'gate', 'PC-C002:C01', 'UNDETERMINED_TARGET_GRANULARITY',
     '与老板预约具体时间',
     'T0076明确表示要安排与老板会面；新reason认可安排意向，同时因未联系、未收到回复、未预约成功而把完整目标写为unknown。具体时间尚未确定不等于安排会面没有计划，但具体时段预约成功比一般意向更细，不能只凭旧标签认定整句必须planned。Elena在新输出中被正确识别为邻居和建议者；没有沿用Memory把她当老板的错误。',
     [68, 69, 71, 76, 78]),
    (112812, 'gate_supplement', 'PS-N001:C01', 'PASS_SCOPED_HISTORICAL_CLASSIFICATION',
     '预算方法与每周检查',
     '历史有学生贷款压力、预算建议和愿意尝试的回应，但没有具体工具与每周固定检查安排。将这个具体新安排视为NEW_PROPOSAL、完整目标unknown有依据。这个PASS仅指此处历史状态分类，不认证候选价值、可实现性或未来完成。',
     [34, 35, 36, 39, 40]),
    (218519, 'gate', 'PC-C001:C01', 'UNDETERMINED_TARGET_GRANULARITY',
     '周六数学辅导',
     '原文支持周六见面、带笔记、一起解决数学困难的计划；没有已完成辅导的主张。完整目标还包含识别具体概念或问题，模型reason将它缩写为开始辅导和翻阅笔记，复合目标的精确支持仍需保留不确定。grounding_refs虽为Memory导航，但两个target_checks都绑定了真实TRANSCRIPT引用，不能误报为仅凭Memory通过。',
     [84, 85, 86, 91, 96, 97]),
    (218519, 'gate_supplement', 'PS-N002:C01', 'FAIL_OUTPUT_LABEL_CONSISTENCY',
     '补充的同一数学辅导',
     'required_transition=planned/SUPPORTED、after_state=unknown/UNKNOWN，却选择PLANNED_OR_INTENDED，因此仍被隔离。新合同已有UNKNOWN可表达这种组合，程序没有替模型改标签。reason用尚未发生解释unknown，仍混淆未完成与无计划。该候选的两个目标与初轮PC-C001完全一致，但after_state分类不同；两次请求的周边候选及recovery上下文不同，不能称为相同messages的纯随机重复实验。即使修正标签，它也是重复发展，不能算第二个独立场景。',
     [84, 85, 86, 91, 96, 97]),
]


def read(path):
    return json.loads(path.read_text())


def main():
    rows = []
    for number, stage, ref, assessment, title, diagnosis, evidence_numbers in NOTES:
        root = f'cc-train-episode-{number}'
        frozen = RUN / 'frozen' / root / stage
        result = RUN / 'results' / root / stage
        plan = read(frozen / 'plan.json')
        event_id, candidate_id = ref.split(':')
        scene = next(s for s in plan['scenes'] if s['memory_event_id'] == event_id)
        candidate = scene['next_step_candidates'][int(candidate_id[1:]) - 1]
        old = next(r for r in read(frozen / 'old_output.json')['candidate_reviews'] if r['candidate_ref'] == ref)
        new = next(r for r in read(result / 'output.json')['candidate_reviews'] if r['candidate_ref'] == ref)
        validation = read(result / 'validation.json')
        catalog = {e['evidence_ref']: e for e in read(frozen / 'context.json')['catalog']['entries']}
        evidence = [catalog[f'{root}:H:T{n:04d}'] for n in evidence_numbers]
        assert all(e['kind'] == 'TRANSCRIPT' for e in evidence)
        rows.append(dict(root_id=root, stage=stage, candidate_ref=ref, title=title,
                         assessment=assessment, diagnosis=diagnosis,
                         required_transition=candidate['required_transition'],
                         after_state=candidate['after_state'], old_review=old, new_review=new,
                         program_errors=validation['row_errors'].get(ref, []),
                         program_eligible=ref in validation['passed_candidate_refs'],
                         evidence=evidence,
                         source_paths=dict(plan=str(frozen / 'plan.json'),
                                           catalog=str(frozen / 'context.json'),
                                           old_output=str(frozen / 'old_output.json'),
                                           new_output=str(result / 'output.json'),
                                           validation=str(result / 'validation.json'))))
    report = dict(reviewer='Codex offline diagnosis', independent_human_certification=False,
                  additional_model_api_calls=0, production_admission_changed=False,
                  scope='Five original candidates; original outputs and validations remain unchanged.',
                  rows=rows)
    (BASE / 'reports/CONTENT_112812_218519.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 112812 / 218519 Gate 内容诊断', '',
             'Codex离线诊断；不是独立人工认证，不构成新增生产语义门槛。未调用API，未改原始输出、候选和准入结果。', '',
             '5条中：1条局部历史分类合理，3条完整复合目标的状态粒度保留不确定，1条输出标签不一致。不是全流程场景合格率。', '']
    for row in rows:
        lines += [f"## {row['root_id']} / {row['stage']} / {row['candidate_ref']}：{row['title']}", '',
                  f"诊断：`{row['assessment']}`。程序准入：{row['program_eligible']}。", '',
                  f"required_transition：{row['required_transition']}", '',
                  f"after_state：{row['after_state']}", '', row['diagnosis'], '',
                  f"[新输出]({row['source_paths']['new_output']}) · [原Planner候选]({row['source_paths']['plan']}) · [验证]({row['source_paths']['validation']})", '',
                  '原文证据（按冻结catalog回取，未改写）：', '']
        for e in row['evidence']:
            lines += [f"- `{e['evidence_ref']}` / `{e['turn_ref']}` / {e['speaker']}: {e['text']}"]
        lines.append('')
    (BASE / 'reports/CONTENT_112812_218519.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'offline_reviewed_candidates': len(rows), 'model_api_calls': 0}))


if __name__ == '__main__':
    main()
