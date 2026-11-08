"""Index retained procedure deviations alongside the completed delivery."""
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
B = HERE.parent


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    output = B / 'delivery/all'
    final = json.loads((output / 'FINAL_STATUS.json').read_text())
    assert final['population_denominator'] == final['terminal_roots'] == 68
    paths = [
        B / 'evidence/BASELINE_RECHECK_EXECUTION_ISSUES_v1.json',
        B / 'evidence/READ_DIRECTORY_RELOCATION_V1.json',
        B / 'evidence/READ_DIRECTORY_RELOCATION_V2.json',
        B / 'evidence/ALL_CONTENT_AUDIT_PROCEDURE_ADAPTER.json',
        B / 'evidence/NATIVE_ATTESTATION_RESUME_V2_CHECKS.json',
        B / 'evidence/SELECTION_LITERAL_PREFIX_SCREEN_V1.json',
        B / 'units/root103866/evidence/QA_REVIEW_EXECUTION_HOLD.json',
        *sorted((B / 'evidence/MECHANICAL_ERRORS').glob('*.json')),
        *sorted((B / 'units').glob('*/evidence/SELECTION_REVIEW_EXECUTION_HOLD.json')),
        *sorted((B / 'evidence').glob('PUBLIC_SCHEMA_RESUME_V2_INDEPENDENT_REVIEW.json')),
    ]
    rows = [{'record_binding': bind(p), 'retained_record': json.loads(p.read_text())} for p in paths]
    value = {
        'status': 'RETAINED_PROCEDURE_DEVIATIONS_DISCLOSED', 'records': rows,
        'input_bindings': [bind(p) for p in [*paths, output / 'FINAL_STATUS.json',
                                           output / 'ALL_ATTEMPTS_INDEX.json', HERE / 'export_procedure_notes_v1.py']],
        'source_question_gold_or_solver_result_changed_by_export': False,
        'semantic_hold_rescued': False, 'external_semantic_certification': False,
        'training_started': False, 'training_pool_merged': False,
    }
    with (output / 'PROCEDURE_DEVIATIONS.json').open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    notes = '''# 程序偏差补充披露

完整逐 root 终态与首稿/审核/失败索引见 ALL_ATTEMPTS_INDEX.json；本文件补充跨 root 的程序偏差。原始记录及完整字段见 PROCEDURE_DEVIATIONS.json。QUESTION_STAGE_LEDGER.json 另列组合前已被拦截的题目槽位及最终交叉核对，避免只看公开题而遗漏失败排序题。

- 103866：六场源有 READY 审核，但 QA 审者提前看到设计，保留本轮程序 HOLD；完整对话在 source_only/，题目未入选。
- 96462、112812 B1：失败 read 目录曾迁移，原 RAW 与失败保留。内容审计只对记录的精确原 path＋SHA 解析到相同原字节档案，未豁免其他哈希失败；修正版只补字段或限定既有引用，不能称为无程序偏差。
- 112812 B2、136202 B1：首次缺失原生事件完整阅读声明，原失败保留；仅允许此前已有完整阅读后的 additive 声明补充，显式 False 改 True 或语义判断变更会被拒绝。
- 136202 A3：作者误用 reviewer 参数创建了 render/context；审核者又迁移该未提交语义判断的目录并重新生成 context。两个目录均保留，实际指定审核者随后完整阅读并提交自己的判断。错误创建的 context 本身不构成审核者阅读证明。
- 156136 公开题审核：原提交漏填复读声明，问题列表结构与两个题级声明亦不完整。原 RAW 和失败保留；仅允许实际审者补缺失声明，并以不变 candidate_id 无损转为字典。关系、题文、引用、理由、排序和原有值均不能改变，显式 False 不能改成 True；这不是程序自动证明实际复读。
- 191749：冻结 selection 的 A1/A3 场景编号与线程/证据交叉冲突，首稿依原 R01 场景及继承约束推进。该根以 selection 程序 HOLD 停止，保留计划与首稿；不将此归为首稿源事实错误，也未改计划恢复生产。
- 162896 出题：酒店入住起始日被写成预订请求开始日，两种候选命题均被独立审核判为 unknown；5 个多选槽位在组合前未填充，两道排序分别复读通过。未将 unknown 当错误选项、未修题或改 Gold 恢复多选。
- 基线审核另保留 246750 的范围执行错误、254852 的可疑 PASS 与另一读者 HOLD，以及 246750/259192 的复核独立性限制；这些不被改写为没有偏差的双读认证。

该索引披露程序执行限制，不提供新的语义 PASS，也不恢复任何失败 root。作者与审者继承同一模型，共享文件系统；阅读和身份为可审计声明。未新训练，未合并正式训练池。
'''
    with (output / 'PROCEDURE_NOTES_ZH.md').open('x') as stream:
        stream.write(notes)
    ledger = []
    for root in final['roots']:
        unit = B / 'units' / ('root' + root['root_id'].rsplit('-', 1)[1])
        folder = unit / 'roots' / unit.name
        row = {'root_id': root['root_id'], 'population_status': root['status'],
               'action': root['action'], 'unfilled_composition_slots': [],
               'final_crosscheck_results': [], 'bindings': []}
        for filename, field, source_field in [
            ('COMPOSITION_RECEIPT.json', 'unfilled_composition_slots', 'unfilled_slots'),
            ('FINAL_CROSSCHECK.json', 'final_crosscheck_results', 'results'),
        ]:
            path = folder / filename
            if path.exists():
                row[field] = json.loads(path.read_text())[source_field]
                row['bindings'].append(bind(path))
        for path in [folder / 'QA_AUTHOR_FAILURE.json', unit / 'evidence/QA_REVIEW_EXECUTION_HOLD.json',
                     unit / 'evidence/SELECTION_REVIEW_EXECUTION_HOLD.json']:
            if path.exists():
                row.setdefault('qa_failures', []).append(json.loads(path.read_text()))
                row['bindings'].append(bind(path))
        ledger.append(row)
    unfilled = [slot for row in ledger for slot in row['unfilled_composition_slots']]
    ids = {q['candidate_id'] for q in json.loads((output / 'CONTENT_QUESTIONS.json').read_text())['questions']}
    assert not ids.intersection(slot['candidate_id'] for slot in unfilled if 'candidate_id' in slot)
    stage_ledger = {
        'status': 'EXHAUSTIVE_NEW_ROOT_QA_STAGE_ACCOUNTING', 'population_roots': len(ledger),
        'unfilled_composition_slots': len(unfilled),
        'unfilled_slot_types': dict(Counter(slot['question_type'] for slot in unfilled)),
        'roots': ledger, 'source_question_gold_or_selection_changed': False,
        'input_bindings': [bind(output / 'FINAL_STATUS.json'), bind(output / 'CONTENT_QUESTIONS.json'),
                           bind(HERE / 'export_procedure_notes_v1.py')],
        'scope': 'Includes slots rejected before public composition, which do not appear in FINAL_CROSSCHECK. Historical REUSE roots keep their bound historical ledgers.'}
    with (output / 'QUESTION_STAGE_LEDGER.json').open('x') as stream:
        json.dump(stage_ledger, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': value['status'], 'retained_records': len(rows),
                      'unfilled_composition_slots': len(unfilled)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
