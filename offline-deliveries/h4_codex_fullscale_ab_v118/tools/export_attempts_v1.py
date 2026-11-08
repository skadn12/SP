"""Add an exhaustive attempts index and source-only QA-hold appendix at delivery."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from production_inventory_v1 import inventory


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def export(output):
    batch = HERE.parent
    data = inventory()
    assert data['population_denominator'] == data['terminal_roots'] == 68
    assert read(output / 'FINAL_STATUS.json')['status'] == 'FULL_FIXED_POPULATION_PROCESSED'
    manifest = read(batch / 'MANIFEST.json')
    for row in data['roots']:
        unit = batch / 'units' / ('root' + row['root_id'].rsplit('-', 1)[1])
        origin = next(r for r in manifest['roots'] if r['root_id'] == row['root_id'])
        row['unit_path'] = str(unit)
        row['original_baseline_binding'] = origin['baseline_binding']
        row['new_and_reused_artifacts_separate'] = True
        row['stage_artifacts'] = [binding(p) for p in sorted(unit.rglob('*.json'))]
        if row['action'] == 'REUSE':
            row['reuse_batch'] = origin['reuse_batch']
        if row['source_only_complete_qa_hold']:
            folder = unit / 'roots' / unit.name
            source = folder / 'CUMULATIVE_SOURCE.json'
            turns = read(source)['turns']
            b3 = folder / 'scenes/B_S03/CUMULATIVE_SOURCE.json'
            assert source.read_bytes() == b3.read_bytes()
            appendix = output / 'source_only' / unit.name
            appendix.mkdir(parents=True, exist_ok=False)
            text = ['# 完整六场对话：QA 程序 HOLD', '',
                    '本源六场已有 READY 审核；QA 审者事先读到设计，因此本轮题目流程终止，未产出入选题目。',
                    '原文、首稿和旧审核均保留，未通过替换审者或改稿恢复。', '',
                    'Source: ' + str(source), 'SHA256: ' + binding(source)['sha256'], '']
            text.extend('[' + t['turn_ref'] + '] ' + t['speaker'] + ': ' + t['text'] for t in turns)
            path = appendix / 'FULL_DIALOGUE.md'
            with path.open('x') as stream:
                stream.write('\n\n'.join(text) + '\n')
            row['source_only_readable_dialogue_binding'] = binding(path)
    data['input_bindings'] = [binding(batch / 'MANIFEST.json'), binding(batch / 'MASTER_FREEZE.json'),
                              binding(batch / 'evidence/READ_DIRECTORY_RELOCATION_V1.json'),
                              binding(batch / 'evidence/ALL_CONTENT_AUDIT.json'),
                              binding(output / 'FINAL_STATUS.json'), binding(Path(__file__).resolve())]
    data['training_started'] = False
    data['training_pool_merged'] = False
    data['external_semantic_certification'] = False
    save(output / 'ALL_ATTEMPTS_INDEX.json', data)
    final = read(output / 'FINAL_STATUS.json')
    lines = ['# 全量执行与失败处置', '',
             '固定68个root全部进入真实终态；终态包括成功、基线HOLD、扩充HOLD与QA失败，不等于68个均成功。', '',
             '主交付报告：REPORT_ZH.md；所有首稿、审核、失败和路径：ALL_ATTEMPTS_INDEX.json；原始阶段文件仍位于units/。', '',
             f"主交付完整对话 {final['complete_dialogues']} 个；另有完整六场但 QA 程序 HOLD 的 source-only 对话 {len(data['source_only_complete_qa_hold_roots'])} 个，见 source_only/。", '',
             '103866：完整扩充源保持六场READY，但审者先读QA_DRAFT设计；没有填写不实匿名声明，没有换审者救题。',
             '75241：后续复读发现发送状态矛盾，追加累计源HOLD，旧READY保留；历史READY不是当前合规证明。',
             '122711：A2旧READY漏检线程错位，在A3累计源审核中HOLD；A3自己的学习线程没有被误称错位。',
             '96462：审者误移动首个失败read目录后新读；原/新判断只差缺失字段补充，原字节均保留。严格relocation审计验证唯一原path+SHA映射，程序偏差仍记录。', '',
             '|root|终态|物化场数|保留的历史READY|失败阶段|',
             '|---|---|---:|---:|---|']
    for row in data['roots']:
        phases = ', '.join(sorted({d['phase'] for d in row['failure_details']}))
        lines.append(f"|{row['root_id']}|{row['status']}|{row['materialized_scenes']}|{row['historical_ready_reviews_retained']}|{phases}|")
    lines.extend(['', '题目质量限制：每根固定同组六事实变体，题间相关；直接属性反转可能较容易。',
                  '排序原始标签偶然为I1到I6不能直接推出Solver难度低，实际配对Solver按冻结协议对每题做12次展示排列。',
                  '本轮不按Solver分数改题；同一模型代理的复核不称外部语义认证。未新训练、未合并正式池。', ''])
    with (output / 'EXECUTION_REPORT_ZH.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps({'status': 'ALL_ATTEMPTS_EXPORTED', 'roots': len(data['roots']),
                      'source_only_qa_hold_roots': data['source_only_complete_qa_hold_roots']}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    export(p.parse_args().output.resolve())
