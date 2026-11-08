"""Build a reviewable fixed-cohort report, then seal only the completed batch."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

B = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def report():
    audit = read(B / 'delivery/POST_RUN_INTEGRITY_AUDIT_V1.json')
    assert audit['pass'], audit['errors']
    scope = read(B / 'REPAIR_SCOPE.json')
    progress = read(B / 'delivery/PROGRESS_FINAL_V1.json')
    summary = read(B / 'delivery/PAIRED_SOLVER_SUMMARY_V1.json')
    candidates = read(B / 'delivery/TRAINING_CANDIDATES_V1.json')
    disposition = read(B / 'delivery/QUESTION_DISPOSITION_V1.json')
    lengths = read(B / 'delivery/DIALOGUE_LENGTHS_ALL_ROOTS_V1.json')
    rows = audit['roots']
    finalized = [r for r in rows if r['status'] == 'FINALIZED']
    scene_complete = [r for r in rows if r.get('scene_state', {}).get('full_expansion_complete')]
    content_roots = [r for r in finalized if r['content_pass'] > 0]
    candidate_roots = {q['root_id'] for q in candidates['questions']}
    per_root = Counter(q['root_id'] for q in candidates['questions'])
    source_counts = Counter(r['source_status'] for r in rows)
    lines = [
        f'历史失败共{scope["historical_failed_roots"]}个 root，v120已救回{len(scope["already_recovered_v120"])}个；本批固定处理其余24个，不重复计算既有成功。', '',
        f'本轮固定24个失败 root 已全部完成一次补救及最终处置。{len(scene_complete)}个具有审核通过的完整六场扩充，{len(content_roots)}个产出内容合格题目，{len(candidate_roots)}个产出 Solver 筛选候选。', '',
        f'新增{progress["new_scenes_generated"]}场；内容通过{progress["content_pass_questions"]}题，其中排序{progress["content_pass_ordering"]}题，多选 Gold 分布为{json.dumps(progress["content_pass_ms_gold_counts"], ensure_ascii=False)}。', '',
        f'Solver 筛选后保留{candidates["candidate_count"]}道训练候选，来自{candidates["root_count"]}个 root；多选 Gold 分布为{json.dumps(candidates["multi_select_gold_counts"], ensure_ascii=False)}。Gold6仅为对照。候选尚未合并训练池，也未训练。', '',
        f'来源标签：原文未改{source_counts["UNCHANGED"]}个，合成修订{source_counts["EDITED_SYNTHETIC"]}个（包含未通过者）。合成修订用于消除既有文本冲突，不声称恢复现实中的原始事实。原文、补丁、历次失败和审核均保留。', '',
        f'完整六场对话的平均累计长度为{lengths["arithmetic_mean"]["full_tokens"]:.1f} tokens，平均{lengths["arithmetic_mean"]["full_turns"]:.1f}条发言；六场范围平均{lengths["arithmetic_mean"]["six_scope_tokens"]:.1f} tokens。按本地Qwen3-4B tokenizer计算，文本仅为speaker: text，不含聊天模板或元数据；这不是生成用量。[逐root长度]({B / "delivery/DIALOGUE_LENGTHS_ALL_ROOTS_V1.json"})。', '',
        '| root | 来源 | 本轮终态 | 内容通过题数 | 筛选候选数 | 记录 |',
        '|---|---|---|---:|---:|---|',
    ]
    for row in rows:
        short = row['root_id'].rsplit('-', 1)[-1]
        unit = B / 'units' / ('root' + short)
        readable = B / 'delivery' / f'root{short}_DIALOGUE_AND_QUESTIONS.md'
        target = readable if readable.exists() else unit
        lines.append(f'| {short} | {row["source_status"]} | {row["status"]} | {row.get("content_pass", 0)} | {per_root[row["root_id"]]} | [完整记录]({target}) |')
    lines += ['', '停止与失败原因：', '']
    for row in rows:
        if row['status'] == 'FINALIZED':
            continue
        short = row['root_id'].rsplit('-', 1)[-1]
        unit = B / 'units' / ('root' + short)
        if row['status'] == 'SOURCE_HOLD':
            lines.append(f'- {short}：来源双审仍有HOLD；婚礼已在前一场告知，后一场又称从未提过，唯一修订候选未消除该矛盾，停止扩充。[来源处置]({unit / "source_repair/FINALIZE_RECEIPT.json"})')
        elif row['status'] == 'PREFIX_HOLD':
            reason = {'135466': '保留A1未完成固定线程要求的联系尝试，继承A2线程也与当前延续约束冲突', '103866': '三个保留B场重复使用服务结果转述、朋友核对、整理笔记的同一骨架，未通过冻结的多样性要求'}.get(short, '固定起点前缀复核未通过')
            lines.append(f'- {short}：{reason}；未前移起点或追加机会。[审核]({B / "prefix_audits" / unit.name / "PREFIX_SOURCE_REVIEW.json"})')
        elif row['status'] == 'SCENE_HOLD':
            failures = [s for s in row['scene_state']['scenes'] if s['status'] in ('STRUCTURAL_FAILURE', 'INTEGRITY_FAILURE', 'REVIEW_INVALID', 'REVIEW_HOLD')]
            detail = '; '.join(s['scene_ref'] + ': ' + s['status'] + ' ' + s.get('reason', '') for s in failures)
            reason = {'197635': 'A1首稿使用中文，违反新场英文合同；原READY误判后已明确阻断，未翻译重写', '254852': 'A2审核引用不存在的turn，字段预检失败后仍执行save，原生schema拒绝且保留首稿；这是审核提交失败，不等同于已判对话语义不合格'}.get(short, detail)
            lines.append(f'- {short}：{reason}。[逐场记录]({unit / "roots" / unit.name / "scenes"})')
        elif row['status'] == 'QA_REVIEW_SCHEMA_HOLD':
            lines.append(f'- {short}：审核首稿缺少题目编号，正式审核存档被结构校验拒绝；完整六场对话保留，未改答案或重交语义首稿。[失败记录]({unit / "roots" / unit.name / "QA_REVIEW_FAILURE.json"})')
        else:
            lines.append(f'- {short}：题目唯一首稿失败，依赖停止。[失败记录]({unit / "roots" / unit.name / "QA_AUTHOR_FAILURE.json"})')
    for row in finalized:
        short = row['root_id'].rsplit('-', 1)[-1]
        folder = B / 'units' / ('root' + short) / 'roots' / ('root' + short)
        for slot in read(folder / 'COMPOSITION_RECEIPT.json')['unfilled_slots']:
            lines.append(f'- {short} 局部题槽未入选：{slot["candidate_id"]}，{slot["reason"]}；问题字段{json.dumps(slot.get("issues", []), ensure_ascii=False)}。其余题目按各自审核结果处理。[失败题素材]({folder / "PUBLIC_ASSETS.json"})；[审核理由]({folder / "ASSET_SOURCE_JUDGMENTS.json"})。')
    lines += ['',
        f'执行偏差：parent曾误改已冻结的进度脚本，三个match-read被哈希门禁拒绝，未产生正式read context或语义首稿；随后按原SHA逐字恢复，终态支持移到独立新脚本。995项冻结文件最终全部一致，但本报告不声称运行中从无协议偏差。[偏差记录]({B / "evidence/PARENT_COLLECTOR_FREEZE_DEVIATION_V1.json"})。', '',
        f'另有一次提交前字段校验调用报TypeError后shell继续正式save；正式原生schema校验和provenance均通过，但前置检查当时未成功执行。未重写原始首稿；后续改为独立调用。[记录]({B / "evidence/REVIEW_PREFLIGHT_FALLTHROUGH_B_V1.json"})。', '',
        f'评估准备曾因10道排序题的JSON字段插入顺序不符合旧导出器要求而停止，未创建评估池或调用Solver。随后新增经过另一代理审核并单独冻结的v3适配器，仅在内存副本中规范字段顺序；标签对应文本、答案、来源、证据、评分、seed及解码未改，原冻结文件保持原样。[准备失败]({B / "evidence/EVALUATION_PREPARE_LABEL_ORDER_FAILURE_V1.json"})；[适配审核]({B / "evidence/EVALUATION_CANONICAL_LABEL_AUDIT_A_V1.json"})。', '',
        f'[两模型配对评估]({B / "delivery/PAIRED_SOLVER_REPORT_ZH_V1.md"})；[全部题目处置]({B / "delivery/QUESTION_DISPOSITION_V1.json"})；[训练候选]({B / "delivery/TRAINING_CANDIDATES_V1.json"})；[完整性审计]({B / "delivery/POST_RUN_INTEGRITY_AUDIT_V1.json"})。', '',
        '两模型使用相同冻结题目、排列、seed和解码协议。Gold2–5、Gold6与排序分别统计；原文来源和合成修订来源分别统计。重复排列、rollout与同root题族不是独立样本，不据此声称独立测试集上的泛化提升。', '',
        '所有原文、修订、扩充及同root题目必须在同一数据划分内。审核为共享文件系统中的不同执行代理记录，未宣称外部独立认证或可证明的盲审。旧v118、v119、v120批次保持封存。', '',
        f'候选去向统计：{json.dumps(disposition["counts"], ensure_ascii=False)}。', '',
    ]
    path = B / 'delivery/BATCH_CLOSEOUT_ZH_V1.md'
    with path.open('x') as stream:
        stream.write('\n'.join(lines))
    save(B / 'delivery/CLOSEOUT_TOTALS_V1.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'fixed_candidates': 24,
        'historical_failed_roots': scope['historical_failed_roots'],
        'previously_recovered_roots': scope['already_recovered_v120'],
        'cumulative_recovered_content_roots': len(scope['already_recovered_v120']) + len(content_roots),
        'terminal_statuses': audit['terminal_statuses'], 'complete_six_scene_roots': len(scene_complete),
        'content_pass_roots': len(content_roots), 'finalized_roots': len(finalized),
        'new_scenes_generated': progress['new_scenes_generated'], 'content_pass_questions': progress['content_pass_questions'],
        'content_pass_ordering': progress['content_pass_ordering'], 'content_pass_ms_gold_counts': progress['content_pass_ms_gold_counts'],
        'training_candidate_count': candidates['candidate_count'], 'training_candidate_roots': candidates['root_count'],
        'training_candidate_types': candidates['question_type_counts'], 'training_candidate_ms_gold_counts': candidates['multi_select_gold_counts'],
        'training_or_pool_merge': False, 'report_binding': bind(path),
    })
    print(str(path))


def seal():
    totals = read(B / 'delivery/CLOSEOUT_TOTALS_V1.json')
    assert read(B / 'delivery/POST_RUN_INTEGRITY_AUDIT_V1.json')['pass']
    target = B / 'COMPLETE_MANIFEST.json'
    assert not target.exists()
    files = [bind(path) for path in sorted(B.rglob('*')) if path.is_file() and path != target]
    save(target, {'status': 'COMPLETE_V121_FIXED_ALL_FAILED_REPAIR_AND_PAIRED_EVALUATION',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'actual_executor': '/root',
        'files': files, 'excludes_only': [str(target)], 'totals': totals,
        'closeout': bind(B / 'delivery/BATCH_CLOSEOUT_ZH_V1.md'), 'training_or_pool_merge': False})
    print(json.dumps({'status': 'BATCH_SEALED', 'files': len(files), 'totals': totals}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('report', 'seal'))
    args = parser.parse_args()
    globals()[args.action]()
