"""Post-call diagnostic reporting only; no Solver calls or content revisions."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import re

BATCH = Path(__file__).resolve().parent.parent
OLD = BATCH.parent / 'h4_codex_serialized_ab_validation_v113'
OLD_ID = 'h4_codex_serialized_ab_validation_v113-ms-5922533334a04633'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    with (BATCH / name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def binding(path):
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def verify(rows):
    for row in rows:
        assert sha(row['path']) == row['sha256'], row['path']


def summarize(run, cid):
    manifest = read(run / 'manifest.json')
    bindings = {r['candidate_id']: r for r in read(run / 'canonical_bindings.json')['candidates']}
    reverse = {content: label for label, content in bindings[cid]['content_id_by_label'].items()}
    gold = {reverse[c] for c in read(run / 'gold.json')[cid]}
    result = next(q for q in read(run / 'RESULTS.json')['questions'] if q['candidate_id'] == cid)
    rows = []
    for req in manifest['requests']:
        if req['candidate_id'] != cid:
            continue
        path = run / 'responses' / (req['request_id'] + '.json')
        answer = read(path)
        match = re.fullmatch(r'\s*<box>\s*([^<>]*)\s*</box>\s*', answer['raw_output'])
        assert match
        display = [v.strip().upper() for v in match.group(1).split(',') if v.strip()]
        labels = sorted(reverse[req['variant']['display_to_content'][v]] for v in display)
        pred = set(labels)
        tp, fp = len(pred & gold), len(pred - gold)
        partial = Fraction(tp, len(gold)) - Fraction(fp, 6 - len(gold))
        assert partial == Fraction(answer['score']['partial_score_fraction'])
        rows.append({'variant_id': req['variant']['variant_id'], 'slot': req['slot'],
                     'seed': req['seed'], 'canonical_prediction': labels,
                     'target_B_selected': 'B' in pred, 'exact': pred == gold,
                     'false_positive_count': fp, 'false_negative_count': len(gold - pred),
                     'partial_score_fraction': str(partial), 'raw_response': binding(path)})
    assert len(rows) == 18
    exact = sum(r['exact'] for r in rows)
    partial_mean = sum((Fraction(r['partial_score_fraction']) for r in rows), Fraction()) / 18
    assert partial_mean == Fraction(result['full_partial_mean'])
    assert Fraction(exact, 18) == Fraction(result['full_exact_rate'])
    return {'candidate_id': cid, 'calls': 18, 'gold': sorted(gold), 'exact_correct': exact,
            'exact_rate': str(Fraction(exact, 18)),
            'target_B_selected': sum(r['target_B_selected'] for r in rows),
            'selected_count_by_canonical_label': {k: sum(k in r['canonical_prediction'] for r in rows) for k in 'ABCDEF'},
            'prediction_cardinality_histogram': dict(sorted(Counter(str(len(r['canonical_prediction'])) for r in rows).items())),
            'mean_predicted_cardinality': str(Fraction(sum(len(r['canonical_prediction']) for r in rows), 18)),
            'calls_with_false_positive': sum(r['false_positive_count'] > 0 for r in rows),
            'false_positive_total': sum(r['false_positive_count'] for r in rows),
            'false_negative_total': sum(r['false_negative_count'] for r in rows),
            'partial_mean': str(partial_mean), 'selected_by_native_policy': result['selected_by_policy'],
            'rows': rows}


def compare(original, revised):
    left = {(r['variant_id'], r['slot']): r for r in original['rows']}
    right = {(r['variant_id'], r['slot']): r for r in revised['rows']}
    assert left.keys() == right.keys() and len(left) == 18
    counts = Counter()
    for key, a in left.items():
        b = right[key]
        assert a['seed'] == b['seed']
        counts['both_correct' if a['exact'] and b['exact'] else
               'corrected' if b['exact'] else 'regressed' if a['exact'] else 'both_wrong'] += 1
    return {k: counts[k] for k in ['both_correct', 'corrected', 'regressed', 'both_wrong']}


def main():
    for side in ['BASE', 'TRAINED']:
        assert read(BATCH / f'evaluation/{side}_COMPLETE.json')['actual_calls'] == 36
    assert read(BATCH / 'INDEPENDENT_SOLVER_AUDIT.json')['status'] == 'PASS'
    assert read(BATCH / 'reviews/ACTUAL_PAIRING_REVIEW.json')['status'] == 'PASS'
    verify(read(BATCH / 'EXECUTION_FROZEN.json')['files'])
    verify(read(BATCH / 'evaluation/PAIRED_PREFLIGHT.json')['files'])
    verify(read(BATCH / 'OLD_BATCH_PROTECTION.json')['files'])
    qs = read(BATCH / 'REVISION_DRAFT.json')['questions']
    by_arm = {q['comparison_arm']: q for q in qs}
    stats = {}
    for side in ['base', 'trained']:
        run = BATCH / 'evaluation/run' / side
        stats[side] = {arm: summarize(run, q['candidate_id']) for arm, q in by_arm.items()}
        stats[side]['historical_v113'] = summarize(OLD / 'evaluation/run' / side, OLD_ID)
        stats[side]['paired_outcomes'] = compare(stats[side]['original_replay'], stats[side]['wording_revision'])
    save('PAIRED_DIAGNOSTIC.json', {
        'status': 'COMPLETE', 'pairs': 1, 'fresh_calls': 72, 'statistics': stats,
        'historical_calls_are_not_fresh_calls': True,
        'score_interpretation': 'Old original Gold is the declared historical Gold; original B remains semantically ambiguous.',
        'limits': ['Posthoc, nonblind, same inherited model source review.',
                   'One question / one root / one fact family; 18 samplings are not 18 independent questions.',
                   'Paired prompt positions and request seeds verified; deterministic replay is not assumed.',
                   'No new ordering review, training, expansion, or merge.']})
    previous = read(OLD / 'SELECTED_CANDIDATES.json')['questions']
    assert len(previous) == 20 and sum(q['candidate_id'] == OLD_ID for q in previous) == 1
    retained = [deepcopy(q) for q in previous if q['candidate_id'] != OLD_ID]
    additions = read(BATCH / 'reviews/NIRALA_EVIDENCE_ADDENDUM.json')
    assert additions['status'] == 'PASS'
    retained_by_id = {q['candidate_id']: q for q in retained}
    for correction in additions['corrections']:
        q = retained_by_id[correction['candidate_id']]
        label = correction['canonical_label']
        assert q['option_evidence'][label] == correction['original_evidence']
        q['option_evidence'][label] = correction['recommended_evidence']
    assert len(additions['corrections']) == 3
    # Export evidence is private metadata: never change public text, Gold, or source.
    for q in retained:
        old_q = next(o for o in previous if o['candidate_id'] == q['candidate_id'])
        assert {k: v for k, v in q.items() if k != 'option_evidence'} == {k: v for k, v in old_q.items() if k != 'option_evidence'}
    revision_selected = stats['trained']['wording_revision']['selected_by_native_policy']
    recommended = retained + ([by_arm['wording_revision']] if revision_selected else [])
    assert len(retained) == 19
    save('RECOMMENDED_CANDIDATES.json', {
        'questions': recommended, 'status': 'RECOMMENDATION_OVERLAY_NOT_MERGED',
        'selection_rule': 'Existing M_clean_policy native full-partial mean in (0,1), plus content review; original ambiguous item excluded.',
        'quarantined_original_ids': [OLD_ID], 'original_replay_is_training_candidate': False,
        'replacement_admitted': revision_selected, 'old_safe_retained': 19,
        'export_evidence_only_corrections': additions['corrections'],
        'export_evidence_corrections_after_freeze': True,
        'export_evidence_corrections_change_public_prompt_or_gold': False,
        'training_started': False, 'training_pool_merged': False,
        'old_selected_artifact': binding(OLD / 'SELECTED_CANDIDATES.json'),
        'diagnostic': binding(BATCH / 'PAIRED_DIAGNOSTIC.json')})
    old_text = by_arm['original_replay']['options']['B']
    new_text = by_arm['wording_revision']['options']['B']
    report = [
        '# v114 歧义题最小修复与配对复测', '',
        '已完成全部 48 条多选真假命题的来源复查：47 条清晰、1 条存在措辞歧义，未发现命题缺乏来源依据或确认需要改 Gold 的项目。此结果不证明不存在其他错误；排序题未在本轮重新审核。', '',
        '本轮只修订 root20432 的 Gold3 题 B 选项。完整对话、题干、其他五个选项和答案 C、D、F 保持一致。', '',
        f'- 原句：{old_text}', f'- 修订：{new_text}',
        '- 原文 R02:A:S02:T004，Emily：“Four cups for this pot. Fill it to the two-cup mark twice, and pour each lot into the pot while it\'s still cold.”', '',
        '原句可能被理解为两次灌注中的一次局部步骤；修订明确“仅一次、总共两杯”，因此与原文“四杯、两次”直接矛盾。', '',
        '两版本各 6 个选项置换 × 3 次采样；训练前后模型分别 36 次，合计 72 次新调用，无自动重试。实际 prompt 配对和全部原始回答复算审核均通过。', '',
        '| 模型 | 历史原题全对 | 本次原题全对 | 修订题全对 | 本次原题误选 B | 修订题误选 B |',
        '|---|---:|---:|---:|---:|---:|']
    for side, label in [('base', 'M0 训练前'), ('trained', 'M_clean_policy 已有训练后')]:
        d = stats[side]; h, a, b = (d[k] for k in ['historical_v113', 'original_replay', 'wording_revision'])
        report.append(f"| {label} | {h['exact_correct']}/18 | {a['exact_correct']}/18 | {b['exact_correct']}/18 | {a['target_B_selected']}/18 | {b['target_B_selected']}/18 |")
    report += ['', '所有 B 计数均按原始内容标签统计，已还原显示选项置换。历史原题成绩单列，不计入 72 次新调用；原题含歧义，其历史 Gold 分数仅用于诊断。', '',
               '配对结果（修订前错误→修订后正确、反向退步、两者都对、两者都错）：']
    for side in ['base', 'trained']:
        d = stats[side]['paired_outcomes']
        report.append(f"- {side}：{d['corrected']}、{d['regressed']}、{d['both_correct']}、{d['both_wrong']}。")
    report += ['', '本次修订没有降低误选 B 或提高全对率。题面歧义和模型错误需分别处理：修订题与来源明确矛盾，但 Solver 仍选择该错项，说明这条事实上的过选仍然存在。不得据此推断全部题目无误，也不能仅为提高 Solver 成绩再次改写。']
    report += ['', f"旧推荐 20 题中隔离原歧义题，保留其他 19 题。修订题{'符合' if revision_selected else '不符合'}已有训练后 Solver 的原生难度入选规则（0 < partial mean < 1），因此本轮推荐为 {len(recommended)} 题。修订题的 partial mean 为 {stats['trained']['wording_revision']['partial_mean']}。",
               '', '在冻结与评估之后核对推荐导出时，另发现 root761 三题复用的 Nirala 价格错项引用不够直接：原引用说 City Stop 昂贵，实际反证应引用同场 T007 明确 Nirala 为 moderate。此项属于导出证据定位不足，命题自身仍有来源依据。新推荐文件只修正这三处私有 evidence 元数据，记录旧值与新值；题文、Gold、来源及冻结评估输入均未改动，因此未增加 Solver 调用。见 reviews/NIRALA_EVIDENCE_ADDENDUM.json。',
               '', '推荐名单仅为独立 overlay，未合并训练池，未启动训练；旧 v113 全部 2218 个受保护文件哈希保持一致。', '',
               '后续沿用现有双路线、各三轮流程，并在出题前加入 WORDING_RULE_ADDENDUM.md 中的总量/单次量、次数、人物和状态检查；不因 Solver 做错就改写正确选项或 Gold。', '',
               '该实验是已接触模型反馈后的单题措辞诊断。来源复查来自同一继承模型、非盲审，不属于外部认证；一次修复和同题多次采样不足以证明大批量稳定性或训练泛化收益。', '',
               '主要文件：', '- PAIRED_DIAGNOSTIC.json：逐次 canonical 回答、历史和新成绩、配对统计。',
               '- RECOMMENDED_CANDIDATES.json：推荐候选及隔离记录。',
               '- PUBLIC_REVISIONS.json / inputs/root20432_CUMULATIVE_SOURCE.json：两版完整题目与原对话。',
               '- evaluation/run/{base,trained}/responses/：所有原始模型回答。',
               '- INDEPENDENT_SOLVER_AUDIT.json / reviews/ACTUAL_PAIRING_REVIEW.json：复算与配对审核。', '']
    with (BATCH / 'REPORT_ZH.md').open('x') as f:
        f.write('\n'.join(report))
    save('FINAL_STATUS.json', {
        'status': 'COMPLETE_POSTHOC_WORDING_DIAGNOSTIC',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'statement_assets_audited': 48, 'clear': 47, 'wording_risk': 1,
        'question_pairs': 1, 'fresh_solver_calls': 72,
        'frozen_eval_content_and_tools_changed_after_freeze': False,
        'postcall_report_tool_added': True,
        'postfreeze_export_evidence_only_corrections': 3,
        'original_ambiguous_question_quarantined': OLD_ID,
        'revision_selected_by_trained_native_policy': revision_selected,
        'recommended_count': len(recommended),
        'recommended_by_type': dict(Counter(q['question_type'] for q in recommended)),
        'old_protected_files_verified_unchanged': 2218,
        'training_started': False, 'training_pool_merged': False,
        'external_or_blind_certification': False,
        'one_pair_not_independent_generalization_test': True,
        'artifacts': [binding(BATCH / name) for name in ['PAIRED_DIAGNOSTIC.json',
                      'RECOMMENDED_CANDIDATES.json', 'REPORT_ZH.md', 'INDEPENDENT_SOLVER_AUDIT.json']]})
    print(json.dumps({'recommended': len(recommended), 'revision_admitted': revision_selected,
                      'stats': {s: {a: {k: v for k, v in d.items() if k != 'rows'}
                                  for a, d in st.items()} for s, st in stats.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
