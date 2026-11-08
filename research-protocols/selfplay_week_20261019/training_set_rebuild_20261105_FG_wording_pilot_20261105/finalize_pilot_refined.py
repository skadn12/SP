#!/usr/bin/env python3
"""Conservative final selection plus read-only integrity checks for the pilot."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
RUN = BASE / 'training_set_rebuild_20261105_FG_exploration_run01'
OLD135 = BASE / 'training_set_rebuild_20261105_F_supplement_run01/repaired_training_export_v1/TRAIN.jsonl'
NOW = datetime.now(timezone.utc).isoformat()
ACCEPT = {162896, 191749, 225575, 45849, 75241}
WHY = {
    103866: '来源盲答的完整顺序与原答案一致，两名可读性评审均偏好改写，但平均清晰度从4.5降至4.0；保留原限定。',
    104814: 'V1评审对改写偏好分歧；时间锚点与“已取得/已记录”范围仍需精确对齐。V2仍有未来汇报指代和记录条件变化风险。',
    109164: 'V1的“最近一次调整后”不如原来的片段截止点明确；V2虽然更顺，却把周二改周三作为前提，可能给答案线索并改变范围。',
    130221: '自然度改善，但原题问草案中应继续采用的已选方向，V1改问哪些已写入草案；操作焦点不同，保留原题。',
    136202: 'V1把组图顺序强化为发送顺序；V2也删除“最初收到照片”限定并改成按给出的组图结构排列，保留原题。',
    186046: '来源盲答发现地图在展示后只明确“将放回”；V1从已完成的装包动作转成结束时的实际状态，最终状态不唯一，保留原题。',
    229686: '两名首轮评审都偏好原题；V2“这次讨论”指代不清，时间终点和“记录准确/已经完成”的判断对象可能变化。',
    162896: '来源盲答与冻结答案一致，仍询问清点时已经实际带来的库存；改写两名评审均偏好，清晰度均值不降。',
    191749: '来源盲答与冻结顺序一致，保留“修改后的草案”状态；改写两名评审均偏好，清晰度均值上升。',
    225575: '来源盲答与冻结顺序一致，仍问火车咨询中符合出发时间限制的方案如何逐步形成；改写两名评审均偏好，清晰度均值不降。',
    45849: '来源盲答与冻结顺序一致，保留下午清理范围及图画书节点；改写两名评审均偏好，清晰度均值上升。原题已有的“检查沙发”推断软问题仍保留。',
    75241: '来源盲答与冻结多选答案一致，保留两人分别查日历、仅判断日历可行性的范围；改写两名评审均偏好，清晰度均值不降。'
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def put(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


v1 = json.loads((OUT / 'DERIVED_CANDIDATES.json').read_text(encoding='utf-8'))['items']
v2 = json.loads((OUT / 'DERIVED_CANDIDATES_V2.json').read_text(encoding='utf-8'))['items']
by_root2 = {x['root_id']: x for x in v2}
decisions = []
selected = []
for item in v1:
    root = item['root_id']
    use_v1 = root in ACCEPT
    expected = item['original_candidate'].copy()
    derived = item['derived_candidate'].copy()
    assert set(expected) == set(derived)
    diffs = [k for k in expected if expected[k] != derived[k]]
    assert diffs == ['question'], (root, diffs)
    chosen = derived if use_v1 else expected
    score = next(x for x in json.loads((OUT / 'BLIND_REVIEW_RESULTS_V1_CORRECTED.json').read_text(encoding='utf-8'))['by_candidate'] if x['candidate_id'] == item['candidate_id'])
    if use_v1:
        assert score['preference_votes'] == {'original': 0, 'revised': 2}
        assert score['revised_clarity_mean'] >= score['original_clarity_mean']
        assert root != 130221
    decisions.append({
        'candidate_id': item['candidate_id'], 'root_id': root,
        'selected_version': 'V1_REWRITE' if use_v1 else 'ORIGINAL',
        'decision': 'RETAIN_AS_RESEARCH_WORDING_CANDIDATE' if use_v1 else 'KEEP_ORIGINAL_WORDING',
        'reason_zh': WHY[root], 'v1_readability_summary': score,
        'candidate_quality_status': 'UNREVIEWED', 'training_admitted': False})
    selected.append({'candidate_id': item['candidate_id'], 'root_id': root,
                     'arm': item['arm'], 'question_type': item['question_type'],
                     'selected_version': 'V1_REWRITE' if use_v1 else 'ORIGINAL',
                     'candidate': chosen})
    if root in by_root2:
        v2i = by_root2[root]
        original = v2i['original_candidate']
        rewritten = v2i['v2_derived_candidate']
        assert set(original) == set(rewritten)
        changed = [k for k in original if original[k] != rewritten[k]]
        assert changed == ['question'], (root, changed)

put('FINAL_WORDING_DECISIONS_REFINED.json', {
    'recorded_at_utc': NOW,
    'supersedes': 'FINAL_WORDING_DECISIONS.json; first pass selected six rewrites before incorporating task-operation and final-state ambiguity checks.',
    'retained_v1_count': len(ACCEPT), 'kept_original_count': 12 - len(ACCEPT),
    'retained_v2_count': 0,
    'selection_rule': 'Research-only conservative judgment: retain V1 only when both internal readability reviewers prefer it, mean clarity does not decrease, blind source reconstruction matches the frozen answer, and the queried operation remains aligned. This rule was applied after seeing the pilot results; it is not a preregistered threshold.',
    'items': decisions,
    'candidate_quality_status': 'UNREVIEWED', 'training_admission': False})
put('FINAL_RESEARCH_CANDIDATES_REFINED.json', {
    'status': 'RESEARCH_COMPARISON_ONLY_NOT_TRAINING_DATA',
    'supersedes': 'FINAL_RESEARCH_CANDIDATES.json; root 130221 returned to original wording after task-operation review.',
    'original_run_unchanged': True, 'items': selected})

# Integrity checks: sealed source package, preserved training export, all 12 source/draft bindings.
seal = json.loads((RUN / 'EXPLORATION_SEAL.json').read_text(encoding='utf-8'))
seal_checks = []
for rec in seal['files']:
    path = RUN / rec['path']
    actual = sha(path)
    seal_checks.append(actual == rec['sha256'])
assert len(seal_checks) == 1095 and all(seal_checks)
assert sha(OLD135) == '46acd3d1725c8af4fea264a5cad2dbdc5b36e604a75ff04f5ea2db1b4b2b50e1'
for item in v1:
    root = item['root_id']
    source = Path(item['source_path'])
    assert sha(source) == item['source_sha256']
    wrapper_path = Path(item['original_draft_path'])
    wrapper = json.loads(wrapper_path.read_text(encoding='utf-8'))
    assert sha(wrapper_path) == item['original_draft_sha256']
    assert wrapper['candidates'][0] == item['original_candidate']
    source_json = json.loads(source.read_text(encoding='utf-8'))
    assert source_json['turns'] and len(source_json['turns']) == item['source_turn_count']

# Verify each recorded structural edge quote binds to the declared source turn exactly.
structure = json.loads((OUT / 'STRUCTURE_DIAGNOSTIC.json').read_text(encoding='utf-8'))
edge_evidence_checks = 0
for case in structure['cases']:
    source = json.loads(Path(case['source_path']).read_text(encoding='utf-8'))
    for edge in case.get('direct_edges', []):
        for ev in edge['evidence']:
            turns = [t for t in source['turns'] if t['turn_ref'] == ev['turn_ref'] or t['turn_ref'].endswith(':' + ev['turn_ref'])]
            assert len(turns) == 1, (case['candidate_id'], ev['turn_ref'], len(turns))
            assert ev['quote'] in turns[0]['text']
            assert ev['origin'] == turns[0]['origin']
            edge_evidence_checks += 1
assert edge_evidence_checks > 0

put('FINAL_INTEGRITY_AUDIT.json', {
    'checked_at_utc': NOW,
    'original_exploration_seal_path': str(RUN / 'EXPLORATION_SEAL.json'),
    'sealed_file_count': len(seal_checks), 'sealed_hash_mismatches': 0,
    'sealed_135_path': str(OLD135), 'sealed_135_sha256_verified': sha(OLD135),
    'sampled_source_and_author_first_draft_bindings_verified': len(v1),
    'sampled_source_or_draft_hash_mismatches': 0,
    'wording_candidate_field_change_assertion': 'question_only',
    'v2_candidate_field_change_assertion': 'question_only',
    'structure_edge_quote_origin_checks': edge_evidence_checks,
    'structure_closure_and_query_audit_flags': {
        c['candidate_id']: {
            'closure_matches_gold': c.get('closure_matches_gold'),
            'query_domain_unique_and_complete': c.get('query_domain_unique_and_complete'),
            'pair_audit_entailments_match_closure': c.get('pair_audit_entailments_match_closure')}
        for c in structure['cases'] if 'direct_edges' in c},
    'training': 'none', 'solver_calls_in_this_pilot': 0,
    'source_or_authoring_files_modified': False,
    'candidate_quality_status': 'UNREVIEWED'})

print(json.dumps({'seal_files_verified': len(seal_checks), 'structure_edge_evidence_checked': edge_evidence_checks,
                  'retained_v1': len(ACCEPT), 'kept_original': 12 - len(ACCEPT)}, ensure_ascii=False))
