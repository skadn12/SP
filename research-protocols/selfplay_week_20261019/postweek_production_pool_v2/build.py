"""Index two production batches, retaining an explicit later content hold."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def save(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    assert not (HERE / 'POOL_MANIFEST.json').exists(), 'ALREADY_BUILT'
    audit = WEEK / 'postweek_production_feedback_audit_v1'
    concern_file = audit / 'TRAINING_CONTENT_CONCERNS.json'
    concerns = read(concern_file)['concerns']
    hold_ids = {c['candidate_id'] for c in concerns}
    batches = [WEEK / f'postweek_production_batch_v{k}' for k in (1, 2)]
    rows, inventories, bindings = [], [], [binding(concern_file), binding(audit / 'FINAL_MANIFEST.json')]
    for batch in batches:
        manifest = read(batch / 'FINAL_MANIFEST.json')
        for b in manifest['artifact_bindings'] + manifest['external_reference_bindings']:
            assert binding(b['path']) == b, b['path']
        path = batch / 'TRAINING_POOL_CANDIDATES.jsonl'
        candidates = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(candidates) == manifest['content_approved_unique_questions']
        assert all(r['root_id'].startswith('cc-train-') for r in candidates)
        rows.extend(candidates)
        inventories.append({'batch': batch.name, 'original_content_approved_candidates': len(candidates),
                            'later_content_concerns': sum(c['candidate_id'] in hold_ids for c in candidates),
                            'planned_slots': manifest['planned_slots'],
                            'material_shortfalls': manifest['material_shortfalls'],
                            'new_solver_requests': manifest['solver_requests']})
        bindings += [binding(batch / 'FINAL_MANIFEST.json'), binding(path)]
    assert len({r['candidate_id'] for r in rows}) == len(rows)
    assert hold_ids <= {r['candidate_id'] for r in rows}
    kept, excluded, signatures = [], [], {}
    for r in rows:
        field = 'options' if r['question_type'] == 'multi_select' else 'items'
        key = json.dumps({'type': r['question_type'], 'question': r['question'],
                          'choices': sorted(r[field].values())}, sort_keys=True, ensure_ascii=False)
        assert key not in signatures, (r['candidate_id'], signatures.get(key))
        signatures[key] = r['candidate_id']
        if r['candidate_id'] in hold_ids:
            excluded.append({'candidate_id': r['candidate_id'], 'reason': 'LATER_INDEPENDENT_CONTENT_HOLD',
                             'concern': next(c for c in concerns if c['candidate_id'] == r['candidate_id'])})
        else:
            kept.append(r)
    out = HERE / 'TRAINING_POOL_CANDIDATES.jsonl'
    with out.open('x', encoding='utf-8') as stream:
        for r in kept:
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    save(HERE / 'EXCLUSION_LEDGER.json', {'excluded': excluded, 'solver_score_filtering': False,
         'partial_fact_overlap_excluded': False, 'original_batches_unchanged': True})
    manifest = {
        'schema_version': 'cumulative_guarded_production_candidate_pool_v2',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'CONTENT_CANDIDATES_INDEXED_NOT_TRAINING_MIX',
        'batches': inventories, 'original_candidate_rows': len(rows),
        'excluded_later_content_holds': len(excluded), 'exported_unique_questions': len(kept),
        'roots': len({r['root_id'] for r in kept}),
        'question_types': dict(Counter(r['question_type'] for r in kept)),
        'mc_gold_histogram': dict(Counter(str(len(r['gold'])) for r in kept if r['question_type'] == 'multi_select')),
        'pool_binding': binding(out), 'exclusion_ledger_binding': binding(HERE / 'EXCLUSION_LEDGER.json'),
        'input_bindings': bindings, 'training_mix_built': False, 'new_training_runs': 0,
        'new_dialogue_turns': 0, 'solver_score_admission_filter': False,
        'solver_reasons_are_training_targets': False, 'formal_B_targeted_generation': False,
        'foundation_or_legacy_training_pools_included': False,
        'duplicates_policy': 'Only exact whole-question duplicates; partial overlap retained. No matches between the two batches.',
        'scope': 'These two new production batches only, not all historical project training inventories.'
    }
    save(HERE / 'POOL_MANIFEST.json', manifest)
    text = (f"# 两批生产候选池\n\n原两批内容通过共{len(rows)}题；后续独立审核暂缓{len(excluded)}题，当前合并候选{len(kept)}题、{manifest['roots']}个Train roots。\n\n"
            f"题型：{manifest['question_types']}；多选Gold分布：{manifest['mc_gold_histogram']}。\n\n"
            "TRAINING_POOL_CANDIDATES.jsonl每道独立题仅一条；未把两个展示当独立题。EXCLUSION_LEDGER.json保留后续内容疑点，原封存题库与成绩未改。\n\n"
            "这是两批新生产题的候选索引，不是最终训练混合；尚未加入历史基础/旧题池，未新训练。没有按solver分数剔题，模型理由不作为训练金标。下一次训练前仍须冻结混合比例、来源/事实族隔离、选项置换、固定验证集和监控配置。\n")
    with (HERE / 'README_ZH.md').open('x') as stream:
        stream.write(text)
    save(HERE / 'FINAL_MANIFEST.json', {'schema_version': 'cumulative_candidate_pool_seal_v2',
         'artifact_bindings': [binding(p) for p in sorted(HERE.iterdir()) if p.is_file()],
         'input_bindings': bindings})
    save(WEEK / 'PRODUCTION_POOL_LATEST.json', {'status': manifest['status'],
         'manifest': binding(HERE / 'POOL_MANIFEST.json'), 'pool': binding(out),
         'readme': binding(HERE / 'README_ZH.md'), 'questions': len(kept), 'roots': manifest['roots']})
    print(json.dumps({k: manifest[k] for k in ['original_candidate_rows', 'excluded_later_content_holds',
          'exported_unique_questions', 'roots', 'question_types', 'mc_gold_histogram']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
