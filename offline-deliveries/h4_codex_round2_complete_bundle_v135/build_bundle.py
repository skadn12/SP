"""Create a relocatable additive delivery; never edit the sealed inputs."""
from pathlib import Path
from collections import Counter
import csv
import copy
import datetime
import hashlib
import json
import shutil
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parent
P = B.parent
VERSIONS = {
    'v132': 'h4_codex_round2_data_bundle_v132',
    'v133': 'h4_codex_held_roots_retry_v133',
    'v134': 'h4_codex_held_roots_eval_v134',
}

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    assert not dst.exists(), dst
    shutil.copy2(src, dst)

def main():
    snapshots = {}
    for version, name in VERSIONS.items():
        src = P / name
        dst = B / 'snapshots' / version
        manifest_path = src / 'COMPLETE_MANIFEST.json'
        manifest = read(manifest_path)
        assert not dst.exists(), dst
        for row in manifest['files']:
            rel = Path(row['path'])
            assert not rel.is_absolute() and '..' not in rel.parts, rel
            f = src / rel
            assert f.is_file() and f.stat().st_size == row['bytes']
            assert sha(f) == row['sha256'], str(f)
            cp(f, dst / rel)
        cp(manifest_path, dst / 'COMPLETE_MANIFEST.json')
        snapshots[version] = {
            'original_path': str(src), 'bundle_path': str(dst.relative_to(B)),
            'manifest_sha256': sha(manifest_path),
            'bound_file_count': len(manifest['files']),
            'status': 'ALL_ORIGINAL_MANIFEST_BINDINGS_VERIFIED_AND_COPIED',
        }
        print(f'{version}: preserved {len(manifest["files"])} files', flush=True)
    save(B / 'SOURCE_SNAPSHOTS.json', snapshots)
    old = B / 'snapshots/v132'
    retry = B / 'snapshots/v133'
    latest = B / 'snapshots/v134'
    root_rows = copy.deepcopy(read(old / 'ROOT_INDEX.json')['roots'])
    new_sources = {r['root_id']: r for r in read(retry / 'ROOT_INDEX.json')['roots']}
    latest_roots = {r['root_id']: r for r in read(latest / 'ROOT_MANIFEST.json')['roots']}
    assert set(new_sources) == set(latest_roots) and len(new_sources) == 7
    roots = {r['root_id']: r for r in root_rows}
    assert len(roots) == 99
    shutil.copytree(old / 'data/train', B / 'data/train')
    question_index = []
    statistics = {}
    pools = {}
    all_ids = set()
    for split, expected in [('train', 363), ('dev', 83), ('test', 101)]:
        if split == 'train':
            pool = read(old / 'data/train/questions_with_answers.json')['questions']
        else:
            pool = read(latest / f'delivery/{split}/EXTENDED_QUESTIONS_WITH_ANSWERS.json')['questions']
            old_pool = read(old / f'data/{split}/questions_with_answers.json')['questions']
            assert pool[:len(old_pool)] == old_pool
            save(B / f'data/{split}/questions_with_answers.json', {
                'schema_version': 'v135_additive_canonical_pool_v1',
                'evaluation_only': True, 'canonical_records_modified': False,
                'old_count': len(old_pool), 'added_count': len(pool) - len(old_pool),
                'evaluation_index': 'EVALUATION_INDEX.json', 'questions': pool,
            })
            with (B / f'data/{split}/questions_with_answers.jsonl').open('x') as f:
                for q in pool:
                    f.write(json.dumps(q, ensure_ascii=False) + '\n')
        assert len(pool) == expected
        pools[split] = pool
        grouped = {}
        for q in pool:
            assert q['candidate_id'] not in all_ids
            all_ids.add(q['candidate_id'])
            assert roots[q['root_id']]['split'] == split
            grouped.setdefault(q['root_id'], []).append(q)
        for r in [r for r in root_rows if r['split'] == split]:
            rid = r['root_id']
            qs = grouped.get(rid, [])
            previous_status = r['status']
            if rid in new_sources:
                nr = new_sources[rid]
                src = retry / nr['dialogue_path']
                original = old / r['dialogue_path']
                assert read(src)['turns'][:len(read(original)['turns'])] == read(original)['turns']
                cp(original, B / f'data/{split}/dialogue_history/{rid}/v132.json')
                r['historical_dialogue_path'] = f'data/{split}/dialogue_history/{rid}/v132.json'
                r['historical_dialogue_sha256'] = sha(original)
                r['historical_status'] = previous_status
                r['historical_original_source_path'] = r['original_source_path']
                r['original_source_path'] = latest_roots[rid]['source_path']
                r['status'] = 'CONTENT_PASS_NEW_SCENES_HISTORICAL_CONFLICTS_RETAINED'
                r['baseline_conflicts_resolved'] = False
                r['current_dialogue_version'] = 'v133_scope_isolated_expansion'
                r['ready_scene_count'] = 6
                r['all_six_scenes_materialized'] = True
                r['new_r02_turns'] = nr['new_turns']
                r['new_r02_scene_turn_counts'] = dict(Counter(
                    t['scene_ref'] for t in read(src)['turns']
                    if str(t.get('scene_ref', '')).startswith('R02:')))
                assert sha(src) == latest_roots[rid]['source_sha256']
            else:
                src = old / r['dialogue_path']
                r['current_dialogue_version'] = 'v132_preserved'
            local_dialogue = f'data/{split}/dialogues/{rid}.json'
            if split != 'train':
                cp(src, B / local_dialogue)
            r['dialogue_path'] = local_dialogue
            r['dialogue_sha256'] = sha(B / local_dialogue)
            r['question_count'] = len(qs)
            r['candidate_ids'] = [q['candidate_id'] for q in qs]
            r['question_type_counts'] = dict(Counter(q['question_type'] for q in qs))
            r['usage'] = 'ACTUALLY_TRAINED' if split == 'train' else 'EVALUATION_ONLY'
            r['total_turns'] = len(read(B / local_dialogue)['turns'])
            r['question_path'] = f'data/{split}/questions_by_root/{rid}.json'
            r['readable_path'] = f'data/{split}/readable/{rid}.md'
            if split != 'train':
                save(B / r['question_path'], {
                    'root_id': rid, 'split': split, 'usage': r['usage'],
                    'status': r['status'], 'dialogue_path': local_dialogue,
                    'dialogue_sha256': r['dialogue_sha256'], 'questions': qs,
                })
                lines = [f'# {rid}', '', f'Split: {split}; questions: {len(qs)}; status: {r["status"]}', '',
                         '当前来源和历史状态见包根 ROOT_INDEX.json。题目保留原始 Gold、证据与来源字段。', '', '## 完整对话', '']
                for t in read(B / local_dialogue)['turns']:
                    lines.extend([f'**{t.get("turn_ref", "")} · {t.get("speaker", "")}**', '', t['text'], ''])
                lines.extend(['## 对应题目与答案', ''])
                for i, q in enumerate(qs, 1):
                    lines.extend([f'### {i}. {q["candidate_id"]}', '', q['question'], ''])
                    for label, text in q.get('options', q.get('items', {})).items():
                        lines.append(f'- {label}: {text}')
                    lines.extend(['', f'Gold: {json.dumps(q["gold"], ensure_ascii=False)}', ''])
                lines.extend([f'完整证据：[逐 root 题库](../questions_by_root/{rid}.json)', ''])
                dest = B / r['readable_path']
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text('\n'.join(lines))
            for q in qs:
                assert q['source_sha256'] == r['dialogue_sha256'], q['candidate_id']
                question_index.append({
                    'candidate_id': q['candidate_id'], 'root_id': rid, 'split': split,
                    'question_type': q['question_type'], 'actual_training_record': split == 'train',
                    'original_source_path': q['source_path'], 'source_sha256': q['source_sha256'],
                    'bundle_dialogue_path': local_dialogue,
                    'bundle_question_path': r['question_path'],
                    'production_batch': 'v134' if rid in new_sources else ('v132_train_snapshot' if split == 'train' else 'v124'),
                    'latest_49_evaluation': rid in new_sources,
                })
        public = [{k: q[k] for k in ['candidate_id', 'root_id', 'question_type', 'question', 'options', 'items'] if k in q} for q in pool]
        save(B / f'data/{split}/questions_public.json', {'questions': public})
        split_roots = [r for r in root_rows if r['split'] == split]
        statistics[split] = {
            'fixed_roots': len(split_roots), 'roots_with_questions': len(grouped),
            'questions': len(pool), 'question_types': dict(Counter(q['question_type'] for q in pool)),
            'multi_select_gold_counts': dict(sorted(Counter(str(len(q['gold'])) for q in pool if q['question_type'] == 'multi_select').items())),
            'actual_training_records': len(pool) if split == 'train' else 0,
        }
    save(B / 'ROOT_INDEX.json', {'schema_version': 'v135_root_index_v1', 'roots': root_rows})
    save(B / 'QUESTION_SOURCE_INDEX.json', {'schema_version': 'v135_question_source_index_v1', 'questions': question_index})
    save(B / 'DATA_STATISTICS.json', {
        'schema_version': 'v135_statistics_v1', 'splits': statistics,
        'total_roots': 99, 'roots_with_questions': 95, 'total_questions': len(all_ids),
        'actual_training_records': 363, 'latest_added_evaluation_questions': 49,
        'new_training_runs': 0, 'new_model_inference_calls': 0,
    })
    with (B / 'ROOT_INDEX.csv').open('x', newline='') as f:
        fields = ['root_id', 'split', 'usage', 'status', 'question_count', 'total_turns', 'dialogue_path', 'question_path', 'readable_path']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(root_rows)
    lines = ['# 第二轮统一 root 索引', '', '99 个固定 root，95 个有题。无题 root 保留现状和失败记录。', '',
             '| Split | Root | 题数 | 完整对话与题目 | 当前状态 |', '|---|---|---:|---|---|']
    for r in root_rows:
        lines.append(f'| {r["split"]} | {r["root_id"]} | {r["question_count"]} | [阅读]({r["readable_path"]}) | {r["status"]} |')
    (B / 'ROOT_INDEX.md').write_text('\n'.join(lines) + '\n')
    save(B / 'EVALUATION_INDEX.json', {
        'schema_version': 'v135_evaluation_index_v1',
        'note': 'Existing result snapshots; no new inference or whole-library re-evaluation.',
        'results': [
            {'scope': 'original_fixed_dev_48', 'question_count': 48, 'report': 'snapshots/v132/evaluations/dev_v127/REPORT_ZH.md', 'details': 'M_clean vs second-round; 780 calls/model'},
            {'scope': 'original_dev_ordering_14_base_control', 'question_count': 14, 'report': 'snapshots/v132/evaluations/original_base_ordering_v130/REPORT_ZH.md'},
            {'scope': 'historical_round1_diagnostic_not_part_of_547', 'question_count': 123, 'report': 'snapshots/v132/evaluations/historical_round1_frontier_v131/REPORT_ZH.md'},
            {'scope': 'latest_new_dev35_test14', 'question_count': 49, 'calls': 2394, 'report': 'snapshots/v134/evaluation/reports/REPORT_ZH.md', 'summary': 'snapshots/v134/evaluation/reports/SUMMARY.json', 'raw_responses': 'snapshots/v134/evaluation/run'},
        ],
        'entire_547_evaluated_after_training': False,
        'entire_extended_dev_test_184_evaluated_here': False,
        'test_historical_exposure': True,
    })
    save(B / 'DATA_BUILD_STATUS.json', {'status': 'DATA_AND_SNAPSHOTS_COMPLETE', 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'questions': len(all_ids)})
    print(json.dumps(statistics, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
