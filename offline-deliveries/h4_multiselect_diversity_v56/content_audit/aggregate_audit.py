"""Aggregate explicit independent reviews; does not infer labels or edit candidates."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent / 'h4_sequential_ab_v56_qa' / 'runs'


def main():
    questions = []
    chunks = []
    status = collections.Counter()
    root_status = collections.defaultdict(collections.Counter)
    ready_status = collections.Counter()
    cardinality = collections.Counter()
    files = []
    for audit_file in sorted((ROOT / 'content_audit').glob('root138343_c*_AUDIT.json')):
        chunk = json.loads(audit_file.read_text())
        run = RUNS / (chunk['chunk'] + '_generation')
        mechanical_path = run / 'MECHANICAL_POOL.json'
        mechanical = json.loads(mechanical_path.read_text())['questions']
        ready_path = run / 'SOLVER_POOL.json'
        ready = {q['candidate_id'] for q in json.loads(ready_path.read_text())['questions']}
        rows = {q['candidate_id']: q for q in mechanical}
        reviews = {q['candidate_id']: q for q in chunk['questions']}
        assert set(rows) == set(reviews), (str(run), 'audit_ids_differ')
        pool_sha = hashlib.sha256(mechanical_path.read_bytes()).hexdigest()
        for cid, review in reviews.items():
            assert review['mechanical_pool_sha256'] == pool_sha
            q = rows[cid]
            assert review['source_sha256'] == q['source_sha256']
            assert hashlib.sha256(Path(q['source_path']).read_bytes()).hexdigest() == q['source_sha256']
            assert review['checked_without_solver_scores']
            is_ready = cid in ready
            item = {'candidate_id': cid, 'root_id': q['root_id'], 'question_type': q['question_type'],
                    'ready_for_solver': is_ready, 'core_status': review['core_status'], 'audit': review}
            questions.append(item)
            status[review['core_status']] += 1
            root_status[q['root_id']][review['core_status']] += 1
            if is_ready:
                ready_status[review['core_status']] += 1
            if review['core_status'] == 'CORE_USABLE' and q['question_type'] == 'multi_select':
                cardinality[len(q['gold'])] += 1
        chunks.append(chunk['chunk'])
        files.append({'path': str(audit_file), 'sha256': hashlib.sha256(audit_file.read_bytes()).hexdigest()})
    out = {'scope': 'independent_read_only_source_content_audit_of_actual_new_v56_mechanical_candidates',
           'new_gate': False, 'generation_or_gold_edits': False, 'solver_scores_used': False,
           'cardinality_rule': '2_to_6_correct_allowed_no_per_question_quota',
           'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'completed_chunks': len(chunks), 'chunks': chunks,
           'mechanical_questions': len(questions), 'audited_questions': len(questions),
           'status_counts': dict(status), 'ready_status_counts': dict(ready_status),
           'by_root': {k: dict(v) for k, v in root_status.items()},
           'core_usable_multiselect_cardinality_histogram': dict(sorted(cardinality.items())),
           'review_files': files, 'questions': questions}
    (ROOT / 'INDEPENDENT_CONTENT_AUDIT.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    lines = ['# v56 独立内容审阅', '',
             '审阅来源全文、题干、六个选项与模型原始 Gold；未查看本轮 Solver 得分，未改题或 Gold，未增加机械门槛。单题允许 2–6 个正确项。', '',
             f'已完成 {len(chunks)} 个分块、{len(questions)} 道机械通过题。结果：{dict(status)}。', '',
             f'内容可用多选正确项数分布：{dict(sorted(cardinality.items()))}。', '',
             '| 题目 | 状态 | 原因 |', '|---|---|---|']
    for q in questions:
        a = q['audit']
        lines.append(f"| {q['candidate_id']} | {q['core_status']} | {a['summary']} |")
    lines += ['', '## 非阻断说明', '']
    for q in questions:
        for note in q['audit']['nonblocking_notes']:
            lines.append(f"- {q['candidate_id']}：{note}")
    (ROOT / 'INDEPENDENT_CONTENT_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'completed_chunks': len(chunks), 'audited_questions': len(questions), 'status_counts': dict(status)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
