"""Export unchanged, score-selected candidates; never merge or train."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

B = Path(__file__).resolve().parents[1]
def read(p): return json.loads(Path(p).read_text())
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def save(p, d):
    with p.open('x') as f: json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')


def main():
    scope = B / 'evaluations/paired_v1'
    poolp = scope / 'EVALUATION_POOL.json'; resultp = scope / 'run/trained/RESULTS.json'
    pool, results = read(poolp), read(resultp)
    by_id = {q['candidate_id']: q for q in results['questions']}
    selected = set(results['selected_questions'])
    assert len(by_id) == len(results['questions']) == len(pool['questions'])
    assert selected == {cid for cid, row in by_id.items() if row['selected_by_policy'] is True}
    candidates, disposition = [], []
    for q in pool['questions']:
        cid = q['candidate_id']; control = q['question_type'] == 'multi_select' and len(q['gold']) == 6
        accepted = cid in selected and not control
        source = pool['source_groups'][q['root_id']]
        status = 'CONTROL_ONLY' if control else 'TRAINING_CANDIDATE_NOT_MERGED' if accepted else 'CONTENT_PASS_NOT_SCORE_SELECTED'
        disposition.append({'candidate_id': cid, 'root_id': q['root_id'], 'question_type': q['question_type'],
            'gold_count': len(q['gold']) if q['question_type'] == 'multi_select' else None,
            'status': status, 'source_provenance': source, 'native_result': by_id[cid]})
        if accepted: candidates.append(q)
    now = datetime.now(timezone.utc).isoformat()
    save(B / 'delivery/TRAINING_CANDIDATES_V1.json', {
        'schema_version': 'v121_frozen_score_selected_candidate_export_v1', 'status': 'CANDIDATES_ONLY_NOT_MERGED_OR_TRAINED',
        'created_at_utc': now, 'actual_executor': '/root', 'selection_model': 'M_clean_policy',
        'native_policy': 'v15_full_only_random_mcq6x3_order12x1_v1',
        'selection_rule': 'Native policy with complete valid calls and 0 < full_partial_mean < 1; Gold6 excluded. No score-guided rewriting.',
        'candidate_count': len(candidates), 'root_count': len({q['root_id'] for q in candidates}),
        'question_type_counts': dict(Counter(q['question_type'] for q in candidates)),
        'multi_select_gold_counts': dict(Counter(len(q['gold']) for q in candidates if q['question_type'] == 'multi_select')),
        'source_provenance_counts': dict(Counter(pool['source_groups'][q['root_id']]['source_status'] for q in candidates)),
        'root_source_provenance': {root: pool['source_groups'][root] for root in sorted({q['root_id'] for q in candidates})},
        'provenance': pool['provenance'], 'source_pool_binding': bind(poolp), 'selection_results_binding': bind(resultp),
        'root_split_rule': 'Keep all original, edited, expanded and related questions from one root in the same split. This export creates no split or pool merge.',
        'questions': candidates, 'training_or_pool_merge': False})
    save(B / 'delivery/QUESTION_DISPOSITION_V1.json', {'schema_version': 'v121_question_disposition_v1',
        'created_at_utc': now, 'selection_model': 'M_clean_policy',
        'counts': dict(Counter(q['status'] for q in disposition)), 'questions': disposition, 'training_or_pool_merge': False})
    print(json.dumps({'status': 'CANDIDATES_EXPORTED_NOT_MERGED', 'candidates': len(candidates),
                      'roots': len({q['root_id'] for q in candidates}), 'dispositions': len(disposition)}))


if __name__ == '__main__': main()
