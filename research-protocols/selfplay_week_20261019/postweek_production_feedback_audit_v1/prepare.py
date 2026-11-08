"""Prospectively bind a semantic review of every FULL response in batch 1."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = HERE.parent / 'postweek_production_batch_v1'


def binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    assert not (HERE / 'REVIEW_FREEZE.json').exists(), 'NO_OVERWRITE'
    manifest = read(BATCH / 'FINAL_MANIFEST.json')
    for b in manifest['artifact_bindings'] + manifest['external_reference_bindings']:
        assert binding(b['path']) == b, b['path']
    candidates = read(BATCH / 'CANDIDATES_PRIVATE.json')
    scores = {r['request_id']: r for r in read(BATCH / 'execution/SCORES.json')['records']}
    packets = read(BATCH / 'execution/REASON_REVIEW_PACKET.json')['requests']
    full = [r for r in packets if r['condition'] == 'FULL']
    roots = sorted({c['root_id'] for c in candidates})
    groups = {'a': roots[:5], 'b': roots[5:]}
    source_bindings = []
    for root in roots:
        candidate = next(c for c in candidates if c['root_id'] == root)
        bound = candidate['source_binding']
        assert binding(bound['path']) == bound
        source = read(bound['path'])
        transcript = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns']) + '\n'
        path = HERE / 'sources' / (root + '.txt')
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            assert path.read_text() == transcript, 'PARTIAL_PREPARATION_TRANSCRIPT_DRIFT'
        else:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(transcript)
        source_bindings.append({'root_id': root, 'source_binding': bound,
                                'transcript_binding': binding(path), 'turns': len(source['turns'])})
    for group, selected in groups.items():
        rows = []
        for c in candidates:
            if c['root_id'] not in selected:
                continue
            responses = []
            for r in full:
                if r['candidate_id'] != c['candidate_id']:
                    continue
                assert binding(r['raw_binding']['path']) == r['raw_binding']
                choices = c['options' if c['question_type'] == 'multi_select' else 'items']
                display_map = {label: next(k for k, v in choices.items() if v == text)
                               for label, text in r['request']['display'].items()}
                responses.append({'request_id': r['request_id'], 'display': r['request']['display'],
                                  'display_to_canonical': display_map, 'raw_binding': r['raw_binding'],
                                  'status': r['status'], 'parsed': r.get('parsed'),
                                  'invalid_error': r.get('invalid_error'),
                                  'mechanical_consistency': r.get('mechanical_consistency'),
                                  'score': scores[r['request_id']]})
            assert len(responses) == 2
            rows.append({'candidate': c, 'responses': responses})
        save(HERE / f'REVIEW_PACKET_{group.upper()}.json', {
            'group': group, 'roots': selected,
            'sources': [b for b in source_bindings if b['root_id'] in selected],
            'questions': rows})
    freeze = {
        'schema_version': 'production_full_semantic_audit_freeze_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection': 'All 132 FULL responses to all 66 admitted questions, both frozen displays; no QUESTION_ONLY responses.',
        'selection_uses_answer_correctness': False,
        'questions': len(candidates), 'responses': len(full), 'roots': len(roots),
        'score_counts': dict(Counter('EXACT' if scores[r['request_id']]['exact'] else 'NOT_EXACT' for r in full)),
        'groups': groups, 'new_solver_calls': 0, 'question_rewrites': False,
        'formal_weakness_admission': False, 'training_admission': False,
        'reviewer_type': 'independent_codex_ai_agent_gold_visible_not_human',
        'preparation_repair': 'Initial CPU preparation stopped on invalid response lacking parsed; retained raw invalid records with parsed=null. Existing identical transcripts verified, no model calls or response edits.',
        'review_contract': {
            'per_question': ['candidate_id', 'root_id', 'content_decision', 'content_notes', 'reconstructed_gold', 'responses'],
            'per_response': ['request_id', 'items', 'answer_reason_support', 'summary_final_consistency',
                             'item_final_consistency', 'error_dimensions', 'notes'],
            'per_display_item': ['judgment_correct', 'citation_support', 'reason_support', 'reviewer_turn_refs', 'note'],
            'statuses': ['PASS', 'FAIL', 'UNKNOWN', 'NA'],
            'dimensions': ['CONTENT_OR_GOLD', 'SOURCE_STATE', 'ENTITY_BINDING', 'QUANTITY_OR_TIME',
                           'RELATION_OR_CONSTRAINT', 'CITATION_UNSUPPORTED', 'REASON_UNSUPPORTED',
                           'ITEM_FINAL_MISMATCH', 'SUMMARY_FINAL_MISMATCH', 'ORDER_RELATION',
                           'INCOMPLETE_OUTPUT', 'FORMAT_OR_LIMIT'],
            'rules': [
                'Read each complete bound source and actual parsed response; reconstruct content independently.',
                'Evaluate display labels against the displayed statements, never canonical letters directly.',
                'Citations must support the submitted judgment and reason; nearby or topically relevant is insufficient.',
                'A false option correctly marked contradicted can have a fully supported reason.',
                'Ordering support concerns item content separately from pair/order rationale.',
                'Ordering explanation of a proposed step does not mean the event has already occurred.',
                'Unknown and invalid outputs remain explicit, never repaired or omitted.',
                'Visible reasons are outputs, not faithful internal chain-of-thought or causal diagnoses.',
                'Any source/gold concern is retained as a post-freeze concern, with no changes to batch 1.',
                'No internal-cause, unseen-root, weakness-reproduction or feedback-gain claim.'
            ]
        },
        'bindings': [binding(BATCH / n) for n in ['FINAL_MANIFEST.json', 'CANDIDATES_PRIVATE.json',
                     'CONTENT_REVIEW_AI.json', 'execution/REASON_REVIEW_PACKET.json', 'execution/SCORES.json']]
                    + [binding(HERE / f'REVIEW_PACKET_{g.upper()}.json') for g in groups]
                    + [binding(Path(__file__))],
        'sources': source_bindings,
    }
    assert len(full) == 132 and len(candidates) == 66
    save(HERE / 'REVIEW_FREEZE.json', freeze)
    print(json.dumps({'questions': len(candidates), 'responses': len(full), 'groups': groups}))


if __name__ == '__main__':
    main()
