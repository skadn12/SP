"""Validate and bind seven explicitly authored source-only review records."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

BATCH = Path(__file__).resolve().parents[1]
GATE_PATH = BATCH / 'evidence/source_gate_disclosure_v89.py'
spec = importlib.util.spec_from_file_location('v89_source_gate_disclosure', GATE_PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
NUMBERS = (25652, 29753, 31457, 38601, 43599, 45849, 50175)


def main():
    inventory_path = BATCH / 'inventory/CANDIDATE_ROOTS_20.json'
    inventory = gate.read(inventory_path)['candidates']
    indexed = {int(r['root_id'].rsplit('-', 1)[1]): r for r in inventory}
    inventory_binding = gate.bind(inventory_path)
    reviews = []
    rows = []
    for n in NUMBERS:
        record = indexed[n]
        review_path = BATCH / 'source_reviews' / f'root{n}_REVIEW.json'
        review = gate.validate_review({'root_id': record['root_id'],
            'source': record['source'], 'review_path': str(review_path)})
        reviews.append(review)
        rows.append({'root_id': record['root_id'], 'source': record['source'],
            'review': gate.bind(review_path), 'decision': review['decision'],
            'full_source_turn_count': len(review['turn_refs_read']),
            'complete_units_reviewed': len(review['units']),
            'unit_decisions': dict(Counter(u['decision'] for u in review['units'])),
            'blocking_findings': [f['finding_id'] for f in review['findings']
                                  if f['severity'] == 'BLOCKING'],
            'warning_findings': [f['finding_id'] for f in review['findings']
                                 if f['severity'] == 'WARNING'],
            'blocking_reasons': [f['reason'] for f in review['findings']
                                if f['severity'] == 'BLOCKING'],
            'exact_review_contract_passed': True,
            'source_unchanged': gate.bind(record['source']['path']) == record['source']})
    assert gate.bind(inventory_path) == inventory_binding
    output = {'schema_version': 'v89_rank14_20_source_review_receipt_v1',
        'created_at_utc': gate.now(), 'inventory_unchanged': inventory_binding,
        'serializer': gate.bind(BATCH / 'evidence/write_rank14_20_source_reviews.py'),
        'summary_builder': gate.bind(__file__), 'validator': gate.bind(GATE_PATH),
        'root_count': len(rows), 'turn_count': sum(r['full_source_turn_count'] for r in rows),
        'unit_count': sum(r['complete_units_reviewed'] for r in rows), 'roots': rows,
        'ready_roots': [r['root_id'] for r in rows if r['decision'] == 'READY'],
        'held_roots': [r['root_id'] for r in rows if r['decision'] == 'HOLD'],
        'latest_review_at_utc': max(r['created_at_utc'] for r in reviews),
        'exposure_disclosure': {
            'historical_question_gold_known': False, 'fresh_blind_review': False,
            'inventory_program_loaded_historical_pools_and_inspected_gold_lengths': True,
            'concrete_question_gold_or_evidence_shown_to_reviewer_for_these_roots': False,
            'current_question_gold_read': False, 'current_solver_scores_read': False,
            'reason_nonblind': 'Conservative disclosure of earlier inventory process '
                               'contact with historical pool fields and locked feedback; '
                               'not a claim of concrete Gold knowledge for these roots.'},
        'semantic_review_method': 'Complete exact-source reading and explicitly authored '
            'per-unit/cross-unit judgments on time, location, identity, plan/completion, '
            'continuity and repetition. Ordinary time gaps, changing stated plans, '
            'uncertainty and distinct repeated events are not automatically conflicts.',
        'automatic_validation_scope': 'Schema, truthful exposure booleans, complete turn/unit '
            'coverage, exact quotes, derived dispositions and frozen source bindings only; '
            'automatic validation does not certify authored semantic conclusions.',
        'full_source_hold_policy': 'Any unresolved BLOCKING unit holds the entire source. '
            'Source omissions, actor renaming and unsupported alias invention are prohibited.',
        'ready_is_question_answer_certification': False,
        'ready_is_parent_question_already_exists': False,
        'training_allowed': False, 'historical_files_modified': 0,
        'source_edits': [], 'question_or_gold_edits': [],
        'http_calls': 0, 'solver_calls': 0, 'model_calls': 0}
    path = BATCH / 'evidence/RANK14_20_SOURCE_REVIEW_RECEIPT.json'
    gate.write(path, output)
    print(json.dumps({'receipt': gate.bind(path), 'ready_roots': output['ready_roots'],
        'held_roots': output['held_roots'], 'turn_count': output['turn_count'],
        'unit_count': output['unit_count']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
