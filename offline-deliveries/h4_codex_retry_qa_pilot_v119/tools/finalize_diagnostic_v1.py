"""Serialize an independent content audit, then run the frozen finalizer.

This additive closeout does not repair the frozen-reviewer procedure deviation.
It cannot authorize training or promotion. No semantic judgments are supplied.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BATCH / 'scripts'))
sys.path.insert(0, str(BATCH / 'tools'))
import codex_sequential_questions as frozen
import review_v3


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def normalize(raw, source, public):
    """Map saved reviewer field names; never infer an answer or choose evidence."""
    attestation = raw['read_attestation']
    assert attestation['full_source_read_completed'] is True
    assert attestation['full_continuation_constraints_read_completed'] is True
    assert attestation['all_source_and_constraints_completed_before_any_public_question_read'] is True
    assert attestation['source_turn_count'] == len(source['turns'])
    assert attestation['no_existing_answers_consulted'] is True
    assert attestation['prohibited_input_content_reads'] == []
    assert all(attestation[key] is False for key in (
        'read_qa_draft', 'read_private', 'read_asset_source_judgments',
        'read_composition', 'read_solver', 'read_other_reviews_or_answers'))
    turns = {t['turn_ref']: t for t in source['turns']}

    def anchor(value):
        result = deepcopy(value)
        # The reviewer explicitly saved the phrase and its offset. Use that
        # exact preselected span to preserve clause order within shared turns.
        if 'anchor_phrase' in result:
            phrase = result['anchor_phrase']
            turn = turns[result['turn_ref']]
            assert phrase in result['quote']
            assert turn['text'].count(phrase) == 1
            assert turn['text'].index(phrase) == result['clause_start_character']
            result['full_turn_quote'] = result['quote']
            result['quote'] = phrase
        return result

    result = deepcopy(raw)
    bindings = {r['role']: r for r in raw['input_bindings']}
    result.update(
        reviewer=raw['executor'],
        source_sha256=bindings['cumulative_source']['sha256'],
        public_sha256=bindings['public_questions']['sha256'],
        full_source_read=attestation['full_source_read_completed'],
        turn_refs_read=[t['turn_ref'] for t in source['turns']],
        private_answers_read_before_save=attestation['read_private'],
        independent_of_question_author=attestation['no_existing_answers_consulted'],
        findings=deepcopy(raw['summary']['invalid_or_nonunique_candidates']),
        questions={},
    )
    expected = {q['candidate_id']: q for q in public['questions']}
    assert len(raw['questions']) == len(expected)
    for original in raw['questions']:
        question = expected[original['candidate_id']]
        value = deepcopy(original)
        if question['question_type'] == 'multi_select':
            value['options'] = {}
            for label, item in original['option_reviews'].items():
                assert item['option_text'] == question['options'][label]
                value['options'][label] = {
                    **deepcopy(item), 'proposition': item['option_text'],
                    'relation': item['judgment'],
                    'scope_valid': original['scope_valid'],
                    'citations': deepcopy(item['source_refs']),
                }
        else:
            for label, item in value['items'].items():
                assert item['item_text'] == question['items'][label]
                item['proposition'] = item['item_text']
                item['support_ref_citations'] = deepcopy(item['support_refs'])
                item['support_refs'] = list(dict.fromkeys(x['turn_ref'] for x in item['support_refs']))
                item['order_anchor'] = anchor(item['order_anchor'])
                item['alternative_matching_anchors'] = [anchor(x) for x in item['alternative_matching_anchors']]
        result['questions'][value['candidate_id']] = value
    assert set(result['questions']) == set(expected)
    result['serialization_mapping'] = {
        'executor': 'reviewer', 'input_bindings': 'source_sha256/public_sha256',
        'read_attestation': 'explicit header attestations plus source ref expansion',
        'questions_list': 'questions_by_candidate_id',
        'option_reviews': 'options', 'option_text/item_text': 'proposition',
        'judgment': 'relation', 'source_refs': 'citations',
        'question_scope_valid': 'option_scope_valid',
        'support_refs_citations': 'support_refs turn IDs; original citations retained',
        'order_anchor_phrase': 'quote exact reviewer-selected span; original full turn retained',
        'semantic_judgments_changed': False,
    }
    return result


def main():
    policy = read(BATCH / 'DIAGNOSTIC_CLOSEOUT_POLICY.json')
    assert policy['batch_procedural_success'] is False
    assert policy['training_pool_eligible'] is False
    for row in read(BATCH / 'MASTER_FREEZE.json')['files']:
        assert bind(row['path']) == row, 'ORIGINAL_FREEZE_CHANGED:' + row['path']
    manifest = read(BATCH / 'MANIFEST.json')
    roles = {r['root_id']: r for r in read(BATCH / 'ROLE_ASSIGNMENTS.json')['roots']}
    prepared = []
    # All source reviews must validate before any private answer is read.
    for row in manifest['roots']:
        folder = Path(row['root_path'])
        source = read(folder / 'CUMULATIVE_SOURCE.json')
        public = read(folder / 'PUBLIC_QUESTIONS.json')
        review_path = folder / 'FINAL_BLIND_SOURCE_REVIEW.json'
        raw = read(review_path)
        for item in raw['input_bindings']:
            assert bind(item['path']) == {k: item[k] for k in ('path', 'sha256')}
        assert raw['executor'] != roles[row['root_id']]['author']
        review = normalize(raw, source, public)
        assert review['root_id'] == row['root_id']
        assert review['reviewer'] == policy['independent_content_reviewer']
        assert review['private_answers_read_before_save'] is False
        assert review['independent_of_question_author'] is True
        assert review['full_source_read'] is True
        assert review['turn_refs_read'] == [t['turn_ref'] for t in source['turns']]
        assert review['source_sha256'] == bind(folder / 'CUMULATIVE_SOURCE.json')['sha256']
        assert review['public_sha256'] == bind(folder / 'PUBLIC_QUESTIONS.json')['sha256']
        data = {'phase': 'public', 'public': public,
                'turns': {t['turn_ref']: t for t in source['turns']}}
        review_v3.validate_qa(review, data)
        assert not (folder / 'PUBLIC_SOURCE_JUDGMENTS.json').exists()
        assert not (folder / 'PRIVATE_GOLD.json').exists()
        canonical = deepcopy(review)
        canonical.update({
            'schema_version': 'diagnostic_independent_content_review_serialization_v1',
            'raw_independent_review_binding': bind(review_path),
            # Only a crosscheck dependency; the blind reviewer did not read it.
            'asset_judgments_sha256': bind(folder / 'ASSET_SOURCE_JUDGMENTS.json')['sha256'],
            'asset_binding_added_by_mechanical_serializer': True,
            'serializer_executor': '/root',
            'serializer_binding': bind(__file__),
            'closeout_policy_binding': bind(BATCH / 'DIAGNOSTIC_CLOSEOUT_POLICY.json'),
            'final_public_questions_actually_reread': True,
            'batch_procedural_success': False,
            'training_pool_eligible': False,
            'review_scope': 'Independent additional content audit; original role deviation remains.',
        })
        prepared.append((row, folder, canonical))
    for row, folder, canonical in prepared:
        save(folder / 'PUBLIC_SOURCE_JUDGMENTS.json', canonical)
        # Frozen checks independently derive the source answer, then compare it
        # against the unchanged private design, including the frozen anchors.
        frozen.finalize(Path(row['unit_path']))
        cross = read(folder / 'FINAL_CROSSCHECK.json')
        save(folder / 'DIAGNOSTIC_CONTENT_DISPOSITION.json', {
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'root_id': row['root_id'],
            'status': 'CONTENT_AUDITED_PROCEDURE_HOLD',
            'batch_procedural_success': False,
            'training_pool_eligible': False,
            'qa_retry_attempt': 1,
            'additional_generation_attempts': 0,
            'original_design_unchanged': True,
            'source_review_serialization_binding': bind(folder / 'PUBLIC_SOURCE_JUDGMENTS.json'),
            'frozen_final_crosscheck_binding': bind(folder / 'FINAL_CROSSCHECK.json'),
            'composition_receipt_binding': bind(folder / 'COMPOSITION_RECEIPT.json'),
            'frozen_finalizer_binding': bind(frozen.__file__),
            'closeout_policy_binding': bind(BATCH / 'DIAGNOSTIC_CLOSEOUT_POLICY.json'),
            'content_pass': sum(r['status'] == 'CONTENT_PASS' for r in cross['results']),
            'content_excluded': sum(r['status'] == 'EXCLUDED' for r in cross['results']),
            'training_started': False,
            'training_pool_merged': False,
        })


if __name__ == '__main__':
    main()
