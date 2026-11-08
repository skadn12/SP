"""Compare already frozen blind answers; never edits or retries reviewer output."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(folder):
    raw_path = folder / 'BLIND_RAW.json'
    if not raw_path.exists() or (folder / 'BLIND_COMPARISON.json').exists():
        return
    private = read(folder / 'DISPLAY_PRIVATE.json')
    program = read(folder / 'PROGRAM_CHECK.json')
    packet = read(folder / 'PUBLIC_PACKET.json')
    errors = []
    try:
        raw = read(raw_path)
    except (ValueError, UnicodeError) as exc:
        raw = {}
        errors.append('UNPARSEABLE:' + str(exc))
    required = ('candidate_id', 'reviewer_type', 'answer_status', 'answer_labels',
                'option_judgments', 'evidence_read', 'hard_issues', 'soft_issues')
    if not isinstance(raw, dict):
        errors.append('REVIEW_NOT_AN_OBJECT')
        raw = {}
    for key in required:
        if key not in raw:
            errors.append('MISSING_FIELD:' + key)
    if raw.get('candidate_id') != private['candidate_id']:
        errors.append('CANDIDATE_ID_MISMATCH')
    if raw.get('reviewer_type') != 'same_model_isolated_blind':
        errors.append('REVIEWER_TYPE_MISMATCH')
    if raw.get('answer_status') not in ('SOLVED', 'HOLD'):
        errors.append('INVALID_ANSWER_STATUS')
    for key in ('hard_issues', 'soft_issues'):
        if not isinstance(raw.get(key), list):
            errors.append('WRONG_FIELD_TYPE:' + key)
    labels = raw.get('answer_labels')
    valid_labels = set(private['label_to_option_id'])
    labels_valid = (isinstance(labels, list)
                    and all(isinstance(label, str) for label in labels)
                    and len(set(labels)) == len(labels)
                    and set(labels) <= valid_labels)
    if not labels_valid:
        errors.append('INVALID_ANSWER_LABELS')
    if raw.get('answer_status') == 'SOLVED' and private['question_type'] == 'full_order':
        if not labels_valid or set(labels) != valid_labels:
            errors.append('ORDER_IS_NOT_A_FULL_PERMUTATION')
    if sha(folder / 'PUBLIC_PACKET.json') != program['public_packet_sha256']:
        errors.append('PUBLIC_PACKET_CHANGED')
    agreement = None
    semantic_answer = None
    if not errors and raw['answer_status'] == 'SOLVED':
        semantic_answer = [private['label_to_option_id'][label] for label in labels]
        agreement = (set(semantic_answer) == set(private['semantic_gold'])
                     if private['question_type'] == 'multi_select'
                     else semantic_answer == private['semantic_gold'])
    reasons = list(errors)
    if raw.get('answer_status') == 'HOLD':
        reasons.append('BLIND_REVIEW_HOLD')
    if agreement is False:
        reasons.append('BLIND_GOLD_DISAGREEMENT')
    if raw.get('hard_issues'):
        reasons.append('BLIND_HARD_ISSUES')
    if program['status'] != 'PASS':
        reasons.append('PROGRAM_DATA_CHECK_HOLD')
    result = {'candidate_id': private['candidate_id'],
              'reviewer_type': 'same_model_isolated_blind',
              'single_question_context': True,
              'review_raw_sha256': sha(raw_path),
              'public_packet_sha256': sha(folder / 'PUBLIC_PACKET.json'),
              'blind_answer_labels': labels,
              'blind_semantic_answer': semantic_answer,
              'frozen_author_gold_labels': private['display_gold'],
              'reviewer_protocol_valid': not errors,
              'reviewer_protocol_errors': errors,
              'gold_agreement': agreement,
              'content_review_status': 'CONTENT_REVIEW_PASS' if not reasons else 'HOLD',
              'hold_reasons': reasons,
              'hard_issues': raw.get('hard_issues'), 'soft_issues': raw.get('soft_issues'),
              'formal_qa_admission': 'NOT_GRANTED_BY_THIS_RUN',
              'source_review_context': packet['source']['existing_review_context'],
              'score': None,
              'limitations': ['Agreement is same-model blind agreement, not independent-model verification.',
                             'No Solver was run; content acceptance is not a difficulty result.',
                             'No question is rejected merely for low auxiliary difficulty.',
                             'Source authorization is tracked separately from text-supported correctness.']}
    with (folder / 'BLIND_COMPARISON.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('candidate_id', 'gold_agreement',
                                                  'content_review_status', 'hold_reasons')},
                     ensure_ascii=False))


if __name__ == '__main__':
    for folder in sorted((BASE / 'items').iterdir()):
        if folder.is_dir():
            compare(folder)
