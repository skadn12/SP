"""Retain a failed public review and normalize its explicit declarations only.

Restricted to the recorded root156136 public failure: rekey the unchanged
question list, add missing reread and two missing question-level declarations.
The actual reviewer supplies those declarations. No existing value can change.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review_public_reread_resume_v1 as previous


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_correction(original, corrected):
    require(isinstance(original, dict) and isinstance(corrected, dict), 'JUDGMENT_OBJECTS_REQUIRED')
    attestation = original.get('read_attestation')
    require(isinstance(attestation, dict), 'ORIGINAL_ATTESTATION_REQUIRED')
    require('final_public_questions_actually_reread' not in attestation, 'ONLY_MISSING_REREAD_FIELD_ALLOWED')
    questions = original.get('questions')
    require(isinstance(questions, list) and bool(questions), 'ORIGINAL_QUESTION_LIST_REQUIRED')
    require(all(isinstance(q, dict) and isinstance(q.get('candidate_id'), str) for q in questions), 'CANDIDATE_IDS_REQUIRED')
    require(len({q['candidate_id'] for q in questions}) == len(questions), 'DUPLICATE_CANDIDATE_ID')
    expected = deepcopy(original)
    expected['read_attestation']['final_public_questions_actually_reread'] = True
    expected['questions'] = {q['candidate_id']: q for q in expected['questions']}
    for q in expected['questions'].values():
        if 'options' in q:
            require(isinstance(q['options'], dict) and set(q['options']) == set('ABCDEF'), 'SIX_ORIGINAL_OPTIONS_REQUIRED')
            for field in ['fact_distinct', 'no_entangled_options']:
                if field not in q:
                    q[field] = True
    require(json.dumps(expected, sort_keys=True, ensure_ascii=False) ==
            json.dumps(corrected, sort_keys=True, ensure_ascii=False), 'ONLY_REKEY_AND_MISSING_DECLARATIONS_ALLOWED')


def engine():
    original = HERE / 'review_attestation_resume_v2.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == previous.ORIGINAL_SHA256
    source = original.read_text()
    replacements = [
        ('require(phase in {"asset", "public"}, "ONLY_SOURCE_FIRST_QA_REVIEW_PHASES")',
         'require(phase == "public" and root == "cc-train-episode-156136", "ONLY_RECORDED_156136_PUBLIC_FAILURE")'),
        ('failure.get("error") == "EXPLICIT_SOURCE_FIRST_ATTESTATION_REQUIRED"',
         'failure.get("error") == "EXPLICIT_FINAL_PUBLIC_REREAD_REQUIRED"'),
        ('"schema_version": "missing_source_first_attestation_resume_v2"',
         '"schema_version": "missing_public_schema_declarations_resume_v2"'),
        ('"only_added_field": "read_attestation.private_answers_read_before_save",\n                  "added_value": False, "semantic_dictionary_unchanged": True',
         '"only_added_fields": ["read_attestation.final_public_questions_actually_reread", "questions.*.fact_distinct", "questions.*.no_entangled_options"],\n                  "added_value": True, "question_list_rekeyed_by_unchanged_candidate_id": True, "existing_semantic_values_unchanged": True'),
        ('directory / "attestation_resume_v2"', 'directory / "public_schema_resume_v2"'),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, 'EXACT_PUBLIC_SCHEMA_ADAPTER_PATCH_REQUIRED'
        source = source.replace(old, new)
    namespace = {'__file__': str(Path(__file__).resolve()), '__name__': 'retained_public_schema_resume_v2'}
    exec(compile(source, str(original), 'exec'), namespace)
    namespace['validate_correction'] = validate_correction
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--executor', required=True)
    parser.add_argument('--judgment', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(engine()['resume'](args.batch, args.root, 'public', args.executor,
                                        args.judgment, args.engine, args.check_only), ensure_ascii=False))


if __name__ == '__main__':
    main()
