"""Resume a rejected public review by adding only its missing reread declaration.

The assigned reviewer must actually have reread the final public questions.
No source, first RAW, failure, semantic judgment, private access declaration,
or Gold is changed. Explicit false declarations cannot be reversed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ORIGINAL_SHA256 = 'ba7eb88bc5192fbc58b4570221a5c16837be15d783514f53bb6e2e4bc909ae8a'


def validate_correction(original, corrected):
    if not isinstance(original, dict) or not isinstance(corrected, dict):
        raise ValueError('JUDGMENT_OBJECTS_REQUIRED')
    att = original.get('read_attestation')
    if not isinstance(att, dict):
        raise ValueError('ORIGINAL_ATTESTATION_REQUIRED')
    if 'final_public_questions_actually_reread' in att:
        raise ValueError('ONLY_MISSING_REREAD_FIELD_ALLOWED')
    expected = deepcopy(original)
    expected['read_attestation']['final_public_questions_actually_reread'] = True
    if json.dumps(expected, sort_keys=True, ensure_ascii=False) != json.dumps(corrected, sort_keys=True, ensure_ascii=False):
        raise ValueError('ONLY_FINAL_PUBLIC_REREAD_ADDITION_ALLOWED')


def engine():
    original = HERE / 'review_attestation_resume_v2.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == ORIGINAL_SHA256
    source = original.read_text()
    replacements = [
        ('require(phase in {"asset", "public"}, "ONLY_SOURCE_FIRST_QA_REVIEW_PHASES")',
         'require(phase == "public", "ONLY_FINAL_PUBLIC_REVIEW_PHASE")'),
        ('failure.get("error") == "EXPLICIT_SOURCE_FIRST_ATTESTATION_REQUIRED"',
         'failure.get("error") == "EXPLICIT_FINAL_PUBLIC_REREAD_REQUIRED"'),
        ('"schema_version": "missing_source_first_attestation_resume_v2"',
         '"schema_version": "missing_final_public_reread_resume_v1"'),
        ('"only_added_field": "read_attestation.private_answers_read_before_save",\n                  "added_value": False',
         '"only_added_field": "read_attestation.final_public_questions_actually_reread",\n                  "added_value": True'),
        ('directory / "attestation_resume_v2"', 'directory / "public_reread_resume_v1"'),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, 'EXACT_REREAD_ADAPTER_PATCH_REQUIRED'
        source = source.replace(old, new)
    namespace = {'__file__': str(Path(__file__).resolve()), '__name__': 'retained_public_reread_resume_v1'}
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
    result = engine()['resume'](args.batch, args.root, 'public', args.executor,
                                args.judgment, args.engine, args.check_only)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
