#!/usr/bin/env python3
"""Independent export: author/export process parses private gold; solver does not.

This exporter is outside the solver runtime. It creates an adapter-whitelisted
public payload plus an exact private copy of all 14 reviewed question objects.
It is not an admission or selection stage and does not alter source questions
or QUESTION_FREEZE.json. Run once; no output is overwritten.
"""
from __future__ import annotations
import ast
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INPUTS = BASE / 'solver_v5' / 'inputs'
QFREEZE = BASE / 'QUESTION_FREEZE.json'
ADAPTER = BASE.parent / 'day2' / 'solver' / 'adapter.py'
PUBLIC = INPUTS / 'PUBLIC_QUESTIONS.json'
PRIVATE = INPUTS / 'PRIVATE_QUESTIONS.json'
EXPORT_FREEZE = INPUTS / 'PUBLIC_EXPORT_FREEZE.json'


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def json_read(path):
    return json.loads(Path(path).read_text())


def main():
    require(QFREEZE.is_file(), 'QUESTION_FREEZE_REQUIRED')
    for p in (PUBLIC, PRIVATE, EXPORT_FREEZE):
        require(not p.exists(), 'REFUSE_OVERWRITE:' + str(p))
    frozen = json_read(QFREEZE)
    require(frozen['status'] == 'FROZEN_BEFORE_SOLVER', 'QUESTION_FREEZE_STATE')
    require(frozen['gates']['content_review_status'] == 'PASS', 'CONTENT_REVIEW_REQUIRED')
    require(frozen['gates']['solver_started'] is False, 'SOLVER_ALREADY_STARTED')

    # Extract the literal whitelist without importing solver modules.
    module = ast.parse(ADAPTER.read_text())
    values = [ast.literal_eval(node.value) for node in module.body
              if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'PUBLIC_KEYS' for t in node.targets)]
    require(len(values) == 1 and isinstance(values[0], set), 'ADAPTER_PUBLIC_KEYS_NOT_LITERAL')
    whitelist = values[0]
    require(whitelist == {'candidate_id', 'root_id', 'question_type', 'question', 'options', 'items', 'source_path', 'source_sha256'}, 'ADAPTER_WHITELIST_CHANGED')

    public_questions, private_questions, parsed = [], [], []
    for slot in frozen['questions']:
        path = Path(slot['question_file']['path'])
        require(digest(path) == slot['question_file']['sha256'], 'FROZEN_QUESTION_HASH_MISMATCH')
        require(path.stat().st_size == slot['question_file']['bytes'], 'FROZEN_QUESTION_BYTES_MISMATCH')
        private = json_read(path)  # Intentionally private, Gold-aware export process.
        require(private['candidate_id'] == slot['candidate_id'], 'CANDIDATE_MISMATCH')
        require(private['root_id'] == slot['root_id'], 'ROOT_MISMATCH')
        require(private['question_type'] == slot['question_type'], 'TYPE_MISMATCH')
        require('gold' in private and isinstance(private['gold'], list), 'PRIVATE_GOLD_REQUIRED')
        private_questions.append(private)
        parsed.append(bind(path))
        field = 'options' if private['question_type'] == 'multi_select' else 'items'
        public = {key: private[key] for key in ('candidate_id', 'root_id', 'question_type', 'question', field)}
        public['source_path'] = private['source_binding']['path']
        public['source_sha256'] = private['source_binding']['sha256']
        require(set(public) <= whitelist, 'PUBLIC_WHITELIST_VIOLATION')
        require(set(public) == {'candidate_id', 'root_id', 'question_type', 'question', field, 'source_path', 'source_sha256'}, 'PUBLIC_KEYS_MISSING')
        require(('items' if field == 'options' else 'options') not in public, 'PUBLIC_NULL_OR_CONFLICT_FIELD')
        public_questions.append(public)

    require(len(public_questions) == len(private_questions) == 14, 'QUESTION_COUNT_MISMATCH')
    require(len({q['candidate_id'] for q in public_questions}) == 14, 'DUPLICATE_ID')
    INPUTS.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(json.dumps({'questions': public_questions}, ensure_ascii=False, indent=2) + '\n')
    PRIVATE.write_text(json.dumps({'questions': private_questions}, ensure_ascii=False, indent=2) + '\n')
    require(json_read(PRIVATE)['questions'] == private_questions, 'PRIVATE_EXPORT_NOT_EXACT_OBJECT_COPY')
    check = json_read(PUBLIC)
    require(set(check) == {'questions'}, 'PUBLIC_WRAPPER_NOT_WHITELISTED')
    for q in check['questions']:
        require(set(q) <= whitelist and not ({'gold', 'evidence', 'gold_count', 'slot', 'display_order', 'display_seed'} & set(q)), 'PUBLIC_PRIVATE_FIELD_LEAK')

    export = {
        'schema_version': 'live_extension_v5_public_export_freeze_v1',
        'status': 'EXPORTED_AND_FROZEN_BEFORE_SOLVER',
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'question_count': 14,
        'by_question_type': dict(collections.Counter(q['question_type'] for q in public_questions)),
        'public_keys': sorted(whitelist),
        'public_wrapper_keys': ['questions'],
        'public_nullable_unused_task_fields_omitted': True,
        'private_export_scope': 'Full exact JSON objects from the 14 question files; original private fields retained.',
        'exposure_disclosure': {
            'author_process_accessed_gold': True,
            'review_process_accessed_gold': True,
            'freeze_process_accessed_gold': True,
            'export_process_accessed_gold': True,
            'export_process_accessed_evidence': True,
            'exporter_is_independent_of_solver_runtime': True,
            'solver_prepare_gold_parse_authorized': False,
            'solver_live_gold_parse_authorized': False,
            'scope_note': 'The author/reviewer/freeze/export processes parsed private Gold. This disclosure makes no blindness claim for them. Only prepare/live use the public allowlist payload; scoring may parse private Gold after a committed raw response freeze.'
        },
        'admission_policy': {
            'selection_performed_by_exporter': False,
            'content_review_required': True,
            'solver_score_filtering': False,
            'source_questions_modified': False,
            'question_freeze_modified': False,
        },
        'bindings': {
            'question_freeze': bind(QFREEZE),
            'public_questions': bind(PUBLIC),
            'private_questions': bind(PRIVATE),
            'adapter_public_key_contract': bind(ADAPTER),
            'exporter_code': bind(__file__),
        },
        'private_question_file_bindings': parsed,
        'checks': [
            '14 question files match QUESTION_FREEZE hashes and identity',
            'PUBLIC_QUESTIONS uses only day2 adapter PUBLIC_KEYS',
            'Unused options/items fields omitted rather than null',
            'No gold/evidence/slot/gold_count/display order field exported publicly',
            'PRIVATE_QUESTIONS preserves all 14 original private JSON objects',
            'No source question, admission decision, or QUESTION_FREEZE modified',
        ]
    }
    EXPORT_FREEZE.write_text(json.dumps(export, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'files': [bind(PUBLIC), bind(PRIVATE), bind(EXPORT_FREEZE)]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
