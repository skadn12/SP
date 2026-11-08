"""Save this reviewer's empty public ordering coverage before source validation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from ordering_review import validate_sources


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {'path': str(path.resolve()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


public_path = HOME / 'ordering' / 'PUBLIC_QUESTIONS.json'
public = read(public_path)
assert public == {'questions': []}, 'This helper records only an actual empty public set.'
sources = []
for slot in read(HOME / 'CONFIG.json')['ordering_slots']:
    path = HOME / 'ordering' / slot['slot_id'] / 'SOURCE.json'
    source = read(path)
    parent_path = HOME / 'parent_review' / ('root' + source['root_id'].split('-')[-1] + '_PUBLIC.json')
    prior = read(parent_path)
    assert path.read_bytes() == Path(prior['source_path']).read_bytes()
    assert binding(path)['sha256'] == prior['source_sha256'] == slot['source_sha256']
    sources.append({'root_id': source['root_id'], **binding(path),
                    'complete_turn_count': len(source['turns'])})

judgment = {
    'review_schema_version': 'source_judgment_v1',
    'reviewer': 'Codex independent source-only final reviewer v86',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'parent_all_true_provenance_known': False,
    'parent_gold_known_before_review': False,
    'fresh_blind_parent_review': False,
    'current_solver_scores_read': False,
    'solver_scores_used': False,
    'full_source_read': True,
    'new_variant_private_artifacts_read_before_save': False,
    'child_gold_read_before_save': False,
    'current_pool_read_before_save': False,
    'declarations_scope': 'The native declarations above apply to current ordering questions only.',
    'historical_multiselect_parent_disclosure': {
        'parent_all_true_provenance_known': True,
        'parent_gold_known_before_review': True,
        'fresh_blind_parent_review': False,
        'reason': 'The historical multiple-select parents are known to have all six options in Gold. No current ordering Gold, private pool, raw writer response, anchor plan, or other review was read.'
    },
    'public_questions': binding(public_path),
    'sources': sources,
    'questions': [],
    'review_scope_note': 'The actual mechanically admitted public ordering set is empty. Four complete sources were previously read in full and their current source bytes were verified identical. No mechanically rejected draft has received source-only semantic certification; this record certifies empty public coverage and source bindings only.'
}
path = HOME / 'ordering' / 'SOURCE_JUDGMENTS.json'
with path.open('x') as stream:
    json.dump(judgment, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
context = validate_sources(HOME)
print(json.dumps({'saved': str(path), 'source_count': len(sources),
                  'public_question_count': len(context['questions']),
                  'checks': context['checks'], 'source_validation': 'PASS'}))
