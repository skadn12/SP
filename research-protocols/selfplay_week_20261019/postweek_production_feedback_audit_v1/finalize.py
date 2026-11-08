"""Seal the complete semantic review without editing the original production."""
from datetime import datetime, timezone
import json
from pathlib import Path

from prepare import HERE, BATCH, binding, read, save


def main():
    assert not (HERE / 'FINAL_MANIFEST.json').exists(), 'ALREADY_SEALED'
    summary = read(HERE / 'SUMMARY.json')
    audit = read(HERE / 'audit/INDEPENDENT_COMPLETION_AUDIT.json')
    assert audit['status'].startswith('PASS')
    assert audit['bindings']
    for b in summary['bindings'] + audit['bindings']:
        assert binding(b['path']) == b, b['path']
    original = read(BATCH / 'FINAL_MANIFEST.json')
    for b in original['artifact_bindings'] + original['external_reference_bindings']:
        assert binding(b['path']) == b, b['path']
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                   and p.name != 'FINAL_MANIFEST.json')
    manifest = {'schema_version': 'production_full_semantic_audit_final_v1',
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'status': summary['status'], 'questions': summary['questions'],
                'responses': summary['responses'], 'content_decisions': summary['content_decisions'],
                'all_review_dimensions_pass': summary['all_review_dimensions_pass'],
                'formal_weakness_admission': False, 'feedback_causal_gain_proven': False,
                'new_solver_calls': 0, 'new_training_runs': 0,
                'original_batch_unchanged': binding(BATCH / 'FINAL_MANIFEST.json'),
                'artifact_bindings': [binding(p) for p in files]}
    save(HERE / 'FINAL_MANIFEST.json', manifest)
    pointer = {'schema_version': 'production_feedback_audit_latest_v1',
               'created_at_utc': manifest['created_at_utc'], 'status': manifest['status'],
               'manifest': binding(HERE / 'FINAL_MANIFEST.json'),
               'report': binding(HERE / 'REPORT_ZH.md'),
               'examples': binding(HERE / 'EXAMPLES_ZH.md'),
               'cards': binding(HERE / 'EXPLORATORY_WEAKNESS_CARDS.json'),
               'content_concerns': binding(HERE / 'TRAINING_CONTENT_CONCERNS.json'),
               'questions': summary['questions'], 'responses': summary['responses'],
               'formal_weakness_admission': False, 'new_solver_calls': 0}
    save(HERE.parent / 'PRODUCTION_FEEDBACK_AUDIT_LATEST.json', pointer)
    for b in manifest['artifact_bindings']:
        assert binding(b['path']) == b
    print(json.dumps({'status': manifest['status'], 'bound_artifacts': len(files)}))


if __name__ == '__main__':
    main()
