"""Project historical feedback to declared training roots before any author reads it."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

source = HERE / 'inputs/HISTORICAL_FEEDBACK.json'
manifest = HERE.parent.parent / 'SP-820-publish/first_training_pipeline/data/matched_training/m_clean_policy_training_manifest.jsonl'
roots = {json.loads(l)['root_id'] for l in manifest.read_text().splitlines()}
feedback = json.loads(source.read_text())
kept = [r for r in feedback['records'] if r['root_id'] in roots]
out = HERE / 'inputs/TRAINING_FEEDBACK.json'
with out.open('x', encoding='utf-8') as f:
    json.dump({'version': 'v110_training_only_historical_feedback',
        'upstream_path': str(source), 'upstream_sha256': sha(source),
        'clean_manifest_sha256': sha(manifest), 'records': kept,
        'root_order': [r['root_id'] for r in kept],
        'excluded_roots': [r['root_id'] for r in feedback['records'] if r['root_id'] not in roots],
        'usage': 'Only this training-only projection is author-readable. Read matching-root feedback if present; otherwise disclose absence. Other-training-root aggregates may guide general emphasis, not supply new answers.',
        'current_validation_feedback_used': False,
        'selection_contract': feedback['selection_contract']}, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': 'TRAINING_FEEDBACK_PROJECTED', 'roots': [r['root_id'] for r in kept],
    'excluded_roots': [r['root_id'] for r in feedback['records'] if r['root_id'] not in roots]}))
