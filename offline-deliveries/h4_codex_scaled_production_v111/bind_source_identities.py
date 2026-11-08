"""Freeze actual original-input byte bindings before authoring.

This binds the training-side source identities before authoring; it is not a claim of external preregistration.
The already fixed selection and declared original identities are not changed.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def bind(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

audit = read(HERE / 'EXPOSURE_AUDIT.json')
selected = set(audit['selected_root_ids'])
holdout = {json.loads(line)['root_id'] for row in audit['holdout_manifests']
           for line in Path(row['path']).read_text().splitlines()}
selection = PROJECT / 'SP-820-publish/first_training_pipeline/data/selection_manifest_h4_unexpanded122.json'
index = {r['root_id']: r for r in read(selection)['records']}
inputs = [{'root_id': rid, **bind(index[rid]['input_path'])} for rid in sorted(selected | holdout)]
declared = {r['root_id']: r for r in audit['selected_roots']}
for row in inputs:
    if row['root_id'] in selected:
        source = read(row['path'])['source']
        original = declared[row['root_id']]
        assert source['source_id'] == original['original_source_id']
        assert source['source_record_sha256'] == original['original_source_record_sha256']
target = HERE / 'evidence/SOURCE_IDENTITY_BINDINGS.json'
with target.open('x', encoding='utf-8') as stream:
    json.dump({'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'BEFORE_AUTHORING',
        'selection_manifest': bind(selection), 'inputs': inputs,
        'selected_root_ids': sorted(selected), 'holdout_root_ids': sorted(holdout),
        'selection_or_declared_identity_changed': False}, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'status': 'BOUND', 'selected_roots': len(selected),
                  'holdout_roots': len(holdout), 'original_inputs': len(inputs)}))
