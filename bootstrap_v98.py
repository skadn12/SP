"""Create a separate working delivery without modifying any sealed evidence."""
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
D = P / 'offline-deliveries'
OLD = D / 'h4_semantic_review_mixed_pilot_v97'
NEW = D / 'h4_event_identity_correction_pilot_v98'
assert not NEW.exists(), NEW
NEW.mkdir()
(NEW / 'evidence').mkdir()
for name in ('expansion_engine', 'qa_engine'):
    (NEW / name).mkdir()
    for source in (OLD / name).iterdir():
        if source.is_file():
            shutil.copy2(source, NEW / name / source.name)
for name in ('materialize_expansion.py', 'build_fresh_shard.py',
             'collect_expansion_results.py', 'audit_actual_qa_inputs.py',
             'actual_expansion_integrity.py', 'audit_actual_expansion.py'):
    source = OLD / 'evidence' / name
    text = source.read_text().replace('h4_semantic_review_mixed_pilot_v97', NEW.name)
    text = text.replace('v97', 'v98').replace('V97', 'V98')
    (NEW / 'evidence' / name).write_text(text)
protected = []
for directory in ('h4_controlled_six_root_expansion_v95', 'h4_v95_ready_mixed_qa_v96', OLD.name):
    for path in sorted((D / directory).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            protected.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
assert len(protected) == 2481, len(protected)
(NEW / 'PRESERVATION_BEFORE.json').write_text(json.dumps({'version': 'v98_preservation_v1', 'files': protected}, indent=2) + '\n')
print(json.dumps({'created': str(NEW), 'protected': len(protected)}))
