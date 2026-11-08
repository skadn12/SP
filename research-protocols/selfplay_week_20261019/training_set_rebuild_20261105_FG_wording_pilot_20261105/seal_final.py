#!/usr/bin/env python3
"""Validate final research-only deliverables and create the local SHA-256 seal."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


decisions = json.loads((OUT / 'FINAL_WORDING_DECISIONS_REFINED.json').read_text(encoding='utf-8'))
selected = json.loads((OUT / 'FINAL_RESEARCH_CANDIDATES_REFINED.json').read_text(encoding='utf-8'))
assert decisions['retained_v1_count'] == 5 and decisions['kept_original_count'] == 7
assert len(selected['items']) == 12
assert sum(x['selected_version'] == 'V1_REWRITE' for x in selected['items']) == 5
assert sum(x['selected_version'] == 'ORIGINAL' for x in selected['items']) == 7
assert decisions['training_admission'] is False
assert decisions['candidate_quality_status'] == 'UNREVIEWED'
audit = json.loads((OUT / 'FINAL_INTEGRITY_AUDIT.json').read_text(encoding='utf-8'))
assert audit['sealed_file_count'] == 1095 and audit['sealed_hash_mismatches'] == 0
assert audit['sampled_source_or_draft_hash_mismatches'] == 0
assert audit['training'] == 'none' and audit['solver_calls_in_this_pilot'] == 0
assert all(v['closure_matches_gold'] and v['query_domain_unique_and_complete'] and
           v['pair_audit_entailments_match_closure']
           for v in audit['structure_closure_and_query_audit_flags'].values())
score_audit = json.loads((OUT / 'READABILITY_SCORE_AUDIT.json').read_text(encoding='utf-8'))
assert score_audit['row_count'] == 24 and score_audit['all_rows_match']

files = []
for path in sorted(OUT.iterdir()):
    if path.name == 'FINAL_SEAL.json' or not path.is_file():
        continue
    files.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
seal = {
    'status': 'FINAL_RESEARCH_PILOT_SEALED_NOT_TRAINING_DATA',
    'sealed_at_utc': datetime.now(timezone.utc).isoformat(),
    'artifact_count_excluding_this_seal': len(files),
    'checks_before_seal': {
        'retained_v1_rewrites': 5, 'original_wordings_retained': 7,
        'retained_v2_rewrites': 0, 'candidate_quality_status': 'UNREVIEWED',
        'parent_exploration_files': 1095, 'parent_hash_mismatches': 0,
        'solver_calls': 0, 'training_or_export': False,
        'structure_closure_and_query_audits_pass': True,
        'readability_rows_verified_against_raw': 24
    },
    'files': files,
    'note': 'Manifest hashes every other regular file in this directory. Any later edit requires a new seal artifact.'
}
with (OUT / 'FINAL_SEAL.json').open('x', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'sealed_files': len(files), 'manifest_sha256': sha(OUT / 'FINAL_SEAL.json')}, ensure_ascii=False))
