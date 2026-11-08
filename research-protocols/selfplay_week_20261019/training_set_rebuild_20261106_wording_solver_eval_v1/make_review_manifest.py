#!/usr/bin/env python3
"""Bind the completed independent review records without changing their findings."""
import json
from pathlib import Path
import adapter

here = Path(__file__).resolve().parent
blind = here / 'BLIND_REVIEW_ZH.md'
dedup = here / 'DEDUP_REVIEW_ZH.md'
final = adapter.read(adapter.PILOT / 'FINAL_RESEARCH_CANDIDATES_REFINED.json')
chosen = [row for row in final['items'] if row['selected_version'] == 'V1_REWRITE']
assert len(chosen) == 5 and {row['candidate_id'] for row in chosen} == set(adapter.IDS)

def record(status, artifact, reviewer, blind_review=False):
    result = {'status': status, 'reviewer': reviewer, 'artifact_path': str(artifact),
              'artifact_sha256': adapter.bind(artifact)['sha256']}
    if blind_review:
        result['blind'] = True
    return result

rows = []
for row in chosen:
    candidate = row['candidate']
    qid = row['candidate_id']
    rows.append({'candidate_id': qid, 'candidate_sha256': adapter.canonical_hash(candidate),
        'source_sha256': candidate['source_sha256'],
        'quality': record('HOLD' if qid == 'FGX1_F_45849_01' else 'PASS', blind,
                          'two independent same-model blind contexts and timing adjudication', True),
        'source': record('PASS', blind, 'two independent same-model blind contexts', True),
        'dedup': record('PASS', dedup, 'pre-solver semantic comparison; PASS means comparison performed')})
adapter.write_new(here / 'EXTERNAL_REVIEWS.json', {'items': rows,
    'scope': 'Research evaluation only. Dedup PASS denotes comparison completed, not novelty or training eligibility.',
    'duplicates': ['FGX1_F_162896_01', 'FGX1_F_191749_01', 'FGX1_G_75241_01'],
    'high_overlap': ['FGX1_F_225575_01', 'FGX1_F_45849_01'],
    'review_limits': 'Same-model contexts; semantic comparison coverage incomplete; see underlying records.'})
print(json.dumps({'created': str(here / 'EXTERNAL_REVIEWS.json'), 'items': len(rows),
                  'quality_hold': ['FGX1_F_45849_01']}, ensure_ascii=False))
