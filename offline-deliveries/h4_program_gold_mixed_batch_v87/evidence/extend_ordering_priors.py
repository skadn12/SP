"""Preserve all v86 dedup priors and add public projections of every v86 draft."""
from pathlib import Path
import json
import sys

BATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BATCH))
from common import read, write, bind, sha, now, need

old = BATCH.parent / 'h4_reason_before_gold_mixed_batch_v86'
need(not (BATCH / 'BATCH_PREPARATION.STARTED.json').exists(), 'already_prepared')
rows = []
inputs = []
for run in sorted((old / 'ordering').glob('root*_ordering')):
    response = next((run / 'requests').glob('*/RESPONSE.json'))
    plan_path = run / 'ANCHOR_PLAN.json'
    plans = {q['candidate_id']: q for q in read(plan_path)['questions']}
    source = run / 'SOURCE.json'
    inputs.extend([bind(response), bind(plan_path), bind(source)])
    for q in json.loads(read(response)['text'])['questions']:
        if q is None:
            continue
        plan = plans[q['candidate_id']]
        rows.append({'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
            'question_type': 'ordering', 'chronology_kind': 'conversation_order',
            'question': plan['question'], 'items': q['items'],
            'source_path': str(source), 'source_sha256': sha(source)})
projection = BATCH / 'evidence/V86_ALL_ORDERING_DRAFT_PUBLIC_TEXT_WITH_METADATA.json'
write(projection, {'created_at_utc': now(), 'questions': rows, 'inputs': inputs,
    'projection': 'Verbatim public items from every non-null writer draft; fixed public stem from bound historical plan.',
    'historical_private_fields_encountered': True, 'current_private_artifacts_read': False,
    'semantic_certification_claimed': False, 'purpose': 'Historical whole-six-text exclusion, including rejected drafts.'})
config = read(BATCH / 'CONFIG.json')
before = [bind(BATCH / n) for n in ('CONFIG.json', 'CONFIG_TEMPLATE.json', 'source_subset.py')]
for slot in config['ordering_slots']:
    root = slot['root_id'].rsplit('-', 1)[1]
    slot['prior_pool_paths'].extend([str(old / 'evidence' / f'ORDERING_PRIORS_UNIQUE_{root}.json'), str(projection)])
for name in ('CONFIG.json', 'CONFIG_TEMPLATE.json'):
    (BATCH / name).write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
pin = BATCH / 'source_subset.py'
old_sha = next(r['sha256'] for r in before if r['path'].endswith('/CONFIG_TEMPLATE.json'))
pin.write_text(pin.read_text().replace(old_sha, sha(BATCH / 'CONFIG_TEMPLATE.json')))
write(BATCH / 'evidence/ORDERING_PRIOR_EXTENSION.json', {'created_at_utc': now(),
    'before': before, 'after': [bind(BATCH / n) for n in ('CONFIG.json', 'CONFIG_TEMPLATE.json', 'source_subset.py')],
    'historical_drafts_added': len(rows), 'public_projection': bind(projection),
    'next': 'Exact normalized whole-set compaction, retaining every distinct six-text set.'})
print({'historical_drafts_added': len(rows)})
