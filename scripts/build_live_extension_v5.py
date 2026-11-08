import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp/gxy-selfplay')
BATCH = ROOT / 'research-protocols/selfplay_week_20261019/postweek_live_extension_v5'
POOL = ROOT / 'offline-deliveries/h4_external_event_source_refresh_v2/multiwoz_train_sample.json'
BASE = ROOT / 'offline-deliveries/h4_codex_train_pool_v125/sources'
OLD_FEEDBACK = ROOT / 'research-protocols/selfplay_week_20261019/postweek_live_extension_v4/feedback/SHORTFALL_LEDGER_ADAPTER.json'

ALLOC = {
    'cc-train-episode-19708': ['PMUL2572.json', 'PMUL2319.json', 'MUL1306.json'],
    'cc-train-episode-20432': ['MUL0097.json', 'MUL0873.json', 'MUL1464.json'],
}

def bind(path):
    raw = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')

def now():
    return datetime.now(timezone.utc).isoformat()

pool = {row['dialogue_id']: row for row in json.loads(POOL.read_text(encoding='utf-8'))}
events = []
for root, ids in ALLOC.items():
    for idx, dialogue_id in enumerate(ids, 1):
        event_path = BATCH / 'inputs' / f'{root}_B_S{idx:02d}_EVENT.json'
        dump(event_path, pool[dialogue_id])
        events.append({
            'root_id': root,
            'scene_ref': f'LEX1:B:S{idx:02d}',
            'source_id': f'multiwoz22:train:{dialogue_id}',
            'split': 'train',
            'pool_record_id': dialogue_id,
            'binding': bind(event_path),
            'pool_binding': bind(POOL),
            'reserved_before_authoring': True,
            'status': 'PERMANENTLY_RESERVED_BEFORE_AUTHORING',
            'release_on_failure': False,
        })

dump(BATCH / 'inputs/SOURCE_RESERVATIONS.json', {
    'schema_version': 'live_extension_source_reservations_v2',
    'created_at_utc': now(),
    'status': 'PERMANENT',
    'policy': 'No reuse and no release on failure or nonexecution; all six B events bound before authoring.',
    'pool_binding': bind(POOL),
    'reservations': events,
})

roots = []
for root in ALLOC:
    roots.append({
        'root_id': root,
        'source_binding': bind(BASE / f'{root}.json'),
        'task_operations': [
            {'operation_id': 'explicit_event_time', 'question_type': 'ordering'},
            {'operation_id': 'source_state_and_plan_completion', 'question_type': 'multi_select'},
            {'operation_id': 'actor_entity_binding', 'question_type': 'multi_select'},
        ],
        'material_targets': [{'slot_id': f'PROD2-{root[-5:]}-ORDER2', 'material_goal': 'Add independently source-grounded event relations without inventing unresolved values.'}],
        'native_events': [e for e in events if e['root_id'] == root],
    })

dump(BATCH / 'CONFIG.json', {
    'schema_version': 'reason_material_feedback_pipeline_v2_config',
    'batch_id': 'postweek_live_extension_v5',
    'namespace': 'LEX1',
    'reason_packet': None,
    'shortfall_ledger_binding': bind(OLD_FEEDBACK),
    'roots': roots,
    'generation_started': False,
})
dump(BATCH / 'PROTOCOL_FREEZE.json', {
    'schema_version': 'live_extension_protocol_freeze_v2',
    'status': 'FROZEN_BEFORE_AUTHORING',
    'created_at_utc': now(),
    'namespace': 'LEX1',
    'roots': list(ALLOC),
    'scene_sequence': ['LEX1:A:S01', 'LEX1:A:S02', 'LEX1:A:S03', 'LEX1:B:S01', 'LEX1:B:S02', 'LEX1:B:S03'],
    'source_budget': {'author_attempts_per_scene': 1, 'semantic_repairs': 0, 'source_replacements': 0, 'dependent_scene_after_failed_review': False},
    'scene_turns': {'min': 12, 'max': 16},
    'full_source_no_truncation': True,
    'question_plan': {'per_root': {'MC2': 1, 'MC3': 1, 'MC4': 1, 'MC5': 1, 'MC6': 1, 'ORDER1': 1, 'ORDER2': 1}, 'solver_score_filtering': False, 'exact_whole_question_duplicate_only': True},
    'shortfall_ledger_binding': bind(OLD_FEEDBACK),
    'reservations_binding': bind(BATCH / 'inputs/SOURCE_RESERVATIONS.json'),
    'pool_binding': bind(POOL),
    'source_bindings': {root: bind(BASE / f'{root}.json') for root in ALLOC},
    'generation_started': False,
})
print(json.dumps({'batch': str(BATCH), 'pool': bind(POOL), 'reservations': bind(BATCH / 'inputs/SOURCE_RESERVATIONS.json'), 'events': len(events)}, ensure_ascii=False))
