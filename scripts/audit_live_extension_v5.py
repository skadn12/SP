import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp/gxy-selfplay')
BATCH = ROOT / 'research-protocols/selfplay_week_20261019/postweek_live_extension_v5'
POOL = ROOT / 'offline-deliveries/h4_external_event_source_refresh_v2/multiwoz_train_sample.json'
IDS = ['PMUL2572.json', 'PMUL2319.json', 'MUL1306.json', 'MUL0097.json', 'MUL0873.json', 'MUL1464.json']
META_NAMES = {
    'SOURCE_RESERVATIONS.json', 'SOURCE_POOL_FREEZE.json', 'SOURCE_CONSUMPTION_FREEZE.json',
    'SOURCE_CONSUMPTION_SCAN.json', 'UNIQUE_EVENT_ALLOCATION.json', 'RESERVATION.json',
    'SELECTION.json', 'selection.json', 'SOURCE_SELECTION.json', 'SOURCE_BINDINGS.json',
    'FINAL_SOURCE_BINDING.json', 'EVALUATION_SOURCE_BINDINGS.json', 'RUN_CONFIG.json',
}

def bind(path):
    raw = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def excluded(path):
    text = str(path)
    return any(f'postweek_live_extension_v{i}' in text for i in range(1, 6))

pool = {row['dialogue_id']: row for row in json.loads(POOL.read_text(encoding='utf-8'))}
hits = {item: [] for item in IDS}
inspected = 0
for path in ROOT.rglob('*.json'):
    if path.name not in META_NAMES or excluded(path):
        continue
    try:
        raw = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        continue
    inspected += 1
    for item in IDS:
        if item in raw or f'multiwoz22:train:{item}' in raw:
            hits[item].append(bind(path))

checks = []
for index, item in enumerate(IDS):
    root = 'cc-train-episode-19708' if index < 3 else 'cc-train-episode-20432'
    scene = index + 1 if index < 3 else index - 2
    event_path = BATCH / 'inputs' / f'{root}_B_S{scene:02d}_EVENT.json'
    event = json.loads(event_path.read_text(encoding='utf-8'))
    original = pool[item]
    checks.append({
        'dialogue_id': item,
        'event_binding': bind(event_path),
        'record_object_exact_match': event == original,
        'pool_turn_count': len(original['turns']),
        'event_turn_count': len(event['turns']),
        'services_exact_match': event.get('services') == original.get('services'),
    })

reservation = json.loads((BATCH / 'inputs/SOURCE_RESERVATIONS.json').read_text(encoding='utf-8'))
reserved = [r['pool_record_id'] for r in reservation['reservations']]
prior_unused = all(not hits[item] for item in IDS)
exact = all(c['record_object_exact_match'] and c['services_exact_match'] for c in checks)
status = 'PASS_PRIOR_UNUSED_POOL_EXACT' if prior_unused and exact and reserved == IDS else 'FAIL_RESERVATION_AUDIT'
result = {
    'schema_version': 'live_extension_v5_reservation_audit_v1',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'status': status,
    'pool_binding': bind(POOL),
    'reservation_binding': bind(BATCH / 'inputs/SOURCE_RESERVATIONS.json'),
    'historical_scan': {
        'metadata_files_inspected': inspected,
        'prior_hit_files_by_id': hits,
        'prior_unused_per_id': {item: not hits[item] for item in IDS},
        'pilot_setup_exclusion': [f'postweek_live_extension_v{i}' for i in range(1, 6)],
    },
    'event_checks': checks,
    'reservation_order_exact': reserved == IDS,
    'conclusion': {
        'all_six_prior_unused': prior_unused,
        'all_six_exact_pool_records': exact,
        'current_reservation_is_not_prior_consumption': True,
        'release_on_failure': False,
    },
}
out = BATCH / 'audit/RESERVATION_AUDIT.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': status, 'audit': str(out), 'metadata_files_inspected': inspected}, ensure_ascii=False))
