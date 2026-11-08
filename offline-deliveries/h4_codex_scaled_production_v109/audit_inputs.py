"""Check immutable inputs, first-draft receipts and mechanical source bindings."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rows = read(HERE / 'SOURCE_RESERVATIONS.json')['allocation']
    checked = []
    for row in rows:
        folder = HERE / 'roots' / ('root' + row['root_id'].rsplit('-', 1)[1])
        draft, source, receipt = [read(folder / n) for n in
            ('EXPANSION_DRAFT.json', 'CUMULATIVE_SOURCE.json', 'AUTHOR_RECEIPT.json')]
        assert sha(Path(row['baseline_path'])) == row['baseline_sha256']
        assert sha(Path(row['source_item_path'])) == row['source_item_sha256']
        baseline = read(Path(row['baseline_path']))
        assert source['turns'] == baseline['turns'] + draft['turns']
        assert draft['root_id'] == row['root_id'] and draft['source_id'] == row['source_id']
        assert 12 <= len(draft['turns']) <= 16
        assert receipt['attempt'] == 1 and receipt['full_baseline_read'] is True
        start = datetime.fromisoformat(receipt['started_at_utc'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(receipt['finished_at_utc'].replace('Z', '+00:00'))
        assert end >= start
        turns = {t['turn_ref']: t for t in baseline['turns']}
        for note in draft['state_notes']:
            assert note['quote'] and note['quote'] in turns[note['turn_ref']]['text']
        checked.append({'root_id': row['root_id'], 'new_turns': len(draft['turns']),
            'prefix_exact': True, 'draft_sha256': sha(folder / 'EXPANSION_DRAFT.json'),
            'source_sha256': sha(folder / 'CUMULATIVE_SOURCE.json'),
            'receipt_sha256': sha(folder / 'AUTHOR_RECEIPT.json')})
    for p in (HERE / 'evidence/precall_code').iterdir():
        assert sha(p) == sha(HERE.parent.parent / 'scripts' / p.name)
    out = {'status': 'PASS_MECHANICAL_NOT_SEMANTIC', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'roots': checked, 'new_turns': sum(r['new_turns'] for r in checked),
        'source_ready_certified': False, 'external_calls': 0}
    with (HERE / 'evidence/INPUT_AND_STRUCTURE_AUDIT.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': out['status'], 'roots': len(checked), 'new_turns': out['new_turns']}))

if __name__ == '__main__':
    main()
