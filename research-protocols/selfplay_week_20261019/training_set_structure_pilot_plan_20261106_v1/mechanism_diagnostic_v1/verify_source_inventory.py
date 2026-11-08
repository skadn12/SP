"""Check source-only inventory bindings, complete ranges, and cited quotations."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
bindings = json.loads((HERE.parent / 'INPUT_BINDINGS.json').read_text())
expected = {r['root_id']: r for r in bindings['roots']}
reports = []
seen = set()
failures = []
for path in sorted((HERE / 'source_inventory').glob('source_inventory_*.json')):
    payload = json.loads(path.read_text())
    for root in payload['roots']:
        root_id = root['root_id']
        if root_id in seen or root_id not in expected:
            failures.append([root_id, 'duplicate or unexpected root'])
            continue
        seen.add(root_id)
        binding = expected[root_id]
        source = Path(binding['source']['path'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        data = json.loads(source.read_text())
        turns = data['turns']
        if digest != binding['source']['sha256'] or root['source_sha256'] != digest:
            failures.append([root_id, 'source hash mismatch'])
        if root['turn_count'] != len(turns) or len(turns) != binding['turn_count']:
            failures.append([root_id, 'turn count mismatch'])
        covered = []
        for chunk in root['read_ranges']:
            start, end = chunk['start_index_1based'], chunk['end_index_1based']
            if start < 1 or end > len(turns) or end < start or end - start + 1 > 30:
                failures.append([root_id, 'invalid chunk', chunk])
                continue
            subset = turns[start - 1:end]
            # Count rendered UTF-8 characters, independent of tokenization.
            rendered = json.dumps(subset, ensure_ascii=False, indent=2)
            if len(rendered) > 12000:
                failures.append([root_id, 'chunk exceeds 12000 characters', start, end])
            refs = chunk.get('turn_refs')
            expected_refs = [x['turn_ref'] for x in subset]
            endpoints = [expected_refs[0], expected_refs[-1]]
            if isinstance(refs, list) and refs != expected_refs and refs != endpoints:
                failures.append([root_id, 'chunk turn_refs mismatch', start, end])
            covered.extend(range(start, end + 1))
        if sorted(covered) != list(range(1, len(turns) + 1)):
            failures.append([root_id, 'ranges do not cover each turn exactly once'])
        turn_text = {t['turn_ref']: t['text'] for t in turns}
        seen_ids = set()
        for item in root['opportunities']:
            ident = item['inventory_id']
            if ident in seen_ids:
                failures.append([root_id, ident, 'duplicate inventory ID'])
            seen_ids.add(ident)
            for evidence in item['source_evidence']:
                ref, quote = evidence['turn_ref'], evidence['quote']
                if ref not in turn_text or quote not in turn_text[ref]:
                    failures.append([root_id, ident, 'quotation mismatch', ref, quote])
            if not item['source_evidence']:
                failures.append([root_id, ident, 'missing evidence'])
        reports.append({'root_id': root_id, 'turn_count': len(turns),
                        'opportunity_count': len(root['opportunities']),
                        'inventory_path': str(path),
                        'inventory_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'source_sha256': digest})
if seen != set(expected):
    failures.append(['root coverage mismatch', sorted(set(expected) - seen), sorted(seen - set(expected))])
result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'status': 'PASS' if not failures else 'FAIL', 'roots': reports,
          'failures': failures,
          'limitations': ['Mechanical checks cannot establish semantic validity, review independence, or completeness of opportunity discovery.']}
print(json.dumps(result, ensure_ascii=False, indent=2))
if not failures:
    with (HERE / 'SOURCE_INVENTORY_FREEZE.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
else:
    raise SystemExit(1)
