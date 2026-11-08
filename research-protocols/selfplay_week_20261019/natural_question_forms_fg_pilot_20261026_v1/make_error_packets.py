#!/usr/bin/env python3
"""Expose only frozen final public responses for prospective target auditing."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def binding(path):
    data = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def main():
    candidates = {c['candidate_id']: c for c in read(HERE / 'CANDIDATES_PRIVATE.json')}
    public = {r['request_id']: r for r in read(HERE / 'execution/PUBLIC_REQUESTS.json')['requests']}
    private = {r['request_id']: r for r in read(HERE / 'execution/PRIVATE_SCORING.json')['requests']}
    records = [r for r in read(HERE / 'execution/SCORES.json')['records'] if r['arm'] == 'F' and r['condition'] == 'FULL']
    for group, tails in [(1, {'218519', '222937'}), (2, {'225575', '229686'})]:
        rows = []
        for record in records:
            if record['root_id'].split('-')[-1] not in tails:
                continue
            rid = record['request_id']
            c = candidates[record['candidate_id']]
            rows.append({'candidate': c, 'score_and_public_final_json': record,
                         'displayed_options': public[rid]['display'],
                         'presentation': private[rid]['presentation']})
        target = HERE / f'error_review/group{group}'
        target.mkdir(parents=True, exist_ok=False)
        packet = {'records': rows, 'input_bindings': [binding(HERE / p) for p in ('CANDIDATES_PRIVATE.json', 'execution/SCORES.json', 'execution/PUBLIC_REQUESTS.json', 'execution/PRIVATE_SCORING.json', 'ERROR_MATCHING_RULES_ZH.md')],
                  'policy': 'Audit every assigned F FULL response, including exact answers and MC6. Inspect final item_evidence and answer_reason only; do not request or infer private deliberation. Preserve predicted targets; distinguish explanation vs selection-only matches. No question edits.'}
        with (target / 'PACKET.json').open('x') as stream:
            json.dump(packet, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({'F_FULL_responses': len(records)}))

if __name__ == '__main__':
    main()
