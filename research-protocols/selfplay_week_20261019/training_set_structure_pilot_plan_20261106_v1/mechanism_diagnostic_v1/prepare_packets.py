"""Create analysis-only copies; never mutate any experiment artifact."""
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
OLD = BASE / 'execution_20261106_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    with Path(path).open('x') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def parse(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def main():
    bindings = json.loads((BASE / 'INPUT_BINDINGS.json').read_text())
    rows = list(csv.DictReader((OLD / 'results/SLOTS.csv').open()))
    assert len(rows) == 45
    protected = []
    for folder in [OLD, BASE / 'execution_solver_infrastructure_amendment_v1']:
        protected += [p for p in folder.rglob('*') if p.is_file()]
    protected += [BASE / 'INPUT_BINDINGS.json', BASE / 'PLAN_ZH.md', BASE / 'PLAN_CONFIG.json']
    for item in bindings['roots']:
        for key in ['source', 'canonical_source', 'neutral_source_info']:
            p = Path(item[key]['path'])
            assert sha(p) == item[key]['sha256'], (key, p)
            protected.append(p)
    protected = sorted(set(protected))
    write_new(OUT / 'PROTECTED_INPUTS_BEFORE.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'note': 'Protected prior experiment files; diagnostic outputs excluded.',
        'files': [{'path': str(p), 'sha256': sha(p)} for p in protected],
    })
    join = []
    for root in bindings['roots']:
        root_id = root['root_id']
        packet_items = []
        for row in rows:
            if row['root_id'] != root_id:
                continue
            blind_id = 'SL_' + hashlib.sha256(
                ('mechanism_diagnostic_v1_20261106:' + row['opportunity_id'] + ':' + row['slot_id']).encode()
            ).hexdigest()[:12]
            fields = ['natural_information_need', 'query_scope', 'answer_program',
                      'semantic_answer', 'required_relations', 'alternative_sufficient_paths',
                      'unresolved_relations']
            content = {key: parse(row[key]) for key in fields}
            raw = parse(row['raw_frozen_slot'])
            refs = {k: v for k, v in raw.items()
                    if k in ['evidence_refs', 'source_evidence', 'evidence',
                             'source_fact_ids', 'source_turn_refs', 'evidence_turn_refs']}
            content['author_evidence_refs_unverified'] = refs
            packet_items.append({'blind_slot_id': blind_id, 'content': content})
            join.append({'blind_slot_id': blind_id, 'opportunity_id': row['opportunity_id'],
                         'slot_id': row['slot_id'], 'root_id': root_id,
                         'group': row['group'], 'selected': parse(row['selected']),
                         'frozen_slot_source': parse(row['frozen_slot_source'])})
        random.Random('mechanism_diagnostic_v1:' + root_id).shuffle(packet_items)
        history = OLD / 'manifests/selector_baseline' / (root_id + '.json')
        packet = {
            'schema_version': 'blind_slot_diagnostic_packet_v1',
            'root_id': root_id,
            'source': root['source'],
            'turn_count': root['turn_count'],
            'fixed_historical_reference': {'path': str(history), 'sha256': sha(history)},
            'slots': packet_items,
            'limitations': ['Target semantics are author claims until independently checked.',
                            'History UNKNOWN is not proof of absent overlap.',
                            'Selection/group/feedback/final questions withheld.'],
        }
        write_new(OUT / 'blind_slot_review' / (root_id + '.packet.json'), packet)
    write_new(OUT / 'PRIVATE_JOIN_MAP.json', {'rows': join})
    write_new(OUT / 'PREPARATION_MANIFEST.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'slot_count': len(join), 'selected_count_private': sum(x['selected'] for x in join),
        'protected_file_count': len(protected),
        'packet_files': [{'path': str(p), 'sha256': sha(p)} for p in sorted((OUT / 'blind_slot_review').glob('*.packet.json'))],
        'protocol_sha256': sha(OUT / 'AUDIT_PROTOCOL.md'),
        'only_diagnostic_outputs_created': True,
    })
    print(json.dumps({'slots': len(join), 'roots': len(bindings['roots']), 'protected_files': len(protected)}))


if __name__ == '__main__':
    main()
