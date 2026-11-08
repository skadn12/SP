#!/usr/bin/env python3
"""Controller-only inventory of frozen prior first drafts, without evaluation outputs."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'execution_20261106_v1'
OUTPUT = HERE / 'PRIOR_ROUND_30_DRAFT_INVENTORY_CONTROLLER.json'
INPUTS = {
    'completion_freeze': (RUN / 'manifests/CONSTRUCTION_COMPLETION_FREEZE.json', 'cbcb17b401e065ffd0b875c26c7e907d077898317cf09899b8312b35f903b78e'),
    'schedule': (RUN / 'manifests/SCHEDULE_RESTRICTED.json', '1ddeb87bb0bd18e33af7d63840072efe53d14c98cb701563aa99523ac61621f3'),
    'invalidation': (RUN / 'quarantine_context_contamination/INVALIDATION.json', 'ec8dcd60d70119d26ef7efeff802f1c2a1ca54ce9b3a89281d603f9d4922078d'),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path, expected in INPUTS.values():
        if sha(path) != expected:
            raise ValueError(f'input changed: {path}')
    freeze, schedule, invalid = [json.loads(INPUTS[k][0].read_text()) for k in ('completion_freeze', 'schedule', 'invalidation')]
    schedules = {x['opportunity_id']: x for x in schedule['opportunities']}
    if len(schedules) != 32 or len(freeze['drafts']) != 30:
        raise ValueError('unexpected schedule or freeze size')
    records, seen_ids, seen_hashes = [], set(), set()
    for item in freeze['drafts']:
        opp = item['opportunity_id']
        path = Path(item['path'])
        if opp in seen_ids or opp not in schedules or sha(path) != item['sha256']:
            raise ValueError(f'freeze identity or hash mismatch: {opp}')
        draft = json.loads(path.read_text())
        sched = schedules[opp]
        if draft.get('root_id') != sched['root_id'] or draft.get('opportunity_id', opp) != opp:
            raise ValueError(f'draft mapping mismatch: {opp}')
        candidates = draft.get('candidates', [draft])
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f'unexpected draft count: {opp}')
        author_input = RUN / 'isolated_authors' / opp / 'INPUT.json'
        input_record = json.loads(author_input.read_text())
        if input_record.get('opportunity_id') != opp or input_record.get('root_id') != sched['root_id']:
            raise ValueError(f'author input mapping mismatch: {opp}')
        source = Path(input_record['source_path'])
        if sha(source) != input_record['source_sha256']:
            raise ValueError(f'source hash mismatch: {opp}')
        seen_ids.add(opp)
        seen_hashes.add(item['sha256'])
        records.append({
            'opportunity_id_controller_only': opp,
            'root_id_controller_only': sched['root_id'],
            'workflow_group_controller_only': sched['group'],
            'candidate_id_controller_only': candidates[0].get('candidate_id', draft.get('candidate_id', 'UNKNOWN')),
            'frozen_draft_path_controller_only': str(path),
            'frozen_draft_sha256': item['sha256'],
            'source_path_controller_only': str(source),
            'source_sha256': input_record['source_sha256'],
            'author_input_sha256': sha(author_input),
            'historical_semantic_fields': {k: 'UNKNOWN_NOT_YET_CODED' for k in ('event', 'natural_information_need', 'query_scope', 'answer_operation', 'evidence_family')},
            'historical_overlap_status': 'UNKNOWN_NOT_YET_COMPARED',
        })
    excluded = set(schedules) - seen_ids
    if excluded != set(invalid['affected_opportunities']) or len(excluded) != 2 or len(seen_hashes) != 30:
        raise ValueError('unexpected invalidated records or duplicate draft hashes')
    quarantined = []
    for item in invalid['files']:
        path = Path(item['path'])
        if sha(path) != item['sha256']:
            raise ValueError(f'quarantine hash mismatch: {path}')
        quarantined.append({'path_controller_only': str(path), 'sha256': item['sha256']})
    absent = []
    for opp in sorted(excluded):
        entry = schedules[opp]
        absent.append({
            'opportunity_id_controller_only': opp,
            'root_id_controller_only': entry['root_id'],
            'status': 'INVALIDATED_ROLE_CONTAMINATION_NO_RETRY',
            'formal_frozen_first_draft': False,
            'quarantined_first_draft_exists': any(opp in x['path_controller_only'] and x['path_controller_only'].endswith('CANDIDATES_FIRST_DRAFT.json') for x in quarantined),
        })
    result = {
        'schema_version': 'prior_round_frozen_draft_inventory_controller_v1',
        'status': 'INVENTORIED_NOT_SEMANTICALLY_REVIEWED_NOT_SELECTOR_READY',
        'visibility': 'CONTROLLER_ONLY',
        'source_bindings': {k: {'path': str(p), 'sha256': h} for k, (p, h) in INPUTS.items()},
        'counts': {'scheduled': 32, 'formal_frozen_first_drafts': len(records), 'distinct_frozen_draft_hashes': len(seen_hashes), 'invalidated_no_formal_frozen_draft': len(absent), 'quarantined_artifacts_hash_checked': len(quarantined), 'records_semantically_coded': 0},
        'per_root_formal_draft_counts': dict(sorted(collections.Counter(x['root_id_controller_only'] for x in records).items())),
        'limitations': [
            'Existing completion freeze is authoritative for the 30 selected paths; byte hashes match it now, but this cannot establish original creation time or a separate signature.',
            'This inventory does not assess question quality, novelty, independence, evaluation, training eligibility, or semantic overlap with the 212 existing records.',
            'All 30 semantic tuples are UNKNOWN_NOT_YET_CODED; absence of coding is not evidence of novelty.',
            'OPP 01/03/05 and OPP 31/32 have disclosures in the completion freeze; this inventory makes no additional protocol-validity judgment.',
            'OPP 02 and 04 were invalidated; the former has a quarantined draft. Neither is a formal frozen first draft.',
            'No Solver or scoring artifact was read by this inventory.',
        ],
        'records': records,
        'invalidated_opportunities': absent,
        'quarantined_artifacts': quarantined,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'sha256': sha(OUTPUT), 'counts': result['counts'], 'per_root': result['per_root_formal_draft_counts']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
