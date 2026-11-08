#!/usr/bin/env python3
"""Prepare the first prequeued root for score-free historical support review.

This is boundary-auditor input, never an evaluation-author release or clearance.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/root/autodl-tmp/gxy-selfplay')
WEEK = BASE / 'research-protocols/selfplay_week_20261019'
PREP = WEEK / 'day3_common_evaluation_v1'
OUT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError('Refusing to overwrite a prior review input: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {'path': str(path), 'sha256': digest(data), 'bytes': len(data)}


def main():
    queue_path = PREP / 'boundary/SEMANTIC_RECONSTRUCTION_QUEUE.json'
    gaps_path = PREP / 'boundary/ROOT_LOCAL_CLEARANCE_GAPS.json'
    queue = json.loads(queue_path.read_text())
    job = queue['jobs'][0]
    root_id = job['root_id']
    gaps = next(r for r in json.loads(gaps_path.read_text())['roots']
                if r['root_id'] == root_id)
    wanted = set(job['record_ids'])
    records = [r for r in gaps['records'] if r['exclusion_record_id'] in wanted]
    assert len(records) == job['unique_unresolved_records'] == 35
    wanted_candidates = {cid for r in records for cid in r['candidate_ids']}
    history_path = BASE / 'restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/difficulty_manifest.jsonl'
    # Deliberately whitelist the public candidate, never serialize the wrapper.
    public_fields = ('candidate_id', 'root_id', 'question_type', 'question', 'options', 'items')
    public = {}
    source_paths = {}
    for line in history_path.open():
        wrapper = json.loads(line)
        if wrapper['root_id'] != root_id or wrapper['candidate_id'] not in wanted_candidates:
            continue
        raw = wrapper['candidate']
        candidate = {k: raw[k] for k in public_fields if k in raw}
        assert set(candidate) <= set(public_fields)
        public[wrapper['candidate_id']] = candidate
        source_paths[wrapper['dialogue_sha256']] = Path(wrapper['dialogue_path'])
    assert set(public) == wanted_candidates
    sources = []
    for source_sha, path in sorted(source_paths.items()):
        data = path.read_bytes()
        assert digest(data) == source_sha
        raw = json.loads(data)
        turns = []
        for turn in raw['turns']:
            turns.append({k: turn[k] for k in ('turn_ref', 'turn_id', 'speaker', 'text',
                                              'utterance', 'scene_ref') if k in turn})
        assert len({t.get('turn_ref', t.get('turn_id')) for t in turns}) == len(turns)
        sources.append({'original_source_sha256': source_sha, 'turns': turns,
                        'full_original_turn_count': len(raw['turns'])})
    rows = []
    for record in sorted(records, key=lambda x: x['exclusion_record_id']):
        rows.append({
            'exclusion_record_id': record['exclusion_record_id'],
            'root_id': root_id,
            'original_split': job['original_split'],
            'source_sha256': record['source_sha256'],
            'known_evidence_turn_refs': record['known_evidence_turn_refs'],
            'public_questions': [public[cid] for cid in sorted(record['candidate_ids'])],
        })
    payload = {
        'schema_version': 'score_free_boundary_support_review_input_v1',
        'role': 'BOUNDARY_AUDITOR_ONLY_NOT_EVALUATION_AUTHOR_INPUT',
        'root_id': root_id,
        'selection_rule': job['priority_rule'],
        'sources': sources,
        'records': rows,
        'limits': [
            'Contains public historical questions; never pass this bundle to a new evaluation author.',
            'No Gold, correctness, predictions, model outcomes or diagnostic summaries are exported.',
            'Known evidence refs are retrieval hints, not proof of all-option support completeness.',
            'Reconstruction is not the five-category candidate-family exclusion or content freeze.',
        ],
    }
    bundle = write(OUT / 'inputs' / (root_id + '.json'), payload)
    # A manifest timestamp is created once to keep repeat execution stable.
    manifest_path = OUT / 'INPUT_MANIFEST.json'
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {
        'schema_version': 'boundary_semantic_first_batch_manifest_v1',
        'created_at_utc': existing.get('created_at_utc', datetime.now(timezone.utc).isoformat()),
        'status': 'INPUTS_PREPARED_SEMANTIC_RECONSTRUCTION_PENDING',
        'root_id': root_id,
        'records': len(rows),
        'selection_rule': job['priority_rule'],
        'input': bundle,
        'original_history_binding_internal_only': {
            'path': str(history_path), 'sha256': digest(history_path.read_bytes())},
        'source_bindings_internal_only': [
            {'path': str(path), 'sha256': source_sha}
            for source_sha, path in sorted(source_paths.items())],
        'queue_sha256': digest(queue_path.read_bytes()),
        'gaps_sha256': digest(gaps_path.read_bytes()),
        'new_evaluation_questions': 0,
        'new_solver_calls': 0,
        'new_training_runs': 0,
    }
    write(manifest_path, manifest)
    print(json.dumps({'status': manifest['status'], 'root_id': root_id,
                      'records': len(rows), 'sources': len(sources), 'input': bundle}))


if __name__ == '__main__':
    main()
