"""Compile observed feedback and reference coverage from frozen artifacts."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import BASELINE, BANK, COMMON, ROOTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verified(path, hashes):
    path = Path(path)
    expected = path.with_name(path.name + '.sha256').read_text().split()[0]
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'input_hash_mismatch:{path}')
    hashes[str(path)] = actual


def observations(value):
    """Only empirical fields survive. No diagnostic/curriculum labels."""
    allowed = {'full_context_accuracy', 'no_context_accuracy', 'context_gain',
               'full_context_correct_count', 'full_context_total_count',
               'no_context_correct_count', 'no_context_total_count',
               'position_mechanism', 'clean_error_set', 'inclusive_error_set',
               'shared_error_set', 'clean_rollout_count', 'inclusive_rollout_count',
               'shared_overlap', 'question_type'}
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if key in allowed:
            result[key] = item
        elif isinstance(item, dict):
            nested = observations(item)
            if nested:
                result[key] = nested
    return result


def compile_inputs(root):
    if root not in ROOTS:
        raise ValueError(f'root_not_in_frozen_scope:{root}')
    hashes = {}
    manifest = BASELINE / 'authoritative_dialogue_manifest.jsonl'
    common_path = COMMON / 'common_feedback_data_manifest.jsonl'
    scope_path = BANK / 'active_train_scope_manifest.jsonl'
    for path in (manifest, common_path, scope_path):
        verified(path, hashes)
    source = next(x for x in rows(manifest) if x['root_id'] == root)
    dialogue_path = Path(source['path'])
    if sha(dialogue_path) != source['sha256']:
        raise ValueError(f'dialogue_hash_mismatch:{root}')
    hashes[str(dialogue_path)] = sha(dialogue_path)
    dialogue = read(dialogue_path)
    refs = [t['turn_ref'] for t in dialogue['turns']]
    if len(refs) != len(set(refs)) or len(refs) != source['turn_count']:
        raise ValueError(f'dialogue_identity_invalid:{root}')
    if not any(r.startswith('R03:') for r in refs):
        raise ValueError(f'r03_missing:{root}')
    common = [r for r in rows(common_path) if r['source_identity']['root_id'] == root]
    feedback = []
    for i, record in enumerate(common, 1):
        feedback.append({
            'feedback_id': f'F{i:03}',
            'source_identity': record['source_identity'],
            'source_qa': record['source_qa'],
            'evidence_refs': record['evidence']['authoritative_evidence_refs'],
            'solver_observations': observations(record['solver_observations']),
            'canonical_confusion': observations(record['transition_behavior'].get('canonical_confusion', {})),
            'source_provenance': {'file': str(common_path), 'sha256': hashes[str(common_path)],
                                  'observations': record['solver_observations'].get('provenance')},
        })
    by_path = defaultdict(set)
    for item in rows(scope_path):
        if item.get('root_id') == root and item.get('split') == 'train':
            by_path[item['source_path']].add(item['candidate_id'])
    history = []
    for path, ids in sorted(by_path.items()):
        # Source pools may predate the sidecar convention; pin their exact bytes.
        hashes[path] = sha(path)
        pool = rows(path) if Path(path).suffix == '.jsonl' else read(path)
        if not isinstance(pool, list):
            raise ValueError(f'candidate_pool_not_array:{path}')
        index = {r['candidate_id']: r for r in pool}
        if ids - index.keys():
            raise ValueError(f'bank_candidate_missing:{path}')
        for cid in sorted(ids):
            row = index[cid]
            history.append({k: row.get(k) for k in (
                'candidate_id', 'question_type', 'question', 'options', 'items',
                'gold', 'correct_option', 'evidence_refs')})
    citation_counts = Counter(ref for h in history for ref in h['evidence_refs'] or [])
    scenes = defaultdict(list)
    for ref in refs:
        if ref.startswith(('R02:', 'R03:')):
            scenes[ref.rsplit(':', 1)[0]].append(ref)
    gaps = []
    for scene, scene_refs in sorted(scenes.items()):
        uncited = [r for r in scene_refs if citation_counts[r] == 0]
        if uncited:
            gaps.append({'gap_id': f'G{len(gaps)+1:03}', 'scene_ref': scene,
                         'uncited_turn_refs': uncited,
                         'measurement': 'no_citation_in_frozen_active_train_bank',
                         'semantic_gap_status': 'UNVERIFIED'})
    packet = {'version': 'r03.feedback_packet.v2', 'root_id': root,
              'dialogue_sha256': hashes[str(dialogue_path)],
              'eligible_feedback': feedback, 'coverage_gaps': gaps,
              'history': history, 'source_manifest': source,
              'source_hashes': hashes}
    return dialogue, packet
