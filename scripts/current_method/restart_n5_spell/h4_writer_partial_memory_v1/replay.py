"""Offline admission replay. Frozen content verdicts are joined only after admission."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from ..core import atomic_json, atomic_text, canonical_json, sha256_file, sha256_bytes
from ..h4_writer_model_comparison_v1.runtime import read, rows, jsonl, directory, POLICY
from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from ..h4_r02_ordering_temporal_contract_v1.contract import register_runtime_context
from ..simple_append_v1.live_runtime import load_json_object_strict
from .admission import normalize_stage, admit_stage, repair_required, duplicate_key


def replay(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("REPLAY_MUST_NOT_WRITE_SOURCE_RUN")
    schedule = read(source / "schedule.json")
    mapping = rows(source / "audit/quality_view_mapping.jsonl")
    by_key = {(r['arm_id'], r['root_id'], r['slot_id'], r['stage'], r['raw_row_index']): r for r in mapping}
    quality = {r['view_id']: r for r in rows(source / 'reports/quality_review.jsonl')}
    assert len(mapping) == len(quality) == 76
    dependencies = [source / name for name in ('schedule.json', 'audit/quality_view_mapping.jsonl',
        'reports/quality_review.jsonl', 'reports/dedup_review.jsonl', 'reports/history_overlap_review.jsonl',
        'reports/cross_arm_and_discarded_pairs.jsonl', 'reports/runtime_final_candidates.jsonl')]
    accepted, slot_results = [], []
    for policy in ('legacy', 'partial_v1'):
        pools = defaultdict(lambda: {'candidates': [], 'records': [], 'views': {}})
        for slot in schedule:
            arm, root_id, intent = slot['arm_id'], slot['root_id'], slot['slot_id']
            original = directory(source, slot)
            target = destination / policy / arm / root_id
            shared = source / 'inputs/shared' / root_id
            expanded = read(shared / 'expanded_dialogue_clean.json')
            contract = read(shared / 'contracts.json')[slot['raw_index']]
            dependencies.extend([shared / 'expanded_dialogue_clean.json', shared / 'contracts.json'])
            retained, history, audits, requests = [], [], [], {}
            stages_used, counterfactual = [], False
            for stage, verb in (('primary', 'generator'), ('repair', 'repair')):
                rid = f'{root_id}-contract_guided-{verb}-{intent}'
                path = original / 'raw/responses' / (rid + '.json.txt')
                if not path.is_file():
                    if stage == 'primary':
                        raise ValueError('MISSING_HISTORICAL_PRIMARY')
                    counterfactual = True
                    break
                dependencies.append(path)
                raw_text = path.read_text()
                try:
                    raw, error = load_json_object_strict(raw_text), None
                except (ValueError, TypeError) as exc:
                    raw, error = None, type(exc).__name__
                atomic_text(target / 'raw/responses' / path.name, raw_text)
                identity = {'run_id': destination.name + ':' + policy, 'arm_id': arm, 'root_id': root_id}
                request = {'intent_id': intent, 'source': 'contract_guided', 'stage': stage,
                           'raw_sha256': sha256_bytes(raw_text.encode())}
                registry_path = target / 'ordering_runtime_context.json'
                previous = read(registry_path).get('requests', {}) if registry_path.exists() else {}
                requests = {**previous, rid: request}
                register_runtime_context(target, POLICY, {**identity, 'requests': requests})
                context = {**identity, **request, 'request_id': rid, 'root_dir': target}
                if policy == 'partial_v1':
                    normalized = normalize_stage(raw, intent_id=intent, source='contract_guided',
                        contract=contract, expanded=expanded, root_id=root_id, stage=stage,
                        request_id=rid, parse_error=error, ordering_policy=POLICY, ordering_context=context)
                    retained, audit = admit_stage(normalized, retained=retained)
                    needs = repair_required(raw, normalized, audit, expanded=expanded)
                else:
                    normalized = normalize_batch(raw, intent_id=intent, source='contract_guided',
                        contract=contract, expanded=expanded, ordering_policy=POLICY, ordering_context=context)
                    if error:
                        normalized.update(valid=False, errors=[*normalized.get('errors', []), error])
                    valid, audit = generation._accepted_candidates(normalized, 'contract_guided')
                    retained = [generation._prefix_candidate(r, root_id=root_id, generation_source='contract_guided')
                                for r in valid] if audit['accepted'] else []
                    needs = not audit['accepted']
                stages_used.append(stage)
                audits.append({'stage': stage, **audit})
                history.append({'request_id': rid, 'intent_id': intent, 'validation': normalized})
                for candidate in retained:
                    own_stage = candidate.get('admission_stage', stage)
                    index = candidate.get('source_row_index') or candidate.get('raw_index')
                    if not index:
                        index = int(candidate.get('raw_candidate_id', candidate['candidate_id']).rsplit('Q', 1)[1].split(':')[0])
                    key = (arm, root_id, intent, own_stage, index)
                    pools[(arm, root_id)]['views'][candidate['candidate_id']] = by_key[key]['view_id']
                if not needs:
                    break
            pool = pools[(arm, root_id)]
            pool['candidates'].extend(retained)
            pool['records'].extend(history)
            historical_repair = (original / 'raw/responses' / f'{root_id}-contract_guided-repair-{intent}.json.txt').exists()
            slot_results.append({**slot, 'admission_policy': policy, 'stages_used': stages_used,
                'historical_repair_observed': historical_repair,
                'historical_repair_unused': historical_repair and 'repair' not in stages_used,
                'counterfactual_status': 'COUNTERFACTUAL_OUTPUT_UNOBSERVED' if counterfactual else 'OBSERVED',
                'retained_before_finalizer': len(retained), 'audits': audits})
        for (arm, root_id), pool in pools.items():
            target = destination / policy / arm / root_id
            generation.finalize_writer_candidates(root_dir=target, root_id=root_id,
                candidates=pool['candidates'], generation_records=pool['records'],
                expanded=read(source / 'inputs/shared' / root_id / 'expanded_dialogue_clean.json'),
                ordering_policy=POLICY, ordering_context={'run_id': destination.name + ':' + policy,
                    'arm_id': arm, 'root_id': root_id}, admission_policy=policy)
            for c in read(target / 'candidates_valid.json'):
                vid = pool['views'][c['candidate_id']]
                accepted.append({'admission_policy': policy, 'arm_id': arm, 'root_id': root_id,
                    'view_id': vid, 'candidate_id': c['candidate_id'], 'question_type': c['question_type'],
                    'exact_duplicate_key': duplicate_key(c), 'quality_judgment': quality[vid]['judgment']})
    frozen_final = {r['view_id'] for r in mapping if r['final_validator_accepted'] and not r['replaced_by_repair']}
    legacy_final = {r['view_id'] for r in accepted if r['admission_policy'] == 'legacy'}
    if frozen_final != legacy_final:
        raise AssertionError('LEGACY_REPLAY_DIFFERS_FROM_FROZEN_FINAL')
    # Semantic clustering is a reporting-only join to existing frozen diagnostics.
    clusters = rows(source / 'reports/dedup_review.jsonl')
    history = {r['view_id']: r for r in rows(source / 'reports/history_overlap_review.jsonl')}
    pair_review = rows(source / 'reports/cross_arm_and_discarded_pairs.jsonl')
    view_mapping = {r['view_id']: r for r in mapping}
    additions = []
    for accepted_row in accepted:
        if accepted_row['admission_policy'] != 'partial_v1' or accepted_row['view_id'] in legacy_final:
            continue
        vid = accepted_row['view_id']
        mapped = view_mapping[vid]
        additions.append({**accepted_row, 'slot_id': mapped['slot_id'], 'stage': mapped['stage'],
            'original_batch_accepted': mapped['minimum_batch_accepted'],
            'original_replaced_by_repair': mapped['replaced_by_repair'],
            'quality_reason': quality[vid]['reason'],
            'old_retained_comparisons': [r for r in pair_review if vid in (r['left_view_id'], r['right_view_id'])
                and (r['left_view_id'] in legacy_final or r['right_view_id'] in legacy_final)],
            'history_overlap_status': 'UNDETERMINED_WHERE_HISTORICAL_ANSWER_CONTENT_MISSING'})
    metrics = []
    for policy in ('legacy', 'partial_v1'):
        for arm, root_id in sorted({(r['arm_id'], r['root_id']) for r in schedule}):
            selected = [r for r in accepted if (r['admission_policy'], r['arm_id'], r['root_id']) == (policy, arm, root_id)]
            passes = [r for r in selected if r['quality_judgment'] == 'PASS']
            parent = {r['view_id']: r['view_id'] for r in passes}
            def find(v):
                while parent[v] != v:
                    v = parent[v]
                return v
            def union(vs):
                vs = [v for v in vs if v in parent]
                for v in vs[1:]:
                    parent[find(v)] = find(vs[0])
            relations, exact = defaultdict(list), defaultdict(list)
            for item in clusters:
                if (item['arm_id'], item['root_id']) != (arm, root_id):
                    continue
                for c in item['clusters']:
                    union(c['member_view_ids'])
                    if c.get('answer_relation'):
                        relations[(item['question_type_scope'], c['answer_relation'])].extend(c['member_view_ids'])
            for members in relations.values():
                union(members)
            for r in passes:
                exact[r['exact_duplicate_key']].append(r['view_id'])
            for members in exact.values():
                union(members)
            groups = defaultdict(list)
            for r in passes:
                groups[find(r['view_id'])].append(r['view_id'])
            judgments = Counter(r['quality_judgment'] for r in selected)
            unique = Counter('ordering' if quality[vid]['question_type'] == 'ordering' else 'SCMS' for vid in groups)
            metrics.append({'admission_policy': policy, 'arm_id': arm, 'root_id': root_id,
                'program_accepted': len(selected), 'content_counts': dict(judgments),
                'supported_clusters_after_known_dedup': dict(unique), 'supported_clusters': list(groups.values()),
                'supported_retained_stages': sorted({view_mapping[r['view_id']]['stage'] for r in passes}),
                'new_vs_legacy_view_ids': [r['view_id'] for r in selected if r['view_id'] not in legacy_final],
                'history_overlap': {r['view_id']: dict(Counter(c['judgment'] for c in history.get(r['view_id'], {}).get('comparisons', []))) for r in passes},
                'dedup_scope': 'Frozen within-layer clusters, named answer relations and exact fingerprints only; missing cross-layer/history relations remain unverified.'})
    frozen = [{'path': str(p), 'sha256': sha256_file(p)} for p in sorted(set(dependencies))]
    jsonl(destination / 'accepted_candidates.jsonl', accepted)
    jsonl(destination / 'newly_retained_candidates.jsonl', additions)
    atomic_json(destination / 'slot_replay.json', slot_results)
    atomic_json(destination / 'input_freeze.json', frozen)
    result = {'status': 'OFFLINE_ADMISSION_REPLAY_COMPLETE', 'raw_candidates': len(mapping),
        'legacy_final_identity_match': True, 'metrics': metrics,
        'identical_initial_program_dedup_state': 'EMPTY_PER_POLICY_ROOT_ARM_SLOT',
        'frozen_history_used_only_for_post_admission_diagnostics': True,
        'unused_historical_repairs_partial': sum(r['historical_repair_unused'] for r in slot_results if r['admission_policy'] == 'partial_v1'),
        'counterfactual_unobserved': [r for r in slot_results if r['counterfactual_status'] != 'OBSERVED'],
        'new_external_calls': 0, 'semantic_admission': False, 'new_semantic_reviews': 0,
        'historical_cost_savings_claimed': False, 'source_run_modified': False}
    atomic_json(destination / 'replay_summary.json', result)
    return result
