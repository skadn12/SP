"""Offline public binding recovery and lossless capacity evidence, append-only."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V90 = HERE.parent
PROJECT = V90.parent.parent
DELIVERIES = V90.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def statements(row):
    return row.get('options') or row.get('items')


def typed_key(row):
    return row['root_id'], row['question_type'], tuple(sorted(' '.join(t.split()).casefold()
                for t in statements(row).values()))


def recover():
    paths = [DELIVERIES / 'h4_anchor_records_mixed_batch_v88/evidence' /
                ('HISTORICAL_PUBLIC_PRIORS_' + root + '.json')
             for root in ['root103866', 'root135466']]
    paths += [DELIVERIES / directory / 'ordering/PUBLIC_QUESTIONS.json' for directory in [
        'h4_independent_roots_mixed_batch_v84', 'h4_label_anchor_minfix_mixed_batch_v85',
        'h4_program_gold_mixed_batch_v87', 'h4_anchor_records_mixed_batch_v88']]
    matches = {}
    for path in paths:
        for row in read(path)['questions']:
            if row['question_type'] == 'ordering':
                matches.setdefault(row['candidate_id'], []).append((row, path))
    receipt = {'version': 'v90_recovered_public_prior_bindings_v1',
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'recovery_sources': [bind(path) for path in paths], 'roots': [],
               'http_attempts': 0, 'private_gold_or_evidence_projected': False,
               'all_distinct_typed_sets_preserved': True, 'cross_type_multiselect_inventory_preserved': True}
    for root in ['root103866', 'root135466']:
        original_path = HERE / ('HISTORICAL_PUBLIC_PRIORS_COMPACT_' + root + '.json')
        original = read(original_path)
        questions = []; recovered = []
        for row in original['questions']:
            if row['question_type'] != 'ordering':
                public = {key: deepcopy(row[key]) for key in
                          ['candidate_id', 'root_id', 'question_type', 'question', 'options']}
            else:
                compatible = []
                for old, path in matches.get(row['candidate_id'], []):
                    if all(old.get(key) == row.get(key) for key in
                           ['candidate_id', 'root_id', 'question_type', 'question', 'items']):
                        compatible.append((old, path))
                assert compatible, 'missing_actual_public_binding:' + row['candidate_id']
                old, path = compatible[0]
                assert all(all(other[k] == old[k] for k in ['chronology_kind', 'source_path', 'source_sha256'])
                           for other, _ in compatible), 'conflicting_public_source_binding'
                assert sha(old['source_path']) == old['source_sha256'], 'historical_source_sha_changed'
                public = {key: deepcopy(old[key]) for key in ['candidate_id', 'root_id', 'question_type',
                          'chronology_kind', 'question', 'items', 'source_path', 'source_sha256']}
                recovered.append({'candidate_id': row['candidate_id'], 'recovered_from': bind(path),
                                  'source_binding': {'path': old['source_path'], 'sha256': old['source_sha256']},
                                  'chronology_kind': old['chronology_kind'], 'public_text_exact_match': True,
                                  'source_sha_verified': True,
                                  'source_binding_may_be_historical_lineage_only':
                                      old['chronology_kind'] == 'unspecified_historical_public_inventory'})
            questions.append(public)
        assert {typed_key(q) for q in questions} == {typed_key(q) for q in original['questions']}
        assert len(questions) == len({typed_key(q) for q in questions})
        all_path = HERE / ('HISTORICAL_PUBLIC_PRIORS_BOUND_' + root + '.json')
        ordering_path = HERE / ('HISTORICAL_ORDERING_PUBLIC_PRIORS_BOUND_' + root + '.json')
        header = {'version': 'v90_bound_public_prior_projection_v1', 'root_id': original['root_id'],
                  'original_compact_prior': bind(original_path), 'private_gold_and_evidence_excluded': True,
                  'all_distinct_typed_sets_preserved': True, 'source_dialogue_truncated': False}
        write(all_path, {**header, 'questions': questions})
        write(ordering_path, {**header, 'projection_question_type': 'ordering',
                             'questions': [q for q in questions if q['question_type'] == 'ordering']})
        receipt['roots'].append({'root_id': original['root_id'], 'all_types_path': bind(all_path),
                                 'ordering_path': bind(ordering_path), 'counts': dict(Counter(q['question_type'] for q in questions)),
                                 'recovered_ordering_rows': recovered,
                                 'initial_historical_lineage_semantics_preserved': True})
    write(HERE / 'PUBLIC_PRIOR_BINDING_RECOVERY_RECEIPT.json', receipt)


def capacity_evidence():
    sys.path.insert(0, str(DELIVERIES / 'h4_output_contract_multiselect_v79'))
    import asset_generation as assets
    adapter_path = V90 / 'engine/prior_rendering.py'
    adapter = load('v90_capacity_prior_rendering', adapter_path)
    _, _, annotate, capacity, dev = assets.base.transport()
    tokenizer = dev.tokenizer_from(str(assets.base.TOKENIZER))
    def count(value):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        return len(tokenizer.encode(text, add_special_tokens=False))
    receipt = {'version': 'v90_lossless_prior_capacity_recovery_v1',
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'adapter': bind(adapter_path),
               'all_cross_type_inventory_preserved': True, 'http_attempts': 0,
               'source_bytes_and_turns_unchanged': True, 'frozen_feedback_suffix_unchanged': True,
               'raw_prior_unchanged': True, 'token_count_components_not_additive_due_to_token_boundaries': True,
               'roots': []}
    for root in ['root103866', 'root135466']:
        run = V90 / 'shards/shard_01r1/multiselect/assets' / (root + '_family')
        ctx = assets.context_for(run); new_ctx, adapter_receipt = adapter.grouped_context(ctx)
        original_payload = read(run / 'INITIAL_REQUEST.json')
        before_sha = {name: sha(run / name) for name in ['SOURCE.json', 'PRIOR.json', 'INITIAL_REQUEST.json']}
        manifest = read(run / 'MANIFEST.json')
        payload, original_feedback_receipt = assets.legacy.legacy.payload_with_feedback(
            run, manifest['root_id'], assets.INTRO, new_ctx, assets.CAP)
        tail = assets.render_task_tail(manifest, assets.FACT_IDS)
        payload['messages'][0]['content'] += tail
        for key in ctx:
            if key != 'previous_candidates_dedup_only':
                assert new_ctx[key] == ctx[key], 'unrelated_context_changed'
        assert adapter.decode_prior(new_ctx['previous_candidates_dedup_only']) == ctx['previous_candidates_dedup_only']
        source_json = json.dumps(ctx['source_turns'], ensure_ascii=False, separators=(',', ':'))
        assert source_json in original_payload['messages'][0]['content'] and source_json in payload['messages'][0]['content']
        suffix_path = run / 'feedback' / manifest['root_id'] / 'writer__M_clean_policy_feedback.txt'
        suffix = suffix_path.read_text()
        assert suffix in original_payload['messages'][0]['content'] and suffix in payload['messages'][0]['content']
        profile = read(run / 'provider_profile.json')
        profile['output_limit'] = payload['max_tokens']; profile['request_parameters']['max_tokens'] = payload['max_tokens']
        actual_capacity = annotate(capacity(profile, tokenizer, payload['messages'], payload['max_tokens']))
        assert actual_capacity['status'] == 'PASS'
        assert before_sha == {name: sha(run / name) for name in before_sha}
        output_path = HERE / ('RECOVERED_GROUPED_REQUEST_' + root + '.json')
        write(output_path, payload)
        raw_prior = read(run / 'PRIOR.json')
        receipt['roots'].append({'root_id': manifest['root_id'], 'failed_prepare_input': bind(run / 'INITIAL_REQUEST.json'),
                                 'unchanged_source': bind(run / 'SOURCE.json'), 'unchanged_raw_prior': bind(run / 'PRIOR.json'),
                                 'unchanged_feedback_suffix': bind(suffix_path), 'replayed_grouped_payload': bind(output_path),
                                 'before_capacity': read(run / 'CAPACITY.json'), 'after_capacity': actual_capacity,
                                 'roundtrip_exact': True, 'adapter_receipt': adapter_receipt,
                                 'original_feedback_receipt': original_feedback_receipt,
                                 'raw_prior_types': dict(Counter(r['question_type'] for r in raw_prior)),
                                 'projected_prior_types': dict(Counter(r['question_type'] for r in ctx['previous_candidates_dedup_only'])),
                                 'component_proxy_tokens': {
                                     'context_before': {k: count(v) for k, v in ctx.items()},
                                     'grouped_public_prior': count(new_ctx['previous_candidates_dedup_only']),
                                     'frozen_feedback_suffix': count(suffix), 'intro': count(assets.INTRO), 'task_tail': count(tail)},
                                 'source_metadata_proxy_tokens': {k: count(v) for k, v in ctx['source_metadata'].items()},
                                 'original_and_grouped_source_json_sha': hashlib.sha256(source_json.encode()).hexdigest()})
    write(HERE / 'LOSSLESS_PRIOR_CAPACITY_RECOVERY_RECEIPT.json', receipt)


def failures():
    rows = []
    for name in ['shard_01', 'shard_01r1']:
        shard = V90 / 'shards' / name
        errors = read(shard / 'BATCH_PREPARATION.ERROR.json')
        assert not list(shard.rglob('HTTP_ATTEMPT.json'))
        rows.append({'shard': str(shard), 'errors': bind(shard / 'BATCH_PREPARATION.ERROR.json'),
                     'multiselect_error': bind(shard / 'multiselect/PREPARATION.ERROR.json'),
                     'errors_summary': errors['errors'], 'qa_http_attempts': 0,
                     'no_http_attempt_artifacts': True, 'failed_shard_overwritten': False,
                     'automatic_retry': False,
                     'inputs': [bind(path) for path in sorted(shard.rglob('CAPACITY.json'))]})
    write(HERE / 'PREPARATION_FAILURE_INDEX.json', {'version': 'v90_offline_prepare_failure_index_v1',
          'created_at_utc': datetime.now(timezone.utc).isoformat(), 'shards': rows,
          'capacity_adapter': bind(V90 / 'engine/prior_rendering.py'),
          'recovery_receipt': bind(HERE / 'LOSSLESS_PRIOR_CAPACITY_RECOVERY_RECEIPT.json'),
          'public_binding_receipt': bind(HERE / 'PUBLIC_PRIOR_BINDING_RECOVERY_RECEIPT.json'),
          'history_and_failed_shards_unchanged': True, 'qa_http_attempts_total': 0})


if __name__ == '__main__':
    recover()
    capacity_evidence()
    failures()
    print(json.dumps({'status': 'PASS', 'http_attempts': 0, 'outputs': [
          'PUBLIC_PRIOR_BINDING_RECOVERY_RECEIPT.json', 'LOSSLESS_PRIOR_CAPACITY_RECOVERY_RECEIPT.json',
          'PREPARATION_FAILURE_INDEX.json']}))
