"""Offline historical-prior coverage proof; never prepares production QA state."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'engine'))
import batch_scope
import common
import prior_rendering


def public_key(row):
    return (row.get('question_type'), tuple(sorted(
        ' '.join(text.casefold().split()) for text in row.get('facts', []))))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def audit():
    assets = common.load_multi('asset_generation')
    spec = common.read(BASE / 'inputs/SHARD_SPEC.json')
    spec_rows = {row['root_id']: row for row in spec['roots']}
    roots = []
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for root in batch_scope.EXPECTED_ROOTS:
            feedback = assets.source_adapter.feedback_data(root)
            historical_pool_binding = feedback['receipt']['historical_pool']
            historical_pool = common.read(historical_pool_binding['path'])
            history = feedback['candidate_history']
            pool_ids = {row['candidate_id'] for row in historical_pool}
            common.need(len(pool_ids) == len(historical_pool)
                        and pool_ids == {row['candidate_id'] for row in history},
                        'original_candidate_pool_history_incomplete')
            history_path = temporary / (root + '_candidate_history.json')
            common.write(history_path, history)
            raw, files = assets.source_adapter.prior_candidates(
                root, feedback, history_path, assets.v15)
            raw_by_id = {}
            for row in raw:
                raw_by_id.setdefault(row.get('candidate_id'), []).append(row)
            historical_keys = set()
            empty_ids = []
            for row in history:
                question = row.get('historical_question') or {}
                facts = list((question.get('options') or question.get('items') or {}).values())
                matches = [prior for prior in raw_by_id.get(row['candidate_id'], [])
                           if prior.get('facts') == facts
                           and prior.get('question_type') == question.get('question_type')
                           and prior.get('question') == question.get('question')]
                common.need(bool(matches), 'original_history_candidate_missing_or_truncated')
                if facts:
                    historical_keys.add(public_key(matches[0]))
                else:
                    empty_ids.append(row['candidate_id'])
            projected = assets.legacy.prompt_prior(raw)
            raw_keys = {public_key(row) for row in raw if row.get('facts')}
            projected_keys = {public_key(row) for row in projected}
            common.need(len(projected_keys) == len(projected)
                        and projected_keys == raw_keys
                        and historical_keys <= projected_keys,
                        'public_projection_lost_distinct_statement_set')
            for row in projected:
                hard = any(prior.get('hard_dedup') for prior in raw
                           if prior.get('facts') and public_key(prior) == public_key(row))
                expected = ('avoid_exact_six_statement_set_only_paraphrases_allowed'
                            if hard else 'warning_only')
                common.need(row['dedup_action'] == expected,
                            'public_projection_lost_dedup_action')
            grouped = prior_rendering.encode_prior(projected)
            decoded = prior_rendering.decode_prior(grouped)
            common.need(decoded == projected, 'grouped_prior_does_not_roundtrip')
            # Read the production context constructor against temporary files only.
            fixture = temporary / root
            fixture.mkdir()
            common.write(fixture / 'MANIFEST.json', {'root_id': root, 'family_id': 'offline_prior_audit'})
            current = spec_rows[root]
            parent = common.read(current['public_parent'])
            source_binding = {'path': parent['source_path'], 'sha256': parent['source_sha256']}
            common.verify_bindings([source_binding])
            source = common.read(source_binding['path'])
            common.write(fixture / 'SOURCE.json', source)
            common.write(fixture / 'PARENT.json', parent)
            common.write(fixture / 'PRIOR.json', raw)
            context = assets.context_for(fixture)
            common.need(context['source_turns'] == source['turns']
                        and context['source_metadata'] == {key: value for key, value in source.items() if key != 'turns'}
                        and context['previous_candidates_dedup_only'] == projected,
                        'context_dropped_source_or_public_prior')
            rendered_context, rendering_receipt = prior_rendering.grouped_context(context)
            common.need(prior_rendering.decode_prior(rendered_context['previous_candidates_dedup_only']) == projected
                        and rendered_context['source_turns'] == source['turns'],
                        'writer_rendering_dropped_history_or_source')
            bound = {Path(row['path']).resolve(): row for row in feedback['receipt']['source_files']}
            for path in files:
                path = Path(path).resolve()
                if path != history_path.resolve():
                    bound[path] = common.bind(path)
            roots.append({
                'root_id': root,
                'original_round_pool': historical_pool_binding,
                'original_round_pool_candidates': len(historical_pool),
                'original_round_pool_types': dict(Counter(row.get('question_type') for row in historical_pool)),
                'original_round_candidate_ids': sorted(pool_ids),
                'historical_candidate_history_count': len(history),
                'historical_candidate_history_canonical_sha256': digest(history),
                'raw_prior_rows_including_additional_historical_sources': len(raw),
                'original_round_candidates_with_complete_public_facts': len(history) - len(empty_ids),
                'original_round_candidates_without_public_options_or_items': empty_ids,
                'original_round_distinct_public_statement_sets': len(historical_keys),
                'all_raw_distinct_public_statement_sets': len(raw_keys),
                'projected_public_statement_sets': len(projected),
                'projected_public_statement_sets_canonical_sha256': digest(projected),
                'rendered_groups': len(grouped['groups']),
                'original_candidate_id_coverage_complete': True,
                'original_public_facts_exact_no_truncation': True,
                'all_distinct_public_sets_preserved': True,
                'dedup_actions_preserved': True,
                'all_projected_indices_roundtrip_exactly': True,
                'current_full_source_turns_preserved': len(source['turns']),
                'current_source': source_binding,
                'bound_historical_files': [bound[path] for path in sorted(bound)],
                'writer_rendering_receipt': rendering_receipt,
            })
    result = {
        'version': 'v96_historical_prior_offline_coverage_audit_v1',
        'status': 'PASS', 'created_at': common.now(),
        'scope': 'original_pool_to_feedback_to_raw_prior_to_public_projection_to_lossless_grouping',
        'method': 'frozen_functions_run_against_temporary_history_and_context_files_only',
        'model_calls': 0, 'actual_production_prepare_performed': False,
        'actual_prepared_PRIOR_still_requires_separate_verification': True,
        'candidate_id_counts_and_distinct_public_statement_set_counts_are_separate': True,
        'original_duplicate_public_sets_follow_existing_frozen_projection': True,
        'audit_script': common.bind(__file__),
        'spec': common.bind(BASE / 'inputs/SHARD_SPEC.json'),
        'source_adapter': common.bind(assets.source_adapter.__file__),
        'asset_generation': common.bind(assets.__file__),
        'legacy_public_projection': common.bind(assets.legacy.__file__),
        'original_public_projection': common.bind(assets.legacy.legacy.__file__),
        'current_lossless_prior_renderer': common.bind(prior_rendering.__file__),
        'roots': roots,
    }
    target = BASE / 'evidence/PRIOR_COVERAGE_OFFLINE.json'
    common.write(target, result)
    print(json.dumps({'status': result['status'], 'output': str(target),
                      'roots': [{key: row[key] for key in ('root_id', 'original_round_pool_candidates',
                                'raw_prior_rows_including_additional_historical_sources',
                                'original_round_distinct_public_statement_sets',
                                'projected_public_statement_sets')} for row in roots]}, ensure_ascii=False))


if __name__ == '__main__':
    audit()
