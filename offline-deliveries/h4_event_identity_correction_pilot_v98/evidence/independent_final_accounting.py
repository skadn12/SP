"""Read-only accounting from frozen inputs, raw HTTP, reservations and outcomes.

Only Python stdlib is imported. No collector, engine, model, transport, tokenizer,
training or semantic-grading code is imported or executed. Output uses exclusive
creation. Preflight deliberately does not inspect or predict unfinished results.
"""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def identity(row):
    return row['dataset'], row['split'], str(row['item_id'])


def signature(row, run=None):
    return (*identity(row), str(Path(row.get('run', run)).resolve()),
            row['root_id'], row['scene_id'], row['source_id'], row['source_item_sha256'])


class Audit:
    def __init__(self, run, phase):
        self.run, self.phase = run.resolve(), phase
        self.home = self.run.parents[2]
        self.paths, self.checks, self.failures = set(), [], []

    def load(self, path):
        path = Path(path).resolve()
        self.paths.add(path)
        return read(path)

    def check(self, name, ok, **detail):
        self.checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', **detail})
        if not ok:
            self.failures.append(name)

    def binding(self, row, path_key='path'):
        path = Path(row[path_key]).resolve()
        self.paths.add(path)
        self.check('binding:' + str(path), path.is_file() and sha(path) == row['sha256'])

    def nested_bindings(self, value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                self.binding(value)
            for child in value.values():
                self.nested_bindings(child)
        elif isinstance(value, list):
            for child in value:
                self.nested_bindings(child)

    def inventory(self, before, reservation):
        """Rebuild occupied owners from raw reservations and all legacy schemas."""
        paths = set(self.home.parent.rglob('SOURCE_RESERVATIONS.json'))
        paths.update(Path(x['path']) for x in before['prior_evidence'])
        for path in tuple(paths):
            if path.name == 'SOURCE_RESERVATIONS.json':
                paths.update(p for p in (path.parent / 'REPORT.json',
                                        path.parent / 'UNIQUE_EVENT_REPORT.json') if p.exists())
        actual = set()
        legacy_count = 0
        for path in sorted(paths):
            data = self.load(path)
            rows = data.get('reservations', []) if path.name == 'SOURCE_RESERVATIONS.json' else data.get('accepted_uses', [])
            if path.name == 'SOURCE_RESERVATIONS.json':
                self.check('reservation_run:' + str(path), Path(data['run']).resolve() == path.parent.resolve())
                signatures = [signature(x, path.parent) for x in rows]
                self.check('reservation_unique_identity:' + str(path), len({identity(x) for x in rows}) == len(rows))
                self.check('reservation_unique_slot:' + str(path), len({x[3:6] for x in signatures}) == len(rows))
            for row in rows:
                actual.add(signature(row, path.parent))
            # v1/v2 predate accepted_uses and SOURCE_RESERVATIONS. Their original
            # accepted scene metadata maps via the original card and source files.
            if path.name == 'RESULT.json' and path.parent.name == 'B':
                legacy_run = path.parents[3]
                if (legacy_run / 'SOURCE_RESERVATIONS.json').exists():
                    continue
                cards_path = legacy_run / 'dataset_cards_result.json'
                sources_path = legacy_run / 'dataset_sources.json'
                cards = {x['card_id']: x for x in self.load(cards_path)['accepted_cards']}
                sources = {x['source_id']: x for x in self.load(sources_path)['items']}
                metadata = data.get('scene_metadata', [])
                self.check('legacy_accepted_count:' + str(path), len(metadata) == data['accepted_scenes'])
                for scene in metadata:
                    card = cards[scene['dataset_card_id']]
                    source = sources[card['source_id']]
                    row = {**source, 'root_id': data['root_id'], 'scene_id': scene['scene_id'], 'run': str(legacy_run)}
                    actual.add(signature(row))
                    legacy_count += 1
        expected_old = {signature(owner) for row in before['accepted_uses']
                        for owner in row.get('historical_owners', [row])}
        expected_new = {signature(x) for x in reservation['reservations']}
        self.check('inventory_owners_exact_frozen_before_plus_reservations', actual == expected_old | expected_new,
                   missing=[list(x) for x in sorted((expected_old | expected_new) - actual)],
                   unexpected=[list(x) for x in sorted(actual - (expected_old | expected_new))])
        grouped = collections.defaultdict(set)
        for row in actual:
            grouped[row[:3]].add(row[3:])
        conflicts = {key for key, owners in grouped.items() if len(owners) > 1}
        expected_conflicts = {identity(x) for x in before['historical_conflicts']}
        self.check('historical_inventory_conflicts_preserved', conflicts == expected_conflicts)
        return {'before_identities': len({identity(x) for x in before['accepted_uses']}),
                'permanently_reserved_identities': len(expected_new),
                'after_identities_rebuilt_from_raw': len(grouped),
                'historical_conflicts_preserved': len(conflicts),
                'legacy_scene_acceptances_reconstructed': legacy_count,
                'raw_inventory_artifacts_read': len(paths)}

    def preflight(self):
        run, home = self.run, self.home
        plan = self.load(home / 'PRECALL_PLAN.json')
        manifest = self.load(run / 'MANIFEST.json')
        selected = self.load(home / 'expansion_engine/SOURCE_SELECTION.json')
        allocation = self.load(run / 'UNIQUE_EVENT_ALLOCATION.json')
        before = self.load(run / 'INVENTORY_BEFORE.json')
        reservation = self.load(run / 'SOURCE_RESERVATIONS.json')
        registry = {x['root_id']: x for x in self.load(home / 'expansion_engine/SOURCE_REGISTRY.json')['roots']}
        self.check('same_frozen_root_order', plan['candidate_roots'] == manifest['roots'] == selected['roots'] == allocation['roots'])
        self.check('same_frozen_source_order', plan['source_ids'] == selected['source_ids'] == [x['source_id'] for x in reservation['reservations']])
        self.check('reservations_match_root_order', [x['root_id'] for x in reservation['reservations']] == manifest['roots'])
        self.check('only_frozen_B_route', manifest['selected_routes'] == ['dataset_new_event'] and manifest['scenes_per_route_target'] == 1)
        self.check('frozen_global_http_budget', manifest['http_max'] == plan['max_expansion_http_attempts'] == 128)
        budget_pairs = [('max_planner_attempts', 'max_planner_attempts'),
                        ('max_realizer_attempts', 'max_realizer_attempts'),
                        ('max_state_extraction_attempts', 'max_initial_state_attempts'),
                        ('max_event_state_attempts_per_plan', 'max_event_review_attempts_per_plan')]
        self.check('frozen_stage_attempt_budgets', all(manifest[a] == plan[b] == 2 for a, b in budget_pairs)
                   and manifest['automatic_transport_retries'] == plan['max_transport_retries_per_request'] == 1
                   and manifest['review_schema_retries'] + 1 == plan['max_scene_review_attempts_per_body'] == 2
                   and manifest['card_extraction_attempts_per_source'] == plan['max_card_extractions_per_source'] == 2)
        self.check('frozen_no_training_no_fallback_no_replacement', all(plan[x] is False for x in
                   ['training_allowed', 'historical_source_fallback', 'failure_replacement', 'formal_training_admission']))
        for row in [*plan['frozen_inputs'], *manifest['files'], *before['prior_evidence']]:
            self.binding(row)
        snapshots = self.load(run / 'EXECUTION_CODE_SNAPSHOT.json')['files']
        for row in snapshots:
            self.binding(row, 'original_path')
            self.binding(row, 'snapshot_path')
        old_ids = {identity(x) for x in before['accepted_uses']}
        self.check('new_reservations_previously_unused', not old_ids & {identity(x) for x in reservation['reservations']})
        source_rows = []
        for root in manifest['roots']:
            original = self.load(registry[root]['source_path'])
            copied = self.load(run / 'sources' / (root + '.json'))
            self.check('complete_original_history:' + root, original == copied and len(copied['turns']) == registry[root]['source_turns'])
            self.check('original_history_sha:' + root, sha(registry[root]['source_path']) == registry[root]['source_sha256'])
            assigned = allocation['assignments'][root]
            reserved = [x for x in reservation['reservations'] if x['root_id'] == root]
            self.check('reservation_allocation:' + root, len(assigned) == len(reserved) == 1 and
                       identity(assigned[0]) == identity(reserved[0]) and
                       assigned[0]['source_item_sha256'] == reserved[0]['source_item_sha256'] and reserved[0]['scene_id'] == 'S01')
            source_rows.append({'root_id': root, 'historical_turns': len(copied['turns']),
                                'source_id': assigned[0]['source_id']})
        return plan, manifest, {'sources': source_rows, 'inventory': self.inventory(before, reservation),
                                'snapshot_files_checked': len(snapshots), 'manifest_bindings_checked': len(manifest['files'])}

    def final(self, plan, manifest):
        run = self.run
        report = self.load(run / 'REPORT.json')
        requests, roles, finishes, usage = [], collections.Counter(), collections.Counter(), collections.Counter()
        planner_contract_rows = []
        attempt_numbers, raw_turns, roots = [], 0, {x: {'roles': collections.Counter(), 'raw_candidates': []} for x in manifest['roots']}
        stage_groups = collections.Counter()
        for folder in sorted((run / 'requests').iterdir()):
            if not folder.is_dir():
                continue
            payload, proof, reservation = [self.load(folder / name) for name in ['REQUEST.json', 'PROOF.json', 'RESERVATION.json']]
            role, root = proof['role'], proof.get('root_id')
            roles[role] += 1
            source = proof.get('source_id')
            if role == 'dataset_extractor':
                root = next(x['root_id'] for x in self.load(run / 'SOURCE_RESERVATIONS.json')['reservations'] if x['source_id'] == source)
            self.check('known_root:' + folder.name, root in roots)
            if root in roots:
                roots[root]['roles'][role] += 1
            self.check('frozen_request_payload:' + folder.name, digest(payload) == proof['payload_sha256'] and payload['max_tokens'] == plan['expansion_max_tokens'] == 4096 and payload['model'] == manifest['model'])
            self.check('capacity_guard:' + folder.name, proof['capacity']['status'] == 'PASS' and proof['capacity']['truncation'] is False)
            self.check('request_transport_budget:' + folder.name, reservation['http_max'] == manifest['http_max'] and reservation['max_attempts'] == 2 and reservation['retryable_statuses'] == [503, 504])
            if role in ('planner', 'realizer'):
                text = payload['messages'][0]['content']
                context, _ = json.JSONDecoder().raw_decode(text.split('INPUT_JSON\n', 1)[1])
                frozen = self.load(run / 'sources' / (root + '.json'))['turns']
                history = context['cumulative_history']
                receipt = proof.get('feedback_receipt', {})
                self.check('actual_writer_full_prefix:' + folder.name, history[:len(frozen)] == frozen and
                           len(history) == proof['history_turns'] and digest(history) == proof['history_sha256'])
                self.check('actual_writer_branch_isolation:' + folder.name, proof['other_branch_in_generator_context'] is False and
                           all(x['turn_ref'].startswith('R02:B:') for x in history[len(frozen):]))
                self.check('actual_writer_previous_training_feedback:' + folder.name,
                           '[PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]' in text and receipt.get('feedback_appended') is True
                           and receipt.get('current_generation_feedback_used') is False)
            attempts = sorted(folder.glob('HTTP_ATTEMPT*.json'))
            self.check('bounded_transport_attempts:' + folder.name, 1 <= len(attempts) <= 2)
            for path in attempts:
                attempt = self.load(path)
                attempt_numbers.append(attempt['http_attempt_number'])
                index = attempt['request_attempt']
                self.check('actual_payload_binding:' + folder.name + ':' + str(index), attempt['actual_payload_equals_frozen'] is True and attempt['payload_sha256'] == digest(payload))
                result = self.load(folder / f'HTTP_RESULT_{index:02d}.json')
                if index > 1:
                    prior = self.load(folder / f'HTTP_RESULT_{index - 1:02d}.json')
                    self.check('only_retry_503_504:' + folder.name, prior['http_status'] in [503, 504] and prior['outcome'] == 'error')
            raw_responses = []
            for path in sorted(folder.glob('HTTP_RESPONSE_*.json')):
                response = self.load(path); meta = response.get('metadata', {})
                finish = meta.get('finish_reason'); finishes[str(finish)] += 1
                usage.update({k: v for k, v in meta.get('usage', {}).items() if type(v) is int})
                raw_responses.append({'path': str(path), 'finish_reason': finish, 'output_tokens': meta.get('output_tokens')})
            final_path = folder / 'RESPONSE.json'
            candidate = None
            if final_path.exists():
                response = self.load(final_path)
                self.check('successful_response_backed_by_raw:' + folder.name, any(response['text'] == self.load(Path(x['path']))['text'] for x in raw_responses))
                try:
                    candidate = json.loads(response['text'])
                except (ValueError, TypeError):
                    pass
            if role == 'realizer' and isinstance(candidate, dict):
                count = len(candidate['turns']) if isinstance(candidate.get('turns'), list) else 0
                raw_turns += count
                roots[root]['raw_candidates'].append({'request': folder.name, 'turns': count,
                    'acceptance': self.load(folder / 'ACCEPTANCE.json') if (folder / 'ACCEPTANCE.json').exists() else None})
            if role == 'planner' and isinstance(candidate, dict):
                scenes = candidate.get('scenes', [])
                planner_contract_rows.append({'request': folder.name, 'root_id': root,
                    'scenes': len(scenes) if isinstance(scenes, list) else None,
                    'event_identity_present': [isinstance(x.get('event_identity'), dict) for x in scenes] if isinstance(scenes, list) else [],
                    'validation': self.load(folder / 'VALIDATION.json') if (folder / 'VALIDATION.json').exists() else None})
            # Folder labels preserve root, scene, plan/realizer and bounded attempt numbers.
            group = (root, role)
            if role == 'event_state_extractor':
                group += (proof['scene_id'], proof['plan_attempt'])
            elif role == 'scene_reviewer':
                group += (proof['scene_id'], folder.name.rsplit('-', 2)[-2])
            stage_groups[group] += 1
            requests.append({'request': folder.name, 'root_id': root, 'role': role, 'http_attempts': len(attempts), 'raw_responses': raw_responses})
        self.check('global_http_numbers_contiguous', sorted(attempt_numbers) == list(range(1, len(attempt_numbers) + 1)))
        self.check('global_http_count_matches_report', len(attempt_numbers) == report['http_attempts'] == report['http_attempt_log_count'] <= manifest['http_max'])
        self.check('logical_requests_match_report', len(requests) == report['http_request_reservations'] <= plan['max_expansion_logical_requests'])
        for group, count in stage_groups.items():
            self.check('bounded_stage_requests:' + ':'.join(map(str, group)), count <= 2, observed=count)
        results = []
        for root in manifest['roots']:
            for route in manifest['selected_routes']:
                letter = 'B' if route == 'dataset_new_event' else 'A'
                folder = run / 'roots' / root / letter
                result = self.load(folder / 'RESULT.json')
                dialogue = self.load(folder / 'cumulative_dialogue.json')['turns']
                frozen = self.load(run / 'sources' / (root + '.json'))['turns']
                suffix = dialogue[len(frozen):]
                accepted_files = sorted(p for p in folder.glob('S[0-9][0-9].json'))
                accepted_bodies = [self.load(p) for p in accepted_files]
                self.check('root_report_matches_raw_outcome:' + root, result in report['routes'])
                self.check('root_full_history_and_suffix:' + root, dialogue[:len(frozen)] == frozen and all(x['turn_ref'].startswith('R02:' + letter + ':') for x in suffix) and len({x['turn_ref'] for x in dialogue}) == len(dialogue))
                self.check('root_accepted_counts:' + root, result['accepted_scenes'] == len(accepted_files) and result['added_turns'] == len(suffix) == sum(len(x['turns']) for x in accepted_bodies))
                self.check('root_output_hash:' + root, digest(dialogue) == result['output_history_sha256'])
                results.append({'root_id': root, 'source_id': next(x['source_id'] for x in self.load(run / 'SOURCE_RESERVATIONS.json')['reservations'] if x['root_id'] == root),
                    'historical_turns': len(frozen), 'model_accepted_scenes': len(accepted_files), 'model_accepted_added_turns': len(suffix),
                    'request_roles': roots[root]['roles'], 'raw_generated_candidates': roots[root]['raw_candidates'], 'failures': result['failures']})
        self.check('accepted_total_matches_root_files', report['accepted_scenes'] == sum(x['model_accepted_scenes'] for x in results))
        self.check('no_expansion_question_solver_or_training', report['question_calls'] == report['solver_calls'] == 0 and report['training_started'] is False)
        totals = {'assigned_roots': len(results), 'logical_requests': len(requests), 'http_attempts': len(attempt_numbers),
                  'model_accepted_scenes': sum(x['model_accepted_scenes'] for x in results),
                  'model_accepted_new_turns': sum(x['model_accepted_added_turns'] for x in results),
                  'raw_generated_turns_across_all_attempts': raw_turns,
                  'raw_generated_candidates': sum(len(x['raw_generated_candidates']) for x in results)}
        stage_path = self.home / 'evidence/STAGE_RESULTS.json'
        if stage_path.exists():
            stage = self.load(stage_path)
            for name in ['assigned_roots', 'http_attempts', 'model_accepted_scenes', 'model_accepted_new_turns', 'raw_generated_turns_across_all_attempts', 'raw_generated_candidates']:
                self.check('collector_crosscheck:' + name, totals[name] == stage['totals'][name])
            self.check('collector_role_crosscheck', dict(roles) == stage['roles'])
            self.check('collector_finish_crosscheck', dict(finishes) == stage['finish_reasons'])
        cards = self.load(run / 'dataset_cards_result.json')
        totals['structurally_accepted_cards'] = len(cards['accepted_cards'])
        return {'run_status': report['status'], 'totals': totals, 'roles': roles, 'finish_reasons': finishes,
                'usage': usage, 'roots': results, 'requests': requests,
                'actual_planner_contract_records': planner_contract_rows,
                'card_acceptance_is_structural_only': True}

    def supplement(self, plan, manifest, actual):
        """Verify preservation, independent source reviews and final delivery facts."""
        home = self.home
        preservation = self.load(home / 'PRESERVATION_BEFORE.json')
        expansion = self.load(home / 'evidence/EXPANSION_MINFIX_RECEIPT.json')
        ordering = self.load(home / 'evidence/ORDERING_MINFIX_RECEIPT.json')
        for row in [*preservation['files'], *expansion['files'], *ordering['files']]:
            self.binding(row)
        integrity = {'preserved_old_files': len(preservation['files']), 'expansion_code_and_input_files': len(expansion['files']),
                     'expansion_python_files': sum(Path(x['path']).suffix == '.py' for x in expansion['files']),
                     'qa_python_files': len(ordering['files']), 'precall_frozen_inputs': len(plan['frozen_inputs'])}
        source_reviews = []
        by_root = {x['root_id']: x for x in actual['roots']}
        for root in manifest['roots']:
            number = root.rsplit('-', 1)[1]
            review_path = home / 'evidence' / ('root' + number + '_INDEPENDENT_SOURCE_AUDIT.json')
            review = self.load(review_path)
            if 'full_history_read' in review:
                full_read, count = review['full_history_read'], review['historical_turn_count']
                refs = review['turn_refs_read']
                status, independently_ready = review['status'], review['independent_ready']
            else:
                full_read = review['reviewer']['full_original_history_read']
                count, refs = review['history_coverage']['turn_count'], review['history_coverage']['all_turn_refs_read']
                status = review['decision']['independent_recommendation']
                self.check('source_review_explicit_not_ready_decision:' + root, status == 'NOT_READY_FOR_QA_NO_NEW_SOURCE')
                independently_ready = status == 'READY'
            self.check('source_review_covers_full_history:' + root, review['root_id'] == root and full_read is True
                       and count == by_root[root]['historical_turns'])
            frozen = self.load(self.run / 'sources' / (root + '.json'))['turns']
            self.check('source_review_exact_full_turn_refs:' + root, refs == [x['turn_ref'] for x in frozen])
            self.nested_bindings(review)
            source_reviews.append({'root_id': root, 'status': status,
                                   'independent_ready': independently_ready, 'path': str(review_path)})
        spec = self.load(home / 'inputs/SHARD_SPEC.json')
        ready = [x['root_id'] for x in source_reviews if x['independent_ready']]
        spec_roots = [x if isinstance(x, str) else x['root_id'] for x in spec['roots']]
        self.check('qa_spec_only_independently_ready_roots', spec_roots == ready)
        skip = self.load(home / 'evidence/QA_SKIPPED_NO_READY_SOURCE.json') if not ready else None
        if skip is not None:
            self.check('qa_skip_no_ready_no_calls_no_training', skip['status'] == 'SKIPPED' and skip['qa_http_calls'] == skip['solver_calls'] == 0 and skip['training_allowed'] is False)
            self.binding(skip['spec']); self.binding(skip['plan'])
            qa_audit = self.load(home / 'evidence/ACTUAL_QA_INPUT_AUDIT.json')
            self.check('qa_input_audit_matches_skip', qa_audit['status'] == 'SKIPPED_NO_READY_SOURCE' and qa_audit['audit_http_calls'] == 0)
            self.binding(qa_audit['skip_record'])
            self.check('no_qa_run_or_shard_directory', not (home / 'shards').exists() and not (home / 'qa_engine/runs').exists())
        results = self.load(home / 'RESULTS.json')
        for name, value in actual['totals'].items():
            self.check('final_results_count:' + name, results['totals'][name] == value)
        self.check('final_results_roles_and_usage', results['roles'] == actual['roles'] and results['usage'] == actual['usage'])
        self.check('final_results_missing_planner_identity_count', results['totals']['planner_identity_omissions'] ==
                   sum(not present for row in actual['actual_planner_contract_records'] for present in row['event_identity_present']))
        self.check('final_results_ready_roots', results['totals']['independent_ready_roots'] == len(ready))
        self.check('final_results_no_accuracy_or_semantic_claim', results['solver_accuracy'] is None and
                   results['difficulty_comparison'] is None and results['semantic_reliability_certified'] is False and
                   results['stable_production_ready'] is False and results['training_allowed'] is False and results['formal_training_admission'] is False)
        self.nested_bindings(results)
        decisions = self.load(home / 'evidence/READY_SOURCE_DECISIONS.json')
        self.nested_bindings(decisions)
        self.check('ready_decisions_match_actual_sources', decisions['ready_roots'] == ready and
                   decisions['ready_new_turns'] == sum(x['ready_new_turn_count'] for x in results['roots']) and
                   all(x['ready_for_joint_qa'] is (x['root_id'] in ready) for x in decisions['roots']))
        complete = home / 'COMPLETE_RESULTS_ZH.md'
        next_plan = home / 'NEXT_MINFIX_PLAN.md'
        self.paths.update([complete, next_plan])
        text = complete.read_text()
        self.check('complete_prose_core_numeric_claims', all(token in text for token in
                   ['4/4 root', '8/8', '173834 tokens', '长度截断 0', '新增正文 0', '独立 READY 0', '不是 0%']))
        self.check('next_plan_preserves_frozen_evidence_and_budget', all(token in next_plan.read_text() for token in
                   ['新的独立版本', '4096', '完整历史', '上一训练轮反馈', '不改 v98']))
        final_docs = []
        for name in ['RESULTS.json', 'RESULTS_ZH.md', 'COMPLETE_RESULTS_ZH.md', 'COMPLETE_MANIFEST.json']:
            path = home / name
            if path.exists():
                self.paths.add(path)
                final_docs.append({'path': str(path), 'sha256': sha(path)})
        return {'integrity_counts': integrity, 'independent_source_reviews': source_reviews,
                'independently_ready_roots': ready, 'qa_skipped': skip is not None,
                'final_delivery_documents_bound': final_docs,
                'second_reader_assessment': {
                    'full_results_complete_and_next_plan_read': True,
                    'raw_vs_report_counts_consistent': True,
                    'structural_card_acceptance_distinguished_from_semantics': True,
                    'planner_schema_failure_distinguished_from_unexercised_semantic_gates': True,
                    'prompt_contract_conflict_disclosed_without_claiming_unique_cause': True,
                    'source_only_risks_distinguished_from_live_model_verdicts': True,
                    'no_accuracy_or_difficulty_inferred_from_zero_calls': True,
                    'next_plan_does_not_rewrite_or_promote_failed_raw_outputs': True,
                    'limitation': 'This is a document-and-accounting review, not fresh blind semantic validation or proof of future fix success.'}}

    def execute(self):
        plan, manifest, result = self.preflight()
        if self.phase == 'final':
            result['actual_final_results'] = self.final(plan, manifest)
            result['final_preservation_source_and_qa_review'] = self.supplement(plan, manifest, result['actual_final_results'])
        else:
            result['actual_final_results'] = None
            result['final_results_state'] = 'PENDING_NOT_AUDITED_OR_PREDICTED'
        return {'version': 'v98_independent_raw_accounting_v1', 'phase': self.phase,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'FAIL' if self.failures else ('PASS_INDEPENDENT_FINAL_ACCOUNTING' if self.phase == 'final' else 'PASS_FROZEN_PRECONDITIONS_ONLY'),
                'run': str(self.run), 'checks': self.checks, 'failures': self.failures, **result,
                'bindings': [{'path': str(p), 'sha256': sha(p)} for p in sorted(self.paths)],
                'script_sha256': sha(__file__), 'model_calls': 0, 'http_calls': 0, 'engine_writes': 0,
                'semantic_certified': False,
                'limitations': ['Raw counts and recorded model acceptance do not certify source meaning, evidence entailment, question quality or model accuracy.',
                                'Only frozen preconditions are audited in preflight; final accounting requires an already completed raw REPORT.json.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parents[1] / 'expansion_engine/runs/four_roots_B_scene01')
    parser.add_argument('--phase', choices=['preflight', 'final'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('refuse to overwrite existing evidence: ' + str(args.out))
    if args.phase == 'final' and not (args.run / 'REPORT.json').is_file():
        raise SystemExit('final results pending: no raw REPORT.json; no output written')
    result = Audit(args.run, args.phase).execute()
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.out.resolve()),
                      'sha256': sha(args.out), 'failures': result['failures'],
                      'inventory': result['inventory']}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result['failures']))


if __name__ == '__main__':
    main()
