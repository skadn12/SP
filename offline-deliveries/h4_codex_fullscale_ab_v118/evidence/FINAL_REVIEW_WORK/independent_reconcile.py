"""Independent read-only final-delivery reconciliation by the named reviewer.

Run only after the full delivery and its procedure supplements exist. This
script does not import delivery/status/scoring code or call a model. It writes
no files itself; the reviewer retains its output separately.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
from copy import deepcopy
from pathlib import Path
import re
import statistics

B = Path(__file__).resolve().parents[2]
D = B / 'delivery/all'
hash_checks = 0


def read(path):
    return json.loads(Path(path).read_text())


def verify(binding):
    global hash_checks
    path = Path(binding['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == binding['sha256'], str(path)
    hash_checks += 1


def unique(rows, key):
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows), 'DUPLICATE:' + key
    return result


def verify_retained_corrections():
    for proof_name in ['READ_DIRECTORY_RELOCATION_V1.json', 'READ_DIRECTORY_RELOCATION_V2.json']:
        proof = read(B / 'evidence' / proof_name)
        for binding in proof['bindings']:
            verify(binding)
        relocation = proof.get('relocation', proof.get('relocations', [None])[0])
        original, archived = relocation['original_binding'], relocation['archived_binding']
        verify(archived)
        assert original['sha256'] == archived['sha256']
        old, current = read(archived['path']), read(original['path'])
        expected = deepcopy(old)
        if proof_name.endswith('V1.json'):
            assert 'new_blocking_findings' not in expected['baseline_constraint_review']
            expected['baseline_constraint_review']['new_blocking_findings'] = []
        else:
            actual_edits = []
            baseline_refs = {turn['turn_ref'] for turn in read(proof['baseline_binding']['path'])['turns']}
            for i, row in enumerate(expected['baseline_constraint_review']['constraint_results']):
                for j, ref in enumerate(row['evidence_turn_refs']):
                    prefix = proof['root_id'] + ':'
                    if prefix + ref in baseline_refs:
                        row['evidence_turn_refs'][j] = prefix + ref
                        actual_edits.append({'constraint_result_index': i, 'evidence_ref_index': j,
                                             'original': ref, 'qualified': prefix + ref})
            assert actual_edits == proof['reference_qualifications']
        assert expected == current
    repaired = 0
    for receipt_path in (B / 'units').glob('*/evidence/reads/**/SAVE_RECEIPT.json'):
        receipt = read(receipt_path)
        if 'only_added_field' not in receipt and 'only_added_fields' not in receipt:
            continue
        old_binding = receipt['original_raw_judgment_binding']
        new_binding = receipt['corrected_archive_binding']
        verify(old_binding)
        verify(new_binding)
        verify(receipt['original_save_failure_binding'])
        expected = read(old_binding['path'])
        corrected = read(new_binding['path'])
        if 'only_added_field' in receipt:
            scope, field = receipt['only_added_field'].split('.')
            assert field not in expected[scope]
            expected[scope][field] = receipt['added_value']
        else:
            assert receipt['schema_version'] == 'missing_public_schema_declarations_resume_v2'
            assert receipt['root_id'] == 'cc-train-episode-156136'
            assert 'final_public_questions_actually_reread' not in expected['read_attestation']
            expected['read_attestation']['final_public_questions_actually_reread'] = True
            expected['questions'] = unique(expected['questions'], 'candidate_id')
            for question in expected['questions'].values():
                if 'options' in question:
                    for field in ['fact_distinct', 'no_entangled_options']:
                        if field not in question:
                            question[field] = True
        assert expected == corrected, str(receipt_path)
        repaired += 1
    return repaired


def verify_scene_chains():
    manifest = read(B / 'MANIFEST.json')
    roles = unique(read(B / 'ROLE_ASSIGNMENTS.json')['roots'], 'root_id')
    frozen_pool = read(B / 'evidence/SOURCE_POOL_FREEZE.json')
    consumed = set(frozen_pool['consumed_source_ids'])
    reserved_ids = []
    counts = Counter()
    naturalness = Counter()
    for root in manifest['roots']:
        if root['action'] not in ['SCREEN', 'INHERIT_SCREEN']:
            continue
        root_id = root['root_id']
        unit = Path(root['unit_path'])
        folder = unit / 'roots' / unit.name
        baseline = read(root['baseline_path'])
        baseline_speakers = {turn['speaker'] for turn in baseline['turns']}
        previous_path = Path(root['baseline_path'])
        previous_turns = baseline['turns']
        reservations_path = unit / 'SOURCE_RESERVATIONS.json'
        allocation = read(reservations_path)['allocation'] if reservations_path.exists() else []
        if allocation:
            assert {row['scene_ref'] for row in allocation} == {'R02:B:S01', 'R02:B:S02', 'R02:B:S03'}
            assert len(allocation) == 3
            for row in allocation:
                assert row['status'] == 'PERMANENTLY_RESERVED_BEFORE_AUTHORING'
                assert row['source_id'] not in consumed
                verify({'path': row['source_item_path'], 'sha256': row['source_item_sha256']})
                reserved_ids.append(row['source_id'])
        by_scene = {row['scene_ref']: row for row in allocation}
        for route, index in [('A', 1), ('A', 2), ('A', 3), ('B', 1), ('B', 2), ('B', 3)]:
            scene = f'R02:{route}:S0{index}'
            sfolder = folder / 'scenes' / f'{route}_S0{index}'
            source_path = sfolder / 'CUMULATIVE_SOURCE.json'
            if not source_path.exists():
                continue
            source = read(source_path)
            draft_path = sfolder / 'EXPANSION_DRAFT.json'
            draft = read(draft_path)
            payload = read(sfolder / 'AUTHOR_PAYLOAD.json')
            assert source['root_id'] == draft['root_id'] == root_id
            assert source['scene_ref'] == draft['scene_ref'] == scene
            assert source['turns'][:len(previous_turns)] == previous_turns
            new_turns = source['turns'][len(previous_turns):]
            assert 12 <= len(new_turns) <= 16
            assert draft['turns'] == new_turns
            assert [{'speaker': turn['speaker'], 'text': turn['text']} for turn in new_turns] == payload['turns']
            assert all(turn['speaker'] in baseline_speakers and turn['scene_ref'] == scene for turn in new_turns)
            assert len({turn['turn_ref'] for turn in source['turns']}) == len(source['turns'])
            previous_sha = hashlib.sha256(previous_path.read_bytes()).hexdigest()
            assert source['input_source_sha256'] == draft['input_source_sha256'] == previous_sha
            assert source['draft_sha256'] == hashlib.sha256(draft_path.read_bytes()).hexdigest()
            assert draft['author'] == roles[root_id]['author']
            if route == 'B':
                assert draft['source_id'] == by_scene[scene]['source_id']
            else:
                assert draft['source_id'] is None
            review_path = sfolder / 'SOURCE_REVIEW.json'
            if review_path.exists():
                review = read(review_path)
                assert review['reviewer'] == roles[root_id]['reviewer'] != draft['author']
                assert review['source_sha256'] == hashlib.sha256(source_path.read_bytes()).hexdigest()
                assert review['turn_refs_read'] == [turn['turn_ref'] for turn in source['turns']]
                assert review['full_source_read'] is True
                naturalness[review['checks']['naturalness']['status']] += 1
                counts['reviews'] += 1
            counts['materialized_scenes'] += 1
            previous_path, previous_turns = source_path, source['turns']
    assert len(reserved_ids) == len(set(reserved_ids))
    return {**dict(counts), 'permanent_native_event_reservations': len(reserved_ids),
            'all_new_review_naturalness': dict(naturalness)}


def metrics_for(pool, scope, side):
    run = scope / 'run' / side
    manifest = read(run / 'manifest.json')
    requests = unique(manifest['requests'], 'request_id')
    assert len(requests) == manifest['request_count']
    gold = read(run / 'gold.json')
    results = read(run / 'RESULTS.json')
    result_by_id = unique(results['questions'], 'candidate_id')
    assert set(pool) == set(gold) == set(result_by_id)
    source_texts = {}
    for question in pool.values():
        if question['source_path'] not in source_texts:
            source = read(question['source_path'])
            source_texts[question['source_path']] = '\n'.join(
                '[' + turn['turn_ref'] + '] ' + turn['speaker'] + ': ' + turn['text'] for turn in source['turns'])
    for request in requests.values():
        question = pool[request['candidate_id']]
        field = 'options' if question['question_type'] == 'multi_select' else 'items'
        public_by_text = {text: label for label, text in question[field].items()}
        assert len(public_by_text) == len(question[field])
        variant = request['variant']
        content_to_text = variant['content_to_text']
        canonical_by_public = {public_by_text[text]: cid for cid, text in content_to_text.items()}
        assert set(canonical_by_public) == set(question[field])
        assert gold[request['candidate_id']] == [canonical_by_public[label] for label in question['gold']]
        display_field = 'options' if question['question_type'] == 'multi_select' else 'items'
        assert variant[display_field] == {label: content_to_text[cid] for label, cid in variant['display_to_content'].items()}
    buckets = defaultdict(list)
    scores_by_q = defaultdict(lambda: defaultdict(list))
    exact_by_q = defaultdict(list)
    seen = set()
    for path in sorted((run / 'responses').glob('*.json')):
        response = read(path)
        request_id = response['request_id']
        assert request_id not in seen
        seen.add(request_id)
        request = requests[request_id]
        candidate_id = request['candidate_id']
        question = pool[candidate_id]
        assert response['evaluation_valid'] is True
        for key in ['candidate_id', 'root_id', 'question_type', 'seed', 'slot']:
            assert response[key] == request[key]
        assert response['variant_id'] == request['variant']['variant_id']
        match = re.fullmatch(r'\s*<box>\s*([^<>]*)\s*</box>\s*', response['raw_output'])
        assert match, str(path)
        labels = [s.strip().upper() for s in match.group(1).split(',') if s.strip()]
        assert len(labels) == len(set(labels))
        mapping = request['variant']['display_to_content']
        predicted = [mapping[label] for label in labels]
        expected = gold[candidate_id]
        keys = [question['question_type'], 'root:' + question['root_id']]
        item = {'valid': True}
        if question['question_type'] == 'multi_select':
            predicted_set, expected_set = set(predicted), set(expected)
            fp, fn = len(predicted_set - expected_set), len(expected_set - predicted_set)
            score = Fraction(len(predicted_set & expected_set), len(expected_set))
            if len(expected_set) < len(mapping):
                score -= Fraction(fp, len(mapping) - len(expected_set))
            exact = predicted_set == expected_set
            count = len(expected_set)
            keys.append('gold' + str(count))
            if 2 <= count <= 5:
                keys.append('multi_select_gold2_5')
            if question.get('role') == 'control6':
                keys.append('gold6_control')
            item.update(fp=fp, fn=fn, predicted_size=len(predicted_set), gold_size=count)
        else:
            assert len(predicted) == 6 and set(predicted) == set(expected)
            position = {label: index for index, label in enumerate(predicted)}
            wrong = sum(position[first] > position[second]
                        for first, second in itertools.combinations(expected, 2))
            score = Fraction(15 - 2 * wrong, 15)
            exact = predicted == expected
        assert score == Fraction(response['score']['partial_score_fraction'])
        assert exact == response['score']['exact_match']
        item.update(partial=score, exact=exact)
        for key in keys:
            buckets[key].append(item)
        scores_by_q[candidate_id][response['variant_id']].append(score)
        exact_by_q[candidate_id].append(exact)
        prompt = run / 'prompts' / (request_id + '.txt')
        assert hashlib.sha256(prompt.read_bytes()).hexdigest() == request['prompt_sha256']
        prompt_text = prompt.read_text()
        assert 'Dialogue:\n' + source_texts[question['source_path']] + '\n\nQuestion (' in prompt_text
        assert question['question'] in prompt_text
        assert all(text in prompt_text for text in request['variant'][
            'options' if question['question_type'] == 'multi_select' else 'items'].values())
    assert seen == set(requests)
    for candidate_id, row in result_by_id.items():
        groups = scores_by_q[candidate_id]
        multi = pool[candidate_id]['question_type'] == 'multi_select'
        assert len(groups) == (6 if multi else 12)
        assert all(len(values) == (3 if multi else 1) for values in groups.values())
        mean = sum((sum(values, Fraction()) / len(values) for values in groups.values()), Fraction()) / len(groups)
        assert mean == Fraction(row['full_partial_mean'])
        assert Fraction(sum(exact_by_q[candidate_id]), len(exact_by_q[candidate_id])) == Fraction(row['full_exact_rate'])
        assert row['selected_by_policy'] == (0 < mean < 1)
        assert not row['data_errors']
    assert not results['call_set_errors']
    assert set(results['selected_questions']) == {cid for cid, row in result_by_id.items() if row['selected_by_policy']}
    summary = {}
    for key, rows in buckets.items():
        item = {'calls': len(rows), 'valid_calls': len(rows), 'invalid_calls': 0,
                'correct': sum(row['exact'] for row in rows),
                'exact_rate': sum(row['exact'] for row in rows) / len(rows),
                'partial_mean': str(sum((row['partial'] for row in rows), Fraction()) / len(rows))}
        multi_rows = [row for row in rows if 'fp' in row]
        if multi_rows:
            item.update(false_positive_total=sum(row['fp'] for row in multi_rows),
                        false_negative_total=sum(row['fn'] for row in multi_rows),
                        calls_with_false_positive=sum(row['fp'] > 0 for row in multi_rows),
                        calls_with_false_negative=sum(row['fn'] > 0 for row in multi_rows),
                        selected_all_six_calls=sum(row['predicted_size'] == 6 for row in multi_rows),
                        over_count_calls=sum(row['predicted_size'] > row['gold_size'] for row in multi_rows),
                        under_count_calls=sum(row['predicted_size'] < row['gold_size'] for row in multi_rows),
                        mean_predicted_size=statistics.mean(row['predicted_size'] for row in multi_rows))
        summary[key] = item
    return summary, result_by_id, requests


def main():
    retained_corrections = verify_retained_corrections()
    scene_chain_checks = verify_scene_chains()
    manifest = read(B / 'MANIFEST.json')
    frozen = unique(manifest['roots'], 'root_id')
    assert len(frozen) == 68
    split_roots = {}
    for split, name in [('train', 'training_manifest'), ('dev', 'dev_manifest'), ('test', 'test_manifest')]:
        binding = manifest[name]
        verify(binding)
        split_roots[split] = {json.loads(line)['root_id'] for line in Path(binding['path']).read_text().splitlines() if line.strip()}
    assert set(frozen) == split_roots['train']
    assert all(not (split_roots[left] & split_roots[right])
               for left, right in [('train', 'dev'), ('train', 'test'), ('dev', 'test')])
    final = read(D / 'FINAL_STATUS.json')
    roots = unique(final['roots'], 'root_id')
    assert set(roots) == set(frozen)
    assert final['population_denominator'] == final['terminal_roots'] == 68
    assert final['status'] == 'FULL_FIXED_POPULATION_PROCESSED'
    for flag in ['training_started', 'training_pool_merged', 'external_semantic_certification', 'unseen_holdout']:
        assert final[flag] is False
    for record in ['MASTER_FREEZE.json', 'evidence/HISTORICAL_PROTECTION.json']:
        for binding in read(B / record)['files']:
            verify(binding)
    audit = read(B / 'evidence/ALL_CONTENT_AUDIT.json')
    assert audit['status'] == 'PASS' and not audit['errors'] and audit['root_denominator'] == 53
    for binding in audit['input_bindings']:
        verify(binding)
    content_doc = read(D / 'CONTENT_QUESTIONS.json')
    content = unique(content_doc['questions'], 'candidate_id')
    selected = unique(read(D / 'SELECTED_CANDIDATES.json')['questions'], 'candidate_id')
    excluded = read(D / 'EXCLUDED_CANDIDATES.json')
    assert set(selected) <= set(content)
    assert all(question == content[cid] for cid, question in selected.items())
    assert all(q.get('role') != 'control6' and not (q['question_type'] == 'multi_select' and len(q['gold']) == 6)
               for q in selected.values())
    overlay = read(B.parent / 'h4_codex_wording_minfix_v114/RECOMMENDED_CANDIDATES.json')
    assert set(excluded['historical_quarantine']) == set(overlay['quarantined_original_ids'])
    assert set(content).isdisjoint(overlay['quarantined_original_ids'])
    assert final['statuses'] == dict(Counter(row['status'] for row in roots.values()))
    for questions, prefix in [(content, 'content'), (selected, 'selected')]:
        assert len(questions) == final[prefix + ('_questions' if prefix == 'content' else '_candidates')]
        assert dict(Counter(q['question_type'] for q in questions.values())) == final[prefix + '_type_counts']
        assert dict(Counter(str(len(q['gold'])) for q in questions.values() if q['question_type'] == 'multi_select')) == final[prefix + '_gold_counts']
    expected_content_exclusions = []
    materialized_scene_total = 0
    for root_id, root in roots.items():
        origin = frozen[root_id]
        unit = Path(origin['unit_path'])
        folder = unit / 'roots' / unit.name
        root_content = {cid: q for cid, q in content.items() if q['root_id'] == root_id}
        root_selected = {cid: q for cid, q in selected.items() if q['root_id'] == root_id}
        assert root['action'] == origin['action']
        assert root['terminal'] is True
        materialized_scene_total += len(list(folder.glob('scenes/*/CUMULATIVE_SOURCE.json')))
        if origin['action'] == 'REUSE':
            derived_status = 'REUSED_COMPLETE'
        elif origin['action'] == 'HISTORICAL_HOLD':
            derived_status = 'HISTORICAL_BASELINE_HOLD'
        elif read(unit / 'inputs/CONTINUATION_CONSTRAINTS.json')['decision'] == 'HOLD':
            derived_status = 'NEW_BASELINE_HOLD'
        elif (unit / 'MATCHING_HOLD.json').exists():
            derived_status = 'MATCHING_HOLD'
        else:
            terminal_values = {'STRUCTURAL_FAILURE', 'INTEGRITY_FAILURE', 'REVIEW_INVALID', 'REVIEW_HOLD'}
            root_receipts = [read(p) for p in folder.glob('receipts/*.json')]
            scene_receipts = [read(p) for p in folder.glob('scenes/*/receipts/*.json')]
            scene_reviews = [read(p) for p in folder.glob('scenes/*/SOURCE_REVIEW.json')]
            stopped = any(r['status'] in terminal_values for r in root_receipts + scene_receipts)
            stopped = stopped or any(r['decision'] == 'HOLD' for r in scene_reviews)
            if stopped:
                derived_status = 'EXPANSION_STOPPED'
            else:
                assert len(scene_reviews) == 6 and all(r['decision'] == 'READY' for r in scene_reviews)
                assert {r['scene_ref'] for r in scene_reviews} == {f'R02:{route}:S0{index}' for route in 'AB' for index in [1, 2, 3]}
                if (folder / 'QA_AUTHOR_FAILURE.json').exists():
                    derived_status = 'QA_FIRST_ATTEMPT_FAILED'
                else:
                    assert (folder / 'FINAL_CROSSCHECK.json').is_file()
                    derived_status = 'NEW_CONTENT_COMPLETED'
        assert root['status'] == derived_status, root_id
        assert root['content_questions'] == len(root_content)
        assert root['selected_candidates'] == len(root_selected)
        if origin['action'] == 'REUSE':
            old = Path(origin['reuse_batch'])
            for binding in origin['reuse_bindings']:
                verify(binding)
            historical = {q['candidate_id']: q for q in read(old / 'evaluation/EVALUATION_POOL.json')['questions'] if q['root_id'] == root_id}
            historical_selected = {q['candidate_id']: q for q in read(old / 'SELECTED_CANDIDATES.json')['questions'] if q['root_id'] == root_id}
            if old.name.endswith('_v113'):
                historical_selected = {q['candidate_id']: q for q in overlay['questions'] if q['root_id'] == root_id}
                historical = {cid: historical_selected.get(cid, q) for cid, q in historical.items() if cid not in overlay['quarantined_original_ids']}
                historical.update(historical_selected)
            assert root_content == historical
            assert root_selected == historical_selected
        elif root['status'] != 'NEW_CONTENT_COMPLETED':
            assert not root_content and not root_selected
        else:
            crosscheck = read(folder / 'FINAL_CROSSCHECK.json')
            assert set(root_content) == {q['candidate_id'] for q in crosscheck['results'] if q['status'] == 'CONTENT_PASS'}
            expected_content_exclusions.extend({'root_id': root_id, 'phase': 'content', **q}
                                               for q in crosscheck['results'] if q['status'] != 'CONTENT_PASS')
        if root['complete_dialogue']:
            verify(root['source_binding'])
            source_path = Path(root['source_binding']['path'])
            source = read(source_path)
            baseline = read(origin['baseline_path'])['turns']
            assert source['turns'][:len(baseline)] == baseline
            assert source['root_id'] == root_id
            scene_counts = Counter(t.get('scene_ref') for t in source['turns'][len(baseline):])
            assert len(scene_counts) == 6 and all(12 <= value <= 16 for value in scene_counts.values())
            lines = ['# ' + root_id, '', 'Source: ' + str(source_path), 'SHA256: ' + root['source_binding']['sha256'], '']
            lines.extend('[' + t['turn_ref'] + '] ' + t['speaker'] + ': ' + t['text'] for t in source['turns'])
            assert Path(root['readable_dialogue_path']).read_text() == '\n\n'.join(lines) + '\n'
            qlines = ['# 题目、答案及来源', '', '内容通过题全部展示；入选状态见候选文件。', '']
            for question in root_content.values():
                qlines.extend(['## ' + question['candidate_id'], '', question['question'], ''])
                field = 'options' if question['question_type'] == 'multi_select' else 'items'
                qlines.extend(label + ': ' + text for label, text in question[field].items())
                qlines.extend(['', 'Gold: ' + ', '.join(question['gold']),
                               '入选候选: ' + str(question['candidate_id'] in selected),
                               '来源: ' + question['source_path'], ''])
            assert Path(root['readable_questions_path']).read_text() == '\n'.join(qlines)
    assert excluded['content_excluded'] == expected_content_exclusions
    for q in content.values():
        verify({'path': q['source_path'], 'sha256': q['source_sha256']})
        source = read(q['source_path'])
        assert source['root_id'] == q['root_id']
        turns = unique(source['turns'], 'turn_ref')
        evidence = q['option_evidence'] if q['question_type'] == 'multi_select' else q['item_evidence']
        for entry in evidence.values():
            for cite in [entry, *entry.get('citations', []), *entry.get('alternative_matching_anchors', [])]:
                turn = turns[cite['turn_ref']]
                assert cite['quote'] and cite['quote'] in turn['text']
                if 'speaker' in cite:
                    assert cite['speaker'] == turn['speaker']
    calls = {}
    for wave in ['validation', 'production']:
        scope = B / 'evaluations' / wave
        pool = unique(read(scope / 'EVALUATION_POOL.json')['questions'], 'candidate_id')
        assert all(q == content[cid] for cid, q in pool.items())
        paired = read(scope / 'ADAPTER_BINDINGS.json')
        assert len(pool) == paired['candidate_count']
        assert paired['calls_per_model'] == sum(18 if q['question_type'] == 'multi_select' else 12 for q in pool.values())
        raw_audit = read(scope / 'INDEPENDENT_SOLVER_AUDIT.json')
        assert raw_audit['status'] == 'PASS' and not raw_audit['errors']
        for binding in raw_audit['input_bindings']:
            verify(binding)
        side_results = {}
        side_requests = {}
        for side in ['base', 'trained']:
            metrics, results, requests = metrics_for(pool, scope, side)
            assert final['solver'][wave]['by_side'][side] == metrics
            assert len(requests) == paired['calls_per_model']
            assert read(scope / (side.upper() + '_COMPLETE.json'))['actual_calls'] == len(requests)
            side_results[side] = results
            side_requests[side] = requests
        assert side_requests['base'] == side_requests['trained']
        for cid, q in pool.items():
            assert (cid in selected) == (side_results['trained'][cid]['selected_by_policy'] and q.get('role') != 'control6')
        calls[wave] = paired['calls_per_model']
    attempts = read(D / 'ALL_ATTEMPTS_INDEX.json')
    attempt_roots = unique(attempts['roots'], 'root_id')
    assert set(attempt_roots) == set(frozen)
    assert attempts['statuses'] == final['statuses']
    assert attempts['new_materialized_scenes'] == materialized_scene_total
    for root_id, row in attempt_roots.items():
        assert row['status'] == roots[root_id]['status']
        unit = Path(frozen[root_id]['unit_path'])
        assert {binding['path'] for binding in row['stage_artifacts']} == {str(path.resolve()) for path in unit.rglob('*.json')}
        for binding in row['stage_artifacts']:
            verify(binding)
    assert attempts['source_only_complete_qa_hold_roots'] == ['cc-train-episode-103866']
    source_only = attempt_roots['cc-train-episode-103866']
    assert source_only['source_only_complete_qa_hold'] is True
    verify(source_only['source_only_readable_dialogue_binding'])
    assert not any(q['root_id'] == 'cc-train-episode-103866' for q in content.values())
    ledger = read(D / 'QUESTION_STAGE_LEDGER.json')
    ledger_roots = unique(ledger['roots'], 'root_id')
    assert set(ledger_roots) == set(frozen)
    unfilled = []
    for root_id, row in ledger_roots.items():
        unit = Path(frozen[root_id]['unit_path'])
        folder = unit / 'roots' / unit.name
        composition = folder / 'COMPOSITION_RECEIPT.json'
        crosscheck = folder / 'FINAL_CROSSCHECK.json'
        assert row['unfilled_composition_slots'] == (read(composition)['unfilled_slots'] if composition.exists() else [])
        assert row['final_crosscheck_results'] == (read(crosscheck)['results'] if crosscheck.exists() else [])
        unfilled.extend(row['unfilled_composition_slots'])
        for binding in row['bindings']:
            verify(binding)
    assert len(unfilled) == ledger['unfilled_composition_slots']
    assert dict(Counter(s['question_type'] for s in unfilled)) == ledger['unfilled_slot_types']
    assert set(content).isdisjoint(s['candidate_id'] for s in unfilled if 'candidate_id' in s)
    ms_missing = ledger_roots['cc-train-episode-162896']['unfilled_composition_slots']
    assert len(ms_missing) == 5 and {s['intended_cardinality'] for s in ms_missing} == {2, 3, 4, 5, 6}
    assert all(s['question_type'] == 'multi_select' and s['reason'] == 'truth_not_supported' for s in ms_missing)
    for root_id, expected in [('cc-train-episode-19053', 2), ('cc-train-episode-104814', 1)]:
        assert sum(s['question_type'] == 'ordering' for s in ledger_roots[root_id]['unfilled_composition_slots']) == expected
    assert read(B / 'units/root191749/evidence/SELECTION_REVIEW_EXECUTION_HOLD.json')['source_semantic_failure'] is False
    deviations = read(D / 'PROCEDURE_DEVIATIONS.json')
    for binding in deviations['input_bindings']:
        verify(binding)
    for row in deviations['records']:
        verify(row['record_binding'])
        assert row['retained_record'] == read(row['record_binding']['path'])
    report = (D / 'REPORT_ZH.md').read_text()
    execution_report = (D / 'EXECUTION_REPORT_ZH.md').read_text()
    for root_id, root in roots.items():
        route = root.get('ready_by_route', {'A': 3 if root['action'] == 'REUSE' else 0,
                                            'B': 3 if root['action'] == 'REUSE' else 0})
        expected_line = (f"|{root_id}|{root['action']}|{root['status']}|{route['A']}/3，{route['B']}/3|"
                         f"{root['content_questions']}|{root['selected_candidates']}|")
        assert expected_line in report
        attempt = attempt_roots[root_id]
        phases = ', '.join(sorted({failure['phase'] for failure in attempt['failure_details']}))
        assert (f"|{root_id}|{attempt['status']}|{attempt['materialized_scenes']}|"
                f"{attempt['historical_ready_reviews_retained']}|{phases}|") in execution_report
    guide = read(D / 'DELIVERY_GUIDE_FACTS.json')
    for binding in guide['input_bindings']:
        verify(binding)
    verify(guide['generated_report_binding'])
    assert guide['population'] == guide['terminal_roots'] == 68
    assert guide['statuses'] == final['statuses']
    assert guide['new_content_success_roots'] == sum(root['status'] == 'NEW_CONTENT_COMPLETED' for root in roots.values())
    assert guide['main_complete_dialogues'] == final['complete_dialogues']
    assert guide['main_new_complete_dialogues'] == final['new_complete_dialogues']
    assert guide['readable_complete_dialogues_including_source_only'] == final['complete_dialogues'] + 1
    assert guide['source_only_qa_procedure_hold_roots'] == ['cc-train-episode-103866']
    for cohort, actions in [('new', {'SCREEN', 'INHERIT_SCREEN'}), ('reused', {'REUSE'}),
                            ('all', {'SCREEN', 'INHERIT_SCREEN', 'REUSE'})]:
        root_ids = {root_id for root_id, root in roots.items() if root['action'] in actions}
        assert guide['cohorts'][cohort]['roots'] == len(root_ids)
        for name, questions in [('content', content), ('selected', selected)]:
            rows = [question for question in questions.values() if question['root_id'] in root_ids]
            distribution = {'questions': len(rows), 'type_counts': dict(Counter(q['question_type'] for q in rows)),
                            'gold_counts': dict(Counter(str(len(q['gold'])) for q in rows if q['question_type'] == 'multi_select'))}
            assert guide['cohorts'][cohort][name] == distribution
    for name, content_only in [('new_content_success_scenes', True), ('all_new_attempt_reviewed_scenes', False)]:
        status_counts, decisions = Counter(), Counter()
        reviewed_roots = set()
        for root_id, root in roots.items():
            if root['action'] not in {'SCREEN', 'INHERIT_SCREEN'}:
                continue
            if content_only and root['status'] != 'NEW_CONTENT_COMPLETED':
                continue
            unit = Path(frozen[root_id]['unit_path'])
            for path in unit.glob('roots/*/scenes/*/SOURCE_REVIEW.json'):
                review = read(path)
                status_counts[review['checks']['naturalness']['status']] += 1
                decisions[review['decision']] += 1
                reviewed_roots.add(root_id)
        assert guide['naturalness_existing_labels'][name] == {
            'reviewed_roots': len(reviewed_roots), 'reviews': sum(status_counts.values()),
            'status_counts': {key: status_counts[key] for key in ['PASS', 'WARNING', 'PASS_WITH_WARNING', 'FAIL']},
            'decision_counts': dict(decisions)}
    for wave in ['validation', 'production']:
        assert guide['solver_new_content_only']['waves'][wave]['calls_per_model'] == calls[wave]
        for side in ['base', 'trained']:
            expected_metrics = {key: value for key, value in final['solver'][wave]['by_side'][side].items() if not key.startswith('root:')}
            assert guide['solver_new_content_only']['waves'][wave]['by_side'][side]['by_type_and_gold'] == expected_metrics
    for side in ['base', 'trained']:
        for key in ['multi_select_gold2_5', 'ordering', 'gold6_control']:
            values = [final['solver'][wave]['by_side'][side][key] for wave in ['validation', 'production']]
            correct, total_calls = sum(row['correct'] for row in values), sum(row['calls'] for row in values)
            assert guide['solver_new_content_only']['overall'][side][key] == {
                'correct': correct, 'calls': total_calls, 'exact_rate': correct / total_calls}
    return {'status': 'PASS', 'reviewer': '/root/v118_final_delivery_review', 'errors': [],
            'population': len(roots), 'statuses': final['statuses'],
            'content_questions': len(content), 'selected_candidates': len(selected),
            'complete_dialogues': final['complete_dialogues'],
            'source_only_qa_hold_roots': attempts['source_only_complete_qa_hold_roots'],
            'unfilled_composition_slots': len(unfilled), 'paired_calls_per_side': calls,
            'file_hash_checks': hash_checks, 'external_semantic_certification': False,
            'additive_review_corrections_checked': retained_corrections,
            'scene_chain_checks': scene_chain_checks,
            'scope': 'Independent mechanical reconciliation and raw-answer metric recomputation; no source-semantic certification or model calls.'}


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
