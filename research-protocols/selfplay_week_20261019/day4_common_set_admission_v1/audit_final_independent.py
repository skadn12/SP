import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_train_pool_v125/data')
issues = []
checks = {}
cache = {}
verified_bindings = {}


def check(condition, code, detail):
    if not condition:
        issues.append({'code': code, 'detail': detail})


def read(path):
    path = str(Path(path).resolve())
    if path not in cache:
        cache[path] = json.loads(Path(path).read_text())
    return cache[path]


def binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def verify_binding(record):
    path = str(Path(record['path']).resolve())
    if path not in verified_bindings:
        verified_bindings[path] = binding(path)
    actual = verified_bindings[path]
    check(record['sha256'] == actual['sha256'], 'BINDING_SHA_MISMATCH', path)
    size = record.get('bytes', record.get('size_bytes'))
    if size is not None:
        check(size == actual['bytes'], 'BINDING_SIZE_MISMATCH', path)


def pointer(obj, value):
    for token in value.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        obj = obj[int(token)] if isinstance(obj, list) else obj[token]
    return obj


def walk_bindings(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get('path'), str) and isinstance(obj.get('sha256'), str):
            verify_binding(obj)
        for value in obj.values():
            walk_bindings(value)
    elif isinstance(obj, list):
        for value in obj:
            walk_bindings(value)


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


ledger_path = BASE / 'ADMISSION_LEDGER.json'
common_path = BASE / 'COMMON_ELIGIBLE_BASE_SET.json'
initial_ledger = binding(ledger_path)
initial_common = binding(common_path)
ledger = read(ledger_path)
common = read(common_path)
old = read(BASE.parent / 'day3_mainline_readiness_v1/training/COMMON_OLD_BASE_SET.json')
inventory = read(BASE / 'PACKET_INVENTORY.json')
cross = read(BASE / 'review/CROSSROOT_REVIEW.json')
questions = {r['candidate_id']: r for r in read(DATA / 'COMBINED_QUESTIONS.json')['questions']}
training = {r['candidate_id']: r for r in (json.loads(line) for line in (DATA / 'TRAIN_NEW_ONLY.jsonl').read_text().splitlines())}
lineage = {r['candidate_id']: r for r in (json.loads(line) for line in (DATA / 'QUESTION_LINEAGE.jsonl').read_text().splitlines())}
groups = {cid: g['group_id'] for g in read(DATA / 'DEPENDENCE_GROUPS.json')['groups'] for cid in g['candidate_ids']}
expected = set(old['foundation']['candidate_ids'])
records = ledger['records']
by_id = {r['candidate_id']: r for r in records}
check(len(records) == len(by_id) == 363 and set(by_id) == expected, 'FOUNDATION_COVERAGE', 'Exactly the fixed 363 candidate IDs must occur once.')
check(set(questions) == set(training) == set(lineage) == expected, 'FROZEN_ID_JOIN', 'Question, training and lineage candidate ID sets must equal the fixed foundation.')
check(Counter(r['question_type'] for r in records) == {'multi_select': 238, 'ordering': 125}, 'QUESTION_TYPE_COUNTS', 'Expected 238 multi_select and 125 ordering.')
check(len({r['root_id'] for r in records}) == 68, 'ROOT_COUNT', 'Expected 68 foundation roots.')
walk_bindings(ledger)
walk_bindings(common)
for input_binding in ledger['input_bindings']:
    if Path(input_binding['path']).suffix == '.json':
        walk_bindings(read(input_binding['path']))

packets = {entry['root_id']: read(entry['packet_binding']['path']) for entry in inventory['packets']}
packet_base = {b['candidate_id']: (b, packet) for packet in packets.values() for b in packet['base_candidates']}
check(len(packet_base) == 102, 'PACKET_BASE_COVERAGE', 'Twenty overlap-root packets must cover 102 unique base candidates.')
same_count = same_pairs = cross_pairs = searched = unretrieved = quote_checks = preserved_links = 0
observed_cross_pairs = set()
confirmed_pairs = set()
for r in records:
    cid = r['candidate_id']
    q, t, lin = questions[cid], training[cid], lineage[cid]
    check(r['root_id'] == q['root_id'] == t['root_id'] == lin['root_id'], 'ROOT_ALIGNMENT', cid)
    check(r['question_type'] == q['question_type'] == t['question_type'] == lin['question_type'], 'TYPE_ALIGNMENT', cid)
    check(r['dependence_group_id'] == t['dependence_group_id'] == lin['dependence_group_id'] == groups[cid], 'DEPENDENCE_GROUP_LOSS', cid)
    check(r['source_binding'] == lin['source_binding'], 'SOURCE_BINDING_LOSS', cid)
    check(r['original_gold'] == q['gold'], 'GOLD_ALIGNMENT', cid)
    hashes = r['input_record_hashes']
    for key, obj in [('question_sha256', q), ('training_record_sha256', t), ('lineage_record_sha256', lin)]:
        check(hashes[key] == canonical_hash(obj), 'RECORD_HASH_ALIGNMENT', cid + ':' + key)
    check(r['decision'] == 'KEEP', 'UNEXPECTED_EXCLUSION', cid)
    check(r['final_training_selected_A'] is False and r['final_training_selected_B'] is False, 'TRAINING_SELECTION', cid)
    check(r['model_scores_used_for_admission'] is False, 'SCORE_SELECTION', cid)
    source = read(r['source_binding']['path'])
    turns = {turn['turn_ref']: turn for turn in source['turns']}
    evidence = q.get('option_evidence', q.get('item_evidence', {}))
    for label, e in evidence.items():
        citations = e.get('citations', [])
        if not citations and e.get('turn_ref') and e.get('quote'):
            citations = [e]
        for c in citations:
            if not c.get('turn_ref') or not c.get('quote'):
                continue
            turn = turns.get(c['turn_ref'])
            check(turn is not None and c['quote'] in turn['text'], 'SOURCE_QUOTE_ALIGNMENT', cid + '#' + label + ':' + c['turn_ref'])
            if turn is not None and c.get('speaker'):
                check(c['speaker'] == turn['speaker'], 'SOURCE_SPEAKER_ALIGNMENT', cid + '#' + label)
            quote_checks += 1
    same = r['same_root_review']
    check(same['applicable'] == (cid in packet_base), 'SAME_ROOT_SCOPE_ALIGNMENT', cid)
    if cid in packet_base:
        same_count += 1
        packet_b, packet = packet_base[cid]
        new_ids = {n['candidate_id'] for n in packet['new_candidates']}
        check(set(same['compared_new_candidate_ids']) == new_ids, 'SAME_ROOT_NEW_UNION', cid)
        same_pairs += len(same['compared_new_candidate_ids'])
        ref = same['evidence_ref']
        verify_binding(ref['artifact'])
        original = pointer(read(ref['artifact']['path']), ref['json_pointer'])
        check(original['candidate_id'] == cid and original['root_id'] == r['root_id'] and original['question_type'] == r['question_type'], 'SAME_ROOT_REVIEW_POINTER', cid)
        check(original['decision'] == 'KEEP', 'SAME_ROOT_DISPOSITION_COPY', cid)
        old_links = original.get('dependence_links', original.get('local_dependency_links', []))
        check(same['dependency_assessment'] == old_links, 'LOCAL_DEPENDENCE_LINK_LOSS', cid)
        preserved_links += len(old_links) if isinstance(old_links, list) else 1
    else:
        check(not same['compared_new_candidate_ids'] and same['evidence_ref'] is None, 'NO_SAME_ROOT_FALSE_REVIEW', cid)
    cr = r['cross_root_review']
    check(cr['negative_result_is_exhaustive_semantic_proof'] is False, 'CROSSROOT_LIMIT_OVERCLAIM', cid)
    searched += cr['all_different_root_pairs_searched']
    unretrieved += cr['unretrieved_pairs_not_individually_semantically_adjudicated']
    cross_pairs += len(cr['retrieved_comparisons'])
    check(len(cr['retrieved_comparisons']) == cr['retrieved_pairs_reviewed'], 'CROSSROOT_PAIR_COUNT', cid)
    for comparison in cr['retrieved_comparisons']:
        p = pointer(cross, comparison['evidence_json_pointer'])
        key = (cid, comparison['new_candidate_id'])
        observed_cross_pairs.add(key)
        check(p['foundation_candidate_id'] == cid and p['new_candidate_id'] == comparison['new_candidate_id'], 'CROSSROOT_POINTER_ALIGNMENT', str(key))
        for field in ['decision', 'semantic_class', 'dependence_link_required']:
            check(comparison[field] == p[field], 'CROSSROOT_DECISION_COPY', str(key) + ':' + field)
        check(p['foundation_root_id'] != p['new_root_id'], 'CROSSROOT_ROOT_ALIGNMENT', str(key))
        if comparison.get('confirmed_necessary_fact_overlap'):
            confirmed_pairs.add(key)
            check(comparison['dependence_link_required'] and comparison['decision'] == 'PASS' and r['decision'] == 'KEEP', 'PARTIAL_OVERLAP_NOT_RETAINED', str(key))

expected_cross_pairs = {(p['foundation_candidate_id'], p['new_candidate_id']) for p in cross['pairs']}
expected_partial = {(p['foundation_candidate_id'], p['new_candidate_id']) for p in cross['summary']['partial_overlap_pair_ids']}
cross_quote_checks = 0
for key, proof in cross['branch_proof_index'].items():
    source = read(proof['source_binding']['path'])
    turns = {turn['turn_ref']: turn for turn in source['turns']}
    check(key == proof['candidate_id'] + '#' + proof['label'], 'CROSSROOT_BRANCH_KEY_ALIGNMENT', key)
    for citation in proof['necessary_citations']:
        turn = turns.get(citation['turn_ref'])
        check(turn is not None and citation['quote'] in turn['text'], 'CROSSROOT_BRANCH_QUOTE_ALIGNMENT', key + ':' + citation['turn_ref'])
        if turn is not None and citation.get('speaker'):
            check(citation['speaker'] == turn['speaker'], 'CROSSROOT_BRANCH_SPEAKER_ALIGNMENT', key)
        cross_quote_checks += 1
for pair in cross['pairs']:
    for role in ['foundation', 'new']:
        for key in pair[role + '_branch_proofs']:
            check(key in cross['branch_proof_index'], 'CROSSROOT_MISSING_BRANCH_PROOF', str(key))
            if key in cross['branch_proof_index']:
                check(cross['branch_proof_index'][key]['candidate_id'] == pair[role + '_candidate_id'], 'CROSSROOT_BRANCH_CANDIDATE_ALIGNMENT', str(key))
check(same_count == 102 and same_pairs == 898, 'SAME_ROOT_TOTALS', f'{same_count} candidates, {same_pairs} pairs')
check(cross_pairs == len(observed_cross_pairs) == 840 and observed_cross_pairs == expected_cross_pairs, 'CROSSROOT_TOTALS', 'All 840 retrieved pairs must be copied exactly once.')
check(searched == 62990 and unretrieved == 62150, 'CROSSROOT_BOUNDED_COUNTS', f'{searched} searched; {unretrieved} not individually adjudicated')
check(confirmed_pairs == expected_partial and len(confirmed_pairs) == 9, 'CONFIRMED_PARTIAL_OVERLAP_COVERAGE', str(sorted(confirmed_pairs)))
check(cross['summary']['whole_task_duplicate_pairs_confirmed'] == 0, 'WHOLE_TASK_DUPLICATE_THRESHOLD', 'The bound crossroot review confirms zero complete-task duplicate pairs.')
check(common['foundation_eligible_A'] == common['foundation_eligible_B'], 'COMMON_ARMS_DIFFER', 'A and B eligible reserves must have identical IDs and order.')
check(len(common['foundation_eligible_A']) == len(set(common['foundation_eligible_A'])) == 363 and set(common['foundation_eligible_A']) == expected, 'COMMON_RESERVE_COVERAGE', 'A/B reserve must equal fixed363 under all-KEEP review.')
check(common['same_ordered_common_set'] is True, 'COMMON_ORDER_FLAG', 'Expected identical ordered reserve.')
check(not common['excluded_candidate_ids'] and not common['hold_candidate_ids'], 'COMMON_EXCLUDE_HOLD', 'Expected no excluded or held candidates.')
check(not common['final_selected_foundation_A'] and not common['final_selected_foundation_B'] and common['final_training_record_counts'] == {'A': 0, 'B': 0}, 'FINAL_TRAINING_POOL_NONEMPTY', 'Reserve must not become an actual training selection.')
check(common['mixture_frozen'] is False and common['q_final'] is None and common['training_start_go'] is False and ledger['training_start_go'] is False, 'NO_GO_LOSS', 'Mixture/budget remains unfrozen and training remains NO_GO.')
check(common['legacy_replay_candidate_ids'] == [] and common['legacy_replay_selected_count'] == 0, 'LEGACY_REOPENED', 'Legacy113 replay remains unselected.')
check(ledger['review_provenance']['new_solver_calls'] == ledger['review_provenance']['new_model_inference'] == ledger['review_provenance']['training_runs'] == 0, 'EXECUTION_BOUNDARY', 'No solver or training runs allowed.')
check(binding(ledger_path) == initial_ledger and binding(common_path) == initial_common, 'INPUT_CHANGED_DURING_AUDIT', 'Final artifacts changed during independent audit.')

checks.update({'foundation_ids_unique_and_complete': len(by_id), 'foundation_roots': 68,
               'question_types': dict(Counter(r['question_type'] for r in records)),
               'canonical_record_hashes_checked': 3 * len(records), 'source_quote_references_checked': quote_checks,
               'crossroot_branch_quote_references_checked': cross_quote_checks,
               'same_root_candidates': same_count, 'same_root_pairs': same_pairs,
               'same_root_original_review_pointers_resolved': same_count,
               'local_dependency_links_preserved': preserved_links,
               'different_root_pairs_searched': searched, 'retrieved_pairs_resolved': cross_pairs,
               'unretrieved_pairs_explicitly_not_individually_adjudicated': unretrieved,
               'confirmed_partial_crossroot_pairs_retained': len(confirmed_pairs),
               'unique_bound_files_verified': len(verified_bindings),
               'common_A_B_ids_and_order_equal': True, 'eligible_reserve_per_arm': 363,
               'actual_final_training_records': {'A': 0, 'B': 0}, 'mixture_frozen': False,
               'training_start_go': False})
result = {'schema_version': 'day4_independent_final_audit_v1', 'status': 'FAIL' if issues else 'PASS',
          'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'reviewer': 'mrc_final_audit; independent reconstruction from frozen source containers and review pointers',
          'issues': issues, 'admission_ledger_binding': initial_ledger, 'common_set_binding': initial_common,
          'checks': checks,
          'scope_limitations': ['This audit verifies identities, record and source bindings, exact quote alignment, review-pointer alignment, dependency preservation, aggregate coverage and reserve boundaries.',
                                'It is not a fresh source-first factual QA or exhaustive semantic adjudication of all candidate pairs. The crossroot screen remains bounded to the disclosed retrieval procedure.',
                                'No solver inference, scorer, training run, mixture selection, epoch or budget freeze is authorized or performed.'],
          'training_start_go': False}
out = BASE / 'review/INDEPENDENT_FINAL_AUDIT.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'audit': binding(out), 'status': result['status'], 'issue_count': len(issues), 'issues': issues[:15], 'checks': checks}, ensure_ascii=False))
