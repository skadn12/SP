#!/usr/bin/env python3
"""Read-only, independent reconstruction of sealed v111 presentation export."""
import copy
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries')
VIEW = BASE / 'h4_codex_scaled_production_v111_training_view_v1'
OUT = VIEW / 'package'
SOURCE = BASE / 'h4_codex_scaled_production_v111_solver_eval/training_candidate_package'
POOL = BASE / 'h4_codex_scaled_production_v111_solver_eval/EVALUATION_POOL.json'
BATCH = BASE / 'h4_codex_scaled_production_v111'
RECORD = BASE / 'h4_codex_scaled_production_v111_delivery/INDEPENDENT_PRESENTATION_REVIEW.json'
SCRIPT_PREPARED_AT = '2026-10-16T03:40:57.603292+00:00'
EXPECTED_SEED = 'v111_training_presentation_v1'
EXPECTED_TOOL_SHA = '2f5958c5d97b1fb8bcb0029b635bae249e312eff449ad913dcb76f423e03c726'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(b):
    return hashlib.sha256(b).hexdigest()


def digest(p):
    return digest_bytes(p.read_bytes())


def read(p):
    return json.loads(p.read_text())


def check(value, why):
    if not value:
        raise AssertionError(why)


def message_digest(messages):
    return digest_bytes(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode('utf-8'))


def snapshot(d):
    return [{'relative_path':str(p.relative_to(d)), 'sha256':digest(p), 'bytes':p.stat().st_size}
            for p in sorted(d.rglob('*')) if p.is_file()]


def manifest_check(directory):
    p = directory / 'COMPLETE_MANIFEST.json'
    m = read(p)
    listed = set()
    for e in m['files']:
        q = Path(e.get('relative_path', e.get('path')))
        q = q if q.is_absolute() else directory / q
        check(q.is_file(), 'Missing manifest file: '+str(q))
        check(q.resolve().is_relative_to(directory.resolve()), 'Manifest escapes directory')
        check(q not in listed, 'Duplicate manifest file')
        listed.add(q)
        check(digest(q) == e['sha256'], 'Hash mismatch: '+str(q))
        if 'bytes' in e:
            check(q.stat().st_size == e['bytes'], 'Size mismatch: '+str(q))
    actual = {p for p in directory.rglob('*') if p.is_file()}
    check(actual == listed | {p}, 'Manifest file set mismatch: '+str(directory))
    return m, len(listed)


def independent_order(namespace, identity, alphabet):
    # Independent implementation: derive a stream of 256-bit draws, consume until
    # an unbiased bounded integer is available, and exchange decreasing positions.
    seed_bytes = hashlib.sha256((namespace+'\0'+identity).encode('utf-8')).digest()
    current = list(alphabet)
    draw_index = 0
    for last in reversed(range(1, len(current))):
        modulus = last + 1
        threshold = 2**256 - (2**256 % modulus)
        draw = threshold
        while draw >= threshold:
            draw = int(hashlib.sha256(seed_bytes+draw_index.to_bytes(8, byteorder='big')).hexdigest(), 16)
            draw_index += 1
        pick = draw % modulus
        current[last], current[pick] = current[pick], current[last]
    return current, seed_bytes.hex()


def patterns(records, field):
    result = {}
    for kind in sorted({r['question_type'] for r in records}):
        group = [r for r in records if r['question_type'] == kind]
        seqs = Counter(','.join(r[field]) for r in group)
        byposition = defaultdict(Counter)
        for r in group:
            for pos, label in enumerate(r[field], 1):
                byposition[str(pos)][label] += 1
        result[kind] = {
            'question_count': len(group),
            'gold_label_sequence_counts': dict(sorted(seqs.items())),
            'unique_gold_label_sequences': len(seqs),
            'maximum_identical_gold_label_sequence_count': max(seqs.values(), default=0),
            'answer_position_label_counts': {k:dict(sorted(v.items())) for k,v in sorted(byposition.items())},
        }
    return result


def main():
    execution_started = now()
    check(not RECORD.exists(), 'Refuse to overwrite review record')
    before = {str(p):snapshot(p) for p in [SOURCE, VIEW]}
    pool_before = digest(POOL)
    plan = read(BATCH/'PRECALL_PLAN.json')
    pre = read(VIEW/'PREAUTHORING_EXPORT_CONFIG.json')
    freeze = read(OUT/'CONFIG_FREEZE.json')
    bindings = read(OUT/'INPUT_BINDINGS.json')
    audit = read(OUT/'LABEL_PATTERN_AUDIT.json')
    complete, output_files = manifest_check(OUT)
    source_complete, source_files = manifest_check(SOURCE)
    root_sealed_at_review = (VIEW/'COMPLETE_MANIFEST.json').exists()
    root_complete, root_files = manifest_check(VIEW) if root_sealed_at_review else (None, None)
    oldm = read(SOURCE/'FRONTIER_MANIFEST.json')
    newm = read(OUT/'FRONTIER_MANIFEST.json')
    oldrows = [json.loads(x) for x in (SOURCE/'frontier_candidates.jsonl').read_text().splitlines() if x.strip()]
    newrows = [json.loads(x) for x in (OUT/'frontier_candidates.jsonl').read_text().splitlines() if x.strip()]
    pool = read(POOL)
    questions = {q['candidate_id']:q for q in pool['questions']}
    check(len(questions) == len(pool['questions']), 'Duplicate pool IDs')
    maps_obj = read(OUT/'PRESENTATION_MAPPING.json')
    maps = maps_obj['records']
    check(maps_obj['message_hash_serialization'] == "json.dumps(messages, ensure_ascii=False, sort_keys=True).encode('utf-8')", 'Hash serialization note')
    n = oldm['candidate_count']
    check(n == len(oldm['candidates']) == len(newm['candidates']) == len(oldrows) == len(newrows) == len(maps), 'Row counts')
    check(len({c['candidate_id'] for c in oldm['candidates']}) == n, 'Duplicate candidate IDs')
    check([c['candidate_id'] for c in oldm['candidates']] == [c['candidate_id'] for c in newm['candidates']], 'Candidate set/order')
    check(pre['stage'] == 'BEFORE_AUTHORING' and pre['one_seed_only'] is True and pre['seed_search_performed'] is False, 'Preauthoring config')
    check(pre['semantic_or_selection_changes_allowed'] is False, 'Semantic config')
    check(plan['presentation_export']['seed_namespace'] == pre['seed_namespace'] == freeze['seed_namespace'] == newm['seed_namespace'] == EXPECTED_SEED, 'Seed binding')
    check(plan['presentation_export']['one_seed_only'] is True and plan['presentation_export']['seed_search_performed'] is False, 'Plan one seed')
    check(plan['presentation_export']['selection_or_semantic_content_changed'] is False, 'Plan semantic restriction')
    check(pre['permutation_algorithm'] == freeze['algorithm'] == newm['presentation_version'] == 'sha256_counter_fisher_yates_v1', 'Algorithm')
    check(digest(VIEW/'export_presentation.py') == EXPECTED_TOOL_SHA == pre['source_script']['sha256'] == freeze['tool']['sha256'] == complete['exporter']['sha256'], 'Exporter frozen hashes')
    check(pre['planned_source_package'] == freeze['source_package'] == bindings['source_package'] == str(SOURCE), 'Source paths')
    check(pre['planned_pool'] == freeze['pool_path'] == bindings['pool_path'] == str(POOL), 'Pool paths')
    check(pre['planned_output'] == str(OUT), 'Output path')
    check(freeze['seed_search_performed'] is False, 'Export says seed search')
    check(plan['created_at_utc'] < pre['created_at_utc'] < freeze['created_at_utc'] < complete['created_at_utc'], 'Freeze chronology')
    # Confirm the actual preauthoring seed freeze predates every saved expansion author interval.
    receipt_count = 0
    for receipt_path in sorted((BATCH/'roots').glob('*/AUTHOR_RECEIPT.json')):
        receipt = read(receipt_path)
        timestamps = [v for k,v in receipt.items() if isinstance(v,str) and ('start' in k.lower()) and '2026-' in v]
        check(timestamps, 'Author receipt lacks start time: '+str(receipt_path))
        check(all(pre['created_at_utc'] < stamp for stamp in timestamps), 'Seed frozen after author start')
        receipt_count += 1
    check(receipt_count == 12, 'Expected twelve author receipts')
    current_source = snapshot(SOURCE)
    check(current_source == freeze['source_package_snapshot'] == bindings['source_package_before'] == bindings['source_package_after'], 'Source snapshots')
    check(bindings['all_source_package_hashes_unchanged'] is True, 'Input unchanged assertion')
    check(pool_before == freeze['pool_sha256'] == bindings['pool_sha256_before'] == bindings['pool_sha256_after'], 'Pool hashes')
    check(digest(SOURCE/'COMPLETE_MANIFEST.json') == bindings['original_complete_manifest_sha256'], 'Source complete binding')
    check(digest(SOURCE/'FRONTIER_MANIFEST.json') == newm['source_frontier_manifest_sha256'], 'Source frontier binding')
    check(str(SOURCE) == newm['source_package'], 'New manifest source')
    check(('--seed '+EXPECTED_SEED) in (OUT/'README.md').read_text(), 'README actual seed')
    if (VIEW/'README.md').exists():
        check(EXPECTED_SEED in (VIEW/'README.md').read_text(), 'Root README actual seed')
    check('--seed v107_training_presentation_v1' not in (OUT/'README.md').read_text(), 'Stale README seed')
    expected_candidates = []
    for idx, (oldrow,newrow,oldc,newc,mapping) in enumerate(zip(oldrows,newrows,oldm['candidates'],newm['candidates'],maps)):
        cid = oldc['candidate_id']
        q = questions[cid]
        kind = oldc['question_type']
        check(kind == q['question_type'] and kind in ['multi_select','ordering'], 'Question type '+cid)
        alphabet = list('ABCDEF') if kind == 'multi_select' else ['I'+str(k) for k in range(1,7)]
        texts = q['options'] if kind == 'multi_select' else q['items']
        check(set(texts) == set(alphabet), 'Label set '+cid)
        check(all(isinstance(v,str) and '\n' not in v for v in texts.values()), 'Single-line item '+cid)
        oldgold = list(oldc['canonical_gold'])
        check(oldgold == q['gold'], 'Pool Gold '+cid)
        if kind == 'multi_select':
            check(len(oldgold) in [2,3,4,5], 'Gold6/non2-5 candidate '+cid)
        check(oldc['source_sha256'] == q['source_sha256'], 'Source digest '+cid)
        oldmsg = oldrow['messages']
        check(len(oldmsg) == 3 and [m['role'] for m in oldmsg] == ['system','user','assistant'], 'Roles '+cid)
        check(oldmsg[2]['content'] == '<box>'+','.join(oldgold)+'</box>', 'Old boxed Gold '+cid)
        check(message_digest(oldmsg) == oldc['messages_sha256'], 'Source message hash '+cid)
        heading = 'Options:' if kind == 'multi_select' else 'Items:'
        oldblock = '\n\n'+heading+'\n'+'\n'.join(k+'. '+texts[k] for k in alphabet)+'\n\n'
        check(oldmsg[1]['content'].count(oldblock) == 1, 'Single source block '+cid)
        prefix,suffix = oldmsg[1]['content'].split(oldblock)
        check(('Question ('+kind+'):\n'+q['question']) in prefix and suffix.startswith('Output only '), 'Source question boundary '+cid)
        shuffled, seedhash = independent_order(EXPECTED_SEED,cid,alphabet)
        n2o = {label:shuffled[position] for position,label in enumerate(alphabet)}
        o2n = {old:new for new,old in n2o.items()}
        relabeled = {label:texts[n2o[label]] for label in alphabet}
        newgold = [o2n[k] for k in oldgold]
        if kind == 'multi_select':
            newgold = sorted(newgold, key=alphabet.index)
        newblock = '\n\n'+heading+'\n'+'\n'.join(k+'. '+relabeled[k] for k in alphabet)+'\n\n'
        expected_row = copy.deepcopy(oldrow)
        expected_row['messages'][1]['content'] = prefix+newblock+suffix
        expected_row['messages'][2]['content'] = '<box>'+','.join(newgold)+'</box>'
        check(newrow == expected_row, 'Full row invariant '+cid)
        newmsg = newrow['messages']
        old_correct = [texts[k] for k in oldgold]
        new_correct = [relabeled[k] for k in newgold]
        check(Counter(texts.values()) == Counter(relabeled.values()), 'Text multiset '+cid)
        check((Counter(old_correct) == Counter(new_correct)) if kind == 'multi_select' else old_correct == new_correct, 'Correct content '+cid)
        expected_map = {
            'row_index_zero_based':idx, 'candidate_id':cid, 'root_id':oldc['root_id'], 'question_type':kind,
            'candidate_seed_sha256':seedhash, 'old_label_to_new_label':o2n, 'new_label_to_old_label':n2o,
            'old_content_by_label':texts, 'new_content_by_label':relabeled, 'old_gold':oldgold, 'new_gold':newgold,
            'old_gold_texts':old_correct, 'new_gold_texts':new_correct,
            'old_messages_sha256':message_digest(oldmsg), 'new_messages_sha256':message_digest(newmsg),
            'old_message_content_sha256':{m['role']:digest_bytes(m['content'].encode()) for m in oldmsg},
            'new_message_content_sha256':{m['role']:digest_bytes(m['content'].encode()) for m in newmsg},
            'unchanged_user_prefix_sha256':digest_bytes(prefix.encode()), 'unchanged_user_suffix_sha256':digest_bytes(suffix.encode()),
            'content_multiset_unchanged':True, 'correct_content_unchanged':True, 'three_roles_unchanged':True,
            'user_outside_replacement_block_unchanged':True, 'system_message_unchanged':True, 'replacement_block_occurrences':1,
        }
        check(mapping == expected_map, 'Full mapping invariant '+cid)
        expected_c = copy.deepcopy(oldc)
        expected_c.update({
            'source_canonical_gold':oldgold, 'source_messages_sha256':oldc['messages_sha256'],
            'canonical_gold':newgold, 'messages_sha256':message_digest(newmsg), 'training_view_content':relabeled,
            'presentation_mapping_row':idx,
            'solver_question_statistics_scope':'Unchanged historical solver statistics use source canonical labels and randomized solver displays; do not reinterpret their gold/content fields as new training-view labels.',
        })
        check(newc == expected_c, 'Full candidate/eligibility/weights/historical stats invariant '+cid)
        expected_candidates.append(expected_c)
    expected_m = copy.deepcopy(oldm)
    expected_m.update({
        'status':'PRESENTATION_ONLY_CANDIDATE_VIEW_NO_TRAINING_EXECUTED',
        'presentation_version':'sha256_counter_fisher_yates_v1', 'seed_namespace':EXPECTED_SEED,
        'candidates':expected_candidates, 'source_package':str(SOURCE),
        'source_frontier_manifest_sha256':digest(SOURCE/'FRONTIER_MANIFEST.json'),
        'semantic_batch_or_selection_changed':False, 'weights_changed':False, 'replay_mixed':False,
    })
    check(newm == expected_m, 'Full manifest metadata invariant')
    types = dict(Counter(c['question_type'] for c in oldm['candidates']))
    golds = dict(sorted(Counter(str(len(c['canonical_gold'])) for c in oldm['candidates'] if c['question_type']=='multi_select').items()))
    check(audit['rows'] == n and audit['type_counts'] == types and audit['multi_gold_counts'] == golds, 'Audit count fields')
    for field in ['content_multisets_unchanged','correct_content_sets_or_ordered_sequences_unchanged','system_messages_unchanged','three_role_rows','single_block_replacements']:
        check(audit[field] == n, 'Audit counter '+field)
    for field in ['candidate_id_order_unchanged','selection_eligibility_and_weights_unchanged','one_row_per_source_question']:
        check(audit[field] is True, 'Audit assertion '+field)
    check(audit['seed_search_performed'] is False, 'Audit seed search')
    check(audit['before'] == patterns(maps,'old_gold') and audit['after'] == patterns(maps,'new_gold'), 'Independent label pattern stats')
    replay = read(SOURCE/'REPLAY_BINDING.json')
    check(replay['mixed_training_file_created'] is False, 'Source mixed replay')
    check(complete['replay_mixed'] is False and complete['training_started'] is False and complete['new_model_calls'] == 0, 'Export no calls/replay/training')
    check(source_complete['training_started'] is False, 'Source no training')
    if root_complete is not None:
        check(root_complete['training_started'] is False, 'Root no training')
    check(before == {str(p):snapshot(p) for p in [SOURCE,VIEW]} and pool_before == digest(POOL), 'Files mutated during readonly review')
    ended = now()
    record = {
        'status':'PASS_INDEPENDENT_READONLY_PRESENTATION_REVIEW',
        'reviewer':'/root/scale_group3', 'script_prepared_by_primary_at_utc':SCRIPT_PREPARED_AT,
        'execution_started_at_utc':execution_started, 'finished_at_utc':ended,
        'rows':n, 'type_counts':types, 'multi_gold_counts':golds,
        'independent_permutations_reconstructed':n, 'full_row_equality_checks':n,
        'full_candidate_metadata_equality_checks':n, 'full_bidirectional_mapping_checks':n,
        'correct_content_checks':n, 'source_author_receipts_predated_by_seed_freeze':receipt_count,
        'manifest_verified_file_counts':{'source_package_excluding_self':source_files,'presentation_package_excluding_self':output_files,'view_root_excluding_self':root_files},
        'view_root_manifest_status':'VERIFIED' if root_sealed_at_review else 'PENDING_FINAL_DELIVERY_BUILDER_EXPECTED',
        'view_root_manifest_note':'The final builder seals the enclosing view root only after this independent package review; source and presentation package manifests are mandatory and verified here.',
        'seed_namespace':EXPECTED_SEED, 'one_predeclared_seed':True, 'seed_search_performed_by_reviewer':False,
        'checks':[
            'All SHA256 counter Fisher-Yates permutations independently reconstructed from preauthoring seed and candidate ID.',
            'Candidate set/order, row and candidate metadata, eligibility, weights and historical solver scores unchanged.',
            'System, supplied dialogue/question prefix and output instruction suffix byte-identical.',
            'All option/item text multisets unchanged; multi-select correct content sets and ordering correct content sequences unchanged.',
            'Every saved mapping field, both label directions, message hashes and candidate seed hashes independently matched.',
            'Full source and presentation package manifest file sets, hashes and sizes verified; enclosing view-root seal is recorded separately as verified or pending final builder.',
            'Every original package file and pool hash matches freeze/before/after/current bindings.',
            'README reproduction command uses actual v111 seed; no stale v107 example.',
            'Gold6 controls excluded; no replay mixed, new training, or external model/Solver calls by exporter/reviewer.',
            'All inputs and sealed outputs remained unchanged throughout the read-only review.',
        ],
        'ordering_max_same_sequence':{
            'before':audit['before'].get('ordering',{}).get('maximum_identical_gold_label_sequence_count'),
            'after':audit['after'].get('ordering',{}).get('maximum_identical_gold_label_sequence_count'),
            'descriptive_only':True,
        },
        'bindings':[{'path':str(p),'sha256':digest(p)} for p in [Path(__file__),VIEW/'PREAUTHORING_EXPORT_CONFIG.json',VIEW/'export_presentation.py',OUT/'COMPLETE_MANIFEST.json',OUT/'frontier_candidates.jsonl',SOURCE/'COMPLETE_MANIFEST.json',SOURCE/'frontier_candidates.jsonl',POOL]+([VIEW/'COMPLETE_MANIFEST.json'] if root_sealed_at_review else [])],
        'independence_limitation':'Separate reviewer role with same inherited model and shared project background; not external, fully blind or different-model certification. Reviewer did not invoke exporter or regenerate an export.',
        'training_started':False, 'replay_mixed':False, 'external_model_or_solver_calls_by_reviewer':0,
    }
    with RECORD.open('x',encoding='utf-8') as f:
        json.dump(record,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(record,ensure_ascii=False))


if __name__ == '__main__':
    main()
