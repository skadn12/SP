#!/usr/bin/env python3
"""Build a deterministic historical reuse REVIEW queue. No model/semantic review."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/gxy-selfplay')
V131 = BASE / 'offline-deliveries/h4_codex_historical_frontier_eval_v131'
HIST = BASE / 'restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1'
SPLIT = BASE / 'restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl'
PUBLIC_FIELDS = ('candidate_id', 'root_id', 'question_type', 'question', 'options', 'items')


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def objhash(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def binding(p):
    p = Path(p)
    return {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size}


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    assert not (HERE / 'PUBLIC_FREEZE.json').exists(), 'Already frozen; do not overwrite review inputs.'
    stamp = datetime.now(timezone.utc).isoformat()
    allq = read(V131 / 'QUESTIONS.json')['candidates']
    scope = read(V131 / 'SCOPE.json')
    quality = read(V131 / 'evidence/HISTORICAL_QUALITY_CROSSCHECK.json')
    qmeta = {x['candidate_id']: x for x in quality['rows']}
    preflight = read(HIST / 'preflight_report.json')
    source_inventory = preflight['source_root_inventory']
    splits = [json.loads(x) for x in SPLIT.read_text().splitlines() if x.strip()]
    split_by_root = {x['root_id']: x for x in splits}
    assert len(split_by_root) == len(splits) == 101
    questions = sorted((x for x in allq if x['question_type'] in {'multi_select', 'ordering'}),
                       key=lambda x: x['candidate_id'])
    excluded = [x for x in allq if x['question_type'] == 'single_choice']
    assert len(allq) == 123 and len(excluded) == 10
    assert Counter(x['question_type'] for x in questions) == {'multi_select': 51, 'ordering': 62}
    assert len({x['candidate_id'] for x in questions}) == 113
    roots = sorted({q['root_id'] for q in questions})
    assert len(roots) == 56
    assert all(split_by_root[r]['split'] == 'train' for r in roots)
    input_paths = [V131 / 'QUESTIONS.json', V131 / 'SCOPE.json',
                   V131 / 'evidence/HISTORICAL_QUALITY_CROSSCHECK.json',
                   HIST / 'preflight_report.json', SPLIT]
    source_records, sources, issue_rows = {}, {}, []
    for root in roots:
        si = source_inventory[root]
        frozen = V131 / 'sources' / (root + '.json')
        original = Path(si['dialogue_path'])
        pool = Path(si['candidate_pool_path'])
        reasons = []
        if not frozen.exists():
            reasons.append('V131_COMPLETE_SOURCE_MISSING')
        elif digest(frozen) != si['dialogue_sha256']:
            reasons.append('V131_SOURCE_HASH_MISMATCH_HISTORICAL_INVENTORY')
        if original.exists() and digest(original) != si['dialogue_sha256']:
            reasons.append('ORIGINAL_DIALOGUE_HASH_MISMATCH')
        if not pool.exists():
            reasons.append('ORIGINAL_CANDIDATE_POOL_MISSING')
        elif digest(pool) != si['candidate_pool_sha256']:
            reasons.append('ORIGINAL_CANDIDATE_POOL_HASH_MISMATCH')
        local = HERE / 'sources' / (root + '.json')
        source = None
        if frozen.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(frozen, local)
            source = read(local)
            turns = source.get('turns', [])
            refs = [t['turn_ref'] for t in turns]
            if source.get('root_id') != root:
                reasons.append('SOURCE_ROOT_MISMATCH')
            if len(refs) != len(set(refs)):
                reasons.append('DUPLICATE_SOURCE_TURN_REF')
            sources[root] = source
        source_records[root] = {
            'root_id': root,
            'source_binding': binding(local) if local.exists() else None,
            'v131_source_binding': binding(frozen) if frozen.exists() else None,
            'historical_original_source_binding': binding(original) if original.exists() else None,
            'historical_expected_dialogue_sha256': si['dialogue_sha256'],
            'original_candidate_pool_binding': binding(pool) if pool.exists() else None,
            'historical_expected_candidate_pool_sha256': si['candidate_pool_sha256'],
            'split_record': split_by_root[root],
            'source_status': 'HOLD' if reasons else 'READY_FOR_SOURCE_REVIEW',
            'hold_reasons': reasons,
            'turn_count': len(source.get('turns', [])) if source else None,
            'source_metadata_turn_count': source.get('turn_count') if source else None,
            'source_origin': 'Historical parent dialogue plus R01 append; unchanged complete source.',
        }
    rows, public = [], []
    pool_cache = {}
    for index, q in enumerate(questions):
        cid, root, kind = q['candidate_id'], q['root_id'], q['question_type']
        sr, qm = source_records[root], qmeta[cid]
        reasons = list(sr['hold_reasons'])
        pool_path = Path(source_inventory[root]['candidate_pool_path'])
        if pool_path.exists():
            if root not in pool_cache:
                pool_cache[root] = {x['candidate_id']: x for x in read(pool_path)}
            if pool_cache[root].get(cid) != q:
                reasons.append('V131_RECORD_DIFFERS_FROM_ORIGINAL_POOL')
        turns = sources.get(root, {}).get('turns', [])
        refs = {t['turn_ref'] for t in turns}
        evidence_refs = list(q.get('evidence_refs', []))
        group_refs = [r for rs in q.get('evidence_groups', {}).values() for r in rs]
        missing = sorted(set(evidence_refs + group_refs) - refs)
        if missing:
            reasons.append('AUTHOR_EVIDENCE_REF_NOT_FOUND_IN_SOURCE')
        labels = list(q['options'] if kind == 'multi_select' else q['items'])
        gold = q['gold']
        schema_ok = (
            len(labels) == (5 if kind == 'multi_select' else 4)
            and isinstance(gold, list) and len(gold) == len(set(gold))
            and set(gold) <= set(labels)
            and (kind != 'ordering' or set(gold) == set(labels))
        )
        if not schema_ok:
            reasons.append('LEGACY_SCHEMA_CHECK_FAILED')
        applicable = qm.get('applicable_historical_reviews', [])
        lookup = 'HISTORICAL_ACCEPT_ONLY' if applicable else 'NO_APPLICABLE_REVIEW_FOUND'
        rows.append({
            'queue_index': index,
            'candidate_id': cid, 'root_id': root, 'question_type': kind,
            'content_status': 'PENDING_REVIEW', 'training_admitted': False,
            'review_readiness': 'HOLD' if reasons else 'READY_FOR_SOURCE_REVIEW',
            'hold_reasons': reasons,
            'split': split_by_root[root]['split'],
            'split_namespace': split_by_root[root]['split_namespace'],
            'split_version': split_by_root[root]['split_version'],
            'source_binding': sr['source_binding'],
            'original_question_record': q,
            'original_question_object_sha256': objhash(q),
            'original_record_reference': {
                'file': str(V131 / 'QUESTIONS.json'),
                'json_pointer': '/candidates/' + str(allq.index(q)),
                'file_sha256': digest(V131 / 'QUESTIONS.json'),
            },
            'legacy_format': {
                'option_or_item_count': len(labels), 'labels': labels,
                'task_definition_status': 'PENDING_SEMANTIC_REVIEW',
                'ordering_definition': 'Interpret exact original public wording; distinguish appearance order from event time before admission.' if kind == 'ordering' else None,
                'do_not_mechanically_pad_to_six': True,
            },
            'mechanical_checks': {
                'schema_ok': schema_ok, 'all_author_refs_resolve': not missing,
                'missing_author_refs': missing,
                'reference_resolution_is_not_semantic_verification': True,
            },
            'historical_review_lookup': {
                'lookup_status': lookup,
                'applicable_reviews': applicable,
                'excluded_same_id_different_version_reviews': qm.get('rejected_same_id_reviews', []),
                'bound_historical_nonvalid_gold_labels': qm.get('bound_historical_nonvalid_gold_labels', []),
                'fresh_semantic_review_done': False,
                'absence_is_not_failure_or_acceptance': True,
            },
            'historical_provenance': {
                'cohort_origin': str(V131 / 'SCOPE.json'),
                'historical_cohort_rule': scope['selection_rule'],
                'historical_partial_definition': scope['historical_selection_partial_means'],
                'first_round_exact_candidate_exposure': qm['first_round_training_exposure'],
                'same_root_used_by_v125': qm['new_v125_root_overlap'],
                'not_unseen_root_or_pristine_evaluation': True,
                'reuse_category': 'HISTORICAL_QUESTION_REUSE_NOT_FIRST_ROUND_TRAINED_REPLAY',
                'current_inventory_selection': 'All113 historical MC/ordering retained; type-only exclusion of10 single_choice; no new score-driven selection.',
            },
        })
        p = {k: q[k] for k in PUBLIC_FIELDS}
        p['source_binding'] = sr['source_binding']
        public.append(p)
        if reasons:
            issue_rows.append({'candidate_id': cid, 'hold_reasons': reasons})
    source_index = {'schema_version': 'historical_reuse_sources_v1', 'roots': [source_records[r] for r in roots]}
    inventory = {
        'schema_version': 'historical_reuse_inventory_v1', 'created_at_utc': stamp,
        'status': 'REVIEW_QUEUE_PREPARED_NOT_ADMITTED',
        'selection_policy': 'Preserve every original v131 multi_select/ordering record; exclude only single_choice. No Solver-score use in this build.',
        'historical_cohort_is_outcome_selected': True,
        'questions': rows,
    }
    write(HERE / 'INVENTORY.json', inventory)
    write(HERE / 'SOURCE_INDEX.json', source_index)
    write(HERE / 'EXCLUDED_SINGLE_CHOICE.json', {
        'count': 10, 'reason': 'Outside current MC/ordering scope; not a content rejection.',
        'candidate_ids': [q['candidate_id'] for q in excluded],
    })
    write(HERE / 'public/QUESTIONS.json', {'schema_version': 'source_first_public_questions_v1', 'questions': public})
    byroot = {r: [q for q in public if q['root_id'] == r] for r in roots}
    chunks, chunk, chunkroots = [], [], []
    for root in roots:
        if chunk and len(chunk) + len(byroot[root]) > 12:
            chunks.append((chunkroots, chunk))
            chunk, chunkroots = [], []
        chunk.extend(byroot[root]); chunkroots.append(root)
    if chunk:
        chunks.append((chunkroots, chunk))
    chunk_index = []
    for ci, (rs, qs) in enumerate(chunks):
        path = HERE / f'public/chunks/CHUNK_{ci:02d}.json'
        write(path, {'questions': qs})
        chunk_index.append({'chunk_id': f'CHUNK_{ci:02d}', 'root_ids': rs,
                            'question_count': len(qs), 'binding': binding(path)})
    write(HERE / 'public/CHUNK_INDEX.json', {'policy': 'Sorted root blocks, at most12 questions/chunk; roots never split.', 'chunks': chunk_index})
    eligible = [r for r in roots if {q['question_type'] for q in byroot[r]} == {'multi_select', 'ordering'}]
    pilot_roots = eligible[:3]
    pilot = [next(q for q in byroot[r] if q['question_type'] == typ)
             for r in pilot_roots for typ in ('multi_select', 'ordering')]
    write(HERE / 'pilot/PUBLIC_QUESTIONS.json', {'questions': pilot})
    for q in pilot:
        fname = hashlib.sha256(q['candidate_id'].encode()).hexdigest()[:16] + '.json'
        write(HERE / 'pilot/packets' / fname, q)
    write(HERE / 'pilot/PILOT_SELECTION.json', {
        'selection_rule': 'Lexicographically first3 roots with both MC and ordering; lowest candidate_id of each type per root.',
        'selection_uses_scores': False, 'root_ids': pilot_roots,
        'candidate_ids': [q['candidate_id'] for q in pilot],
        'question_count': len(pilot), 'content_status': 'PENDING_REVIEW',
        'public_binding': binding(HERE / 'pilot/PUBLIC_QUESTIONS.json'),
    })
    bound_inputs = [binding(p) for p in input_paths]
    for r in roots:
        for key in ['v131_source_binding', 'historical_original_source_binding', 'original_candidate_pool_binding']:
            if source_records[r][key] is not None:
                bound_inputs.append(source_records[r][key])
    write(HERE / 'INPUT_BINDINGS.json', {'files': list({x['path']: x for x in bound_inputs}.values())})
    validation = {
        'schema_version': 'historical_reuse_mechanical_validation_v1', 'status': 'PASS' if not issue_rows else 'HOLDS_PRESENT',
        'questions_original': len(allq), 'questions_retained': len(rows),
        'question_types': dict(Counter(q['question_type'] for q in rows)),
        'retained_roots': len(roots), 'single_choice_excluded': len(excluded),
        'pending_semantic_review': len(rows), 'admitted_questions': 0,
        'historical_accept_only': sum(bool(x['historical_review_lookup']['applicable_reviews']) for x in rows),
        'no_applicable_review_found': sum(not x['historical_review_lookup']['applicable_reviews'] for x in rows),
        'source_hold_questions': len(issue_rows), 'holds': issue_rows,
        'legacy_counts': {'multi_select_option_count': 5, 'ordering_item_count': 4},
        'all113_have_source_files': all(r['source_binding'] is not None for r in rows),
        'all113_are_train_in_original_namespace': all(r['split'] == 'train' for r in rows),
        'all113_preserve_original_record_losslessly': all(x['original_question_record'] == next(q for q in allq if q['candidate_id'] == x['candidate_id']) for x in rows),
        'all113_public_questions_use_whitelist': all(set(x) == set(PUBLIC_FIELDS) | {'source_binding'} for x in public),
        'source_first_chunks': len(chunks), 'pilot_questions': len(pilot),
        'no_model_calls': True, 'no_training': True, 'no_new_semantic_claims': True,
        'inventory_builder_had_historical_gold_access': True,
        'review_must_be_performed_by_other_source_first_reviewer': True,
    }
    write(HERE / 'VALIDATION.json', validation)
    public_paths = [p for folder in ['public', 'pilot', 'sources'] for p in sorted((HERE / folder).rglob('*')) if p.is_file()]
    write(HERE / 'PUBLIC_FREEZE.json', {
        'schema_version': 'historical_reuse_public_freeze_v1', 'created_at_utc': stamp,
        'status': 'FROZEN_FOR_SOURCE_FIRST_REVIEW',
        'files': [{'path': str(p.relative_to(HERE)), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in public_paths],
        'public_question_schema': list(PUBLIC_FIELDS) + ['source_binding'],
        'not_admitted': True,
    })
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    main()
