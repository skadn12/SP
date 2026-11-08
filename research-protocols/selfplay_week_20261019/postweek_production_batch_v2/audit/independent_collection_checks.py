"""Independent accounting helpers. Read only except the caller's new audit file."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    p = Path(path).resolve()
    raw = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def check(b):
    actual = binding(b['path'])
    assert actual['sha256'] == b['sha256'], b['path']
    if 'bytes' in b:
        assert actual['bytes'] == b['bytes']
    if 'size_bytes' in b:
        assert actual['bytes'] == b['size_bytes']


def signature(c):
    choices = c['options' if c['question_type'] == 'multi_select' else 'items']
    return json.dumps({'type': c['question_type'], 'question': c['question'],
                       'choices': sorted(choices.values())}, ensure_ascii=False, sort_keys=True)


def selection_checks(here):
    s = read(here / 'SELECTION_FREEZE.json')
    inventory = read(here / 'SOURCE_SELECTION_INVENTORY.json')
    check(s['source_selection_inventory_binding'])
    for key in ('source_manifest_binding', 'root_boundary_binding',
                'authoritative_split_binding', 'historical_dependency_binding'):
        check(inventory[key])
    manifest_path = Path(inventory['source_manifest_binding']['path'])
    manifest = read(manifest_path)
    assert len({r['root_id'] for r in manifest}) == len(manifest)
    assert all(r['split'] == 'train' for r in manifest)
    # Split files are metadata only. No DEV/TEST dialogue, question or answer is read.
    split = {r['root_id']: r['split'] for r in
             (json.loads(line) for line in Path(inventory['authoritative_split_binding']['path']).read_text().splitlines())}
    boundary = read(inventory['root_boundary_binding']['path'])['split_root_ids']
    train_ids = set(boundary['train'])
    assert all(split[r['root_id']] == 'train' and r['root_id'] in train_ids for r in manifest)
    previous = read(here.parent / 'postweek_production_batch_v1/SELECTION_FREEZE.json')
    excluded = {r['root_id'] for r in previous['roots']}
    assert excluded == set(s['excluded_production_batch1_roots'])
    excluded |= {'cc-train-episode-' + value for value in s['excluded_quality_validation_roots']}
    assert excluded == set(inventory['excluded_root_ids']) and len(excluded) == 19
    available = sorted(r['root_id'] for r in manifest if r['root_id'] not in excluded)
    chosen = [r['root_id'] for r in s['roots']]
    assert available == inventory['all_available_root_ids'] and len(available) == 49
    assert chosen == available[:10] == inventory['recommended_root_ids']
    by_id = {r['root_id']: r for r in manifest}
    sources = {}
    paths = [here / 'SELECTION_FREEZE.json', here / 'SOURCE_SELECTION_INVENTORY.json']
    for r in s['roots']:
        m = by_id[r['root_id']]
        packaged = manifest_path.parent.parent / m['packaged_source']
        b = {k: r[k] for k in ('path', 'sha256', 'bytes')}
        assert binding(packaged) == b
        check(m['source_binding'])
        assert m['source_binding']['sha256'] == b['sha256']
        source = read(packaged)
        assert source['root_id'] == r['root_id']
        assert len(source['turns']) == len({t['turn_ref'] for t in source['turns']})
        sources[r['root_id']] = source
        paths.extend([packaged, Path(m['source_binding']['path'])])
    baseline = read(here / 'audit/HISTORICAL_BASELINE.json')
    for b in baseline['bindings']:
        check(b)  # Hash only; never parses historical solver text.
    check(baseline['preserved_prior_batch_manifest'])
    prior_manifest = read(baseline['preserved_prior_batch_manifest']['path'])
    for b in prior_manifest['artifact_bindings'] + prior_manifest['external_reference_bindings']:
        check(b)
    return {'selected_train_roots': 10, 'manifest_train_roots': len(manifest),
            'available_after_19_exclusions': len(available),
            'lexicographic_first_ten_verified': True,
            'source_original_and_packaged_hashes_equal': True,
            'prior_batch_and_historical_baseline_unchanged': True,
            'historical_baseline_bindings_verified': len(baseline['bindings']),
            'dev_test_content_read': False,
            'bindings': [binding(p) for p in paths]}, s


def audit_collection(here):
    root = here.parents[2]
    sys.path.insert(0, str(root))
    from scripts.current_method.production_question_gate import (
        bind_candidates, fingerprint, validate_review,
    )
    source_result, selection = selection_checks(here)
    planned = {f"PROD2-{r['root_id'].split('-')[-1]}-{slot['slot']}": (r, slot)
               for r in selection['roots'] for slot in r['slot_plan']}
    assert len(planned) == selection['planned_candidates'] == 70
    assert Counter(p['question_type'] for _, p in planned.values()) == {'multi_select': 50, 'ordering': 20}
    assert selection['initial_joint_review_scaleup_threshold'] == .9
    assert selection['per_candidate_draft_budget'] == selection['pre_execution_review_repair_budget'] == 1
    assert not selection['post_solver_rewrite_or_retry']
    initial, final, candidates, slots, paths, groups, repaired = {}, {}, {}, {}, [], [], []
    for group in ('a', 'b'):
        dp = here / f'author_{group}/CANDIDATES_PRIVATE_DRAFT.json'
        fp = here / f'author_{group}/CANDIDATES_PRIVATE.json'
        ip = here / f'review_{group}/INITIAL_CONTENT_REVIEW_AI.json'
        rp = here / f'review_{group}/CONTENT_REVIEW_AI.json'
        lp = here / f'author_{group}/SLOT_LEDGER_DRAFT.json'
        draft_rows, final_rows = read(dp), read(fp)
        old, new = ({c['candidate_id']: c for c in rows} for rows in (draft_rows, final_rows))
        ir, fr = ({r['candidate_id']: r for r in read(p)['reviews']} for p in (ip, rp))
        assert len(old) == len(draft_rows) and len(new) == len(final_rows)
        assert set(old) == set(new) == set(ir) == set(fr)
        bound, _ = bind_candidates(final_rows, here / 'SELECTION_FREEZE.json')
        assert bound == final_rows
        ledger = read(lp)
        ledger = ledger['slots'] if isinstance(ledger, dict) else ledger
        assert len(ledger) == len({r['candidate_id'] for r in ledger}) == 35
        for row in ledger:
            cid = row['candidate_id']
            assert cid not in slots
            slots[cid] = row
            if cid not in new:
                assert row['status'] == 'MATERIAL_SHORTFALL'
        for cid, c in new.items():
            r, p = planned[cid]
            assert r['author_group'] == group and c['root_id'] == r['root_id']
            assert c['question_type'] == p['question_type'] and c['task_operation'] == p['task_operation']
            assert (len(c['gold']) == p['gold_count'] and len(c['options']) == 6) if c['question_type'] == 'multi_select' else 4 <= len(c['items']) <= 6
            assert ir[cid]['candidate_sha256'] == fingerprint(old[cid])
            assert fr[cid]['candidate_sha256'] == fingerprint(c)
            assert c['author_id'] == fr[cid]['author_id'] != fr[cid]['reviewer_id']
            old_audit = validate_review(old[cid], ir[cid], here / 'SELECTION_FREEZE.json')
            audit = validate_review(c, fr[cid], here / 'SELECTION_FREEZE.json')
            assert old_audit['review_record_complete'] and audit['review_record_complete']
            if fr[cid]['decision'] == 'PASS':
                assert audit['admitted']
            if c != old[cid]:
                repaired.append(cid)
                # A completed initial content review must predate any one-round
                # fix, including fixes to metadata; the original stays preserved.
                assert ir[cid]['decision'] in ('HOLD', 'FAIL')
            assert cid not in candidates
            candidates[cid], initial[cid], final[cid] = c, ir[cid], fr[cid]
        groups.append({'group': group, 'submitted': len(new),
                       'initial': dict(Counter(r['decision'] for r in ir.values())),
                       'final': dict(Counter(r['decision'] for r in fr.values())),
                       'authors': sorted({c['author_id'] for c in new.values()}),
                       'reviewers': sorted({r['reviewer_id'] for r in fr.values()})})
        paths.extend([dp, fp, ip, rp, lp])
        logs = list((here / f'author_{group}').glob('*REPAIR*.json'))
        for log in logs:
            d = read(log)
            if isinstance(d, dict) and 'repair_budget_used' in d:
                assert d['repair_budget_used'] <= 1
            paths.append(log)
    assert set(slots) == set(planned)
    assert {c['candidate_id']: c for c in read(here / 'CANDIDATES_PRIVATE.json')} == candidates
    merged = {r['candidate_id']: r for r in read(here / 'CONTENT_REVIEW_AI.json')['reviews']}
    assert set(merged) == set(candidates)
    for cid, r in merged.items():
        assert all(r[k] == value for k, value in final[cid].items())
        assert r['initial_decision'] == initial[cid]['decision']
        assert r['initial_candidate_sha256'] == initial[cid]['candidate_sha256']
        assert r['repaired_after_initial_review'] == (cid in repaired)
    collection = read(here / 'COLLECTION_AUDIT.json')
    joint = sum(r['decision'] == 'PASS' for r in initial.values())
    admitted = [c for cid, c in candidates.items() if final[cid]['decision'] == 'PASS']
    shortfalls = [row for cid, row in slots.items() if cid not in candidates]
    assert collection['initial_joint_pass'] == joint
    assert collection['initial_joint_pass_rate_planned'] == joint / 70
    assert collection['initial_joint_pass_rate_submitted'] == joint / len(candidates)
    assert collection['final_pass'] == len(admitted)
    assert collection['initial_joint_90_percent_scaleup_threshold_met'] == (joint / 70 >= .9)
    assert len(collection['material_shortfalls']) == len(shortfalls)
    assert set(collection['repaired_candidates']) == set(repaired)
    whole = [signature(c) for c in candidates.values()]
    assert len(whole) == len(set(whole))
    # The audit inventory lists only Train question files. No held-out content.
    import importlib.util
    spec = importlib.util.spec_from_file_location('independent_dupe_inventory', here / 'audit_batch.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    history, count = set(), 0
    for rel in mod.HISTORICAL_QUESTION_FILES:
        p = here.parent / rel
        d = read(p)
        rows = d if isinstance(d, list) else next(d[k] for k in ('questions', 'candidates') if k in d)
        for c in rows:
            assert c['root_id'].startswith('cc-train-')
            history.add(signature(c))
            count += 1
        paths.append(p)
    assert not set(whole) & history
    return {'source_selection': source_result, 'planned_slots': 70,
            'submitted': len(candidates), 'admitted': len(admitted),
            'initial_joint_pass': joint, 'initial_rate_including_shortfalls': joint / 70,
            'initial_90_percent_threshold_met': joint / 70 >= .9,
            'material_shortfalls': len(shortfalls), 'repaired_candidates': repaired,
            'groups': groups, 'author_reviewer_separation_verified': True,
            'whole_question_duplicates_in_batch': 0,
            'whole_question_duplicates_against_history': 0,
            'historical_question_rows': count, 'historical_unique_signatures': len(history),
            'bindings': [binding(p) for p in paths]}


def aggregate(records):
    mc = [s for s in records if s['question_type'] == 'multi_select']
    order = [s for s in records if s['question_type'] == 'ordering']
    n = len(records)
    correct = sum(s['exact'] for s in records)
    pair_count = sum(s['pairwise_total'] for s in order)
    pair_right = sum(s['pairwise_correct'] for s in order)
    return {'frozen_requests': n, 'valid_requests': sum(s['valid'] for s in records),
            'invalid_requests': sum(not s['valid'] for s in records),
            'correct': correct, 'exact_accuracy': correct / n if n else None,
            'mc_requests': len(mc), 'mc_correct': sum(s['exact'] for s in mc),
            'mc_false_positive': sum(s['false_positive'] for s in mc),
            'mc_false_negative': sum(s['false_negative'] for s in mc),
            'mc_allselect_correct': sum(s['allselect_exact'] for s in mc),
            'mc_allselect_accuracy': sum(s['allselect_exact'] for s in mc) / len(mc) if mc else None,
            'ordering_requests': len(order), 'ordering_correct': sum(s['exact'] for s in order),
            'ordering_pairwise_correct': pair_right, 'ordering_pairwise_total': pair_count,
            'ordering_pairwise_accuracy': pair_right / pair_count if pair_count else None}


def audit_completed_summary(here, rows, scores, private, summary):
    # These formulas are local to this audit, not the root summary assertions.
    assert summary['valid'] == sum(s['valid'] for s in scores)
    assert summary['correct'] == sum(s['exact'] for s in scores)
    for cell in summary['cells']:
        expected = aggregate([s for s in scores if s['condition'] == cell['condition'] and s['variant_id'] == cell['variant_id']])
        assert all(cell[k] == v for k, v in expected.items())
    for key, group in [('conditions', ('condition',)), ('root_cells', ('root_id', 'condition')),
                       ('new_mc_gold_buckets', ('condition', 'gold_count'))]:
        for cell in summary[key]:
            expected = aggregate([s for s in scores if all(s[k] == cell[k] for k in group)])
            assert all(cell[k] == v for k, v in expected.items())
    for pair in summary['variant_pairs']:
        chosen = sorted([s for s in scores if s['candidate_id'] == pair['candidate_id'] and s['condition'] == pair['condition']], key=lambda s: s['variant_id'])
        assert len(chosen) == 2
        a, b = chosen
        same = a['valid'] and b['valid'] and (set(a['prediction']) == set(b['prediction']) if a['question_type'] == 'multi_select' else a['prediction'] == b['prediction'])
        assert pair['both_valid'] == (a['valid'] and b['valid'])
        assert pair['canonical_prediction_same'] == same
        assert pair['both_exact'] == (a['exact'] and b['exact'])
        assert pair['neither_exact'] == (not a['exact'] and not b['exact'])
        assert pair['accuracy_changed'] == (a['exact'] != b['exact'])
    # The execution audit separately reconstructs labels directly from actual
    # displayed texts, recomputes each score and verifies sealed raw hashes.
    assert len(rows) == len(scores) == 4 * summary['admitted_questions']
