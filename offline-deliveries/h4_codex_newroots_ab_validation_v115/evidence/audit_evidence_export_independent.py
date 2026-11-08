"""Read-only, independent comparison of the prepared v115 evidence export.

Does not import or invoke the exporter, controllers, Solver, or training code.
Writes only the batch-level EVIDENCE_EXPORT_AUDIT.json, exclusively.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import itertools
import json

BATCH = Path(__file__).resolve().parent.parent
PROJECT = BATCH.parent.parent
ERRORS = []
COUNTS = Counter()
BINDINGS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    BINDINGS[str(path)] = sha(path)
    return json.loads(path.read_text())


def check(condition, where):
    if not condition:
        ERRORS.append(where)
    return bool(condition)


def equality(actual, expected, where):
    return check(actual == expected, where)


def indexed(rows, where):
    result = {row['candidate_id']: row for row in rows}
    check(len(result) == len(rows), where + ':duplicate_ids')
    return result


def citation(c, turns, where, speaker_required=True):
    t = turns.get(c.get('turn_ref'))
    if not check(t is not None, where + ':turn_ref'):
        return
    check(isinstance(c.get('quote'), str) and bool(c['quote']) and c['quote'] in t['text'], where + ':exact_quote')
    if speaker_required or 'speaker' in c:
        equality(c.get('speaker'), t['speaker'], where + ':speaker')
    COUNTS['source_citations_checked'] += 1


def normalized_anchor(c, turns):
    return {**c, 'speaker': turns[c['turn_ref']]['speaker']}


def hash_link(doc, key, paths, name, where):
    equality(doc.get(key), sha(paths[name]), where + ':' + key)


def inspect_path_adapter(pool):
    archive = BATCH / 'evaluation/failed_prepare_v1'
    old_pool = read(archive / 'EVALUATION_POOL.json')
    old_bindings = read(archive / 'ADAPTER_BINDINGS.json')
    bindings = read(BATCH / 'evaluation/ADAPTER_BINDINGS.json')
    failure = read(BATCH / 'evidence/PREPARE_V1_FAILURE_RECORD.json')
    freeze = read(BATCH / 'evidence/PATH_ADAPTER_TOOLING_FREEZE.json')
    receipt = read(BATCH / 'evaluation/PREPARE_PATH_ADAPTER_RECEIPT.json')
    for item in failure['original_artifacts']:
        path = Path(item['path'])
        if path.name in ('EVALUATION_POOL.json', 'ADAPTER_BINDINGS.json'):
            path = archive / path.name
        equality(sha(path), item['sha256'], 'path_adapter:failed_original_bytes:' + path.name)
    equality(failure['solver_calls'], 0, 'path_adapter:failure_no_solver_calls')
    equality(failure['exit_code'], 1, 'path_adapter:failed_prepare_recorded')
    for item in freeze['files'] + receipt['input_bindings']:
        equality(sha(item['path']), item['sha256'], 'path_adapter:frozen_or_receipt_binding:' + item['path'])
    equality(sha(freeze['original_tooling_freeze']['path']), freeze['original_tooling_freeze']['sha256'], 'path_adapter:original_tooling_freeze_unchanged')
    equality(freeze['status'], 'FROZEN_BEFORE_PATH_NORMALIZATION_AND_SOLVER', 'path_adapter:freeze_status')
    check(freeze['created_at_utc'] <= receipt['completed_at_utc'], 'path_adapter:freeze_before_receipt')
    equality(receipt['status'], 'PASS', 'path_adapter:receipt_status')
    equality(receipt['solver_calls'], 0, 'path_adapter:receipt_no_solver_calls')
    equality({k: v for k, v in old_pool.items() if k != 'questions'}, {k: v for k, v in pool.items() if k != 'questions'}, 'path_adapter:nonquestion_metadata_unchanged')
    equality(len(old_pool['questions']), len(pool['questions']), 'path_adapter:candidate_count_unchanged')
    changes = []
    actual_changed = 0
    for old, new in zip(old_pool['questions'], pool['questions']):
        equality(old['candidate_id'], new['candidate_id'], 'path_adapter:candidate_order_unchanged')
        w = 'path_adapter:' + old['candidate_id']
        equality({k: v for k, v in old.items() if k != 'source_path'}, {k: v for k, v in new.items() if k != 'source_path'}, w + ':all_other_fields_unchanged')
        path = Path(old['source_path'])
        expected = str((path if path.is_absolute() else PROJECT / path).resolve())
        equality(new['source_path'], expected, w + ':resolve_only')
        equality(sha(expected), old['source_sha256'], w + ':unchanged_source_bytes')
        actual_changed += old['source_path'] != new['source_path']
        changes.append({'candidate_id': old['candidate_id'], 'before': old['source_path'], 'after': new['source_path'], 'source_sha256': old['source_sha256']})
    equality(receipt['changes'], changes, 'path_adapter:receipt_exact_candidate_changes')
    for key, value in old_bindings.items():
        if key != 'files':
            equality(bindings.get(key), value, 'path_adapter:old_binding_metadata:' + key)
    old_files, new_files = old_bindings['files'], bindings['files']
    equality(len(new_files), len(old_files) + 2, 'path_adapter:exactly_two_additional_tool_bindings')
    canonical = str(BATCH / 'evaluation/EVALUATION_POOL.json')
    for old, new in zip(old_files, new_files):
        expected = {**old, 'sha256': sha(canonical)} if old['path'] == canonical else old
        equality(new, expected, 'path_adapter:old_binding_change_only_pool_hash:' + old['path'])
    COUNTS['path_adapter_questions_compared'] = len(changes)
    COUNTS['path_strings_changed'] = actual_changed
    COUNTS['path_strings_already_absolute_unchanged'] = len(changes) - actual_changed
    return {'failed_prepare_original_bytes_preserved': True, 'candidate_count': len(changes), 'changed_path_strings': actual_changed, 'already_absolute_unchanged': len(changes) - actual_changed, 'all_other_candidate_fields_verbatim': True}


def inspect_root(row, pool_by):
    rid = row['root_id']
    folder = BATCH / 'roots' / ('root' + rid.rsplit('-', 1)[1])
    names = ['CUMULATIVE_SOURCE.json', 'PUBLIC_QUESTIONS.json', 'PRIVATE_GOLD.json',
             'PUBLIC_SOURCE_JUDGMENTS.json', 'ASSET_SOURCE_JUDGMENTS.json',
             'PRIVATE_DESIGN.json', 'FINAL_CROSSCHECK.json', 'PUBLIC_ASSETS.json']
    paths = {name: folder / name for name in names}
    docs = {name: read(path) for name, path in paths.items()}
    source, public, gold, review, assets, design, final, public_assets = [docs[name] for name in names]
    turns = {t['turn_ref']: t for t in source['turns']}
    check(len(turns) == len(source['turns']), rid + ':source_refs_unique')
    equality(source['root_id'], rid, rid + ':source_root')
    equality(paths['CUMULATIVE_SOURCE.json'].read_bytes(), (folder / 'scenes/B_S03/CUMULATIVE_SOURCE.json').read_bytes(), rid + ':final_source_bytecopy')
    for doc, keys, where in [
        (gold, {'source_sha256': 'CUMULATIVE_SOURCE.json', 'public_sha256': 'PUBLIC_QUESTIONS.json', 'source_judgments_sha256': 'PUBLIC_SOURCE_JUDGMENTS.json', 'original_design_sha256': 'PRIVATE_DESIGN.json'}, 'gold'),
        (review, {'source_sha256': 'CUMULATIVE_SOURCE.json', 'public_sha256': 'PUBLIC_QUESTIONS.json', 'asset_judgments_sha256': 'ASSET_SOURCE_JUDGMENTS.json'}, 'final_review'),
        (assets, {'source_sha256': 'CUMULATIVE_SOURCE.json', 'public_sha256': 'PUBLIC_ASSETS.json'}, 'asset_review'),
        (design, {'source_sha256': 'CUMULATIVE_SOURCE.json', 'public_assets_sha256': 'PUBLIC_ASSETS.json', 'asset_judgments_sha256': 'ASSET_SOURCE_JUDGMENTS.json'}, 'design'),
        (final, {'private_sha256': 'PRIVATE_GOLD.json', 'public_sha256': 'PUBLIC_QUESTIONS.json', 'source_judgments_sha256': 'PUBLIC_SOURCE_JUDGMENTS.json'}, 'final_crosscheck')]:
        for key, name in keys.items():
            hash_link(doc, key, paths, name, rid + ':' + where)
    by_public = indexed(public['questions'], rid + ':public')
    by_gold = indexed(gold['questions'], rid + ':gold')
    by_design = indexed(design['design'], rid + ':design')
    by_final = indexed(final['results'], rid + ':crosscheck')
    for name, ids in [('gold', by_gold), ('design', by_design), ('crosscheck', by_final), ('review', review['questions'])]:
        equality(set(ids), set(by_public), rid + ':' + name + '_coverage')
    asset_ordering = indexed(assets['ordering_questions'], rid + ':asset_ordering')
    expected = {cid for cid, v in by_final.items() if v['status'] == 'CONTENT_PASS'}
    equality({cid for cid, q in pool_by.items() if q['root_id'] == rid}, expected, rid + ':pool_CONTENT_PASS_coverage')
    record = {'root_id': rid, 'public_count': len(by_public), 'content_pass_count': len(expected), 'questions': []}
    for cid in sorted(expected):
        before = len(ERRORS)
        q, g, d, j, e = by_public[cid], by_gold[cid], by_design[cid], review['questions'][cid], pool_by[cid]
        w = rid + ':' + cid
        for key, value in q.items():
            if key == 'source_path':
                source_path = Path(value)
                expected_path = str((source_path if source_path.is_absolute() else PROJECT / source_path).resolve())
                equality(e.get(key), expected_path, w + ':public_field:source_path_absolute_format_adapter')
            else:
                equality(e.get(key), value, w + ':public_field:' + key)
        equality(e.get('gold'), g['gold'], w + ':gold')
        equality(e.get('role'), g['role'], w + ':role')
        equality(by_final[cid]['source_derived_answer'], g['gold'], w + ':crosscheck_gold')
        check(by_final[cid]['original_gold_unchanged'] is True and not by_final[cid]['issues'], w + ':crosscheck_issues')
        equality(q['source_sha256'], sha(paths['CUMULATIVE_SOURCE.json']), w + ':source_hash')
        source_path = Path(e['source_path'])
        check(source_path.is_absolute(), w + ':native_source_path_absolute')
        equality(source_path.read_bytes(), paths['CUMULATIVE_SOURCE.json'].read_bytes(), w + ':source_path_bytes')
        meta = e.get('evidence_export', {})
        equality(meta.get('version'), 'source_review_lossless_evidence_export_v1', w + ':export_version')
        check(meta.get('all_reviewed_citations_preserved') is True and meta.get('semantic_entailment_checked_by_exporter') is False, w + ':export_claims')
        equality(meta.get('input_sha256'), {name: sha(path) for name, path in paths.items()}, w + ':all_input_hashes')
        ms = q['question_type'] == 'multi_select'
        field, evidence_field = ('options', 'option_evidence') if ms else ('items', 'item_evidence')
        labels = list('ABCDEF') if ms else ['I' + str(i) for i in range(1, 7)]
        equality(list(q[field]), labels, w + ':label_order')
        equality(set(e.get(evidence_field, {})), set(labels), w + ':evidence_coverage')
        label_records = []
        for label in labels:
            v, saved = e[evidence_field][label], j[field][label]
            lw = w + ':' + label
            # Every original judgment field must survive verbatim. The added
            # primary span may resolve a missing speaker only for ordering.
            for key, value in saved.items():
                if key == 'primary_citation' and not ms:
                    equality(v.get(key), normalized_anchor(value, turns), lw + ':review_field:' + key)
                else:
                    equality(v.get(key), value, lw + ':review_field:' + key)
            equality(v.get('proposition'), q[field][label], lw + ':proposition')
            if ms:
                COUNTS['multiselect_labels'] += 1
                citations = saved['citations']
                equality(v['citations'], citations, lw + ':all_final_citations_verbatim')
                equality(g['evidence'][label], citations, lw + ':private_citations_verbatim')
                for index, c in enumerate(citations):
                    citation(c, turns, lw + ':final_citation:' + str(index))
                primary = saved.get('primary_citation')
                if len(citations) > 1:
                    COUNTS['multicitations_labels'] += 1
                    check(primary is not None, lw + ':reviewer_explicit_multi_primary')
                if primary is None and len(citations) == 1:
                    primary = citations[0]
                check(primary in citations, lw + ':primary_is_saved_citation')
                equality(v.get('primary_citation'), primary, lw + ':primary_verbatim')
                equality(v.get('primary_citation_origin'), 'source_judgment' if saved.get('primary_citation') is not None else 'sole_reviewed_citation', lw + ':primary_origin')
                if len(citations) > 1 and primary != citations[0]:
                    COUNTS['explicit_nonfirst_primary_labels'] += 1
                check((saved['relation'] == 'supports') == (label in g['gold']), lw + ':relation_gold')
                check(saved['relation'] in ('supports', 'contradicts'), lw + ':no_unknown')
                sid = d['labels_to_statement_ids'][label]
                a = assets['statements'][sid]
                equality(v.get('statement_id'), sid, lw + ':asset_statement_id')
                equality(v.get('asset_source_judgment'), a, lw + ':complete_asset_judgment_verbatim')
                equality(a['relation'], saved['relation'], lw + ':asset_final_relation')
                equality(a['proposition'], saved['proposition'], lw + ':asset_final_proposition')
                for index, c in enumerate(a['citations']):
                    citation(c, turns, lw + ':asset_citation:' + str(index))
                COUNTS['final_citation_entries_preserved'] += len(citations)
                COUNTS['asset_citation_entries_preserved'] += len(a['citations'])
                equality(v.get('reason'), saved['reason'], lw + ':final_reason_verbatim')
                label_records.append({'label': label, 'citation_count': len(citations), 'primary_turn_ref': primary['turn_ref'], 'primary_explicit': saved.get('primary_citation') is not None, 'primary_is_first': primary == citations[0], 'statement_id': sid})
            else:
                COUNTS['ordering_labels'] += 1
                anchor = saved['order_anchor']
                primary = normalized_anchor(anchor, turns)
                equality(g['evidence'][label], anchor, lw + ':private_anchor_verbatim')
                equality(v['citations'], [anchor, *saved['alternative_matching_anchors']], lw + ':all_ordering_citations_verbatim')
                equality(v.get('primary_citation'), primary, lw + ':ordering_primary')
                equality(v.get('primary_citation_origin'), 'reviewed_order_anchor', lw + ':primary_origin')
                equality(e['order_anchor_spans'][label], primary, lw + ':native_anchor')
                equality(d['fixed_turns'][label], anchor['turn_ref'], lw + ':design_fixed_turn_preserved_in_anchor')
                equality(v['support_refs'], saved['support_refs'], lw + ':support_refs_verbatim')
                equality(v['alternative_matching_anchors'], saved['alternative_matching_anchors'], lw + ':alternatives_verbatim')
                check(anchor['turn_ref'] in v['support_refs'] and all(ref in turns for ref in v['support_refs']), lw + ':support_refs_valid')
                for index, c in enumerate([anchor, *saved['alternative_matching_anchors']]):
                    citation(c, turns, lw + ':order_citation:' + str(index), False)
                equality(asset_ordering[cid]['items'][label], saved, lw + ':asset_final_ordering_judgment')
                COUNTS['ordering_alternative_entries_preserved'] += len(saved['alternative_matching_anchors'])
                label_records.append({'label': label, 'fixed_turn_ref': d['fixed_turns'][label], 'support_refs_count': len(saved['support_refs']), 'alternative_count': len(saved['alternative_matching_anchors'])})
            for key in ('turn_ref', 'speaker', 'quote'):
                equality(v.get(key), primary[key], lw + ':compatibility_' + key)
        if not ms:
            positions = {t['turn_ref']: i for i, t in enumerate(source['turns'])}
            choices = [[x['order_anchor'], *x['alternative_matching_anchors']] for x in j['items'].values()]
            orders = set()
            for combination in itertools.product(*choices):
                order = sorted(zip(labels, combination), key=lambda item: (positions[item[1]['turn_ref']], turns[item[1]['turn_ref']]['text'].index(item[1]['quote'])))
                orders.add(tuple(label for label, _ in order))
            equality(orders, {tuple(g['gold'])}, w + ':all_reasonable_anchor_orders')
            equality(j['derived_order'], g['gold'], w + ':review_derived_order')
        COUNTS[q['question_type'] + '_questions'] += 1
        record['questions'].append({'candidate_id': cid, 'question_type': q['question_type'], 'gold_count': len(g['gold']), 'labels': label_records, 'status': 'PASS' if len(ERRORS) == before else 'FAIL'})
    return record


def main():
    started = datetime.now(timezone.utc).isoformat()
    # The parent authorizes running this only after prepare has completed.
    pool_path = BATCH / 'evaluation/EVALUATION_POOL.json'
    if not pool_path.is_file():
        raise SystemExit('WAIT_FOR_PARENT_PREPARE: no audit output written')
    roots = []
    for name in ['EXECUTION_INPUTS_FROZEN.json', 'evidence/TOOLING_FREEZE.json', 'evidence/PREEXECUTION_CODE_BINDINGS.json', 'evaluation/ADAPTER_BINDINGS.json']:
        doc = read(BATCH / name)
        for item in doc['files']:
            equality(sha(item['path']), item['sha256'], name + ':binding:' + item['path'])
            if item.get('snapshot_path'):
                equality(sha(item['snapshot_path']), item['snapshot_sha256'], name + ':snapshot:' + item['snapshot_path'])
                equality(item['sha256'], item['snapshot_sha256'], name + ':snapshot_current_hash')
    cfg = read(BATCH / 'RUN_CONFIG.json')
    pool = read(pool_path)
    path_adapter = inspect_path_adapter(pool)
    by_pool = indexed(pool['questions'], 'pool')
    equality(pool['status'], 'SEALED_CONTENT_READY_FOR_SOLVER_EVALUATION', 'pool_status')
    equality(pool['provenance']['source_batch_id'], BATCH.name, 'pool_batch')
    for row in cfg['roots']:
        try:
            roots.append(inspect_root(row, by_pool))
        except Exception as exc:
            ERRORS.append(row['root_id'] + ':AUDIT_EXCEPTION:' + type(exc).__name__ + ':' + str(exc))
    equality(len(by_pool), sum(root['content_pass_count'] for root in roots), 'global_CONTENT_PASS_coverage')
    equality(set(q['root_id'] for q in by_pool.values()), set(r['root_id'] for r in cfg['roots']), 'pool_configured_root_coverage')
    result = {'status': 'PASS' if not ERRORS else 'FAIL', 'auditor': '/root/v113_14304_recheck',
              'audit_kind': 'independent_programmatic_verbatim_export_and_binding_comparison',
              'semantic_recertification': False, 'exporter_imported_or_called': False,
              'solver_or_training_invoked': False, 'root_files_modified': False,
              'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
              'audit_code_sha256': sha(__file__), 'evaluation_pool_sha256': sha(pool_path),
              'candidate_count': len(by_pool), 'counts': dict(COUNTS), 'roots': roots,
              'path_adapter_comparison': path_adapter,
              'checked_input_files': [{'path': path, 'sha256': digest} for path, digest in BINDINGS.items()],
              'limitations': ['Mechanical export audit only; semantic reviews remain the saved source-first judgments.', 'Auditor authored two roots and shares the inherited model; this is not external semantic certification.'],
              'errors': ERRORS}
    output = BATCH / 'EVIDENCE_EXPORT_AUDIT.json'
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ['status', 'candidate_count', 'counts', 'errors']}))
    if ERRORS:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
