"""Produce a provisional training-side frontier package without training a model."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = HERE.parent / 'h4_codex_scaled_production_v111'
REPLAY = HERE.parent.parent / 'SP-820-publish/first_training_pipeline/data/matched_training/m_clean_policy_train.jsonl'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bind(p):
    return {'path': str(p), 'sha256': sha(p)}

def save(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def verify_manifest(path):
    doc = read(path)
    for row in doc['files']:
        file = Path(row['path'])
        file = file if file.is_absolute() else path.parent / file
        assert sha(file) == row['sha256'], str(file)

def messages(question):
    source = read(question['source_path'])
    multi = question['question_type'] == 'multi_select'
    field, heading = ('options', 'Options') if multi else ('items', 'Items')
    system = ('You are a strict structured dialogue QA solver. Use only the supplied dialogue when it is present. '
              'Return the final answer in exactly one <box>...</box> pair. Do not provide explanations, '
              'reasoning, or any text outside the box.')
    parts = ['Dialogue:\n' + '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns']),
        f"Question ({question['question_type']}):\n{question['question']}",
        heading + ':\n' + '\n'.join(f'{k}. {v}' for k, v in question[field].items()),
        'Output only selected letters separated by commas, e.g. <box>A,C</box>.' if multi else
        'Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>.']
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': '\n\n'.join(parts)},
            {'role': 'assistant', 'content': '<box>' + ','.join(question['gold']) + '</box>'}]

def main():
    audit = read(HERE / 'AUDIT_RESULTS.json')
    assert audit['status'] == 'VALID' and not audit['data_errors']
    verify_manifest(HERE / 'EVALUATION_COMPLETE_MANIFEST.json')
    verify_manifest(BATCH / 'COMPLETE_MANIFEST.json')
    exposure = read(BATCH / 'EXPOSURE_AUDIT.json')
    assert exposure['purpose'] == 'TRAINING_SIDE_DATA_PRODUCTION'
    assert all(exposure[k] == [] for k in ('holdout_root_overlap', 'holdout_source_id_overlap', 'holdout_source_record_overlap'))
    pool = read(HERE / 'EVALUATION_POOL.json')['questions']
    stats = {q['candidate_id']: q for q in audit['models']['trained']['questions']}
    eligible, quarantined, rows = [], [], []
    for q in pool:
        s = stats[q['candidate_id']]
        control = q['question_type'] == 'multi_select' and len(q['gold']) == 6
        reason = 'GOLD6_CONTROL' if control else 'NATIVE_FRONTIER_PASS' if s['selected_by_policy'] else 'NOT_NATIVE_FRONTIER'
        item = {'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
            'family_id': q['root_id'] + ':' + read(BATCH / 'RUN_CONFIG.json')['scene_ref'],
            'question_type': q['question_type'], 'gold_count': len(q['gold']),
            'reason': reason, 'source_sha256': q['source_sha256'],
            'solver_question_statistics': s}
        if reason != 'NATIVE_FRONTIER_PASS':
            quarantined.append(item)
            continue
        chat = messages(q)
        eligible.append({**item, 'canonical_gold': q['gold'], 'messages_sha256':
            hashlib.sha256(json.dumps(chat, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
        rows.append({'messages': chat})
    directory = HERE / 'training_candidate_package'
    directory.mkdir()
    with (directory / 'frontier_candidates.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    save(directory / 'FRONTIER_MANIFEST.json', {'status': 'PROVISIONAL_NO_TRAINING_EXECUTED',
        'basis_model': 'M_clean_policy', 'candidates': eligible,
        'candidate_count': len(eligible), 'family_split_unit': 'root',
        'related_variants_are_not_independent': True, 'training_execution_authorized_by_this_file': False})
    save(directory / 'QUARANTINE.json', {'candidates': quarantined,
        'content_failed_candidates': [r for r in read(BATCH / 'ALL_OUTCOMES_INDEX.json')['results']
                                      if r['status'] != 'CONTENT_PASS'],
        'all_content_outcomes_index': bind(BATCH / 'ALL_OUTCOMES_INDEX.json'),
        'source_or_slot_failures': read(BATCH / 'UNFILLED_SLOTS.json')})
    save(directory / 'REPLAY_BINDING.json', {'existing_clean_replay': bind(REPLAY),
        'rows': sum(1 for _ in REPLAY.open(encoding='utf-8')),
        'mixing_ratio_frozen': False, 'mixed_training_file_created': False,
        'note': 'Original replay is referenced, not duplicated or silently resampled; train budget/mix remains a later experiment configuration.'})
    counts = Counter(q['question_type'] for q in eligible)
    gold = Counter(str(q['gold_count']) for q in eligible if q['question_type'] == 'multi_select')
    report = {'status': 'PROVISIONAL_CANDIDATE_PACKAGE_COMPLETE', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'content_manifest': bind(BATCH / 'COMPLETE_MANIFEST.json'),
        'evaluation_manifest': bind(HERE / 'EVALUATION_COMPLETE_MANIFEST.json'),
        'eligible_candidates': len(eligible), 'type_counts': dict(counts), 'multi_gold_counts': dict(sorted(gold.items())),
        'eligible_roots': len({q['root_id'] for q in eligible}), 'quarantined_candidates': len(quarantined),
        'gold6_in_export': 0, 'validation_test_in_export': 0,
        'external_semantic_review_performed': False, 'training_started': False,
        'limitations': ['Candidate export is not evidence of training benefit.',
            'Author and semantic reviewer use the same inherited model and share project background.',
            'Training mix, family weights and training budget are not silently chosen by this export.']}
    save(directory / 'PACKAGE_REPORT.json', report)
    save(directory / 'COMPLETE_MANIFEST.json', {'status': 'PROVISIONAL_TRAINING_CANDIDATE_PACKAGE_SEALED',
        'files': [bind(p) for p in sorted(directory.iterdir()) if p.is_file()],
        'script': bind(Path(__file__)), 'training_started': False})
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    main()
