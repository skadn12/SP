"""Append a uniform stage index while preserving original role receipts verbatim."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    timing_path = HERE / 'evidence/GROUP1_REVIEW_TIMING_ADDENDUM.json'
    timing = {r['root_id']: r for r in read(timing_path)['roots']} if timing_path.exists() else {}
    stages = [('EXPANSION', 'AUTHOR_RECEIPT.json'), ('SOURCE_REVIEW', 'SOURCE_REVIEW.json'),
        ('QA_AUTHOR', 'QA_AUTHOR_RECEIPT.json'), ('ASSET_REVIEW', 'ASSET_SOURCE_JUDGMENTS.json'),
        ('COMPOSITION', 'COMPOSITION_RECEIPT.json'), ('PUBLIC_REVIEW', 'PUBLIC_SOURCE_JUDGMENTS.json'),
        ('GOLD_CROSSCHECK', 'FINAL_CROSSCHECK.json')]
    rows = []
    for reservation in read(HERE / 'SOURCE_RESERVATIONS.json')['allocation']:
        rid = reservation['root_id']
        folder = HERE / 'roots' / ('root' + rid.rsplit('-', 1)[1])
        for stage, name in stages:
            path = folder / name
            if not path.exists():
                continue
            doc = read(path)
            timing_note = timing.get(rid) if stage == 'ASSET_REVIEW' else None
            if timing_note:
                assert bind(path)['sha256'] == timing_note['asset_source_judgments']['sha256']
            actor = doc.get('reviewer') or doc.get('author') or doc.get('author_agent') or doc.get('author_role')
            backend = 'PROGRAMMATIC' if stage in ('COMPOSITION', 'GOLD_CROSSCHECK') else 'CURRENT_CODEX_SESSION'
            if backend == 'PROGRAMMATIC':
                actor = 'codex_controlled_batch/' + stage.lower()
            rows.append({'schema_version': 'v109_stage_ledger_v1', 'run_id': HERE.name,
                'root_id': rid, 'family_id': rid + ':R02:C:S01', 'source_id': reservation['source_id'],
                'stage': stage, 'attempt': doc.get('attempt', 1), 'actor': actor,
                'backend': backend, 'model_identifier': None,
                'model_identifier_precision': 'not exposed by session' if backend != 'PROGRAMMATIC' else 'not applicable',
                'policy_version': 'v105_content_contract_inheriting_v98_v103',
                'policy_binding': bind(HERE / 'evidence/CONTENT_CONTRACT.md'),
                'raw_record': bind(path), 'source_sha256': doc.get('source_sha256'),
                'public_sha256': doc.get('public_sha256'),
                'started_at_utc': doc.get('started_at_utc'),
                'finished_at_utc': timing_note['actual_asset_finished_at_utc'] if timing_note else
                    doc.get('finished_at_utc') or doc.get('created_at_utc'),
                'original_reported_finished_at_utc': doc.get('finished_at_utc') or doc.get('created_at_utc'),
                'timing_addendum': bind(timing_path) if timing_note else None,
                'status': doc.get('decision') or doc.get('status') or 'RECORDED',
                'full_source_read': doc.get('full_source_read', doc.get('full_baseline_read')),
                'private_answers_read_before_save': doc.get('private_answers_read_before_save', doc.get('private_gold_read')),
                'independent_of_question_author': doc.get('independent_of_question_author'),
                'usage_input_tokens': None, 'usage_output_tokens': None, 'billed_cost': None,
                'usage_limitation': 'Generation usage not exposed; mechanical stages make no model calls.',
                'controller_bindings': [bind(p) for p in sorted((HERE / 'evidence/precall_code').glob('*.py'))]})
    target = HERE / 'NORMALIZED_STAGE_LEDGER.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump({'status': 'APPEND_ONLY_NORMALIZED_INDEX_NOT_REPLACEMENT_RECEIPTS',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'records': rows,
            'limitation': 'The policy document is frozen; exact per-turn orchestration prompts and exact generator usage are not fully exposed by this entry point.'},
            f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'ledger_records': len(rows), 'original_receipts_modified': 0}))

if __name__ == '__main__':
    main()
