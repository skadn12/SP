"""Gold-aware export of all three predeclared V2 operation revisions; no inference."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/postweek_live_extension_v5')
OLD = BASE / 'reason_diagnostic_v5'
REVISION = BASE / 'order_operation_revision_v1/OPERATION_METADATA_REVISION_V2.json'
REVIEW = BASE / 'order_operation_revision_v1/OPERATION_METADATA_REVIEW_V2.json'
REVISION_SHA = '22f3a6093f32773d6daf9ed2d1c063a1029a50301aeef1e1793268793d630a6d'

def read(path): return json.loads(Path(path).read_text())
def bind(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
def check(bound): assert bind(bound['path']) == bound, 'BOUND_FILE_CHANGED:' + bound['path']
def bindings(value):
    if isinstance(value, dict):
        if {'path','sha256','bytes'} <= set(value): yield {k:value[k] for k in ('path','sha256','bytes')}
        else:
            for child in value.values(): yield from bindings(child)
    elif isinstance(value, list):
        for child in value: yield from bindings(child)
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

def main():
    assert not (HERE / 'inputs').exists(), 'NO_EXPORT_OVERWRITE'
    assert bind(REVISION)['sha256'] == REVISION_SHA, 'V2_PROPOSAL_HASH_CHANGED'
    revision, review = read(REVISION), read(REVIEW)
    assert bind(REVIEW)['sha256'] == '77954b87e68e8e39bbe95e7809da80f376fc8cddcde322f2f0c8094d05039354', 'V2_REVIEW_HASH_CHANGED'
    assert review['status'] == 'PASS', 'INDEPENDENT_V2_METADATA_PASS_REQUIRED'
    assert any(b == bind(REVISION) for b in bindings(review)), 'REVIEW_NOT_BOUND_TO_V2'
    assert review['policy_checks']['old_scores_reusable'] is False and review['policy_checks']['solver_score_filtering'] is False
    assert all(m['verdict'] == 'PASS' for m in review['mapping_reviews'])
    for bound in bindings(review): check(bound)
    changes = revision['changes']
    assert len(changes) == 3 and len({c['candidate_id'] for c in changes}) == 3
    assert all(c['new_task_operation'] == 'ORDER_ENQUIRY_PROGRESSION' for c in changes)
    old_freeze = read(OLD / 'inputs/EXPORT_FREEZE.json')
    for bound in old_freeze['bindings'] + [old_freeze['private_binding']]: check(bound)
    public = {q['candidate_id']:q for q in read(OLD / 'inputs/PUBLIC_QUESTIONS.json')['questions']}
    private = {q['candidate_id']:q for q in read(old_freeze['private_binding']['path'])['questions']}
    selected, keys = [], []
    for change in changes:
        cid = change['candidate_id']; row = dict(public[cid])
        assert row['question_type'] == 'ordering' and row['task_operation'] == change['old_task_operation']
        row['task_operation'] = change['new_task_operation']
        assert {k:v for k,v in row.items() if k != 'task_operation'} == {k:v for k,v in public[cid].items() if k != 'task_operation'}
        selected.append(row); keys.append(private[cid])
    save(HERE / 'inputs/PUBLIC_QUESTIONS.json', {'questions': selected})
    private_path = HERE / 'inputs/PRIVATE_SCORING.json'
    save(private_path, {'questions': keys}); private_path.chmod(0o600)
    save(HERE / 'inputs/EXPORT_FREEZE.json', {'schema_version':'order_operation_reason_retest_v2_export_v1',
        'requests':3, 'fixed_candidate_ids':[c['candidate_id'] for c in changes],
        'selection':'All three V2 metadata changes, fixed before new outputs; no score-based selection.',
        'exporter_gold_aware':True, 'inference_gold_aware':False,
        'stem_display_source_gold_unchanged':True, 'operation_only_revision':True,
        'bindings':[bind(p) for p in (REVISION, REVIEW, OLD/'inputs/EXPORT_FREEZE.json', OLD/'inputs/PUBLIC_QUESTIONS.json',
            HERE/'inputs/PUBLIC_QUESTIONS.json', HERE/'PROTOCOL_ZH.md', Path(__file__))],
        'private_binding':bind(private_path), 'original_private_binding':old_freeze['private_binding']})
    print('EXPORTED: fixed three V2 public questions and private scoring; no model call')

if __name__ == '__main__': main()
