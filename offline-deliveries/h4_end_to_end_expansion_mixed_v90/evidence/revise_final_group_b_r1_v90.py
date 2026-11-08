"""Preserve the initial whole-question review and record its temporal correction."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARD = HERE.parent / 'shards/shard_01r2'
FOLDER = SHARD / 'multiselect/final_review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    original_path = FOLDER / 'group_b_AUTHORED_JUDGMENTS.json'
    archived_path = HERE / 'superseded_reviews/group_b_SOURCE_JUDGMENTS_initial.json'
    assert archived_path.is_file() and not (FOLDER / 'group_b_SOURCE_JUDGMENTS.json').exists()
    public_path = SHARD / 'multiselect/PUBLIC_QUESTIONS.json'
    public = {q['candidate_id']: q for q in read(public_path)['questions']}
    old = read(original_path)
    revised = deepcopy(old)
    revised['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    revised['version'] = 'v90_authored_whole_question_semantic_review_r1_v1'
    revised['supersedes_authored_judgments'] = bind(original_path)
    revised['supersedes_initial_source_judgments'] = bind(archived_path)
    revised['revision_reason'] = (
        "Source-only reassessment and an independent temporal spotcheck agree that "
        "'is staying in to run errands later' permits a future-plan reading. The "
        "current 'heading out ... now' report cannot explicitly refute that later "
        "arrangement. No private Gold or Solver was consulted."
    )
    updates = {
        'cc-train-episode-135466-v79-root135466_family-5e2c7e8dce9d': 'B',
        'cc-train-episode-135466-v79-root135466_family-4f42d8018a92': 'A',
        'cc-train-episode-135466-v79-root135466_family-09bfd40f4509': 'C',
    }
    changed = []
    for row in revised['questions']:
        q = public[row['candidate_id']]
        source = read(q['source_path'])
        assert sha(q['source_path']) == q['source_sha256'] == revised['source_sha256']
        assert len(source['turns']) == revised['complete_source_turns_read'] == 204
        if row['candidate_id'] not in updates:
            continue
        label = updates[row['candidate_id']]
        judgment = row['options'][label]
        assert judgment['proposition'] == q['options'][label] == (
            'In the R02:B:S01 scene, Morgan says Morgan is staying in to run errands later.')
        assert judgment['relation'] == 'contradicts' and judgment['scope_valid'] is True
        judgment['relation'] = 'unknown'
        judgment['scope_valid'] = False
        judgment['reason'] = (
            'Later errands may be a separate future plan; heading out now does not '
            'explicitly rule out staying in later or identify the same errand episode.'
        )
        changed.append({'candidate_id': row['candidate_id'], 'label': label,
                        'changed_fields': ['relation', 'scope_valid', 'reason'],
                        'relation_before': 'contradicts', 'relation_after': 'unknown'})
    assert len(changed) == 3
    # Confirm the revision changes exactly the three disclosed option decisions.
    for old_row, new_row in zip(old['questions'], revised['questions']):
        assert old_row['candidate_id'] == new_row['candidate_id'] and old_row['stem_valid'] == new_row['stem_valid']
        for label, previous in old_row['options'].items():
            current = new_row['options'][label]
            if updates.get(old_row['candidate_id']) == label:
                assert {k for k in previous if previous[k] != current[k]} == {'relation', 'scope_valid', 'reason'}
            else:
                assert previous == current
    revised['revisions'] = changed
    output = FOLDER / 'group_b_AUTHORED_JUDGMENTS_r1.json'
    with output.open('x') as stream:
        json.dump(revised, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    helper_path = HERE / 'save_authored_final_review_v90.py'
    spec = importlib.util.spec_from_file_location('v90_save_revised_final_multiselect_review', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.save(SHARD, 'b', output)
    print(json.dumps({'status': 'SAVED', 'revised_option_decisions': 3,
                      'other_option_decisions_unchanged': 27,
                      'current_private_artifacts_read': False,
                      'initial_reviews_preserved': True}))


if __name__ == '__main__':
    main()
